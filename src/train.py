"""Two-stage clean-image transfer learning from a versioned YAML configuration."""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from src.data.datasets import ManifestImageDataset, load_manifest
from src.data.transforms import build_transform, preprocessing_from_model_config
from src.metrics.classification import classification_metrics
from src.models.build import (
    VisionClassifier,
    build_model,
    set_backbone_trainable,
    trainable_parameters,
)
from src.utils import create_run_directory, load_config, set_seed, write_json


def _device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _worker_seed(worker_id: int) -> None:
    worker_seed = (torch.initial_seed() + worker_id) % (2**32)
    np.random.seed(worker_seed)


def build_loaders(config: dict[str, Any], seed: int) -> tuple[DataLoader, DataLoader]:
    data = config["dataset"]
    model_config = config["model"]
    training = config["training"]
    manifest = load_manifest(data["manifest"], fold=data.get("fold"))
    image_size = int(model_config["input_size"])
    preprocessing = preprocessing_from_model_config(model_config)
    train_dataset = ManifestImageDataset(
        manifest,
        "train",
        data_root=data["root"],
        transform=build_transform(True, image_size, **preprocessing),
    )
    val_dataset = ManifestImageDataset(
        manifest,
        "val",
        data_root=data["root"],
        transform=build_transform(False, image_size, **preprocessing),
    )
    generator = torch.Generator().manual_seed(seed)
    common = {
        "batch_size": int(training["batch_size"]),
        "num_workers": int(training.get("num_workers", 0)),
        "pin_memory": torch.cuda.is_available(),
        "worker_init_fn": _worker_seed,
        "generator": generator,
    }
    return (
        DataLoader(train_dataset, shuffle=True, **common),
        DataLoader(val_dataset, shuffle=False, **common),
    )


def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    if (
        training
        and isinstance(model, VisionClassifier)
        and not any(parameter.requires_grad for parameter in model.backbone.parameters())
    ):
        # Frozen parameters do not freeze BatchNorm running statistics by themselves.
        model.backbone.eval()
    loss_sum = 0.0
    logits_parts: list[np.ndarray] = []
    label_parts: list[np.ndarray] = []
    context = torch.enable_grad if training else torch.inference_mode
    with context():
        for batch in loader:
            inputs = batch["image"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = model(inputs)
            loss = criterion(logits, labels)
            if training:
                loss.backward()
                optimizer.step()
            loss_sum += float(loss.detach()) * len(labels)
            logits_parts.append(logits.detach().cpu().numpy())
            label_parts.append(labels.detach().cpu().numpy())
    logits_array = np.concatenate(logits_parts)
    labels_array = np.concatenate(label_parts)
    probabilities = torch.softmax(torch.from_numpy(logits_array), dim=1).numpy()
    metrics = classification_metrics(probabilities, labels_array)
    metrics["loss"] = loss_sum / len(labels_array)
    return metrics


def _train_stage(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    stage: str,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    patience: int,
    history: list[dict[str, Any]],
    best_score: float,
    best_state: dict[str, torch.Tensor],
) -> tuple[float, dict[str, torch.Tensor]]:
    if epochs <= 0:
        return best_score, best_state
    optimizer = AdamW(trainable_parameters(model), lr=learning_rate, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=max(1, epochs))
    criterion = nn.CrossEntropyLoss()
    stale = 0
    for epoch in range(1, epochs + 1):
        started = time.monotonic()
        train_metrics = _run_epoch(model, train_loader, criterion, device, optimizer)
        val_metrics = _run_epoch(model, val_loader, criterion, device)
        row: dict[str, Any] = {
            "stage": stage,
            "epoch": epoch,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "duration_seconds": time.monotonic() - started,
        }
        row.update({f"train_{key}": value for key, value in train_metrics.items()})
        row.update({f"val_{key}": value for key, value in val_metrics.items()})
        history.append(row)
        score = val_metrics["macro_f1"]
        if score > best_score:
            best_score = score
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
        scheduler.step()
        print(
            f"{stage} epoch {epoch:02d}/{epochs:02d} "
            f"train_loss={train_metrics['loss']:.4f} "
            f"val_macro_f1={score:.4f}"
        )
        if stale >= patience:
            print(f"early stopping {stage} after {epoch} epochs")
            break
    return best_score, best_state


def train(config: dict[str, Any]) -> dict[str, Any]:
    training = config["training"]
    seed = int(training["seed"])
    set_seed(seed, deterministic=bool(training.get("deterministic", True)))
    run_id, run_dir = create_run_directory(
        config, seed, base_dir=config.get("outputs", {}).get("runs_dir", "results/runs")
    )
    device = _device(str(training.get("device", "auto")))
    train_loader, val_loader = build_loaders(config, seed)
    model_config = config["model"]
    data_config = config["dataset"]
    model = build_model(
        model_config["backbone"],
        num_classes=int(data_config["num_classes"]),
        pretrained=bool(model_config.get("pretrained", True)),
        dropout=float(model_config.get("dropout", 0.2)),
    ).to(device)
    history: list[dict[str, Any]] = []
    best_score = -float("inf")
    best_state = copy.deepcopy(model.state_dict())

    set_backbone_trainable(model, False)
    best_score, best_state = _train_stage(
        model,
        train_loader,
        val_loader,
        device,
        "head",
        epochs=int(training["head_epochs"]),
        learning_rate=float(training["head_lr"]),
        weight_decay=float(training.get("weight_decay", 1e-4)),
        patience=int(training.get("patience", 3)),
        history=history,
        best_score=best_score,
        best_state=best_state,
    )
    model.load_state_dict(best_state)
    set_backbone_trainable(model, True)
    best_score, best_state = _train_stage(
        model,
        train_loader,
        val_loader,
        device,
        "fine_tune",
        epochs=int(training["finetune_epochs"]),
        learning_rate=float(training["finetune_lr"]),
        weight_decay=float(training.get("weight_decay", 1e-4)),
        patience=int(training.get("patience", 3)),
        history=history,
        best_score=best_score,
        best_state=best_state,
    )
    model.load_state_dict(best_state)
    final_val = _run_epoch(model, val_loader, nn.CrossEntropyLoss(), device)

    checkpoint_dir = Path(config.get("outputs", {}).get("checkpoints_dir", "results/checkpoints"))
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / (
        f"{data_config['name']}-{model.backbone_name}-seed{seed}-{run_id}.pt"
    )
    manifest_sha256 = hashlib.sha256(Path(data_config["manifest"]).read_bytes()).hexdigest()
    torch.save(
        {
            "format_version": 1,
            "run_id": run_id,
            "config": config,
            "manifest_sha256": manifest_sha256,
            "model_state": model.state_dict(),
            "best_val_macro_f1": best_score,
            "seed": seed,
        },
        checkpoint_path,
    )
    if history:
        with (run_dir / "curves.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(history[0]))
            writer.writeheader()
            writer.writerows(history)
    result = {
        "run_id": run_id,
        "checkpoint": str(checkpoint_path),
        "device": str(device),
        "best_val_macro_f1": best_score,
        "manifest_sha256": manifest_sha256,
        "final_val": final_val,
        "history": history,
    }
    write_json(run_dir / "metrics.json", result)
    return result


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--seed", type=int, help="override the versioned training seed")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="one epoch per stage with random initialization; for pipeline verification only",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    config = load_config(args.config)
    if args.seed is not None:
        config["training"]["seed"] = args.seed
    if args.smoke:
        config["training"].update({"head_epochs": 1, "finetune_epochs": 1, "patience": 1})
        config["model"]["pretrained"] = False
    result = train(config)
    print(f"checkpoint: {result['checkpoint']}")


if __name__ == "__main__":
    main()
