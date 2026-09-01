"""Checkpoint inference with deterministic evaluation-only corruptions and caching."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.special import softmax
from torch.utils.data import DataLoader

from src.data.datasets import ManifestImageDataset, load_manifest
from src.data.transforms import build_transform, preprocessing_from_model_config
from src.models.build import VisionClassifier, build_model
from src.shifts.corruptions import make_corruption
from src.shifts.severity import corruption_protocol_digest
from src.utils import load_config, set_seed


def resolve_device(requested: str = "auto") -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@dataclass(frozen=True)
class PredictionBundle:
    logits: np.ndarray
    labels: np.ndarray
    features: np.ndarray
    paths: np.ndarray

    @property
    def probabilities(self) -> np.ndarray:
        return softmax(self.logits, axis=1)


class CheckpointEvaluator:
    """Load one checkpoint and reuse it across the complete corruption grid."""

    def __init__(
        self,
        checkpoint_path: str | Path,
        config: dict[str, Any] | None = None,
        device: str = "auto",
        cache_dir: str | Path = "results/cache",
    ) -> None:
        self.checkpoint_path = Path(checkpoint_path)
        self.device = resolve_device(device)
        checkpoint = torch.load(self.checkpoint_path, map_location="cpu", weights_only=False)
        checkpoint_config = copy.deepcopy(checkpoint["config"])
        self.config = self._resolved_config(checkpoint_config, config)
        model_config = self.config["model"]
        data_config = self.config["dataset"]
        self.model: VisionClassifier = build_model(
            model_config["backbone"],
            num_classes=int(data_config["num_classes"]),
            pretrained=False,
            dropout=float(model_config.get("dropout", 0.2)),
        )
        self.model.load_state_dict(checkpoint["model_state"], strict=True)
        self.model.to(self.device).eval()
        self.seed = int(checkpoint.get("seed", self.config["training"]["seed"]))
        self.run_id = str(checkpoint.get("run_id", self.checkpoint_path.stem))
        self.cache_dir = Path(cache_dir) / self.checkpoint_path.stem
        manifest_path = Path(data_config["manifest"])
        manifest_bytes = manifest_path.read_bytes()
        self.manifest_digest = hashlib.sha256(manifest_bytes).hexdigest()
        self.corruption_protocol_sha256 = corruption_protocol_digest()
        expected_manifest_digest = checkpoint.get("manifest_sha256")
        if (
            expected_manifest_digest is not None
            and str(expected_manifest_digest) != self.manifest_digest
        ):
            raise ValueError(
                "evaluation manifest differs from the split manifest saved with the checkpoint"
            )
        fingerprint = hashlib.sha256()
        fingerprint.update(self.checkpoint_path.read_bytes())
        fingerprint.update(manifest_bytes)
        inference_config = {
            "dataset": {key: data_config.get(key) for key in ("name", "root", "manifest", "fold")},
            "model": {
                "backbone": model_config.get("backbone"),
                "input_size": model_config.get("input_size"),
                "preprocessing": preprocessing_from_model_config(model_config),
            },
            "evaluation": self.config.get("evaluation", {}),
            "corruption_protocol_sha256": self.corruption_protocol_sha256,
        }
        fingerprint.update(
            json.dumps(inference_config, sort_keys=True, default=str).encode("utf-8")
        )
        self._source_fingerprint = fingerprint.digest()
        self.manifest = load_manifest(data_config["manifest"], fold=data_config.get("fold"))

    @staticmethod
    def _resolved_config(
        checkpoint_config: dict[str, Any], override: dict[str, Any] | None
    ) -> dict[str, Any]:
        """Preserve checkpoint identity while allowing relocated evaluation data."""

        if override is None:
            return checkpoint_config
        resolved = copy.deepcopy(checkpoint_config)
        immutable = {
            "dataset.name": (
                checkpoint_config["dataset"].get("name"),
                override["dataset"].get("name"),
            ),
            "dataset.num_classes": (
                checkpoint_config["dataset"].get("num_classes"),
                override["dataset"].get("num_classes"),
            ),
            "model.backbone": (
                checkpoint_config["model"].get("backbone"),
                override["model"].get("backbone"),
            ),
            "model.input_size": (
                checkpoint_config["model"].get("input_size"),
                override["model"].get("input_size"),
            ),
            "model.dropout": (
                checkpoint_config["model"].get("dropout", 0.2),
                override["model"].get("dropout", 0.2),
            ),
        }
        checkpoint_preprocessing = preprocessing_from_model_config(checkpoint_config["model"])
        override_preprocessing = preprocessing_from_model_config(override["model"])
        immutable.update(
            {
                f"model.{field}": (
                    checkpoint_preprocessing[field],
                    override_preprocessing[field],
                )
                for field in ("mean", "std", "interpolation", "crop_pct")
            }
        )
        mismatches = [name for name, values in immutable.items() if values[0] != values[1]]
        if mismatches:
            raise ValueError(
                "checkpoint and evaluation config disagree on immutable fields: "
                + ", ".join(mismatches)
            )
        for field in ("root", "manifest"):
            if field in override["dataset"]:
                resolved["dataset"][field] = override["dataset"][field]
        resolved["evaluation"] = copy.deepcopy(override.get("evaluation", {}))
        resolved["outputs"] = copy.deepcopy(override.get("outputs", {}))
        return resolved

    def _cache_path(self, split: str, corruption: str, severity: int) -> Path:
        fingerprint = hashlib.sha256()
        fingerprint.update(self._source_fingerprint)
        fingerprint.update(f"{split}|{corruption}|{severity}".encode())
        return (
            self.cache_dir / f"{split}-{corruption}-s{severity}-{fingerprint.hexdigest()[:12]}.npz"
        )

    def loader(self, split: str, corruption: str = "clean", severity: int = 0) -> DataLoader:
        if corruption == "clean":
            if severity != 0:
                raise ValueError("clean condition requires severity 0")
            callback = None
        else:
            if split == "train":
                raise ValueError("corruptions cannot be applied to the training split")
            shift_seed = int(self.config.get("evaluation", {}).get("corruption_seed", 1729))
            callback = make_corruption(corruption, severity, base_seed=shift_seed)
        data = self.config["dataset"]
        model_config = self.config["model"]
        dataset = ManifestImageDataset(
            self.manifest,
            split,
            data_root=data["root"],
            transform=build_transform(
                False,
                int(model_config["input_size"]),
                **preprocessing_from_model_config(model_config),
            ),
            corruption=callback,
        )
        return DataLoader(
            dataset,
            batch_size=int(
                self.config.get("evaluation", {}).get(
                    "batch_size", self.config["training"]["batch_size"]
                )
            ),
            shuffle=False,
            num_workers=int(self.config.get("evaluation", {}).get("num_workers", 0)),
            pin_memory=self.device.type == "cuda",
        )

    @torch.inference_mode()
    def infer(
        self,
        split: str,
        corruption: str = "clean",
        severity: int = 0,
        use_cache: bool = True,
    ) -> PredictionBundle:
        path = self._cache_path(split, corruption, severity)
        if use_cache and path.exists():
            cached = np.load(path, allow_pickle=False)
            return PredictionBundle(
                logits=cached["logits"],
                labels=cached["labels"],
                features=cached["features"],
                paths=cached["paths"],
            )
        set_seed(self.seed)
        logits: list[np.ndarray] = []
        labels: list[np.ndarray] = []
        features: list[np.ndarray] = []
        paths: list[str] = []
        for batch in self.loader(split, corruption, severity):
            inputs = batch["image"].to(self.device, non_blocking=True)
            pooled = self.model.forward_features(inputs)
            output = self.model.head(pooled)
            logits.append(output.cpu().numpy())
            features.append(pooled.cpu().numpy())
            labels.append(batch["label"].numpy())
            paths.extend(str(item) for item in batch["path"])
        bundle = PredictionBundle(
            logits=np.concatenate(logits),
            labels=np.concatenate(labels),
            features=np.concatenate(features),
            paths=np.asarray(paths),
        )
        if use_cache:
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                path,
                logits=bundle.logits,
                labels=bundle.labels,
                features=bundle.features,
                paths=bundle.paths,
            )
        return bundle

    @torch.inference_mode()
    def infer_mc_dropout(
        self,
        split: str,
        corruption: str,
        severity: int,
        passes: int = 30,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return passes x samples x classes logits, labels, and paths."""

        if passes < 2:
            raise ValueError("passes must be at least two")
        # BatchNorm and all other layers stay in eval mode; only Dropout is enabled.
        self.model.eval()
        for module in self.model.modules():
            if isinstance(module, torch.nn.modules.dropout._DropoutNd):
                module.train()
        pass_logits: list[list[np.ndarray]] = [[] for _ in range(passes)]
        all_labels: list[np.ndarray] = []
        all_paths: list[str] = []
        with torch.random.fork_rng(devices=[self.device] if self.device.type == "cuda" else []):
            torch.manual_seed(self.seed)
            for batch in self.loader(split, corruption, severity):
                inputs = batch["image"].to(self.device, non_blocking=True)
                for pass_index in range(passes):
                    pass_logits[pass_index].append(self.model(inputs).cpu().numpy())
                all_labels.append(batch["label"].numpy())
                all_paths.extend(str(item) for item in batch["path"])
        self.model.eval()
        stacked = np.stack([np.concatenate(parts) for parts in pass_logits])
        return stacked, np.concatenate(all_labels), np.asarray(all_paths)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--split", default="test")
    parser.add_argument("--corruption", default="clean")
    parser.add_argument("--severity", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--no-cache", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config) if args.config else None
    evaluator = CheckpointEvaluator(args.checkpoint, config=config, device=args.device)
    bundle = evaluator.infer(
        args.split, args.corruption, args.severity, use_cache=not args.no_cache
    )
    summary = {
        "checkpoint": str(args.checkpoint),
        "split": args.split,
        "corruption": args.corruption,
        "severity": args.severity,
        "samples": len(bundle.labels),
        "logit_shape": list(bundle.logits.shape),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
