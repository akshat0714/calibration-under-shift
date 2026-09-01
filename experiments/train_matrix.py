"""Train all configured seeds/folds and write a checkpoint registry."""

from __future__ import annotations

import argparse
import copy
from pathlib import Path

import pandas as pd

from src.train import train
from src.utils import load_config


def train_matrix(
    config: dict,
    seeds: list[int] | None = None,
    folds: list[int] | None = None,
    smoke: bool = False,
) -> pd.DataFrame:
    selected_seeds = seeds or [
        int(value) for value in config["training"].get("seeds", [config["training"]["seed"]])
    ]
    selected_folds: list[int | None]
    if folds is not None:
        selected_folds = folds
    elif "folds" in config["dataset"]:
        selected_folds = [int(value) for value in config["dataset"]["folds"]]
    else:
        selected_folds = [None]
    rows = []
    for fold in selected_folds:
        for seed in selected_seeds:
            run_config = copy.deepcopy(config)
            run_config["training"]["seed"] = seed
            if fold is not None:
                run_config["dataset"]["fold"] = fold
            if smoke:
                run_config["model"]["pretrained"] = False
                run_config["training"].update(
                    {"head_epochs": 1, "finetune_epochs": 1, "patience": 1}
                )
            result = train(run_config)
            rows.append(
                {
                    "dataset": run_config["dataset"]["name"],
                    "model": run_config["model"]["backbone"],
                    "seed": seed,
                    "fold": "" if fold is None else fold,
                    "run_id": result["run_id"],
                    "checkpoint": result["checkpoint"],
                    "best_val_macro_f1": result["best_val_macro_f1"],
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--folds", nargs="+", type=int)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--registry", type=Path, default=Path("results/checkpoint_registry.csv"))
    args = parser.parse_args()
    registry = train_matrix(
        load_config(args.config), seeds=args.seeds, folds=args.folds, smoke=args.smoke
    )
    args.registry.parent.mkdir(parents=True, exist_ok=True)
    if args.registry.exists():
        registry = pd.concat([pd.read_csv(args.registry), registry], ignore_index=True)
        registry = registry.drop_duplicates("checkpoint", keep="last")
    registry.to_csv(args.registry, index=False)
    print(f"wrote {len(registry)} checkpoint rows to {args.registry}")


if __name__ == "__main__":
    main()
