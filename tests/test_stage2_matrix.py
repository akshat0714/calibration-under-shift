from __future__ import annotations

import importlib
import json
import subprocess
from pathlib import Path

import pandas as pd
import pytest

from experiments.evaluate_clean_matrix import EXPECTED_STAGE1_MEMBERS
from experiments.run_grid import KEY_COLUMNS

stage2 = importlib.import_module("experiments.run_stage2_matrix")


def _write_exact_registry(tmp_path: Path) -> Path:
    rows = []
    for member in sorted(EXPECTED_STAGE1_MEMBERS):
        run_id = member.label().replace("/", "-").replace("=", "-")
        checkpoint = tmp_path / "checkpoints" / f"{run_id}.pt"
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        checkpoint.write_bytes(b"checkpoint")
        rows.append(
            {
                "dataset": member.dataset,
                "model": member.model,
                "seed": member.seed,
                "fold": "" if member.fold is None else member.fold,
                "run_id": run_id,
                "checkpoint": str(checkpoint),
                "best_val_macro_f1": 0.9,
            }
        )
    registry = tmp_path / "checkpoint_registry-stage1.csv"
    pd.DataFrame(rows).to_csv(registry, index=False)
    return registry


def _row(dataset: str, model: str, index: int, revision: str) -> dict:
    row = {
        "dataset": dataset,
        "model": model,
        "seed": index,
        "fold": "",
        "corruption": "clean",
        "severity": 0,
        "method": "raw_softmax",
        "metric": f"metric-{index}",
        "value": 0.5,
        "evaluation_git_revision": revision,
    }
    assert set(KEY_COLUMNS).issubset(row)
    return row


def test_frozen_stage2_cardinalities():
    assert stage2.EXPECTED_GROUP_ROWS == {
        ("smids", "resnet50"): 14_565,
        ("smids", "xception"): 8_295,
        ("smids", "mobilenet_v3_large"): 8_295,
        ("hushem", "resnet50"): 14_385,
    }
    assert sum(stage2.EXPECTED_GROUP_ROWS.values()) == stage2.EXPECTED_TOTAL_ROWS == 45_540
    assert stage2.EXPECTED_DETAIL_JSONS == 17


def test_orchestrator_runs_groups_in_order_and_publishes_only_after_strict_validation(
    tmp_path, monkeypatch
):
    repository = Path(__file__).resolve().parents[1]
    registry = _write_exact_registry(tmp_path)
    output = tmp_path / "metrics.csv"
    parts_dir = tmp_path / "parts"
    details_dir = tmp_path / "details"
    revision = "a" * 40
    monkeypatch.setattr(stage2, "require_clean_git_revision", lambda *args, **kwargs: revision)
    validated: list[tuple[int, dict]] = []
    monkeypatch.setattr(
        stage2,
        "validate_complete_grid",
        lambda metrics, protocol: validated.append((len(metrics), protocol)),
    )
    commands: list[list[str]] = []
    config_identities = {
        "smids_resnet50.yaml": ("smids", "resnet50"),
        "smids_xception.yaml": ("smids", "xception"),
        "smids_mobilenetv3.yaml": ("smids", "mobilenet_v3_large"),
        "hushem_resnet50.yaml": ("hushem", "resnet50"),
    }

    def fake_run(command, *, cwd, check):
        assert cwd == repository
        assert check is True
        commands.append(command)
        config = Path(command[command.index("--config") + 1]).name
        identity = config_identities[config]
        part = Path(command[command.index("--output") + 1])
        count = stage2.EXPECTED_GROUP_ROWS[identity]
        pd.DataFrame(
            [_row(identity[0], identity[1], index, revision) for index in range(count)]
        ).to_csv(part, index=False)
        shared_details = Path(command[command.index("--details-dir") + 1])
        shared_details.mkdir(parents=True, exist_ok=True)
        start = command.index("--checkpoints") + 1
        stop = command.index("--output")
        for checkpoint in command[start:stop]:
            run_id = Path(checkpoint).stem
            (shared_details / f"{run_id}.json").write_text(
                json.dumps({"evaluation_git_revision": revision}), encoding="utf-8"
            )
        if identity == ("smids", "resnet50"):
            (shared_details / stage2.ENSEMBLE_DETAIL_NAME).write_text(
                json.dumps({"evaluation_git_revision": revision}), encoding="utf-8"
            )
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(stage2.subprocess, "run", fake_run)

    metrics = stage2.run_stage2_matrix(
        registry_path=registry,
        output=output,
        parts_dir=parts_dir,
        details_dir=details_dir,
        device="cuda",
        root=repository,
        python="python-under-test",
    )

    assert len(metrics) == len(pd.read_csv(output)) == stage2.EXPECTED_TOTAL_ROWS
    assert [Path(command[command.index("--config") + 1]).name for command in commands] == [
        "smids_resnet50.yaml",
        "smids_xception.yaml",
        "smids_mobilenetv3.yaml",
        "hushem_resnet50.yaml",
    ]
    assert all(command[0] == "python-under-test" for command in commands)
    assert all(command[command.index("--device") + 1] == "cuda" for command in commands)
    assert all(
        Path(command[command.index("--details-dir") + 1]) == details_dir for command in commands
    )
    assert validated and validated[0][0] == stage2.EXPECTED_TOTAL_ROWS
    assert validated[0][1]["aggregation"]["severities"] == [1, 2, 3, 4, 5]
    assert not output.with_suffix(".csv.tmp").exists()


def test_duplicate_tidy_keys_are_rejected_before_output_replacement(tmp_path):
    duplicate = _row("smids", "resnet50", 1, "a" * 40)
    metrics = pd.DataFrame([duplicate, duplicate])
    output = tmp_path / "metrics.csv"
    output.write_text("sentinel\n", encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate tidy keys"):
        stage2.reject_duplicate_tidy_keys(metrics)

    assert output.read_text(encoding="utf-8") == "sentinel\n"


def test_missing_checkpoint_fails_before_any_group_runs(tmp_path, monkeypatch):
    repository = Path(__file__).resolve().parents[1]
    registry = _write_exact_registry(tmp_path)
    frame = pd.read_csv(registry)
    Path(frame.iloc[0]["checkpoint"]).unlink()
    monkeypatch.setattr(stage2, "require_clean_git_revision", lambda *args, **kwargs: "a" * 40)
    called = False

    def unexpected_run(*args, **kwargs):
        nonlocal called
        called = True

    monkeypatch.setattr(stage2, "run_group", unexpected_run)

    with pytest.raises(FileNotFoundError, match="checkpoint paths are missing"):
        stage2.run_stage2_matrix(registry_path=registry, root=repository)

    assert called is False


def test_git_revision_gate_rejects_tracked_and_unrelated_untracked_changes(tmp_path):
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=tmp_path, check=True)
    tracked = tmp_path / "tracked.txt"
    tracked.write_text("clean\n", encoding="utf-8")
    subprocess.run(["git", "add", "tracked.txt"], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-qm", "fixture"], cwd=tmp_path, check=True)

    revision = stage2.require_clean_git_revision(tmp_path)
    assert len(revision) == 40

    generated = tmp_path / "results/parts/group.csv"
    generated.parent.mkdir(parents=True)
    generated.write_text("generated\n", encoding="utf-8")
    assert (
        stage2.require_clean_git_revision(tmp_path, allowed_untracked=(Path("results/parts"),))
        == revision
    )

    unrelated = tmp_path / "unversioned.py"
    unrelated.write_text("pass\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="dirty or contains unversioned"):
        stage2.require_clean_git_revision(tmp_path, allowed_untracked=(Path("results/parts"),))
    unrelated.unlink()

    tracked.write_text("dirty\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="dirty or contains unversioned"):
        stage2.require_clean_git_revision(tmp_path, allowed_untracked=(Path("results/parts"),))
