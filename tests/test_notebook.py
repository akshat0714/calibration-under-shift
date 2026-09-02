from __future__ import annotations

from pathlib import Path

import nbformat
import pytest
from nbclient import NotebookClient

REPOSITORY = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = REPOSITORY / "notebooks/01_walkthrough.ipynb"
CI_PATH = REPOSITORY / ".github/workflows/ci.yml"


def _notebook_source() -> str:
    notebook = nbformat.read(NOTEBOOK_PATH, as_version=4)
    return "\n".join(str(cell.source) for cell in notebook.cells)


def _setup_source() -> str:
    notebook = nbformat.read(NOTEBOOK_PATH, as_version=4)
    return str(next(cell.source for cell in notebook.cells if cell.id == "cell-02"))


def _execution_text(notebook: nbformat.NotebookNode) -> str:
    parts: list[str] = []
    for cell in notebook.cells:
        for output in cell.get("outputs", []):
            if output.get("output_type") == "stream":
                parts.append(str(output.get("text", "")))
            data = output.get("data", {})
            if "text/plain" in data:
                parts.append(str(data["text/plain"]))
    return "\n".join(parts)


def _execute_setup() -> dict[str, object]:
    namespace: dict[str, object] = {}
    exec(compile(_setup_source(), str(NOTEBOOK_PATH), "exec"), namespace)
    return namespace


def test_walkthrough_defaults_to_real_results_and_guards_demo_overrides():
    source = _notebook_source()

    assert 'os.getenv("CALIBRATION_RESULTS_DIR")' in source
    assert 'os.getenv("CALIBRATION_NOTEBOOK_METRICS")' in source
    assert 'os.getenv("CALIBRATION_NOTEBOOK_THRESHOLDS")' in source
    assert "demo mode requires CALIBRATION_NOTEBOOK_METRICS" in source
    assert "fixture path overrides require CALIBRATION_NOTEBOOK_ALLOW_DEMO=1" in source
    assert 'ATTRIBUTION / "attribution_stability.csv"' in source
    assert 'FIGURES / "attribution_stability.csv"' not in source
    assert "The early-warning hypothesis was not supported at these thresholds" in source
    assert source.count("Secondary and exploratory") >= 3


def test_walkthrough_accepts_standard_results_directory_override(tmp_path, monkeypatch):
    monkeypatch.setenv("CALIBRATION_RESULTS_DIR", str(tmp_path))
    for name in (
        "CALIBRATION_NOTEBOOK_ALLOW_DEMO",
        "CALIBRATION_NOTEBOOK_METRICS",
        "CALIBRATION_NOTEBOOK_THRESHOLDS",
    ):
        monkeypatch.delenv(name, raising=False)
    namespace = _execute_setup()

    assert namespace["RESULTS"] == tmp_path.resolve()
    assert namespace["METRICS_PATH"] == (tmp_path / "metrics.csv").resolve()
    assert namespace["THRESHOLDS_PATH"] == (tmp_path / "thresholds.csv").resolve()
    assert namespace["ATTRIBUTION"] == tmp_path.resolve() / "attribution"


def test_walkthrough_rejects_fixture_paths_without_demo_opt_in(tmp_path, monkeypatch):
    monkeypatch.delenv("CALIBRATION_NOTEBOOK_ALLOW_DEMO", raising=False)
    monkeypatch.setenv("CALIBRATION_NOTEBOOK_METRICS", str(tmp_path / "metrics.csv"))
    monkeypatch.setenv("CALIBRATION_NOTEBOOK_THRESHOLDS", str(tmp_path / "thresholds.csv"))

    with pytest.raises(RuntimeError, match="fixture path overrides require"):
        _execute_setup()


def test_walkthrough_demo_opt_in_requires_both_fixture_paths(monkeypatch):
    monkeypatch.setenv("CALIBRATION_NOTEBOOK_ALLOW_DEMO", "1")
    monkeypatch.setenv("CALIBRATION_NOTEBOOK_METRICS", "results/demo_metrics.csv")
    monkeypatch.delenv("CALIBRATION_NOTEBOOK_THRESHOLDS", raising=False)

    with pytest.raises(RuntimeError, match="demo mode requires"):
        _execute_setup()


def test_walkthrough_executes_default_canonical_results(monkeypatch):
    for name in (
        "CALIBRATION_RESULTS_DIR",
        "CALIBRATION_NOTEBOOK_ALLOW_DEMO",
        "CALIBRATION_NOTEBOOK_METRICS",
        "CALIBRATION_NOTEBOOK_THRESHOLDS",
    ):
        monkeypatch.delenv(name, raising=False)

    notebook = nbformat.read(NOTEBOOK_PATH, as_version=4)
    executed = NotebookClient(notebook, timeout=120, kernel_name="python3").execute(
        cwd=str(REPOSITORY)
    )
    output = _execution_text(executed)

    assert "validated complete prespecified SMIDS/HuSHeM grid" in output
    assert "In the primary prespecified result" in output
    assert "10 comparisons with both crossings observed" in output
    assert "6 further signals never crossed" in output


def test_ci_demo_notebook_uses_explicit_fixture_paths_without_overwrite():
    workflow = CI_PATH.read_text(encoding="utf-8")

    assert "cp results/demo_metrics.csv results/metrics.csv" not in workflow
    assert "CALIBRATION_NOTEBOOK_ALLOW_DEMO=1" in workflow
    assert "CALIBRATION_NOTEBOOK_METRICS=results/demo_metrics.csv" in workflow
    assert "CALIBRATION_NOTEBOOK_THRESHOLDS=results/demo_thresholds.csv" in workflow
    assert "ruff format --check ." in workflow
    assert "bash -n run.sh scripts/download_data.sh" in workflow
