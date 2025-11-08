from pathlib import Path
from scripts._shared import resolve_run, RUNS_DIR


def test_resolve_run_creates_layout(tmp_path, monkeypatch):
    # Use a temp RUNS_DIR to avoid touching the repo
    monkeypatch.setattr("scripts._shared.RUNS_DIR", Path(tmp_path) / "runs", raising=False)
    run_id, run_dir = resolve_run(run_id="unit_test_run")
    assert run_id == "unit_test_run"
    assert run_dir.exists()
    # Ensure standard subdirs are present
    assert (run_dir / "charts").exists()
    assert (run_dir / "tables").exists()

