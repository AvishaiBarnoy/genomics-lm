import json
from pathlib import Path
import subprocess


def test_compare_runs_scan_minimal(tmp_path: Path, monkeypatch):
    # Create minimal metrics for two runs
    repo = Path.cwd()
    runs = repo / "runs"
    runA = runs / f"{tmp_path.name}_runA"
    runB = runs / f"{tmp_path.name}_runB"
    runA.mkdir(parents=True, exist_ok=True)
    runB.mkdir(parents=True, exist_ok=True)
    (runA / "meta.json").write_text(json.dumps({"val_ppl": 55.0, "test_ppl": 60.0}))
    (runB / "meta.json").write_text(json.dumps({"val_ppl": 45.0, "test_ppl": 50.0}))

    out_csv = tmp_path / "summary.csv"

    try:
        # Ensure compare runs without args scans runs/*
        subprocess.run(["python", "-m", "scripts.compare_runs", "--out", str(out_csv)], check=True)
    finally:
        for path in (runA / "meta.json", runB / "meta.json"):
            path.unlink(missing_ok=True)
        runA.rmdir()
        runB.rmdir()

    assert out_csv.exists()
    text = out_csv.read_text()
    assert runA.name in text and runB.name in text
