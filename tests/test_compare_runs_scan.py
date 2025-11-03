import json
from pathlib import Path
import subprocess


def test_compare_runs_scan_minimal(tmp_path: Path, monkeypatch):
    # Create minimal metrics for two runs
    repo = Path.cwd()
    scores = repo / "outputs" / "scores"
    (scores / "compare").mkdir(parents=True, exist_ok=True)
    runA = scores / "runA"
    runB = scores / "runB"
    runA.mkdir(parents=True, exist_ok=True)
    runB.mkdir(parents=True, exist_ok=True)
    (runA / "metrics.json").write_text(json.dumps({"val_ppl": 55.0, "test_ppl": 60.0}))
    (runB / "metrics.json").write_text(json.dumps({"val_ppl": 45.0, "test_ppl": 50.0}))

    # Ensure compare runs without args scans outputs/scores/*
    subprocess.run(["python", "-m", "scripts.compare_runs"], check=True)

    out_csv = scores / "compare" / "summary.csv"
    assert out_csv.exists()
    text = out_csv.read_text()
    assert "runA" in text and "runB" in text

