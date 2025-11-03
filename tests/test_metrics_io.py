from pathlib import Path

from src.codonlm.metrics_io import read_metrics, write_merge_metrics


def test_write_merge_metrics_roundtrip(tmp_path: Path):
    p = tmp_path / "metrics.json"
    out = write_merge_metrics(p, {"a": 1, "b": 2})
    assert out["a"] == 1 and out["b"] == 2
    out2 = write_merge_metrics(p, {"b": 3, "c": 4})
    # existing preserved except keys explicitly updated
    assert out2 == {"a": 1, "b": 3, "c": 4}
    data = read_metrics(p)
    assert data == out2

