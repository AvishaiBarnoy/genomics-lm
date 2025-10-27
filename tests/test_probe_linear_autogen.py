import numpy as np

import scripts.probe_linear as PL
import scripts._shared as SH


def test_probe_linear_autogenerates_labels(tmp_path, monkeypatch):
    # Set temporary runs dir
    monkeypatch.setattr(SH, "RUNS_DIR", tmp_path / "runs", raising=True)

    run_id = "TESTID"
    paths = SH.ensure_run_layout(run_id)
    run_dir = paths["run"]
    (run_dir / "itos.txt").write_text("ATG\nTAA\nAAA\n")

    artifacts = {
        "token_embeddings": np.random.randn(3, 8).astype(np.float32),
        "pos_embeddings": np.zeros((0,), dtype=np.float32),
        "logits": np.zeros((0,), dtype=np.float32),
        "probs": np.zeros((0,), dtype=np.float32),
        "attn": np.zeros((0,), dtype=np.float32),
        "token_counts": np.zeros((3,), dtype=np.int64),
        "first_token_counts": np.zeros((3,), dtype=np.int64),
        "val_inputs": np.zeros((0,), dtype=np.int64),
        "val_targets": np.zeros((0,), dtype=np.int64),
    }
    np.savez_compressed(run_dir / "artifacts.npz", **artifacts)
    # Run the probe; it should auto-write probe_labels.csv and results
    PL.main([run_id])
    results = (run_dir / "tables" / "probe_results.csv")
    assert results.exists(), "probe_results.csv was not created"

