import torch
import numpy as np
from scripts.mine_motifs import main as mine_main
from scripts._shared import write_meta

def test_mine_motifs_cli(tmp_path):
    # Setup a mock run directory
    run_id = "test_run"
    run_dir = tmp_path / "runs" / run_id
    run_dir.mkdir(parents=True)
    
    # Mock meta.json
    meta = {
        "model_spec": {
            "model_type": "tiny_gpt",
            "vocab_size": 10,
            "block_size": 32,
            "n_layer": 1,
            "n_head": 1,
            "n_embd": 16
        }
    }
    write_meta(run_dir, meta)
    
    # Mock itos.txt
    (run_dir / "itos.txt").write_text("\n".join([f"tok{i}" for i in range(10)]))
    
    # Mock weights.pt
    from src.codonlm.model_tiny_gpt import TinyGPT
    model = TinyGPT(10, 32, n_layer=1, n_head=1, n_embd=16)
    torch.save({"model": model.state_dict()}, run_dir / "weights.pt")
    
    # Mock data.npz
    data_path = tmp_path / "data.npz"
    X = np.random.randint(0, 10, (5, 32))
    np.savez(data_path, X=X)
    
    # Run CLI (mocking args)
    from unittest.mock import patch
    
    # Patch RUNS_DIR in _shared to point to tmp_path/runs
    with patch("scripts._shared.RUNS_DIR", tmp_path / "runs"):
        with patch("sys.argv", [
            "mine_motifs.py", 
            "--run_id", run_id, 
            "--data_npz", str(data_path), 
            "--n_samples", "2", 
            "--window_size", "5", 
            "--stride", "5", 
            "--n_clusters", "2"
        ]):
            mine_main()
            
    # Verify output
    out_path = run_dir / "motif_mining" / "clusters.npz"
    assert out_path.exists()
    
    data = np.load(out_path)
    assert "labels" in data
    assert "centers" in data
    assert data["window_size"] == 5
