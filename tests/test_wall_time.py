import os
import sys
import yaml
import numpy as np
import pytest
import subprocess
import json
from pathlib import Path

def test_train_codon_lm_wall_time_limit(tmp_path):
    # Create tiny config
    config_data = {
        "vocab_size": 69,
        "block_size": 8,
        "n_layer": 1,
        "n_head": 1,
        "n_embd": 16,
        "dropout": 0.0,
        "batch_size": 1,
        "grad_accum_steps": 1,
        "lr": 0.001,
        "weight_decay": 0.0,
        "epochs": 10,
        "optimizer": "adamw",
        "amp": False,
        "use_checkpoint": False,
        "scheduler": "cosine",
        "early_stop_patience": 5,
        "out_dir": str(tmp_path / "checkpoints"),
        "scores_dir": str(tmp_path / "scores"),
        "log_csv": "curves.csv",
        "seed": 42,
        "max_time_minutes": 0.0001,  # extremely small limit: 6ms
    }
    
    config_file = tmp_path / "config.yaml"
    with open(config_file, "w") as f:
        yaml.safe_dump(config_data, f)
        
    # Generate dummy npz datasets
    train_npz = tmp_path / "train.npz"
    val_npz = tmp_path / "val.npz"
    test_npz = tmp_path / "test.npz"
    
    # Needs to have at least a few steps so wall-time limit can be caught mid-epoch
    # We write 10 samples of length 8 with non-zero indices so they aren't ignored by ignore_index=0
    x_data = np.ones((10, 8), dtype=np.int64)
    y_data = np.ones((10, 8), dtype=np.int64)
    np.savez_compressed(train_npz, X=x_data, Y=y_data)
    np.savez_compressed(val_npz, X=x_data, Y=y_data)
    np.savez_compressed(test_npz, X=x_data, Y=y_data)
    
    # Run train_codon_lm.py in a subprocess from workspace root with a custom run_id
    run_id = "test_run_wall_time_temp"
    cmd = [
        sys.executable,
        "-m", "src.codonlm.train_codon_lm",
        "--config", str(config_file),
        "--train_npz", str(train_npz),
        "--val_npz", str(val_npz),
        "--test_npz", str(test_npz),
        "--run_id", run_id,
    ]
    
    workspace_root = Path(__file__).resolve().parents[1]
    runs_dir = Path(workspace_root) / "runs" / run_id
    
    # Ensure any previous temp run dir is clean
    import shutil
    if runs_dir.exists():
        shutil.rmtree(runs_dir)
        
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=workspace_root)
        print("STDOUT:", result.stdout)
        print("STDERR:", result.stderr)
        
        assert result.returncode == 0
        
        # Verify that checkpoints directory contains last.pt
        last_pt = runs_dir / "checkpoints" / "last.pt"
        assert last_pt.exists()
        
        # Verify metadata status
        meta_json = runs_dir / "checkpoints" / "meta.json"
        assert meta_json.exists()
        
        meta = json.loads(meta_json.read_text())
        assert meta["status"] == "stopped"
    finally:
        if runs_dir.exists():
            shutil.rmtree(runs_dir)
