import os
import sys
import yaml
import pytest
import subprocess
import json
import shutil
from pathlib import Path

def test_train_multi_task_wall_time_limit(tmp_path):
    # Create tiny train and val jsonl files
    train_data = tmp_path / "train.jsonl"
    val_data = tmp_path / "val.jsonl"
    
    sample = {
        "sequence": "MKLVFLVLLFLGAVG",
        "pfam_id": 0,
        "ec_id": 0,
        "stability_id": 0
    }
    
    with open(train_data, "w") as f:
        # Write multiple samples to allow multiple batches/steps
        for _ in range(20):
            f.write(json.dumps(sample) + "\n")
            
    with open(val_data, "w") as f:
        for _ in range(5):
            f.write(json.dumps(sample) + "\n")
            
    # Create tiny config
    config_data = {
        "device": "cpu",
        "train_data": str(train_data),
        "val_data": str(val_data),
        "block_size": 32,
        "n_layer": 1,
        "n_head": 1,
        "n_embd": 16,
        "dropout": 0.0,
        "batch_size": 1,
        "grad_accum_steps": 1,
        "epochs": 10,
        "lr": 0.001,
        "out_dir": str(tmp_path / "checkpoints"),
        "max_time_minutes": 0.0001,  # extremely small limit: 6ms
    }
    
    config_file = tmp_path / "config.yaml"
    with open(config_file, "w") as f:
        yaml.safe_dump(config_data, f)
        
    # Run train_multi_task.py in a subprocess from workspace root
    cmd = [
        sys.executable,
        "-m", "src.protein_lm.train_multi_task",
        "--config", str(config_file),
    ]
    
    workspace_root = "/Users/User/github/genomics-lm"
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=workspace_root)
    print("STDOUT:", result.stdout)
    print("STDERR:", result.stderr)
    
    assert result.returncode == 0
    
    # Verify that checkpoints directory contains last_critic.pt
    last_critic_pt = tmp_path / "checkpoints" / "last_critic.pt"
    assert last_critic_pt.exists()
