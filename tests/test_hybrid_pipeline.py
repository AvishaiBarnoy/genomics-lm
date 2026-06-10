from __future__ import annotations

import json
import subprocess
from pathlib import Path
import numpy as np
import pytest
import yaml

def test_hybrid_pipeline_end_to_end(tmp_path):
    # 1. Setup mock/temporary directories and config
    config_data = {
        "block_size": 128,
        "windows_per_seq": 1,
        "val_frac": 0.2,
        "test_frac": 0.2,
        "datasets": [
            {
                "name": "test_ds",
                "gbff": "data/raw/Enterobacteriaceae/GCF_000005845.gbff",
                "min_len": 90,
            }
        ]
    }
    
    config_file = tmp_path / "test_config.yaml"
    config_file.write_text(yaml.dump(config_data))
    
    run_dir = tmp_path / "runs" / "test_run"
    run_dir.mkdir(parents=True, exist_ok=True)
    
    # 2. Run pipeline_prepare_hybrid as a subprocess
    cmd = [
        "python",
        "-m",
        "src.codonlm.pipeline_prepare_hybrid",
        "--config",
        str(config_file),
        "--run-id",
        "test_run",
        "--run-dir",
        str(run_dir),
        "--upstream",
        "30",
        "--downstream",
        "60",
        "--force",
    ]
    
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 0, f"Pipeline preparation failed: {res.stderr}"
    
    # 3. Verify outputs
    # Manifest files
    combined_manifest_path = run_dir / "combined_manifest.json"
    assert combined_manifest_path.exists()
    
    manifest = json.loads(combined_manifest_path.read_text())
    assert "train" in manifest
    assert "val" in manifest
    assert "test" in manifest
    
    # NPZ files exist
    train_npz = Path(manifest["train"])
    val_npz = Path(manifest["val"])
    test_npz = Path(manifest["test"])
    
    assert train_npz.exists()
    assert val_npz.exists()
    assert test_npz.exists()
    
    # Check shape & bounds of loaded arrays
    with np.load(train_npz) as data:
        assert "X" in data
        assert "Y" in data
        X = data["X"]
        Y = data["Y"]
        assert X.ndim == 2
        assert X.shape[1] == 128
        assert Y.shape == X.shape
        
        # Verify tokens inside valid vocab range (0-73)
        assert np.all((X >= 0) & (X < 74))
