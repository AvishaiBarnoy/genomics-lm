import os
import yaml
import torch
import numpy as np
from pathlib import Path

from src.codonlm.model_tiny_gpt import TinyGPT
import scripts.expand_model as EM

def test_model_expansion_logic(tmp_path):
    # 1. Setup Source Configuration and Save Temporary Checkpoint
    src_config = {
        "vocab_size": 69,
        "block_size": 128,
        "n_layer": 2,
        "n_head": 2,
        "n_embd": 128,
        "dropout": 0.1,
    }
    
    src_model = TinyGPT(
        vocab_size=src_config["vocab_size"],
        block_size=src_config["block_size"],
        n_layer=src_config["n_layer"],
        n_head=src_config["n_head"],
        n_embd=src_config["n_embd"],
        dropout=src_config["dropout"],
    )
    
    src_checkpoint_path = tmp_path / "src_best.pt"
    src_payload = {
        "model": src_model.state_dict(),
        "cfg": src_config,
    }
    torch.save(src_payload, src_checkpoint_path)

    # 2. Setup Destination Configuration File
    dst_config = {
        "vocab_size": 69,
        "block_size": 128,
        "n_layer": 4,  # Expanded layers
        "n_head": 4,   # Expanded heads
        "n_embd": 256,  # Expanded dimension
        "dropout": 0.1,
        "use_swiglu": True,
        "termination_loss_enabled": True,
        "termination_n_classes": 5,
        "multi_offset_targets": [2, 4],
    }
    
    dst_config_path = tmp_path / "dst_config.yaml"
    with open(dst_config_path, "w") as f:
        yaml.safe_dump(dst_config, f)

    # 3. Invoke Model Expansion script
    out_checkpoint_path = tmp_path / "dst_expanded.pt"
    
    # Mock args
    class Args:
        src_checkpoint = str(src_checkpoint_path)
        dst_config = str(dst_config_path)
        out_checkpoint = str(out_checkpoint_path)
        
    from unittest.mock import patch
    with patch("scripts.expand_model.parse_args", return_value=Args()):
        EM.main()

    # 4. Verify Shape Alignment
    assert out_checkpoint_path.exists(), "Expanded checkpoint was not written"
    dst_ckpt = torch.load(out_checkpoint_path, map_location="cpu")
    
    assert "model" in dst_ckpt
    assert "cfg" in dst_ckpt
    assert dst_ckpt["cfg"]["n_embd"] == 256
    
    # Load into a real model and do a forward pass to check shapes
    target_model = TinyGPT(
        vocab_size=dst_config["vocab_size"],
        block_size=dst_config["block_size"],
        n_layer=dst_config["n_layer"],
        n_head=dst_config["n_head"],
        n_embd=dst_config["n_embd"],
        dropout=dst_config["dropout"],
        termination_aux=True,
        termination_n_classes=5,
        multi_offset_targets=[2, 4],
        use_swiglu=True,
    )
    
    # Verify exact loading
    target_model.load_state_dict(dst_ckpt["model"], strict=True)
    
    # Verify forward pass
    dummy_input = torch.randint(0, 69, (2, 64), dtype=torch.long)
    logits, next_loss, aux = target_model(dummy_input, targets=dummy_input, return_aux=True)
    
    assert logits.shape == (2, 64, 69)
    assert next_loss is not None
    assert "termination_logits" in aux
    assert aux["termination_logits"].shape == (2, 64, 5)
    assert "offset_logits" in aux
    assert 2 in aux["offset_logits"]
    assert aux["offset_logits"][2].shape == (2, 64, 69)
