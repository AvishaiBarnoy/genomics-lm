from pathlib import Path
from scripts._shared import resolve_run


def test_resolve_run_creates_layout(tmp_path, monkeypatch):
    # Use a temp RUNS_DIR to avoid touching the repo
    monkeypatch.setattr("scripts._shared.RUNS_DIR", Path(tmp_path) / "runs", raising=False)
    run_id, run_dir = resolve_run(run_id="unit_test_run")
    assert run_id == "unit_test_run"
    assert run_dir.exists()
    # Ensure standard subdirs are present
    assert (run_dir / "charts").exists()
    assert (run_dir / "tables").exists()


def test_load_model_resolves_consolidated(tmp_path, monkeypatch):
    import json
    import torch
    monkeypatch.setattr("scripts._shared.RUNS_DIR", Path(tmp_path) / "runs", raising=False)
    run_dir = tmp_path / "runs" / "test_run"
    run_dir.mkdir(parents=True)
    
    # Write mock meta.json
    meta = {
        "model_spec": {
            "model_type": "tiny_gpt",
            "vocab_size": 10,
            "block_size": 16,
            "n_layer": 1,
            "n_head": 1,
            "n_embd": 16
        }
    }
    with open(run_dir / "meta.json", "w") as f:
        json.dump(meta, f)
        
    # Write mock weights in checkpoints subfolder
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir()
    
    from src.codonlm.model_tiny_gpt import TinyGPT
    model = TinyGPT(vocab_size=10, block_size=16, n_layer=1, n_head=1, n_embd=16)
    torch.save(model.state_dict(), ckpt_dir / "weights.pt")
    
    # Load model
    from scripts._shared import load_model
    loaded_model, spec = load_model(run_dir, device="cpu")
    assert loaded_model is not None
    assert spec.vocab_size == 10


def test_load_model_dynamic_vocab_mismatch(tmp_path, monkeypatch):
    import json
    import torch
    monkeypatch.setattr("scripts._shared.RUNS_DIR", Path(tmp_path) / "runs", raising=False)
    run_dir = tmp_path / "runs" / "mismatch_run"
    run_dir.mkdir(parents=True)
    
    # Write mock meta.json with mismatched vocab_size = 99
    meta = {
        "model_spec": {
            "model_type": "tiny_gpt",
            "vocab_size": 99,
            "block_size": 16,
            "n_layer": 1,
            "n_head": 1,
            "n_embd": 16
        }
    }
    with open(run_dir / "meta.json", "w") as f:
        json.dump(meta, f)
        
    # Checkpoints weigh has vocab_size = 12
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir()
    
    from src.codonlm.model_tiny_gpt import TinyGPT
    model = TinyGPT(vocab_size=12, block_size=16, n_layer=1, n_head=1, n_embd=16)
    torch.save(model.state_dict(), ckpt_dir / "weights.pt")
    
    # Load model should dynamically adjust spec.vocab_size to 12
    from scripts._shared import load_model
    loaded_model, spec = load_model(run_dir, device="cpu")
    assert loaded_model is not None
    assert spec.vocab_size == 12


def test_load_codon_model_dynamic_vocab(tmp_path):
    import torch
    run_dir = tmp_path / "mismatch_run"
    run_dir.mkdir(parents=True)
    
    # Checkpoint has vocab_size = 15 saved in cfg, but actual weight is 8
    cfg = {
        "vocab_size": 15,
        "block_size": 16,
        "n_layer": 1,
        "n_head": 1,
        "n_embd": 16
    }
    
    from src.codonlm.model_tiny_gpt import TinyGPT
    model = TinyGPT(vocab_size=8, block_size=16, n_layer=1, n_head=1, n_embd=16)
    
    state = {
        "model": model.state_dict(),
        "cfg": cfg
    }
    torch.save(state, run_dir / "best.pt")
    (run_dir / "itos.txt").write_text(
        "\n".join(f"token_{index}" for index in range(7)) + "\n"
    )
    
    from src.codonlm.checkpoints import load_codon_model
    loaded_model, loaded_cfg, _ = load_codon_model(run_dir, device="cpu")
    assert loaded_model is not None
    assert loaded_cfg["vocab_size"] == 8
    compatibility = loaded_cfg["vocabulary_compatibility"]
    assert compatibility["legacy_adaptation"] is True
    assert compatibility["configured_size"] == 15
    assert compatibility["embedding_rows"] == 8
    assert compatibility["artifact_size"] == 7


