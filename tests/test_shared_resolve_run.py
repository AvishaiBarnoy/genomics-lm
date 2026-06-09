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


