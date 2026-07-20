import json

import numpy as np
import pytest
import torch

from scripts import extract_embeddings
from scripts.extract_embeddings import _pool_hidden, _validate_vocabulary
from src.codonlm.model_tiny_gpt import TinyGPT


def _model(*, sep_id=3):
    torch.manual_seed(7)
    return TinyGPT(
        vocab_size=12,
        block_size=8,
        n_layer=2,
        n_head=2,
        n_embd=16,
        dropout=0.0,
        use_checkpoint=False,
        sep_id=sep_id,
        use_sdpa=True,
    ).eval()


def test_future_tokens_cannot_change_prefix_hidden_states():
    model = _model()
    first = torch.tensor([[1, 4, 5, 6, 7]])
    second = torch.tensor([[1, 4, 5, 8, 9]])
    with torch.no_grad():
        first_hidden = model.forward_hidden(first)
        second_hidden = model.forward_hidden(second)
    torch.testing.assert_close(first_hidden[:, :3], second_hidden[:, :3])


def test_previous_segment_cannot_change_later_segment_hidden_states():
    model = _model()
    first = torch.tensor([[1, 4, 3, 6, 7]])
    second = torch.tensor([[1, 9, 3, 6, 7]])
    with torch.no_grad():
        first_hidden = model.forward_hidden(first)
        second_hidden = model.forward_hidden(second)
    torch.testing.assert_close(first_hidden[:, 2:], second_hidden[:, 2:])


def test_pooling_rejects_unverified_model_api():
    class Unsupported(torch.nn.Module):
        pass

    ids = torch.tensor([[1, 2]])
    with pytest.raises(TypeError, match="verified forward_hidden"):
        _pool_hidden(Unsupported(), ids, ids.ne(0))


def test_shape_guided_pooling_requires_explicit_shapes():
    model = _model()
    model.use_shape_guidance = True
    ids = torch.tensor([[1, 2]])
    with pytest.raises(RuntimeError, match="shape-guided extraction requires"):
        _pool_hidden(model, ids, ids.ne(0))


def test_checkpoint_vocabulary_rows_must_match(tmp_path):
    vocabulary = tmp_path / "itos.txt"
    vocabulary.write_text("<PAD>\nA\nB\n")
    state = {
        "tok_emb.weight": torch.zeros(3, 4),
        "head.weight": torch.zeros(4, 4),
    }
    with pytest.raises(RuntimeError, match="output rows=4"):
        _validate_vocabulary(tuple(vocabulary.read_text().splitlines()), state, {}, vocabulary)


def test_cli_writes_verified_provenance_sidecar(tmp_path, monkeypatch):
    run_dir = tmp_path / "run"
    (run_dir / "checkpoints").mkdir(parents=True)
    tokens = ["<PAD>", "<BOS_CDS>", "<EOS_CDS>", "<SEP>", "ATG"] + [
        f"T{i}" for i in range(7)
    ]
    (run_dir / "itos.txt").write_text("\n".join(tokens) + "\n")
    model = _model()
    cfg = {
        "vocab_size": 12,
        "block_size": 8,
        "n_layer": 2,
        "n_head": 2,
        "n_embd": 16,
        "dropout": 0.0,
        "use_sdpa": True,
    }
    torch.save(
        {"model": model.state_dict(), "cfg": cfg},
        run_dir / "checkpoints" / "best.pt",
    )
    fasta = tmp_path / "input.fasta"
    fasta.write_text(">gene-1\nATG\n")
    output = tmp_path / "embeddings.npz"
    monkeypatch.setattr(extract_embeddings.Q, "dev", lambda: torch.device("cpu"))
    monkeypatch.setattr(
        "sys.argv",
        [
            "extract_embeddings",
            "--run_dir",
            str(run_dir),
            "--fasta",
            str(fasta),
            "--out",
            str(output),
        ],
    )
    extract_embeddings.main()

    with np.load(output, allow_pickle=True) as data:
        assert data["X"].shape == (1, 16)
    metadata = json.loads(output.with_suffix(".npz.metadata.json").read_text())
    assert metadata["validation_status"] == "causal_verified"
    assert metadata["mask_mode"] == "canonical_causal_segment"
    assert metadata["vocabulary"]["size"] == 12
    assert metadata["inputs"][0]["path"] == str(fasta.resolve())
