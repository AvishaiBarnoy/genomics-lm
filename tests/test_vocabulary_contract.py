from __future__ import annotations

import hashlib

import numpy as np
import pytest
import torch

from src.codonlm.model_tiny_gpt import TinyGPT
from src.codonlm.training.vocabulary import (
    VocabularyContractError,
    load_itos,
    resolve_vocabulary_contract,
    snapshot_vocabulary,
    validate_resume_checkpoint,
)


TOKENS = ("<PAD>", "<BOS_CDS>", "<EOS_CDS>", "<SEP>", "AAA")


def _write_vocab(path, tokens=TOKENS):
    path.write_text("\n".join(tokens) + "\n")
    return path


def test_matching_fixed_dataset_contract_and_snapshot(tmp_path):
    vocab = _write_vocab(tmp_path / "itos.txt")
    dataset = tmp_path / "train.npz"
    np.savez_compressed(
        dataset,
        X=np.array([[1, 4, 0]], dtype=np.int32),
        Y=np.array([[4, 2, 0]], dtype=np.int32),
    )

    contract = resolve_vocabulary_contract(
        [dataset], configured_path=vocab, configured_size=len(TOKENS)
    )
    snapshot = snapshot_vocabulary(contract, tmp_path / "run" / "itos.txt")

    assert contract.tokens == TOKENS
    assert contract.size == 5
    assert contract.dataset_bounds[0].minimum == 0
    assert contract.dataset_bounds[0].maximum == 4
    assert contract.sha256 == hashlib.sha256(vocab.read_bytes()).hexdigest()
    assert snapshot.read_text() == vocab.read_text()


def test_dynamic_and_npy_sidecar_bounds_are_supported(tmp_path):
    _write_vocab(tmp_path / "itos.txt")
    dynamic = tmp_path / "dynamic.npz"
    np.savez_compressed(
        dynamic,
        X=np.array([1, 4, 2], dtype=np.int32),
        lengths=np.array([3], dtype=np.int32),
    )
    sidecar = tmp_path / "sidecar.npz"
    np.savez_compressed(sidecar, X=np.array([[99]], dtype=np.int32))
    np.save(tmp_path / "sidecar_X.npy", np.array([[1, 4]], dtype=np.int32))
    np.save(tmp_path / "sidecar_Y.npy", np.array([[4, 2]], dtype=np.int32))

    contract = resolve_vocabulary_contract(
        [dynamic, sidecar], configured_path=None, configured_size=5
    )

    assert [bound.maximum for bound in contract.dataset_bounds] == [4, 4]


@pytest.mark.parametrize(
    ("configured_size", "data", "message"),
    [
        pytest.param(6, [1, 4], "Configured vocab_size=6", id="stale-config"),
        pytest.param(5, [1, 5], "contains token ID 5", id="out-of-range"),
        pytest.param(5, [-1, 2], "negative token ID -1", id="negative"),
    ],
)
def test_contract_rejects_mismatches(tmp_path, configured_size, data, message):
    _write_vocab(tmp_path / "itos.txt")
    dataset = tmp_path / "train.npz"
    values = np.asarray([data], dtype=np.int32)
    np.savez_compressed(dataset, X=values, Y=values)

    with pytest.raises(VocabularyContractError, match=message):
        resolve_vocabulary_contract(
            [dataset], configured_path=None, configured_size=configured_size
        )


def test_vocab_rejects_duplicates_and_empty_ids(tmp_path):
    duplicate = _write_vocab(tmp_path / "duplicate.txt", ("A", "B", "A"))
    empty = _write_vocab(tmp_path / "empty.txt", ("A", "", "B"))

    with pytest.raises(VocabularyContractError, match="duplicate"):
        load_itos(duplicate)
    with pytest.raises(VocabularyContractError, match="empty token IDs"):
        load_itos(empty)


def test_resume_requires_exact_embedding_output_and_checkpoint_vocab(tmp_path):
    vocab = _write_vocab(tmp_path / "itos.txt")
    dataset = tmp_path / "train.npz"
    np.savez_compressed(dataset, X=np.array([[1, 4]]), Y=np.array([[4, 2]]))
    contract = resolve_vocabulary_contract(
        [dataset], configured_path=vocab, configured_size=5
    )
    model = TinyGPT(5, 4, n_layer=1, n_head=1, n_embd=4, dropout=0.0)
    checkpoint = tmp_path / "matching.pt"
    torch.save(
        {
            "model": model.state_dict(),
            "cfg": {
                "vocab_size": 5,
                "vocabulary": {"sha256": contract.sha256},
            },
        },
        checkpoint,
    )

    validate_resume_checkpoint(checkpoint, contract)

    legacy = TinyGPT(6, 4, n_layer=1, n_head=1, n_embd=4, dropout=0.0)
    legacy_checkpoint = tmp_path / "legacy.pt"
    torch.save(
        {"model": legacy.state_dict(), "cfg": {"vocab_size": 6}},
        legacy_checkpoint,
    )
    with pytest.raises(VocabularyContractError, match="Use transfer_from"):
        validate_resume_checkpoint(legacy_checkpoint, contract)


def test_resume_rejects_different_dataset_identity(tmp_path):
    vocab = _write_vocab(tmp_path / "itos.txt")
    dataset = tmp_path / "train.npz"
    np.savez(dataset, X=np.array([[0, 1]]), Y=np.array([[1, 0]]))
    contract = resolve_vocabulary_contract(
        [dataset], configured_path=vocab, configured_size=None
    )
    checkpoint = tmp_path / "checkpoint.pt"
    torch.save(
        {
            "model": {
                "tok_emb.weight": torch.zeros(contract.size, 4),
                "head.weight": torch.zeros(contract.size, 4),
            },
            "cfg": {
                "vocab_size": contract.size,
                "vocabulary": {"sha256": contract.sha256},
                "dataset_manifest": {"dataset_id": "dataset-a"},
            },
        },
        checkpoint,
    )
    with pytest.raises(VocabularyContractError, match="current dataset_id='dataset-b'"):
        validate_resume_checkpoint(checkpoint, contract, dataset_id="dataset-b")
