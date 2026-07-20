import json
from pathlib import Path

import numpy as np
import pytest

from scripts.eval_ppl_baselines import evaluate_baselines, fit_baselines
from src.codonlm.training.vocabulary import (
    VocabularyContractError,
    resolve_vocabulary_contract,
)


def _write_vocab(directory: Path, size: int = 6):
    (directory / "itos.txt").write_text("\n".join(["<PAD>", "<BOS>", "A", "B", "C", "<EOS>"]) + "\n")


def _formats(tmp_path: Path):
    x = np.array([[1, 2, 3, 4], [1, 3, 2, 4]], dtype=np.int32)
    y = np.array([[2, 3, 4, 0], [3, 2, 4, 0]], dtype=np.int32)
    fixed = tmp_path / "fixed.npz"
    np.savez(fixed, X=x, Y=y)
    dynamic = tmp_path / "dynamic.npz"
    flat = np.array([1, 2, 3, 4, 1, 3, 2, 4], dtype=np.int32)
    np.savez(dynamic, X=flat, lengths=np.array([4, 4], dtype=np.int32))
    mmap = tmp_path / "mmap.npz"
    mmap.touch()
    np.save(tmp_path / "mmap_X.npy", flat)
    np.save(tmp_path / "mmap_lengths.npy", np.array([4, 4], dtype=np.int32))
    return fixed, dynamic, mmap


def test_identical_streams_match_across_storage_formats(tmp_path):
    _write_vocab(tmp_path)
    fixed, dynamic, mmap = _formats(tmp_path)
    expected = None
    for path in (fixed, dynamic, mmap):
        counts = fit_baselines(path, 6)
        results, tokens, best = evaluate_baselines(path, counts, 6)
        comparable = json.dumps([results, tokens, best], sort_keys=True)
        expected = expected or comparable
        assert comparable == expected
        assert tokens == 6


def test_empty_targets_fail_closed(tmp_path):
    path = tmp_path / "empty.npz"
    np.savez(path, X=np.zeros((1, 2), dtype=np.int32), Y=np.zeros((1, 2), dtype=np.int32))
    with pytest.raises(ValueError, match="no evaluable"):
        fit_baselines(path, 6)


def test_vocabulary_mismatch_fails_closed(tmp_path):
    _write_vocab(tmp_path)
    path = tmp_path / "invalid.npz"
    np.savez(path, X=np.array([[1, 9]]), Y=np.array([[9, 0]]))
    with pytest.raises(VocabularyContractError, match="token ID 9"):
        resolve_vocabulary_contract([path], configured_path=None, configured_size=None)


def test_unseen_contexts_are_smoothed(tmp_path):
    train = tmp_path / "train.npz"
    test = tmp_path / "test.npz"
    np.savez(train, X=np.array([[1, 2]]), Y=np.array([[2, 0]]))
    np.savez(test, X=np.array([[4, 3]]), Y=np.array([[3, 0]]))
    results, tokens, _ = evaluate_baselines(test, fit_baselines(train, 6), 6)
    assert tokens == 1
    assert all(np.isfinite(item["perplexity"]) for item in results.values())
