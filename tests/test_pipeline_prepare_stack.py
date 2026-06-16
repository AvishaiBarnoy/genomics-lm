from pathlib import Path
import numpy as np
from scripts.pipeline_prepare import _stack_npz


def test_stack_npz_concatenates_rows(tmp_path: Path):
    a = tmp_path / "a.npz"
    b = tmp_path / "b.npz"
    X1 = np.arange(12, dtype=np.int64).reshape(3, 4)
    Y1 = np.ones_like(X1)
    X2 = np.arange(8, dtype=np.int64).reshape(2, 4)
    Y2 = np.ones_like(X2)
    np.savez_compressed(a, X=X1, Y=Y1)
    np.savez_compressed(b, X=X2, Y=Y2)
    out = tmp_path / "out.npz"
    _stack_npz([str(a), str(b)], out)
    with np.load(out, allow_pickle=False) as blob:
        X = blob["X"]
        Y = blob["Y"]
    assert X.shape == (5, 4)
    assert Y.shape == (5, 4)
    # Order preserved
    assert np.array_equal(X[:3], X1)
    assert np.array_equal(X[3:], X2)


def test_stack_npz_dynamic(tmp_path: Path):
    a = tmp_path / "a.npz"
    b = tmp_path / "b.npz"
    X1 = np.array([1, 2, 3, 4], dtype=np.int32)
    len1 = np.array([2, 2], dtype=np.int32)
    X2 = np.array([5, 6, 7], dtype=np.int32)
    len2 = np.array([1, 2], dtype=np.int32)

    np.savez_compressed(a, X=X1, lengths=len1)
    np.savez_compressed(b, X=X2, lengths=len2)

    out = tmp_path / "out.npz"
    _stack_npz([str(a), str(b)], out)

    with np.load(out, allow_pickle=False) as blob:
        X = blob["X"]
        lengths = blob["lengths"]

    assert np.array_equal(X, np.array([1, 2, 3, 4, 5, 6, 7], dtype=np.int32))
    assert np.array_equal(lengths, np.array([2, 2, 1, 2], dtype=np.int32))
