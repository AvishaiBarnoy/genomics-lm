import numpy as np
import torch

from src.codonlm.train_codon_lm import PackedDataset, _ensure_path_list


def test_ensure_path_list_from_string():
    assert _ensure_path_list(None, "foo.npz", "train_npz") == ["foo.npz"]


def test_ensure_path_list_from_list():
    data = ["a.npz", "b.npz"]
    assert _ensure_path_list(data, None, "train_npz") == data


def test_ensure_path_list_missing_raises():
    try:
        _ensure_path_list(None, None, "train_npz")
    except ValueError as exc:
        assert "train_npz" in str(exc)
    else:
        raise AssertionError("Expected ValueError for missing paths")


def test_packed_dataset_concatenates(tmp_path):
    def make_npz(path, start):
        x = np.arange(start, start + 12, dtype=np.int64).reshape(2, 6)
        y = np.arange(start + 100, start + 112, dtype=np.int64).reshape(2, 6)
        np.savez_compressed(path, X=x, Y=y)

    first = tmp_path / "first.npz"
    second = tmp_path / "second.npz"
    make_npz(first, 0)
    make_npz(second, 1000)

    ds = PackedDataset([first, second])
    assert len(ds) == 4

    x0, y0 = ds[0]
    assert isinstance(x0, torch.Tensor) and isinstance(y0, torch.Tensor)
    assert x0.shape == (6,)
    expected = torch.from_numpy(np.arange(1000, 1012, dtype=np.int64).reshape(2, 6)[0])
    assert torch.equal(ds[2][0], expected)
