import inspect

import numpy as np
import torch

from src.codonlm.data_loading import PackedDataset
from src.codonlm.training.config import _ensure_path_list
from src.codonlm.training.loop import _average_accumulated_gradients, run_training


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


def test_packed_dataset_preserves_compact_backing_dtype(tmp_path):
    path = tmp_path / "compact.npz"
    x = np.arange(12, dtype=np.int32).reshape(2, 6)
    np.savez_compressed(path, X=x, Y=x)

    ds = PackedDataset(path)

    assert ds.storage_mode == "npz_memory"
    assert ds.X.dtype == torch.int32
    assert ds.Y.dtype == torch.int32
    assert ds[0][0].dtype == torch.long
    assert ds[0][1].dtype == torch.long


def test_average_accumulated_gradients_matches_mean_loss():
    accumulated = torch.nn.Linear(2, 1, bias=False)
    reference = torch.nn.Linear(2, 1, bias=False)
    reference.load_state_dict(accumulated.state_dict())
    batches = [
        (torch.tensor([[1.0, 2.0]]), torch.tensor([[0.5]])),
        (torch.tensor([[2.0, -1.0]]), torch.tensor([[1.5]])),
        (torch.tensor([[-1.0, 3.0]]), torch.tensor([[-0.5]])),
    ]

    for inputs, targets in batches:
        torch.nn.functional.mse_loss(accumulated(inputs), targets).backward()
    _average_accumulated_gradients(accumulated.parameters(), len(batches))

    mean_loss = torch.stack(
        [torch.nn.functional.mse_loss(reference(inputs), targets) for inputs, targets in batches]
    ).mean()
    mean_loss.backward()

    assert torch.allclose(accumulated.weight.grad, reference.weight.grad)


def test_training_loop_does_not_clear_mps_cache_on_happy_path():
    assert "empty_cache" not in inspect.getsource(run_training)
