from pathlib import Path
import numpy as np
import torch

from src.codonlm.data_loading import MmapPackedDataset, build_codon_lm_dataloaders


def test_mmap_packed_dataset_npz_fallback(tmp_path: Path):
    # Create mock fixed-length NPZ
    X = np.random.randint(1, 10, size=(10, 8), dtype=np.int64)
    Y = np.random.randint(1, 10, size=(10, 8), dtype=np.int64)
    
    npz_path = tmp_path / "train.npz"
    np.savez(npz_path, X=X, Y=Y)
    
    # Init MmapPackedDataset (will fall back to PackedDataset)
    ds = MmapPackedDataset(npz_path)
    assert not ds.is_dynamic
    assert ds.storage_mode == "npz_memory"
    assert len(ds) == 10
    
    x_sample, y_sample = ds[3]
    assert torch.equal(x_sample, torch.from_numpy(X[3]))
    assert torch.equal(y_sample, torch.from_numpy(Y[3]))
    
    # Check seq_lengths
    assert np.array_equal(ds.seq_lengths, np.full(10, 8, dtype=np.int32))


def test_mmap_packed_dataset_npy_fixed(tmp_path: Path):
    # Create mock fixed-length NPZ AND its corresponding NPY files
    X = np.random.randint(1, 10, size=(10, 8), dtype=np.int64)
    Y = np.random.randint(1, 10, size=(10, 8), dtype=np.int64)
    
    npz_path = tmp_path / "train.npz"
    np.savez(npz_path, X=X, Y=Y)
    
    # Save uncompressed NPY versions
    np.save(tmp_path / "train_X.npy", X)
    np.save(tmp_path / "train_Y.npy", Y)
    
    # Init MmapPackedDataset (should detect and use npy mmap)
    ds = MmapPackedDataset(npz_path)
    assert ds.use_npy_mmap
    assert ds.storage_mode == "npy_mmap"
    assert not ds.is_dynamic
    assert len(ds) == 10
    
    x_sample, y_sample = ds[5]
    assert torch.equal(x_sample, torch.from_numpy(X[5]))
    assert torch.equal(y_sample, torch.from_numpy(Y[5]))
    
    # Check seq_lengths
    assert np.array_equal(ds.seq_lengths, np.full(10, 8, dtype=np.int32))

    xb, yb = ds.fetch_batch([5, 2, 7])
    assert xb.dtype == torch.long
    assert yb.dtype == torch.long
    assert torch.equal(xb, torch.from_numpy(X[[5, 2, 7]]).long())
    assert torch.equal(yb, torch.from_numpy(Y[[5, 2, 7]]).long())


def test_mmap_dataloader_uses_batched_fixed_fetch(tmp_path: Path):
    X = np.arange(40, dtype=np.uint8).reshape(5, 8)
    Y = X + 1

    npz_path = tmp_path / "train.npz"
    np.savez(npz_path, X=X, Y=Y)
    np.save(tmp_path / "train_X.npy", X)
    np.save(tmp_path / "train_Y.npy", Y)

    train_ds = MmapPackedDataset(npz_path)
    val_ds = MmapPackedDataset(npz_path)
    train_loader, val_loader, _, _ = build_codon_lm_dataloaders(
        train_ds,
        val_ds,
        {"batch_size": 2, "dataloader_seed": 123},
    )

    xb, yb = next(iter(val_loader))
    assert xb.dtype == torch.long
    assert yb.dtype == torch.long
    assert torch.equal(xb, torch.from_numpy(X[:2]).long())
    assert torch.equal(yb, torch.from_numpy(Y[:2]).long())
    assert len(train_loader) == 3


def test_mmap_packed_dataset_npy_dynamic(tmp_path: Path):
    # Create mock dynamic-length NPZ and its NPY versions
    lengths = np.array([4, 6, 2], dtype=np.int32)
    X = np.random.randint(1, 10, size=(12,), dtype=np.int64)
    
    npz_path = tmp_path / "train_dyn.npz"
    np.savez(npz_path, X=X, lengths=lengths)
    
    # Save uncompressed NPY versions
    np.save(tmp_path / "train_dyn_X.npy", X)
    np.save(tmp_path / "train_dyn_lengths.npy", lengths)
    
    # Init MmapPackedDataset (should detect and use npy mmap)
    ds = MmapPackedDataset(npz_path)
    assert ds.use_npy_mmap
    assert ds.storage_mode == "npy_mmap"
    assert ds.is_dynamic
    assert len(ds) == 3
    
    # Check sequences
    assert torch.equal(ds[0], torch.from_numpy(X[0:4]))
    assert torch.equal(ds[1], torch.from_numpy(X[4:10]))
    assert torch.equal(ds[2], torch.from_numpy(X[10:12]))
    
    # Check seq_lengths
    assert np.array_equal(ds.seq_lengths, lengths)

    xb, yb = ds.fetch_batch([1, 0])
    assert xb.dtype == torch.long
    assert yb.dtype == torch.long
    assert torch.equal(xb[0], torch.tensor([X[4], X[5], X[6], X[7], X[8]], dtype=torch.long))
    assert torch.equal(yb[0], torch.tensor([X[5], X[6], X[7], X[8], X[9]], dtype=torch.long))
    assert torch.equal(xb[1], torch.tensor([X[0], X[1], X[2], 0, 0], dtype=torch.long))
    assert torch.equal(yb[1], torch.tensor([X[1], X[2], X[3], 0, 0], dtype=torch.long))


def test_mmap_dataloader_uses_batched_dynamic_fetch(tmp_path: Path):
    seq1 = np.array([1, 10, 20, 2], dtype=np.uint8)
    seq2 = np.array([1, 15, 25, 35, 45, 2], dtype=np.uint8)
    X = np.concatenate([seq1, seq2])
    lengths = np.array([len(seq1), len(seq2)], dtype=np.int32)

    npz_path = tmp_path / "train_dyn.npz"
    np.savez(npz_path, X=X, lengths=lengths)
    np.save(tmp_path / "train_dyn_X.npy", X)
    np.save(tmp_path / "train_dyn_lengths.npy", lengths)

    train_ds = MmapPackedDataset(npz_path)
    val_ds = MmapPackedDataset(npz_path)
    _, val_loader, _, _ = build_codon_lm_dataloaders(
        train_ds,
        val_ds,
        {"batch_size": 2},
    )

    xb, yb = next(iter(val_loader))
    assert torch.equal(xb[0], torch.tensor([1, 10, 20, 0, 0], dtype=torch.long))
    assert torch.equal(yb[0], torch.tensor([10, 20, 2, 0, 0], dtype=torch.long))
    assert torch.equal(xb[1], torch.tensor([1, 15, 25, 35, 45], dtype=torch.long))
    assert torch.equal(yb[1], torch.tensor([15, 25, 35, 45, 2], dtype=torch.long))
