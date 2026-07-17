from pathlib import Path
import numpy as np
import torch

from src.codonlm.data_loading import MmapPackedDataset


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
