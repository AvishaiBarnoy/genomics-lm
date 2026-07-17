from __future__ import annotations

import os
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

PAD_ID = 0


def dataset_length_audit(dataset, block_size: int) -> dict:
    if len(dataset) == 0:
        return {
            "n_sequences": 0,
            "min": None,
            "p50": None,
            "p90": None,
            "p99": None,
            "max": None,
            "at_block_size": 0,
            "at_block_size_frac": 0.0,
            "mode": "dynamic" if getattr(dataset, "is_dynamic", False) else "fixed",
        }
    if getattr(dataset, "is_dynamic", False):
        lengths = np.asarray(dataset.seq_lengths, dtype=np.int64)
    else:
        lengths = np.full(len(dataset), int(block_size), dtype=np.int64)
    return {
        "n_sequences": int(len(lengths)),
        "min": int(lengths.min()),
        "p50": float(np.percentile(lengths, 50)),
        "p90": float(np.percentile(lengths, 90)),
        "p99": float(np.percentile(lengths, 99)),
        "max": int(lengths.max()),
        "at_block_size": int((lengths >= int(block_size)).sum()),
        "at_block_size_frac": float((lengths >= int(block_size)).mean()),
        "mode": "dynamic" if getattr(dataset, "is_dynamic", False) else "fixed",
    }


class PackedDataset(Dataset):
    def __init__(self, paths):
        if isinstance(paths, (str, os.PathLike)):
            paths = [paths]
        else:
            paths = list(paths)

        self.is_dynamic = False
        if len(paths) > 0:
            with np.load(paths[0], allow_pickle=False) as data:
                if "lengths" in data:
                    self.is_dynamic = True

        if self.is_dynamic:
            self.seqs = []
            for path in paths:
                with np.load(path, allow_pickle=False) as data:
                    flat_X = data["X"]
                    lengths = data["lengths"]
                    offsets = np.insert(np.cumsum(lengths), 0, 0)
                    for i in range(len(lengths)):
                        seq = flat_X[offsets[i] : offsets[i + 1]]
                        self.seqs.append(torch.from_numpy(seq.astype(np.int64)))
        else:
            totals = []
            tail_shape = None
            y_tail_shape = None
            for path in paths:
                with np.load(path, allow_pickle=False) as data:
                    X = data["X"]
                    Y = data["Y"]
                    if tail_shape is None:
                        tail_shape = X.shape[1:]
                        y_tail_shape = Y.shape[1:]
                    totals.append(X.shape[0])
            total_rows = sum(totals)
            if total_rows == 0:
                self.X = torch.empty((0,) + (tail_shape or (0,)), dtype=torch.long)
                self.Y = torch.empty((0,) + (y_tail_shape or (0,)), dtype=torch.long)
                return

            X_agg = np.empty((total_rows,) + tail_shape, dtype=np.int64)
            Y_agg = np.empty((total_rows,) + y_tail_shape, dtype=np.int64)

            offset = 0
            for path in paths:
                with np.load(path, allow_pickle=False) as data:
                    X = np.asarray(data["X"], dtype=np.int64)
                    Y = np.asarray(data["Y"], dtype=np.int64)
                    rows = X.shape[0]
                    X_agg[offset : offset + rows] = X
                    Y_agg[offset : offset + rows] = Y
                    offset += rows
            self.X = torch.from_numpy(X_agg)
            self.Y = torch.from_numpy(Y_agg)

    def __len__(self):
        if self.is_dynamic:
            return len(self.seqs)
        return self.X.shape[0]

    def __getitem__(self, i):
        if self.is_dynamic:
            return self.seqs[i]
        return self.X[i], self.Y[i]

    @property
    def seq_lengths(self) -> np.ndarray:
        if self.is_dynamic:
            return np.array([len(seq) for seq in self.seqs], dtype=np.int32)
        return np.full(len(self), self.X.shape[1], dtype=np.int32)


class MmapPackedDataset(Dataset):
    """Memory-mapped alternative to PackedDataset supporting uncompressed NPY or dynamic NPZ."""

    def __init__(self, paths):
        from pathlib import Path
        if isinstance(paths, (str, os.PathLike)):
            paths = [paths]
        else:
            paths = list(paths)

        # 1. Check if uncompressed .npy files are available next to all paths
        self.use_npy_mmap = True
        npy_configs = []
        for path in paths:
            p = Path(path)
            x_path = p.with_name(p.stem + "_X.npy")
            y_path = p.with_name(p.stem + "_Y.npy")
            len_path = p.with_name(p.stem + "_lengths.npy")
            if x_path.exists():
                npy_configs.append({
                    "X": x_path,
                    "Y": y_path if y_path.exists() else None,
                    "lengths": len_path if len_path.exists() else None,
                    "is_dynamic": len_path.exists()
                })
            else:
                self.use_npy_mmap = False
                break

        if self.use_npy_mmap:
            self.is_dynamic = npy_configs[0]["is_dynamic"]
            self._mmaps_X = []
            self._mmaps_Y = []
            self._offsets = []
            self._lengths = []
            
            total_seqs = 0
            for cfg in npy_configs:
                mmap_X = np.load(cfg["X"], mmap_mode="r")
                self._mmaps_X.append(mmap_X)
                
                if self.is_dynamic:
                    lengths = np.load(cfg["lengths"])
                    offsets = np.concatenate([[0], np.cumsum(lengths[:-1])])
                    self._offsets.append(offsets)
                    self._lengths.append(lengths)
                    total_seqs += len(lengths)
                else:
                    mmap_Y = np.load(cfg["Y"], mmap_mode="r")
                    self._mmaps_Y.append(mmap_Y)
                    total_seqs += mmap_X.shape[0]
                    self._lengths.append(mmap_X.shape[0])
            
            global_file = []
            global_local = []
            for fi, length_info in enumerate(self._lengths):
                n_seq = len(length_info) if self.is_dynamic else length_info
                global_file.append(np.full(n_seq, fi, dtype=np.int32))
                global_local.append(np.arange(n_seq, dtype=np.int32))
            
            self._global_file = np.concatenate(global_file)
            self._global_local = np.concatenate(global_local)
            self._total = total_seqs
            self._delegate = None
            return

        # 2. Fallback to original NPZ loading logic
        with np.load(paths[0], allow_pickle=False) as probe:
            is_dynamic = "lengths" in probe

        if not is_dynamic:
            self._delegate = PackedDataset(paths)
            self.is_dynamic = False
            return

        self.is_dynamic = True
        self._delegate = None
        self._mmaps: list[np.ndarray] = []
        self._offsets: list[np.ndarray] = []
        self._lengths: list[np.ndarray] = []

        total_seqs = 0
        for path in paths:
            data = np.load(path, allow_pickle=False, mmap_mode="r")
            flat_X = data["X"]
            lengths = data["lengths"]
            offsets = np.concatenate([[0], np.cumsum(lengths[:-1])])
            self._mmaps.append(flat_X)
            self._offsets.append(offsets)
            self._lengths.append(lengths)
            total_seqs += len(lengths)

        global_file = []
        global_local = []
        for fi, lengths in enumerate(self._lengths):
            global_file.append(np.full(len(lengths), fi, dtype=np.int32))
            global_local.append(np.arange(len(lengths), dtype=np.int32))
        self._global_file = np.concatenate(global_file)
        self._global_local = np.concatenate(global_local)
        self._total = total_seqs

    def __len__(self):
        if self._delegate is not None:
            return len(self._delegate)
        return self._total

    def __getitem__(self, i):
        if self._delegate is not None:
            return self._delegate[i]
            
        fi = int(self._global_file[i])
        li = int(self._global_local[i])
        
        if self.use_npy_mmap:
            if self.is_dynamic:
                start = int(self._offsets[fi][li])
                length = int(self._lengths[fi][li])
                seq = np.array(self._mmaps_X[fi][start : start + length], dtype=np.int64)
                return torch.from_numpy(seq)
            else:
                x = np.array(self._mmaps_X[fi][li], dtype=np.int64)
                y = np.array(self._mmaps_Y[fi][li], dtype=np.int64)
                return torch.from_numpy(x), torch.from_numpy(y)
                
        start = int(self._offsets[fi][li])
        length = int(self._lengths[fi][li])
        seq = np.array(self._mmaps[fi][start : start + length], dtype=np.int64)
        return torch.from_numpy(seq)

    @property
    def seq_lengths(self) -> np.ndarray:
        if self._delegate is not None:
            return self._delegate.seq_lengths
        if self.use_npy_mmap:
            if self.is_dynamic:
                return np.concatenate(self._lengths)
            else:
                all_lens = []
                for fi, length_info in enumerate(self._lengths):
                    all_lens.append(np.full(length_info, self._mmaps_X[fi].shape[1], dtype=np.int32))
                return np.concatenate(all_lens)
        return np.concatenate(self._lengths)


class BucketBatchSampler(torch.utils.data.Sampler):
    def __init__(
        self,
        lengths: np.ndarray,
        batch_size: int,
        n_buckets: int = 8,
        shuffle: bool = True,
        drop_last: bool = False,
        seed: int | None = None,
    ) -> None:
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.seed = seed
        edges = np.linspace(lengths.min(), lengths.max() + 1, n_buckets + 1)
        bucket_ids = np.digitize(lengths, edges[1:])
        self.buckets: list[list[int]] = [[] for _ in range(n_buckets)]
        for idx, bid in enumerate(bucket_ids):
            self.buckets[bid].append(idx)

    def __iter__(self):
        all_batches: list[list[int]] = []
        rng = np.random.default_rng(self.seed)
        for bucket in self.buckets:
            if not bucket:
                continue
            indices = list(bucket)
            if self.shuffle:
                rng.shuffle(indices)
            for start in range(0, len(indices), self.batch_size):
                batch = indices[start : start + self.batch_size]
                if self.drop_last and len(batch) < self.batch_size:
                    continue
                all_batches.append(batch)
        if self.shuffle:
            rng.shuffle(all_batches)
        yield from all_batches

    def __len__(self):
        total = 0
        for bucket in self.buckets:
            n = len(bucket) // self.batch_size
            if not self.drop_last and len(bucket) % self.batch_size:
                n += 1
            total += n
        return total


def dynamic_lm_collate_fn(batch):
    lengths = [len(seq) for seq in batch]
    max_len = max(lengths)
    xs, ys = [], []
    for seq in batch:
        x_seq = seq[:-1]
        y_seq = seq[1:]
        pad_len = (max_len - 1) - len(x_seq)
        if pad_len > 0:
            x_seq = torch.cat([x_seq, torch.full((pad_len,), PAD_ID, dtype=torch.long)])
            y_seq = torch.cat([y_seq, torch.full((pad_len,), PAD_ID, dtype=torch.long)])
        xs.append(x_seq)
        ys.append(y_seq)
    return torch.stack(xs), torch.stack(ys)


def dataloader_kwargs(cfg: dict[str, Any]) -> dict[str, Any]:
    num_workers = int(cfg.get("num_workers", 0))
    pin_memory = bool(cfg.get("pin_memory", False))
    prefetch_factor = int(cfg.get("prefetch_factor", 2)) if num_workers > 0 else None
    persistent_workers = bool(cfg.get("persistent_workers", True)) if num_workers > 0 else False
    kwargs: dict[str, Any] = dict(num_workers=num_workers, pin_memory=pin_memory)
    if prefetch_factor is not None:
        kwargs["prefetch_factor"] = prefetch_factor
    if persistent_workers:
        kwargs["persistent_workers"] = True
    return kwargs


def build_codon_lm_datasets(train_paths, val_paths, use_mmap: bool = False):
    dataset_cls = MmapPackedDataset if use_mmap else PackedDataset
    return dataset_cls(train_paths), dataset_cls(val_paths)


def build_codon_lm_dataloaders(train_ds, val_ds, cfg: dict[str, Any]):
    kwargs = dataloader_kwargs(cfg)
    collate_fn = dynamic_lm_collate_fn if getattr(train_ds, "is_dynamic", False) else None
    bucket_batching = bool(cfg.get("bucket_batching", False))
    train_sampler = None
    train_shuffle = True
    dataloader_seed = cfg.get("dataloader_seed")
    if bucket_batching and getattr(train_ds, "is_dynamic", False):
        lengths = train_ds.seq_lengths
        n_buckets = int(cfg.get("n_buckets", 8))
        train_sampler = BucketBatchSampler(
            lengths,
            batch_size=int(cfg["batch_size"]),
            n_buckets=n_buckets,
            shuffle=True,
            seed=int(dataloader_seed) if dataloader_seed is not None else None,
        )
        train_shuffle = False

    if train_sampler is not None:
        train_loader = DataLoader(train_ds, batch_sampler=train_sampler, collate_fn=collate_fn, **kwargs)
    else:
        generator = None
        if dataloader_seed is not None:
            generator = torch.Generator()
            generator.manual_seed(int(dataloader_seed))
        train_loader = DataLoader(
            train_ds,
            batch_size=int(cfg["batch_size"]),
            shuffle=train_shuffle,
            collate_fn=collate_fn,
            generator=generator,
            **kwargs,
        )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(cfg["batch_size"]),
        collate_fn=collate_fn,
        **kwargs,
    )
    return train_loader, val_loader, train_sampler, kwargs
