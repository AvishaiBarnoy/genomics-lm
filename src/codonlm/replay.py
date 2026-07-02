"""Generated-state replay data for CodonLM auxiliary heads."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import torch
from torch.utils.data import Dataset


IGNORE_INDEX = -100


def _iter_jsonl(path: Path) -> Iterable[dict]:
    with path.open() as fh:
        for line_no, line in enumerate(fh, start=1):
            raw = line.strip()
            if not raw:
                continue
            try:
                yield json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL record at {path}:{line_no}: {exc}") from exc


def _normalize_label_items(record: dict) -> list[tuple[int, int]]:
    items = record.get("labels")
    if items is None and "label_position" in record and "target_class" in record:
        items = [{"pos": record["label_position"], "class": record["target_class"]}]
    if not isinstance(items, list):
        return []

    out: list[tuple[int, int]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        try:
            out.append((int(item["pos"]), int(item["class"])))
        except (KeyError, TypeError, ValueError):
            continue
    return out


class GeneratedTerminationReplayDataset(Dataset):
    """Fixed-length generated contexts with sparse termination-head labels.

    Replay JSONL records must contain ``ids`` and either ``labels`` entries of
    ``{"pos": int, "class": int}`` or the legacy pair ``label_position`` /
    ``target_class``. Positions are absolute within ``ids`` before left
    clipping. The returned label tensor is ``IGNORE_INDEX`` everywhere except
    supervised generated states.
    """

    def __init__(
        self,
        path: str | Path,
        block_size: int,
        *,
        pad_id: int = 0,
        ignore_index: int = IGNORE_INDEX,
    ) -> None:
        self.path = Path(path)
        self.block_size = int(block_size)
        self.pad_id = int(pad_id)
        self.ignore_index = int(ignore_index)
        if self.block_size <= 0:
            raise ValueError("block_size must be positive")
        if not self.path.exists():
            raise FileNotFoundError(f"replay dataset not found: {self.path}")

        records: list[tuple[list[int], list[tuple[int, int]]]] = []
        for record in _iter_jsonl(self.path):
            raw_ids = record.get("ids")
            if not isinstance(raw_ids, list):
                continue
            try:
                ids = [int(x) for x in raw_ids]
            except (TypeError, ValueError):
                continue
            if not ids:
                continue
            label_items = _normalize_label_items(record)
            if not label_items:
                continue
            offset = max(0, len(ids) - self.block_size)
            clipped_len = min(len(ids), self.block_size)
            valid = [
                (pos - offset, cls)
                for pos, cls in label_items
                if offset <= pos < offset + clipped_len
            ]
            if valid:
                records.append((ids, valid))
        if not records:
            raise ValueError(f"replay dataset has no usable records: {self.path}")
        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        ids, labels = self.records[idx]
        ids = ids[-self.block_size :]
        x = torch.full((self.block_size,), self.pad_id, dtype=torch.long)
        y = torch.full((self.block_size,), self.ignore_index, dtype=torch.long)
        x[: len(ids)] = torch.tensor(ids, dtype=torch.long)
        for pos, cls in labels:
            if 0 <= pos < len(ids):
                y[pos] = int(cls)
        return x, y

