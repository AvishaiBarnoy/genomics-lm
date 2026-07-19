"""Lossless token chunking and auditable multi-window packing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping

import numpy as np


PACKING_METADATA_FIELDS = [
    "split",
    "window_index",
    "window_token_count",
    "window_token_start",
    "window_token_end",
    "source_id",
    "source_line_idx",
    "fragment_line_idx",
    "fragment_index",
    "chunk_index",
    "source_token_start",
    "source_token_end",
    "codon_start",
    "codon_end",
    "continues_from_previous",
    "continues_to_next",
    "starts_fragment",
    "ends_fragment",
]


@dataclass(frozen=True)
class TokenChunk:
    """One transition-complete chunk derived from a tokenized CDS fragment."""

    tokens: tuple[int, ...]
    source_id: str
    source_line_idx: int
    fragment_line_idx: int
    fragment_index: int
    chunk_index: int
    split: str
    token_start: int
    token_end: int
    codon_start: int
    codon_end: int
    continues_from_previous: bool
    continues_to_next: bool


@dataclass(frozen=True)
class PackedSpan:
    """Location and provenance of a chunk inside a packed token window."""

    source_id: str
    source_line_idx: int
    fragment_line_idx: int
    fragment_index: int
    chunk_index: int
    split: str
    source_token_start: int
    source_token_end: int
    codon_start: int
    codon_end: int
    window_token_start: int
    window_token_end: int
    continues_from_previous: bool
    continues_to_next: bool


@dataclass(frozen=True)
class PackedWindow:
    """A token window and the source spans placed within it."""

    tokens: tuple[int, ...]
    spans: tuple[PackedSpan, ...]


def chunk_record(record: Mapping[str, Any], block_size: int) -> list[TokenChunk]:
    """Chunk one fragment with one-token overlap and complete transition coverage.

    ``block_size`` is the maximum number of next-token transitions consumed by the
    model, so a chunk contains at most ``block_size + 1`` tokens.
    """
    if block_size < 1:
        raise ValueError("block_size must be at least 1")
    tokens = tuple(int(token) for token in record["tokens"])
    if len(tokens) < 2:
        return []

    fragment_codon_start = int(record["fragment_codon_start"])
    fragment_codon_end = int(record["fragment_codon_end"])
    fragment_codon_count = fragment_codon_end - fragment_codon_start
    capacity = block_size + 1
    chunks: list[TokenChunk] = []
    token_start = 0

    while token_start < len(tokens) - 1:
        token_end = min(token_start + capacity, len(tokens))
        local_codon_start = max(0, token_start - 1)
        local_codon_end = min(fragment_codon_count, token_end - 1)
        chunks.append(
            TokenChunk(
                tokens=tokens[token_start:token_end],
                source_id=str(record["source_id"]),
                source_line_idx=int(record["source_line_idx"]),
                fragment_line_idx=int(record["fragment_line_idx"]),
                fragment_index=int(record["fragment_index"]),
                chunk_index=len(chunks),
                split=str(record["split"]),
                token_start=token_start,
                token_end=token_end,
                codon_start=fragment_codon_start + local_codon_start,
                codon_end=fragment_codon_start + local_codon_end,
                continues_from_previous=token_start > 0,
                continues_to_next=token_end < len(tokens),
            )
        )
        if token_end == len(tokens):
            break
        token_start = token_end - 1

    return chunks


def _span_for_chunk(
    chunk: TokenChunk, window_token_start: int, window_token_end: int
) -> PackedSpan:
    return PackedSpan(
        source_id=chunk.source_id,
        source_line_idx=chunk.source_line_idx,
        fragment_line_idx=chunk.fragment_line_idx,
        fragment_index=chunk.fragment_index,
        chunk_index=chunk.chunk_index,
        split=chunk.split,
        source_token_start=chunk.token_start,
        source_token_end=chunk.token_end,
        codon_start=chunk.codon_start,
        codon_end=chunk.codon_end,
        window_token_start=window_token_start,
        window_token_end=window_token_end,
        continues_from_previous=chunk.continues_from_previous,
        continues_to_next=chunk.continues_to_next,
    )


def pack_chunks(
    chunks: Iterable[TokenChunk],
    *,
    block_size: int,
    mode: str,
    sep_id: int,
) -> list[PackedWindow]:
    """Pack chunks without losing or duplicating their source transitions."""
    if mode not in {"single", "dynamic", "multi"}:
        raise ValueError(f"Unsupported pack mode: {mode!r}")
    capacity = block_size + 1
    chunk_list = list(chunks)
    if any(len(chunk.tokens) > capacity for chunk in chunk_list):
        raise ValueError("Chunk exceeds block_size + 1 token capacity")

    if mode in {"single", "dynamic"}:
        return [
            PackedWindow(
                tokens=chunk.tokens,
                spans=(_span_for_chunk(chunk, 0, len(chunk.tokens)),),
            )
            for chunk in chunk_list
        ]

    windows: list[PackedWindow] = []
    window_tokens: list[int] = []
    window_spans: list[PackedSpan] = []

    def flush() -> None:
        nonlocal window_tokens, window_spans
        if len(window_tokens) >= 2:
            windows.append(
                PackedWindow(tokens=tuple(window_tokens), spans=tuple(window_spans))
            )
        window_tokens = []
        window_spans = []

    for chunk in chunk_list:
        if chunk.continues_from_previous and window_tokens:
            flush()
        separator_count = 1 if window_tokens else 0
        if len(window_tokens) + separator_count + len(chunk.tokens) > capacity:
            flush()
            separator_count = 0
        if separator_count:
            window_tokens.append(sep_id)
        window_start = len(window_tokens)
        window_tokens.extend(chunk.tokens)
        window_spans.append(
            _span_for_chunk(chunk, window_start, len(window_tokens))
        )
        if chunk.continues_to_next or len(window_tokens) == capacity:
            flush()
    flush()
    return windows


def packing_metadata_rows(
    split: str, windows: Iterable[PackedWindow]
) -> list[dict[str, Any]]:
    """Create portable tabular provenance rows for packed windows."""
    rows = []
    for window_index, window in enumerate(windows):
        for span in window.spans:
            rows.append(
                {
                    "split": split,
                    "window_index": window_index,
                    "window_token_count": len(window.tokens),
                    "window_token_start": span.window_token_start,
                    "window_token_end": span.window_token_end,
                    "source_id": span.source_id,
                    "source_line_idx": span.source_line_idx,
                    "fragment_line_idx": span.fragment_line_idx,
                    "fragment_index": span.fragment_index,
                    "chunk_index": span.chunk_index,
                    "source_token_start": span.source_token_start,
                    "source_token_end": span.source_token_end,
                    "codon_start": span.codon_start,
                    "codon_end": span.codon_end,
                    "continues_from_previous": int(span.continues_from_previous),
                    "continues_to_next": int(span.continues_to_next),
                    "starts_fragment": int(span.source_token_start == 0),
                    "ends_fragment": int(not span.continues_to_next),
                }
            )
    return rows


def packed_arrays(
    windows: Iterable[PackedWindow], *, block_size: int, mode: str
) -> dict[str, np.ndarray]:
    """Convert packed windows to loader-compatible arrays with aligned provenance."""
    window_list = list(windows)
    segment_rows = []
    position_rows = []
    chunk_rows = []
    for window in window_list:
        segment_ids = np.full(len(window.tokens), -1, dtype=np.int32)
        source_positions = np.full(len(window.tokens), -1, dtype=np.int32)
        chunk_ids = np.full(len(window.tokens), -1, dtype=np.int32)
        for span in window.spans:
            start = span.window_token_start
            end = span.window_token_end
            segment_ids[start:end] = span.fragment_line_idx
            source_positions[start:end] = np.arange(
                span.source_token_start, span.source_token_end, dtype=np.int32
            )
            chunk_ids[start:end] = span.chunk_index
        segment_rows.append(segment_ids)
        position_rows.append(source_positions)
        chunk_rows.append(chunk_ids)

    if mode == "dynamic":
        return {
            "X": np.concatenate(
                [np.asarray(window.tokens, dtype=np.int32) for window in window_list]
            )
            if window_list
            else np.zeros((0,), dtype=np.int32),
            "lengths": np.asarray(
                [len(window.tokens) for window in window_list], dtype=np.int32
            ),
            "segment_ids": np.concatenate(segment_rows)
            if segment_rows
            else np.zeros((0,), dtype=np.int32),
            "source_positions": np.concatenate(position_rows)
            if position_rows
            else np.zeros((0,), dtype=np.int32),
            "chunk_ids": np.concatenate(chunk_rows)
            if chunk_rows
            else np.zeros((0,), dtype=np.int32),
        }

    X = np.zeros((len(window_list), block_size), dtype=np.int32)
    Y = np.zeros((len(window_list), block_size), dtype=np.int32)
    segment_ids = np.full((len(window_list), block_size), -1, dtype=np.int32)
    source_positions = np.full((len(window_list), block_size), -1, dtype=np.int32)
    chunk_ids = np.full((len(window_list), block_size), -1, dtype=np.int32)
    for window_index, window in enumerate(window_list):
        tokens = np.asarray(window.tokens, dtype=np.int32)
        transition_count = len(tokens) - 1
        X[window_index, :transition_count] = tokens[:-1]
        Y[window_index, :transition_count] = tokens[1:]
        segment_ids[window_index, :transition_count] = segment_rows[window_index][:-1]
        source_positions[window_index, :transition_count] = position_rows[window_index][
            :-1
        ]
        chunk_ids[window_index, :transition_count] = chunk_rows[window_index][:-1]
    return {
        "X": X,
        "Y": Y,
        "segment_ids": segment_ids,
        "source_positions": source_positions,
        "chunk_ids": chunk_ids,
    }


__all__ = [
    "PackedSpan",
    "PackedWindow",
    "PACKING_METADATA_FIELDS",
    "TokenChunk",
    "chunk_record",
    "pack_chunks",
    "packed_arrays",
    "packing_metadata_rows",
]
