"""Vocabulary contracts for reproducible CodonLM training."""

from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch


class VocabularyContractError(ValueError):
    """Raised when tokenizer, dataset, config, and model token spaces disagree."""


@dataclass(frozen=True)
class DatasetTokenBounds:
    path: str
    minimum: int | None
    maximum: int | None
    arrays: tuple[str, ...]


@dataclass(frozen=True)
class VocabularyContract:
    source_path: Path
    tokens: tuple[str, ...]
    sha256: str
    configured_size: int | None
    dataset_bounds: tuple[DatasetTokenBounds, ...]

    @property
    def size(self) -> int:
        return len(self.tokens)

    def provenance(self, resolved_path: Path | None = None) -> dict:
        return {
            "schema_version": 1,
            "source_path": str(self.source_path),
            "resolved_path": str(resolved_path or self.source_path),
            "sha256": self.sha256,
            "size": self.size,
            "configured_size": self.configured_size,
            "token_ids_contiguous": True,
            "dataset_bounds": [
                {
                    "path": bound.path,
                    "minimum": bound.minimum,
                    "maximum": bound.maximum,
                    "arrays": list(bound.arrays),
                }
                for bound in self.dataset_bounds
            ],
            "legacy_adaptation": False,
        }


def load_itos(path: Path) -> tuple[str, ...]:
    if not path.exists():
        raise VocabularyContractError(f"Tokenizer vocabulary not found: {path}")
    raw_lines = path.read_text().splitlines()
    if not raw_lines:
        raise VocabularyContractError(f"Tokenizer vocabulary is empty: {path}")
    tokens = tuple(line.strip() for line in raw_lines)
    empty_ids = [index for index, token in enumerate(tokens) if not token]
    if empty_ids:
        raise VocabularyContractError(
            f"Tokenizer vocabulary contains empty token IDs {empty_ids}: {path}"
        )
    duplicates = sorted({token for token in tokens if tokens.count(token) > 1})
    if duplicates:
        raise VocabularyContractError(
            f"Tokenizer vocabulary contains duplicate tokens {duplicates}: {path}"
        )
    return tokens


def resolve_itos_path(
    dataset_paths: Sequence[str | Path], configured_path: str | Path | None
) -> Path:
    adjacent = {
        Path(path).expanduser().resolve().parent / "itos.txt" for path in dataset_paths
    }
    existing_adjacent = sorted(path for path in adjacent if path.exists())
    if existing_adjacent:
        if len(existing_adjacent) != 1 or any(path != existing_adjacent[0] for path in adjacent):
            raise VocabularyContractError(
                "Dataset shards do not resolve to one shared adjacent itos.txt: "
                + ", ".join(str(path) for path in sorted(adjacent))
            )
        resolved = existing_adjacent[0]
        if configured_path is not None:
            configured = Path(configured_path).expanduser().resolve()
            if configured.exists() and configured.read_bytes() != resolved.read_bytes():
                raise VocabularyContractError(
                    f"Configured tokenizer {configured} differs from dataset tokenizer {resolved}"
                )
        return resolved
    if configured_path is None:
        raise VocabularyContractError(
            "No dataset-adjacent itos.txt or explicit itos_path was found"
        )
    return Path(configured_path).expanduser().resolve()


def _bounds(arrays: Iterable[tuple[str, np.ndarray]]) -> tuple[int | None, int | None, tuple[str, ...]]:
    minimum = None
    maximum = None
    names = []
    for name, array in arrays:
        names.append(name)
        if array.size == 0:
            continue
        array_min = int(np.min(array))
        array_max = int(np.max(array))
        minimum = array_min if minimum is None else min(minimum, array_min)
        maximum = array_max if maximum is None else max(maximum, array_max)
    return minimum, maximum, tuple(names)


def dataset_token_bounds(path_value: str | Path) -> DatasetTokenBounds:
    path = Path(path_value).expanduser().resolve()
    x_sidecar = path.with_name(f"{path.stem}_X.npy")
    y_sidecar = path.with_name(f"{path.stem}_Y.npy")
    if x_sidecar.exists():
        arrays = [("X", np.load(x_sidecar, mmap_mode="r"))]
        if y_sidecar.exists():
            arrays.append(("Y", np.load(y_sidecar, mmap_mode="r")))
        minimum, maximum, names = _bounds(arrays)
    else:
        if not path.exists():
            raise VocabularyContractError(f"Dataset shard not found: {path}")
        with np.load(path, allow_pickle=False) as data:
            names = tuple(name for name in ("X", "Y") if name in data)
            if "X" not in names:
                raise VocabularyContractError(f"Dataset shard has no X array: {path}")
            minimum, maximum, names = _bounds((name, data[name]) for name in names)
    return DatasetTokenBounds(str(path), minimum, maximum, names)


def resolve_vocabulary_contract(
    dataset_paths: Sequence[str | Path],
    *,
    configured_path: str | Path | None,
    configured_size: int | None,
) -> VocabularyContract:
    source_path = resolve_itos_path(dataset_paths, configured_path)
    tokens = load_itos(source_path)
    if configured_size is not None and int(configured_size) != len(tokens):
        raise VocabularyContractError(
            f"Configured vocab_size={configured_size} does not match tokenizer vocabulary "
            f"size={len(tokens)} from {source_path}"
        )
    bounds = tuple(dataset_token_bounds(path) for path in dataset_paths)
    for bound in bounds:
        if bound.minimum is not None and bound.minimum < 0:
            raise VocabularyContractError(
                f"Dataset {bound.path} contains negative token ID {bound.minimum}"
            )
        if bound.maximum is not None and bound.maximum >= len(tokens):
            raise VocabularyContractError(
                f"Dataset {bound.path} contains token ID {bound.maximum}, but tokenizer "
                f"{source_path} defines valid IDs 0..{len(tokens) - 1}"
            )
    return VocabularyContract(
        source_path=source_path,
        tokens=tokens,
        sha256=hashlib.sha256(source_path.read_bytes()).hexdigest(),
        configured_size=(int(configured_size) if configured_size is not None else None),
        dataset_bounds=bounds,
    )


def snapshot_vocabulary(contract: VocabularyContract, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if contract.source_path != destination.resolve():
        shutil.copy2(contract.source_path, destination)
    if hashlib.sha256(destination.read_bytes()).hexdigest() != contract.sha256:
        raise VocabularyContractError(f"Vocabulary snapshot hash mismatch: {destination}")
    return destination.resolve()


def checkpoint_embedding_rows(checkpoint: dict) -> tuple[int | None, int | None]:
    state = checkpoint.get("model", checkpoint)
    embedding = state.get("tok_emb.weight")
    output = state.get("head.weight")
    return (
        int(embedding.shape[0]) if embedding is not None else None,
        int(output.shape[0]) if output is not None else None,
    )


def validate_resume_checkpoint(checkpoint_path: str | Path, contract: VocabularyContract) -> None:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    embedding_rows, output_rows = checkpoint_embedding_rows(checkpoint)
    checkpoint_cfg = checkpoint.get("cfg", {}) if isinstance(checkpoint, dict) else {}
    checkpoint_size = checkpoint_cfg.get("vocab_size")
    mismatches = []
    if embedding_rows != contract.size:
        mismatches.append(f"embedding rows={embedding_rows}")
    if output_rows != contract.size:
        mismatches.append(f"output rows={output_rows}")
    if checkpoint_size is not None and int(checkpoint_size) != contract.size:
        mismatches.append(f"checkpoint cfg vocab_size={checkpoint_size}")
    checkpoint_vocab = checkpoint_cfg.get("vocabulary", {})
    checkpoint_hash = checkpoint_vocab.get("sha256") if isinstance(checkpoint_vocab, dict) else None
    if checkpoint_hash is not None and checkpoint_hash != contract.sha256:
        mismatches.append(f"checkpoint vocabulary sha256={checkpoint_hash}")
    if mismatches:
        raise VocabularyContractError(
            f"Resume checkpoint {checkpoint_path} is incompatible with tokenizer "
            f"{contract.source_path} (size={contract.size}, sha256={contract.sha256}): "
            + ", ".join(mismatches)
            + ". Use transfer_from only for explicit legacy vocabulary adaptation."
        )


def write_vocabulary_manifest(provenance: dict, path: Path) -> None:
    path.write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n")


__all__ = [
    "DatasetTokenBounds",
    "VocabularyContract",
    "VocabularyContractError",
    "checkpoint_embedding_rows",
    "dataset_token_bounds",
    "load_itos",
    "resolve_itos_path",
    "resolve_vocabulary_contract",
    "snapshot_vocabulary",
    "validate_resume_checkpoint",
    "write_vocabulary_manifest",
]
