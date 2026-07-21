#!/usr/bin/env python3
"""Generate manifest-bound synonymous and sequence-shuffling controls."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Callable

import numpy as np

from src.codonlm.dataset_manifest import manifest_artifact_path
from src.codonlm.evaluation_provenance import artifact_provenance, bind_dataset_manifest
from src.codonlm.training.vocabulary import load_itos


AA_TO_CODONS = {
    "A": ["GCT", "GCC", "GCA", "GCG"],
    "R": ["CGT", "CGC", "CGA", "CGG", "AGA", "AGG"],
    "N": ["AAT", "AAC"], "D": ["GAT", "GAC"], "C": ["TGT", "TGC"],
    "Q": ["CAA", "CAG"], "E": ["GAA", "GAG"],
    "G": ["GGT", "GGC", "GGA", "GGG"], "H": ["CAT", "CAC"],
    "I": ["ATT", "ATC", "ATA"],
    "L": ["TTA", "TTG", "CTT", "CTC", "CTA", "CTG"],
    "K": ["AAA", "AAG"], "M": ["ATG"], "F": ["TTT", "TTC"],
    "P": ["CCT", "CCC", "CCA", "CCG"],
    "S": ["TCT", "TCC", "TCA", "TCG", "AGT", "AGC"],
    "T": ["ACT", "ACC", "ACA", "ACG"], "W": ["TGG"],
    "Y": ["TAT", "TAC"], "V": ["GTT", "GTC", "GTA", "GTG"],
    "*": ["TAA", "TAG", "TGA"],
}
CODON_TO_AA = {
    codon: amino_acid
    for amino_acid, codons in AA_TO_CODONS.items()
    for codon in codons
}


def _codon_spans(sequence: np.ndarray, itos: list[str]):
    start = None
    for index, value in enumerate(sequence):
        token_id = int(value)
        is_codon = 0 <= token_id < len(itos) and itos[token_id] in CODON_TO_AA
        if is_codon and start is None:
            start = index
        elif not is_codon and start is not None:
            yield start, index
            start = None
    if start is not None:
        yield start, len(sequence)


def synonymous_mutate_sequence(
    sequence: np.ndarray, rng: random.Random, itos: list[str], stoi: dict[str, int]
) -> np.ndarray:
    output = np.copy(sequence)
    for start, end in _codon_spans(sequence, itos):
        for index in range(start, end):
            amino_acid = CODON_TO_AA[itos[int(sequence[index])]]
            output[index] = stoi[rng.choice(AA_TO_CODONS[amino_acid])]
    return output


def codon_shuffle_sequence(
    sequence: np.ndarray, rng: random.Random, itos: list[str], _stoi: dict[str, int]
) -> np.ndarray:
    output = np.copy(sequence)
    for start, end in _codon_spans(sequence, itos):
        values = list(sequence[start:end])
        rng.shuffle(values)
        output[start:end] = values
    return output


def protein_shuffle_sequence(
    sequence: np.ndarray, rng: random.Random, itos: list[str], stoi: dict[str, int]
) -> np.ndarray:
    output = np.copy(sequence)
    for start, end in _codon_spans(sequence, itos):
        amino_acids = [CODON_TO_AA[itos[int(value)]] for value in sequence[start:end]]
        rng.shuffle(amino_acids)
        output[start:end] = [
            stoi[rng.choice(AA_TO_CODONS[amino_acid])]
            for amino_acid in amino_acids
        ]
    return output


def _transform_dataset(
    X: np.ndarray,
    Y: np.ndarray | None,
    lengths: np.ndarray | None,
    transform: Callable,
    *,
    rng: random.Random,
    itos: list[str],
    stoi: dict[str, int],
) -> tuple[np.ndarray, np.ndarray | None]:
    if lengths is not None:
        output = np.copy(X)
        offset = 0
        for raw_length in lengths:
            length = int(raw_length)
            output[offset : offset + length] = transform(
                X[offset : offset + length], rng, itos, stoi
            )
            offset += length
        if offset != len(X):
            raise ValueError("dynamic lengths do not cover the flat token array")
        return output, None

    if Y is None or X.ndim != 2 or Y.shape != X.shape:
        raise ValueError("fixed controls require equally shaped two-dimensional X and Y")
    output_x = np.zeros_like(X)
    output_y = np.zeros_like(Y)
    for row in range(len(X)):
        transition_count = int(np.count_nonzero(Y[row]))
        if transition_count == 0:
            continue
        if transition_count > 1 and not np.array_equal(
            X[row, 1:transition_count], Y[row, : transition_count - 1]
        ):
            raise ValueError("source dataset violates the next-token X/Y shift invariant")
        if np.any(X[row, transition_count:]) or np.any(Y[row, transition_count:]):
            raise ValueError("source dataset contains non-PAD tokens after padding begins")
        stream = np.concatenate(
            (X[row, :transition_count], Y[row, transition_count - 1 : transition_count])
        )
        transformed = transform(stream, rng, itos, stoi)
        output_x[row, :transition_count] = transformed[:-1]
        output_y[row, :transition_count] = transformed[1:]
    return output_x, output_y


def _write_control(
    path: Path,
    X: np.ndarray,
    Y: np.ndarray | None,
    lengths: np.ndarray | None,
    *,
    kind: str,
    seed: int,
    manifest_provenance: dict | None,
    source_provenance: dict,
    vocabulary_provenance: dict,
) -> None:
    if lengths is None:
        np.savez_compressed(path, X=X, Y=Y)
    else:
        np.savez_compressed(path, X=X, lengths=lengths)
    provenance = {
        "schema_version": 1,
        "status": "derived_control_verified" if manifest_provenance else "legacy_unverified",
        "control": kind,
        "seed": seed,
        "dataset_id": manifest_provenance.get("dataset_id") if manifest_provenance else None,
        "dataset_manifest": manifest_provenance,
        "vocabulary": vocabulary_provenance,
        "source_test": source_provenance,
        "output": artifact_provenance(path),
    }
    sidecar = path.with_suffix(path.suffix + ".provenance.json")
    sidecar.write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n")
    print(f"[controls] saved {path} and {sidecar}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_npz", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--vocabulary", type=Path, help="Required for legacy data without a manifest.")
    parser.add_argument("--out_dir", type=Path)
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()

    test_path = args.test_npz.expanduser().resolve()
    if not test_path.is_file():
        raise FileNotFoundError(f"test dataset not found: {test_path}")

    manifest_provenance = None
    if args.manifest:
        manifest, manifest_provenance = bind_dataset_manifest(
            args.manifest,
            expected_artifacts={"test_tokens": test_path},
            require_scientific=False,
        )
        vocabulary_path = manifest_artifact_path(
            manifest, args.manifest.expanduser().resolve(), "vocabulary"
        )
    elif args.vocabulary:
        vocabulary_path = args.vocabulary.expanduser().resolve()
    else:
        raise ValueError("provide --manifest or --vocabulary; vocabulary IDs are not inferred")

    itos = load_itos(vocabulary_path)
    stoi = {token: index for index, token in enumerate(itos)}
    missing_codons = sorted(set(CODON_TO_AA) - set(stoi))
    if missing_codons:
        raise ValueError(f"vocabulary lacks standard codons: {missing_codons}")

    with np.load(test_path, allow_pickle=False) as data:
        X = np.asarray(data["X"])
        Y = np.asarray(data["Y"]) if "Y" in data else None
        lengths = np.asarray(data["lengths"]) if "lengths" in data else None

    controls = (
        ("synonymous", synonymous_mutate_sequence, 0),
        ("codon_shuffle", codon_shuffle_sequence, 1),
        ("protein_shuffle", protein_shuffle_sequence, 2),
    )
    out_dir = (args.out_dir or test_path.parent).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    block_label = X.shape[1] if X.ndim > 1 else int(lengths.max(initial=0))
    for kind, transform, seed_offset in controls:
        control_x, control_y = _transform_dataset(
            X,
            Y,
            lengths,
            transform,
            rng=random.Random(args.seed + seed_offset),
            itos=itos,
            stoi=stoi,
        )
        output = out_dir / f"test_control_{kind}_bs{block_label}.npz"
        _write_control(
            output,
            control_x,
            control_y,
            lengths,
            kind=kind,
            seed=args.seed + seed_offset,
            manifest_provenance=manifest_provenance,
            source_provenance=artifact_provenance(test_path),
            vocabulary_provenance=artifact_provenance(vocabulary_path),
        )


if __name__ == "__main__":
    main()
