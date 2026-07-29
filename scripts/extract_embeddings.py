#!/usr/bin/env python3
"""Extract sequence-level embeddings from a trained TinyGPT run.

Inputs:
  - run_id or run_dir (to locate weights.pt and itos.txt)
  - sequences: --fasta FASTA or --csv CSV (--seq_col column name)
  - mode: dna_cds (default; chunk into codons), or codon_tokens (space-separated codons)

Output:
  - NPZ with X (N,D) + optional ids list; saved to --out

Example:
  python -m scripts.extract_embeddings --run_id 2025-11-05_tiny_8L6H_d384_e5 \
    --fasta data/my_genes.fasta --out outputs/reports/e1/train_embeddings.npz
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch

from . import query_model as Q
from src.codonlm.checkpoints import build_codon_model_from_cfg, load_codon_checkpoint
from src.codonlm.evaluation_provenance import (
    bind_checkpoint_dataset,
    bind_dataset_manifest,
)
from src.codonlm.training.vocabulary import load_itos


def _read_fasta(path: Path) -> List[Tuple[str, str]]:
    out = []
    name = None
    seq_chunks: List[str] = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith(">"):
            if name is not None:
                out.append((name, "".join(seq_chunks)))
            name = line[1:].strip()
            seq_chunks = []
        else:
            seq_chunks.append(line)
    if name is not None:
        out.append((name, "".join(seq_chunks)))
    return out


def _dna_to_codon_tokens(dna: str) -> List[str]:
    s = dna.strip().upper().replace("U", "T")
    L = (len(s) // 3) * 3
    toks: List[str] = []
    for i in range(0, L, 3):
        toks.append(s[i : i + 3])
    return toks


def _pool_hidden(
    model: torch.nn.Module,
    idx: torch.Tensor,
    nonpad_mask: torch.Tensor,
    *,
    shape_embeddings: torch.Tensor | None = None,
) -> torch.Tensor:
    """Mean-pool canonical causal hidden states over non-PAD positions."""
    forward_hidden = getattr(model, "forward_hidden", None)
    if not callable(forward_hidden):
        raise TypeError(
            f"{type(model).__name__} does not expose the verified forward_hidden API"
        )
    if getattr(model, "use_shape_guidance", False) and shape_embeddings is None:
        raise RuntimeError(
            "shape-guided extraction requires embeddings from the checkpoint's shape encoder"
        )
    with torch.no_grad():
        x = forward_hidden(idx, shape_embeddings=shape_embeddings)
        mask = nonpad_mask.to(x.dtype).unsqueeze(-1)  # (B,T,1)
        summed = (x * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp_min(1.0)
        pooled = summed / counts
        return pooled  # (B, D)


def _pool_state(
    hidden: torch.Tensor,
    idx: torch.Tensor,
    nonpad_mask: torch.Tensor,
    *,
    mode: str,
    content_ids: set[int],
) -> torch.Tensor:
    if mode == "mean_nonpad":
        mask = nonpad_mask
    elif mode == "mean_content":
        mask = torch.zeros_like(nonpad_mask)
        for token_id in content_ids:
            mask |= idx.eq(token_id)
    elif mode == "eos":
        positions = nonpad_mask.long().sum(dim=1).sub(1).clamp_min(0)
        return hidden[torch.arange(hidden.size(0), device=hidden.device), positions]
    else:
        raise ValueError(f"unsupported pooling mode: {mode}")
    weights = mask.to(hidden.dtype).unsqueeze(-1)
    return (hidden * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_sha() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _validate_vocabulary(itos: tuple[str, ...], state: dict, cfg: dict, path: Path) -> None:
    rows = int(state["tok_emb.weight"].shape[0])
    if rows != len(itos):
        raise RuntimeError(
            f"checkpoint embedding rows={rows} do not match vocabulary size={len(itos)}"
        )
    output_rows = int(state["head.weight"].shape[0])
    if output_rows != len(itos):
        raise RuntimeError(
            f"checkpoint output rows={output_rows} do not match vocabulary size={len(itos)}"
        )
    configured = cfg.get("vocab_size")
    if configured is not None and int(configured) != len(itos):
        raise RuntimeError(
            f"checkpoint vocab_size={configured} does not match vocabulary size={len(itos)}"
        )
    vocabulary_metadata = cfg.get("vocabulary") or {}
    if not isinstance(vocabulary_metadata, dict):
        raise RuntimeError("checkpoint vocabulary metadata must be a mapping")
    expected_hash = vocabulary_metadata.get("sha256")
    if expected_hash and expected_hash != _sha256(path):
        raise RuntimeError("checkpoint vocabulary hash does not match run itos.txt")


def _shape_runtime(checkpoint_path: Path, itos: tuple[str, ...], device: torch.device):
    from scripts.train_biophysics_fusion import build_one_hot_lookup
    from src.codonlm.biophysics import NucleotideEncoder

    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or "encoder" not in payload:
        raise RuntimeError(
            f"shape-guided checkpoint does not contain its trained encoder: {checkpoint_path}"
        )
    encoder = NucleotideEncoder(d_shape=3).to(device)
    encoder.load_state_dict(payload["encoder"])
    encoder.eval()
    return encoder, build_one_hot_lookup(list(itos), device)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_id")
    ap.add_argument("--run_dir")
    ap.add_argument("--fasta")
    ap.add_argument("--csv")
    ap.add_argument("--seq_col", default="seq")
    ap.add_argument("--mode", choices=["dna_cds", "codon_tokens"], default="dna_cds")
    ap.add_argument(
        "--manifest",
        type=Path,
        help="Frozen pretraining manifest; required for corrected checkpoints.",
    )
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument(
        "--random-init-seed",
        type=int,
        help="Extract from a deterministic random model with the checkpoint architecture.",
    )
    ap.add_argument(
        "--hidden-layers",
        default="final",
        help="Comma-separated layer stages: 0..n_layer and/or final.",
    )
    ap.add_argument(
        "--pooling-modes",
        default="mean_nonpad",
        help="Comma-separated modes: mean_nonpad, mean_content, eos.",
    )
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    if args.batch_size < 1:
        ap.error("--batch-size must be at least 1")
    layer_values: list[int | str] = []
    for value in args.hidden_layers.split(","):
        value = value.strip()
        layer_values.append("final" if value == "final" else int(value))
    pooling_modes = [value.strip() for value in args.pooling_modes.split(",")]

    # Resolve run directory and load model + vocab
    if args.run_dir:
        rd = Path(args.run_dir)
    else:
        rd = Path(__file__).resolve().parents[1] / "runs" / args.run_id
    itos_path = rd / "itos.txt"
    itos = load_itos(itos_path)
    stoi = {token: index for index, token in enumerate(itos)}
    state_dict, cfg, checkpoint_path = load_codon_checkpoint(rd)
    valid_layers = set(range(int(cfg["n_layer"]) + 1)) | {"final"}
    if not layer_values or any(layer not in valid_layers for layer in layer_values):
        ap.error(f"--hidden-layers must be drawn from {sorted(map(str, valid_layers))}")
    valid_pooling = {"mean_nonpad", "mean_content", "eos"}
    if not pooling_modes or any(mode not in valid_pooling for mode in pooling_modes):
        ap.error(f"--pooling-modes must be drawn from {sorted(valid_pooling)}")
    manifest_provenance = None
    if args.manifest is not None:
        _, manifest_provenance = bind_dataset_manifest(args.manifest)
    checkpoint_dataset = bind_checkpoint_dataset(cfg, manifest_provenance)
    _validate_vocabulary(itos, state_dict, cfg, itos_path)
    if args.random_init_seed is None:
        model = Q.build_model_from_state(
            state_dict, cfg, setup_shape_runtime=False
        )
        model_initialization = {"kind": "trained_checkpoint"}
        model_weights_sha256 = _sha256(checkpoint_path)
    else:
        if bool(cfg.get("use_shape_guidance", False)):
            raise RuntimeError(
                "random-init extraction is not supported for shape-guided checkpoints"
            )
        torch.manual_seed(args.random_init_seed)
        model = build_codon_model_from_cfg(cfg)
        model.eval()
        random_contract = json.dumps(
            {
                "architecture": {
                    key: cfg.get(key)
                    for key in (
                        "vocab_size",
                        "block_size",
                        "n_layer",
                        "n_head",
                        "n_embd",
                        "dropout",
                        "tie_embeddings",
                        "n_kv_head",
                        "use_sdpa",
                        "use_swiglu",
                        "use_rope",
                    )
                },
                "seed": args.random_init_seed,
            },
            sort_keys=True,
        ).encode("utf-8")
        model_weights_sha256 = hashlib.sha256(random_contract).hexdigest()
        model_initialization = {
            "kind": "random",
            "seed": args.random_init_seed,
        }
    device = Q.dev()
    model.to(device).eval()
    encoder = lookup = None
    if getattr(model, "use_shape_guidance", False):
        encoder, lookup = _shape_runtime(checkpoint_path, itos, device)

    # Load sequences
    seqs: List[Tuple[str, str]] = []
    if args.fasta:
        seqs += _read_fasta(Path(args.fasta))
    if args.csv:
        import csv

        with open(args.csv, "r", newline="") as f:
            for row in csv.DictReader(f):
                seqs.append((row.get("id", f"row{len(seqs)}"), row[args.seq_col]))
    if not seqs:
        raise SystemExit("No sequences provided (use --fasta or --csv)")

    bos = stoi.get("<BOS_CDS>")
    eos = stoi.get("<EOS_CDS>")
    pad = stoi.get("<PAD>", 0)
    examples: List[Tuple[str, List[int]]] = []
    max_T = int(cfg.get("block_size", getattr(model, "block_size", 512)))
    for sid, seq in seqs:
        if args.mode == "dna_cds":
            codons = _dna_to_codon_tokens(seq)
        else:
            codons = [t for t in seq.strip().upper().split() if t]
        toks = []
        if bos is not None:
            toks.append(bos)
        toks.extend(stoi[c] for c in codons if c in stoi)
        if eos is not None:
            toks.append(eos)
        if toks:
            examples.append((sid, toks[:max_T]))

    representation_vectors: dict[str, List[np.ndarray]] = {
        f"layer_{layer}__{mode}": []
        for layer in layer_values
        for mode in pooling_modes
    }
    ids: List[str] = []
    content_ids = {
        index for index, token in enumerate(itos) if len(token) == 3 and token.isalpha()
    }
    with torch.no_grad():
        for start in range(0, len(examples), args.batch_size):
            batch = examples[start : start + args.batch_size]
            batch_width = max(len(toks) for _, toks in batch)
            ids_tensor = torch.full(
                (len(batch), batch_width),
                pad,
                dtype=torch.long,
                device=device,
            )
            for row, (_, toks) in enumerate(batch):
                ids_tensor[row, : len(toks)] = torch.tensor(
                    toks, dtype=torch.long, device=device
                )
            nonpad = ids_tensor.ne(pad)
            shapes = None
            if encoder is not None:
                one_hots = lookup[ids_tensor].view(
                    ids_tensor.size(0), 3 * ids_tensor.size(1), 4
                )
                shapes = encoder(one_hots)
            requested = set(layer_values)
            iterator = getattr(model, "iter_hidden_states", None)
            if not callable(iterator):
                raise TypeError(
                    f"{type(model).__name__} does not expose iter_hidden_states"
                )
            for layer, hidden in iterator(ids_tensor, shape_embeddings=shapes):
                if layer not in requested:
                    continue
                for mode in pooling_modes:
                    pooled = _pool_state(
                        hidden,
                        ids_tensor,
                        nonpad,
                        mode=mode,
                        content_ids=content_ids,
                    )
                    representation_vectors[f"layer_{layer}__{mode}"].extend(
                        pooled.cpu().numpy()
                    )
            ids.extend(sid for sid, _ in batch)

    if not ids:
        raise SystemExit("No valid sequences after tokenization")
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {
        f"X__{name}": np.stack(vectors, axis=0)
        for name, vectors in representation_vectors.items()
    }
    if len(arrays) == 1:
        arrays["X"] = next(iter(arrays.values()))
    np.savez_compressed(out_path, **arrays, ids=np.array(ids, dtype=object))
    input_paths = [Path(path) for path in (args.fasta, args.csv) if path]
    metadata = {
        "schema_version": 1,
        "validation_status": "causal_verified",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "checkpoint": {"path": str(checkpoint_path.resolve()), "sha256": _sha256(checkpoint_path)},
        "model_weights": {
            "sha256": model_weights_sha256,
            "initialization": model_initialization,
        },
        "dataset_manifest": manifest_provenance or {"status": "legacy_unverified"},
        "checkpoint_dataset": checkpoint_dataset,
        "vocabulary": {"path": str(itos_path.resolve()), "size": len(itos), "sha256": _sha256(itos_path)},
        "inputs": [{"path": str(path.resolve()), "sha256": _sha256(path)} for path in input_paths],
        "mask_mode": (
            "canonical_causal_segment"
            if getattr(model, "sep_id", None) is not None
            else "canonical_causal"
        ),
        "pooling_mode": (
            "mean_nonpad_including_special_tokens"
            if pooling_modes == ["mean_nonpad"]
            else "multi_representation"
        ),
        "representations": sorted(representation_vectors),
        "shape_guidance": bool(getattr(model, "use_shape_guidance", False)),
        "block_size": max_T,
        "extraction_batch_size": args.batch_size,
        "truncation_policy": "right_truncate",
        "code_git_sha": _git_sha(),
    }
    out_path.with_suffix(out_path.suffix + ".metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    )
    shapes_written = {name: list(array.shape) for name, array in arrays.items()}
    print(f"[extract] wrote {args.out} with arrays={shapes_written}")


if __name__ == "__main__":
    main()
