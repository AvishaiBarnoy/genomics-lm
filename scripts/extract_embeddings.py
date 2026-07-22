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
from src.codonlm.checkpoints import load_codon_checkpoint
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
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    # Resolve run directory and load model + vocab
    if args.run_dir:
        rd = Path(args.run_dir)
    else:
        rd = Path(__file__).resolve().parents[1] / "runs" / args.run_id
    itos_path = rd / "itos.txt"
    itos = load_itos(itos_path)
    stoi = {token: index for index, token in enumerate(itos)}
    state_dict, cfg, checkpoint_path = load_codon_checkpoint(rd)
    manifest_provenance = None
    if args.manifest is not None:
        _, manifest_provenance = bind_dataset_manifest(args.manifest)
    checkpoint_dataset = bind_checkpoint_dataset(cfg, manifest_provenance)
    _validate_vocabulary(itos, state_dict, cfg, itos_path)
    model = Q.build_model_from_state(
        state_dict, cfg, setup_shape_runtime=False
    )
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
    out_vecs: List[np.ndarray] = []
    ids: List[str] = []
    max_T = int(cfg.get("block_size", getattr(model, "block_size", 512)))
    with torch.no_grad():
        for sid, seq in seqs:
            if args.mode == "dna_cds":
                codons = _dna_to_codon_tokens(seq)
            else:
                codons = [t for t in seq.strip().upper().split() if t]
            # Map to ids; add BOS/EOS if available; truncate to block_size
            toks = []
            if bos is not None:
                toks.append(bos)
            for c in codons:
                if c in stoi:
                    toks.append(stoi[c])
            if eos is not None:
                toks.append(eos)
            if not toks:
                continue
            ids_tensor = torch.tensor(
                toks[:max_T], dtype=torch.long, device=device
            ).unsqueeze(0)
            nonpad = ids_tensor.ne(pad)
            shapes = None
            if encoder is not None:
                one_hots = lookup[ids_tensor].view(1, 3 * ids_tensor.size(1), 4)
                shapes = encoder(one_hots)
            pooled = _pool_hidden(
                model, ids_tensor, nonpad, shape_embeddings=shapes
            )
            out_vecs.append(pooled.squeeze(0).cpu().numpy())
            ids.append(sid)

    if not out_vecs:
        raise SystemExit("No valid sequences after tokenization")
    X = np.stack(out_vecs, axis=0)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, X=X, ids=np.array(ids, dtype=object))
    input_paths = [Path(path) for path in (args.fasta, args.csv) if path]
    metadata = {
        "schema_version": 1,
        "validation_status": "causal_verified",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "checkpoint": {"path": str(checkpoint_path.resolve()), "sha256": _sha256(checkpoint_path)},
        "dataset_manifest": manifest_provenance or {"status": "legacy_unverified"},
        "checkpoint_dataset": checkpoint_dataset,
        "vocabulary": {"path": str(itos_path.resolve()), "size": len(itos), "sha256": _sha256(itos_path)},
        "inputs": [{"path": str(path.resolve()), "sha256": _sha256(path)} for path in input_paths],
        "mask_mode": (
            "canonical_causal_segment"
            if getattr(model, "sep_id", None) is not None
            else "canonical_causal"
        ),
        "pooling_mode": "mean_nonpad_including_special_tokens",
        "shape_guidance": bool(getattr(model, "use_shape_guidance", False)),
        "block_size": max_T,
        "truncation_policy": "right_truncate",
        "code_git_sha": _git_sha(),
    }
    out_path.with_suffix(out_path.suffix + ".metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    )
    print(f"[extract] wrote {args.out} with X.shape={X.shape}")


if __name__ == "__main__":
    main()
