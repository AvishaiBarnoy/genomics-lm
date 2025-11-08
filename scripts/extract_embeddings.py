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
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch

from . import query_model as Q


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


def _pool_hidden(model: torch.nn.Module, idx: torch.Tensor, nonpad_mask: torch.Tensor) -> torch.Tensor:
    # Reconstruct forward pass up to ln_f to get token embeddings; then mean-pool over non-pad tokens
    with torch.no_grad():
        pos = torch.arange(0, idx.shape[1], device=idx.device).unsqueeze(0)
        x = model.tok_emb(idx) + model.pos_emb(pos)
        x = model.drop(x)
        attn_mask = None
        if getattr(model, "sep_id", None) is not None:
            sep = (idx == int(model.sep_id))
            seg = torch.cumsum(sep, dim=1)
            attn_mask = (seg.unsqueeze(-1) == seg.unsqueeze(-2)).unsqueeze(1)
        for blk in model.blocks:
            x = blk(x, attn_mask=attn_mask)
        x = model.ln_f(x)
        # Pool
        mask = nonpad_mask.to(x.dtype).unsqueeze(-1)  # (B,T,1)
        summed = (x * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp_min(1.0)
        pooled = summed / counts
        return pooled  # (B, D)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_id")
    ap.add_argument("--run_dir")
    ap.add_argument("--fasta")
    ap.add_argument("--csv")
    ap.add_argument("--seq_col", default="seq")
    ap.add_argument("--mode", choices=["dna_cds", "codon_tokens"], default="dna_cds")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    # Resolve run directory and load model + vocab
    if args.run_dir:
        rd = Path(args.run_dir)
    else:
        rd = Path(__file__).resolve().parents[1] / "runs" / args.run_id
    itos, stoi = Q._load_vocab(rd)
    state_dict, cfg = Q._load_checkpoint(rd)
    model = Q.build_model_from_state(state_dict, cfg)
    device = Q.dev()
    model.to(device).eval()

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
            ids_tensor = torch.tensor(toks[:max_T], dtype=torch.long, device=device).unsqueeze(0)
            nonpad = ids_tensor.ne(pad)
            pooled = _pool_hidden(model, ids_tensor, nonpad)
            out_vecs.append(pooled.squeeze(0).cpu().numpy())
            ids.append(sid)

    if not out_vecs:
        raise SystemExit("No valid sequences after tokenization")
    X = np.stack(out_vecs, axis=0)
    np.savez_compressed(args.out, X=X, ids=np.array(ids, dtype=object))
    print(f"[extract] wrote {args.out} with X.shape={X.shape}")


if __name__ == "__main__":
    main()

