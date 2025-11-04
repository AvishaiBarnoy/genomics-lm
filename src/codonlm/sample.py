#!/usr/bin/env python3
"""
Simple sampling utilities with early-stop on EOS or biological stop codons.

CLI:
  python -m src.codonlm.sample --run_dir outputs/checkpoints/<RUN_ID> \
    --max_codons 300 --stop_on_eos --stop_on_bio_stop --prompt "ATG GCT"
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import torch

from .model_tiny_gpt import TinyGPT
from .codon_tokenize import STOP_CODONS
from scripts.query_model import _load_checkpoint as load_ckpt, _load_vocab as load_vocab, build_model_from_state


def dev():
    return torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")


@torch.no_grad()
def generate_ids(model: TinyGPT, device: torch.device, ctx_ids: List[int], max_new_tokens: int, eos_idx: int | None, stop_on_eos: bool, stop_on_bio_stop: bool, itos: List[str]) -> List[int]:
    ids = list(ctx_ids)
    for _ in range(max_new_tokens):
        x = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)
        logits, _ = model(x)
        logits = logits[0, -1]
        probs = torch.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, 1).item()
        ids.append(next_id)
        tok = itos[next_id] if 0 <= next_id < len(itos) else ""
        if stop_on_eos and eos_idx is not None and next_id == eos_idx:
            break
        if stop_on_bio_stop and tok in STOP_CODONS:
            break
    return ids


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--prompt", default="", help="space-separated codons or empty for <BOS_CDS>")
    ap.add_argument("--max_codons", type=int, default=300)
    ap.add_argument("--stop_on_eos", action="store_true")
    ap.add_argument("--stop_on_bio_stop", action="store_true")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    itos, stoi = load_vocab(run_dir)
    state_dict, cfg = load_ckpt(run_dir)
    model = build_model_from_state(state_dict, cfg).to(dev()).eval()

    if args.prompt.strip():
        codons = [c.strip().upper() for c in args.prompt.split() if c.strip()]
        ids = [stoi.get("<BOS_CDS>", 1)] + [stoi[c] for c in codons if c in stoi]
    else:
        ids = [stoi.get("<BOS_CDS>", 1)]

    eos_idx = stoi.get("<EOS_CDS>")
    gen = generate_ids(model, dev(), ids, args.max_codons, eos_idx, args.stop_on_eos, args.stop_on_bio_stop, itos)
    toks = [itos[i] for i in gen]
    print(" ".join(toks))


if __name__ == "__main__":
    main()

