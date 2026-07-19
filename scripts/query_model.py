#!/usr/bin/env python3
"""
Interactive/query interface for a trained codon-level LM.

Examples:
  python -m scripts.query_model <RUN_ID> --mode next --dna ATGAAACCC
  python -m scripts.query_model <RUN_ID> --mode generate --dna ATG --max_new 30 --temperature 0.8 --topk 5
  python -m scripts.query_model <RUN_ID> --mode score --dna ATGAAATGA
  python -m scripts.query_model <RUN_ID> --interactive

Loads runs/<RUN_ID>/weights.pt and runs/<RUN_ID>/itos.txt (written by collect_artifacts_yaml).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import yaml

from src.codonlm.checkpoints import build_codon_model_from_cfg, load_codon_checkpoint
from src.codonlm.model_tiny_gpt import TinyGPT


def dev() -> torch.device:
    return (
        torch.device("mps")
        if torch.backends.mps.is_available()
        else torch.device("cpu")
    )


def _load_checkpoint(run_dir: Path) -> Tuple[Dict, Dict]:
    try:
        state_dict, cfg, _ = load_codon_checkpoint(run_dir)
        return state_dict, cfg
    except FileNotFoundError:
        alt = Path("outputs/checkpoints") / run_dir.name
        state_dict, cfg, _ = load_codon_checkpoint(alt)
        return state_dict, cfg


def _load_vocab(run_dir: Path) -> Tuple[List[str], Dict[str, int]]:
    itos_path = run_dir / "itos.txt"
    if not itos_path.exists():
        cfg_path = run_dir / "checkpoints" / "config.yaml"
        if cfg_path.exists():
            cfg = yaml.safe_load(cfg_path.read_text()) or {}
            fallback = cfg.get("itos_path")
            if fallback and Path(fallback).exists():
                itos_path = Path(fallback)
        if not itos_path.exists():
            raise FileNotFoundError(
                f"Missing itos.txt at {run_dir / 'itos.txt'} and no usable itos_path "
                "was found in checkpoints/config.yaml."
            )
    tokens = [
        line.strip() for line in itos_path.read_text().splitlines() if line.strip()
    ]
    stoi = {tok: i for i, tok in enumerate(tokens)}
    return tokens, stoi


def dna_to_ids(dna: str, stoi: Dict[str, int]) -> List[int]:
    dna = dna.strip().upper().replace("U", "T")
    if len(dna) < 3:
        return []
    L = len(dna) // 3 * 3
    bos = stoi.get("<BOS_CDS>", None)
    eos = stoi.get("<EOS_CDS>", None)
    arr: List[int] = []
    if bos is not None:
        arr.append(bos)
    for i in range(0, L, 3):
        codon = dna[i : i + 3]
        idx = stoi.get(codon)
        if idx is None:
            raise ValueError(f"Unknown codon: {codon}")
        arr.append(idx)
    if eos is not None:
        arr.append(eos)
    return arr


def dna_prefix_to_ids(dna: str, stoi: Dict[str, int]) -> List[int]:
    dna = dna.strip().upper().replace("U", "T")
    if len(dna) < 3:
        return []
    L = len(dna) // 3 * 3
    bos = stoi.get("<BOS_CDS>", None)
    arr: List[int] = []
    if bos is not None:
        arr.append(bos)
    for i in range(0, L, 3):
        codon = dna[i : i + 3]
        idx = stoi.get(codon)
        if idx is None:
            raise ValueError(f"Unknown codon: {codon}")
        arr.append(idx)
    return arr


def ids_to_codons(ids: List[int], itos: List[str]) -> List[str]:
    return [itos[i] if 0 <= i < len(itos) else f"<{i}>" for i in ids]


def build_model_from_state(state_dict: Dict, cfg: Dict, checkpoint: Optional[Dict] = None) -> TinyGPT:
    model = build_codon_model_from_cfg(cfg)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    
    if getattr(model, "use_shape_guidance", False):
        from src.codonlm.biophysics import NucleotideEncoder
        from scripts.train_biophysics_fusion import build_one_hot_lookup
        
        encoder = NucleotideEncoder(d_shape=3)
        loaded = False
        if checkpoint is not None and isinstance(checkpoint, dict) and "encoder" in checkpoint:
            print("[biophysics] Loading NucleotideEncoder weights directly from training checkpoint.")
            encoder.load_state_dict(checkpoint["encoder"])
            loaded = True
        else:
            enc_ckpt = Path("runs/biophysics_encoder.pt")
            if enc_ckpt.exists():
                encoder.load_state_dict(torch.load(enc_ckpt, map_location="cpu"))
                loaded = True
                
        if not loaded:
            print("[warn] No biophysics encoder checkpoint or runs/biophysics_encoder.pt found.")
            
        encoder.eval()
        
        # Load vocabulary
        fallback = cfg.get("itos_path")
        itos_path = Path(fallback) if (fallback and Path(fallback).exists()) else Path("itos.txt")
        if itos_path.exists():
            itos = [line.strip() for line in itos_path.read_text().splitlines() if line.strip()]
        else:
            try:
                from src.codonlm.generate import CODON_ITOS
                itos = CODON_ITOS
            except ImportError:
                from src.codonlm.codon_tokenize import VOCAB
                itos = VOCAB
            
        model.encoder = encoder
        model.lookup_table = build_one_hot_lookup(itos, torch.device("cpu"))
        
    return model


@torch.no_grad()
def next_token(
    model: TinyGPT, device: torch.device, ctx_ids: List[int]
) -> torch.Tensor:
    max_T = getattr(model, "block_size", None)
    ids = ctx_ids[-max_T:] if max_T is not None else ctx_ids
    x = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)  # (1, T)
    
    shapes = None
    if getattr(model, "use_shape_guidance", False) and hasattr(model, "encoder") and hasattr(model, "lookup_table"):
        if model.lookup_table.device != device:
            model.lookup_table = model.lookup_table.to(device)
        if next(model.encoder.parameters()).device != device:
            model.encoder = model.encoder.to(device)
            
        one_hots = model.lookup_table[x]
        one_hots = one_hots.view(1, 3 * x.size(1), 4)
        shapes = model.encoder(one_hots)
        
    logits, _ = model(x, shape_embeddings=shapes)
    return logits[0, -1]  # (V,)


@torch.no_grad()
def generate(
    model: TinyGPT,
    device: torch.device,
    ctx_ids: List[int],
    max_new: int,
    temperature: float = 1.0,
    topk: int = 0,
    eos_idx: int | None = None,
) -> List[int]:
    ids = list(ctx_ids)
    max_T = getattr(model, "block_size", None)
    for _ in range(max_new):
        logits = next_token(model, device, ids)
        if temperature != 1.0:
            logits = logits / max(1e-6, float(temperature))
        probs = torch.softmax(logits, dim=-1)
        if topk and topk > 0:
            vals, idxs = torch.topk(probs, k=min(topk, probs.numel()))
            idx = torch.multinomial(vals, 1).item()
            next_id = idxs[idx].item()
        else:
            next_id = torch.multinomial(probs, 1).item()
        ids.append(next_id)
        if max_T is not None and len(ids) > max_T:
            ids = ids[-max_T:]
        if eos_idx is not None and next_id == eos_idx:
            break
    return ids


@torch.no_grad()
def score_sequence(
    model: TinyGPT, device: torch.device, ids: List[int]
) -> Dict[str, float]:
    x = torch.tensor(ids[:-1], dtype=torch.long, device=device).unsqueeze(0)
    y = torch.tensor(ids[1:], dtype=torch.long, device=device).unsqueeze(0)
    logits, loss = model(x, y)
    loss_val = float(loss.item()) if loss is not None else float("nan")
    ppl = float(np.exp(min(20.0, loss_val))) if loss is not None else float("nan")
    return {"nll": loss_val, "ppl": ppl}


def run_once(args) -> Dict:
    if getattr(args, "run_dir", None):
        rd = Path(args.run_dir)
        run_dir = rd if (rd / "itos.txt").exists() else (Path("runs") / rd.name)
    else:
        run_dir = Path("runs") / args.run_id
    itos, stoi = _load_vocab(run_dir)
    state_dict, cfg = _load_checkpoint(run_dir)
    model = build_model_from_state(state_dict, cfg)
    device = dev()
    model.to(device)

    if args.dna is None and not args.interactive:
        raise SystemExit("Provide --dna or use --interactive mode")

    if args.interactive:
        print("[interactive] enter DNA strings (CTRL+D to exit)")
        while True:
            try:
                line = input("> ").strip()
            except EOFError:
                break
            if not line:
                continue
            res = _answer(line, args, itos, stoi, model, device)
            print(json.dumps(res, indent=2))
        return {}
    else:
        return _answer(args.dna, args, itos, stoi, model, device)


def _answer(
    dna: str,
    args,
    itos: List[str],
    stoi: Dict[str, int],
    model: TinyGPT,
    device: torch.device,
) -> Dict:
    ids = dna_to_ids(dna, stoi)
    if not ids:
        return {"error": "prompt too short (<3 nt)"}
    eos_idx = stoi.get("<EOS_CDS>")
    if args.mode == "next":
        logits = next_token(model, device, ids)
        probs = torch.softmax(logits, dim=-1)
        topv, topi = torch.topk(probs, k=min(args.topk, probs.numel()))
        out = []
        for p, i in zip(topv.tolist(), topi.tolist()):
            out.append({"token": itos[i], "prob": float(p)})
        return {"prompt": dna, "topk": out}
    elif args.mode == "generate":
        gen_ids = generate(
            model,
            device,
            ids,
            max_new=args.max_new,
            temperature=args.temperature,
            topk=args.topk if args.topk > 0 else 0,
            eos_idx=eos_idx,
        )
        gen_toks = ids_to_codons(gen_ids, itos)
        return {"prompt": dna, "tokens": gen_toks}
    elif args.mode == "score":
        return score_sequence(model, device, ids)
    else:
        raise SystemExit(f"Unknown mode: {args.mode}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Query a trained codon LM by run_id or run_dir."
    )
    ap.add_argument(
        "run_id",
        nargs="?",
        help="Run identifier under runs/<RUN_ID> (mutually exclusive with --run_dir)",
    )
    ap.add_argument(
        "--run_dir",
        help="Alternative to run_id; path to outputs/checkpoints/<RUN_ID> or runs/<RUN_ID>",
    )
    ap.add_argument("--mode", choices=["next", "generate", "score"], default="next")
    ap.add_argument("--dna", help="DNA prompt (uppercase ACGT)")
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--max_new", type=int, default=30)
    ap.add_argument("--interactive", action="store_true")
    ap.add_argument("--out", help="optional JSON output path")
    args = ap.parse_args()
    # Argument validation: require either run_id or run_dir
    if not args.run_id and not args.run_dir:
        ap.error("provide either run_id or --run_dir")
    if args.run_id and args.run_dir:
        # Prefer run_dir but warn the user for clarity
        print("[warn] both run_id and --run_dir provided; using --run_dir")
        args.run_id = None

    result = run_once(args)
    if args.out:
        outp = Path(args.out)
        outp.parent.mkdir(parents=True, exist_ok=True)
        outp.write_text(json.dumps(result, indent=2) + "\n")
    elif result:
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
