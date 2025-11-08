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

from src.codonlm.model_tiny_gpt import TinyGPT


def dev() -> torch.device:
    return torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")


def _load_checkpoint(run_dir: Path) -> Tuple[Dict, Dict]:
    ckpt_path = run_dir / "weights.pt"
    if not ckpt_path.exists():
        # fallback to outputs layout
        alt = Path("outputs/checkpoints") / run_dir.name / "best.pt"
        if alt.exists():
            ckpt_path = alt
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found under {run_dir} (weights.pt) or outputs/checkpoints/{run_dir.name}/best.pt")
    state = torch.load(ckpt_path, map_location="cpu")
    if isinstance(state, dict) and "model" in state:
        return state["model"], state.get("cfg", {})
    return state, {}


def _load_vocab(run_dir: Path) -> Tuple[List[str], Dict[str, int]]:
    itos_path = run_dir / "itos.txt"
    if not itos_path.exists():
        raise FileNotFoundError(f"Missing itos.txt at {itos_path}. Run analysis/post_process first or provide labels.")
    tokens = [line.strip() for line in itos_path.read_text().splitlines() if line.strip()]
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


def ids_to_codons(ids: List[int], itos: List[str]) -> List[str]:
    return [itos[i] if 0 <= i < len(itos) else f"<{i}>" for i in ids]


def build_model_from_state(state_dict: Dict, cfg: Dict) -> TinyGPT:
    # Pull model dims from cfg saved in checkpoint
    required = ["vocab_size", "block_size", "n_layer", "n_head", "n_embd"]
    missing = [k for k in required if k not in cfg]
    if missing:
        raise RuntimeError(f"Checkpoint missing model config fields: {missing}")
    model = TinyGPT(
        vocab_size=int(cfg["vocab_size"]),
        block_size=int(cfg["block_size"]),
        n_layer=int(cfg["n_layer"]),
        n_head=int(cfg["n_head"]),
        n_embd=int(cfg["n_embd"]),
        dropout=float(cfg.get("dropout", 0.0)),
        use_checkpoint=False,
        label_smoothing=float(cfg.get("label_smoothing", 0.0)),
    )
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model


@torch.no_grad()
def next_token(model: TinyGPT, device: torch.device, ctx_ids: List[int]) -> torch.Tensor:
    max_T = getattr(model, "block_size", None)
    ids = ctx_ids[-max_T:] if max_T is not None else ctx_ids
    x = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)  # (1, T)
    logits, _ = model(x)
    return logits[0, -1]  # (V,)


@torch.no_grad()
def generate(model: TinyGPT, device: torch.device, ctx_ids: List[int], max_new: int, temperature: float = 1.0, topk: int = 0, eos_idx: int | None = None) -> List[int]:
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
def score_sequence(model: TinyGPT, device: torch.device, ids: List[int]) -> Dict[str, float]:
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


def _answer(dna: str, args, itos: List[str], stoi: Dict[str, int], model: TinyGPT, device: torch.device) -> Dict:
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
        gen_ids = generate(model, device, ids, max_new=args.max_new, temperature=args.temperature, topk=args.topk if args.topk>0 else 0, eos_idx=eos_idx)
        gen_toks = ids_to_codons(gen_ids, itos)
        return {"prompt": dna, "tokens": gen_toks}
    elif args.mode == "score":
        return score_sequence(model, device, ids)
    else:
        raise SystemExit(f"Unknown mode: {args.mode}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_id", nargs="?")
    ap.add_argument("--run_dir", help="Alternative to run_id; path to outputs/checkpoints/<RUN_ID> or runs/<RUN_ID>")
    ap.add_argument("--mode", choices=["next", "generate", "score"], default="next")
    ap.add_argument("--dna", help="DNA prompt (uppercase ACGT)")
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--max_new", type=int, default=30)
    ap.add_argument("--interactive", action="store_true")
    ap.add_argument("--out", help="optional JSON output path")
    args = ap.parse_args()

    result = run_once(args)
    if args.out:
        outp = Path(args.out); outp.parent.mkdir(parents=True, exist_ok=True)
        outp.write_text(json.dumps(result, indent=2) + "\n")
    elif result:
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
