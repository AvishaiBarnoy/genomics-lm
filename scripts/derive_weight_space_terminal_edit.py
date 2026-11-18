#!/usr/bin/env python3
"""Derive a closed-form stop-row edit from terminal/nonterminal hidden states."""

from __future__ import annotations

import argparse
import copy
import json
import math
import shutil
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from src.codonlm.checkpoints import load_codon_model


def collect(model, npz: Path, rows: int, stop_ids: list[int], eos_id: int, *,
            keep_logits: bool, neg_per_row: int = 24):
    d = np.load(npz)
    x = torch.from_numpy(d["X"][:rows]).long()
    y = torch.from_numpy(d["Y"][:rows]).long()
    rng = np.random.default_rng(20261110)
    pos, neg, all_h, all_logits, all_y, all_terminal = [], [], [], [], [], []
    captured = []
    hook = model.ln_f.register_forward_hook(lambda _m, _i, o: captured.append(o.detach()))
    try:
        with torch.no_grad():
            for i in range(0, rows, 2):
                yy = y[i:i+2]
                logits, _ = model(x[i:i+2])
                h = captured.pop()
                term = torch.zeros_like(yy, dtype=torch.bool)
                term[:, :-1] = torch.isin(yy[:, :-1], torch.tensor(stop_ids)) & (yy[:, 1:] == eos_id)
                pos.append(h[term].cpu())
                for j in range(yy.size(0)):
                    valid = torch.where((yy[j] != 0) & ~term[j] & ~torch.isin(yy[j], torch.tensor(stop_ids + [eos_id])))[0].numpy()
                    if len(valid):
                        chosen = rng.choice(valid, min(neg_per_row, len(valid)), replace=False)
                        neg.append(h[j, chosen].cpu())
                if keep_logits:
                    valid = yy != 0
                    all_h.append(h[valid].cpu())
                    all_logits.append(logits[valid].cpu())
                    all_y.append(yy[valid].cpu())
                    all_terminal.append(term[valid].cpu())
                if (i + 2) % 100 == 0:
                    print(npz.stem, i + 2, "/", rows, flush=True)
    finally:
        hook.remove()
    return {
        "pos": torch.cat(pos), "neg": torch.cat(neg),
        "h": torch.cat(all_h) if keep_logits else None,
        "logits": torch.cat(all_logits) if keep_logits else None,
        "y": torch.cat(all_y) if keep_logits else None,
        "terminal": torch.cat(all_terminal) if keep_logits else None,
    }


def score(val, direction, stop_ids, scale):
    shift = (val["h"] @ direction) * scale
    logits = val["logits"].clone()
    logits[:, stop_ids] += shift[:, None]
    nll = F.cross_entropy(logits, val["y"]).item()
    p_stop = logits.softmax(-1)[:, stop_ids].sum(-1)
    term = val["terminal"]
    return {
        "scale": scale,
        "nll": nll,
        "ppl": math.exp(nll),
        "mean_stop_prob_terminal": p_stop[term].mean().item(),
        "mean_stop_prob_nonterminal": p_stop[~term].mean().item(),
        "median_shift_terminal": shift[term].median().item(),
        "median_shift_nonterminal": shift[~term].median().item(),
    }


def main(args):
    src = Path(args.source)
    model, cfg, src_ckpt = load_codon_model(src, device="cpu")
    itos = (src / "itos.txt").read_text().splitlines()
    stop_ids = [itos.index(t) for t in ("TAA", "TAG", "TGA")]
    eos_id = itos.index("<EOS_CDS>")
    data = Path("data/processed/corrected/corrected-codonlm-v1/genome")
    train = collect(model, data / "train_bs512.npz", args.train_rows, stop_ids, eos_id, keep_logits=False)
    val = collect(model, data / "val_bs512.npz", args.val_rows, stop_ids, eos_id, keep_logits=True)
    pos = train["pos"].double()
    neg = train["neg"].double()
    print("fit samples", len(pos), len(neg), "val terminal", int(val["terminal"].sum()), flush=True)
    mu_p, mu_n = pos.mean(0), neg.mean(0)
    gram = pos.T @ pos / len(pos) + neg.T @ neg / len(neg)
    ridge = args.ridge * float(torch.trace(gram) / gram.size(0))
    direction = torch.linalg.solve(gram + ridge * torch.eye(gram.size(0), dtype=torch.double), mu_p - mu_n).float()
    # Normalize so one scale unit changes the mean terminal logit by roughly one.
    terminal_response = float((val["h"][val["terminal"]] @ direction).mean())
    if terminal_response <= 0:
        raise RuntimeError("Derived direction does not increase held-out terminal logits")
    direction /= terminal_response
    results = [score(val, direction, stop_ids, float(s)) for s in args.scales]
    for result in results:
        print(result, flush=True)
    selected = min((r for r in results if r["nll"] <= results[0]["nll"] + args.max_nll_cost),
                   key=lambda r: (-r["mean_stop_prob_terminal"] / max(r["mean_stop_prob_nonterminal"], 1e-10), r["nll"]))
    scale = selected["scale"]
    if scale <= 0:
        print("No positive scale passed validation gate; no checkpoint written.", flush=True)
        return
    ckpt = torch.load(src_ckpt, map_location="cpu", weights_only=False)
    state = ckpt["model"]
    source_row_norms = {itos[sid]: float(state["head.weight"][sid].norm()) for sid in stop_ids}
    update_norm = float((scale * direction).norm())
    for sid in stop_ids:
        state["head.weight"][sid] += scale * direction
    edited_ckpt = {
        "model": state,
        "cfg": {**ckpt["cfg"], "run_id": args.target.name},
        "source_checkpoint": str(src_ckpt),
    }
    out = args.target / "checkpoints"
    out.mkdir(parents=True, exist_ok=True)
    torch.save(edited_ckpt, out / "best.pt")
    shutil.copy2(out / "best.pt", out / "last.pt")
    for name in ("itos.txt", "vocabulary.json"):
        shutil.copy2(src / name, args.target / name)
    metadata = {
        "source": str(src), "method": "closed_form_terminal_contrastive_stop_row_edit",
        "train_rows": args.train_rows, "val_rows": args.val_rows, "ridge": args.ridge,
        "selected_scale": scale, "selected_validation": selected, "all_validation": results,
        "edited_tensor": "head.weight", "edited_rows": [itos[i] for i in stop_ids],
        "training_terminal_examples": len(pos), "training_nonterminal_examples": len(neg),
        "validation_terminal_examples": int(val["terminal"].sum()),
        "source_stop_row_norms": source_row_norms, "update_norm_per_stop_row": update_norm,
        "status": "experimental_unvalidated", "promotion_recommendation": "do_not_promote_without_generation_review",
    }
    (out / "meta.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print("wrote", args.target, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="runs/corrected-codonlm-v1-batch64-lr-ablation-lr_1_5e4")
    parser.add_argument("--target", type=Path, default=Path("runs/corrected-targeted-terminal-weight-edit-seed1337"))
    parser.add_argument("--train-rows", type=int, default=300)
    parser.add_argument("--val-rows", type=int, default=100)
    parser.add_argument("--ridge", type=float, default=0.1)
    parser.add_argument("--max-nll-cost", type=float, default=0.003)
    parser.add_argument("--scales", type=float, nargs="+", default=[0, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0])
    main(parser.parse_args())
