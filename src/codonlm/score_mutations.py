#!/usr/bin/env python3
"""
For a single CDS (as DNA), compute ΔlogP for substituting each position with all 64 codons.

Output: TSV with per-position WT codon and 64 mutant log-probs
Use: identify sensitive sites (proxy for conservation/importance).
"""

import argparse, torch, math, os
from pathlib import Path
import torch.nn.functional as F
from .model_tiny_gpt import TinyGPT

CODONS = [a+b+c for a in "ACGT" for b in "ACGT" for c in "ACGT"]
SPECIALS = ["<pad>", "<bos>", "<eog>", "<unk>", "<eos>"]
VOCAB = SPECIALS + CODONS
stoi = {t:i for i,t in enumerate(VOCAB)}
itos = {i:t for t,i in stoi.items()}

def dna_to_ids(dna):
    dna = dna.strip().upper().replace("U","T")
    L = (len(dna)//3)*3
    ids = [stoi["<bos>"]]
    for i in range(0,L,3):
        ids.append(stoi.get(dna[i:i+3], stoi["<unk>"]))
    ids.append(stoi["<eog>"])
    return ids

def dev():
    return torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument('--dna', required=True, help='DNA string (ACGT...) or path to file (raw or FASTA)')
    ap.add_argument("--out", default=None, help="Write TSV here (default: auto path)")
    args = ap.parse_args()

    # load model
    state = torch.load(args.ckpt, map_location="cpu")
    sd = state.get("model", state)
    cfg_saved = state.get("cfg", {})

    cfg_src = cfg_saved or {
        "vocab_size": 69,
        "block_size": 256,
        "n_layer": 2,
        "n_head": 4,
        "n_embd": 128,
        "dropout": 0.0,
        "use_checkpoint": False,
        "label_smoothing": 0.0,
    }
    model = TinyGPT(
        cfg_src.get("vocab_size", 69),
        cfg_src.get("block_size", 256),
        n_layer=cfg_src.get("n_layer", 2),
        n_head=cfg_src.get("n_head", 4),
        n_embd=cfg_src.get("n_embd", 128),
        dropout=cfg_src.get("dropout", 0.0),
        use_checkpoint=cfg_src.get("use_checkpoint", False),
        label_smoothing=cfg_src.get("label_smoothing", 0.0),
    )

    missing, unexpected = model.load_state_dict(sd, strict=False)
    print("[state] missing:", missing)
    print("[state] unexpected:", unexpected)

    device = dev()
    model.to(device).eval()

    if len(args.dna) <= 80 and all(c in "ACGTUacgtu" for c in args.dna):
        dna = args.dna
    else:
        with open(args.dna) as f: dna = f.read().strip()

    ids = dna_to_ids(dna)
    x = torch.tensor([ids], device=dev()).long()

    # compute baseline log-probs at each position (sliding window if needed)
    with torch.no_grad():
        T = x.size(1)                      # total tokens, including <bos> and <eog>
        # Determine block size from model attributes or saved config
        block_attr = getattr(model, "block_size", None)
        if isinstance(block_attr, int) and block_attr > 0:
            block = block_attr
        elif hasattr(model, "pos_emb") and hasattr(model.pos_emb, "num_embeddings"):
            block = int(model.pos_emb.num_embeddings)
        else:
            block = int(cfg_saved.get("block_size", 256))

        if T <= block:
            # simple case: one pass; predictions for tokens 1..T-1
            logits, _ = model(x)                                # (1, T, V)
            logp_pred = torch.log_softmax(logits, dim=-1)[0][:-1]  # (T-1, V)
        else:
            # streaming: for each position t, predict token ids[t] given prefix ids[:t]
            logps = []
            for t in range(1, T):                    # predict positions 1..T-1
                s = max(0, t - block)                # left edge of window
                ctx = x[:, s:t]                      # (1, ctx_len) prefix up to t-1
                logits, _ = model(ctx)               # (1, ctx_len, V)
                lp_last = torch.log_softmax(logits[:, -1, :], dim=-1)  # (1, V)
                logps.append(lp_last[0].detach().cpu())
            logp_pred = torch.stack(logps, dim=0)    # (T-1, V)

    # positions 1..T-2 are codons (0 is <bos>, last is <eog>)
    print_codons = False
    if print_codons:
        print("pos\twt\t" + "\t".join(CODONS))
        for pos in range(1, len(ids)-1):
            wt_id = ids[pos]; wt = itos[wt_id]
            baseline = float(logp_pred[pos-1, wt_id])
            row = [f"{pos}\t{wt}"]
            for cod in CODONS:
                mid = stoi[cod]
                row.append(f"{float(logp_pred[pos-1, mid] - baseline):.4f}")
            print("\t".join(row))

    output_path: Path
    base_scores = Path("outputs/scores")
    run_id = os.environ.get("RUN_ID", "").strip()
    if run_id:
        base_scores = base_scores / run_id
    base_scores.mkdir(parents=True, exist_ok=True)

    if args.out is None:
        stem = Path(args.dna).stem
        ckpt_stem = Path(args.ckpt).stem
        output_path = base_scores / f"{stem}__{ckpt_stem}.tsv"
    else:
        output_path = Path(args.out)
        output_path.parent.mkdir(parents=True, exist_ok=True)

    import csv
    with output_path.open("w", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["pos","wt"] + CODONS)
        for pos in range(1, len(ids)-1):
            wt_id = ids[pos]; wt = itos[wt_id]
            baseline = float(logp_pred[pos-1, wt_id])
            row = [pos, wt] + [f"{float(logp_pred[pos-1, stoi[c]] - baseline):.4f}" for c in CODONS]
            w.writerow(row)
    print(f"[save] {output_path}")

if __name__ == "__main__":
    main()
