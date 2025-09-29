#!/usr/bin/env python3
"""
For a single CDS (as DNA), compute ΔlogP for substituting each position with all 64 codons.

Output: TSV with per-position WT codon and 64 mutant log-probs
Use: identify sensitive sites (proxy for conservation/importance).
"""

import argparse, torch, math
import torch.nn.functional as F
from .model_tiny_gpt import TinyGPT, Cfg

# --- utility: device + legacy ckpt conversion + layer inference ---
def _is_legacy_state(sd: dict) -> bool:
    return any(".attn.query.weight" in k for k in sd.keys()) or any(k.endswith(".attn.mask") for k in sd.keys())

def _convert_legacy_state(sd_old: dict) -> dict:
    sd_new = {}
    for k, v in sd_old.items():
        # drop old attention mask buffer; new model creates attn_mask itself
        if k.endswith(".attn.mask"):
            continue

        # fuse query/key/value -> qkv (WEIGHTS)
        if k.endswith(".attn.query.weight"):
            base = k[:-len(".attn.query.weight")]
            qw = sd_old[k]
            kw = sd_old[base + ".attn.key.weight"]
            vw = sd_old[base + ".attn.value.weight"]
            sd_new[base + ".attn.qkv.weight"] = torch.cat([qw, kw, vw], dim=0)
            continue

        # drop separate q/k/v weights (handled above)
        if (".attn.key.weight" in k) or (".attn.value.weight" in k):
            continue

        # drop old biases if new layers use bias=False on attn.proj or qkv
        if (".attn.query.bias" in k) or (".attn.key.bias" in k) or (".attn.value.bias" in k) or k.endswith(".attn.proj.bias"):
            continue

        # keep everything else
        sd_new[k] = v
    return sd_new

def _infer_n_layers_from_state(sd: dict) -> int:
    max_idx = -1
    for k in sd.keys():
        if k.startswith("blocks."):
            try:
                i = int(k.split('.')[1])
                if i > max_idx: max_idx = i
            except: pass
    return max_idx + 1 if max_idx >= 0 else 0

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
    sd = state.get("model", state)         # supports ckpts saved as {"model": ...}
    cfg_saved = state.get("cfg", {})       # saved at train time in your new trainer

    # prefer the ckpt config; fall back to CLI args if some fields are missing
    mconf = Cfg(
        vocab_size = cfg_saved.get("vocab_size", getattr(args, "vocab_size", 69)),
        n_layer    = cfg_saved.get("n_layer",    getattr(args, "n_layer",    2)),
        n_head     = cfg_saved.get("n_head",     getattr(args, "n_head",     4)),
        n_embd     = cfg_saved.get("n_embd",     getattr(args, "n_embd",     128)),
        block_size = cfg_saved.get("block_size", getattr(args, "block_size", 256)),
        dropout    = cfg_saved.get("dropout",    getattr(args, "dropout",    0.0)),
    )
    model = TinyGPT(mconf)

    # load weights; attn_mask is a buffer created at init, so strict=False is fine
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
        block = int(mconf.block_size)

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

    if args.out is None:
        import os, pathlib
        stem = pathlib.Path(args.dna).stem
        args.out = f"outputs/scores/{stem}__{pathlib.Path(args.ckpt).stem}.tsv"

    os.makedirs("outputs/scores", exist_ok=True)

    import csv
    with open(args.out, "w", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["pos","wt"] + CODONS)
        for pos in range(1, len(ids)-1):
            wt_id = ids[pos]; wt = itos[wt_id]
            baseline = float(logp_pred[pos-1, wt_id])
            row = [pos, wt] + [f"{float(logp_pred[pos-1, stoi[c]] - baseline):.4f}" for c in CODONS]
            w.writerow(row)
    print(f"[save] {args.out}")

if __name__ == "__main__":
    main()

