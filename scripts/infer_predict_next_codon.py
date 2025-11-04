#!/usr/bin/env python3
import argparse, json
from pathlib import Path
import torch
from scripts.query_model import _load_checkpoint, _load_vocab, build_model_from_state, dev


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--prompt", required=True, help='space-separated codons, e.g., "ATG GCT GCT"')
    ap.add_argument("--topk", type=int, default=10)
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    itos, stoi = _load_vocab(run_dir)
    sd, cfg = _load_checkpoint(run_dir)
    model = build_model_from_state(sd, cfg).to(dev()).eval()

    ids = [stoi.get("<BOS_CDS>", 1)] + [stoi[c] for c in args.prompt.strip().upper().split() if c in stoi]
    x = torch.tensor(ids, dtype=torch.long, device=dev()).unsqueeze(0)
    with torch.no_grad():
        logits, _ = model(x)
        probs = torch.softmax(logits[0, -1], dim=-1)
        vals, idxs = torch.topk(probs, k=min(args.topk, probs.numel()))
    out = [{"token": itos[i], "prob": float(v)} for v, i in zip(vals.tolist(), idxs.tolist())]
    print(json.dumps(out, indent=2))

if __name__ == "__main__":
    main()

