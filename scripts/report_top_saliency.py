"""Top-saliency segments report.

Loads saliency.csv for a run and reports the top-K windows (default k=9, top=20)
by summed saliency. If attention tensors are present in artifacts.npz, the
script also reports a simple attention score (mean within-window attention
mass for the first captured sample).

Outputs: runs/<RUN_ID>/tables/top_saliency_segments.csv
"""
from __future__ import annotations

import argparse
import csv
from typing import Iterable, Optional

import numpy as np

from ._shared import ensure_run_layout, load_artifacts, load_token_list


def main(argv: Optional[Iterable[str]] = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_id")
    ap.add_argument("--window", type=int, default=9)
    ap.add_argument("--top", type=int, default=20)
    args = ap.parse_args(argv)

    paths = ensure_run_layout(args.run_id)
    run_dir, tables_dir = paths["run"], paths["tables"]

    # Load saliency
    sal_path = tables_dir / "saliency.csv"
    if not sal_path.exists():
        print(f"[top-saliency] no saliency.csv at {sal_path}; skipping")
        return
    pos, token, sal = [], [], []
    with sal_path.open("r") as fh:
        header = fh.readline()
        for line in fh:
            parts = line.strip().split(",")
            if len(parts) < 3:
                parts = line.strip().split("\t")
            if len(parts) < 3:
                continue
            try:
                p = int(parts[0]); s = float(parts[2])
            except Exception:
                continue
            pos.append(p); token.append(parts[1]); sal.append(s)
    if not sal:
        print("[top-saliency] empty saliency; aborting")
        return

    sal_arr = np.array(sal, dtype=np.float32)
    tok_list = token
    T = len(sal_arr)
    w = max(1, int(args.window))
    if T < w:
        print("[top-saliency] sequence shorter than window; aborting")
        return

    # Compute summed saliency over sliding windows
    sums = np.array([sal_arr[i:i+w].sum() for i in range(0, T - w + 1)], dtype=np.float32)
    order = np.argsort(-sums)[: int(args.top)]

    # Try to load attention
    artifacts = load_artifacts(args.run_id)
    attn = artifacts.get("attn")
    attn_score = None
    if attn is not None and attn.size > 0:
        # attn shape: (layers, batch, heads, T, T); take layer0, batch0, mean over heads
        A = attn[0, 0].mean(axis=0)  # (T, T)
        attn_score = A

    # Prepare rows
    rows = []
    for i in order.tolist():
        tok_span = tok_list[i : i + w]
        score = float(sums[i])
        att_score = ""
        if attn_score is not None and attn_score.shape[0] >= i + w:
            block = attn_score[i : i + w, i : i + w]
            att_score = f"{float(block.mean()):.6f}"
        rows.append((i, "-".join(tok_span), f"{score:.6f}", att_score))

    out_path = tables_dir / "top_saliency_segments.csv"
    with out_path.open("w", newline="") as fh:
        wtr = csv.writer(fh)
        wtr.writerow(["start_pos", "tokens", "saliency_sum", "mean_attn_in_window"])
        for r in rows:
            wtr.writerow(r)

    print(f"[top-saliency] wrote {out_path}")


if __name__ == "__main__":
    main()

