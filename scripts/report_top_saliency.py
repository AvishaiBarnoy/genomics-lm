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
import torch

from ._shared import ensure_run_layout, load_artifacts, load_token_list, load_model, stoi, resolve_run


def main(argv: Optional[Iterable[str]] = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_id", nargs="?")
    ap.add_argument("--run_dir", help="Alternative to run_id; path to runs/<RUN_ID>")
    ap.add_argument("--window", type=int, default=9)
    ap.add_argument("--top", type=int, default=20)
    args = ap.parse_args(argv)

    run_id, run_dir = resolve_run(args.run_id, args.run_dir)
    paths = ensure_run_layout(run_id)
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
    artifacts = load_artifacts(run_id)
    attn = artifacts.get("attn")
    attn_score = None
    if attn is not None and attn.size > 0:
        # attn shape: (layers, batch, heads, T, T); take layer0, batch0, mean over heads
        A = attn[0, 0].mean(axis=0)  # (T, T)
        attn_score = A

    # Prepare rows
    rows = []
    # Optional motif cluster cross-reference
    cluster_path = run_dir / "motif_clusters.npz"
    centers = None
    if cluster_path.exists():
        with np.load(cluster_path, allow_pickle=False) as data:
            centers = data.get("centers")
    model = None
    stoi_map = None
    device = torch.device("cpu")
    if centers is not None:
        try:
            model, _spec = load_model(run_dir, device)
            model.eval()
            stoi_map = stoi(load_token_list(run_dir))
        except Exception as exc:
            print(f"[top-saliency] motif cross-ref disabled: {exc}")
            centers = None
    for i in order.tolist():
        tok_span = tok_list[i : i + w]
        score = float(sums[i])
        att_score = ""
        if attn_score is not None and attn_score.shape[0] >= i + w:
            block = attn_score[i : i + w, i : i + w]
            att_score = f"{float(block.mean()):.6f}"
        cluster_id = ""
        cluster_dist = ""
        if centers is not None and stoi_map is not None:
            try:
                unk_idx = stoi_map.get("<unk>", 0)
                ids = [stoi_map.get(tok, unk_idx) for tok in tok_span]
                x = torch.tensor([ids], dtype=torch.long, device=device)
                feats: list[torch.Tensor] = []

                def hook(_module, _inp, out):
                    feats.append(out.detach())

                handle = model.ln_f.register_forward_hook(hook)
                try:
                    with torch.no_grad():
                        model(x)
                finally:
                    handle.remove()
                if feats:
                    hidden = feats[-1][0]  # (T, d)
                    embed = hidden.mean(dim=0).cpu().numpy()
                    dists = np.linalg.norm(centers - embed, axis=1)
                    nearest = int(np.argmin(dists))
                    cluster_id = str(nearest)
                    cluster_dist = f"{float(dists[nearest]):.6f}"
            except Exception as exc:
                print(f"[top-saliency] failed cluster match: {exc}")
        rows.append((i, "-".join(tok_span), f"{score:.6f}", att_score, cluster_id, cluster_dist))

    out_path = tables_dir / "top_saliency_segments.csv"
    with out_path.open("w", newline="") as fh:
        wtr = csv.writer(fh)
        wtr.writerow([
            "start_pos",
            "tokens",
            "saliency_sum",
            "mean_attn_in_window",
            "nearest_cluster",
            "cluster_distance",
        ])
        for r in rows:
            wtr.writerow(r)

    print(f"[top-saliency] wrote {out_path}")


if __name__ == "__main__":
    main()
