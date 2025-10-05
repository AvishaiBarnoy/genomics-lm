"""Probe next-token predictions for fixed genomic prefixes."""
from __future__ import annotations

import argparse
import csv
from typing import Iterable, Optional

import torch

from ._shared import ensure_run_layout, load_model, load_token_list, stoi, softmax

PREFIXES = ["ATG", "ATG-AAA", "ATG-GAA", "TAA"]
TOP_K = 5


def main(argv: Optional[Iterable[str]] = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_id")
    args = ap.parse_args(argv)

    paths = ensure_run_layout(args.run_id)
    run_dir, tables_dir = paths["run"], paths["tables"]

    tokens = load_token_list(run_dir)
    vocab = stoi(tokens)

    try:
        model, spec = load_model(run_dir)
    except Exception as exc:
        print(f"[probe-next] failed to load model: {exc}")
        return

    rows = []
    for prefix in PREFIXES:
        codons = prefix.split("-")
        try:
            indices = [vocab[c] for c in codons]
        except KeyError:
            rows.append([prefix, "NA", ""] + [""] * (2 * (TOP_K - 1)))
            continue
        input_ids = torch.tensor(indices, dtype=torch.long).unsqueeze(0)
        with torch.no_grad():
            outputs = model(input_ids)
            if isinstance(outputs, tuple):
                logits = outputs[0]
            elif isinstance(outputs, dict):
                logits = outputs["logits"]
            else:
                raise RuntimeError("Unexpected model output type")
            next_logits = logits[0, -1]
            probs = softmax(next_logits, dim=-1)
            values, idx = torch.topk(probs, k=min(TOP_K, probs.size(0)))
        pred_tokens = [tokens[i] if i < len(tokens) else f"tok_{i}" for i in idx.tolist()]
        pred_probs = [f"{p:.4f}" for p in values.tolist()]
        row = [prefix]
        for tok, prob in zip(pred_tokens, pred_probs):
            row.extend([tok, prob])
        # pad to fixed width
        while len(row) < 1 + 2 * TOP_K:
            row.append("")
        rows.append(row)

    header = ["prefix"]
    for k in range(1, TOP_K + 1):
        header.extend([f"pred_{k}", f"prob_{k}"])

    out_path = tables_dir / "next_token_tests.csv"
    with out_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)

    print(f"[probe-next] wrote results to {out_path}")


if __name__ == "__main__":
    main()

