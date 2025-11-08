"""Compute gradient × input saliency for a validation sequence."""
from __future__ import annotations

import argparse
import csv
from typing import Iterable, Optional

import numpy as np
import torch
import torch.nn.functional as F

from ._shared import ensure_run_layout, load_artifacts, load_model, load_token_list, resolve_run


def main(argv: Optional[Iterable[str]] = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_id", nargs="?")
    ap.add_argument("--run_dir", help="Alternative to run_id; path to runs/<RUN_ID>")
    args = ap.parse_args(argv)

    run_id, run_dir = resolve_run(args.run_id, args.run_dir)
    paths = ensure_run_layout(run_id)
    run_dir, tables_dir = paths["run"], paths["tables"]

    artifacts = load_artifacts(run_id)
    val_inputs = artifacts.get("val_inputs")
    val_targets = artifacts.get("val_targets")
    if val_inputs is None or val_inputs.size == 0:
        print("[saliency] validation inputs missing; aborting")
        return
    if val_targets is None or val_targets.size == 0:
        val_targets = np.roll(val_inputs, -1, axis=1)

    tokens = load_token_list(run_dir)

    model, _ = load_model(run_dir)
    device = next(model.parameters()).device
    seq = torch.from_numpy(val_inputs[0:1]).long().to(device)
    tgt = torch.from_numpy(val_targets[0:1]).long().to(device)

    embedding_output = {}

    def capture(module, inputs, output):
        embedding_output["activations"] = output
        output.retain_grad()

    handle = model.tok_emb.register_forward_hook(capture)

    model.zero_grad(set_to_none=True)
    outputs = model(seq, tgt)
    if isinstance(outputs, tuple):
        logits, loss = outputs
        if loss is None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), tgt.view(-1))
    elif isinstance(outputs, dict):
        logits = outputs["logits"]
        loss = outputs.get("loss")
        if loss is None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), tgt.view(-1))
    else:
        raise RuntimeError("Unexpected model output")

    loss.backward()

    handle.remove()

    acts = embedding_output.get("activations")
    if acts is None or acts.grad is None:
        print("[saliency] failed to capture embedding gradients")
        return

    grad = acts.grad[0]
    act = acts.detach()[0]
    saliency = torch.sum(grad * act, dim=-1)

    rows = []
    for idx, value in enumerate(saliency.tolist()):
        token_idx = int(seq[0, idx].cpu())
        token = tokens[token_idx] if token_idx < len(tokens) else f"tok_{token_idx}"
        rows.append((idx, token, value))

    out_path = tables_dir / "saliency.csv"
    with out_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["position", "token", "saliency"])
        for idx, token, val in rows:
            writer.writerow([idx, token, f"{val:.6f}"])

    print(f"[saliency] wrote saliency scores to {out_path}")


if __name__ == "__main__":
    main()
