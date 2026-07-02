#!/usr/bin/env python3
"""Build generated-state replay data from hard-cap prefix generations."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from scripts import query_model as Q
from scripts.eval_generation_prefix import (
    PRESETS,
    _cfg_from,
    _load_run_meta,
    _load_vocab_for_run,
    _model_spec_from,
    _resolve_cds_dna_path,
    _select_device,
    _set_seed,
)
from src.codonlm.generate import generate_cds_constrained


def _is_codon(tok: str) -> bool:
    return len(tok) == 3 and set(tok) <= set("ACGT")


def _codon_positions(ids: list[int], itos: list[str]) -> list[tuple[int, int, str]]:
    out: list[tuple[int, int, str]] = []
    codon_count = 0
    for pos, idx in enumerate(ids):
        tok = itos[idx] if 0 <= int(idx) < len(itos) else ""
        if _is_codon(tok):
            codon_count += 1
            out.append((pos, codon_count, tok))
    return out


def _replay_labels(
    ids: list[int],
    itos: list[str],
    *,
    prefix_codons: int,
    target_codons: int,
    window: int,
    near_class: int,
    immediate_class: int,
) -> list[dict[str, int]]:
    labels: list[dict[str, int]] = []
    start_generated = max(0, int(target_codons) - max(0, int(window)))
    for pos, total_codons, _tok in _codon_positions(ids, itos):
        generated_codons = int(total_codons) - int(prefix_codons)
        if generated_codons < start_generated:
            continue
        target_class = int(immediate_class) if generated_codons >= int(target_codons) else int(near_class)
        labels.append({"pos": int(pos), "class": target_class})
    return labels


def _load_model(repo: Path, run_id: str, ckpt_name: str, device: torch.device):
    run_dir = repo / "runs" / run_id
    meta = _load_run_meta(run_dir)
    weights_path = run_dir / "checkpoints" / ckpt_name
    if not weights_path.exists():
        weights_path = run_dir / ckpt_name
    if not weights_path.exists():
        weights_path = repo / "outputs" / "checkpoints" / run_id / ckpt_name
    if not weights_path.exists():
        raise FileNotFoundError(f"checkpoint not found: {ckpt_name}")

    ckpt = torch.load(weights_path, map_location=device)
    state_dict = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    spec = _model_spec_from(meta, ckpt)
    model = Q.build_model_from_state(state_dict, spec).to(device).eval()
    cfg = _cfg_from(meta, ckpt)
    itos, stoi = _load_vocab_for_run(run_dir, repo, cfg)
    return run_dir, cfg, model, itos, stoi


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_id", required=True)
    ap.add_argument("--ckpt", default="best.pt")
    ap.add_argument("--preset", choices=sorted(PRESETS), default="quick")
    ap.add_argument("--k_list", default="1,3,5,10")
    ap.add_argument("--samples", type=int, default=None)
    ap.add_argument("--max_genes", type=int, default=None)
    ap.add_argument("--max_new", type=int, default=None)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--min_aa_len", type=int, default=100)
    ap.add_argument("--target_aa_len", type=int, default=256)
    ap.add_argument("--max_aa_len", type=int, default=400)
    ap.add_argument("--special_margin", type=int, default=6)
    ap.add_argument("--replay_window", type=int, default=12)
    ap.add_argument("--near_class", type=int, default=1)
    ap.add_argument("--immediate_class", type=int, default=0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--progress_every", type=int, default=20)
    ap.add_argument(
        "--allow_non_cds_tokens",
        action="store_true",
        help="Permit non-codon tokens during CDS continuation generation for diagnostics.",
    )
    ap.add_argument(
        "--allow_empty",
        action="store_true",
        help="Write an empty replay file instead of failing when no hard-cap failures are found.",
    )
    args = ap.parse_args()

    preset = PRESETS.get(args.preset or "quick", {})
    args.max_genes = int(args.max_genes if args.max_genes is not None else preset.get("max_genes", 10))
    args.samples = int(args.samples if args.samples is not None else preset.get("samples", 2))
    args.max_new = int(args.max_new if args.max_new is not None else preset.get("max_new", 100))
    if not (0 < args.min_aa_len <= args.target_aa_len <= args.max_aa_len):
        raise SystemExit("require 0 < min_aa_len <= target_aa_len <= max_aa_len")

    _set_seed(int(args.seed))
    repo = Path(__file__).resolve().parents[1]
    device = _select_device(args.device)
    run_dir, cfg, model, itos, stoi = _load_model(repo, args.run_id, args.ckpt, device)
    dna_path = _resolve_cds_dna_path(repo, run_dir, cfg, max_genes=args.max_genes)
    if dna_path is None or not dna_path.exists():
        raise SystemExit("[replay] could not locate CDS DNA via run manifests/config")

    out_path = Path(args.out) if args.out else run_dir / "scores" / "generated_prefix_replay.jsonl"
    if not out_path.is_absolute():
        out_path = repo / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path = out_path.with_suffix(".summary.json")

    cds: list[str] = []
    with dna_path.open() as fh:
        for line in fh:
            seq = line.strip().upper().replace("U", "T")
            if len(seq) >= 9:
                cds.append(seq)
            if len(cds) >= args.max_genes:
                break

    k_list = [int(x) for x in str(args.k_list).split(",") if x]
    total_expected = len(cds) * len(k_list) * int(args.samples)
    done = 0
    written = 0
    hard_caps = 0
    terminal_stops = 0
    wall0 = time.perf_counter()
    block_size = int(cfg.get("block_size", getattr(model, "block_size", 512)))

    with out_path.open("w") as out_fh:
        for gene_idx, dna in enumerate(cds):
            truth_codons = [dna[i : i + 3] for i in range(0, (len(dna) // 3) * 3, 3)]
            for k in k_list:
                prefix_k = min(k, len(truth_codons))
                prefix = dna[: 3 * prefix_k]
                ctx_ids = Q.dna_prefix_to_ids(prefix, stoi)
                for sample_id in range(args.samples):
                    max_window_codons = block_size - int(prefix_k) - int(args.special_margin)
                    if max_window_codons < args.min_aa_len:
                        raise ValueError("block_size too small for requested replay generation lengths")
                    hard_cap = int(min(max_window_codons, args.max_aa_len, args.max_new))
                    target_codons = int(max(args.min_aa_len, min(args.target_aa_len, hard_cap)))
                    gen_ids, info = generate_cds_constrained(
                        model=model,
                        device=device,
                        ctx_ids=ctx_ids,
                        stoi=stoi,
                        itos=itos,
                        target_codons=target_codons,
                        hard_cap=hard_cap,
                        require_terminal_stop=True,
                        temperature=float(args.temperature),
                        topk=int(args.topk) if args.topk > 0 else 0,
                        cds_only=not bool(args.allow_non_cds_tokens),
                    )
                    done += 1
                    if info.get("had_terminal_stop"):
                        terminal_stops += 1
                    if info.get("hit_hard_cap") and not info.get("had_terminal_stop"):
                        hard_caps += 1
                        labels = _replay_labels(
                            gen_ids,
                            itos,
                            prefix_codons=prefix_k,
                            target_codons=target_codons,
                            window=int(args.replay_window),
                            near_class=int(args.near_class),
                            immediate_class=int(args.immediate_class),
                        )
                        if labels:
                            record = {
                                "source_run_id": args.run_id,
                                "source_ckpt": args.ckpt,
                                "gene_idx": gene_idx,
                                "k": prefix_k,
                                "sample_id": sample_id,
                                "ids": [int(x) for x in gen_ids],
                                "labels": labels,
                                "target_codons": int(target_codons),
                                "generated_codons": int(info.get("generated_codons", 0)),
                                "hard_cap": int(hard_cap),
                                "last_termination_class": info.get("last_termination_class"),
                            }
                            out_fh.write(json.dumps(record, separators=(",", ":")) + "\n")
                            written += 1
                    if args.progress_every and done % int(args.progress_every) == 0:
                        elapsed = time.perf_counter() - wall0
                        rate = done / max(elapsed, 1e-9)
                        remaining = max(0, total_expected - done)
                        eta = remaining / max(rate, 1e-9)
                        print(
                            f"[replay] progress {done}/{total_expected} "
                            f"written={written} rate={rate:.2f}/sec eta_sec={eta:.1f}",
                            flush=True,
                        )

    summary = {
        "run_id": args.run_id,
        "ckpt": args.ckpt,
        "device": str(device),
        "preset": args.preset,
        "samples": int(args.samples),
        "max_genes": int(args.max_genes),
        "k_list": k_list,
        "generated_samples": int(done),
        "hard_cap_without_stop": int(hard_caps),
        "terminal_stops": int(terminal_stops),
        "records_written": int(written),
        "replay_window": int(args.replay_window),
        "near_class": int(args.near_class),
        "immediate_class": int(args.immediate_class),
        "cds_only": not bool(args.allow_non_cds_tokens),
        "out": str(out_path),
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    if written == 0 and not args.allow_empty:
        raise SystemExit(
            "[replay] no hard-cap replay records were written; "
            "use --allow_empty only when this is expected"
        )
    print(f"[replay] wrote {out_path}")
    print(f"[replay] wrote {summary_path}")


if __name__ == "__main__":
    main()
