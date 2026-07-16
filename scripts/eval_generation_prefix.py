#!/usr/bin/env python3
"""
Prefix-generation benchmark for codon LM.

Given a run_id and a list of prefix lengths (k in codons), generate continuations
from real CDS prefixes, score validity/fidelity/coherence metrics, and produce:

- outputs/scores/<RUN_ID>/gen_prefix/samples.csv
- outputs/scores/<RUN_ID>/gen_prefix/summary.csv
- simple plots under outputs/scores/<RUN_ID>/gen_prefix/

CLI:
  python -m scripts.eval_generation_prefix --run_id <RUN_ID> \
    --k_list 1,3,5,10 --samples 5 --max_genes 50 --max_new 300 --temperature 0.8 --topk 5
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(Path(os.environ.get("TMPDIR", "/tmp")) / "genomics-lm-mplconfig"))

import numpy as np
import torch
import matplotlib.pyplot as plt

from . import query_model as Q
from src.codonlm.generate import generate_cds_constrained, generate_cds_critic_guided
from .generative_design_loop import load_critic, score_with_critic

PRESETS = {
    "quick": {"max_genes": 10, "samples": 2, "max_new": 100},
    "standard": {"max_genes": 20, "samples": 3, "max_new": 300},
    "full": {"max_genes": 50, "samples": 5, "max_new": 300},
}


def _select_device(requested: str) -> torch.device:
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise SystemExit("[gen-prefix] requested --device cuda but CUDA is not available")
        return torch.device("cuda")
    if requested == "mps":
        if not torch.backends.mps.is_available():
            raise SystemExit("[gen-prefix] requested --device mps but MPS is not available")
        return torch.device("mps")
    if requested == "cpu":
        return torch.device("cpu")
    raise SystemExit(f"[gen-prefix] unknown device: {requested}")


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _load_run_meta(run_dir: Path) -> dict:
    for path in (run_dir / "meta.json", run_dir / "checkpoints" / "meta.json"):
        if path.exists():
            return json.loads(path.read_text())
    raise FileNotFoundError(f"meta.json missing under {run_dir}")


def _resolve_repo_path(repo: Path, raw_path: str | Path | None) -> Path | None:
    if not raw_path:
        return None
    path = Path(raw_path)
    return path if path.is_absolute() else repo / path


def _extract_hybrid_cds_file(repo: Path, run_dir: Path, manifest_path: Path, max_genes: int) -> Path | None:
    data = json.loads(manifest_path.read_text())
    datasets = data.get("datasets") or []
    if not datasets:
        return None

    out_path = run_dir / "scores" / "_eval_hybrid_cds_dna.txt"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with out_path.open("w") as out_fh:
        for item in datasets:
            tsv_path = _resolve_repo_path(repo, item.get("tsv") or item.get("hybrid_data"))
            if tsv_path is None or not tsv_path.exists():
                continue
            with tsv_path.open(newline="") as fh:
                reader = csv.DictReader(fh, delimiter="\t")
                required = {"sequence", "cds_start", "cds_end"}
                if not required.issubset(set(reader.fieldnames or [])):
                    continue
                for row in reader:
                    seq = str(row["sequence"]).strip().upper().replace("U", "T")
                    try:
                        start = int(row["cds_start"])
                        end = int(row["cds_end"])
                    except (TypeError, ValueError):
                        continue
                    cds = seq[start:end]
                    if len(cds) >= 9:
                        out_fh.write(cds + "\n")
                        written += 1
                    if written >= max_genes:
                        break
            if written >= max_genes:
                break
    return out_path if written else None


def _resolve_cds_dna_path(repo: Path, run_dir: Path, cfg: dict, max_genes: int) -> Path | None:
    # Legacy/main.sh manifest layout.
    manifest = run_dir / "combined_manifest.json"
    if manifest.exists():
        data = json.loads(manifest.read_text())
        if data.get("datasets"):
            dna_path = _resolve_repo_path(repo, data["datasets"][0].get("dna"))
            if dna_path is not None and dna_path.exists():
                return dna_path

    # Direct config override for future runs.
    for key in ("dna_path", "cds_dna", "primary_dna"):
        dna_path = _resolve_repo_path(repo, cfg.get(key))
        if dna_path is not None and dna_path.exists():
            return dna_path

    manifest_candidates = [cfg.get("hybrid_manifest"), cfg.get("combined_manifest")]
    train_npz = cfg.get("train_npz")
    train_npz_values = train_npz if isinstance(train_npz, list) else [train_npz]
    for item in train_npz_values:
        train_path = _resolve_repo_path(repo, item)
        if train_path is not None:
            manifest_candidates.append(train_path.parent / "manifest.json")

    # Hybrid CDS+UTR manifest layout from pipeline_prepare_hybrid.
    for raw_manifest in manifest_candidates:
        manifest_path = _resolve_repo_path(repo, raw_manifest)
        if manifest_path is not None and manifest_path.exists():
            dna_path = _extract_hybrid_cds_file(repo, run_dir, manifest_path, max_genes=max(10000, max_genes))
            if dna_path is not None and dna_path.exists():
                return dna_path
    return None


def _model_spec_from(meta: dict, ckpt: object) -> dict:
    spec = meta.get("model_spec") or {}
    cfg = {}
    if isinstance(ckpt, dict):
        cfg = ckpt.get("cfg", {}) or {}
    cfg = meta.get("cfg", cfg) or cfg

    keys = [
        "vocab_size", "block_size", "n_layer", "n_head", "n_embd",
        "multi_offset_targets", "termination_aux", "termination_n_classes", "termination_loss_enabled",
        "use_swiglu", "use_rope", "use_shape_guidance", "itos_path"
    ]
    for k in keys:
        if k in cfg and k not in spec:
            spec[k] = cfg[k]

    # Validate required keys
    required = ["vocab_size", "block_size", "n_layer", "n_head", "n_embd"]
    missing = [k for k in required if k not in spec]
    if missing:
        raise KeyError(f"model_spec missing and checkpoint cfg lacks: {missing}")
    return spec


def _cfg_from(meta: dict, ckpt: object) -> dict:
    ckpt_cfg = ckpt.get("cfg", {}) if isinstance(ckpt, dict) else {}
    return meta.get("cfg", ckpt_cfg) or ckpt_cfg or {}


def _load_vocab_for_run(run_dir: Path, repo: Path, cfg: dict) -> Tuple[List[str], Dict[str, int]]:
    try:
        return Q._load_vocab(run_dir)
    except FileNotFoundError:
        pass

    itos_path = cfg.get("itos_path")
    if not itos_path:
        raise FileNotFoundError(
            f"Missing itos.txt at {run_dir / 'itos.txt'} and checkpoint cfg has no itos_path"
        )
    path = Path(str(itos_path))
    if not path.is_absolute():
        path = repo / path
    if not path.exists():
        raise FileNotFoundError(
            f"Missing itos.txt at {run_dir / 'itos.txt'} and configured itos_path does not exist: {path}"
        )
    tokens = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    if not tokens:
        raise ValueError(f"Configured itos_path is empty: {path}")
    return tokens, {tok: i for i, tok in enumerate(tokens)}


# --- Biology helpers ---
CODON_TO_AA: Dict[str, str] = {
    # fmt: off
    "TTT": "F",
    "TTC": "F",
    "TTA": "L",
    "TTG": "L",
    "TCT": "S",
    "TCC": "S",
    "TCA": "S",
    "TCG": "S",
    "TAT": "Y",
    "TAC": "Y",
    "TAA": "Stop",
    "TAG": "Stop",
    "TGT": "C",
    "TGC": "C",
    "TGA": "Stop",
    "TGG": "W",
    "CTT": "L",
    "CTC": "L",
    "CTA": "L",
    "CTG": "L",
    "CCT": "P",
    "CCC": "P",
    "CCA": "P",
    "CCG": "P",
    "CAT": "H",
    "CAC": "H",
    "CAA": "Q",
    "CAG": "Q",
    "CGT": "R",
    "CGC": "R",
    "CGA": "R",
    "CGG": "R",
    "ATT": "I",
    "ATC": "I",
    "ATA": "I",
    "ATG": "M",
    "ACT": "T",
    "ACC": "T",
    "ACA": "T",
    "ACG": "T",
    "AAT": "N",
    "AAC": "N",
    "AAA": "K",
    "AAG": "K",
    "AGT": "S",
    "AGC": "S",
    "AGA": "R",
    "AGG": "R",
    "GTT": "V",
    "GTC": "V",
    "GTA": "V",
    "GTG": "V",
    "GCT": "A",
    "GCC": "A",
    "GCA": "A",
    "GCG": "A",
    "GAT": "D",
    "GAC": "D",
    "GAA": "E",
    "GAG": "E",
    "GGT": "G",
    "GGC": "G",
    "GGA": "G",
    "GGG": "G",
    # fmt: on
}


def _codon_to_aa(codon: str) -> str:
    return CODON_TO_AA.get(codon, "?")


def _aa_seq(codons: List[str]) -> List[str]:
    return [_codon_to_aa(c) for c in codons if len(c) == 3]


def _ngram_repeat_ratio(tokens: List[str], n: int = 3) -> float:
    """Fraction of repeated n-grams using non-overlapping windows.

    This matches legacy expectations in tests where sequences are chunked by codons.
    """
    if len(tokens) < n:
        return 0.0
    grams = [tuple(tokens[i : i + n]) for i in range(0, len(tokens) - n + 1, n)]
    uniq = len(set(grams))
    total = len(grams)
    return 1.0 - (uniq / total) if total else 0.0


def _score_stop_behavior(
    gen_codons: List[str], truth_len_codons: int
) -> Tuple[float, bool, bool]:
    """Return (StopScore, valid_end_stop, early_stop_flag).

    StopScore = 1 if ends with canonical stop and <eog> present; else decays with normalized
    termination error.
    """
    stops = {"TAA", "TAG", "TGA"}
    valid_end = len(gen_codons) > 0 and gen_codons[-1] in stops
    # early stop: any stop before 90% of truth length
    early = False
    cutoff = max(1, int(0.9 * truth_len_codons))
    for i, c in enumerate(gen_codons[:-1]):
        if c in stops and i < cutoff:
            early = True
            break
    if valid_end:
        stop_score = 1.0 if not early else 0.5
    else:
        # termination error: absolute distance from truth length (normalized)
        tau = abs(len(gen_codons) - truth_len_codons) / max(1, truth_len_codons)
        stop_score = max(0.0, 1.0 - tau / 0.2)  # decay to 0 after ~20% error
    return float(stop_score), bool(valid_end), bool(early)


@dataclass
class SampleResult:
    run_id: str
    gene_idx: int
    k: int
    sample_id: int
    aa_identity: float
    syn_rate: float
    stop_score: float
    frame_integrity: float
    ppl_stability: float
    no_repeat: float
    usage_agree: float
    gqs: float
    gen_len: int
    valid_end: bool
    early_stop: bool
    # long-protein generation metadata
    gen_len_codons: int
    had_terminal_stop: bool
    hit_hard_cap: bool
    target_codons: int
    termination_bias_steps: int
    last_termination_class: object
    critic_stability: float = 0.0
    critic_family_prob: float = 0.0
    critic_function_prob: float | None = None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_id", required=True)
    ap.add_argument(
        "--preset",
        choices=sorted(PRESETS),
        default=None,
        help="Evaluation size preset. Explicit --max_genes/--samples/--max_new override it.",
    )
    ap.add_argument("--k_list", default="1,3,5,10")
    ap.add_argument("--samples", type=int, default=None)
    ap.add_argument("--max_genes", type=int, default=None)
    ap.add_argument("--max_new", type=int, default=None)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument(
        "--device",
        choices=["auto", "cpu", "mps", "cuda"],
        default="auto",
        help="Device for generation/evaluation. Explicit unavailable devices fail fast.",
    )
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument(
        "--out_label",
        default="gen_prefix",
        help="Subdirectory under runs/<RUN_ID>/scores for outputs.",
    )
    ap.add_argument(
        "--progress_every",
        type=int,
        default=20,
        help="Print progress every N generated samples; 0 disables progress logs.",
    )
    # Long-protein controls
    ap.add_argument("--min_aa_len", type=int, default=100)
    ap.add_argument("--target_aa_len", type=int, default=256)
    ap.add_argument("--max_aa_len", type=int, default=400)
    ap.add_argument("--require_terminal_stop", action="store_true", default=False)
    ap.add_argument("--special_margin", type=int, default=6)
    ap.add_argument("--termination_bias", action="store_true", default=False)
    ap.add_argument("--termination_stop_bias", type=float, default=0.0)
    ap.add_argument("--termination_trigger_class_max", type=int, default=0)
    ap.add_argument(
        "--termination_bias_window",
        type=int,
        default=0,
        help="Only apply termination stop bias within this many codons of target length.",
    )
    ap.add_argument(
        "--allow_non_cds_tokens",
        action="store_true",
        default=False,
        help="Permit non-codon tokens during CDS continuation generation for diagnostics.",
    )
    ap.add_argument(
        "--multi_offset_prior",
        action="store_true",
        default=False,
        help="Enable multi-offset prior-guided decoding.",
    )
    ap.add_argument(
        "--multi_offset_prior_weights",
        type=str,
        default=None,
        help="JSON dict mapping offsets to prior weights (e.g. '{\"4\":0.1,\"8\":0.05}').",
    )
    # Normalization option for GQS
    ap.add_argument(
        "--gqs_normalize",
        choices=["none", "len"],
        default="none",
        help="Normalize GQS by reference length (truth length if available, else gen length)",
    )
    ap.add_argument(
        "--ckpt",
        default="best.pt",
        help="Which checkpoint to use (e.g., best.pt or last.pt)",
    )
    ap.add_argument(
        "--critic_stability",
        action="store_true",
        default=False,
        help="Enable MultiTask ProteinCritic stability and family classification evaluation.",
    )
    ap.add_argument(
        "--critic_ckpt",
        default="runs/protein_critic/checkpoints/best_critic.pt",
        help="Path to MultiTask ProteinCritic checkpoint.",
    )
    ap.add_argument(
        "--critic_cfg",
        default="configs/protein_critic.yaml",
        help="Path to ProteinCritic YAML config.",
    )
    ap.add_argument(
        "--critic_guidance",
        action="store_true",
        default=False,
        help="Enable active critic-guided blending during generation.",
    )
    ap.add_argument(
        "--ebm_guidance",
        action="store_true",
        default=False,
        help="Enable active EBM-guided blending during generation.",
    )
    ap.add_argument(
        "--ebm_ckpt",
        default="runs/protein_ebm_1024/checkpoints/best_ebm.pt",
        help="Path to EBM checkpoint.",
    )
    ap.add_argument(
        "--ebm_hidden_dim",
        type=int,
        default=1024,
        help="Hidden dimension of the EBM.",
    )
    ap.add_argument(
        "--guide_alpha",
        type=float,
        default=0.5,
        help="Blending weight alpha for critic/EBM guidance.",
    )
    ap.add_argument(
        "--guide_top_k",
        type=int,
        default=5,
        help="Pruning size guide_top_k for guided generation.",
    )
    args = ap.parse_args()
    preset = PRESETS.get(args.preset or "full", {})
    args.max_genes = int(args.max_genes if args.max_genes is not None else preset.get("max_genes", 50))
    args.samples = int(args.samples if args.samples is not None else preset.get("samples", 5))
    args.max_new = int(args.max_new if args.max_new is not None else preset.get("max_new", 300))
    _set_seed(int(args.seed))

    repo = Path(__file__).resolve().parents[1]
    run_dir = repo / "runs" / args.run_id
    out_dir = repo / "runs" / args.run_id / "scores" / args.out_label
    out_dir.mkdir(parents=True, exist_ok=True)

    # Custom loading for specific checkpoint
    meta = _load_run_meta(run_dir)

    # Try looking in consolidated layout first, then runs/ base, then legacy fallback
    weights_path = run_dir / "checkpoints" / args.ckpt
    if not weights_path.exists():
        weights_path = run_dir / args.ckpt
    if not weights_path.exists():
        weights_path = repo / "outputs" / "checkpoints" / args.run_id / args.ckpt
        if not weights_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {args.ckpt}")

    device = _select_device(args.device)
    print(
        f"[gen-prefix] run_id={args.run_id} ckpt={args.ckpt} device={device} "
        f"preset={args.preset or 'full'} max_genes={args.max_genes} samples={args.samples} "
        f"max_new={args.max_new} seed={args.seed}",
        flush=True,
    )
    ckpt = torch.load(weights_path, map_location=device)
    state_dict = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    spec = _model_spec_from(meta, ckpt)
    model = Q.build_model_from_state(state_dict, spec)
    cfg = _cfg_from(meta, ckpt)
    itos, stoi = _load_vocab_for_run(run_dir, repo, cfg)

    multi_offset_prior_weights = None
    if args.multi_offset_prior:
        if args.multi_offset_prior_weights:
            import json
            try:
                parsed = json.loads(args.multi_offset_prior_weights)
                multi_offset_prior_weights = {int(k): float(v) for k, v in parsed.items()}
            except Exception as exc:
                raise ValueError(f"Failed to parse --multi_offset_prior_weights: {exc}")
        else:
            # Load from model config
            raw_weights = cfg.get("multi_offset_weights", {})
            if isinstance(raw_weights, dict):
                multi_offset_prior_weights = {int(k): float(v) for k, v in raw_weights.items()}
            elif isinstance(raw_weights, (list, tuple)):
                targets = cfg.get("multi_offset_targets", [])
                multi_offset_prior_weights = {int(t): float(w) for t, w in zip(targets, raw_weights)}
            else:
                multi_offset_prior_weights = {}

    critic_model = None
    critic_tokenizer = None
    critic_task_dims = {}
    if args.critic_stability or args.critic_guidance or args.ebm_guidance:
        critic_model, critic_tokenizer, critic_task_dims = load_critic(
            args.critic_ckpt, args.critic_cfg, device
        )

    ebm_model = None
    if args.ebm_guidance:
        from src.protein_lm.ebm import ProteinLatentEBM
        ebm_model = ProteinLatentEBM(n_embd=256, hidden_dim=args.ebm_hidden_dim)
        ebm_state = torch.load(args.ebm_ckpt, map_location=device)
        if isinstance(ebm_state, dict) and "model" in ebm_state:
            ebm_state = ebm_state["model"]
        ebm_model.load_state_dict(ebm_state)
        ebm_model.to(device).eval()

    model.to(device).eval()
    # Validate AA length constraints
    if not (0 < args.min_aa_len <= args.target_aa_len <= args.max_aa_len):
        raise SystemExit("require 0 < min_aa_len ≤ target_aa_len ≤ max_aa_len")

    # Choose CDS corpus from legacy combined manifest, config, or hybrid manifest.
    dna_path = _resolve_cds_dna_path(repo, run_dir, cfg, max_genes=args.max_genes)
    if dna_path is None or not dna_path.exists():
        raise SystemExit(
            "[gen-prefix] could not locate a CDS dna file via combined or hybrid manifest"
        )

    # Build reference corpus codon unigram from test manifest if present (fallback: from dna file)
    codon_mask = np.array(
        [1 if (len(t) == 3 and all(c in "ACGT" for c in t)) else 0 for t in itos],
        dtype=np.int32,
    )
    unigram = np.zeros((len(itos),), dtype=np.float64)
    # quick pass over dna file
    lines = 0
    with open(dna_path) as f:
        for line in f:
            seq = line.strip().upper().replace("U", "T")
            L = (len(seq) // 3) * 3
            for i in range(0, L, 3):
                tok = seq[i : i + 3]
                j = stoi.get(tok, None)
                if j is not None:
                    unigram[j] += 1
            lines += 1
            if lines >= 10000:
                break
    if unigram.sum() == 0:
        unigram[:] = 1.0
    unigram_cod = unigram * codon_mask
    unigram_cod = unigram_cod / max(1e-9, unigram_cod.sum())

    # Load CDS list
    cds: List[str] = []
    with open(dna_path) as f:
        for line in f:
            s = line.strip().upper().replace("U", "T")
            if len(s) >= 9:
                cds.append(s)
            if len(cds) >= args.max_genes:
                break

    def aa_identity(truth: List[str], gen: List[str]) -> float:
        L = min(len(truth), len(gen))
        if L == 0:
            return 0.0
        return float(sum(1 for i in range(L) if truth[i] == gen[i])) / L

    def syn_rate(truth_cod: List[str], gen_cod: List[str]) -> float:
        L = min(len(truth_cod), len(gen_cod))
        if L == 0:
            return 0.0
        cnt = 0
        for i in range(L):
            a, b = _codon_to_aa(truth_cod[i]), _codon_to_aa(gen_cod[i])
            if a != "Stop" and b != "Stop" and a == b:
                cnt += 1
        return float(cnt) / L

    def ppl_stability(ids: List[int]) -> float:
        # Use mean NLL of first/last 10 tokens of continuation (approximate drift)
        if len(ids) < 22:
            return 1.0
        x = torch.tensor(ids[:-1], dtype=torch.long, device=device).unsqueeze(0)
        y = torch.tensor(ids[1:], dtype=torch.long, device=device).unsqueeze(0)
        with torch.no_grad():
            logits, _ = model(x, y)
            loss_all = torch.nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)),
                y.view(-1),
                ignore_index=0,
                reduction="none",
            ).view(1, -1)
        # first/last window
        w = min(10, loss_all.shape[1] // 4)
        first = loss_all[0, :w].mean().item()
        last = loss_all[0, -w:].mean().item()
        s = max(0.0, last - first)
        # map slope to [0,1]
        return float(np.exp(-s / 0.02))

    def usage_agree(gen_ids: List[int]) -> float:
        counts = np.zeros_like(unigram)
        for j in gen_ids:
            counts[int(j)] += 1
        p = counts * codon_mask
        s = p.sum()
        if s <= 0:
            return 0.0
        p = p / s
        kl = float((p * (np.log((p + 1e-12) / (unigram_cod + 1e-12)))).sum())
        # scale to [0,1] with heuristic KL0
        KL0 = 0.5
        return float(max(0.0, 1.0 - min(1.0, kl / KL0)))

    def frame_integrity_ok(gen_codons: List[str]) -> float:
        ok = all(len(c) == 3 and set(c) <= set("ACGT") for c in gen_codons)
        return 1.0 if ok else 0.0

    def gqs(stop_score, aaid, syn, stab, norep, usage, frame) -> float:
        return 100.0 * (
            0.30 * stop_score
            + 0.20 * aaid
            + 0.15 * syn
            + 0.10 * stab
            + 0.10 * norep
            + 0.10 * usage
            + 0.05 * frame
        )

    rows: List[SampleResult] = []
    k_list = [int(x) for x in args.k_list.split(",") if x]
    total_expected = len(cds) * len(k_list) * int(args.samples)
    done = 0
    wall0 = time.perf_counter()

    block_size = int(cfg.get("block_size", getattr(model, "block_size", 512)))
    for gene_idx, dna in enumerate(cds):
        truth_codons = [dna[i : i + 3] for i in range(0, (len(dna) // 3) * 3, 3)]
        truth_aa = _aa_seq(truth_codons)
        for k in k_list:
            prefix = dna[: 3 * min(k, len(truth_codons))]
            # tokenize prefix
            ctx_ids = Q.dna_prefix_to_ids(prefix, stoi)
            for sidx in range(args.samples):
                # Compute safe generation lengths (AA == codons)
                max_window_codons = block_size - int(k) - int(args.special_margin)
                if max_window_codons < args.min_aa_len:
                    raise ValueError("block_size too small for requested lengths and k")
                hard_cap = int(min(max_window_codons, args.max_aa_len, args.max_new))
                target_codons = int(min(args.target_aa_len, hard_cap))
                target_codons = int(max(target_codons, args.min_aa_len))
                # Constrained generation
                if args.critic_guidance or args.ebm_guidance:
                    gen_ids, info = generate_cds_critic_guided(
                        model=model,
                        critic_model=critic_model,
                        c_tokenizer=critic_tokenizer,
                        device=device,
                        ctx_ids=ctx_ids,
                        stoi=stoi,
                        itos=itos,
                        target_codons=target_codons,
                        hard_cap=hard_cap,
                        alpha=float(args.guide_alpha),
                        guide_top_k=int(args.guide_top_k),
                        target_task="ebm" if args.ebm_guidance else "stability",
                        target_class_idx=None,
                        ebm_model=ebm_model if args.ebm_guidance else None,
                        temperature=float(args.temperature),
                        cds_only=not bool(args.allow_non_cds_tokens),
                        require_terminal_stop=bool(args.require_terminal_stop)
                    )
                else:
                    gen_ids, info = generate_cds_constrained(
                        model=model,
                        device=device,
                        ctx_ids=ctx_ids,
                        stoi=stoi,
                        itos=itos,
                        target_codons=target_codons,
                        hard_cap=hard_cap,
                        require_terminal_stop=bool(args.require_terminal_stop),
                        temperature=float(args.temperature),
                        topk=int(args.topk) if args.topk > 0 else 0,
                        termination_bias_enabled=bool(args.termination_bias),
                        termination_stop_bias=float(args.termination_stop_bias),
                        termination_trigger_class_max=int(args.termination_trigger_class_max),
                        termination_bias_window=int(args.termination_bias_window),
                        cds_only=not bool(args.allow_non_cds_tokens),
                        multi_offset_prior_enabled=bool(args.multi_offset_prior),
                        multi_offset_prior_weights=multi_offset_prior_weights,
                    )
                gen_toks = Q.ids_to_codons(gen_ids, itos)
                # strip BOS and anything before first codon
                codons = [t for t in gen_toks if len(t) == 3 and set(t) <= set("ACGT")]
                # continuation after prefix length
                gen_cont_cod = codons[min(k, len(codons)) :]
                gen_cont_ids = [stoi[c] for c in gen_cont_cod if c in stoi]
                gen_cont_aa = _aa_seq(gen_cont_cod)
                # metrics
                aaid = aa_identity(truth_aa[k:], gen_cont_aa)
                syn = syn_rate(truth_codons[k:], gen_cont_cod)
                stop_score, valid_end, early = _score_stop_behavior(
                    codons, truth_len_codons=len(truth_codons)
                )
                stab = ppl_stability([stoi.get(c, 0) for c in codons])
                norep = 1.0 - _ngram_repeat_ratio(codons, n=3)
                usage = usage_agree(gen_cont_ids)
                frame = frame_integrity_ok(codons)
                score = gqs(stop_score, aaid, syn, stab, norep, usage, frame)

                critic_stability = 0.0
                critic_family_prob = 0.0
                critic_function_prob = 0.0
                if critic_model is not None:
                    aa_list = []
                    for c in codons:
                        aa = CODON_TO_AA.get(c, "X")
                        if aa == "Stop":
                            break
                        aa_list.append(aa)
                    aa_seq = "".join(aa_list)
                    crit_scores = score_with_critic(critic_model, critic_tokenizer, critic_task_dims, aa_seq, device)
                    if "stability" in critic_task_dims:
                        critic_stability = crit_scores.get("stability_prob", 0.0)
                    if "family" in critic_task_dims:
                        critic_family_prob = crit_scores.get("family_top1_conf", 0.0)
                    if "function" in critic_task_dims:
                        critic_function_prob = crit_scores.get("function_top1_conf", 0.0)

                rows.append(
                    SampleResult(
                        args.run_id,
                        gene_idx,
                        k,
                        sidx,
                        aaid,
                        syn,
                        stop_score,
                        frame,
                        stab,
                        norep,
                        usage,
                        score,
                        len(codons),
                        valid_end,
                        early,
                        gen_len_codons=len(codons),
                        had_terminal_stop=bool(info.get("had_terminal_stop", False)),
                        hit_hard_cap=bool(info.get("hit_hard_cap", False)),
                        target_codons=int(target_codons),
                        termination_bias_steps=int(info.get("termination_bias_steps", 0)),
                        last_termination_class=info.get("last_termination_class"),
                        critic_stability=critic_stability,
                        critic_family_prob=critic_family_prob,
                        critic_function_prob=critic_function_prob,
                    )
                )
                done += 1
                if args.progress_every and done % int(args.progress_every) == 0:
                    elapsed = time.perf_counter() - wall0
                    rate = done / max(elapsed, 1e-9)
                    remaining = max(0, total_expected - done)
                    eta = remaining / max(rate, 1e-9)
                    print(
                        f"[gen-prefix] progress {done}/{total_expected} "
                        f"rate={rate:.2f} samples/sec eta_sec={eta:.1f}",
                        flush=True,
                    )

    # write samples.csv
    samples_csv = out_dir / "samples.csv"
    with samples_csv.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([c for c in SampleResult.__annotations__.keys()])
        for r in rows:
            writer.writerow(
                [getattr(r, c) for c in SampleResult.__annotations__.keys()]
            )
    print(f"[gen-prefix] wrote {samples_csv}")

    # summary per k
    import statistics as stats

    summary = []
    for k in k_list:
        rks = [r for r in rows if r.k == k]
        if not rks:
            continue
        term_rate = sum(1 for r in rks if r.valid_end) / len(rks)
        early_rate = sum(1 for r in rks if r.early_stop) / len(rks)
        median_gqs = float(stats.median([r.gqs for r in rks]))
        mean_aa = float(sum(r.aa_identity for r in rks) / len(rks))
        best_aa = float(max(r.aa_identity for r in rks))
        mean_len = float(sum(r.gen_len_codons for r in rks) / len(rks))
        median_len = float(stats.median([r.gen_len_codons for r in rks]))
        stop_rate = sum(1 for r in rks if r.had_terminal_stop) / len(rks)
        hard_cap_rate = sum(1 for r in rks if r.hit_hard_cap) / len(rks)
        # Optional length-normalized GQS
        mean_gqs_norm = None
        median_gqs_norm = None
        if args.gqs_normalize == "len":
            # Use target_codons (proxy for truth length) when available; fallback to gen_len
            norms = []
            for r in rks:
                denom = max(1, getattr(r, "target_codons", 0) or r.gen_len_codons)
                norms.append(r.gqs / float(denom))
            if norms:
                mean_gqs_norm = float(sum(norms) / len(norms))
                median_gqs_norm = float(stats.median(norms))

        # Optional critic summary
        mean_crit_stab = None
        mean_crit_fam = None
        mean_crit_func = None
        if any(getattr(r, "critic_stability", 0.0) > 0.0 for r in rks):
            mean_crit_stab = float(sum(r.critic_stability for r in rks) / len(rks))
            mean_crit_fam = float(sum(r.critic_family_prob for r in rks) / len(rks))
            mean_crit_func = float(sum(r.critic_function_prob for r in rks) / len(rks))

        summary.append(
            {
                "k": k,
                "termination_rate": term_rate,
                "early_stop_rate": early_rate,
                "median_gqs": median_gqs,
                "mean_aa_identity": mean_aa,
                "best_aa_identity": best_aa,
                "mean_aa_len": mean_len,
                "median_aa_len": median_len,
                "terminal_stop_rate": stop_rate,
                "hard_cap_rate": hard_cap_rate,
                **(
                    {"mean_gqs_norm": mean_gqs_norm, "median_gqs_norm": median_gqs_norm}
                    if args.gqs_normalize == "len"
                    else {}
                ),
                **(
                    {
                        "mean_critic_stability": mean_crit_stab,
                        "mean_critic_family_prob": mean_crit_fam,
                        "mean_critic_function_prob": mean_crit_func,
                    }
                    if mean_crit_stab is not None
                    else {}
                ),
                "n": len(rks),
            }
        )
    summary_csv = out_dir / "summary.csv"
    with summary_csv.open("w", newline="") as f:
        base_cols = [
            "k",
            "termination_rate",
            "early_stop_rate",
            "median_gqs",
            "mean_aa_identity",
            "best_aa_identity",
            "mean_aa_len",
            "median_aa_len",
            "terminal_stop_rate",
            "hard_cap_rate",
            "n",
        ]
        extra = []
        if any("mean_gqs_norm" in s for s in summary):
            extra += ["mean_gqs_norm", "median_gqs_norm"]
        if any("mean_critic_stability" in s for s in summary):
            extra += ["mean_critic_stability", "mean_critic_family_prob", "mean_critic_function_prob"]

        writer = csv.DictWriter(f, fieldnames=base_cols + extra)
        writer.writeheader()
        writer.writerows(summary)
    print(f"[gen-prefix] wrote {summary_csv}")

    # simple plots
    try:
        ks = [s["k"] for s in summary]
        tr = [s["termination_rate"] for s in summary]
        gq = [s["median_gqs"] for s in summary]
        aa = [s["mean_aa_identity"] for s in summary]
        plt.figure()
        plt.plot(ks, tr, marker="o")
        plt.xlabel("k")
        plt.ylabel("termination_rate")
        plt.title("Termination vs k")
        plt.tight_layout()
        plt.savefig(out_dir / "termination_vs_k.png")
        plt.close()
        plt.figure()
        plt.plot(ks, gq, marker="o")
        plt.xlabel("k")
        plt.ylabel("median_gqs")
        plt.title("GQS vs k")
        plt.tight_layout()
        plt.savefig(out_dir / "gqs_vs_k.png")
        plt.close()
        plt.figure()
        plt.plot(ks, aa, marker="o")
        plt.xlabel("k")
        plt.ylabel("mean_aa_identity")
        plt.title("AA identity vs k")
        plt.tight_layout()
        plt.savefig(out_dir / "aa_vs_k.png")
        plt.close()
        ml = [s["mean_aa_len"] for s in summary]
        plt.figure()
        plt.plot(ks, ml, marker="o")
        plt.xlabel("k")
        plt.ylabel("mean_aa_len")
        plt.title("AA length vs k")
        plt.tight_layout()
        plt.savefig(out_dir / "aa_len_vs_k.png")
        try:
            figs_root = Path(__file__).resolve().parents[1] / "outputs" / "figs"
            figs_root.mkdir(parents=True, exist_ok=True)
            plt.savefig(figs_root / "aa_len_vs_k.png")
        except Exception:
            pass
        plt.close()
    except Exception as exc:
        print(f"[gen-prefix] plotting failed: {exc}")


if __name__ == "__main__":
    main()
