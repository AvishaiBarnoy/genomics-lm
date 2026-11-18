"""External protein sequence plausibility critic using pretrained ESM-2 (8M).

Establishes the generator -> translation -> external critic evaluation interface.
Evaluates translated coding sequences by computing mean masked-residue log-likelihood
from the frozen ESM-2 (8M parameter, esm2_t6_8M_UR50D) protein language model.
"""

from __future__ import annotations

import csv
import json
import math
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from src.eval.translation import TranslationReport, translate_codons


@dataclass
class ProteinScoreResult:
    sequence_id: str
    source_type: str  # "generated" or "real_validation"
    protein_length: int
    evaluated_positions: int
    mean_masked_log_likelihood: float
    perplexity: float
    is_valid_orf: bool
    protein_sequence: str


@dataclass
class ESM2CriticBenchmarkSummary:
    num_generated: int
    num_real: int
    num_positions_per_protein: int
    seed: int
    all_scores_finite: bool
    median_real_score: float
    mean_real_score: float
    std_real_score: float
    median_generated_score: float
    mean_generated_score: float
    std_generated_score: float
    delta_median_score: float  # real - generated (expected > 0)
    delta_mean_score: float
    mann_whitney_u: float | None
    p_value: float | None
    success: bool
    results: list[ProteinScoreResult] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["results"] = [asdict(r) for r in self.results]
        return data


class ESM2ProteinCritic:
    """Evaluates protein sequence plausibility via masked residue log-likelihood in ESM-2."""

    def __init__(
        self,
        model_name: str = "esm2_t6_8M_UR50D",
        device: torch.device | str | None = None,
    ):
        import esm

        if device is None:
            if torch.backends.mps.is_available():
                self.device = torch.device("mps")
            elif torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)

        load_fn = getattr(esm.pretrained, model_name, None)
        if load_fn is None:
            raise ValueError(f"Unknown ESM model {model_name!r}")

        self.model, self.alphabet = load_fn()
        self.model.to(self.device).eval()
        self.batch_converter = self.alphabet.get_batch_converter()
        self.mask_idx = self.alphabet.mask_idx

    @torch.no_grad()
    def score_protein(
        self,
        protein_sequence: str,
        *,
        num_positions: int = 16,
        seed: int | None = 42,
    ) -> tuple[float, int]:
        """Compute mean masked-residue log-likelihood across random positions.

        Parameters
        ----------
        protein_sequence
            Amino acid sequence string (e.g. "MKALVAG...").
        num_positions
            Number of random positions to evaluate (default: 16).
        seed
            RNG seed for position sampling (ensures deterministic evaluation).

        Returns
        -------
        (mean_masked_log_likelihood, evaluated_position_count)
        """
        # Filter for valid uppercase amino acid characters
        clean_seq = "".join(c for c in protein_sequence.upper() if c in "ACDEFGHIKLMNPQRSTVWY")
        seq_len = len(clean_seq)
        if seq_len == 0:
            return float("-inf"), 0

        # Sample positions (0-indexed relative to clean_seq)
        k = min(num_positions, seq_len)
        rng = random.Random(seed) if seed is not None else random.Random()
        positions = sorted(rng.sample(range(seq_len), k))

        # Convert to tokens: [<cls>, x_0, x_1, ..., x_{L-1}, <eos>]
        _, _, batch_tokens = self.batch_converter([("target", clean_seq)])
        batch_tokens = batch_tokens.to(self.device)  # shape (1, seq_len + 2)

        # Build batch of k masked copies
        masked_batch = batch_tokens.repeat(k, 1)
        for i, pos in enumerate(positions):
            # Token index is pos + 1 due to <cls> at index 0
            masked_batch[i, pos + 1] = self.mask_idx

        # Forward pass
        results = self.model(masked_batch)
        logits = results["logits"]  # shape (k, seq_len + 2, vocab_size)

        log_likelihoods: list[float] = []
        for i, pos in enumerate(positions):
            token_idx = pos + 1
            target_token_id = batch_tokens[0, token_idx].item()
            token_logits = logits[i, token_idx].float()
            log_probs = F.log_softmax(token_logits, dim=-1)
            target_lp = float(log_probs[target_token_id].item())
            log_likelihoods.append(target_lp)

        mean_ll = float(np.mean(log_likelihoods))
        return mean_ll, k

    def score_cds(
        self,
        cds_sequence: str | Sequence[str],
        *,
        sequence_id: str = "seq",
        source_type: str = "generated",
        num_positions: int = 16,
        seed: int = 42,
    ) -> tuple[ProteinScoreResult, TranslationReport]:
        """Translate CDS and evaluate translated protein with ESM-2."""
        report = translate_codons(cds_sequence)
        protein = report.protein_sequence

        if not protein:
            result = ProteinScoreResult(
                sequence_id=sequence_id,
                source_type=source_type,
                protein_length=0,
                evaluated_positions=0,
                mean_masked_log_likelihood=float("-inf"),
                perplexity=float("inf"),
                is_valid_orf=report.is_valid_orf,
                protein_sequence="",
            )
            return result, report

        mean_ll, evaluated_k = self.score_protein(
            protein, num_positions=num_positions, seed=seed
        )
        ppl = float(math.exp(min(20.0, -mean_ll))) if mean_ll > float("-inf") else float("inf")

        result = ProteinScoreResult(
            sequence_id=sequence_id,
            source_type=source_type,
            protein_length=len(protein),
            evaluated_positions=evaluated_k,
            mean_masked_log_likelihood=mean_ll,
            perplexity=ppl,
            is_valid_orf=report.is_valid_orf,
            protein_sequence=protein,
        )
        return result, report


def translate_and_score_cds(
    cds: str | Sequence[str],
    critic: ESM2ProteinCritic | None = None,
    *,
    num_positions: int = 16,
    seed: int = 42,
) -> tuple[float, TranslationReport]:
    """Convenience functional interface for scoring a single CDS."""
    if critic is None:
        critic = ESM2ProteinCritic()
    res, report = critic.score_cds(cds, num_positions=num_positions, seed=seed)
    return res.mean_masked_log_likelihood, report


def extract_length_matched_validation_proteins(
    manifest_or_dir: Path | str,
    target_lengths: list[int],
    *,
    seed: int = 1337,
    tolerance: int = 10,
) -> list[tuple[str, str]]:
    """Extract real validation proteins length-matched to target lengths.

    Returns list of (source_id, protein_sequence).
    """
    root = Path(manifest_or_dir).expanduser().resolve()
    if root.is_file() and root.name == "manifest.json":
        manifest_path = root
        dataset_dir = root.parent
    elif root.is_dir():
        manifest_path = root / "manifest.json"
        dataset_dir = root
    else:
        raise FileNotFoundError(f"Cannot locate dataset manifest at {root}")

    manifest = json.loads(manifest_path.read_text())
    artifacts = manifest.get("artifacts", {})

    frag_meta_path = dataset_dir / artifacts.get("fragment_metadata", {}).get("path", "cds_fragments.tsv")
    source_dna_path = dataset_dir / artifacts.get("source_dna", {}).get("path", "cds_dna.txt")

    with open(source_dna_path, "r", encoding="utf-8") as f:
        dna_lines = [line.strip() for line in f]

    pool: list[tuple[str, str, int]] = []
    with open(frag_meta_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            if row.get("split") != "val":
                continue
            src_idx = int(row.get("source_line_idx", -1))
            if 0 <= src_idx < len(dna_lines):
                dna = dna_lines[src_idx]
                report = translate_codons(dna)
                if report.is_valid_orf and len(report.protein_sequence) >= 20:
                    pool.append((
                        row.get("source_id", f"val_{src_idx}"),
                        report.protein_sequence,
                        len(report.protein_sequence),
                    ))

    if not pool:
        raise ValueError(
            f"No valid ORFs found in validation split ({manifest_path}) for length matching"
        )

    rng = random.Random(seed)
    selected: list[tuple[str, str]] = []
    used_indices: set[int] = set()

    for target_len in target_lengths:
        candidates = [
            i for i, (_, _, l) in enumerate(pool)
            if i not in used_indices and abs(l - target_len) <= tolerance
        ]
        if not candidates:
            # Fallback to nearest available in pool
            candidates = sorted(
                [i for i in range(len(pool)) if i not in used_indices] or list(range(len(pool))),
                key=lambda i: abs(pool[i][2] - target_len),
            )
        chosen_idx = rng.choice(candidates[:min(5, len(candidates))])
        used_indices.add(chosen_idx)
        selected.append((pool[chosen_idx][0], pool[chosen_idx][1]))

    return selected


def run_esm2_critic_benchmark(
    critic: ESM2ProteinCritic,
    generated_proteins: list[tuple[str, str]],
    real_proteins: list[tuple[str, str]],
    *,
    num_positions: int = 16,
    seed: int = 42,
) -> ESM2CriticBenchmarkSummary:
    """Run ESM-2 plausibility benchmark comparing generated vs real validation proteins."""
    results: list[ProteinScoreResult] = []

    # Score generated proteins
    for idx, (seq_id, protein) in enumerate(generated_proteins):
        mean_ll, k = critic.score_protein(
            protein, num_positions=num_positions, seed=seed + idx
        )
        ppl = float(math.exp(min(20.0, -mean_ll))) if mean_ll > float("-inf") else float("inf")
        results.append(
            ProteinScoreResult(
                sequence_id=seq_id,
                source_type="generated",
                protein_length=len(protein),
                evaluated_positions=k,
                mean_masked_log_likelihood=mean_ll,
                perplexity=ppl,
                is_valid_orf=True,
                protein_sequence=protein,
            )
        )

    # Score real validation proteins
    for idx, (seq_id, protein) in enumerate(real_proteins):
        mean_ll, k = critic.score_protein(
            protein, num_positions=num_positions, seed=seed + idx
        )
        ppl = float(math.exp(min(20.0, -mean_ll))) if mean_ll > float("-inf") else float("inf")
        results.append(
            ProteinScoreResult(
                sequence_id=seq_id,
                source_type="real_validation",
                protein_length=len(protein),
                evaluated_positions=k,
                mean_masked_log_likelihood=mean_ll,
                perplexity=ppl,
                is_valid_orf=True,
                protein_sequence=protein,
            )
        )

    gen_scores = [r.mean_masked_log_likelihood for r in results if r.source_type == "generated"]
    real_scores = [r.mean_masked_log_likelihood for r in results if r.source_type == "real_validation"]

    all_finite = all(math.isfinite(s) for s in gen_scores + real_scores)

    median_gen = float(np.median(gen_scores))
    mean_gen = float(np.mean(gen_scores))
    std_gen = float(np.std(gen_scores, ddof=1)) if len(gen_scores) > 1 else 0.0

    median_real = float(np.median(real_scores))
    mean_real = float(np.mean(real_scores))
    std_real = float(np.std(real_scores, ddof=1)) if len(real_scores) > 1 else 0.0

    delta_median = median_real - median_gen
    delta_mean = mean_real - mean_gen

    mw_stat = None
    p_val = None
    try:
        from scipy.stats import mannwhitneyu

        stat, p = mannwhitneyu(real_scores, gen_scores, alternative="greater")
        mw_stat = float(stat)
        p_val = float(p)
    except Exception:
        pass

    success = all_finite and (median_real > median_gen)

    return ESM2CriticBenchmarkSummary(
        num_generated=len(gen_scores),
        num_real=len(real_scores),
        num_positions_per_protein=num_positions,
        seed=seed,
        all_scores_finite=all_finite,
        median_real_score=median_real,
        mean_real_score=mean_real,
        std_real_score=std_real,
        median_generated_score=median_gen,
        mean_generated_score=mean_gen,
        std_generated_score=std_gen,
        delta_median_score=delta_median,
        delta_mean_score=delta_mean,
        mann_whitney_u=mw_stat,
        p_value=p_val,
        success=success,
        results=results,
    )


__all__ = [
    "ESM2CriticBenchmarkSummary",
    "ESM2ProteinCritic",
    "ProteinScoreResult",
    "extract_length_matched_validation_proteins",
    "run_esm2_critic_benchmark",
    "translate_and_score_cds",
]
