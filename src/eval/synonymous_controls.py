"""Synonymous control generation and paired likelihood evaluation for CodonLM.

Tests whether CodonLM learns biological codon choice (codon usage bias and
higher-order sequence context) beyond trivial amino-acid protein translation.
Natural bacterial coding sequences are paired with synonymous alternatives
encoding the exact same protein sequence, preserving start codon, stop codon,
gene length, and reading frame.

Success check:
    ΔNLL = NLL_shuffled - NLL_natural
The pilot succeeds if median(ΔNLL) > 0, meaning natural sequences consistently
receive lower negative log-likelihood (higher probability) than synonymous controls.
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

from src.eval.translation import (
    BACTERIAL_START_CODONS,
    CODON_TABLE_11,
    STOP_CODONS,
    translate_codons,
)

# Reverse mapping: Amino Acid -> list of synonymous codons in Table 11
AA_TO_CODONS: dict[str, list[str]] = {}
for _codon, _aa in sorted(CODON_TABLE_11.items()):
    AA_TO_CODONS.setdefault(_aa, []).append(_codon)


@dataclass
class PairedGeneEvaluation:
    gene_id: str
    length_codons: int
    evaluated_tokens: int
    nll_natural: float
    nll_control: float
    delta_nll: float
    ppl_natural: float
    ppl_control: float
    natural_start: str
    natural_stop: str
    control_start: str
    control_stop: str
    protein_length: int
    protein_match: bool


@dataclass
class SynonymousPilotSummary:
    num_genes: int
    mode: str
    seed: int
    median_delta_nll: float
    mean_delta_nll: float
    std_delta_nll: float
    min_delta_nll: float
    max_delta_nll: float
    q25_delta_nll: float
    q75_delta_nll: float
    win_rate: float
    mean_natural_nll: float
    mean_control_nll: float
    median_natural_nll: float
    median_control_nll: float
    mean_natural_ppl: float
    mean_control_ppl: float
    wilcoxon_statistic: float | None
    wilcoxon_p_value: float | None
    success: bool
    gene_results: list[PairedGeneEvaluation] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["gene_results"] = [asdict(r) for r in self.gene_results]
        return data


def _to_codon_list(sequence: str | Sequence[str]) -> list[str]:
    """Convert string or sequence of codons into a list of uppercase codon strings."""
    if isinstance(sequence, str):
        cleaned = sequence.strip()
        if "-" in cleaned or " " in cleaned:
            tokens = cleaned.replace("-", " ").split()
        elif len(cleaned) % 3 == 0 and all(c in "ACGTNacgtn" for c in cleaned):
            tokens = [cleaned[i : i + 3] for i in range(0, len(cleaned), 3)]
        else:
            tokens = cleaned.split()
    else:
        tokens = list(sequence)
    return [t.upper() for t in tokens]


def reshuffle_synonymous_gene(
    gene: str | Sequence[str] | Sequence[int],
    *,
    rng: random.Random | int | None = None,
    mode: str = "uniform_alternative",
    preserve_start: bool = True,
    preserve_stop: bool = True,
    itos: Sequence[str] | None = None,
    stoi: dict[str, int] | None = None,
    table: dict[str, str] | None = None,
) -> Any:
    """Synonymously recode each gene, preserving protein sequence and gene length.

    Parameters
    ----------
    gene
        Sequence of codons (str, list[str], or list[int] token IDs if itos/stoi are provided).
    rng
        Random number generator or integer seed.
    mode
        - "uniform_alternative": For each codon, uniformly choose from synonymous codons
          excluding the current codon (if alternatives exist). Degenerate 1-fold codons
          (M, W) remain unchanged.
        - "uniform_any": For each codon, uniformly choose from all synonymous codons
          for that amino acid (including the current codon).
        - "within_gene_shuffle": Permute the existing synonymous codons for each amino acid
          within this gene, strictly preserving within-gene codon frequencies.
    preserve_start
        If True, locks the start codon (index 0) so initiation bias is unperturbed.
    preserve_stop
        If True, locks the terminal stop codon (index -1) if it is a recognized stop.
    itos / stoi
        Vocabulary lookups required if `gene` is passed as integer token IDs.
    table
        Codon to amino acid lookup table (defaults to NCBI Table 11).

    Returns
    -------
    Mutated sequence in the same type/format as the input.
    """
    if table is None:
        table = CODON_TABLE_11

    if isinstance(rng, int):
        rand = random.Random(rng)
    elif isinstance(rng, random.Random):
        rand = rng
    else:
        rand = random.Random()

    is_token_ids = False
    if len(gene) > 0 and isinstance(gene[0], (int, np.integer)):
        if itos is None or stoi is None:
            raise ValueError("itos and stoi are required when gene is a sequence of token IDs")
        is_token_ids = True
        codons = [itos[int(idx)] for idx in gene]
    else:
        codons = _to_codon_list(gene)  # type: ignore[arg-type]

    if not codons:
        return [] if is_token_ids or not isinstance(gene, str) else ""

    # Reverse lookup map: AA -> list of synonymous codons
    aa_to_codons: dict[str, list[str]] = {}
    for c, aa in table.items():
        aa_to_codons.setdefault(aa, []).append(c)

    start_idx = 1 if preserve_start and len(codons) > 1 else 0
    end_idx = len(codons) - 1 if preserve_stop and len(codons) > 1 and codons[-1] in STOP_CODONS else len(codons)

    mutated = list(codons)

    if mode == "within_gene_shuffle":
        # Group mutable positions by amino acid
        aa_positions: dict[str, list[int]] = {}
        for pos in range(start_idx, end_idx):
            codon = codons[pos]
            aa = table.get(codon)
            if aa is not None:
                aa_positions.setdefault(aa, []).append(pos)

        # Permute within each amino acid group
        for aa, positions in aa_positions.items():
            if len(positions) > 1:
                codon_pool = [codons[p] for p in positions]
                rand.shuffle(codon_pool)
                for p, new_codon in zip(positions, codon_pool):
                    mutated[p] = new_codon

    elif mode in ("uniform_alternative", "uniform_any"):
        for pos in range(start_idx, end_idx):
            codon = codons[pos]
            aa = table.get(codon)
            if aa is None or aa == "*":
                # Unknown token or internal stop; leave as is
                continue
            family = aa_to_codons.get(aa, [codon])
            if mode == "uniform_alternative":
                alternatives = [c for c in family if c != codon]
                if alternatives:
                    mutated[pos] = rand.choice(alternatives)
            else:
                mutated[pos] = rand.choice(family)
    else:
        raise ValueError(
            f"Unknown mode {mode!r}; expected 'uniform_alternative', 'uniform_any', or 'within_gene_shuffle'"
        )

    # Invariant verification: exact protein match
    orig_protein = translate_codons(codons, table=table).protein_sequence
    mut_protein = translate_codons(mutated, table=table).protein_sequence
    if orig_protein != mut_protein:
        raise RuntimeError(
            f"Synonymous mutation invariant violated: natural translates to {orig_protein[:20]}... "
            f"while mutated translates to {mut_protein[:20]}..."
        )

    if is_token_ids:
        assert stoi is not None
        return [stoi[c] for c in mutated]
    elif isinstance(gene, str):
        if "-" in gene:
            return "-".join(mutated)
        elif " " in gene:
            return " ".join(mutated)
        return "".join(mutated)
    return mutated


@torch.no_grad()
def calculate_token_sequence_nll(
    model: torch.nn.Module,
    device: torch.device,
    token_ids: Sequence[int],
) -> tuple[float, float, int]:
    """Calculate length-normalized NLL and perplexity for a single sequence.

    Parameters
    ----------
    model
        Causal language model.
    device
        Evaluation torch device.
    token_ids
        Sequence of token IDs (e.g. including BOS/EOS if trained with them).

    Returns
    -------
    (nll_nats, perplexity, valid_token_count)
    """
    if len(token_ids) < 2:
        raise ValueError("Sequence must contain at least 2 tokens (context and target)")

    x = torch.tensor([token_ids[:-1]], dtype=torch.long, device=device)
    y = torch.tensor([token_ids[1:]], dtype=torch.long, device=device)

    logits, _ = model(x)
    flat_logits = logits.float().view(-1, logits.size(-1))
    flat_targets = y.view(-1)

    loss_sum = F.cross_entropy(
        flat_logits,
        flat_targets,
        ignore_index=0,
        reduction="sum",
    ).item()
    valid_tokens = int((flat_targets != 0).sum().item())
    if valid_tokens == 0:
        return 0.0, 1.0, 0

    mean_nll = float(loss_sum / valid_tokens)
    ppl = float(math.exp(min(20.0, mean_nll)))
    return mean_nll, ppl, valid_tokens


def evaluate_gene_pair_nll(
    model: torch.nn.Module,
    device: torch.device,
    natural_codons: list[str],
    control_codons: list[str],
    *,
    gene_id: str,
    stoi: dict[str, int],
    itos: Sequence[str],
    include_bos: bool = True,
    include_eos: bool = True,
    bos_token: str = "<BOS_CDS>",
    eos_token: str = "<EOS_CDS>",
    table: dict[str, str] | None = None,
) -> PairedGeneEvaluation:
    """Evaluate natural vs synonymous control pair and compute ΔNLL."""
    if table is None:
        table = CODON_TABLE_11

    # Prepare token sequences
    def _to_ids(codon_list: list[str]) -> list[int]:
        ids: list[int] = []
        if include_bos and bos_token in stoi:
            ids.append(stoi[bos_token])
        for c in codon_list:
            if c in stoi:
                ids.append(stoi[c])
            else:
                raise KeyError(f"Codon {c} not found in model vocabulary")
        if include_eos and eos_token in stoi:
            ids.append(stoi[eos_token])
        return ids

    nat_ids = _to_ids(natural_codons)
    ctrl_ids = _to_ids(control_codons)

    nll_nat, ppl_nat, tokens_nat = calculate_token_sequence_nll(model, device, nat_ids)
    nll_ctrl, ppl_ctrl, tokens_ctrl = calculate_token_sequence_nll(model, device, ctrl_ids)

    delta_nll = nll_ctrl - nll_nat

    nat_report = translate_codons(natural_codons, table=table)
    ctrl_report = translate_codons(control_codons, table=table)

    return PairedGeneEvaluation(
        gene_id=gene_id,
        length_codons=len(natural_codons),
        evaluated_tokens=tokens_nat,
        nll_natural=nll_nat,
        nll_control=nll_ctrl,
        delta_nll=delta_nll,
        ppl_natural=ppl_nat,
        ppl_control=ppl_ctrl,
        natural_start=natural_codons[0] if natural_codons else "",
        natural_stop=natural_codons[-1] if natural_codons else "",
        control_start=control_codons[0] if control_codons else "",
        control_stop=control_codons[-1] if control_codons else "",
        protein_length=nat_report.translated_length,
        protein_match=nat_report.protein_sequence == ctrl_report.protein_sequence,
    )


def extract_validation_genes(
    manifest_or_dir: Path | str,
    *,
    num_genes: int = 100,
    min_codons: int = 30,
    max_codons: int = 500,
    seed: int = 1337,
) -> list[tuple[str, list[str]]]:
    """Extract intact coding sequences belonging to the validation split.

    Returns list of (gene_id, list_of_codons).
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

    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    manifest = json.loads(manifest_path.read_text())
    artifacts = manifest.get("artifacts", {})

    frag_meta_rel = artifacts.get("fragment_metadata", {}).get("path", "cds_fragments.tsv")
    frag_meta_path = dataset_dir / frag_meta_rel
    source_dna_rel = artifacts.get("source_dna", {}).get("path", "cds_dna.txt")
    source_dna_path = dataset_dir / source_dna_rel

    if not frag_meta_path.exists() or not source_dna_path.exists():
        raise FileNotFoundError(
            f"Missing required artifacts: {frag_meta_path} or {source_dna_path}"
        )

    # Read DNA lines into indexed access
    with open(source_dna_path, "r", encoding="utf-8") as f:
        dna_lines = [line.strip() for line in f]

    candidates: list[tuple[str, list[str]]] = []
    with open(frag_meta_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            if row.get("split") != "val":
                continue
            codon_count = int(row.get("codon_count", 0))
            if not (min_codons <= codon_count <= max_codons):
                continue
            src_idx = int(row.get("source_line_idx", -1))
            if src_idx < 0 or src_idx >= len(dna_lines):
                continue
            dna = dna_lines[src_idx]
            if len(dna) != codon_count * 3:
                continue
            codons = [dna[i : i + 3] for i in range(0, len(dna), 3)]
            if not codons:
                continue
            # Validate start codon and stop codon
            if codons[0] not in BACTERIAL_START_CODONS:
                continue
            if codons[-1] not in STOP_CODONS:
                continue
            # Ensure no internal stops
            report = translate_codons(codons)
            if not report.is_valid_orf:
                continue

            gene_id = row.get("source_id", f"val_gene_{src_idx}")
            candidates.append((gene_id, codons))

    if len(candidates) < num_genes:
        raise ValueError(
            f"Only {len(candidates)} valid ORF candidates found in validation set, "
            f"requested {num_genes}"
        )

    rng = random.Random(seed)
    selected = rng.sample(candidates, num_genes)
    return selected


def run_synonymous_pilot(
    model: torch.nn.Module,
    device: torch.device,
    genes: list[tuple[str, list[str]]],
    *,
    itos: Sequence[str],
    stoi: dict[str, int],
    mode: str = "uniform_alternative",
    seed: int = 1337,
    table: dict[str, str] | None = None,
) -> SynonymousPilotSummary:
    """Run paired natural vs synonymous control evaluation across genes."""
    if table is None:
        table = CODON_TABLE_11

    rng = random.Random(seed)
    gene_results: list[PairedGeneEvaluation] = []

    for gene_id, nat_codons in genes:
        # Create 1 matched synonymous control
        ctrl_codons = reshuffle_synonymous_gene(
            nat_codons,
            rng=rng,
            mode=mode,
            preserve_start=True,
            preserve_stop=True,
            table=table,
        )
        res = evaluate_gene_pair_nll(
            model,
            device,
            nat_codons,
            ctrl_codons,
            gene_id=gene_id,
            stoi=stoi,
            itos=itos,
            table=table,
        )
        gene_results.append(res)

    deltas = np.array([r.delta_nll for r in gene_results], dtype=np.float64)
    nat_nlls = np.array([r.nll_natural for r in gene_results], dtype=np.float64)
    ctrl_nlls = np.array([r.nll_control for r in gene_results], dtype=np.float64)
    nat_ppls = np.array([r.ppl_natural for r in gene_results], dtype=np.float64)
    ctrl_ppls = np.array([r.ppl_control for r in gene_results], dtype=np.float64)

    median_delta = float(np.median(deltas))
    mean_delta = float(np.mean(deltas))
    std_delta = float(np.std(deltas, ddof=1)) if len(deltas) > 1 else 0.0
    win_rate = float(np.mean(deltas > 0.0))

    wilcoxon_stat = None
    wilcoxon_p = None
    try:
        from scipy.stats import wilcoxon

        w_res = wilcoxon(ctrl_nlls - nat_nlls, alternative="greater")
        wilcoxon_stat = float(w_res.statistic)
        wilcoxon_p = float(w_res.pvalue)
    except Exception:
        pass

    return SynonymousPilotSummary(
        num_genes=len(gene_results),
        mode=mode,
        seed=seed,
        median_delta_nll=median_delta,
        mean_delta_nll=mean_delta,
        std_delta_nll=std_delta,
        min_delta_nll=float(np.min(deltas)),
        max_delta_nll=float(np.max(deltas)),
        q25_delta_nll=float(np.percentile(deltas, 25)),
        q75_delta_nll=float(np.percentile(deltas, 75)),
        win_rate=win_rate,
        mean_natural_nll=float(np.mean(nat_nlls)),
        mean_control_nll=float(np.mean(ctrl_nlls)),
        median_natural_nll=float(np.median(nat_nlls)),
        median_control_nll=float(np.median(ctrl_nlls)),
        mean_natural_ppl=float(np.mean(nat_ppls)),
        mean_control_ppl=float(np.mean(ctrl_ppls)),
        wilcoxon_statistic=wilcoxon_stat,
        wilcoxon_p_value=wilcoxon_p,
        success=bool(median_delta > 0.0),
        gene_results=gene_results,
    )


__all__ = [
    "AA_TO_CODONS",
    "PairedGeneEvaluation",
    "SynonymousPilotSummary",
    "calculate_token_sequence_nll",
    "evaluate_gene_pair_nll",
    "extract_validation_genes",
    "reshuffle_synonymous_gene",
    "run_synonymous_pilot",
]
