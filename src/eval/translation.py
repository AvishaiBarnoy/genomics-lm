"""Centralised DNA-to-protein translation evaluation for generated CDS.

Uses NCBI Translation Table 11 (Bacterial, Archaeal, and Plant Plastid Code).
This is the **standard** genetic code — identical to Table 1 except that
Table 11 additionally allows GTG/TTG/CTG as alternative start codons in
bacteria.  The amino-acid mapping for sense codons is the same as the
universal table, so the codon→AA dict below is correct for both.

This module is the single source of truth for codon translation in the eval
pipeline.  Downstream callers (generation quality scoring, learned protein
critic) should import from here rather than defining their own tables.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Sequence

# ── NCBI Translation Table 11 ──────────────────────────────────────────────
# Bacterial / Archaeal / Plant-plastid genetic code.
# Stop codons are mapped to '*' (standard IUPAC convention).
# fmt: off
CODON_TABLE_11: dict[str, str] = {
    "TTT": "F", "TTC": "F", "TTA": "L", "TTG": "L",
    "TCT": "S", "TCC": "S", "TCA": "S", "TCG": "S",
    "TAT": "Y", "TAC": "Y", "TAA": "*", "TAG": "*",
    "TGT": "C", "TGC": "C", "TGA": "*", "TGG": "W",
    "CTT": "L", "CTC": "L", "CTA": "L", "CTG": "L",
    "CCT": "P", "CCC": "P", "CCA": "P", "CCG": "P",
    "CAT": "H", "CAC": "H", "CAA": "Q", "CAG": "Q",
    "CGT": "R", "CGC": "R", "CGA": "R", "CGG": "R",
    "ATT": "I", "ATC": "I", "ATA": "I", "ATG": "M",
    "ACT": "T", "ACC": "T", "ACA": "T", "ACG": "T",
    "AAT": "N", "AAC": "N", "AAA": "K", "AAG": "K",
    "AGT": "S", "AGC": "S", "AGA": "R", "AGG": "R",
    "GTT": "V", "GTC": "V", "GTA": "V", "GTG": "V",
    "GCT": "A", "GCC": "A", "GCA": "A", "GCG": "A",
    "GAT": "D", "GAC": "D", "GAA": "E", "GAG": "E",
    "GGT": "G", "GGC": "G", "GGA": "G", "GGG": "G",
}
# fmt: on

STOP_CODONS = frozenset(c for c, aa in CODON_TABLE_11.items() if aa == "*")
SENSE_CODONS = frozenset(c for c, aa in CODON_TABLE_11.items() if aa != "*")
# Bacterial alternative start codons (Table 11).
BACTERIAL_START_CODONS = frozenset({"ATG", "GTG", "TTG", "CTG"})

# Tokens emitted by the CodonLM tokenizer that are not biological codons.
SPECIAL_TOKENS = frozenset({"<PAD>", "<BOS_CDS>", "<EOS_CDS>", "<SEP>", "<UNK>"})


@dataclass
class TranslationReport:
    """Per-sequence translation quality report.

    This dataclass is intentionally kept flat so a learned protein critic can
    later extend it with additional score fields (e.g. ``critic_score``,
    ``stability_pred``) without changing the interface.
    """

    # ── input ────────────────────────────────────────────────────────────
    sequence_index: int = 0
    raw_tokens: list[str] = field(default_factory=list)

    # ── codon-level checks ──────────────────────────────────────────────
    num_codons: int = 0
    invalid_codons: list[str] = field(default_factory=list)
    num_invalid_codons: int = 0

    # ── translation ─────────────────────────────────────────────────────
    protein_sequence: str = ""
    translated_length: int = 0          # number of amino acids (excl. stops)
    starts_with_met: bool = False       # first codon is ATG → M
    has_terminal_stop: bool = False     # last biological codon is a stop
    internal_stop_count: int = 0        # stop codons before the last position
    internal_stop_positions: list[int] = field(default_factory=list)

    # ── summary flag ────────────────────────────────────────────────────
    is_valid_orf: bool = False          # starts M, ends *, no internal *

    def to_dict(self) -> dict:
        """Flat dict suitable for CSV / pandas DataFrame rows."""
        return {
            "sequence_index": self.sequence_index,
            "num_codons": self.num_codons,
            "num_invalid_codons": self.num_invalid_codons,
            "invalid_codons": ";".join(self.invalid_codons) if self.invalid_codons else "",
            "translated_length": self.translated_length,
            "protein_sequence": self.protein_sequence,
            "starts_with_met": self.starts_with_met,
            "has_terminal_stop": self.has_terminal_stop,
            "internal_stop_count": self.internal_stop_count,
            "internal_stop_positions": ";".join(map(str, self.internal_stop_positions))
            if self.internal_stop_positions
            else "",
            "is_valid_orf": self.is_valid_orf,
        }


# ── Public API ──────────────────────────────────────────────────────────────


def strip_special_tokens(tokens: Sequence[str]) -> list[str]:
    """Remove tokenizer special tokens, keeping only biological codons."""
    return [t for t in tokens if t not in SPECIAL_TOKENS]


def translate_codons(
    sequence: str | Sequence[str],
    *,
    sequence_index: int = 0,
    table: dict[str, str] | None = None,
) -> TranslationReport:
    """Translate a generated codon sequence and report biological validity.

    Parameters
    ----------
    sequence
        Either a single string of dash- or space-separated codons
        (e.g. ``"ATG-AAA-GCT-TAA"``), or an iterable of codon strings.
    sequence_index
        Optional index for bookkeeping in batch evaluation.
    table
        Codon→AA mapping.  Defaults to NCBI Table 11.

    Returns
    -------
    TranslationReport
        Flat report with codon-level and translation-level checks.
    """
    if table is None:
        table = CODON_TABLE_11

    # ── normalise input ─────────────────────────────────────────────────
    if isinstance(sequence, str):
        cleaned = sequence.strip()
        if "-" in cleaned or " " in cleaned:
            tokens = cleaned.replace("-", " ").split()
        elif len(cleaned) % 3 == 0 and len(cleaned) > 3 and all(c in "ACGTUacgtu" for c in cleaned):
            tokens = [cleaned[i : i + 3] for i in range(0, len(cleaned), 3)]
        else:
            tokens = cleaned.split()
    else:
        tokens = list(sequence)
    tokens = [t.upper() for t in tokens]

    # Strip special tokens
    codons = strip_special_tokens(tokens)

    report = TranslationReport(
        sequence_index=sequence_index,
        raw_tokens=list(tokens),
        num_codons=len(codons),
    )

    if not codons:
        return report

    # ── translate ───────────────────────────────────────────────────────
    amino_acids: list[str] = []
    invalid: list[str] = []
    internal_stops: list[int] = []

    for i, codon in enumerate(codons):
        aa = table.get(codon)
        if aa is None:
            invalid.append(codon)
            amino_acids.append("X")      # unknown
            continue
        amino_acids.append(aa)
        # An internal stop is any '*' that is *not* the last codon
        if aa == "*" and i < len(codons) - 1:
            internal_stops.append(i)

    # ── populate report ─────────────────────────────────────────────────
    report.invalid_codons = invalid
    report.num_invalid_codons = len(invalid)

    # Protein = all amino acids excluding stops (and the sentinel 'X')
    protein_chars = [aa for aa in amino_acids if aa not in ("*", "X")]
    report.protein_sequence = "".join(protein_chars)
    report.translated_length = len(protein_chars)

    report.starts_with_met = amino_acids[0] == "M" if amino_acids else False
    report.has_terminal_stop = amino_acids[-1] == "*" if amino_acids else False
    report.internal_stop_count = len(internal_stops)
    report.internal_stop_positions = internal_stops

    report.is_valid_orf = (
        report.starts_with_met
        and report.has_terminal_stop
        and report.internal_stop_count == 0
        and report.num_invalid_codons == 0
    )

    return report


def batch_translate(
    sequences: Sequence[str | Sequence[str]],
    *,
    table: dict[str, str] | None = None,
) -> list[TranslationReport]:
    """Translate a batch of generated codon sequences.

    Parameters
    ----------
    sequences
        Iterable of codon sequences (each either a string or list of codons).
    table
        Codon→AA mapping.  Defaults to NCBI Table 11.

    Returns
    -------
    list[TranslationReport]
    """
    return [
        translate_codons(seq, sequence_index=i, table=table)
        for i, seq in enumerate(sequences)
    ]
