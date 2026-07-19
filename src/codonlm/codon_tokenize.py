#!/usr/bin/env python3
"""
Convert CDS DNA (one sequence per line) → sequences of codon ids.

Vocabulary (fixed order):
  0: <PAD>
  1: <BOS_CDS>
  2: <EOS_CDS>
  3: <SEP>
  4..67: the 64 codons (AAA..TTT in lexical order)

Encoding a single CDS yields:
  ["<BOS_CDS>", CODON1, ..., STOPCODON, "<EOS_CDS>"]

When packing multiple CDS into one sequence, separate by <SEP>. We include
<EOS_CDS> before every <SEP> so models can learn explicit termination.

Outputs:
- data/processed/codon_ids.txt  (space-separated ints, one CDS per line)
- data/processed/vocab_codon.txt (id -> token)
- data/processed/itos_codon.txt  (token per line)
"""

from pathlib import Path
import argparse
import csv
from dataclasses import dataclass

CODONS = [a + b + c for a in "ACGT" for b in "ACGT" for c in "ACGT"]
IUPAC_DNA_BASES = frozenset("ACGTRYSWKMBDHVN")
STOP_CODONS = {"TAA", "TAG", "TGA"}
SPECIALS = ["<PAD>", "<BOS_CDS>", "<EOS_CDS>", "<SEP>"]
VOCAB = SPECIALS + CODONS
stoi = {tok: i for i, tok in enumerate(VOCAB)}
# itos must map to canonical tokens; build from VOCAB only
itos = {i: tok for i, tok in enumerate(VOCAB)}
# Backward-compat aliases for legacy tests/configs (affect stoi only)
ALIASES = {
    "<bos>": "<BOS_CDS>",
    "<eog>": "<EOS_CDS>",
    "<eos>": "<EOS_CDS>",
}
for alias, canonical in ALIASES.items():
    stoi[alias] = stoi[canonical]


class AmbiguousCodonError(ValueError):
    """Raised when a single-sequence tokenization would erase an ambiguous codon."""


@dataclass(frozen=True)
class TokenizedCDSFragment:
    """A retained contiguous run of unambiguous codons in oriented CDS coordinates."""

    ids: list[int]
    source_id: str | None
    fragment_index: int
    codon_start: int
    codon_end: int
    base_start: int
    base_end: int


@dataclass(frozen=True)
class CDSTokenizationResult:
    """Fragments and audit counts produced by ambiguity-aware CDS tokenization."""

    fragments: list[TokenizedCDSFragment]
    ambiguous_codons: int
    discarded_fragments: int
    partial_trailing_bases: int

    @property
    def source_had_ambiguity(self) -> bool:
        return self.ambiguous_codons > 0


def _normalize_dna(dna: str) -> str:
    return dna.strip().upper().replace("U", "T")


def _terminated_ids(codon_ids: list[int], termination: str) -> list[int]:
    ids = [stoi["<BOS_CDS>"], *codon_ids]
    if termination == "eos":
        ids.append(stoi["<EOS_CDS>"])
    elif termination == "sep":
        ids.append(stoi["<SEP>"])
    elif termination != "none":
        raise ValueError(f"Unsupported termination policy: {termination!r}")
    return ids


def tokenize_cds_fragments(
    dna: str,
    *,
    source_id: str | None = None,
    min_fragment_codons: int = 1,
    termination: str = "eos",
) -> CDSTokenizationResult:
    """Split a CDS at ambiguous codons without creating cross-gap adjacency.

    Coordinates are zero-based, half-open offsets in the oriented CDS string. A
    trailing partial codon is excluded and reported through ``partial_trailing_bases``.
    Empty runs created by leading, trailing, or consecutive ambiguity are ignored.
    """
    if min_fragment_codons < 1:
        raise ValueError("min_fragment_codons must be at least 1")

    normalized = _normalize_dna(dna)
    complete_length = (len(normalized) // 3) * 3
    partial_trailing_bases = len(normalized) - complete_length
    fragments: list[TokenizedCDSFragment] = []
    ambiguous_codons = 0
    discarded_fragments = 0
    fragment_index = 0
    run_start: int | None = None
    run_ids: list[int] = []

    def flush(end_codon: int) -> None:
        nonlocal discarded_fragments, fragment_index, run_start, run_ids
        if run_start is None:
            return
        if len(run_ids) >= min_fragment_codons:
            fragments.append(
                TokenizedCDSFragment(
                    ids=_terminated_ids(run_ids, termination),
                    source_id=source_id,
                    fragment_index=fragment_index,
                    codon_start=run_start,
                    codon_end=end_codon,
                    base_start=run_start * 3,
                    base_end=end_codon * 3,
                )
            )
        else:
            discarded_fragments += 1
        fragment_index += 1
        run_start = None
        run_ids = []

    for codon_index in range(complete_length // 3):
        codon = normalized[codon_index * 3 : codon_index * 3 + 3]
        token_id = stoi.get(codon)
        if token_id is None:
            ambiguous_codons += 1
            flush(codon_index)
            continue
        if run_start is None:
            run_start = codon_index
        run_ids.append(token_id)
    flush(complete_length // 3)

    return CDSTokenizationResult(
        fragments=fragments,
        ambiguous_codons=ambiguous_codons,
        discarded_fragments=discarded_fragments,
        partial_trailing_bases=partial_trailing_bases,
    )

def to_ids(dna: str, termination: str = "eos") -> list[int]:
    """Converts a DNA sequence string into a list of token IDs, wrapping it in BOS and EOS/SEP tokens."""
    dna = _normalize_dna(dna)
    if len(dna) < 3:
        return []
    # trim to a multiple of 3, left-aligned (GenBank CDS already in-frame)
    L = (len(dna) // 3) * 3
    trailing = dna[L:]
    if trailing and not set(trailing) <= set("ACGT"):
        raise AmbiguousCodonError(
            f"ambiguous partial codon {trailing!r} at codon index {L // 3}; "
            "use tokenize_cds_fragments() for dataset preparation"
        )
    codon_ids = []
    for i in range(0, L, 3):
        codon = dna[i : i + 3]
        idx = stoi.get(codon)
        if idx is None:
            raise AmbiguousCodonError(
                f"ambiguous codon {codon!r} at codon index {i // 3}; "
                "use tokenize_cds_fragments() for dataset preparation"
            )
        codon_ids.append(idx)
    # Ensure we terminate correctly even if sequence lacked a canonical stop
    if not codon_ids:
        return []
    return _terminated_ids(codon_ids, termination)


def main():
    """Reads CDS DNA lines, tokenizes them to codon IDs, and writes tokenized text and vocabulary mapping files."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--inp", default="data/processed/cds_dna.txt")
    ap.add_argument("--out_ids", default="data/processed/codon_ids.txt")
    ap.add_argument("--out_vocab", default="data/processed/vocab_codon.txt")
    ap.add_argument("--out_itos", default="data/processed/itos_codon.txt")
    ap.add_argument(
        "--out_fragments",
        default=None,
        help="Fragment provenance TSV (default: <out_ids>.fragments.tsv).",
    )
    ap.add_argument("--min_fragment_codons", type=int, default=10)
    ap.add_argument("--termination", choices=["eos", "sep", "none"], default="eos",
                    help="Termination token: 'eos' (id 2, default), 'sep' (id 3, legacy default), or 'none'")
    args = ap.parse_args()

    ids_path = Path(args.out_ids)
    ids_path.parent.mkdir(parents=True, exist_ok=True)
    fragments_path = Path(args.out_fragments or f"{args.out_ids}.fragments.tsv")
    fragments_path.parent.mkdir(parents=True, exist_ok=True)
    stats = {
        "source_records": 0,
        "source_records_with_ambiguity": 0,
        "ambiguous_codons": 0,
        "retained_fragments": 0,
        "discarded_fragments": 0,
        "partial_trailing_bases": 0,
    }
    with (
        open(args.inp) as fin,
        open(args.out_ids, "w") as fout,
        open(fragments_path, "w", newline="") as fragment_handle,
    ):
        fields = [
            "fragment_line_idx",
            "source_line_idx",
            "source_id",
            "fragment_index",
            "codon_start",
            "codon_end",
            "base_start",
            "base_end",
        ]
        writer = csv.DictWriter(fragment_handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        for source_line_idx, line in enumerate(fin):
            source_id = f"line:{source_line_idx}"
            result = tokenize_cds_fragments(
                line,
                source_id=source_id,
                min_fragment_codons=args.min_fragment_codons,
                termination=args.termination,
            )
            stats["source_records"] += 1
            stats["source_records_with_ambiguity"] += int(
                result.source_had_ambiguity
            )
            stats["ambiguous_codons"] += result.ambiguous_codons
            stats["discarded_fragments"] += result.discarded_fragments
            stats["partial_trailing_bases"] += result.partial_trailing_bases
            for fragment in result.fragments:
                fragment_line_idx = stats["retained_fragments"]
                fout.write(" ".join(map(str, fragment.ids)) + "\n")
                writer.writerow(
                    {
                        "fragment_line_idx": fragment_line_idx,
                        "source_line_idx": source_line_idx,
                        "source_id": source_id,
                        "fragment_index": fragment.fragment_index,
                        "codon_start": fragment.codon_start,
                        "codon_end": fragment.codon_end,
                        "base_start": fragment.base_start,
                        "base_end": fragment.base_end,
                    }
                )
                stats["retained_fragments"] += 1
    with open(args.out_vocab, "w") as f:
        for i, tok in enumerate(VOCAB):
            f.write(f"{i}\t{tok}\n")
    with open(args.out_itos, "w") as f:
        for tok in VOCAB:
            f.write(f"{tok}\n")
    print(
        f"[tokenize] wrote {stats['retained_fragments']} fragments → {ids_path} "
        f"| provenance {fragments_path} | stats {stats} | vocab size {len(VOCAB)} "
        f"| itos {args.out_itos}"
    )

__all__ = [
    "ALIASES",
    "AmbiguousCodonError",
    "CDSTokenizationResult",
    "CODONS",
    "IUPAC_DNA_BASES",
    "SPECIALS",
    "STOP_CODONS",
    "TokenizedCDSFragment",
    "VOCAB",
    "itos",
    "stoi",
    "to_ids",
    "tokenize_cds_fragments",
]

if __name__ == "__main__":
    main()
