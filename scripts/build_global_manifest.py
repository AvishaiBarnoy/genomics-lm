#!/usr/bin/env python3
"""
scripts/build_global_manifest.py — Global Manifest Builder and Group-Aware Splitter

This script implements a global, leakage-resistant data preparation pipeline.
Instead of splitting datasets individually and stacking them (which causes P0 genomic/homology leakage),
it:
  1. Gathers all CDS sequences and metadata across all configured GenBank (GBFF) files.
  2. Resolves stable genome accessions and groups records by genome or genus.
  3. Splits groups globally into train/val/test partitions (Option A).
  4. Tokenizes all sequences using the standard codon tokenizer.
  5. Packs split tokenized IDs into provenance NPZ files plus mmap-ready NPY sidecars.
  6. Emits pipeline_prepare.json and manifest.json for downstream training compatibility.

Usage:
  python -m scripts.build_global_manifest --config configs/tiny_mps.yaml --run-id my_run --run-dir runs/my_run --group-by genus
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import re
from pathlib import Path
from typing import Dict, List
import numpy as np
import yaml
from Bio import SeqIO

from src.codonlm.codon_tokenize import (
    IUPAC_DNA_BASES,
    itos,
    stoi,
    tokenize_cds_fragments,
)
from src.codonlm.extract_cds_from_genbank import reverse_complement, _first_qualifier, _join_qualifier
from src.codonlm.lossless_packing import (
    PACKING_METADATA_FIELDS,
    PackedWindow,
    chunk_record,
    pack_chunks,
    packed_arrays,
    packing_metadata_rows,
)
from src.codonlm.dataset_manifest import (
    SCHEMA_NAME,
    SCHEMA_VERSION,
    artifact_entry,
    file_sha256,
    finalize_manifest,
    validate_dataset_manifest,
)
from src.codonlm.leakage_audit import LeakageAuditError, audit_source_records
from src.codonlm.leakage_audit import quarantine_cross_split_exact_duplicates


ASSEMBLY_ACCESSION_RE = re.compile(
    r"(?<![A-Z0-9])(GC[AF]_\d+(?:\.\d+)?)(?![\d.])", re.IGNORECASE
)

def _load_config(path: Path) -> dict:
    cfg = yaml.safe_load(path.read_text()) or {}
    if not isinstance(cfg, dict):
        raise SystemExit(f"[error] Config at {path} must be a mapping.")
    if "data" in cfg and isinstance(cfg["data"], dict):
        for k, v in cfg["data"].items():
            cfg.setdefault(k, v)
    return cfg


def _parse_extra_dataset(spec: str) -> dict:
    parts = spec.split(",")
    if len(parts) < 2:
        raise SystemExit(
            f"[error] Bad --extra-dataset spec (need name,gbff[,min_len]): {spec}"
        )
    entry: dict = {"name": parts[0], "gbff": parts[1]}
    if len(parts) > 2:
        entry["min_len"] = int(parts[2])
    return entry


def validate_pinned_source(dataset: dict, gbff_path: Path) -> None:
    """Fail before extraction when a configured immutable source has drifted."""
    expected_sha256 = str(dataset.get("sha256", "")).strip().lower()
    expected_bytes = dataset.get("bytes")
    if expected_bytes is not None and gbff_path.stat().st_size != int(expected_bytes):
        raise ValueError(
            f"Source size mismatch for {gbff_path}: expected {expected_bytes}, "
            f"found {gbff_path.stat().st_size}"
        )
    if expected_sha256:
        if not re.fullmatch(r"[0-9a-f]{64}", expected_sha256):
            raise ValueError(f"Invalid configured sha256 for {gbff_path}")
        observed_sha256 = file_sha256(gbff_path)
        if observed_sha256 != expected_sha256:
            raise ValueError(
                f"Source SHA-256 mismatch for {gbff_path}: expected "
                f"{expected_sha256}, found {observed_sha256}"
            )


def resolve_genome_identity(dataset: dict, gbff_path: Path, record) -> tuple[str, str]:
    """Resolve a stable genome identity and describe its provenance."""
    for key in ("genome_id", "assembly_accession", "accession"):
        value = str(dataset.get(key, "")).strip()
        if value:
            return value, f"config.{key}"

    for component in (gbff_path.name, *(parent.name for parent in gbff_path.parents)):
        match = ASSEMBLY_ACCESSION_RE.search(component)
        if match:
            return match.group(1).upper(), "path_accession"

    accessions = record.annotations.get("accessions", [])
    if isinstance(accessions, str):
        accessions = [accessions]
    for value in accessions:
        accession = str(value).strip()
        if accession:
            return accession, "genbank.annotations.accessions"

    record_id = str(getattr(record, "id", "")).strip()
    if record_id and record_id.lower() not in {"unknown", "<unknown id>"}:
        return record_id, "genbank.record_id"

    raise ValueError(
        f"Cannot resolve genome identity for {gbff_path}; set genome_id or assembly_accession in the dataset config"
    )


def _assign_sequence_splits(
    records: List[dict], rng: random.Random, val_frac: float, test_frac: float
) -> None:
    if len(records) < 3:
        raise ValueError("Sequence-level splitting requires at least 3 records")
    indices = list(range(len(records)))
    rng.shuffle(indices)
    n_test = min(max(1, int(len(records) * test_frac)), len(records) - 2)
    n_val = min(max(1, int(len(records) * val_frac)), len(records) - n_test - 1)
    test_idx = set(indices[:n_test])
    val_idx = set(indices[n_test : n_test + n_val])
    for idx, record in enumerate(records):
        record["split"] = "test" if idx in test_idx else "val" if idx in val_idx else "train"


def _assign_group_splits(
    records: List[dict],
    group_key: str,
    rng: random.Random,
    val_frac: float,
    test_frac: float,
) -> dict[str, set[str]]:
    groups = sorted({str(record[group_key]) for record in records})
    if len(groups) < 3:
        raise ValueError(
            f"Scientific splitting requires at least 3 distinct {group_key} groups; found {len(groups)}"
        )
    rng.shuffle(groups)
    n_test = min(max(1, int(len(groups) * test_frac)), len(groups) - 2)
    n_val = min(max(1, int(len(groups) * val_frac)), len(groups) - n_test - 1)
    split_groups = {
        "test": set(groups[:n_test]),
        "val": set(groups[n_test : n_test + n_val]),
        "train": set(groups[n_test + n_val :]),
    }
    for record in records:
        group = str(record[group_key])
        record["split"] = next(
            split for split, assigned in split_groups.items() if group in assigned
        )
    return split_groups

def extract_genus(rec) -> str:
    """Extract Genus from BioPython SeqRecord annotation taxonomy or organism."""
    organism = rec.annotations.get("organism", "")
    if organism:
        parts = organism.split()
        if parts:
            return parts[0]
    taxonomy = rec.annotations.get("taxonomy", [])
    if taxonomy:
        return taxonomy[0]
    return "Unknown"

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="YAML config file path")
    ap.add_argument("--run-id", required=True, help="Run identifier")
    ap.add_argument("--run-dir", required=True, help="Run output directory")
    ap.add_argument("--group-by", choices=["genome", "genus", "sequence"], default=None,
                    help="Split grouping criterion to avoid leakage.")
    ap.add_argument(
        "--allow-sequence-split",
        action="store_true",
        help="Explicitly allow a non-scientific sequence-level split when grouped splitting is impossible.",
    )
    ap.add_argument(
        "--extra-dataset", action="append", default=[], help="NAME,GBFF[,MIN_LEN]"
    )
    ap.add_argument(
        "--output-dir",
        help="Prepared dataset directory (default: data/processed/global/<run-id>).",
    )
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument(
        "--skip-homology-audit",
        action="store_true",
        help="NON-SCIENTIFIC: skip the required MMseqs2 protein-homology audit.",
    )
    ap.add_argument(
        "--allow-cross-split-exact-duplicates",
        action="store_true",
        help="NON-SCIENTIFIC: report but do not block cross-split exact CDS duplicates.",
    )
    ap.add_argument(
        "--mmseqs-executable",
        default="mmseqs",
        help="MMseqs2 executable used for clustering and nearest-neighbor searches.",
    )
    ap.add_argument("--audit-threads", type=int, default=1)
    ap.add_argument(
        "--nucleotide-executable",
        default="minimap2",
        help="Minimap2 executable used for low-memory nucleotide nearest-neighbor mapping.",
    )
    ap.add_argument("--force", action="store_true", help="Force rebuild")
    args = ap.parse_args()

    cfg = _load_config(Path(args.config))
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    block_size = int(cfg.get("block_size", 256))
    val_frac = float(cfg.get("val_frac", 0.1))
    test_frac = float(cfg.get("test_frac", 0.1))
    if not (0.0 < val_frac < 1.0 and 0.0 < test_frac < 1.0):
        raise SystemExit("[error] val_frac and test_frac must both be between 0 and 1")
    if val_frac + test_frac >= 1.0:
        raise SystemExit("[error] val_frac + test_frac must be less than 1")
    pack_mode = cfg.get("pack_mode", "multi")
    windows_per_seq = int(float(cfg.get("windows_per_seq", 1)))
    min_len = int(cfg.get("min_len", 90))
    min_fragment_codons = int(cfg.get("min_fragment_codons", 10))
    min_protein_identity = float(cfg.get("max_cross_split_protein_identity", 0.3))
    min_homology_coverage = float(cfg.get("min_homology_coverage", 0.8))
    exact_duplicate_policy = str(cfg.get("exact_duplicate_policy", "block"))
    protein_homology_policy = str(cfg.get("protein_homology_policy", "block"))
    nearest_query_batch_size = int(cfg.get("mmseqs_query_batch_size", 4096))
    split_memory_limit = str(cfg.get("mmseqs_split_memory_limit", "0"))
    nucleotide_preset = str(cfg.get("nucleotide_alignment_preset", "asm20"))
    if min_fragment_codons < 1:
        raise SystemExit("[error] min_fragment_codons must be at least 1")
    if not (0.0 < min_protein_identity <= 1.0):
        raise SystemExit("[error] max_cross_split_protein_identity must be in (0, 1]")
    if not (0.0 < min_homology_coverage <= 1.0):
        raise SystemExit("[error] min_homology_coverage must be in (0, 1]")
    if exact_duplicate_policy not in {"block", "quarantine"}:
        raise SystemExit("[error] exact_duplicate_policy must be block or quarantine")
    if protein_homology_policy not in {"block", "report"}:
        raise SystemExit("[error] protein_homology_policy must be block or report")
    if nearest_query_batch_size < 1:
        raise SystemExit("[error] mmseqs_query_batch_size must be at least 1")
    if exact_duplicate_policy == "quarantine" and args.allow_cross_split_exact_duplicates:
        raise SystemExit(
            "[error] exact_duplicate_policy=quarantine cannot be combined with "
            "--allow-cross-split-exact-duplicates"
        )
    requested_group_by = args.group_by or str(cfg.get("split_group_by", "genome"))
    if requested_group_by not in {"genome", "genus", "sequence"}:
        raise SystemExit(
            f"[error] split_group_by must be genome, genus, or sequence; got {requested_group_by!r}"
        )
    if requested_group_by == "sequence" and not args.allow_sequence_split:
        raise SystemExit(
            "[error] Sequence-level splitting is non-scientific and requires --allow-sequence-split"
        )

    rng = random.Random(args.seed)
    datasets = list(cfg.get("datasets", []))
    datasets.extend(_parse_extra_dataset(spec) for spec in args.extra_dataset)
    if not datasets:
        raise SystemExit("[error] No datasets found in config.")

    # 1. Extraction phase
    print(f"[global-prep] Extracting sequences from {len(datasets)} datasets...")
    all_records: List[dict] = []
    genome_sources: Dict[str, dict] = {}
    announced_genomes: set[str] = set()
    
    for ds in datasets:
        name = ds["name"]
        gbff_path = Path(ds["gbff"])
        if not gbff_path.exists():
            raise FileNotFoundError(f"GBFF not found: {gbff_path}")
        try:
            validate_pinned_source(ds, gbff_path)
        except ValueError as exc:
            raise SystemExit(f"[error] {exc}") from exc
        
        dataset_min_len = int(ds.get("min_len", min_len))
        for rec in SeqIO.parse(gbff_path, "genbank"):
            try:
                genome_id, identity_source = resolve_genome_identity(ds, gbff_path, rec)
            except ValueError as exc:
                raise SystemExit(f"[error] {exc}") from exc
            resolved_path = str(gbff_path.resolve())
            previous = genome_sources.get(genome_id)
            if previous is not None and previous["gbff"] != resolved_path:
                raise SystemExit(
                    f"[error] Genome identity collision for {genome_id!r}: "
                    f"{previous['gbff']} and {resolved_path}. Set distinct genome_id values explicitly."
                )
            genome_sources.setdefault(
                genome_id,
                {
                    "gbff": resolved_path,
                    "identity_source": identity_source,
                    "sha256": file_sha256(gbff_path),
                    "bytes": gbff_path.stat().st_size,
                },
            )
            if genome_id not in announced_genomes:
                print(
                    f"  Processing {name} ({gbff_path.name}) with "
                    f"genome_id={genome_id} ({identity_source})..."
                )
                announced_genomes.add(genome_id)
            seq = str(rec.seq).upper()
            genus = extract_genus(rec)
            
            for feature_index, feat in enumerate(rec.features):
                if feat.type != "CDS":
                    continue
                s, e = int(feat.location.start), int(feat.location.end)
                strand = int(feat.location.strand or 1)
                cds_seq = seq[s:e]
                if strand == -1:
                    cds_seq = reverse_complement(cds_seq)
                
                if len(cds_seq) >= dataset_min_len and set(cds_seq) <= IUPAC_DNA_BASES:
                    source_id = (
                        f"{genome_id}:{rec.id}:cds:{s}-{e}:{strand}:{feature_index}"
                    )
                    all_records.append({
                        "sequence": cds_seq,
                        "source_id": source_id,
                        "genome": genome_id,
                        "genome_identity_source": identity_source,
                        "genus": genus,
                        "dataset": name,
                        "record_id": str(rec.id),
                        "protein_id": _first_qualifier(feat, "protein_id"),
                        "locus_tag": _first_qualifier(feat, "locus_tag"),
                        "gene": _first_qualifier(feat, "gene"),
                        "product": _first_qualifier(feat, "product"),
                        "translation": _first_qualifier(feat, "translation"),
                        "db_xref": _join_qualifier(feat, "db_xref"),
                        "start": s,
                        "end": e,
                        "strand": strand,
                    })

    extracted_total_seqs = len(all_records)
    total_seqs = extracted_total_seqs
    print(f"[global-prep] Extracted {total_seqs} total CDS records.")
    if not all_records:
        raise SystemExit("[error] No eligible CDS records were extracted.")

    # 2. Splitting phase
    effective_group_by = requested_group_by
    split_groups: dict[str, set[str]] | None = None
    if requested_group_by == "sequence":
        print("[global-prep] Performing sequence-level random split...")
        _assign_sequence_splits(all_records, rng, val_frac, test_frac)
    else:
        try:
            split_groups = _assign_group_splits(
                all_records, requested_group_by, rng, val_frac, test_frac
            )
        except ValueError as exc:
            if not args.allow_sequence_split:
                raise SystemExit(f"[error] {exc}") from exc
            print(
                f"[global-prep] NON-SCIENTIFIC: falling back from {requested_group_by} "
                "to an explicitly allowed sequence-level split."
            )
            effective_group_by = "sequence"
            _assign_sequence_splits(all_records, rng, val_frac, test_frac)
        else:
            for split, groups in split_groups.items():
                print(f"  {split.title()} groups: {groups}")

    duplicate_quarantine = None
    if exact_duplicate_policy == "quarantine":
        all_records, duplicate_quarantine = quarantine_cross_split_exact_duplicates(
            all_records
        )
        print(
            "[global-prep] Quarantined "
            f"{duplicate_quarantine['removed_record_count']} cross-split exact "
            "duplicate records."
        )
        total_seqs = len(all_records)

    # Count split stats after any preventive quarantine.
    counts = {"train": 0, "val": 0, "test": 0}
    for rec in all_records:
        counts[rec["split"]] += 1
    print(f"[global-prep] Split counts: train={counts['train']}, val={counts['val']}, test={counts['test']}")

    # 3. Save combined metadata & DNA
    out_dir = (
        Path(args.output_dir)
        if args.output_dir
        else Path("data/processed/global") / args.run_id
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    audit_path = out_dir / "leakage_audit.json"
    if args.skip_homology_audit:
        print(
            "[global-prep] NON-SCIENTIFIC: MMseqs2 protein-homology audit was explicitly skipped."
        )
    if args.allow_cross_split_exact_duplicates:
        print(
            "[global-prep] NON-SCIENTIFIC: cross-split exact CDS duplicates are allowed."
        )
    try:
        leakage_report = audit_source_records(
            all_records,
            audit_path,
            min_protein_identity=min_protein_identity,
            min_coverage=min_homology_coverage,
            threads=args.audit_threads,
            executable=args.mmseqs_executable,
            nucleotide_executable=args.nucleotide_executable,
            nucleotide_preset=nucleotide_preset,
            skip_homology=args.skip_homology_audit,
            allow_exact_duplicates=args.allow_cross_split_exact_duplicates,
            protein_homology_policy=protein_homology_policy,
            nearest_query_batch_size=nearest_query_batch_size,
            split_memory_limit=split_memory_limit,
        )
    except LeakageAuditError as exc:
        raise SystemExit(f"[error] {exc}; see {audit_path}") from exc
    print(f"[global-prep] Leakage audit passed: {audit_path}")

    quarantine_path = None
    if duplicate_quarantine is not None:
        quarantine_path = out_dir / "exact_duplicate_quarantine.json"
        quarantine_path.write_text(json.dumps(duplicate_quarantine, indent=2) + "\n")
    
    meta_path = out_dir / "cds_meta.tsv"
    dna_path = out_dir / "cds_dna.txt"
    
    with open(meta_path, "w", newline="") as fm, open(dna_path, "w") as fd:
        metadata_fields = [
            key
            for key in all_records[0]
            if key not in {"sequence", "line_idx", "split"}
        ]
        writer = csv.DictWriter(
            fm,
            fieldnames=["line_idx", "split", *metadata_fields],
            delimiter="\t",
        )
        writer.writeheader()
        
        for idx, rec in enumerate(all_records):
            fd.write(rec["sequence"] + "\n")
            row = {k: v for k, v in rec.items() if k != "sequence"}
            row["line_idx"] = idx
            writer.writerow(row)
            
    print(f"[global-prep] Wrote global metadata to {meta_path} and DNA to {dna_path}")

    # 4. Tokenization phase
    print("[global-prep] Tokenizing sequences...")
    fragment_records: List[dict] = []
    token_ids_list: List[List[int]] = []
    tokenization_counts = {
        "ambiguous_codons": 0,
        "source_records_with_ambiguity": 0,
        "retained_fragments": 0,
        "discarded_fragments": 0,
        "partial_trailing_bases": 0,
    }
    per_split_counts = {
        split: {key: 0 for key in tokenization_counts}
        for split in ("train", "val", "test")
    }
    for source_line_idx, rec in enumerate(all_records):
        result = tokenize_cds_fragments(
            rec["sequence"],
            source_id=rec["source_id"],
            min_fragment_codons=min_fragment_codons,
        )
        split_counts = per_split_counts[rec["split"]]
        tokenization_counts["ambiguous_codons"] += result.ambiguous_codons
        split_counts["ambiguous_codons"] += result.ambiguous_codons
        tokenization_counts["discarded_fragments"] += result.discarded_fragments
        split_counts["discarded_fragments"] += result.discarded_fragments
        tokenization_counts["partial_trailing_bases"] += result.partial_trailing_bases
        split_counts["partial_trailing_bases"] += result.partial_trailing_bases
        if result.source_had_ambiguity:
            tokenization_counts["source_records_with_ambiguity"] += 1
            split_counts["source_records_with_ambiguity"] += 1
        for fragment in result.fragments:
            token_ids_list.append(fragment.ids)
            tokenization_counts["retained_fragments"] += 1
            split_counts["retained_fragments"] += 1
            fragment_records.append(
                {
                    "fragment_line_idx": len(fragment_records),
                    "source_line_idx": source_line_idx,
                    "source_id": rec["source_id"],
                    "split": rec["split"],
                    "fragment_index": fragment.fragment_index,
                    "codon_start": fragment.codon_start,
                    "codon_end": fragment.codon_end,
                    "base_start": fragment.base_start,
                    "base_end": fragment.base_end,
                    "codon_count": fragment.codon_end - fragment.codon_start,
                }
            )

    # Save global codon ids
    ids_path = out_dir / "codon_ids.txt"
    with open(ids_path, "w") as f:
        for tids in token_ids_list:
            f.write(" ".join(str(i) for i in tids) + "\n")

    fragments_path = out_dir / "cds_fragments.tsv"
    fragment_fields = [
        "fragment_line_idx",
        "source_line_idx",
        "source_id",
        "split",
        "fragment_index",
        "codon_start",
        "codon_end",
        "base_start",
        "base_end",
        "codon_count",
    ]
    with open(fragments_path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fragment_fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(fragment_records)

    # 5. Lossless chunking and packing phase
    chunks_by_split = {"train": [], "val": [], "test": []}
    for fragment, token_ids in zip(fragment_records, token_ids_list):
        record = {
            "tokens": token_ids,
            "source_id": fragment["source_id"],
            "source_line_idx": fragment["source_line_idx"],
            "fragment_line_idx": fragment["fragment_line_idx"],
            "fragment_index": fragment["fragment_index"],
            "fragment_codon_start": fragment["codon_start"],
            "fragment_codon_end": fragment["codon_end"],
            "split": fragment["split"],
        }
        chunks_by_split[fragment["split"]].extend(
            chunk_record(record, block_size=block_size)
        )

    if windows_per_seq != 1:
        print(
            "[global-prep] Ignoring legacy windows_per_seq="
            f"{windows_per_seq}; lossless packing emits every source transition once."
        )

    def write_packing_metadata(
        split: str, windows: List[PackedWindow]
    ) -> Path:
        path = out_dir / f"{split}_packing.tsv"
        with open(path, "w", newline="") as handle:
            writer = csv.DictWriter(
                handle, fieldnames=PACKING_METADATA_FIELDS, delimiter="\t"
            )
            writer.writeheader()
            writer.writerows(packing_metadata_rows(split, windows))
        return path

    out_paths = {}
    mmap_sidecars: dict[str, dict[str, Path]] = {}
    empty_windows: dict[str, int] = {}
    packing_metadata_paths: dict[str, str] = {}
    packed_window_counts: dict[str, int] = {}
    packed_chunk_counts: dict[str, int] = {}
    for name in ("train", "val", "test"):
        windows = pack_chunks(
            chunks_by_split[name],
            block_size=block_size,
            mode=pack_mode,
            sep_id=stoi["<SEP>"],
        )
        arrays = packed_arrays(windows, block_size=block_size, mode=pack_mode)
        X = arrays["X"]
        out_npz = out_dir / f"{name}_bs{block_size}.npz"
        np.savez_compressed(out_npz, **arrays)
        split_sidecars: dict[str, Path] = {}
        for key in ("X", "Y", "lengths"):
            if key not in arrays:
                continue
            sidecar = out_dir / f"{out_npz.stem}_{key}.npy"
            values = arrays[key]
            if key in {"X", "Y"}:
                if values.size and (values.min() < 0 or values.max() > 255):
                    raise ValueError(f"{name} {key} token IDs do not fit in uint8")
                values = values.astype(np.uint8, copy=False)
            np.save(sidecar, values, allow_pickle=False)
            split_sidecars[key] = sidecar
        mmap_sidecars[name] = split_sidecars
        if pack_mode == "dynamic":
            empty_windows[name] = int(np.count_nonzero(arrays["lengths"] == 0))
        else:
            empty_windows[name] = int(
                np.count_nonzero((arrays["Y"] != 0).sum(axis=1) == 0)
            )
        out_paths[name] = out_npz
        packing_metadata_paths[name] = write_packing_metadata(name, windows).name
        packed_window_counts[name] = len(windows)
        packed_chunk_counts[name] = len(chunks_by_split[name])
        print(f"[global-prep] Packed split {name} to {out_npz} with shape {X.shape}")

    # Write itos/vocab files
    itos_path = out_dir / "itos.txt"
    vocab_path = out_dir / "vocab_codon.txt"
    
    with open(itos_path, "w") as f:
        for i in sorted(itos.keys()):
            f.write(itos[i] + "\n")
            
    with open(vocab_path, "w") as f:
        for k, v in stoi.items():
            f.write(f"{k} {v}\n")

    # 6. Save combined manifest & pipeline_prepare.json
    achieved_fractions = {
        split: (counts[split] / total_seqs if total_seqs else 0.0)
        for split in ("train", "val", "test")
    }
    group_counts = None
    groups_by_split = None
    if split_groups is not None and effective_group_by != "sequence":
        groups_by_split = {
            split: sorted(groups) for split, groups in split_groups.items()
        }
        group_counts = {
            split: len(groups) for split, groups in split_groups.items()
        }

    scientific_valid = (
        effective_group_by != "sequence"
        and not args.skip_homology_audit
        and not args.allow_cross_split_exact_duplicates
    )
    artifacts = {
        "train_tokens": artifact_entry(out_paths["train"], out_dir, "train_tokens"),
        "val_tokens": artifact_entry(out_paths["val"], out_dir, "val_tokens"),
        "test_tokens": artifact_entry(out_paths["test"], out_dir, "test_tokens"),
        "vocabulary": artifact_entry(itos_path, out_dir, "vocabulary"),
        "vocabulary_map": artifact_entry(vocab_path, out_dir, "vocabulary_map"),
        "source_metadata": artifact_entry(meta_path, out_dir, "source_metadata"),
        "source_dna": artifact_entry(dna_path, out_dir, "source_dna"),
        "token_ids": artifact_entry(ids_path, out_dir, "token_ids"),
        "fragment_metadata": artifact_entry(fragments_path, out_dir, "fragment_metadata"),
        "leakage_audit": artifact_entry(audit_path, out_dir, "leakage_audit"),
    }
    for split, sidecars in mmap_sidecars.items():
        for key, path in sidecars.items():
            artifacts[f"{split}_{key.lower()}_npy"] = artifact_entry(
                path, out_dir, f"{split}_{key.lower()}_npy"
            )
    protein_homology = leakage_report.get("protein_homology") or {}
    audit_evidence = {
        "protein_clusters": protein_homology.get("cluster_artifact"),
        "nearest_nucleotide": (
            protein_homology.get("nearest_neighbors", {})
            .get("nucleotide", {})
            .get("artifact")
        ),
        "nearest_protein": (
            protein_homology.get("nearest_neighbors", {})
            .get("protein", {})
            .get("artifact")
        ),
    }
    for name, path in audit_evidence.items():
        if path:
            artifacts[name] = artifact_entry(Path(path), out_dir, name)
    if quarantine_path is not None:
        artifacts["exact_duplicate_quarantine"] = artifact_entry(
            quarantine_path, out_dir, "exact_duplicate_quarantine"
        )
    for split in ("train", "val", "test"):
        artifacts[f"{split}_packing_metadata"] = artifact_entry(
            out_dir / packing_metadata_paths[split], out_dir, f"{split}_packing_metadata"
        )

    manifest = {
        "schema": {"name": SCHEMA_NAME, "version": SCHEMA_VERSION},
        "dataset": {
            "id": "pending",
            "scientific_valid": scientific_valid,
            "source_record_count": total_seqs,
            "extracted_source_record_count": extracted_total_seqs,
        },
        "train": str(out_paths["train"]),
        "val": str(out_paths["val"]),
        "test": str(out_paths["test"]),
        "datasets": datasets,
        "seed": args.seed,
        "split_policy": {
            "requested_group_by": requested_group_by,
            "effective_group_by": effective_group_by,
            "allow_sequence_split": bool(args.allow_sequence_split),
            "scientific_valid": (
                effective_group_by != "sequence"
                and not args.skip_homology_audit
                and not args.allow_cross_split_exact_duplicates
            ),
            "requested_fractions": {"val": val_frac, "test": test_frac},
            "achieved_record_fractions": achieved_fractions,
            "record_counts": counts,
            "group_counts": group_counts,
            "groups_by_split": groups_by_split,
        },
        "genome_sources": genome_sources,
        "sources": {
            genome: {
                "path": source["gbff"],
                "sha256": source["sha256"],
                "bytes": source["bytes"],
                "identity_source": source["identity_source"],
            }
            for genome, source in sorted(genome_sources.items())
        },
        "vocabulary": {
            "schema_version": 1,
            "itos_path": str(itos_path),
            "sha256": hashlib.sha256(itos_path.read_bytes()).hexdigest(),
            "size": len(itos),
            "token_ids_contiguous": sorted(itos) == list(range(len(itos))),
            "special_tokens": {
                token: stoi[token]
                for token in ("<PAD>", "<BOS_CDS>", "<EOS_CDS>", "<SEP>")
            },
        },
        "leakage_audit": {
            "artifact": "leakage_audit",
            "status": leakage_report["status"],
            "homology_audit_skipped": args.skip_homology_audit,
            "exact_duplicate_override": args.allow_cross_split_exact_duplicates,
            "exact_duplicate_policy": exact_duplicate_policy,
            "protein_homology_policy": protein_homology_policy,
            "thresholds": leakage_report["thresholds"],
        },
        "tokenization": {
            "fragment_metadata_artifact": "fragment_metadata",
            "ambiguous_codon_policy": {
                "name": "split",
                "min_fragment_codons": min_fragment_codons,
                **tokenization_counts,
                "per_split": per_split_counts,
            },
        },
        "packing": {
            "schema_version": 1,
            "mode": pack_mode,
            "block_size": block_size,
            "token_capacity": block_size + 1,
            "chunk_overlap_tokens": 1,
            "transition_policy": "exactly_once",
            "legacy_windows_per_seq_ignored": windows_per_seq,
            "seed": args.seed,
            "metadata": {
                split: f"{split}_packing_metadata" for split in ("train", "val", "test")
            },
            "window_counts": packed_window_counts,
            "chunk_counts": packed_chunk_counts,
        },
        "artifacts": artifacts,
        "reproducibility": {
            "split_seed": args.seed,
            "packing_seed": args.seed,
            "deterministic_packing": True,
        },
    }
    manifest = finalize_manifest(manifest)
    manifest_json_path = out_dir / "manifest.json"
    validate_dataset_manifest(manifest, manifest_json_path, verify_artifacts=True)
    manifest_json_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    run_manifest = json.loads(json.dumps(manifest))
    for entry in run_manifest["artifacts"].values():
        artifact_path = Path(entry["path"])
        if not artifact_path.is_absolute():
            entry["path"] = str((out_dir / artifact_path).resolve())
    run_manifest_path = run_dir / "combined_manifest.json"
    validate_dataset_manifest(run_manifest, run_manifest_path, verify_artifacts=True)
    run_manifest_path.write_text(json.dumps(run_manifest, indent=2, sort_keys=True))
    
    pipeline_prepare_json = {
        "train_npz": str(out_paths["train"]),
        "val_npz": str(out_paths["val"]),
        "test_npz": str(out_paths["test"]),
        "primary_dna": str(dna_path),
        "combined_manifest": str(manifest_json_path),
        "itos_path": str(itos_path),
        "dataset_id": manifest["dataset"]["id"],
        "dataset_schema": manifest["schema"],
    }
    
    result_path = run_dir / "pipeline_prepare.json"
    result_path.write_text(json.dumps(pipeline_prepare_json, indent=2))
    
    # Save integrity check data
    integrity = {
        "train_npz": str(out_paths["train"]),
        "val_npz": str(out_paths["val"]),
        "test_npz": str(out_paths["test"]),
        "empty_windows": empty_windows,
    }
    (run_dir / "integrity.json").write_text(json.dumps(integrity, indent=2))
    
    print(f"[global-prep] Completed global manifest data preparation. Output summary in {result_path}")

if __name__ == "__main__":
    main()
