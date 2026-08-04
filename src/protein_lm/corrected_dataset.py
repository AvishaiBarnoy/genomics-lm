from __future__ import annotations

import hashlib
import random
from collections import Counter, defaultdict


SPLIT_FRACTIONS = {"train": 0.8, "validation": 0.1, "test": 0.1}


def sequence_sha256(sequence: str) -> str:
    return hashlib.sha256(sequence.encode("ascii")).hexdigest()


def normalize_protein(sequence: str) -> str:
    normalized = "".join(str(sequence).upper().split()).rstrip("*")
    if not normalized or any(residue not in "ACDEFGHIKLMNPQRSTVWY" for residue in normalized):
        raise ValueError("protein sequence contains empty, ambiguous, or non-amino-acid symbols")
    return normalized


def assign_clusters(
    records: list[dict],
    *,
    seed: int,
    fractions: dict[str, float] | None = None,
    required_task_keys: tuple[str, ...] = (),
) -> dict[str, str]:
    fractions = fractions or SPLIT_FRACTIONS
    if set(fractions) != {"train", "validation", "test"}:
        raise ValueError("fractions must define train, validation, and test")
    if abs(sum(fractions.values()) - 1.0) > 1e-9:
        raise ValueError("split fractions must sum to one")

    cluster_sizes = Counter(record["protein_cluster"] for record in records)
    rng = random.Random(seed)
    tie_breakers = {cluster: rng.random() for cluster in cluster_sizes}
    # Place large homology groups first so one large family cannot accidentally
    # consume an entire validation or test split late in the assignment.
    clusters = sorted(
        cluster_sizes,
        key=lambda cluster: (-cluster_sizes[cluster], tie_breakers[cluster], cluster),
    )
    targets = {split: len(records) * fraction for split, fraction in fractions.items()}
    assigned_counts = {split: 0 for split in fractions}
    assignments = {}
    for key in required_task_keys:
        task_cluster_set = {
            record["protein_cluster"]
            for record in records
            if record.get(key) is not None
        }
        task_clusters = [
            cluster
            for cluster in clusters
            if cluster in task_cluster_set
        ]
        if len(task_clusters) < 3:
            raise ValueError(f"task {key} has fewer than three protein clusters")
        for cluster, split in zip(task_clusters[:3], ("train", "validation", "test")):
            existing = assignments.get(cluster)
            if existing is not None and existing != split:
                raise ValueError(f"task coverage constraints conflict for cluster {cluster}")
            assignments[cluster] = split
            assigned_counts[split] += cluster_sizes[cluster]
    for cluster in clusters:
        if cluster in assignments:
            continue
        split = max(
            fractions,
            key=lambda name: (
                targets[name] - assigned_counts[name],
                fractions[name],
                name,
            ),
        )
        assignments[cluster] = split
        assigned_counts[split] += cluster_sizes[cluster]
    return assignments


def eligible_labels(
    records: list[dict],
    label_key: str,
    minimums: dict[str, int],
) -> set[str | int]:
    counts = {
        split: Counter(
            record.get(label_key)
            for record in records
            if record["split"] == split and record.get(label_key) is not None
        )
        for split in minimums
    }
    labels = set.intersection(*(set(counter) for counter in counts.values()))
    return {
        label
        for label in labels
        if all(counts[split][label] >= minimums[split] for split in minimums)
    }


def split_report(records: list[dict], label_keys: tuple[str, ...]) -> dict:
    report = {}
    for split in ("train", "validation", "test"):
        members = [record for record in records if record["split"] == split]
        report[split] = {
            "records": len(members),
            "clusters": len({record["protein_cluster"] for record in members}),
            "sources": dict(sorted(Counter(record["source"] for record in members).items())),
            "labels": {
                key: dict(
                    sorted(
                        Counter(
                            str(record[key])
                            for record in members
                            if record.get(key) is not None
                        ).items()
                    )
                )
                for key in label_keys
            },
        }
    split_clusters = {
        split: {
            record["protein_cluster"]
            for record in records
            if record["split"] == split
        }
        for split in report
    }
    crossing = set()
    for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
        crossing.update(split_clusters[left] & split_clusters[right])
    report["cross_split_clusters"] = sorted(crossing)
    return report


def group_by_sequence(records: list[dict]) -> tuple[list[dict], list[dict]]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        grouped[record["sequence"]].append(record)

    merged, quarantined = [], []
    for sequence, members in grouped.items():
        combined = {
            "record_id": sequence_sha256(sequence),
            "sequence": sequence,
            "source": "+".join(sorted({member["source"] for member in members})),
            "source_ids": sorted({source_id for member in members for source_id in member["source_ids"]}),
        }
        conflict = False
        for key in ("pfam_label", "ec_label"):
            values = {member.get(key) for member in members if member.get(key) is not None}
            if len(values) > 1:
                conflict = True
            combined[key] = next(iter(values)) if len(values) == 1 else None
        stability = [float(member["stability_score"]) for member in members if member.get("stability_score") is not None]
        if stability and max(stability) - min(stability) > 1e-6:
            conflict = True
        combined["stability_score"] = stability[0] if stability else None
        if conflict:
            quarantined.append({"sequence_sha256": combined["record_id"], "source_ids": combined["source_ids"]})
        else:
            merged.append(combined)
    return merged, quarantined
