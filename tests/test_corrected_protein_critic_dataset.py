import pytest

from src.protein_lm.corrected_dataset import (
    assign_clusters,
    eligible_labels,
    group_by_sequence,
    normalize_protein,
    split_report,
)


def test_normalize_protein_rejects_ambiguous_residues():
    assert normalize_protein(" acd ef* ") == "ACDEF"
    with pytest.raises(ValueError):
        normalize_protein("ACDX")


def test_cluster_assignment_is_deterministic_and_disjoint():
    records = [
        {"protein_cluster": f"c{index // 2}", "source": "fixture"}
        for index in range(60)
    ]
    first = assign_clusters(records, seed=1337)
    second = assign_clusters(records, seed=1337)
    assert first == second
    for record in records:
        record["split"] = first[record["protein_cluster"]]
    report = split_report(records, ())
    assert report["cross_split_clusters"] == []
    assert all(report[split]["records"] > 0 for split in ("train", "validation", "test"))


def test_large_cluster_is_assigned_before_small_split_can_be_consumed():
    records = [
        *({"protein_cluster": "large", "source": "fixture"} for _ in range(60)),
        *(
            {"protein_cluster": f"small-{index}", "source": "fixture"}
            for index in range(40)
        ),
    ]
    assignments = assign_clusters(records, seed=1337)
    assert assignments["large"] == "train"


def test_required_task_clusters_seed_all_three_splits():
    records = [
        *(
            {"protein_cluster": "stability-large", "source": "fixture", "score": 1.0}
            for _ in range(20)
        ),
        {"protein_cluster": "stability-medium", "source": "fixture", "score": 2.0},
        {"protein_cluster": "stability-small", "source": "fixture", "score": 3.0},
        *(
            {"protein_cluster": f"other-{index}", "source": "fixture", "score": None}
            for index in range(30)
        ),
    ]
    assignments = assign_clusters(records, seed=1337, required_task_keys=("score",))
    task_splits = {
        assignments[record["protein_cluster"]]
        for record in records
        if record["score"] is not None
    }
    assert task_splits == {"train", "validation", "test"}


def test_eligible_labels_requires_support_in_every_split():
    records = []
    for split, count in (("train", 4), ("validation", 2), ("test", 2)):
        records.extend({"split": split, "label": "kept"} for _ in range(count))
    records.extend({"split": "train", "label": "train-only"} for _ in range(10))
    assert eligible_labels(
        records,
        "label",
        {"train": 4, "validation": 2, "test": 2},
    ) == {"kept"}


def test_exact_sequence_label_conflicts_are_quarantined():
    base = {
        "sequence": "ACDE",
        "source": "fixture",
        "source_ids": ["a"],
        "ec_label": 1,
        "stability_score": None,
    }
    merged, quarantined = group_by_sequence(
        [
            {**base, "pfam_label": "PF1"},
            {**base, "source_ids": ["b"], "pfam_label": "PF2"},
        ]
    )
    assert merged == []
    assert quarantined[0]["source_ids"] == ["a", "b"]
