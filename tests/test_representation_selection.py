import csv
import json

import numpy as np

from scripts.select_grouped_representation import main


def test_grouped_representation_selection_uses_training_groups(tmp_path):
    ids = [f"gene-{index}" for index in range(32)]
    labels = np.asarray([index % 2 for index in range(32)])
    groups = [f"cluster-{index // 4}" for index in range(32)]
    embeddings = tmp_path / "train.npz"
    signal = np.column_stack([labels, 1 - labels]).astype(np.float32)
    constant = np.zeros_like(signal)
    np.savez(
        embeddings,
        ids=np.asarray(ids, dtype=object),
        X__layer_2__mean_content=signal,
        X__layer_final__mean_nonpad=constant,
    )
    metadata = {
        "validation_status": "causal_verified",
        "dataset_manifest": {"status": "frozen_manifest_verified"},
        "checkpoint_dataset": {"status": "checkpoint_manifest_verified"},
    }
    embeddings.with_suffix(".npz.metadata.json").write_text(json.dumps(metadata))
    labels_path = tmp_path / "labels.csv"
    with labels_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["id", "label"])
        writer.writerows(zip(ids, labels))
    groups_path = tmp_path / "groups.tsv"
    with groups_path.open("w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["id", "protein_cluster"])
        writer.writerows(zip(ids, groups))
    output = tmp_path / "selection.json"

    main(
        [
            "--embeddings",
            str(embeddings),
            "--labels",
            str(labels_path),
            "--groups",
            str(groups_path),
            "--folds",
            "4",
            "--output",
            str(output),
        ]
    )

    report = json.loads(output.read_text())
    assert report["selection_split"] == "amr_train_only"
    assert report["groups"] == 8
    assert report["winner"] == "layer_2__mean_content"
