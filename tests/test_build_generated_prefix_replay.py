from scripts.build_generated_prefix_replay import _codon_positions, _replay_labels


def test_codon_positions_counts_only_codon_tokens():
    itos = ["<PAD>", "<BOS_CDS>", "ATG", "AAA", "N", "TAA"]
    ids = [1, 2, 4, 3, 5]

    assert _codon_positions(ids, itos) == [
        (1, 1, "ATG"),
        (3, 2, "AAA"),
        (4, 3, "TAA"),
    ]


def test_replay_labels_targets_boundary_window_after_prefix():
    ids = [1, 2, 3, 4, 5, 3, 4, 5]

    labels = _replay_labels(
        ids,
        prefix_tokens=2,
        window=4,
        bucket_edges=(0, 1, 3),
    )

    assert labels == [
        {"pos": 3, "class": 3},
        {"pos": 4, "class": 2},
        {"pos": 5, "class": 2},
        {"pos": 6, "class": 1},
        {"pos": 7, "class": 0},
    ]


def test_replay_labels_do_not_supervise_the_original_prefix():
    labels = _replay_labels(
        [1, 2, 3],
        prefix_tokens=2,
        window=30,
        bucket_edges=(0, 3, 10, 30),
    )

    assert labels == [{"pos": 2, "class": 0}]
