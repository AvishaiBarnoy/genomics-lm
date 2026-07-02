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
    itos = ["<PAD>", "<BOS_CDS>", "ATG", "AAA", "CCC", "GGG", "TAA"]
    ids = [1, 2, 3, 4, 5, 3, 4, 5]

    labels = _replay_labels(
        ids,
        itos,
        prefix_codons=2,
        target_codons=4,
        window=2,
        near_class=1,
        immediate_class=0,
    )

    assert labels == [
        {"pos": 4, "class": 1},
        {"pos": 5, "class": 1},
        {"pos": 6, "class": 0},
        {"pos": 7, "class": 0},
    ]
