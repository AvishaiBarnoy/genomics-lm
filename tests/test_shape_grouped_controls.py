import numpy as np
import pytest

from scripts.eval_shape_baselines import (
    _local_mer,
    make_group_folds,
    select_window_indices,
)


def test_group_folds_are_disjoint_and_deterministic():
    groups = np.array(["a", "a", "b", "b", "c", "d", "e", "e"])
    first, assignments = make_group_folds(groups, n_splits=3, seed=17)
    second, repeated = make_group_folds(groups, n_splits=3, seed=17)
    assert assignments == repeated
    for (train, test), (train2, test2) in zip(first, second):
        np.testing.assert_array_equal(train, train2)
        np.testing.assert_array_equal(test, test2)
        assert set(groups[train]).isdisjoint(set(groups[test]))


def test_group_folds_require_enough_groups():
    with pytest.raises(ValueError, match="at least 3 groups"):
        make_group_folds(np.array(["a", "a", "b"]), n_splits=3, seed=1)


def test_local_mers_pad_sequence_boundaries():
    assert _local_mer("ATGCCC", 0, 5) == "NATGC"
    assert _local_mer("ATGCCC", 1, 5) == "GCCCN"
    assert _local_mer("ATGCCC", 0, 7) == "NNATGCC"
    assert _local_mer("ATGCCC", 1, 7) == "TGCCCNN"


def test_window_sampling_balances_genomes_and_is_deterministic():
    spans = {
        index: [{"source_id": f"gene-{index}"}]
        for index in range(10)
    }
    genomes = {
        f"gene-{index}": "genome-a" if index < 8 else "genome-b"
        for index in range(10)
    }
    first, counts = select_window_indices(
        spans, genomes, "genome", max_windows=4, seed=17
    )
    repeated, repeated_counts = select_window_indices(
        spans, genomes, "genome", max_windows=4, seed=17
    )
    assert first == repeated
    assert counts == repeated_counts == {"genome-a": 2, "genome-b": 2}
