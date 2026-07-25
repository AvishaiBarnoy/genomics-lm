import csv
from pathlib import Path

import numpy as np
import pytest

from scripts.diagnose_context_learning import (
    _bootstrap_paired_rows,
    _packing_window_flags,
    _parse_windows,
    _trigram_nll,
)


def test_context_window_parser_preserves_full_and_rejects_zero():
    assert _parse_windows("1,2,full,2") == [1, 2, None]
    with pytest.raises(ValueError, match="positive"):
        _parse_windows("0,full")


def test_packing_window_flags_identify_continuations(tmp_path: Path):
    path = tmp_path / "packing.tsv"
    fields = [
        "window_index",
        "continues_from_previous",
        "continues_to_next",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerow(
            {
                "window_index": 0,
                "continues_from_previous": 0,
                "continues_to_next": 0,
            }
        )
        writer.writerow(
            {
                "window_index": 1,
                "continues_from_previous": 1,
                "continues_to_next": 0,
            }
        )
    assert _packing_window_flags(path, 2).tolist() == [False, True]


def test_segment_aware_trigram_ignores_pre_separator_token():
    counts = np.zeros(6, dtype=np.int64)
    counts[2] = 10
    trigram = {(0, 4): counts}
    x_a = np.array([1, 4])
    x_b = np.array([5, 4])
    y = np.array([0, 2])
    kwargs = {
        "trigram": trigram,
        "bigram": {},
        "alpha": 0.01,
        "active_size": 5,
        "reset_token_ids": frozenset({4}),
    }
    assert np.array_equal(
        _trigram_nll(x_a, y, **kwargs),
        _trigram_nll(x_b, y, **kwargs),
    )


def test_paired_bootstrap_reports_positive_model_gap():
    result = _bootstrap_paired_rows(
        np.array([3.0, 6.0]),
        np.array([2.0, 4.0]),
        np.array([1, 2]),
        seed=7,
        samples=100,
    )
    assert result["codonlm_minus_trigram_nats_per_token"] == pytest.approx(1.0)
    assert result["ci95"][0] > 0
