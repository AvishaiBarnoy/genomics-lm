from scripts.run_decoding_termination_ablation import _sample_seed, _summary


def test_decoding_ablation_summary_and_seeds_are_deterministic():
    rows = [
        {
            "had_terminal_stop": True,
            "stop_reason": "biological_stop",
            "hit_hard_cap": False,
            "generated_tokens": 10,
            "generated_codons": 10,
            "gc_fraction": 0.4,
        },
        {
            "had_terminal_stop": False,
            "stop_reason": "max_new_tokens",
            "hit_hard_cap": True,
            "generated_tokens": 20,
            "generated_codons": 20,
            "gc_fraction": 0.6,
        },
    ]

    summary = _summary(rows)
    assert summary["natural_stop_rate"] == 0.5
    assert summary["hard_cap_rate"] == 0.5
    assert summary["mean_generated_tokens"] == 15.0
    assert summary["mean_gc_fraction"] == 0.5
    assert _sample_seed(3, 2, "x") == _sample_seed(3, 2, "x")
