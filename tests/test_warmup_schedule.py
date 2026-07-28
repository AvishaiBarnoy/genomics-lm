import pytest

from src.codonlm.training.loop import resolve_warmup_steps


def test_fixed_warmup_steps_remain_backward_compatible():
    assert resolve_warmup_steps({"warmup_steps": 100}, 1000) == 100
    assert resolve_warmup_steps({}, 1000) == 200


@pytest.mark.parametrize(
    ("total_steps", "expected"),
    [(1000, 100), (2000, 200), (4000, 400)],
)
def test_fractional_warmup_tracks_scheduler_horizon(total_steps, expected):
    assert resolve_warmup_steps({"warmup_fraction": 0.1}, total_steps) == expected


def test_warmup_configuration_fails_closed():
    with pytest.raises(ValueError, match="only one"):
        resolve_warmup_steps(
            {"warmup_steps": 100, "warmup_fraction": 0.1},
            1000,
        )
    with pytest.raises(ValueError, match=r"\[0, 1\)"):
        resolve_warmup_steps({"warmup_fraction": 1.0}, 1000)
    with pytest.raises(ValueError, match="non-negative"):
        resolve_warmup_steps({"warmup_steps": -1}, 1000)
