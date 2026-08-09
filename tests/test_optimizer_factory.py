import pytest
import torch

from src.training.optimizers import build_optimizer


def _parameters():
    return torch.nn.Linear(2, 1).parameters()


def test_optimizer_factory_preserves_adamw_default():
    optimizer = build_optimizer(
        _parameters(), {"lr": 1e-3, "weight_decay": 0.02}
    )
    assert isinstance(optimizer, torch.optim.AdamW)
    assert optimizer.param_groups[0]["lr"] == 1e-3
    assert optimizer.param_groups[0]["weight_decay"] == 0.02


@pytest.mark.parametrize(
    ("name", "expected"),
    [("adam", torch.optim.Adam), ("sgd", torch.optim.SGD)],
)
def test_optimizer_factory_selects_registered_algorithms(name, expected):
    optimizer = build_optimizer(
        _parameters(),
        {"lr": 1e-3, "optimizer": {"name": name, "weight_decay": 0.0}},
    )
    assert isinstance(optimizer, expected)


def test_optimizer_factory_allows_algorithm_specific_options():
    optimizer = build_optimizer(
        _parameters(),
        {
            "lr": 1e-3,
            "optimizer": {
                "name": "sgd",
                "lr": 0.1,
                "momentum": 0.9,
                "nesterov": True,
            },
        },
    )
    assert optimizer.param_groups[0]["lr"] == 0.1
    assert optimizer.param_groups[0]["momentum"] == 0.9
    assert optimizer.param_groups[0]["nesterov"] is True


def test_optimizer_factory_rejects_unknown_algorithm_or_option():
    with pytest.raises(ValueError, match="unsupported optimizer"):
        build_optimizer(_parameters(), {"lr": 1e-3, "optimizer": "mystery"})
    with pytest.raises(ValueError, match="unsupported adamw optimizer options"):
        build_optimizer(
            _parameters(),
            {"lr": 1e-3, "optimizer": {"name": "adamw", "momentum": 0.9}},
        )


@pytest.mark.parametrize("lr", [None, 0.0, -1.0, True])
def test_optimizer_factory_rejects_invalid_learning_rate(lr):
    with pytest.raises((TypeError, ValueError), match="learning rate|required|lr"):
        build_optimizer(_parameters(), {"lr": lr})

