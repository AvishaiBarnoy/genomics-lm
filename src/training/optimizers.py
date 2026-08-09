"""Validated optimizer construction shared by training entry points."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from numbers import Real
from typing import Any

import torch


OPTIMIZER_REGISTRY = {
    "adamw": (
        torch.optim.AdamW,
        {"lr", "weight_decay", "betas", "eps", "amsgrad"},
    ),
    "adam": (
        torch.optim.Adam,
        {"lr", "weight_decay", "betas", "eps", "amsgrad"},
    ),
    "sgd": (
        torch.optim.SGD,
        {"lr", "weight_decay", "momentum", "dampening", "nesterov"},
    ),
}


def _real(name: str, value: Any, *, minimum: float, inclusive: bool) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"optimizer {name} must be numeric")
    numeric = float(value)
    valid = numeric >= minimum if inclusive else numeric > minimum
    if not valid:
        relation = ">=" if inclusive else ">"
        raise ValueError(f"optimizer {name} must be {relation} {minimum}")
    return numeric


def build_optimizer(
    parameters: Iterable[torch.nn.Parameter],
    training_config: Mapping[str, Any],
) -> torch.optim.Optimizer:
    """Build an allow-listed optimizer with backward-compatible defaults."""

    configured = training_config.get("optimizer", "adamw")
    if isinstance(configured, str):
        name = configured.lower()
        options: dict[str, Any] = {}
    elif isinstance(configured, Mapping):
        name = str(configured.get("name", "adamw")).lower()
        options = {key: value for key, value in configured.items() if key != "name"}
    else:
        raise TypeError("training.optimizer must be a name or mapping")
    if name not in OPTIMIZER_REGISTRY:
        supported = ", ".join(sorted(OPTIMIZER_REGISTRY))
        raise ValueError(f"unsupported optimizer {name!r}; choose one of: {supported}")

    optimizer_class, allowed = OPTIMIZER_REGISTRY[name]
    unexpected = set(options).difference(allowed)
    if unexpected:
        raise ValueError(
            f"unsupported {name} optimizer options: {sorted(unexpected)}"
        )
    options.setdefault("lr", training_config.get("lr"))
    options.setdefault("weight_decay", training_config.get("weight_decay", 0.01))
    if options["lr"] is None:
        raise ValueError("optimizer learning rate is required")
    options["lr"] = _real("lr", options["lr"], minimum=0.0, inclusive=False)
    options["weight_decay"] = _real(
        "weight_decay", options["weight_decay"], minimum=0.0, inclusive=True
    )
    if "momentum" in options:
        options["momentum"] = _real(
            "momentum", options["momentum"], minimum=0.0, inclusive=True
        )
    if "dampening" in options:
        options["dampening"] = _real(
            "dampening", options["dampening"], minimum=0.0, inclusive=True
        )
    if "eps" in options:
        options["eps"] = _real(
            "eps", options["eps"], minimum=0.0, inclusive=False
        )
    if "betas" in options:
        betas = tuple(options["betas"])
        if len(betas) != 2 or any(
            isinstance(beta, bool)
            or not isinstance(beta, Real)
            or not 0.0 <= float(beta) < 1.0
            for beta in betas
        ):
            raise ValueError("optimizer betas must contain two values in [0, 1)")
        options["betas"] = tuple(float(beta) for beta in betas)
    return optimizer_class(parameters, **options)

