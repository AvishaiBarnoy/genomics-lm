from __future__ import annotations

import torch

from src.codonlm.training.task import (
    adapt_codon_lm_checkpoint,
    decode_codon_lm_checkpoint,
)


def test_legacy_codon_checkpoint_decodes_to_shared_contract():
    model = {"weight": torch.tensor([1.0])}
    optimizer = {"state": {}, "param_groups": []}
    scheduler = {"last_epoch": 3}
    payload = {
        "model": model,
        "optimizer": optimizer,
        "scheduler": scheduler,
        "cfg": {"scheduler": "cosine"},
        "epoch": 1,
        "step": 3,
        "consumed_train_tokens": 24,
        "epoch_microbatch_idx": 2,
        "run_progress": {
            "completed_epochs": 1,
            "current_epoch": 1,
            "microbatch": 2,
            "optimizer_step": 3,
        },
        "accumulation_health": {
            "active_microbatches": 0,
            "nonfinite_microbatches": 1,
            "aborted_groups": 1,
            "discarded_finite_microbatches": 2,
        },
        "rng_state": {},
    }

    checkpoint = decode_codon_lm_checkpoint(payload)

    assert checkpoint.engine.completed_epochs == 1
    assert checkpoint.engine.current_epoch == 1
    assert checkpoint.engine.microbatch == 2
    assert checkpoint.engine.optimizer_step == 3
    assert checkpoint.task["model"] == model
    assert checkpoint.strategy["optimizer"] == optimizer
    assert checkpoint.strategy["scheduler"] == scheduler
    assert checkpoint.metadata["committed_units"] == {"tokens": 24}
    assert checkpoint.metadata["aborted_groups"] == 1


def test_engine_checkpoint_keeps_legacy_codon_aliases():
    payload = {
        "engine": {
            "completed_epochs": 2,
            "current_epoch": 2,
            "microbatch": 0,
            "optimizer_step": 4,
        },
        "task": {"model": {"weight": torch.tensor([2.0])}, "runtime_memory": {}},
        "strategy": {
            "optimizer": {"state": {}},
            "scheduler": {"last_epoch": 4},
            "accumulation_health": {
                "active_microbatches": 0,
                "nonfinite_microbatches": 0,
                "aborted_groups": 0,
                "discarded_finite_microbatches": 0,
            },
        },
        "rng": {},
        "metadata": {
            "reason": "epoch",
            "best_metric": 1.25,
            "best_epoch": 2,
            "metrics": {"loss": 1.25, "next_loss": 1.0},
            "training_metrics": {"loss": 1.5, "next_loss": 1.2},
            "training_metric_weights": {"loss": 5, "next_loss": 5},
            "committed_units": {"tokens": 80},
            "no_improve": 0,
        },
    }

    adapted = adapt_codon_lm_checkpoint(
        payload,
        config={"device": "cpu"},
        batch_size=2,
        grad_accum_steps=2,
        train_examples=5,
        train_batches=3,
        max_nonfinite_groups=3,
    )

    assert adapted["model"] is adapted["task"]["model"]
    assert adapted["optimizer"] is adapted["strategy"]["optimizer"]
    assert adapted["scheduler"] is adapted["strategy"]["scheduler"]
    assert adapted["step"] == 4
    assert adapted["epoch"] == 2
    assert adapted["consumed_train_tokens"] == 80
    assert adapted["epoch_microbatch_idx"] == 0
    assert adapted["epoch_train_metrics"]["microbatches"] == 5
