from __future__ import annotations

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.codonlm.model_tiny_gpt import TinyGPT
from src.codonlm.training.callbacks import CodonLMConsole
from src.codonlm.training.task import (
    CodonLMTask,
    adapt_codon_lm_checkpoint,
    decode_codon_lm_checkpoint,
)
from src.training.contracts import (
    EngineEvent,
    MetricValue,
    StepContext,
    TrainingPhase,
)
from src.training.engine import EngineConfig, TrainingEngine
from src.training.run_lifecycle import TrainingRun
from src.training.strategies import AccumulatedBackpropStrategy


class _ExpireAfterFirstGroup:
    def __init__(self):
        self.calls = 0

    def expired(self):
        self.calls += 1
        return self.calls == 1


def _replay_engine(tmp_path, run, *, timer=None):
    model = TinyGPT(
        vocab_size=8,
        block_size=4,
        n_layer=1,
        n_head=1,
        n_embd=8,
        dropout=0.0,
        termination_aux=True,
        termination_n_classes=5,
    )
    primary = [
        (torch.tensor([[1, 4, 5, 6]]), torch.tensor([[4, 5, 6, 2]])),
        (torch.tensor([[1, 5, 6, 7]]), torch.tensor([[5, 6, 7, 2]])),
        (torch.tensor([[1, 6, 7, 4]]), torch.tensor([[6, 7, 4, 2]])),
        (torch.tensor([[1, 7, 4, 5]]), torch.tensor([[7, 4, 5, 2]])),
    ]
    replay_x = torch.tensor(
        [[1, 4, 5, 6], [1, 5, 6, 7], [1, 6, 7, 4], [1, 7, 4, 5]]
    )
    replay_y = torch.full((4, 4), -100, dtype=torch.long)
    replay_y[:, -1] = torch.tensor([0, 1, 2, 3])
    replay_generator = torch.Generator().manual_seed(991)
    replay_loader = DataLoader(
        TensorDataset(replay_x, replay_y),
        batch_size=1,
        shuffle=True,
        generator=replay_generator,
    )
    task = CodonLMTask(
        model=model,
        train_loader_factory=lambda epoch: primary,
        validation_loader=primary[:1],
        device=torch.device("cpu"),
        config={
            "replay_loss_enabled": True,
            "replay_loss_weight": 0.25,
            "replay_every_microbatches": 1,
        },
        replay_loader=replay_loader,
        replay_generator=replay_generator,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda step: 1.0
    )
    engine = TrainingEngine(
        task=task,
        strategy=AccumulatedBackpropStrategy(optimizer, scheduler=scheduler),
        run=run,
        config=EngineConfig(epochs=1, grad_accum_steps=1),
        device=torch.device("cpu"),
        wall_timer=timer,
    )
    return engine, model, optimizer, scheduler


def _assert_nested_equal(actual, expected):
    if isinstance(expected, torch.Tensor):
        assert torch.equal(actual, expected)
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            _assert_nested_equal(actual[key], expected[key])
    elif isinstance(expected, list):
        assert len(actual) == len(expected)
        for actual_item, expected_item in zip(actual, expected):
            _assert_nested_equal(actual_item, expected_item)
    else:
        assert actual == expected


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
            "training_initial_metrics": {"loss": 2.25, "next_loss": 2.0},
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
    assert adapted["epoch_train_metrics"]["initial_loss"] == 2.25


def test_replay_enabled_resume_matches_uninterrupted_parameters(tmp_path):
    torch.manual_seed(77)
    reference_run = TrainingRun.open(tmp_path, "reference")
    reference_engine, reference_model, reference_optimizer, reference_scheduler = (
        _replay_engine(tmp_path, reference_run)
    )
    reference_result = reference_engine.fit()
    reference_state = {
        name: value.detach().clone()
        for name, value in reference_model.state_dict().items()
    }
    reference_optimizer_state = reference_optimizer.state_dict()
    reference_scheduler_state = reference_scheduler.state_dict()
    reference_run.close()

    torch.manual_seed(77)
    interrupted_run = TrainingRun.open(tmp_path, "resumable")
    interrupted_engine, _, _, _ = _replay_engine(
        tmp_path, interrupted_run, timer=_ExpireAfterFirstGroup()
    )
    interrupted_result = interrupted_engine.fit()
    checkpoint = interrupted_run.checkpoints / "last.pt"
    interrupted_run.close()

    assert interrupted_result.status == "interrupted"
    assert interrupted_result.state.microbatch == 1

    resumed_run = TrainingRun.open(
        tmp_path, "resumable", resume=checkpoint, target_epochs=1
    )
    resumed_engine, resumed_model, resumed_optimizer, resumed_scheduler = _replay_engine(
        tmp_path, resumed_run
    )
    resumed_result = resumed_engine.fit()

    assert reference_result.state == resumed_result.state
    for name, expected in reference_state.items():
        assert torch.equal(resumed_model.state_dict()[name], expected), name
    _assert_nested_equal(resumed_optimizer.state_dict(), reference_optimizer_state)
    _assert_nested_equal(resumed_scheduler.state_dict(), reference_scheduler_state)
    resumed_run.close()


def test_replay_console_allows_epoch_without_replay_metric(tmp_path):
    curves = tmp_path / "curves.csv"
    curves.write_text("epoch,train_loss,val_loss,train_next,val_next,ppl,lr,replay\n")
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.SGD([parameter], lr=0.1)
    callback = CodonLMConsole(
        curves,
        multi_offset_weights={},
        termination_enabled=False,
        replay_enabled=True,
        optimizer=optimizer,
    )

    callback.on_event(
        EngineEvent(
            "epoch_completed",
            metrics={
                "loss": MetricValue(1.5),
                "next_loss": MetricValue(1.25),
            },
            metadata={
                "epoch": 1,
                "training_metrics": {
                    "loss": MetricValue(2.0),
                    "next_loss": MetricValue(1.75),
                },
            },
        )
    )

    assert curves.read_text().splitlines()[-1].endswith(",")


def test_codon_task_emits_multi_offset_and_termination_metrics():
    model = TinyGPT(
        vocab_size=8,
        block_size=4,
        n_layer=1,
        n_head=1,
        n_embd=8,
        dropout=0.0,
        termination_aux=True,
        termination_n_classes=5,
        multi_offset_targets=[2],
    )
    task = CodonLMTask(
        model=model,
        train_loader_factory=lambda epoch: [],
        validation_loader=[],
        device=torch.device("cpu"),
        config={
            "termination_loss_enabled": True,
            "termination_loss_weight": 0.1,
            "termination_stop_ids": [2],
            "termination_bucket_edges": [0, 1, 2, 3],
        },
        multi_offset_weights={2: 0.5},
    )
    batch = (
        torch.tensor([[1, 4, 5, 6]]),
        torch.tensor([[4, 5, 6, 2]]),
    )

    output = task.training_step(
        batch,
        StepContext(TrainingPhase.TRAIN, 0, 0, 0, torch.device("cpu")),
    )

    assert torch.isfinite(output.loss)
    assert {"loss", "next_loss", "offset_2", "term_loss"} <= set(output.metrics)


@pytest.mark.parametrize("unfreeze", [False, True])
def test_shape_encoder_phase_matches_unfreeze_configuration(unfreeze):
    model = torch.nn.Linear(1, 1)
    encoder = torch.nn.Linear(1, 1)
    task = CodonLMTask(
        model=model,
        train_loader_factory=lambda epoch: [],
        validation_loader=[],
        device=torch.device("cpu"),
        config={"unfreeze_encoder": unfreeze},
        encoder=encoder,
    )

    task.begin_phase(TrainingPhase.TRAIN, 0)
    assert encoder.training is unfreeze
    task.begin_phase(TrainingPhase.VALIDATION, 0)
    assert encoder.training is False
