from __future__ import annotations

import copy

import pytest
import torch

from src.protein_lm.classifier_task import (
    ProteinClassifierTask,
    adapt_protein_classifier_checkpoint,
    decode_protein_classifier_checkpoint,
)
from src.training.contracts import TrainingPhase
from src.training.engine import EngineConfig, TrainingEngine
from src.training.run_lifecycle import TrainingRun
from src.training.strategies import AccumulatedBackpropStrategy


class TinyClassifier(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(8, 4)
        self.head = torch.nn.Linear(4, 2)

    def forward(self, input_ids, attention_mask=None):
        hidden = self.embedding(input_ids)
        weights = attention_mask.unsqueeze(-1)
        pooled = (hidden * weights).sum(1) / weights.sum(1).clamp_min(1)
        return self.head(pooled)


def _batches():
    return [
        (torch.tensor([[1, 2, 0]]), torch.tensor([0])),
        (torch.tensor([[1, 3, 0]]), torch.tensor([1])),
        (torch.tensor([[1, 4, 5]]), torch.tensor([1])),
    ]


def _task(model, *, batches=None, label_map=None):
    batches = batches or _batches()
    return ProteinClassifierTask(
        model=model,
        train_loader=batches,
        validation_loader=batches,
        criterion=torch.nn.CrossEntropyLoss(),
        device=torch.device("cpu"),
        train_generator=torch.Generator(),
        seed=17,
        label_map=label_map or {"a": 0, "b": 1},
        pad_token_id=0,
        log_every_microbatches=0,
    )


def test_classifier_task_matches_actual_size_accumulation_and_metrics(tmp_path):
    torch.manual_seed(11)
    model = TinyClassifier()
    reference = copy.deepcopy(model)
    batches = _batches()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1)
    task = _task(model, batches=batches)
    run = TrainingRun.open(tmp_path, "classifier-parity")
    engine = TrainingEngine(
        task=task,
        strategy=AccumulatedBackpropStrategy(
            optimizer,
            scheduler=scheduler,
            parameters=model.parameters(),
            scheduler_interval="epoch",
        ),
        run=run,
        config=EngineConfig(epochs=1, grad_accum_steps=2),
        device=torch.device("cpu"),
    )
    result = engine.fit()

    reference_optimizer = torch.optim.AdamW(
        reference.parameters(), lr=1e-3, weight_decay=0.01
    )
    for group in (batches[:2], batches[2:]):
        reference_optimizer.zero_grad(set_to_none=True)
        losses = []
        for input_ids, labels in group:
            logits = reference(input_ids, attention_mask=input_ids.ne(0))
            losses.append(torch.nn.functional.cross_entropy(logits, labels))
        torch.stack(losses).mean().backward()
        reference_optimizer.step()

    for actual, expected in zip(model.parameters(), reference.parameters()):
        assert torch.allclose(actual, expected)
    assert result.state.optimizer_step == 2
    assert scheduler.last_epoch == 1
    task.begin_phase(TrainingPhase.VALIDATION, 1)
    for index, batch in enumerate(batches):
        task.validation_step(batch, _context(index))
    metrics = task.end_phase(TrainingPhase.VALIDATION, 1)
    assert set(metrics) == {"accuracy", "weighted_f1"}
    run.close()


def _context(microbatch):
    from src.training.contracts import StepContext

    return StepContext(
        TrainingPhase.VALIDATION,
        epoch=0,
        microbatch=microbatch,
        optimizer_step=0,
        device=torch.device("cpu"),
    )


def test_classifier_task_rejects_changed_checkpoint_label_map():
    task = _task(TinyClassifier())
    with pytest.raises(ValueError, match="label map differs"):
        task.load_state_dict(
            {"model": task.model.state_dict(), "label_map": {"different": 0}}
        )


def test_legacy_classifier_checkpoint_translation_and_aliases():
    model = TinyClassifier()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=2)
    legacy = {
        "epoch": 1,
        "epoch_complete": False,
        "microbatch_idx": 3,
        "optimizer_step": 2,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "label_map": {"a": 0, "b": 1},
        "rng_state": {},
    }
    decoded = decode_protein_classifier_checkpoint(legacy)
    assert decoded.engine.current_epoch == 1
    assert decoded.engine.microbatch == 3
    assert decoded.task["label_map"] == legacy["label_map"]

    payload = decoded.to_payload()
    payload["metadata"] = {
        "reason": "epoch",
        "best_metric": 0.5,
        "metrics": {"loss": 0.6},
    }
    adapted = adapt_protein_classifier_checkpoint(payload, config={"epochs": 2})
    assert adapted["model_state_dict"] is payload["task"]["model"]
    assert adapted["label_map"] == legacy["label_map"]
    assert adapted["loss"] == 0.6


class ExpireAfterFirstGroup:
    def __init__(self):
        self.calls = 0

    def expired(self):
        self.calls += 1
        return self.calls == 1


def test_classifier_interrupted_resume_matches_uninterrupted(tmp_path):
    torch.manual_seed(23)
    initial = TinyClassifier().state_dict()

    def build(run, *, timer=None):
        model = TinyClassifier()
        model.load_state_dict(initial)
        task = _task(model, batches=_batches() + _batches())
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1)
        return task, TrainingEngine(
            task=task,
            strategy=AccumulatedBackpropStrategy(
                optimizer,
                scheduler=scheduler,
                scheduler_interval="epoch",
            ),
            run=run,
            config=EngineConfig(epochs=1, grad_accum_steps=2),
            device=torch.device("cpu"),
            wall_timer=timer,
        )

    reference_run = TrainingRun.open(tmp_path, "reference")
    reference_task, reference_engine = build(reference_run)
    reference_engine.fit()
    reference_state = copy.deepcopy(reference_task.model.state_dict())
    reference_run.close()

    interrupted_run = TrainingRun.open(tmp_path, "resumed")
    _, interrupted_engine = build(interrupted_run, timer=ExpireAfterFirstGroup())
    interrupted = interrupted_engine.fit()
    checkpoint = interrupted_run.checkpoints / "last.pt"
    interrupted_run.close()
    assert interrupted.state.microbatch == 2

    resumed_run = TrainingRun.open(
        tmp_path, "resumed", resume=checkpoint, target_epochs=1
    )
    resumed_task, resumed_engine = build(resumed_run)
    resumed_engine.fit()
    for name, tensor in resumed_task.model.state_dict().items():
        assert torch.allclose(tensor, reference_state[name])
    resumed_run.close()
