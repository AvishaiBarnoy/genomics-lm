from __future__ import annotations

import copy

import torch

from src.protein_lm.config import ProteinLMConfig
from src.protein_lm.models import ProteinConditionalTransformer
from src.protein_lm.tasks import ProteinLMTask, decode_protein_lm_checkpoint
from src.protein_lm.tokenizer import ProteinTokenizer
from src.training.engine import EngineConfig, TrainingEngine
from src.training.run_lifecycle import TrainingRun
from src.training.strategies import AccumulatedBackpropStrategy


def _model(tokenizer):
    return ProteinConditionalTransformer(
        ProteinLMConfig(
            vocab_size=len(tokenizer.vocab),
            n_layer=1,
            n_head=1,
            n_embd=8,
            block_size=6,
            dropout=0.0,
        )
    )


def _batches(tokenizer):
    pad = tokenizer.pad_token_id
    return [
        torch.tensor([[tokenizer.bos_token_id, 3, 4, 5, pad, pad]]),
        torch.tensor([[tokenizer.bos_token_id, 6, 7, 8, pad, pad]]),
    ]


def _loss(model, batch, tokenizer):
    targets = batch[:, 1:].contiguous()
    logits = model(batch[:, :-1]).contiguous()
    return torch.nn.functional.cross_entropy(
        logits.view(-1, logits.size(-1)),
        targets.view(-1),
        ignore_index=tokenizer.pad_token_id,
    )


class _ExpireOnce:
    def __init__(self):
        self.expired_once = False

    def expired(self):
        if self.expired_once:
            return False
        self.expired_once = True
        return True


def _engine_for_task(run, task, epochs=1, timer=None):
    optimizer = torch.optim.AdamW(task.model.parameters(), lr=1e-3, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=2)
    engine = TrainingEngine(
        task=task,
        strategy=AccumulatedBackpropStrategy(
            optimizer,
            scheduler=scheduler,
            parameters=task.model.parameters(),
            scheduler_interval="epoch",
        ),
        run=run,
        config=EngineConfig(epochs=epochs, grad_accum_steps=2),
        device=torch.device("cpu"),
        wall_timer=timer,
    )
    return engine, scheduler


def test_protein_lm_task_matches_former_update_and_scheduler_equations(tmp_path):
    torch.manual_seed(19)
    tokenizer = ProteinTokenizer()
    model = _model(tokenizer)
    reference = copy.deepcopy(model)
    batches = _batches(tokenizer)
    validation = _batches(tokenizer)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=2)
    task = ProteinLMTask(
        model=model,
        train_loader=batches,
        validation_loader=validation,
        tokenizer=tokenizer,
        device=torch.device("cpu"),
        train_generator=torch.Generator(),
        seed=19,
        log_every_microbatches=0,
    )
    run = TrainingRun.open(tmp_path, "protein-parity")
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
    reference_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        reference_optimizer, T_max=2
    )
    reference_optimizer.zero_grad(set_to_none=True)
    for batch in batches:
        (_loss(reference, batch, tokenizer) / 2).backward()
    reference_optimizer.step()
    reference_optimizer.zero_grad(set_to_none=True)
    reference_scheduler.step()
    expected_validation = sum(
        _loss(reference, batch, tokenizer).item() for batch in validation
    ) / len(validation)

    for actual, expected in zip(model.parameters(), reference.parameters()):
        assert torch.allclose(actual, expected)
    assert result.state.optimizer_step == 1
    assert scheduler.state_dict() == reference_scheduler.state_dict()
    assert result.best_metric == expected_validation
    run.close()


def test_legacy_protein_lm_checkpoint_is_translated_without_guessing():
    tokenizer = ProteinTokenizer()
    model = _model(tokenizer)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=2)
    legacy = {
        "epoch": 0,
        "epoch_complete": False,
        "microbatch_idx": 3,
        "optimizer_step": 2,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "rng_state": {},
    }

    translated = decode_protein_lm_checkpoint(legacy)

    assert translated.engine.completed_epochs == 0
    assert translated.engine.current_epoch == 0
    assert translated.engine.microbatch == 3
    assert translated.engine.optimizer_step == 2
    for name, tensor in translated.task["model"].items():
        assert torch.equal(tensor, legacy["model_state_dict"][name])
    assert translated.strategy["optimizer"].keys() == legacy[
        "optimizer_state_dict"
    ].keys()


def test_protein_lm_interrupted_resume_matches_uninterrupted_run(tmp_path):
    tokenizer = ProteinTokenizer()
    batches = _batches(tokenizer) * 2
    torch.manual_seed(31)
    initial = _model(tokenizer).state_dict()

    def make_task():
        model = _model(tokenizer)
        model.load_state_dict(initial)
        return ProteinLMTask(
            model=model,
            train_loader=batches,
            validation_loader=_batches(tokenizer),
            tokenizer=tokenizer,
            device=torch.device("cpu"),
            train_generator=torch.Generator(),
            seed=31,
            log_every_microbatches=0,
        )

    reference_task = make_task()
    reference_run = TrainingRun.open(tmp_path, "protein-reference")
    reference_engine, _ = _engine_for_task(reference_run, reference_task)
    reference_result = reference_engine.fit()
    reference_state = copy.deepcopy(reference_task.model.state_dict())
    reference_run.close()

    interrupted_task = make_task()
    interrupted_run = TrainingRun.open(tmp_path, "protein-resume")
    interrupted_engine, _ = _engine_for_task(
        interrupted_run, interrupted_task, timer=_ExpireOnce()
    )
    interrupted_result = interrupted_engine.fit()
    checkpoint = interrupted_run.checkpoints / "last.pt"
    interrupted_run.close()

    assert interrupted_result.status == "interrupted"
    assert interrupted_result.state.microbatch == 2

    resumed_task = make_task()
    resumed_run = TrainingRun.open(
        tmp_path, "protein-resume", resume=checkpoint, target_epochs=1
    )
    resumed_engine, resumed_scheduler = _engine_for_task(resumed_run, resumed_task)
    resumed_result = resumed_engine.fit()

    assert resumed_result.state.optimizer_step == reference_result.state.optimizer_step
    assert resumed_scheduler.last_epoch == 1
    for name, tensor in resumed_task.model.state_dict().items():
        assert torch.allclose(tensor, reference_state[name])
    resumed_run.close()
