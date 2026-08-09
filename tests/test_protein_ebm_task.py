from __future__ import annotations

import copy
import json
import random
from types import SimpleNamespace

import pytest
import torch

from src.protein_lm.ebm import ProteinLatentEBM
from src.protein_lm.ebm_task import (
    ProteinEBMTask,
    adapt_protein_ebm_checkpoint,
    decode_protein_ebm_checkpoint,
)
from src.protein_lm.tokenizer import ProteinTokenizer
from src.protein_lm.config import ProteinClassifierConfig
from src.protein_lm.models_multi import MultiTaskProteinClassifier
from src.protein_lm.train_ebm import resolve_critic_checkpoint, train_ebm
from src.training.engine import EngineConfig, TrainingEngine
from src.training.run_lifecycle import TrainingRun
from src.training.strategies import AccumulatedBackpropStrategy


class FrozenCritic(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(32, 4)
        for parameter in self.parameters():
            parameter.requires_grad = False

    def extract_latent(self, input_ids, attention_mask):
        hidden = self.embedding(input_ids)
        weights = attention_mask.unsqueeze(-1)
        return (hidden * weights).sum(1) / weights.sum(1).clamp_min(1)


def _batch(tokenizer, sequences=("MKTAA", "MQQVV")):
    encoded = []
    for sequence in sequences:
        tokens = [tokenizer.bos_token_id] + tokenizer.encode_sequence(sequence)
        encoded.append(tokens + [tokenizer.pad_token_id] * (8 - len(tokens)))
    input_ids = torch.tensor(encoded)
    return {
        "input_ids": input_ids,
        "attention_mask": input_ids.ne(tokenizer.pad_token_id).long(),
        "sequence": list(sequences),
    }


def _task(model, critic, batches):
    tokenizer = ProteinTokenizer()
    return ProteinEBMTask(
        model=model,
        critic=critic,
        train_loader=batches,
        validation_loader=batches[:1],
        tokenizer=tokenizer,
        device=torch.device("cpu"),
        block_size=8,
        log_every_microbatches=0,
    )


def test_ebm_task_updates_only_head_and_reports_energy_metrics(tmp_path):
    tokenizer = ProteinTokenizer()
    batch = _batch(tokenizer)
    torch.manual_seed(5)
    critic = FrozenCritic()
    critic_before = copy.deepcopy(critic.state_dict())
    model = ProteinLatentEBM(n_embd=4, hidden_dim=8, dropout=0.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    task = _task(model, critic, [batch])
    run = TrainingRun.open(tmp_path, "ebm-frozen")
    engine = TrainingEngine(
        task=task,
        strategy=AccumulatedBackpropStrategy(optimizer),
        run=run,
        config=EngineConfig(epochs=1),
        device=torch.device("cpu"),
    )

    random.seed(7)
    result = engine.fit()

    assert result.state.optimizer_step == 1
    assert set(critic.state_dict()) == set(critic_before)
    for name, tensor in critic.state_dict().items():
        assert torch.equal(tensor, critic_before[name])
        assert tensor.grad is None
    payload = torch.load(
        run.checkpoints / "last.pt", map_location="cpu", weights_only=False
    )
    assert set(payload["metadata"]["metrics"]) >= {
        "loss",
        "energy_pos",
        "energy_neg",
        "energy_gap",
    }
    run.close()


def test_ebm_task_rejects_trainable_critic():
    critic = FrozenCritic()
    next(critic.parameters()).requires_grad = True
    with pytest.raises(ValueError, match="must be frozen"):
        _task(
            ProteinLatentEBM(n_embd=4, hidden_dim=8, dropout=0.0),
            critic,
            [_batch(ProteinTokenizer())],
        )


def test_legacy_ebm_checkpoint_translation_preserves_one_based_epoch():
    model = ProteinLatentEBM(n_embd=4, hidden_dim=8)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    legacy = {
        "model": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": 3,
        "epoch_complete": True,
        "best_val_loss": 0.4,
        "best_epoch": 2,
        "rng_state": {},
        "run_progress": {"optimizer_step": 12},
    }
    decoded = decode_protein_ebm_checkpoint(legacy)
    assert decoded.engine.completed_epochs == 3
    assert decoded.engine.current_epoch == 3
    assert decoded.engine.optimizer_step == 12
    assert decoded.metadata["best_epoch"] == 2

    payload = decoded.to_payload()
    payload["metadata"] = {
        "reason": "epoch",
        "best_metric": 0.3,
        "best_epoch": 3,
        "metrics": {"loss": 0.35},
    }
    adapted = adapt_protein_ebm_checkpoint(
        payload,
        critic_checkpoint="critic.pt",
        model_spec={"n_embd": 4},
    )
    assert adapted["epoch"] == 3
    assert adapted["val_loss"] == 0.35
    assert adapted["best_val_loss"] == 0.3
    assert adapted["best_epoch"] == 3


class ExpireOnce:
    def __init__(self):
        self.done = False

    def expired(self):
        if self.done:
            return False
        self.done = True
        return True


def test_ebm_interrupted_resume_matches_uninterrupted(tmp_path):
    tokenizer = ProteinTokenizer()
    batches = [_batch(tokenizer, (sequence,)) for sequence in ("MKTAA", "MQQVV")]
    torch.manual_seed(13)
    critic_state = FrozenCritic().state_dict()
    initial_model = ProteinLatentEBM(n_embd=4, hidden_dim=8, dropout=0.0).state_dict()

    def build(run, timer=None):
        critic = FrozenCritic()
        critic.load_state_dict(critic_state)
        model = ProteinLatentEBM(n_embd=4, hidden_dim=8, dropout=0.0)
        model.load_state_dict(initial_model)
        task = _task(model, critic, batches)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        return task, TrainingEngine(
            task=task,
            strategy=AccumulatedBackpropStrategy(optimizer),
            run=run,
            config=EngineConfig(epochs=1),
            device=torch.device("cpu"),
            wall_timer=timer,
        )

    random.seed(19)
    reference_run = TrainingRun.open(tmp_path, "reference")
    reference_task, reference_engine = build(reference_run)
    reference_engine.fit()
    reference_state = copy.deepcopy(reference_task.model.state_dict())
    reference_run.close()

    random.seed(19)
    interrupted_run = TrainingRun.open(tmp_path, "resumed")
    _, interrupted_engine = build(interrupted_run, ExpireOnce())
    interrupted = interrupted_engine.fit()
    checkpoint = interrupted_run.checkpoints / "last.pt"
    interrupted_run.close()
    assert interrupted.state.microbatch == 1

    resumed_run = TrainingRun.open(
        tmp_path, "resumed", resume=checkpoint, target_epochs=1
    )
    resumed_task, resumed_engine = build(resumed_run)
    resumed_engine.fit()
    for name, tensor in resumed_task.model.state_dict().items():
        assert torch.allclose(tensor, reference_state[name])
    resumed_run.close()


def test_ebm_trainer_accepts_versioned_critic_and_writes_legacy_aliases(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(
        "src.protein_lm.train_ebm.default_device", lambda: torch.device("cpu")
    )
    tokenizer = ProteinTokenizer()
    task_dims = {"family": 2, "function": 2, "stability": 1}
    critic_config = ProteinClassifierConfig(
        vocab_size=len(tokenizer),
        n_layer=1,
        n_head=1,
        n_embd=8,
        block_size=8,
        dropout=0.0,
        num_classes=2,
        pooling="attention",
    )
    critic = MultiTaskProteinClassifier(critic_config, task_dims)
    critic_path = tmp_path / "critic.pt"
    torch.save(
        {
            "training_contract_version": 1,
            "task": {"model": critic.state_dict()},
            "model_spec": {
                "vocab_size": len(tokenizer),
                "n_layer": 1,
                "n_head": 1,
                "n_embd": 8,
                "block_size": 8,
                "dropout": 0.0,
                "pooling": "attention",
                "bidirectional": True,
                "task_dims": task_dims,
            },
        },
        critic_path,
    )
    state, spec = resolve_critic_checkpoint(critic_path)
    assert state.keys() == critic.state_dict().keys()
    assert spec["task_dims"] == task_dims

    train_path = tmp_path / "train.jsonl"
    val_path = tmp_path / "validation.jsonl"
    records = [{"sequence": "MKTAA"}, {"sequence": "MQQVV"}]
    text = "".join(json.dumps(record) + "\n" for record in records)
    train_path.write_text(text)
    val_path.write_text(text)
    config_path = tmp_path / "ebm.yaml"
    config_path.write_text(
        "\n".join(
            [
                f"train_data: {train_path}",
                f"val_data: {val_path}",
                "n_layer: 1",
                "n_head: 1",
                "n_embd: 8",
                "block_size: 8",
                "dropout: 0.0",
                "batch_size: 2",
            ]
        )
    )
    args = SimpleNamespace(
        config=str(config_path),
        critic_ckpt=str(critic_path),
        epochs=1,
        lr=1e-3,
        pooling="attention",
        family_dim=2,
        function_dim=2,
        hidden_dim=8,
        out_dir=str(tmp_path / "runs" / "protein_ebm"),
        run_id="ebm-smoke",
        resume=None,
        seed=7,
    )

    result = train_ebm(args)

    run = tmp_path / "runs" / "ebm-smoke"
    payload = torch.load(
        run / "checkpoints" / "last_ebm.pt",
        map_location="cpu",
        weights_only=False,
    )
    assert result.status == "complete"
    assert payload["training_contract_version"] == 1
    assert payload["epoch"] == 1
    assert payload["best_epoch"] == 1
    assert payload["model"].keys() == payload["task"]["model"].keys()
    assert payload["critic_checkpoint"] == str(critic_path.resolve())
    assert (run / "checkpoints" / "ebm_epoch_1.pt").exists()
    assert (run / "checkpoints" / "best_ebm.pt").exists()
    assert (run / "summary.md").exists()
