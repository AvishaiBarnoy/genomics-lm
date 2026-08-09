import copy

import torch.nn.functional as F
import torch

from src.codonlm.model_tiny_gpt import NoPropTinyGPT, NoPropBlock
from src.codonlm.noprop_task import (
    NoPropTask,
    NoPropUpdateStrategy,
    adapt_noprop_checkpoint,
    decode_noprop_checkpoint,
)
from src.training.engine import EngineConfig, TrainingEngine
from src.training.contracts import TrainingPhase
from src.training.run_lifecycle import TrainingRun

def test_noprop_initialization():
    vocab_size = 64
    block_size = 32
    n_layer = 3
    n_head = 2
    n_embd = 32

    model = NoPropTinyGPT(
        vocab_size=vocab_size,
        block_size=block_size,
        n_layer=n_layer,
        n_head=n_head,
        n_embd=n_embd,
        sep_id=3
    )

    assert len(model.blocks) == n_layer
    assert isinstance(model.blocks[0], NoPropBlock)
    assert model.tok_emb.weight.shape == (vocab_size, n_embd)

def test_noprop_gradient_isolation():
    vocab_size = 64
    block_size = 10
    n_layer = 3
    n_head = 2
    n_embd = 32

    model = NoPropTinyGPT(
        vocab_size=vocab_size,
        block_size=block_size,
        n_layer=n_layer,
        n_head=n_head,
        n_embd=n_embd
    )

    # Mock inputs and targets
    idx = torch.randint(1, vocab_size, (2, block_size))
    targets = torch.randint(1, vocab_size, (2, block_size))

    # Forward embeddings
    pos = torch.arange(0, block_size).unsqueeze(0)
    h = model.tok_emb(idx) + model.pos_emb(pos)

    y_clean = model.tok_emb(targets)
    noise = torch.randn_like(y_clean) * 0.1
    y_noisy = y_clean + noise

    # Check block 1 backprop isolation
    # Forward block 0
    h0_out, pred_y0 = model.blocks[0](h, noisy_targets=y_noisy)

    # Forward block 1 (with detached input from block 0)
    h1_in = h0_out.detach()
    h1_out, pred_y1 = model.blocks[1](h1_in, noisy_targets=y_noisy)

    # Compute local MSE loss for block 1
    loss_b1 = F.mse_loss(pred_y1, y_clean)
    loss_b1.backward()

    # Assertions:
    # 1. Block 1 parameters must have gradients.
    # 2. Block 0 and Block 2 parameters must NOT have gradients (grad is None).
    for name, param in model.blocks[1].named_parameters():
        assert param.grad is not None, f"Block 1 param {name} should have gradients."
        assert param.grad.abs().sum() > 0

    for name, param in model.blocks[0].named_parameters():
        assert param.grad is None, f"Block 0 param {name} should have NO gradients (isolated)."

    for name, param in model.blocks[2].named_parameters():
        assert param.grad is None, f"Block 2 param {name} should have NO gradients (isolated)."

def test_noprop_inference_forward():
    vocab_size = 64
    block_size = 10
    model = NoPropTinyGPT(vocab_size=vocab_size, block_size=block_size, n_layer=2, n_head=2, n_embd=16)
    model.eval()

    idx = torch.randint(1, vocab_size, (2, block_size))
    with torch.no_grad():
        logits, preds = model(idx)

    assert logits.shape == (2, block_size, vocab_size)
    assert len(preds) == 2
    assert preds[0].shape == (2, block_size, 16)


def _small_model():
    return NoPropTinyGPT(
        vocab_size=16,
        block_size=5,
        n_layer=2,
        n_head=1,
        n_embd=8,
        dropout=0.0,
        sep_id=None,
    )


def _optimizers(model):
    lr = 1e-3
    embedding = torch.optim.AdamW(
        [*model.tok_emb.parameters(), *model.pos_emb.parameters()], lr=lr
    )
    blocks = [torch.optim.AdamW(block.parameters(), lr=lr) for block in model.blocks]
    head = torch.optim.AdamW(
        [*model.ln_f.parameters(), *model.head.parameters()], lr=lr
    )
    return embedding, blocks, head


def _task_and_strategy(model, batches, *, sigma=0.1):
    embedding, blocks, head = _optimizers(model)
    task = NoPropTask(
        model=model,
        train_loader=batches,
        validation_loader=batches[:1],
        device=torch.device("cpu"),
        noise_sigma=sigma,
        seed=17,
    )
    return task, NoPropUpdateStrategy(embedding, blocks, head)


def _assert_nested_equal(actual, expected):
    if torch.is_tensor(expected):
        assert torch.equal(actual, expected)
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            _assert_nested_equal(actual[key], expected[key])
    elif isinstance(expected, list):
        assert len(actual) == len(expected)
        for actual_item, expected_item in zip(actual, expected, strict=True):
            _assert_nested_equal(actual_item, expected_item)
    else:
        assert actual == expected


def test_noprop_shared_engine_matches_direct_local_update(tmp_path):
    batches = [(torch.tensor([[1, 2, 3, 4, 5]]), torch.tensor([[2, 3, 4, 5, 6]]))]
    torch.manual_seed(7)
    engine_model = _small_model()
    direct_model = copy.deepcopy(engine_model)
    engine_task, strategy = _task_and_strategy(engine_model, batches)
    direct_task, direct_strategy = _task_and_strategy(direct_model, batches)

    torch.manual_seed(29)
    direct_strategy.begin_group(1)
    direct_task.begin_phase(TrainingPhase.TRAIN, 0)
    direct_strategy.process_microbatch(direct_task, batches[0], None)
    direct_strategy.commit_group()

    torch.manual_seed(29)
    run = TrainingRun.open(tmp_path, "noprop-parity")
    engine = TrainingEngine(
        task=engine_task,
        strategy=strategy,
        run=run,
        config=EngineConfig(epochs=1),
        device=torch.device("cpu"),
    )
    result = engine.fit()
    run.close()

    assert result.state.optimizer_step == 1
    for actual, expected in zip(engine_model.parameters(), direct_model.parameters()):
        assert torch.allclose(actual, expected)


class _ExpireOnce:
    def __init__(self):
        self.calls = 0

    def expired(self):
        self.calls += 1
        return self.calls == 1


def test_noprop_interrupted_resume_matches_uninterrupted(tmp_path):
    batches = [
        (torch.tensor([[1, 2, 3, 4, 5]]), torch.tensor([[2, 3, 4, 5, 6]])),
        (torch.tensor([[6, 7, 8, 9, 10]]), torch.tensor([[7, 8, 9, 10, 11]])),
    ]
    torch.manual_seed(41)
    initial = copy.deepcopy(_small_model().state_dict())

    def build(run, timer=None):
        model = _small_model()
        model.load_state_dict(initial)
        task, strategy = _task_and_strategy(model, batches)
        return model, TrainingEngine(
            task=task,
            strategy=strategy,
            run=run,
            config=EngineConfig(epochs=1),
            device=torch.device("cpu"),
            wall_timer=timer,
            checkpoint_decoder=decode_noprop_checkpoint,
            checkpoint_payload_adapter=adapt_noprop_checkpoint,
        )

    torch.manual_seed(53)
    reference_run = TrainingRun.open(tmp_path, "noprop-reference")
    reference_model, reference_engine = build(reference_run)
    reference_engine.fit()
    expected = copy.deepcopy(reference_model.state_dict())
    reference_checkpoint = torch.load(
        reference_run.checkpoints / "last.pt", map_location="cpu", weights_only=False
    )
    reference_run.close()

    torch.manual_seed(53)
    interrupted_run = TrainingRun.open(tmp_path, "noprop-resume")
    _, interrupted_engine = build(interrupted_run, _ExpireOnce())
    result = interrupted_engine.fit()
    assert result.status == "interrupted"
    checkpoint = interrupted_run.checkpoints / "last.pt"
    interrupted_run.close()

    resumed_run = TrainingRun.open(
        tmp_path, "noprop-resume", resume=checkpoint, target_epochs=1
    )
    resumed_model, resumed_engine = build(resumed_run)
    resumed = resumed_engine.fit()
    resumed_run.close()

    assert resumed.status == "complete"
    assert resumed.state.optimizer_step == 2
    for name, tensor in resumed_model.state_dict().items():
        assert torch.allclose(tensor, expected[name])
    resumed_checkpoint = torch.load(
        resumed_run.checkpoints / "last.pt", map_location="cpu", weights_only=False
    )
    _assert_nested_equal(
        resumed_checkpoint["strategy"]["optimizers"],
        reference_checkpoint["strategy"]["optimizers"],
    )


def test_noprop_legacy_checkpoint_translation_preserves_optimizer_groups():
    model = _small_model()
    embedding, blocks, head = _optimizers(model)
    legacy = {
        "model": model.state_dict(),
        "optimizers": {
            "embedding": embedding.state_dict(),
            "blocks": [optimizer.state_dict() for optimizer in blocks],
            "head": head.state_dict(),
        },
        "epoch": 2,
        "best_val_loss": 1.5,
        "rng_state": {},
    }
    checkpoint = decode_noprop_checkpoint(legacy)

    assert checkpoint.engine.completed_epochs == 2
    assert checkpoint.metadata["best_metric"] == 1.5
    assert len(checkpoint.strategy["optimizers"]["blocks"]) == 2
