import numpy as np
import torch

from src.codonlm.data_loading import PackedDataset, dataset_length_audit
from src.codonlm.training.objectives import (
    multi_offset_lm_loss,
    offset_target_mask,
    termination_aux_loss,
    termination_distance_bucket_labels,
)
from src.codonlm.model_tiny_gpt import TinyGPT
from src.codonlm.generate import _apply_multi_offset_priors


def test_offset_target_mask_blocks_boundaries_before_target():
    # yb contains future tokens relative to each input position.
    yb = torch.tensor(
        [
            [10, 11, 12, 13, 0],
            [10, 2, 12, 13, 0],
            [10, 11, 3, 13, 0],
        ],
        dtype=torch.long,
    )

    mask = offset_target_mask(yb, offset=4, boundary_ids=(2, 3))

    assert mask.tolist() == [
        [True, False],
        [False, False],
        [False, False],
    ]


def test_multi_offset_lm_loss_skips_offsets_without_valid_targets():
    logits = torch.randn(2, 4, 16)
    yb = torch.zeros((2, 4), dtype=torch.long)

    total, losses = multi_offset_lm_loss(logits, yb, {4: 0.1})

    assert total.item() == 0.0
    assert losses == {}


def test_multi_offset_lm_loss_uses_shifted_targets():
    logits = torch.full((1, 5, 8), -10.0)
    yb = torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.long)
    logits[0, 0, 4] = 10.0

    total, losses = multi_offset_lm_loss(logits, yb, {4: 1.0}, boundary_ids=())

    assert 4 in losses
    assert torch.isclose(total, losses[4])


def test_dataset_length_audit_reports_dynamic_clipped_fraction(tmp_path):
    flat = np.array([1, 2, 3, 1, 2, 3, 4, 5], dtype=np.int32)
    lengths = np.array([3, 5], dtype=np.int32)
    npz = tmp_path / "train_bs5.npz"
    np.savez_compressed(npz, X=flat, lengths=lengths)

    dataset = PackedDataset(npz)
    audit = dataset_length_audit(dataset, block_size=5)

    assert audit["mode"] == "dynamic"
    assert audit["n_sequences"] == 2
    assert audit["max"] == 5
    assert audit["at_block_size"] == 1
    assert audit["at_block_size_frac"] == 0.5


def test_termination_distance_bucket_labels():
    yb = torch.tensor(
        [
            [10, 11, 2, 12, 0],
            [2, 10, 11, 12, 13],
            [10, 11, 12, 13, 14],
        ],
        dtype=torch.long,
    )

    labels = termination_distance_bucket_labels(
        yb,
        stop_ids=(2,),
        bucket_edges=(0, 1, 3),
    )

    assert labels.tolist() == [
        [2, 1, 0, 3, -100],
        [0, 3, 3, 3, 3],
        [3, 3, 3, 3, 3],
    ]


def _reference_termination_labels(yb, stop_ids, bucket_edges, ignore_index=-100):
    labels = torch.full_like(yb, fill_value=ignore_index, dtype=torch.long)
    n_classes = len(bucket_edges) + 1
    for row_idx, row in enumerate(yb):
        stop_positions = [
            pos for pos, token in enumerate(row.tolist()) if token in stop_ids
        ]
        for pos, token in enumerate(row.tolist()):
            if token == 0:
                continue
            future_stops = [stop for stop in stop_positions if stop >= pos]
            if not future_stops:
                labels[row_idx, pos] = n_classes - 1
                continue
            distance = future_stops[0] - pos
            labels[row_idx, pos] = sum(
                distance > edge for edge in bucket_edges
            )
    return labels


def test_vectorized_termination_labels_match_reference():
    generator = torch.Generator().manual_seed(1337)
    for shape in ((1, 1), (2, 17), (4, 512)):
        yb = torch.randint(0, 12, shape, generator=generator)
        for stop_ids in ((2,), (2, 3)):
            for bucket_edges in ((), (0,), (0, 3, 10, 30)):
                expected = _reference_termination_labels(
                    yb,
                    stop_ids=stop_ids,
                    bucket_edges=bucket_edges,
                )
                actual = termination_distance_bucket_labels(
                    yb,
                    stop_ids=stop_ids,
                    bucket_edges=bucket_edges,
                )
                assert torch.equal(actual, expected)


def test_termination_aux_loss_accepts_labels():
    logits = torch.randn(2, 4, 5)
    labels = torch.tensor(
        [
            [0, 1, 4, -100],
            [4, 3, 2, 1],
        ],
        dtype=torch.long,
    )

    loss = termination_aux_loss(logits, labels)

    assert torch.isfinite(loss)


def test_termination_aux_loss_accepts_class_weights():
    logits = torch.zeros((1, 2, 2), dtype=torch.float32)
    labels = torch.tensor([[0, 1]], dtype=torch.long)

    loss = termination_aux_loss(
        logits,
        labels,
        class_weights=torch.tensor([4.0, 1.0]),
    )

    assert torch.isfinite(loss)


def test_multi_offset_projection_heads():
    # Instantiate model with multi-offset targets
    model = TinyGPT(
        vocab_size=16,
        block_size=32,
        n_layer=2,
        n_head=2,
        n_embd=64,
        multi_offset_targets=[4, 8]
    )

    assert hasattr(model, "offset_projs")
    assert "4" in model.offset_projs
    assert "8" in model.offset_projs

    # Test identity initialization of projection weights in both MLP layers
    eye_weight = torch.eye(64)
    assert torch.allclose(model.offset_projs["4"][0].weight, eye_weight)
    assert torch.allclose(model.offset_projs["4"][2].weight, eye_weight)
    assert torch.allclose(model.offset_projs["8"][0].weight, eye_weight)
    assert torch.allclose(model.offset_projs["8"][2].weight, eye_weight)
    if model.offset_projs["4"][0].bias is not None:
        assert torch.allclose(model.offset_projs["4"][0].bias, torch.zeros(64))
        assert torch.allclose(model.offset_projs["4"][2].bias, torch.zeros(64))

    # Test forward pass returns offset logits in aux
    idx = torch.randint(1, 16, (2, 10))
    logits, loss, aux = model(idx, return_aux=True)

    assert "offset_logits" in aux
    assert 4 in aux["offset_logits"]
    assert 8 in aux["offset_logits"]
    assert aux["offset_logits"][4].shape == (2, 10, 16)
    assert aux["offset_logits"][8].shape == (2, 10, 16)

    # Test multi_offset_lm_loss with dictionary logits
    yb = torch.randint(1, 16, (2, 10))
    total_loss, offset_losses = multi_offset_lm_loss(
        logits=aux["offset_logits"],
        yb=yb,
        offset_weights={4: 0.1, 8: 0.05},
        boundary_ids=()
    )
    assert 4 in offset_losses
    assert 8 in offset_losses
    assert total_loss.item() > 0.0


def test_apply_multi_offset_priors_guided_decoding():
    logits = torch.randn(16)
    aux = {
        "offset_logits": {
            4: torch.full((1, 5, 16), 1.5),
            8: torch.full((1, 5, 16), -2.0)
        }
    }

    # Let ctx_len be 5.
    # Prior for offset 4 is predicted by index ctx_len - 4 = 5 - 4 = 1.
    # Prior for offset 8 is predicted by index ctx_len - 8 = 5 - 8 = -3 (which is < 0, so ignored).
    # Therefore, only offset 4's prior should be added: 0.1 * 1.5 = 0.15
    modified_logits = _apply_multi_offset_priors(
        logits=logits,
        aux=aux,
        ctx_len=5,
        offsets=[4, 8],
        weights={4: 0.1, 8: 0.5}
    )
    expected = logits + 0.15
    assert torch.allclose(modified_logits, expected)


def test_backbone_freezing():
    from src.codonlm.model_tiny_gpt import TinyGPT

    model = TinyGPT(
        vocab_size=16,
        block_size=32,
        n_layer=2,
        n_head=2,
        n_embd=64,
        multi_offset_targets=[4, 8],
        termination_aux=True
    )

    # Simulate backbone freezing
    cfg = {"freeze_backbone": True}
    freeze_backbone = bool(cfg.get("freeze_backbone", False))
    if freeze_backbone:
        for name, param in model.named_parameters():
            if "offset_projs" in name or "termination_head" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

    # Check that transformer weights are frozen
    assert model.tok_emb.weight.requires_grad is False
    assert model.blocks[0].attn.proj.weight.requires_grad is False
    assert model.head.weight.requires_grad is False

    # Check that offset projections and termination head are trainable
    assert model.termination_head.weight.requires_grad is True
    assert model.offset_projs["4"][0].weight.requires_grad is True
    assert model.offset_projs["4"][2].weight.requires_grad is True
    assert model.offset_projs["8"][0].weight.requires_grad is True
    assert model.offset_projs["8"][2].weight.requires_grad is True
