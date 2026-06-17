import numpy as np
import torch

from src.codonlm.train_codon_lm import (
    PackedDataset,
    dataset_length_audit,
    multi_offset_lm_loss,
    offset_target_mask,
)


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
