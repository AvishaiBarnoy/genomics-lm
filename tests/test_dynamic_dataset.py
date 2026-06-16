import numpy as np
import torch
from torch.utils.data import DataLoader
from src.codonlm.train_codon_lm import PackedDataset


def test_dynamic_dataset_loading_and_collate(tmp_path):
    # Create two dummy variable-length sequences
    seq1 = np.array([1, 10, 20, 2], dtype=np.int32)
    seq2 = np.array([1, 15, 25, 35, 45, 2], dtype=np.int32)

    flat_X = np.concatenate([seq1, seq2])
    lengths = np.array([len(seq1), len(seq2)], dtype=np.int32)

    npz_path = tmp_path / "dynamic_test.npz"
    np.savez_compressed(npz_path, X=flat_X, lengths=lengths)

    # Load dataset
    ds = PackedDataset(npz_path)
    assert ds.is_dynamic is True
    assert len(ds) == 2
    assert torch.equal(ds[0], torch.tensor([1, 10, 20, 2], dtype=torch.long))
    assert torch.equal(ds[1], torch.tensor([1, 15, 25, 35, 45, 2], dtype=torch.long))

    # Setup dynamic_collate_fn
    def dynamic_collate_fn(batch):
        lengths = [len(seq) for seq in batch]
        max_len = max(lengths)
        xs, ys = [], []
        for seq in batch:
            x_seq = seq[:-1]
            y_seq = seq[1:]
            pad_len = (max_len - 1) - len(x_seq)
            if pad_len > 0:
                x_seq = torch.cat([x_seq, torch.zeros(pad_len, dtype=torch.long)])
                y_seq = torch.cat([y_seq, torch.zeros(pad_len, dtype=torch.long)])
            xs.append(x_seq)
            ys.append(y_seq)
        return torch.stack(xs), torch.stack(ys)

    loader = DataLoader(ds, batch_size=2, shuffle=False, collate_fn=dynamic_collate_fn)

    for xb, yb in loader:
        # max_len of batch is 6. max_len - 1 is 5.
        # x for seq1: [1, 10, 20] -> pad to 5 -> [1, 10, 20, 0, 0]
        # y for seq1: [10, 20, 2] -> pad to 5 -> [10, 20, 2, 0, 0]
        # x for seq2: [1, 15, 25, 35, 45] -> length 5
        # y for seq2: [15, 25, 35, 45, 2] -> length 5

        assert xb.shape == (2, 5)
        assert yb.shape == (2, 5)

        assert torch.equal(xb[0], torch.tensor([1, 10, 20, 0, 0], dtype=torch.long))
        assert torch.equal(yb[0], torch.tensor([10, 20, 2, 0, 0], dtype=torch.long))
        assert torch.equal(xb[1], torch.tensor([1, 15, 25, 35, 45], dtype=torch.long))
        assert torch.equal(yb[1], torch.tensor([15, 25, 35, 45, 2], dtype=torch.long))
