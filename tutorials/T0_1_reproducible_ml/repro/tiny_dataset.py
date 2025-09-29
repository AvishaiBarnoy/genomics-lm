import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class TinyLinearDataset(Dataset):
    # y = x @ w + b + noise
    def __init__(self, n=2048, d=16, seed=123, noise_std=0.0):
        rng = np.random.default_rng(seed)
        self.X = rng.normal(size=(n, d)).astype(np.float32)
        self.w = rng.normal(size=(d, 1)).astype(np.float32)
        self.b = np.array([[0.5]], dtype=np.float32)
        y = self.X @ self.w + self.b
        if noise_std > 0:
            y += rng.normal(scale=noise_std, size=y.shape).astype(np.float32)
        self.y = y

    def __len__(self): return self.X.shape[0]
    def __getitem__(self, i):
        return torch.from_numpy(self.X[i]), torch.from_numpy(self.y[i])

def make_loader(ds, batch_size, seed=123):
    # Deterministic DataLoader: workers=0 + seeded generator
    g = torch.Generator()
    g.manual_seed(seed)
    return DataLoader(
        ds, batch_size=batch_size, shuffle=True,
        num_workers=0, persistent_workers=False, generator=g, drop_last=False
    )

