from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from .probes import compute_metrics


class MLP(nn.Module):
    def __init__(self, d_in: int, n_classes: int, hidden: int = 128, depth: int = 1, dropout: float = 0.1):
        super().__init__()
        layers = []
        last = d_in
        for i in range(depth):
            layers += [nn.Linear(last, hidden), nn.ReLU(), nn.Dropout(dropout)]
            last = hidden
        layers.append(nn.Linear(last, n_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@dataclass
class MLPResult:
    model: MLP
    metrics: Dict[str, float]
    y_pred: np.ndarray
    y_proba: np.ndarray


def fit_mlp(X: np.ndarray, y: np.ndarray, epochs: int = 20, lr: float = 1e-3, batch_size: int = 64, hidden: int = 128, depth: int = 1, dropout: float = 0.1, device: str = "auto") -> MLPResult:
    X = X.astype(np.float32)
    n_classes = int(np.max(y)) + 1
    d_in = X.shape[1]
    dev = torch.device("mps") if (device == "auto" and torch.backends.mps.is_available()) else torch.device("cpu") if device in {"auto", "cpu"} else torch.device(device)
    ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(y.astype(np.int64)))
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True)
    model = MLP(d_in, n_classes, hidden=hidden, depth=depth, dropout=dropout).to(dev)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    model.train()
    for _ in range(epochs):
        for xb, yb in dl:
            xb = xb.to(dev); yb = yb.to(dev)
            logits = model(xb)
            loss = criterion(logits, yb)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
    # Evaluate on training set
    model.eval()
    with torch.no_grad():
        X_t = torch.from_numpy(X).to(dev)
        logits = model(X_t)
        y_pred = torch.argmax(logits, dim=1).cpu().numpy()
        y_proba = torch.softmax(logits, dim=1).cpu().numpy()
    metrics = compute_metrics(y, y_pred, y_proba)
    return MLPResult(model, metrics, y_pred, y_proba)

