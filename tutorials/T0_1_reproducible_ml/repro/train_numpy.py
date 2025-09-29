import numpy as np
from repro.tiny_model_numpy import LinearNP
from repro.logutil import get_logger

def train_numpy(seed=42, n=2048, d=16, epochs=10, batch=128, lr=1e-2):
    logger = get_logger("np")
    rng = np.random.default_rng(seed)

    # Deterministic synthetic data
    X = rng.normal(size=(n, d)).astype(np.float32)
    w_true = rng.normal(size=(d,1)).astype(np.float32)
    b_true = np.array([0.5], dtype=np.float32)
    y = X @ w_true + b_true

    model = LinearNP(d=d, seed=seed)
    for ep in range(epochs):
        # deterministic shuffle
        idx = rng.permutation(n)
        for s in range(0, n, batch):
            sl = idx[s:s+batch]
            model.step(X[sl], y[sl], lr)
        pred = model.forward(X)
        loss = model.mse(pred, y)
        logger.info(f"epoch {ep:02d} | loss {loss:.6f}")
    return model.W.copy(), model.b.copy()

if __name__ == "__main__":
    train_numpy()

