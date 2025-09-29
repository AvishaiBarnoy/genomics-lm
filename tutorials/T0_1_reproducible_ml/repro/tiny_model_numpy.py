import numpy as np

class LinearNP:
    def __init__(self, d, seed=0):
        rng = np.random.default_rng(seed)
        self.W = rng.normal(scale=0.02, size=(d,1)).astype(np.float32)
        self.b = np.zeros((1,), dtype=np.float32)

    def forward(self, X):  # (n,d) @ (d,1) + (1,)
        return X @ self.W + self.b

    def mse(self, pred, y):
        return np.mean((pred - y)**2)

    def step(self, X, y, lr):
        pred = self.forward(X)
        err = (pred - y)  # (n,1)
        dW = (X.T @ err) / X.shape[0]  # (d,1)
        db = np.mean(err, axis=0)
        self.W -= lr * dW
        self.b -= lr * db

