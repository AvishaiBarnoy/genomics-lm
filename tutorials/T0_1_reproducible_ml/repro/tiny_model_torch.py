import torch, torch.nn as nn

class LinearTorch(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.lin = nn.Linear(d, 1, bias=True)

    def forward(self, x):
        return self.lin(x)

