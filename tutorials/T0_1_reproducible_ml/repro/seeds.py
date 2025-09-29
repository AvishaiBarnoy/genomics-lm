import os, random
import numpy as np
import torch

def set_global_seed(seed: int, mps: bool = False):
    os.environ["PYTHONHASHSEED"] = str(seed)   # hash randomization
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if mps and torch.backends.mps.is_available():
        # MPS shares the CPU RNG; torch.manual_seed covers it, but keep flag for clarity
        pass

