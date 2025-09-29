import os, torch

def enable_full_determinism(raise_on_nondet: bool = True):
    # 1) Torch deterministic algorithms
    torch.use_deterministic_algorithms(True, warn_only=not raise_on_nondet)

    # 2) Threading (helps reproducibility for BLAS/reductions)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"

    # 3) Backend-specific toggles
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True  # N/A on MPS but harmless

    # 4) Safe MPS fallback (let ops fall back to CPU deterministically if needed)
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

