"""Stage-2 classifier components for benchmarking LM representations.

Modules:
- probes: common utilities (I/O, metrics, plotting)
- linear_probe: logistic regression / linear SVM wrappers
- mlp_head: small MLP head on frozen embeddings (PyTorch)
- kmer_baselines: TF-IDF k-mer baselines + classical classifiers
"""

__all__ = [
    "probes",
    "linear_probe",
    "mlp_head",
    "kmer_baselines",
]

