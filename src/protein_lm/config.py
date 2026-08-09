from dataclasses import dataclass, fields
from numbers import Integral, Real
from pathlib import Path

import yaml

@dataclass
class ProteinLMConfig:
    """Configuration for the Protein Language Model."""
    vocab_size: int
    n_layer: int
    n_head: int
    n_embd: int
    block_size: int
    dropout: float


def validate_protein_lm_config(config, model_config, tokenizer) -> None:
    """Fail fast on invalid ProteinLM model, training, data, or tokenizer values."""

    for name in ("vocab_size", "n_layer", "n_head", "n_embd", "block_size"):
        value = getattr(model_config, name)
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise TypeError(f"model.{name} must be an integer")
        if value < 1:
            raise ValueError(f"model.{name} must be positive")
    if model_config.block_size < 2:
        raise ValueError("model.block_size must be at least 2 for shifted targets")
    if model_config.n_embd % model_config.n_head:
        raise ValueError("model.n_embd must be divisible by model.n_head")
    if isinstance(model_config.dropout, bool) or not isinstance(
        model_config.dropout, Real
    ):
        raise TypeError("model.dropout must be numeric")
    if not 0.0 <= float(model_config.dropout) < 1.0:
        raise ValueError("model.dropout must be in [0, 1)")

    if len(tokenizer.vocab) != model_config.vocab_size:
        raise ValueError("tokenizer vocabulary size must match model.vocab_size")
    special_ids = {
        "pad_token_id": tokenizer.pad_token_id,
        "bos_token_id": tokenizer.bos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if len(set(special_ids.values())) != len(special_ids):
        raise ValueError("tokenizer PAD, BOS, and EOS IDs must be distinct")
    for name, token_id in special_ids.items():
        if not isinstance(token_id, Integral) or not 0 <= token_id < len(tokenizer.vocab):
            raise ValueError(f"tokenizer {name} is outside the vocabulary")

    training = config.get("training", {})
    for name in ("epochs", "batch_size", "grad_accum_steps"):
        value = training.get(name, 1 if name == "grad_accum_steps" else None)
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise TypeError(f"training.{name} must be an integer")
        if value < 1:
            raise ValueError(f"training.{name} must be positive")
    workers = training.get("num_workers", 0)
    if isinstance(workers, bool) or not isinstance(workers, Integral):
        raise TypeError("training.num_workers must be an integer")
    if workers < 0:
        raise ValueError("training.num_workers must be non-negative")
    seed = training.get("seed", 1337)
    if isinstance(seed, bool) or not isinstance(seed, Integral):
        raise TypeError("training.seed must be an integer")
    for name in ("log_every_steps", "checkpoint_every_steps"):
        value = training.get(name, 0)
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise TypeError(f"training.{name} must be an integer")
        if value < 0:
            raise ValueError(f"training.{name} must be non-negative")
    for name in ("max_time_minutes", "checkpoint_every_minutes"):
        value = training.get(name)
        if value is not None:
            if isinstance(value, bool) or not isinstance(value, Real):
                raise TypeError(f"training.{name} must be numeric")
            if value < 0:
                raise ValueError(f"training.{name} must be non-negative")
    clip = training.get("grad_clip_norm")
    if clip is not None:
        if isinstance(clip, bool) or not isinstance(clip, Real):
            raise TypeError("training.grad_clip_norm must be numeric")
        if clip <= 0:
            raise ValueError("training.grad_clip_norm must be positive")

    data = config.get("data", {})
    for name in ("train_path", "val_path"):
        value = data.get(name)
        if not isinstance(value, (str, Path)) or not str(value):
            raise ValueError(f"data.{name} is required")
        path = Path(value)
        if not path.is_file():
            raise FileNotFoundError(f"data.{name} not found: {path}")
        if path.stat().st_size == 0:
            raise ValueError(f"data.{name} is empty: {path}")

@dataclass
class ProteinClassifierConfig:
    """Configuration for the Protein Classifier."""
    vocab_size: int
    n_layer: int
    n_head: int
    n_embd: int
    block_size: int
    dropout: float
    num_classes: int
    use_checkpoint: bool = False
    pooling: str = "mean"
    bidirectional: bool = True

def load_config(path: str, config_class, overrides=None):
    """
    Loads a model configuration from a YAML file.

    Args:
        path: The path to the YAML file.
        config_class: The dataclass to instantiate (e.g., ProteinLMConfig).

    Returns:
        An instance of the provided config_class.
    """
    with open(path, 'r') as f:
        data = yaml.safe_load(f)

    # The configuration is expected to be under a 'model' key in the YAML
    model_data = data.get('model', {})

    # Filter the loaded data to include only the fields expected by the dataclass
    expected_fields = {f.name for f in fields(config_class)}
    filtered_data = {k: v for k, v in model_data.items() if k in expected_fields}
    filtered_data.update(overrides or {})

    return config_class(**filtered_data)
