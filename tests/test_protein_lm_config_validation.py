from copy import deepcopy

import pytest

from src.protein_lm.config import ProteinLMConfig, validate_protein_lm_config
from src.protein_lm.tokenizer import ProteinTokenizer


def _valid(tmp_path):
    train = tmp_path / "train.jsonl"
    validation = tmp_path / "validation.jsonl"
    train.write_text('{"sequence":"MKT"}\n')
    validation.write_text('{"sequence":"MQQ"}\n')
    tokenizer = ProteinTokenizer()
    model = ProteinLMConfig(
        vocab_size=len(tokenizer.vocab),
        n_layer=1,
        n_head=2,
        n_embd=8,
        block_size=8,
        dropout=0.1,
    )
    config = {
        "training": {
            "epochs": 1,
            "batch_size": 2,
            "grad_accum_steps": 1,
            "lr": 1e-3,
            "num_workers": 0,
        },
        "data": {"train_path": str(train), "val_path": str(validation)},
    }
    return config, model, tokenizer


def test_valid_protein_lm_configuration_passes(tmp_path):
    config, model, tokenizer = _valid(tmp_path)
    validate_protein_lm_config(config, model, tokenizer)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("n_layer", 0, "positive"),
        ("block_size", 1, "at least 2"),
        ("dropout", 1.0, "in \\[0, 1\\)"),
    ],
)
def test_invalid_model_values_fail_early(tmp_path, field, value, message):
    config, model, tokenizer = _valid(tmp_path)
    setattr(model, field, value)
    with pytest.raises(ValueError, match=message):
        validate_protein_lm_config(config, model, tokenizer)


def test_attention_geometry_must_be_divisible(tmp_path):
    config, model, tokenizer = _valid(tmp_path)
    model.n_embd = 7
    with pytest.raises(ValueError, match="divisible"):
        validate_protein_lm_config(config, model, tokenizer)


def test_training_counts_and_paths_fail_before_run_allocation(tmp_path):
    config, model, tokenizer = _valid(tmp_path)
    invalid = deepcopy(config)
    invalid["training"]["batch_size"] = 1.5
    with pytest.raises(TypeError, match="batch_size must be an integer"):
        validate_protein_lm_config(invalid, model, tokenizer)

    invalid = deepcopy(config)
    invalid["data"]["train_path"] = str(tmp_path / "missing.jsonl")
    with pytest.raises(FileNotFoundError, match="train_path not found"):
        validate_protein_lm_config(invalid, model, tokenizer)
