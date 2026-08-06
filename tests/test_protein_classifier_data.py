import json

import pytest

from src.protein_lm.data import ProteinClassificationDataset
from src.protein_lm.tokenizer import ProteinTokenizer


def _write_jsonl(path, records):
    path.write_text("".join(json.dumps(record) + "\n" for record in records))


def test_validation_reuses_training_label_ids(tmp_path):
    train_path = tmp_path / "train.jsonl"
    val_path = tmp_path / "val.jsonl"
    _write_jsonl(
        train_path,
        [
            {"sequence": "ACD", "func_label": "alpha"},
            {"sequence": "EFG", "func_label": "beta"},
        ],
    )
    _write_jsonl(val_path, [{"sequence": "HIK", "func_label": "beta"}])
    tokenizer = ProteinTokenizer()

    train = ProteinClassificationDataset(
        str(train_path), tokenizer, block_size=8, label_field="func_label"
    )
    validation = ProteinClassificationDataset(
        str(val_path),
        tokenizer,
        block_size=8,
        label_field="func_label",
        label_map=train.label_map,
    )

    assert train.label_map == {"alpha": 0, "beta": 1}
    assert validation[0][1].item() == 1
    assert validation[0][0][1].item() == tokenizer.token_to_id["H"]


def test_validation_rejects_labels_unseen_during_training(tmp_path):
    path = tmp_path / "validation.jsonl"
    _write_jsonl(path, [{"sequence": "ACD", "func_label": "unknown"}])

    with pytest.raises(ValueError, match="absent from the training label map"):
        ProteinClassificationDataset(
            str(path),
            ProteinTokenizer(),
            block_size=8,
            label_field="func_label",
            label_map={"known": 0},
        )
