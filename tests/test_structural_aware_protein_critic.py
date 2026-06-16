import json

import torch

from scripts.prepare_protein_type_dataset import PROTEIN_TYPE_LABELS, protein_type_labels, row_to_sample
from src.protein_lm.config import ProteinClassifierConfig
from src.protein_lm.models_multi import MultiTaskProteinClassifier
from src.protein_lm.tokenizer import ProteinTokenizer
from src.protein_lm.train_multi_task import (
    LengthBucketBatchSampler,
    MultiTaskProteinDataset,
    collate_protein_batch,
)


def test_protein_type_label_extraction():
    row = {
        "Entry": "P0",
        "Sequence": "M" * 42,
        "Length": "42",
        "EC number": "1.1.1.1",
        "Keywords": "3D-structure;Membrane",
        "Features": "Signal (1); Transmembrane (1); Low complexity (1)",
        "Subcellular location [CC]": "SUBCELLULAR LOCATION: Secreted.",
    }

    labels = protein_type_labels(row)
    assert labels["structured_pdb"] == 1
    assert labels["membrane"] == 1
    assert labels["signal_secreted"] == 1
    assert labels["disordered_low_complexity"] == 1
    assert labels["enzyme"] == 1
    assert labels["short_peptide"] == 1
    assert labels["soluble_candidate"] == 0


def test_row_to_sample_multi_label_vector_order():
    sample = row_to_sample(
        {
            "Entry": "P1",
            "Entry Name": "MOCK",
            "Sequence": "M" * 80,
            "Length": "80",
            "Keywords": "3D-structure",
            "Features": "",
            "Subcellular location [CC]": "",
        },
        short_len=50,
    )

    assert sample is not None
    assert len(sample["protein_type"]) == len(PROTEIN_TYPE_LABELS)
    assert sample["protein_type"][PROTEIN_TYPE_LABELS.index("structured_pdb")] == 1
    assert sample["protein_type"][PROTEIN_TYPE_LABELS.index("soluble_candidate")] == 1


def test_dynamic_collate_pads_to_batch_max(tmp_path):
    tokenizer = ProteinTokenizer()
    data = tmp_path / "data.jsonl"
    rows = [
        {"sequence": "MKT", "protein_type": [1, 0]},
        {"sequence": "MKTWVV", "protein_type": [0, 1]},
    ]
    data.write_text("".join(json.dumps(row) + "\n" for row in rows))

    dataset = MultiTaskProteinDataset(
        data,
        tokenizer,
        max_length=32,
        dynamic_padding=True,
        multi_label_tasks=["protein_type"],
    )
    batch = collate_protein_batch([dataset[0], dataset[1]], tokenizer.pad_token_id)

    assert batch["input_ids"].shape == batch["attention_mask"].shape
    assert batch["input_ids"].shape[1] == dataset.sequence_length(1)
    assert batch["attention_mask"][0].sum().item() == dataset.sequence_length(0)
    assert batch["attention_mask"][1].sum().item() == dataset.sequence_length(1)
    assert batch["protein_type"].shape == (2, 2)


def test_length_bucket_sampler_groups_sorted_lengths(tmp_path):
    tokenizer = ProteinTokenizer()
    data = tmp_path / "data.jsonl"
    rows = [
        {"sequence": "M" * 10},
        {"sequence": "M" * 2},
        {"sequence": "M" * 8},
        {"sequence": "M" * 3},
    ]
    data.write_text("".join(json.dumps(row) + "\n" for row in rows))
    dataset = MultiTaskProteinDataset(data, tokenizer, max_length=32, dynamic_padding=True)

    batches = list(LengthBucketBatchSampler(dataset, batch_size=2, shuffle=False))
    lengths = [[dataset.sequence_length(i) for i in batch] for batch in batches]
    assert lengths == [sorted(lengths[0]), sorted(lengths[1])]
    assert max(lengths[0]) <= min(lengths[1])


def test_multitask_model_uses_attention_mask():
    config = ProteinClassifierConfig(
        vocab_size=len(ProteinTokenizer().vocab),
        block_size=8,
        n_layer=1,
        n_head=1,
        n_embd=8,
        dropout=0.0,
        num_classes=0,
        use_checkpoint=False,
    )
    model = MultiTaskProteinClassifier(config, {"protein_type": 3})
    input_ids = torch.ones((2, 8), dtype=torch.long)
    attention_mask = torch.tensor(
        [
            [1, 1, 1, 0, 0, 0, 0, 0],
            [1, 1, 1, 1, 1, 1, 1, 1],
        ],
        dtype=torch.long,
    )

    logits = model(input_ids, attention_mask=attention_mask)
    assert logits["protein_type"].shape == (2, 3)
