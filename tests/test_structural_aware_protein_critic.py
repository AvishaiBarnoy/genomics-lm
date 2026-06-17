import json

import torch

from scripts.eval_multi_task_critic import threshold_metrics, top_fraction_enrichment
from scripts.prepare_protein_type_dataset import PROTEIN_TYPE_LABELS, protein_type_labels, row_to_sample
from src.protein_lm.config import ProteinClassifierConfig
from src.protein_lm.models_multi import MultiTaskProteinClassifier
from src.protein_lm.tokenizer import ProteinTokenizer
from src.protein_lm.train_multi_task import (
    LengthBucketBatchSampler,
    MultiTaskProteinDataset,
    collate_protein_batch,
    compute_multi_label_pos_weight,
    load_compatible_model_weights,
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


def test_transfer_loads_matching_backbone_and_skips_heads(tmp_path):
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
    source = MultiTaskProteinClassifier(config, {"family": 5})
    target = MultiTaskProteinClassifier(config, {"protein_type": 3})

    with torch.no_grad():
        source.backbone.token_embedding.weight.fill_(0.25)
        target.backbone.token_embedding.weight.zero_()

    checkpoint = tmp_path / "source.pt"
    torch.save(source.state_dict(), checkpoint)

    loaded, skipped = load_compatible_model_weights(target, checkpoint)

    assert loaded > 0
    assert "heads.family.weight" in skipped
    assert torch.allclose(target.backbone.token_embedding.weight, source.backbone.token_embedding.weight)


def test_compute_multi_label_pos_weight_from_dataset(tmp_path):
    tokenizer = ProteinTokenizer()
    data = tmp_path / "data.jsonl"
    rows = [
        {"sequence": "MKT", "protein_type": [1, 0, 0]},
        {"sequence": "MKT", "protein_type": [0, 0, 0]},
        {"sequence": "MKT", "protein_type": [0, 1, 0]},
        {"sequence": "MKT", "protein_type": [0, 1, 0]},
    ]
    data.write_text("".join(json.dumps(row) + "\n" for row in rows))
    dataset = MultiTaskProteinDataset(
        data,
        tokenizer,
        max_length=32,
        dynamic_padding=True,
        multi_label_tasks=["protein_type"],
    )

    weights = compute_multi_label_pos_weight(dataset, "protein_type", max_weight=10)

    assert torch.allclose(weights, torch.tensor([3.0, 1.0, 1.0]))


def test_eval_helpers_report_thresholds_and_enrichment():
    y_true = torch.tensor([1, 0, 1, 0, 0], dtype=torch.float32).numpy()
    y_prob = torch.tensor([0.95, 0.9, 0.4, 0.3, 0.1], dtype=torch.float32).numpy()

    thresholds = threshold_metrics(y_true, y_prob, [0.5, 0.2])
    enrichment = top_fraction_enrichment(y_true, y_prob, [0.4, 1.0])

    assert thresholds[0]["threshold"] == 0.5
    assert thresholds[0]["precision"] == 0.5
    assert thresholds[0]["recall"] == 0.5
    assert thresholds[0]["predicted_fraction"] == 0.4
    assert thresholds[1]["recall"] == 1.0

    assert enrichment[0]["k"] == 2
    assert enrichment[0]["positive_rate"] == 0.5
    assert abs(enrichment[0]["enrichment"] - 1.25) < 1e-6
    assert abs(enrichment[1]["positive_rate"] - 0.4) < 1e-6
