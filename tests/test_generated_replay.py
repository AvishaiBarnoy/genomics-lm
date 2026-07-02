import json

import torch

from src.codonlm.replay import GeneratedTerminationReplayDataset


def test_generated_replay_dataset_sparse_labels(tmp_path):
    path = tmp_path / "replay.jsonl"
    record = {
        "ids": [10, 11, 12, 13],
        "labels": [{"pos": 2, "class": 1}, {"pos": 3, "class": 0}],
    }
    path.write_text(json.dumps(record) + "\n")

    ds = GeneratedTerminationReplayDataset(path, block_size=6, pad_id=0)

    x, y = ds[0]
    assert x.tolist() == [10, 11, 12, 13, 0, 0]
    assert y.tolist() == [-100, -100, 1, 0, -100, -100]


def test_generated_replay_dataset_left_clips_positions(tmp_path):
    path = tmp_path / "replay.jsonl"
    record = {
        "ids": [1, 2, 3, 4, 5, 6],
        "labels": [{"pos": 1, "class": 1}, {"pos": 4, "class": 0}],
    }
    path.write_text(json.dumps(record) + "\n")

    ds = GeneratedTerminationReplayDataset(path, block_size=3, pad_id=0)

    x, y = ds[0]
    assert x.tolist() == [4, 5, 6]
    assert y.tolist() == [-100, 0, -100]


def test_generated_replay_dataset_supports_legacy_single_label(tmp_path):
    path = tmp_path / "replay.jsonl"
    record = {"ids": [1, 2, 3], "label_position": 2, "target_class": 0}
    path.write_text(json.dumps(record) + "\n")

    ds = GeneratedTerminationReplayDataset(path, block_size=3)
    _x, y = ds[0]

    assert torch.equal(y, torch.tensor([-100, -100, 0]))

