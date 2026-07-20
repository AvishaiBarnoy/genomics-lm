import json
import subprocess
import sys

import torch

from src.codonlm.training.loop import dev


def test_explicit_cpu_device_never_auto_selects_accelerator():
    assert dev(requested="cpu") == torch.device("cpu")


def test_cpu_train_checkpoint_resume_preflight(tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.training_preflight",
            "--device",
            "cpu",
            "--work-dir",
            str(tmp_path),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads((tmp_path / "preflight_report.json").read_text())
    assert report["status"] == "passed"
    assert report["requested_device"] == report["actual_device"] == "cpu"
    assert report["initial"]["step"] == 2
    assert report["resumed"]["step"] == 4
    assert report["initial"]["consumed_train_tokens"] == 40
    assert report["resumed"]["consumed_train_tokens"] == 80
    assert report["resumed"]["scheduler_last_epoch"] == 4
    assert report["resumed"]["accumulation_health"] == {
        "active_microbatches": 0,
        "nonfinite_microbatches": 0,
        "aborted_groups": 0,
        "discarded_finite_microbatches": 0,
    }
    assert report["initial"]["dataset_manifest"]["dataset_id"] == report["dataset_id"]
    assert report["resumed"]["runtime_memory"]["process_max_rss_raw"] > 0
