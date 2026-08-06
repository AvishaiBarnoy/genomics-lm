import json
import random

import numpy as np
import pytest
import torch

from src.training.run_lifecycle import (
    RunLifecycleError,
    TrainingRun,
    capture_rng_state,
    configuration_fingerprint,
    restore_rng_state,
)


def _checkpoint(path, *, completed_epochs, current_epoch=0, microbatch=0):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "run_progress": {
                "completed_epochs": completed_epochs,
                "current_epoch": current_epoch,
                "microbatch": microbatch,
                "optimizer_step": 17,
            }
        },
        path,
    )


def test_fresh_duplicate_run_allocates_serial_directory(tmp_path):
    first = TrainingRun.open(tmp_path, "experiment")
    first.close()
    second = TrainingRun.open(tmp_path, "experiment")
    assert first.run_dir.name == "experiment"
    assert second.run_dir.name == "experiment-r002"
    second.close()


def test_active_run_lock_rejects_second_writer(tmp_path):
    run = TrainingRun.open(tmp_path, "experiment")
    checkpoint = run.checkpoints / "last.pt"
    _checkpoint(checkpoint, completed_epochs=1)
    with pytest.raises(RunLifecycleError, match="already locked"):
        TrainingRun.open(
            tmp_path, "experiment", resume=checkpoint, target_epochs=2
        )
    run.close()


def test_resume_requires_newest_last_checkpoint(tmp_path):
    run = TrainingRun.open(tmp_path, "experiment")
    last = run.checkpoints / "last.pt"
    best = run.checkpoints / "best.pt"
    _checkpoint(last, completed_epochs=3)
    _checkpoint(best, completed_epochs=2)
    run.close()
    with pytest.raises(RunLifecycleError, match="newest last.pt"):
        TrainingRun.open(tmp_path, "experiment", resume=best, target_epochs=4)


def test_resume_rejects_non_increasing_epoch_target(tmp_path):
    run = TrainingRun.open(tmp_path, "experiment")
    last = run.checkpoints / "last.pt"
    _checkpoint(last, completed_epochs=5)
    run.close()
    with pytest.raises(RunLifecycleError, match="5 completed epochs"):
        TrainingRun.open(tmp_path, "experiment", resume=last, target_epochs=5)


def test_completed_run_rejects_equal_target_and_allows_extension(tmp_path):
    run = TrainingRun.open(tmp_path, "experiment")
    last = run.checkpoints / "last.pt"
    _checkpoint(last, completed_epochs=2)
    run.mark_complete({"completed_epochs": 2})
    run.close()
    assert json.loads((run.run_dir / "run_complete.json").read_text())["status"] == "complete"
    with pytest.raises(RunLifecycleError, match="2 completed epochs"):
        TrainingRun.open(tmp_path, "experiment", resume=last, target_epochs=2)
    resumed = TrainingRun.open(tmp_path, "experiment", resume=last, target_epochs=3)
    assert not resumed.completion_path.exists()
    assert (run.run_dir / "run_complete_epoch_002.json").exists()
    resumed.close()


def test_rng_state_round_trip():
    random.seed(7)
    np.random.seed(7)
    torch.manual_seed(7)
    state = capture_rng_state()
    expected = (random.random(), np.random.random(), torch.rand(1))
    restore_rng_state(state)
    actual = (random.random(), np.random.random(), torch.rand(1))
    assert actual[0] == expected[0]
    assert actual[1] == expected[1]
    assert torch.equal(actual[2], expected[2])


def test_resume_rejects_duplicate_curve_history(tmp_path):
    run = TrainingRun.open(tmp_path, "experiment")
    last = run.checkpoints / "last.pt"
    _checkpoint(last, completed_epochs=2)
    (run.scores / "curves.csv").write_text(
        "epoch,train_loss,val_loss\n1,2,3\n1,2,3\n"
    )
    run.close()
    with pytest.raises(RunLifecycleError, match="duplicate or decreasing"):
        TrainingRun.open(tmp_path, "experiment", resume=last, target_epochs=3)


def test_configuration_fingerprint_ignores_operational_settings():
    baseline = {"n_layer": 8, "lr": 1e-4, "epochs": 10, "max_time_minutes": 30}
    operational_change = {
        **baseline,
        "epochs": 15,
        "max_time_minutes": 60,
        "checkpoint_every_minutes": 5,
    }
    assert configuration_fingerprint(baseline) == configuration_fingerprint(
        operational_change
    )
    assert configuration_fingerprint(baseline) != configuration_fingerprint(
        {**baseline, "lr": 2e-4}
    )
    assert configuration_fingerprint(
        {"training": {"epochs": 10, "lr": 1e-4}}
    ) == configuration_fingerprint(
        {"training": {"epochs": 20, "lr": 1e-4}}
    )
