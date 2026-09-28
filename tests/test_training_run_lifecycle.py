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


def _checkpoint(
    path,
    *,
    completed_epochs,
    current_epoch=0,
    microbatch=0,
    run_fingerprint=None,
):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "run_progress": {
                "completed_epochs": completed_epochs,
                "current_epoch": current_epoch,
                "microbatch": microbatch,
                "optimizer_step": 17,
            },
            "run_fingerprint": run_fingerprint,
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


def test_run_close_records_exception_before_releasing_lock(tmp_path):
    run = TrainingRun.open(tmp_path, "logged")

    with pytest.raises(RuntimeError, match="expected failure"):
        try:
            run.start_logging()
            raise RuntimeError("expected failure")
        finally:
            run.close()

    log_text = (run.logs / "train.log").read_text()
    assert "unhandled exception" in log_text
    assert "expected failure" in log_text
    reopened = TrainingRun(run.run_dir, None)
    reopened.close()


def test_resume_requires_newest_last_checkpoint(tmp_path):
    run = TrainingRun.open(tmp_path, "experiment")
    last = run.checkpoints / "last.pt"
    best = run.checkpoints / "best.pt"
    _checkpoint(last, completed_epochs=3)
    _checkpoint(best, completed_epochs=2)
    run.close()
    with pytest.raises(RunLifecycleError, match="newest last.pt"):
        TrainingRun.open(tmp_path, "experiment", resume=best, target_epochs=4)


def test_fork_accepts_best_checkpoint_and_records_lineage(tmp_path):
    source = TrainingRun.open(tmp_path, "source")
    best = source.checkpoints / "best.pt"
    _checkpoint(
        best,
        completed_epochs=2,
        current_epoch=2,
        run_fingerprint="source-fingerprint",
    )
    source.close()

    fork = TrainingRun.open(
        tmp_path,
        "forked",
        fork_from=best,
        target_epochs=4,
        config_fingerprint="fork-fingerprint",
    )

    assert fork.run_dir == tmp_path / "forked"
    assert fork.resume_checkpoint == best.resolve()
    assert fork.launch_mode == "fork"
    lineage = json.loads((fork.run_dir / "run_lineage.json").read_text())
    assert lineage["launch_mode"] == "fork"
    assert lineage["fork_run_id"] == "forked"
    assert lineage["source_checkpoint"] == str(best.resolve())
    assert lineage["source_run_id"] == "source"
    assert lineage["source_run_fingerprint"] == "source-fingerprint"
    assert lineage["fork_run_fingerprint"] == "fork-fingerprint"
    assert lineage["source_progress"]["completed_epochs"] == 2
    assert len(lineage["source_checkpoint_sha256"]) == 64
    fork.close()


def test_fork_requires_distinct_run_id(tmp_path):
    source = TrainingRun.open(tmp_path, "source")
    checkpoint = source.checkpoints / "best.pt"
    _checkpoint(checkpoint, completed_epochs=1)
    source.close()

    with pytest.raises(RunLifecycleError, match="requires a new run ID"):
        TrainingRun.open(
            tmp_path,
            "source",
            fork_from=checkpoint,
            target_epochs=2,
        )


def test_fork_and_resume_are_mutually_exclusive(tmp_path):
    checkpoint = tmp_path / "checkpoint.pt"
    _checkpoint(checkpoint, completed_epochs=1)
    with pytest.raises(RunLifecycleError, match="mutually exclusive"):
        TrainingRun.open(
            tmp_path,
            "forked",
            resume=checkpoint,
            fork_from=checkpoint,
            target_epochs=2,
        )


def test_fork_rejects_non_increasing_epoch_target(tmp_path):
    source = TrainingRun.open(tmp_path, "source")
    checkpoint = source.checkpoints / "best.pt"
    _checkpoint(checkpoint, completed_epochs=3)
    source.close()

    with pytest.raises(RunLifecycleError, match="3 completed epochs"):
        TrainingRun.open(
            tmp_path,
            "forked",
            fork_from=checkpoint,
            target_epochs=3,
        )


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
