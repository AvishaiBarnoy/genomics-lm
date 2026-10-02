import fcntl
import json
from pathlib import Path

import pytest

from scripts.check_ci import summarize_ci
from src.training.inspection.runs import inspect_run
from src.training.inspection.reporting import render_markdown


def test_ci_missing_is_not_success():
    report = summarize_ci(
        {"headRefOid": "abc", "statusCheckRollup": []}, ["lint", "core-tests"]
    )
    assert report["status"] == "missing"
    assert report["missing_checks"] == ["core-tests", "lint"]


def test_ci_pending_failure_and_cancelled():
    base = {
        "headRefOid": "abc",
        "statusCheckRollup": [
            {"name": "lint", "status": "COMPLETED", "conclusion": "SUCCESS"},
            {"name": "core-tests", "status": "IN_PROGRESS", "conclusion": ""},
        ],
    }
    assert summarize_ci(base, ["lint", "core-tests"])["status"] == "pending"
    base["statusCheckRollup"][1].update(status="COMPLETED", conclusion="FAILURE")
    assert summarize_ci(base, ["lint", "core-tests"])["status"] == "failed"
    base["statusCheckRollup"][1]["conclusion"] = "CANCELLED"
    assert summarize_ci(base, ["lint", "core-tests"])["status"] == "cancelled"


def test_missing_run_not_created(tmp_path):
    path = tmp_path / "absent"
    with pytest.raises(ValueError, match="does not exist"):
        inspect_run(path)
    assert not path.exists()


def test_stale_lock_file_does_not_prove_running(tmp_path):
    (tmp_path / ".run.lock").write_text("pid=123\n")
    assert inspect_run(tmp_path)["status"] == "incomplete"


def test_live_lock_overrides_old_completion(tmp_path):
    (tmp_path / "run_complete.json").write_text('{"status":"complete"}')
    with (tmp_path / ".run.lock").open("w") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert inspect_run(tmp_path)["status"] == "active"


def test_completed_run_and_metrics(tmp_path):
    (tmp_path / "run_complete.json").write_text(
        '{"status":"complete","completed_epochs":3}'
    )
    scores = tmp_path / "scores"
    scores.mkdir()
    (scores / "curves.csv").write_text("epoch,train_loss,val_loss\n1,2,3\n2,1,2\n")
    (scores / "metrics.json").write_text('{"test_ppl":4}')
    result = inspect_run(tmp_path, historical=True)
    assert result["status"] == "complete"
    assert result["curves"]["best_validation"]["epoch"] == "2"
    assert result["evaluations"][0]["data"]["test_ppl"] == 4
    assert result["curves"]["validation_loss_change"] == -1


def test_corrupt_evidence_reported(tmp_path):
    (tmp_path / "run_complete.json").write_text("{")
    result = inspect_run(tmp_path)
    assert result["status"] != "complete"
    assert result["warnings"]


def test_markdown_preserves_evidence():
    rendered = render_markdown(
        {"kind": "test", "status": "unknown", "warnings": ["missing data"]}
    )
    assert "unknown" in rendered and "missing data" in rendered


def test_ci_success_and_legacy_context():
    pr = {
        "statusCheckRollup": [
            dict(context="lint", state="SUCCESS", targetUrl="https://example.com"),
            dict(name="core-tests", status="COMPLETED", conclusion="SUCCESS"),
        ]
    }
    result = summarize_ci(pr, ["lint", "core-tests"])
    assert result["status"] == "passed"
    assert result["checks"][1]["url"] == "https://example.com"
    pr["statusCheckRollup"][1]["conclusion"] = "SKIPPED"
    assert summarize_ci(pr, ["lint", "core-tests"])["status"] == "skipped"


def test_active_quiet_and_tail_evidence(tmp_path):
    import os

    (tmp_path / "train.log").write_text(
        "[progress] epoch=1/3 optimizer_step=20 lr=0.001\nRuntimeError: failure\n"
    )
    os.utime(tmp_path / "train.log", (1, 1))
    with (tmp_path / ".run.lock").open("w") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        report = inspect_run(tmp_path, now=4000, quiet_minutes=30)
    assert report["status"] == "active_quiet"
    assert "optimizer_step=20" in report["logs"][0]["latest_progress"]
    assert report["logs"][0]["error_evidence"]


def test_checkpoint_type_detection_and_unsafe_legacy(tmp_path):
    import torch

    torch.save(
        {
            "cfg": {"trainer": "protein_multitask"},
            "run_progress": {"optimizer_step": 20},
        },
        tmp_path / "last.pt",
    )
    result = inspect_run(tmp_path, historical=True)
    assert result["run_type"] == "protein-critic"
    assert result["checkpoint_state"]["progress"]["optimizer_step"] == 20
    (tmp_path / "last.pt").write_bytes(b"invalid checkpoint")
    result = inspect_run(tmp_path, historical=True)
    assert any("checkpoint metadata unavailable" in w for w in result["warnings"])


def test_malformed_csv_nonfinite_and_type_conflict(tmp_path):
    (tmp_path / "config.json").write_text(
        json.dumps({"trainer": "protein_multitask", "train_npz": "data.npz"})
    )
    (tmp_path / "curves.csv").write_text("epoch,train_loss,val_loss\n1,2,nan\n2,3\n")
    result = inspect_run(tmp_path)
    assert result["run_type"] == "ambiguous"
    assert any("incomplete CSV" in w for w in result["warnings"])
    assert any("non-finite" in w for w in result["warnings"])


def test_benchmark_requires_explicit_inputs_and_matching_type(tmp_path):
    from src.training.inspection.benchmarks import run_benchmark

    with pytest.raises(ValueError, match="requires run type"):
        run_benchmark(
            tmp_path, "unknown", "codon-test", checkpoint=tmp_path / "best.pt"
        )
    with pytest.raises(ValueError, match="requires --manifest"):
        run_benchmark(
            tmp_path, "codonlm", "codon-test", checkpoint=tmp_path / "best.pt"
        )
    assert not (tmp_path / "reports").exists()


@pytest.mark.parametrize(
    "outcome", ["pass", "fail", "timeout", "missing_result", "changed_input"]
)
def test_benchmark_execution_receipt_and_isolation(tmp_path, monkeypatch, outcome):
    import subprocess
    from src.training.inspection import benchmarks

    checkpoint = tmp_path / "best.pt"
    checkpoint.write_bytes(b"checkpoint")
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}")
    original_scores = tmp_path / "scores"
    original_scores.mkdir()
    (original_scores / "metrics.json").write_text('{"original":1}')

    def fake_run(command, **kwargs):
        directory = Path(command[command.index("--run_dir") + 1])
        assert directory != tmp_path
        assert kwargs["env"]["FORCE_CPU"] == "1"
        if outcome == "timeout":
            raise subprocess.TimeoutExpired(command, 5)
        if outcome in ("pass", "changed_input"):
            (directory / "scores").mkdir()
            (directory / "scores/metrics.json").write_text('{"test_ppl":2}')
        if outcome == "changed_input":
            checkpoint.write_bytes(b"replaced checkpoint")
        return subprocess.CompletedProcess(command, 1 if outcome == "fail" else 0)

    monkeypatch.setattr(benchmarks.subprocess, "run", fake_run)
    result = benchmarks.run_benchmark(
        tmp_path, "codonlm", "codon-test", checkpoint=checkpoint, manifest=manifest
    )
    assert (
        result["status"]
        == {
            "pass": "passed",
            "fail": "failed",
            "timeout": "timeout",
            "missing_result": "failed",
            "changed_input": "failed",
        }[outcome]
    )
    assert result["inputs"]["checkpoint"]["sha256"]
    assert (original_scores / "metrics.json").read_text() == '{"original":1}'
    report = inspect_run(tmp_path, historical=True)
    assert any("execution.json" in r["path"] for r in report["evaluations"])


def test_critic_benchmark_command(tmp_path, monkeypatch):
    import subprocess
    from src.training.inspection import benchmarks

    inputs = []
    for name in ["best.pt", "config.yaml", "validation.jsonl"]:
        path = tmp_path / name
        path.write_text("placeholder")
        inputs.append(path)

    def fake_run(command, **kwargs):
        assert command[2] == "scripts.eval_multi_task_critic"
        assert command[command.index("--split") + 1] == "validation"
        Path(command[command.index("--out_json") + 1]).write_text('{"accuracy":0.7}')
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(benchmarks.subprocess, "run", fake_run)
    result = benchmarks.run_benchmark(
        tmp_path,
        "protein-critic",
        "critic-validation",
        checkpoint=inputs[0],
        config=inputs[1],
        data=inputs[2],
    )
    assert result["status"] == "passed"


def test_ci_cli_error_is_structured(monkeypatch, capsys):
    import subprocess
    import sys
    from scripts import check_ci

    monkeypatch.setattr(sys, "argv", ["check_ci", "--pr", "12", "--format", "json"])
    monkeypatch.setattr(
        check_ci.subprocess,
        "run",
        lambda *a, **kw: subprocess.CompletedProcess(
            a[0], 1, "", "authentication required"
        ),
    )
    assert check_ci.main() == 2
    assert json.loads(capsys.readouterr().out)["status"] == "error"


def test_training_cli_missing_and_output(tmp_path, monkeypatch, capsys):
    import sys
    from scripts import check_training

    monkeypatch.setattr(
        sys,
        "argv",
        ["check_training", "--run-dir", str(tmp_path / "absent"), "--format", "json"],
    )
    assert check_training.main() == 2
    assert json.loads(capsys.readouterr().out)["status"] == "error"
    output = tmp_path / "report.md"
    monkeypatch.setattr(
        sys,
        "argv",
        ["check_training", "--run-dir", str(tmp_path), "--output", str(output)],
    )
    assert check_training.main() == 0
    assert "| Field | Value |" in output.read_text()


def test_analyze_cli_includes_benchmark_failure(tmp_path, monkeypatch, capsys):
    import sys
    from scripts import analyze_run

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "analyze_run",
            "--run-dir",
            str(tmp_path),
            "--run-type",
            "codonlm",
            "--benchmark",
            "codon-test",
            "--checkpoint",
            "best.pt",
            "--format",
            "json",
        ],
    )
    monkeypatch.setattr(
        analyze_run, "run_benchmark", lambda *a, **kw: {"status": "failed"}
    )
    assert analyze_run.main() == 1
    assert (
        json.loads(capsys.readouterr().out)["benchmark_execution"]["status"] == "failed"
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "analyze_run",
            "--run-dir",
            str(tmp_path),
            "--benchmark",
            "codon-test",
            "--format",
            "json",
        ],
    )
    assert analyze_run.main() == 2
    assert "--checkpoint" in json.loads(capsys.readouterr().out)["error"]


def test_saved_yaml_and_strict_json(tmp_path):
    from src.training.inspection.reporting import clean

    (tmp_path / "config.yaml").write_text("trainer: protein_multitask\n")
    report = inspect_run(tmp_path)
    assert report["run_type"] == "protein-critic"
    assert clean({"value": float("nan")}) == {"value": "nan"}


def test_ci_cli_success_repo_and_expected(monkeypatch, capsys):
    import sys
    import subprocess
    from scripts import check_ci

    monkeypatch.setattr(
        sys,
        "argv",
        ["check_ci", "--repo", "owner/repo", "--expect", "custom", "--format", "json"],
    )

    def fake_run(command, **kwargs):
        assert command[command.index("--repo") + 1] == "owner/repo"
        return subprocess.CompletedProcess(
            command,
            0,
            json.dumps(
                {"statusCheckRollup": [{"context": "custom", "state": "SUCCESS"}]}
            ),
            "",
        )

    monkeypatch.setattr(check_ci.subprocess, "run", fake_run)
    assert check_ci.main() == 0
    assert json.loads(capsys.readouterr().out)["status"] == "passed"


def test_inspection_preserves_artifacts(tmp_path):
    (tmp_path / ".run.lock").write_text("pid=999999\n")
    (tmp_path / "train.log").write_text("[progress] epoch=1/2 optimizer_step=3\n")
    before = {
        p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in tmp_path.iterdir()
    }
    inspect_run(tmp_path)
    after = {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in tmp_path.iterdir()}
    assert before == after


def test_unreadable_lock_cannot_confirm_completion(tmp_path, monkeypatch):
    from src.training.inspection import runs

    (tmp_path / "run_complete.json").write_text('{"status":"complete"}')
    monkeypatch.setattr(runs, "lock_state", lambda root: "unknown")
    assert inspect_run(tmp_path)["status"] == "unknown"


def test_legacy_critic_progress_and_target(tmp_path):
    import torch

    torch.save(
        {
            "epoch": 9,
            "epoch_complete": True,
            "optimizer_step": 4090,
            "checkpoint_reason": "epoch",
            "cfg": {
                "epochs": 10,
                "trainer": "protein_multitask",
                "multi_label_tasks": [],
                "regression_tasks": ["stability"],
            },
            "model_spec": {"task_dims": {"family": 43, "function": 7, "stability": 1}},
        },
        tmp_path / "last_critic.pt",
    )
    r = inspect_run(tmp_path, historical=True)
    assert r["checkpoint_state"]["progress"]["completed_epochs"] == 10
    assert r["checkpoint_state"]["progress"]["optimizer_step"] == 4090
    assert r["completion_evidence"]["target_reached"] is True
    assert r["status"] != "complete"
    assert "regression" in r["prediction_tasks"][2]["meaning"]
    assert r["multilabel"]["status"] == "disabled"


def test_legacy_codonlm_progress_and_target(tmp_path):
    import torch

    torch.save(
        {
            "epoch": 10,
            "step": 5000,
            "epoch_microbatch_idx": 0,
            "cfg": {"epochs": 10, "train_npz": "train.npz"},
        },
        tmp_path / "last.pt",
    )
    report = inspect_run(tmp_path, historical=True)

    assert report["run_type"] == "codonlm"
    assert report["checkpoint_state"]["progress"]["completed_epochs"] == 10
    assert report["checkpoint_state"]["progress"]["optimizer_step"] == 5000
    assert report["completion_evidence"]["target_reached"] is True
    assert report["status"] != "complete"


def test_markdown_human_duration_and_empty_mapping():
    md = render_markdown(
        {
            "kind": "test",
            "status": "unknown",
            "checkpoints": [{"age_seconds": 4928718}],
            "metadata": {},
        }
    )
    assert "57d 1h" in md
    assert "Not recorded" in md


def test_status_sidecar_uses_model_type_and_terminal_error(tmp_path):
    (tmp_path / "run_status.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "status": "failed",
                "run_type": "protein-critic",
                "error": {"message": "nonfinite limit"},
            }
        )
    )
    r = inspect_run(tmp_path)
    assert r["status"] == "failed"
    assert r["run_type"] == "protein-critic"
    assert r["session"]["error"]["message"] == "nonfinite limit"
