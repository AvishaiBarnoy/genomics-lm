from __future__ import annotations

import ast
import tomllib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REGISTRY_PATH = ROOT / "docs" / "training_entrypoints.toml"


def _registry() -> dict[str, dict[str, str]]:
    data = tomllib.loads(REGISTRY_PATH.read_text())
    assert data["schema_version"] == 1
    entries = data["entrypoints"]
    paths = [entry["path"] for entry in entries]
    assert len(paths) == len(set(paths)), "duplicate trainer registry path"
    return {entry["path"]: entry for entry in entries}


def _discovered_trainers() -> set[str]:
    return {
        path.relative_to(ROOT).as_posix()
        for directory in (ROOT / "src", ROOT / "scripts")
        for path in directory.rglob("train_*.py")
        if "tests" not in path.parts
    }


def test_all_training_entrypoints_are_registered_and_exist():
    registry = _registry()
    discovered = _discovered_trainers()

    assert discovered == set(registry), (
        f"unregistered={sorted(discovered - set(registry))}; "
        f"stale={sorted(set(registry) - discovered)}"
    )

    for path, entry in registry.items():
        assert (ROOT / path).is_file(), f"registered trainer does not exist: {path}"
        status = entry["status"]
        assert status in {"engine", "deferred", "exempt"}, (path, status)
        if status != "engine":
            assert entry.get("reason", "").strip(), (
                f"{path} needs an explicit reason for status={status}"
            )


def test_engine_trainers_use_shared_engine_and_run_lifecycle():
    registry = _registry()
    for path, entry in registry.items():
        if entry["status"] != "engine":
            continue
        tree = ast.parse((ROOT / path).read_text(), filename=path)
        names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
        opens_training_run = any(
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "open"
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "TrainingRun"
            for node in ast.walk(tree)
        )
        run_variables = {
            target.id
            for node in ast.walk(tree)
            if isinstance(node, ast.Assign)
            and isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Attribute)
            and node.value.func.attr == "open"
            and isinstance(node.value.func.value, ast.Name)
            and node.value.func.value.id == "TrainingRun"
            for target in node.targets
            if isinstance(target, ast.Name)
        }
        starts_run_logging = any(
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "start_logging"
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id in run_variables
            for node in ast.walk(tree)
        )
        closes_run = any(
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "close"
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id in run_variables
            for node in ast.walk(tree)
        )
        assert "TrainingEngine" in names, f"{path} does not assemble TrainingEngine"
        assert "TrainingRun" in names, f"{path} does not use TrainingRun"
        assert opens_training_run, f"{path} does not open a managed run"
        assert starts_run_logging, f"{path} does not start run logging"
        assert closes_run, f"{path} does not close the managed run"


def test_shared_run_lifecycle_contract_tests_remain_registered():
    required_tests = {
        "tests/test_training_run_lifecycle.py": {
            "test_fresh_duplicate_run_allocates_serial_directory",
            "test_active_run_lock_rejects_second_writer",
            "test_resume_requires_newest_last_checkpoint",
            "test_completed_run_rejects_equal_target_and_allows_extension",
        },
        "tests/test_training_engine.py": {
            "test_interrupted_resume_matches_uninterrupted_parameters",
        },
    }

    for path, required_names in required_tests.items():
        tree = ast.parse((ROOT / path).read_text(), filename=path)
        found = {
            node.name
            for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
        assert required_names <= found, (
            f"missing lifecycle coverage in {path}: "
            f"{sorted(required_names - found)}"
        )


def test_engine_trainers_expose_explicit_checkpoint_forks():
    registry = _registry()
    for path, entry in registry.items():
        if entry["status"] != "engine":
            continue
        tree = ast.parse((ROOT / path).read_text(), filename=path)
        string_literals = {
            node.value
            for node in ast.walk(tree)
            if isinstance(node, ast.Constant) and isinstance(node.value, str)
        }
        fork_keywords = [
            keyword
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "open"
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "TrainingRun"
            for keyword in node.keywords
            if keyword.arg == "fork_from"
        ]
        assert "--fork-from" in string_literals, f"{path} has no --fork-from CLI"
        assert fork_keywords, f"{path} does not pass fork_from to TrainingRun.open"


def test_engine_trainers_do_not_write_canonical_checkpoints_directly():
    registry = _registry()
    for path, entry in registry.items():
        if entry["status"] != "engine":
            continue
        tree = ast.parse((ROOT / path).read_text(), filename=path)
        forbidden_calls = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            if isinstance(node.func, ast.Name) and node.func.id == "save_checkpoint_atomic":
                forbidden_calls.append(node.lineno)
            elif (
                isinstance(node.func, ast.Attribute)
                and node.func.attr == "save"
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "torch"
            ):
                forbidden_calls.append(node.lineno)
        assert not forbidden_calls, (
            f"{path} writes checkpoints directly at lines {forbidden_calls}; "
            "canonical last/best files belong to TrainingEngine and selected "
            "artifacts must use save_artifact_atomic"
        )
