from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "tools/run_qwen38_tp_correctness.py"


def _load():
    assert MODULE_PATH.is_file(), "Qwen3.8 correctness runner is missing"
    spec = importlib.util.spec_from_file_location(
        "run_qwen38_tp_correctness_under_test",
        MODULE_PATH,
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


runner = _load()


def test_production_root_is_the_approved_remote_data_mount():
    assert runner.APPROVED_REMOTE_ROOT == Path(
        "/data00/home/sitian/tinyllmforge-workspaces/"
        "command-timeline-20260818"
    )


def _plan(tmp_path, monkeypatch):
    approved = tmp_path / "approved"
    source = approved / "source"
    model = approved / "models" / "qwen38"
    source.mkdir(parents=True)
    model.mkdir(parents=True)
    manifest = model.parent / "model_manifest.json"
    manifest.write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(
        runner,
        "APPROVED_REMOTE_ROOT",
        approved.resolve(),
    )
    return runner.build_correctness_plan(
        attempt_root=approved / "attempts" / "correctness-r1",
        source_root=source,
        model_root=model,
        model_manifest_path=manifest,
        source_tree_sha256="1" * 64,
        model_manifest_sha256="2" * 64,
        python_executable=approved / "env" / "bin" / "python",
        torchrun_executable=approved / "env" / "bin" / "torchrun",
        gpu_indices=(2, 3, 4, 5),
        rendezvous_ports=(32101, 32102),
        prompt_token_ids=(11, 22, 33, 44),
        generated_tokens=2,
        topk=8,
        timeout_s=900,
    )


def test_plan_emits_three_text_only_greedy_commands(tmp_path, monkeypatch):
    plan = _plan(tmp_path, monkeypatch)

    assert plan["schema_version"] == runner.PLAN_SCHEMA
    assert plan["command_order"] == [
        "official_tp1",
        "tinyllmforge_tp1",
        "tinyllmforge_tp4",
        "assemble",
        "verify",
    ]
    commands = plan["commands"]
    assert commands["official_tp1"]["argv"][0].endswith("/python")
    assert commands["tinyllmforge_tp1"]["argv"][0].endswith("/torchrun")
    assert commands["tinyllmforge_tp4"]["argv"][0].endswith("/torchrun")
    assert "--nproc-per-node=1" in commands["tinyllmforge_tp1"]["argv"]
    assert "--nproc-per-node=4" in commands["tinyllmforge_tp4"]["argv"]
    assert "--master-port=32101" in commands["tinyllmforge_tp1"]["argv"]
    assert "--master-port=32102" in commands["tinyllmforge_tp4"]["argv"]
    assert plan["rendezvous_ports"] == [32101, 32102]

    for name in ("official_tp1", "tinyllmforge_tp1", "tinyllmforge_tp4"):
        argv = commands[name]["argv"]
        assert "--text-only" in argv
        assert "--greedy" in argv
        assert "--temperature=0" in argv
        assert "--generated-tokens=2" in argv
        assert "--prompt-token-ids=[11,22,33,44]" in argv
        assert "--disable-profiler" in argv
    assert commands["official_tp1"]["env"]["CUDA_VISIBLE_DEVICES"] == "2"
    assert commands["tinyllmforge_tp1"]["env"]["CUDA_VISIBLE_DEVICES"] == "2"
    assert commands["tinyllmforge_tp4"]["env"]["CUDA_VISIBLE_DEVICES"] == (
        "2,3,4,5"
    )


def test_plan_only_writes_below_approved_root_and_has_no_forbidden_actions(
    tmp_path,
    monkeypatch,
):
    plan = _plan(tmp_path, monkeypatch)
    approved = runner.APPROVED_REMOTE_ROOT
    attempt = Path(plan["attempt_root"])
    assert attempt.is_relative_to(approved)
    for path in plan["write_paths"]:
        assert Path(path).is_relative_to(attempt)

    serialized = json.dumps(plan, sort_keys=True)
    for forbidden in (
        "pkill",
        "killall",
        "kinit",
        "krenew",
        "adaptive-ngram",
        "/private/tmp",
    ):
        assert forbidden not in serialized


@pytest.mark.parametrize(
    "mutator,match",
    (
        (
            lambda kwargs, root: kwargs.update(
                attempt_root=root.parent / "outside"
            ),
            "approved remote root",
        ),
        (
            lambda kwargs, root: kwargs.update(
                rendezvous_ports=(32101, 32101)
            ),
            "distinct",
        ),
        (
            lambda kwargs, root: kwargs.update(gpu_indices=(0, 1, 2, 2)),
            "four distinct",
        ),
    ),
)
def test_plan_rejects_unsafe_scope_ports_or_gpu_identity(
    tmp_path,
    monkeypatch,
    mutator,
    match,
):
    approved = tmp_path / "approved"
    source = approved / "source"
    model = approved / "models" / "qwen38"
    source.mkdir(parents=True)
    model.mkdir(parents=True)
    manifest = model.parent / "model_manifest.json"
    manifest.write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(runner, "APPROVED_REMOTE_ROOT", approved.resolve())
    kwargs = {
        "attempt_root": approved / "attempts" / "r1",
        "source_root": source,
        "model_root": model,
        "model_manifest_path": manifest,
        "source_tree_sha256": "1" * 64,
        "model_manifest_sha256": "2" * 64,
        "python_executable": approved / "env/bin/python",
        "torchrun_executable": approved / "env/bin/torchrun",
        "gpu_indices": (0, 1, 2, 3),
        "rendezvous_ports": (32101, 32102),
        "prompt_token_ids": (1, 2),
        "generated_tokens": 2,
        "topk": 8,
        "timeout_s": 900,
    }
    mutator(kwargs, approved)

    with pytest.raises(ValueError, match=match):
        runner.build_correctness_plan(**kwargs)


def test_executor_records_owned_processes_cleanup_and_verification(
    tmp_path,
    monkeypatch,
):
    plan = _plan(tmp_path, monkeypatch)
    calls = []

    def run_command(command, *, timeout_s):
        calls.append(command["name"])
        return {
            "returncode": 0,
            "pid": 1000 + len(calls),
            "pgid": 2000 + len(calls),
            "process_group_destroyed": True,
            "stdout": "",
            "stderr": "",
        }

    receipt = runner.execute_correctness_plan(
        plan,
        run_command=run_command,
        verify_bundle=lambda root: {"classification": "PASS"},
    )

    assert receipt["classification"] == "PASS"
    assert receipt["failed_stage"] is None
    assert calls == plan["command_order"][:-1]
    assert [row["pid"] for row in receipt["processes"]] == [
        1001,
        1002,
        1003,
        1004,
    ]
    assert all(
        row["process_group_destroyed"]
        for row in receipt["processes"]
    )
    assert receipt["owned_children_remaining"] == []
    receipt_path = Path(plan["attempt_root"]) / "runner_receipt.json"
    assert json.loads(receipt_path.read_text()) == receipt


def test_executor_returns_stage_specific_failure_receipt(
    tmp_path,
    monkeypatch,
):
    plan = _plan(tmp_path, monkeypatch)

    def run_command(command, *, timeout_s):
        return {
            "returncode": 7 if command["name"] == "tinyllmforge_tp1" else 0,
            "pid": 101,
            "pgid": 201,
            "process_group_destroyed": True,
            "stdout": "",
            "stderr": "injected failure",
        }

    receipt = runner.execute_correctness_plan(
        plan,
        run_command=run_command,
        verify_bundle=lambda root: {"classification": "PASS"},
    )

    assert receipt["classification"] == "FAIL"
    assert receipt["failed_stage"] == "tinyllmforge_tp1"
    assert receipt["verification"] is None
    assert receipt["owned_children_remaining"] == []


def test_executor_rejects_tampered_output_path_before_launch(
    tmp_path,
    monkeypatch,
):
    plan = _plan(tmp_path, monkeypatch)
    plan["commands"]["official_tp1"]["output_paths"][0] = str(
        runner.APPROVED_REMOTE_ROOT.parent / "escaped.jsonl"
    )
    calls = []

    with pytest.raises(ValueError, match="attempt_root"):
        runner.execute_correctness_plan(
            plan,
            run_command=lambda command, *, timeout_s: calls.append(command),
            verify_bundle=lambda root: {"classification": "PASS"},
        )

    assert calls == []


def test_executor_rejects_tampered_command_before_launch(
    tmp_path,
    monkeypatch,
):
    plan = _plan(tmp_path, monkeypatch)
    plan["commands"]["official_tp1"]["argv"] = [
        "/bin/sh",
        "-c",
        "true",
    ]
    calls = []

    with pytest.raises(ValueError, match="command executable"):
        runner.execute_correctness_plan(
            plan,
            run_command=lambda command, *, timeout_s: calls.append(command),
            verify_bundle=lambda root: {"classification": "PASS"},
        )

    assert calls == []


def test_executor_persists_failure_for_malformed_command_result(
    tmp_path,
    monkeypatch,
):
    plan = _plan(tmp_path, monkeypatch)

    receipt = runner.execute_correctness_plan(
        plan,
        run_command=lambda command, *, timeout_s: {
            "returncode": 0,
            "pid": 101,
            "pgid": 201,
            "process_group_destroyed": True,
            "owned_children_remaining": None,
        },
        verify_bundle=lambda root: {"classification": "PASS"},
    )

    assert receipt["classification"] == "FAIL"
    assert receipt["failed_stage"] == "official_tp1"
    assert "owned_children_remaining" in receipt["failure_reason"]
    assert json.loads(
        (
            Path(plan["attempt_root"]) / "runner_receipt.json"
        ).read_text()
    ) == receipt
