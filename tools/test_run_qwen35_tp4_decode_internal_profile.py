from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"


def _load():
    path = TOOLS / "run_qwen35_tp4_decode_internal_profile.py"
    spec = importlib.util.spec_from_file_location(
        "run_qwen35_tp4_decode_internal_profile",
        path,
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _authorization(runner):
    return runner.WorkerAuthorization(
        prerequisites_sha256="a" * 64,
        source_tree_sha256="b" * 64,
        model_manifest_sha256="c" * 64,
        workload_manifest_sha256="d" * 64,
        gpu_indices=(2, 4, 5, 6),
    )


def _runtime(runner):
    return runner.WorkerRuntimeArtifacts(
        model_dir="/remote/model",
        model_manifest_path="/remote/model_manifest.json",
        correctness_prerequisites_path="/remote/prerequisites.json",
        workload_manifest_path="/remote/workload_manifest.json",
    )


def test_build_profile_cases_freezes_one_warmup_and_five_pairs():
    runner = _load()

    cases = runner.build_profile_cases()

    assert len(cases) == 12
    assert [(case.phase, case.repetition) for case in cases[:2]] == [
        ("warmup", 0),
        ("warmup", 0),
    ]
    assert {
        (case.phase, case.repetition, case.policy)
        for case in cases[2:]
    } == {
        ("measured", repetition, policy)
        for repetition in range(5)
        for policy in ("recompute", "exact_restore")
    }
    assert all(case.workload == "w2_long_reuse" for case in cases)


def test_structured_commands_enable_eight_token_decode_profile():
    runner = _load()
    ports = [
        (22000 + index * 2, 22001 + index * 2)
        for index in range(12)
    ]

    commands = runner.build_structured_commands(
        remote_source="/remote/source",
        remote_cases="/remote/output/cases",
        ports=ports,
        authorization=_authorization(runner),
        runtime_artifacts=_runtime(runner),
    )

    assert len(commands) == 12
    assert all(
        command["env"]["CUDA_VISIBLE_DEVICES"] == "2,4,5,6"
        for command in commands
    )
    for command in commands:
        assert "--profile" in command["argv"]
        assert command["argv"][
            command["argv"].index("--generated-tokens-override") + 1
        ] == "8"
        assert "--decode-internal-profile" in command["argv"]
        assert command["output_dir"] in command["argv"]
        assert command["output_dir"].endswith(command["case_id"])


def test_nsys_command_uses_separate_replay_identity():
    runner = _load()
    structured = runner.build_structured_commands(
        remote_source="/remote/source",
        remote_cases="/remote/output/cases",
        ports=[(22000, 22001)] * 12,
        authorization=_authorization(runner),
        runtime_artifacts=_runtime(runner),
    )[2]

    command = runner.build_nsys_command(
        structured_command=structured,
        remote_nsys="/usr/local/bin/nsys",
        report_prefix="/remote/nsys/r2-recompute",
        repetition=2,
    )

    assert command["argv"][:4] == [
        "/usr/local/bin/nsys",
        "profile",
        "--trace=cuda,nvtx,osrt",
        "--force-overwrite=true",
    ]
    assert "nccl" not in command["argv"][2]
    phase_index = command["argv"].index("--phase")
    assert command["argv"][phase_index + 1] == "nsys_replay"
    assert command["phase"] == "nsys_replay"
    assert command["case_id"] == (
        "w2_long_reuse__nsys_replay__r2__recompute"
    )


def test_nsys_stats_report_rejects_zero_exit_error_output():
    runner = _load()

    available = runner._nsys_stats_report_available(
        returncode=0,
        output=(
            "Processing [trace.sqlite] with [nccl_sum]...\n"
            "ERROR: Report 'nccl_sum' could not be found.\n"
        ),
    )

    assert available is False


def test_nsys_stats_reports_cover_nvtx_ranges_and_kernels():
    runner = _load()

    assert runner._nsys_stats_reports() == (
        "cuda_gpu_kern_sum",
        "nvtx_pushpop_sum",
        "nvtx_kern_sum",
        "nccl_sum",
    )


def test_shared_guard_allows_processes_but_enforces_memory_and_utilization():
    runner = _load()
    rows = [
        {
            "gpu_index": index,
            "gpu_uuid": f"GPU-{index}",
            "free_bytes": runner.MIN_GPU_FREE_BYTES,
            "utilization_percent": 10,
            "compute_processes": [{"pid": 1000 + index}],
        }
        for index in runner.GPU_INDICES
    ]

    ready = runner.evaluate_shared_gpu_guard(rows)

    assert ready["classification"] == "READY"
    assert ready["resource_policy"] == "shared-low-utilization"
    assert ready["exclusive"] is False
    assert [row["gpu_index"] for row in ready["selected_gpus"]] == [
        2,
        4,
        5,
        6,
    ]

    rows[1]["free_bytes"] -= 1
    assert runner.evaluate_shared_gpu_guard(rows)["classification"] == (
        "BLOCKED_RESOURCES"
    )
    rows[1]["free_bytes"] = runner.MIN_GPU_FREE_BYTES
    rows[2]["utilization_percent"] = 11
    assert runner.evaluate_shared_gpu_guard(rows)["classification"] == (
        "BLOCKED_RESOURCES"
    )


def test_wait_for_guard_polls_until_resources_are_ready(tmp_path):
    runner = _load()
    rows = [
        {
            "gpu_index": index,
            "gpu_uuid": f"GPU-{index}",
            "free_bytes": 30 * 1024**3,
            "utilization_percent": 0,
            "compute_processes": [],
        }
        for index in runner.GPU_INDICES
    ]
    blocked_rows = [dict(row) for row in rows]
    blocked_rows[2]["utilization_percent"] = 93
    observations = iter((blocked_rows, rows))
    sleeps = []

    receipt = runner._wait_for_guard(
        tmp_path,
        "worker-test",
        query_rows=lambda: next(observations),
        sleep=lambda seconds: sleeps.append(seconds),
        poll_interval_s=30,
        max_wait_s=60,
    )

    assert receipt["classification"] == "READY"
    assert receipt["waited_for_resources"] is True
    assert len(receipt["samples"]) == 2
    assert sleeps == [30]
    written = json.loads(
        (tmp_path / "guards" / "worker-test.json").read_text()
    )
    assert written == receipt


def test_wait_for_guard_survives_transient_gpu_query_error(tmp_path):
    runner = _load()
    rows = [
        {
            "gpu_index": index,
            "gpu_uuid": f"GPU-{index}",
            "free_bytes": 30 * 1024**3,
            "utilization_percent": 0,
            "compute_processes": [],
        }
        for index in runner.GPU_INDICES
    ]
    observations = iter((RuntimeError("ssh 255"), rows))
    sleeps = []

    def query_rows():
        observation = next(observations)
        if isinstance(observation, BaseException):
            raise observation
        return observation

    receipt = runner._wait_for_guard(
        tmp_path,
        "worker-query-retry",
        query_rows=query_rows,
        sleep=lambda seconds: sleeps.append(seconds),
        poll_interval_s=30,
        max_wait_s=60,
    )

    assert receipt["classification"] == "READY"
    assert receipt["waited_for_resources"] is True
    assert receipt["samples"][0]["classification"] == "GPU_QUERY_ERROR"
    assert receipt["samples"][0]["reasons"] == ["ssh 255"]
    assert sleeps == [30]


def test_entry_guard_blocked_stops_before_stage():
    runner = _load()
    events = []

    result = runner.orchestrate_attempt(
        entry_guard=lambda: {
            "classification": "BLOCKED_RESOURCES",
        },
        stage=lambda: events.append("stage"),
        worker_guard=lambda case: events.append(("guard", case.case_id)),
        run_worker=lambda command: events.append(("worker", command)),
        aggregate=lambda: events.append("aggregate"),
        run_nsys=lambda repetition: events.append(("nsys", repetition)),
        cleanup=lambda: events.append("cleanup"),
    )

    assert result["classification"] == "BLOCKED_RESOURCES"
    assert events == []


def test_worker_guard_blocked_stops_before_measured_worker():
    runner = _load()
    events = []
    calls = {"guard": 0}

    def worker_guard(case):
        calls["guard"] += 1
        events.append(("guard", case.case_id))
        if calls["guard"] == 3:
            return {"classification": "BLOCKED_RESOURCES"}
        return {"classification": "READY"}

    result = runner.orchestrate_attempt(
        entry_guard=lambda: {"classification": "READY"},
        stage=lambda: events.append("stage"),
        worker_guard=worker_guard,
        run_worker=lambda command: events.append(
            ("worker", command.case_id)
        ) or {"returncode": 0},
        aggregate=lambda: {"representative_repetition": 2},
        run_nsys=lambda repetition: {"classification": "COMPLETE"},
        cleanup=lambda: {"classification": "CLEAN"},
    )

    assert result["classification"] == "BLOCKED_WORKER_ENTRY"
    assert [event for event in events if event[0] == "worker"] == [
        ("worker", runner.build_profile_cases()[0].case_id),
        ("worker", runner.build_profile_cases()[1].case_id),
    ]


def test_worker_failure_preserves_attempt_and_runs_cleanup():
    runner = _load()
    cleanup_calls = []

    result = runner.orchestrate_attempt(
        entry_guard=lambda: {"classification": "READY"},
        stage=lambda: None,
        worker_guard=lambda case: {"classification": "READY"},
        run_worker=lambda command: {
            "returncode": 9,
            "case_id": command.case_id,
        },
        aggregate=lambda: {"representative_repetition": 2},
        run_nsys=lambda repetition: {"classification": "COMPLETE"},
        cleanup=lambda: cleanup_calls.append(True) or {
            "classification": "CLEAN",
        },
    )

    assert result["classification"] == "FAILED_STRUCTURED_WORKER"
    assert result["preserve_attempt"] is True
    assert cleanup_calls == [True]


def test_nsys_unavailable_preserves_structured_result():
    runner = _load()

    result = runner.orchestrate_attempt(
        entry_guard=lambda: {"classification": "READY"},
        stage=lambda: None,
        worker_guard=lambda case: {"classification": "READY"},
        run_worker=lambda command: {"returncode": 0},
        aggregate=lambda: {
            "measured_pairs": 5,
            "representative_repetition": 3,
        },
        run_nsys=lambda repetition: {
            "classification": "NSYS_UNAVAILABLE",
        },
        cleanup=lambda: {"classification": "CLEAN"},
    )

    assert result["classification"] == "COMPLETE_WITHOUT_NSYS"
    assert result["structured"]["measured_pairs"] == 5
    assert result["nsys"]["classification"] == "NSYS_UNAVAILABLE"


def test_cleanup_incomplete_fails_final_classification():
    runner = _load()

    result = runner.orchestrate_attempt(
        entry_guard=lambda: {"classification": "READY"},
        stage=lambda: None,
        worker_guard=lambda case: {"classification": "READY"},
        run_worker=lambda command: {"returncode": 0},
        aggregate=lambda: {
            "measured_pairs": 5,
            "representative_repetition": 4,
        },
        run_nsys=lambda repetition: {"classification": "COMPLETE"},
        cleanup=lambda: {
            "classification": "DIRTY",
            "remaining_attempt_scoped_pids": [123],
        },
    )

    assert result["classification"] == "FAILED_CLEANUP"
    assert result["cleanup"]["remaining_attempt_scoped_pids"] == [123]


def test_shell_ssh_command_keeps_operators_inside_inner_script():
    runner = _load()

    argv = runner._shell_ssh_argv(
        "cd /remote/source && env CUDA_VISIBLE_DEVICES=2,4,5,6 true"
    )

    assert argv[:2] == ["ssh", "-o"]
    assert argv[-3] == "bash"
    assert argv[-2] == "-lc"
    assert "&&" in argv[-1]
    assert "'&&'" not in argv[-1]


def test_subprocess_retries_transient_ssh_255():
    runner = _load()
    calls = []
    results = iter([
        SimpleNamespace(
            returncode=255,
            stdout="",
            stderr="connection closed",
        ),
        SimpleNamespace(
            returncode=0,
            stdout="ok",
            stderr="",
        ),
    ])
    original_run = runner.subprocess.run
    original_sleep = runner.time.sleep
    runner.subprocess.run = lambda *args, **kwargs: (
        calls.append((args, kwargs)) or next(results)
    )
    runner.time.sleep = lambda seconds: calls.append(("sleep", seconds))
    try:
        result = runner._run_subprocess(
            ["ssh", "host", "true"],
            timeout_s=1,
        )
    finally:
        runner.subprocess.run = original_run
        runner.time.sleep = original_sleep

    assert result.returncode == 0
    assert len([call for call in calls if call[0] != "sleep"]) == 2
    assert ("sleep", 1.0) in calls
