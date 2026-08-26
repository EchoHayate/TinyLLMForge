from __future__ import annotations

import importlib.util
import json
from datetime import datetime, timezone
from pathlib import Path
import shlex
import sys
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"


def _load():
    path = TOOLS / "run_qwen38_tp4_communication_profile.py"
    spec = importlib.util.spec_from_file_location(
        "run_qwen38_tp4_communication_profile",
        path,
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _gpu(
    index: int,
    *,
    uuid: str | None = None,
    memory_used_mib: int = 0,
    utilization_percent: int = 0,
    compute_processes: list[dict] | None = None,
) -> dict:
    return {
        "gpu_index": index,
        "gpu_uuid": uuid or f"GPU-{index}",
        "memory_used_mib": memory_used_mib,
        "utilization_percent": utilization_percent,
        "compute_processes": (
            [] if compute_processes is None else compute_processes
        ),
    }


def test_select_strict_clean_gpus_accepts_exact_boundaries():
    runner = _load()
    inventory = [
        _gpu(
            index,
            memory_used_mib=runner.MAX_GPU_MEMORY_USED_MIB,
            utilization_percent=runner.MAX_GPU_UTILIZATION_PERCENT,
        )
        for index in (7, 2, 9, 4)
    ]

    selected = runner.select_strict_clean_gpus(inventory)

    assert isinstance(selected, tuple)
    assert [row["gpu_index"] for row in selected] == [2, 4, 7, 9]
    assert len({row["gpu_uuid"] for row in selected}) == 4


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("memory_used_mib", 1025, "four strict-clean GPUs"),
        ("utilization_percent", 6, "four strict-clean GPUs"),
        (
            "compute_processes",
            [{"pid": 4321, "process_name": "python"}],
            "four strict-clean GPUs",
        ),
    ],
)
def test_select_strict_clean_gpus_rejects_non_clean_gpu(
    field,
    value,
    message,
):
    runner = _load()
    inventory = [_gpu(index) for index in range(4)]
    inventory[0][field] = value

    with pytest.raises(ValueError, match=message):
        runner.select_strict_clean_gpus(inventory)


def test_select_strict_clean_gpus_rejects_insufficient_inventory():
    runner = _load()

    with pytest.raises(ValueError, match="four strict-clean GPUs"):
        runner.select_strict_clean_gpus(
            [_gpu(index) for index in range(3)]
        )


def test_select_strict_clean_gpus_rejects_duplicate_uuid():
    runner = _load()
    inventory = [_gpu(index) for index in range(4)]
    inventory[3]["gpu_uuid"] = inventory[0]["gpu_uuid"]

    with pytest.raises(ValueError, match="duplicate GPU UUID"):
        runner.select_strict_clean_gpus(inventory)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda rows: rows.append("not-a-row"),
        lambda rows: rows[0].update(gpu_index=True),
        lambda rows: rows[0].update(gpu_uuid=""),
        lambda rows: rows[0].update(memory_used_mib=True),
        lambda rows: rows[0].update(memory_used_mib=-1),
        lambda rows: rows[0].update(utilization_percent=1.5),
        lambda rows: rows[0].update(utilization_percent=-1),
        lambda rows: rows[0].update(compute_processes={}),
        lambda rows: rows[0].update(extra="unexpected"),
    ],
)
def test_select_strict_clean_gpus_rejects_malformed_telemetry(mutate):
    runner = _load()
    inventory = [_gpu(index) for index in range(4)]
    mutate(inventory)

    with pytest.raises(ValueError, match="GPU telemetry"):
        runner.select_strict_clean_gpus(inventory)


def test_observed_gpu_processes_must_be_owned_after_launch():
    runner = _load()
    selected = runner.select_strict_clean_gpus(
        [_gpu(index) for index in range(4)]
    )
    observed = [_gpu(index) for index in range(4)]
    observed[1]["compute_processes"] = [
        {
            "pid": 8001,
            "process_name": "owned-worker",
            "used_memory_mib": 256,
        },
    ]

    runner.validate_selected_gpu_processes(
        selected=selected,
        observed=observed,
        owned_pids={8001},
    )

    observed[2]["compute_processes"] = [
        {
            "pid": 9999,
            "process_name": "unrelated",
            "used_memory_mib": 256,
        },
    ]
    with pytest.raises(ValueError, match="unrelated GPU process"):
        runner.validate_selected_gpu_processes(
            selected=selected,
            observed=observed,
            owned_pids={8001},
        )


def test_build_workload_cases_freezes_complete_deterministic_campaign():
    runner = _load()

    cases = runner.build_workload_cases()

    assert isinstance(cases, tuple)
    assert len(cases) == 60
    assert len({case.case_id for case in cases}) == 60
    assert [case.workload for case in cases] == [
        workload
        for workload in runner.WORKLOAD_ORDER
        for _ in range(12)
    ]

    expected_workloads = {
        "P0": ("causal", 256, 128, 1),
        "P1": ("causal", 2048, 128, 1),
        "Q0": ("online", 256, 128, 4),
        "Q1": ("online", 256, 128, 8),
        "Q2": ("online", 2048, 128, 4),
    }
    for workload, expected in expected_workloads.items():
        workload_cases = [
            case for case in cases if case.workload == workload
        ]
        assert {
            (
                case.family,
                case.prompt_tokens,
                case.output_tokens,
                case.concurrency,
            )
            for case in workload_cases
        } == {expected}
        assert [
            (case.phase, case.repetition)
            for case in workload_cases
        ] == [
            ("warmup", 0),
            ("warmup", 1),
            *[
                ("measured", repetition)
                for repetition in range(5)
            ],
            *[
                ("nsys_replay", repetition)
                for repetition in range(5)
            ],
        ]

    warmups = [case for case in cases if case.phase == "warmup"]
    measured = [case for case in cases if case.phase == "measured"]
    replays = [
        case for case in cases if case.phase == "nsys_replay"
    ]
    assert len(warmups) == 10
    assert len(measured) == 25
    assert len(replays) == 25
    assert all(case.profiled is False for case in warmups + measured)
    assert all(case.profiled is True for case in replays)
    assert all(case.representative is False for case in cases)
    assert {
        case.overhead_pair_id for case in measured
    } == {
        case.overhead_pair_id for case in replays
    }
    assert None not in {
        case.overhead_pair_id for case in measured + replays
    }


def test_representatives_require_all_structured_timings_and_keep_all_replays():
    runner = _load()
    cases = runner.build_workload_cases()
    timings = {
        workload: {
            repetition: float(100 + repetition * 10)
            for repetition in range(5)
        }
        for workload in runner.WORKLOAD_ORDER
    }

    marked = runner.mark_representative_replays(cases, timings)

    assert len(marked) == len(cases)
    replay_rows = [
        case for case in marked if case.phase == "nsys_replay"
    ]
    assert len(replay_rows) == 25
    assert sum(case.representative for case in replay_rows) == 5
    for workload in runner.WORKLOAD_ORDER:
        representatives = [
            case
            for case in replay_rows
            if case.workload == workload and case.representative
        ]
        assert [case.repetition for case in representatives] == [2]

    incomplete = {
        workload: dict(rows)
        for workload, rows in timings.items()
    }
    incomplete["Q2"].pop(4)
    with pytest.raises(ValueError, match="five structured timings"):
        runner.mark_representative_replays(cases, incomplete)


def test_representative_tie_breaks_to_lower_repetition():
    runner = _load()
    cases = runner.build_workload_cases()
    timings = {
        workload: {
            0: 100.0,
            1: 110.0,
            2: 120.0,
            3: 130.0,
            4: 140.0,
        }
        for workload in runner.WORKLOAD_ORDER
    }
    timings["P0"] = {
        0: 90.0,
        1: 100.0,
        2: 100.0,
        3: 110.0,
        4: 120.0,
    }

    marked = runner.mark_representative_replays(cases, timings)

    assert [
        case.repetition
        for case in marked
        if case.phase == "nsys_replay"
        and case.workload == "P0"
        and case.representative
    ] == [1]


def _selected(runner):
    return runner.select_strict_clean_gpus(
        [_gpu(index) for index in (2, 4, 6, 7)]
    )


def _topology():
    return {
        "gpu_rows": [
            {
                "gpu_index": index,
                "gpu_uuid": f"GPU-{index}",
                "pci_bus_id": f"00000000:{index:02x}:00.0",
            }
            for index in (2, 4, 6, 7)
        ],
        "interconnect_matrix": (
            "GPU2 GPU4 GPU6 GPU7 CPU Affinity\n"
            "GPU2 X NV18 NV18 NV18 0-31\n"
            "GPU4 NV18 X NV18 NV18 0-31\n"
            "GPU6 NV18 NV18 X NV18 0-31\n"
            "GPU7 NV18 NV18 NV18 X 0-31\n"
        ),
    }


def _plan_kwargs(runner):
    remote_root = runner.APPROVED_REMOTE_ROOT
    return {
        "ssh_target": "sitian@10.232.195.203",
        "remote_root": remote_root,
        "model_root": (
            f"{remote_root}/models/"
            "Qwen3.8-27B/snapshots/0123456789abcdef"
        ),
        "attempt_tag": "qwen38-tp4-comm-20260826-r001",
        "source_revision": "a" * 40,
        "model_revision": "b" * 40,
        "selected_gpus": _selected(runner),
        "gpu_topology": _topology(),
        "command_timeout_s": 7200,
        "retry_count": 2,
    }


def test_build_attempt_plan_keeps_every_path_under_approved_root():
    runner = _load()

    plan = runner.build_attempt_plan(**_plan_kwargs(runner))

    attempt_root = (
        f"{runner.APPROVED_REMOTE_ROOT}/attempts/"
        "qwen38-tp4-comm-20260826-r001"
    )
    assert plan["schema_version"] == (
        "qwen38.tp4-communication-profile-plan.v1"
    )
    assert plan["attempt_root"] == attempt_root
    assert plan["temporary_root"] == f"{attempt_root}/.staging"
    assert plan["artifact_root"] == f"{attempt_root}/artifacts"
    assert plan["nsys_root"] == f"{attempt_root}/nsys"
    assert plan["source_revision"] == "a" * 40
    assert plan["model_revision"] == "b" * 40
    assert plan["command_timeout_s"] == 7200
    assert plan["retry_count"] == 2
    assert plan["gpu_rank_mapping"] == [
        {
            "rank": rank,
            "gpu_index": gpu["gpu_index"],
            "gpu_uuid": gpu["gpu_uuid"],
        }
        for rank, gpu in enumerate(_selected(runner))
    ]
    assert len(plan["workload_cases"]) == 60
    assert all(
        runner.is_path_below_approved_remote_root(value)
        for key, value in plan.items()
        if key.endswith("_root")
    )
    assert all(
        isinstance(command["argv"], list)
        and command["argv"]
        and all(isinstance(value, str) for value in command["argv"])
        for command in plan["commands"]
    )
    assert plan["commands"][0] == {
        "name": "create-attempt-root",
        "argv": ["mkdir", "--", attempt_root],
    }
    flattened = [
        value
        for command in plan["commands"]
        for value in command["argv"]
    ]
    assert "kinit" not in flattened
    assert "kill" not in flattened
    assert "pkill" not in flattened
    assert "killall" not in flattened
    assert plan["manifests"]["source"]["revision"] == "a" * 40
    assert plan["manifests"]["model"] == {
        "root": _plan_kwargs(runner)["model_root"],
        "revision": "b" * 40,
    }
    assert plan["manifests"]["environment"] == {
        "dtype": "bfloat16",
        "tensor_parallel_size": 4,
        "decoding": "greedy",
        "temperature": 0.0,
        "fixed_output_tokens": 128,
        "scheduler_policy": "identical",
        "cuda_graph_policy": "identical",
    }
    assert plan["manifests"]["workloads"]["counts"] == {
        "warmup": 10,
        "measured": 25,
        "nsys_replay": 25,
        "overhead_pairs": 25,
    }
    assert plan["manifests"]["topology"]["strict_clean_limits"] == {
        "maximum_memory_used_mib": 1024,
        "maximum_utilization_percent": 5,
        "compute_processes": [],
    }
    assert plan["manifests"]["topology"]["gpu_rows"] == (
        _topology()["gpu_rows"]
    )
    assert plan["manifests"]["topology"]["interconnect_matrix"] == (
        _topology()["interconnect_matrix"]
    )


def test_build_attempt_plan_rejects_topology_identity_drift():
    runner = _load()
    kwargs = _plan_kwargs(runner)
    kwargs["gpu_topology"]["gpu_rows"][0]["gpu_uuid"] = "GPU-other"

    with pytest.raises(ValueError, match="topology"):
        runner.build_attempt_plan(**kwargs)


def test_build_attempt_plan_rejects_topology_matrix_without_selected_gpu():
    runner = _load()
    kwargs = _plan_kwargs(runner)
    kwargs["gpu_topology"]["interconnect_matrix"] = (
        kwargs["gpu_topology"]["interconnect_matrix"].replace(
            "GPU7",
            "GPU9",
        )
    )

    with pytest.raises(ValueError, match="topology"):
        runner.build_attempt_plan(**kwargs)


@pytest.mark.parametrize(
    "remote_root",
    [
        "/",
        "/tmp",
        "/private/tmp",
        "/data00/home/sitian/old-checkout",
        (
            "/data00/home/sitian/tinyllmforge-workspaces/"
            "command-timeline-20260818/../escape"
        ),
        (
            "/data00/home/sitian/tinyllmforge-workspaces/"
            "command-timeline-20260818-adaptive-ngram"
        ),
    ],
)
def test_build_attempt_plan_rejects_unapproved_remote_root(remote_root):
    runner = _load()
    kwargs = _plan_kwargs(runner)
    kwargs["remote_root"] = remote_root

    with pytest.raises(ValueError, match="approved remote root"):
        runner.build_attempt_plan(**kwargs)


@pytest.mark.parametrize(
    "attempt_tag",
    [
        "",
        "../escape",
        "nested/attempt",
        "attempt;touch-pwned",
        "attempt$(id)",
        "attempt`id`",
        "attempt with spaces",
        "-option",
    ],
)
def test_build_attempt_plan_rejects_unsafe_attempt_tag(attempt_tag):
    runner = _load()
    kwargs = _plan_kwargs(runner)
    kwargs["attempt_tag"] = attempt_tag

    with pytest.raises(ValueError, match="attempt tag"):
        runner.build_attempt_plan(**kwargs)


def test_build_attempt_plan_rejects_model_path_outside_root():
    runner = _load()
    kwargs = _plan_kwargs(runner)
    kwargs["model_root"] = "/tmp/Qwen3.8-27B"

    with pytest.raises(ValueError, match="approved remote root"):
        runner.build_attempt_plan(**kwargs)


def test_build_attempt_plan_rejects_remote_symlink_escape():
    runner = _load()
    kwargs = _plan_kwargs(runner)

    def resolve(path):
        if "/models/" in path:
            return "/tmp/escaped-model"
        return path

    kwargs["resolve_remote_path"] = resolve
    with pytest.raises(ValueError, match="symlink escape"):
        runner.build_attempt_plan(**kwargs)


def test_build_attempt_plan_rejects_attempt_parent_symlink_escape():
    runner = _load()
    kwargs = _plan_kwargs(runner)

    def resolve(path):
        if "/attempts/" in path:
            return "/tmp/escaped-attempt"
        return path

    kwargs["resolve_remote_path"] = resolve
    with pytest.raises(ValueError, match="symlink escape"):
        runner.build_attempt_plan(**kwargs)


def test_build_attempt_plan_rejects_reused_attempt_tag():
    runner = _load()
    kwargs = _plan_kwargs(runner)
    kwargs["attempt_exists"] = lambda path: True

    with pytest.raises(ValueError, match="already exists"):
        runner.build_attempt_plan(**kwargs)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("command_timeout_s", True),
        ("command_timeout_s", 0),
        ("command_timeout_s", -1),
        ("retry_count", True),
        ("retry_count", 0),
        ("retry_count", 11),
    ],
)
def test_build_attempt_plan_rejects_unbounded_execution_policy(
    field,
    value,
):
    runner = _load()
    kwargs = _plan_kwargs(runner)
    kwargs[field] = value

    with pytest.raises(ValueError, match="execution policy"):
        runner.build_attempt_plan(**kwargs)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda plan: plan.__setitem__(
            "ssh_target",
            "not a valid target",
        ),
        lambda plan: plan["commands"][0].__setitem__(
            "argv",
            ["mkdir", "--", plan["artifact_root"]],
        ),
        lambda plan: plan["commands"].append({
            "name": "unexpected",
            "argv": ["true"],
        }),
        lambda plan: plan.__setitem__(
            "benchmark_execution_authorized",
            True,
        ),
        lambda plan: plan.__setitem__("unexpected", "field"),
        lambda plan: plan["gpu_rank_mapping"][1].__setitem__(
            "gpu_index",
            plan["gpu_rank_mapping"][0]["gpu_index"],
        ),
    ],
)
def test_attempt_plan_validation_rejects_identity_or_command_drift(
    mutate,
):
    runner = _load()
    plan = runner.build_attempt_plan(**_plan_kwargs(runner))
    mutate(plan)

    with pytest.raises(ValueError, match="attempt plan"):
        runner.run_attempt(
            plan=plan,
            plan_only=True,
            query_inventory=lambda: pytest.fail(
                "tampered plan must fail before GPU query"
            ),
            run_correctness=lambda **kwargs: None,
            run_case=lambda **kwargs: None,
            cleanup_owned=lambda **kwargs: None,
        )


def test_cli_exposes_required_remote_and_bounded_execution_options():
    runner = _load()
    parser = runner.build_parser()

    arguments = parser.parse_args([
        "--ssh-target",
        "sitian@10.232.195.203",
        "--remote-root",
        runner.APPROVED_REMOTE_ROOT,
        "--model-root",
        (
            f"{runner.APPROVED_REMOTE_ROOT}/models/"
            "Qwen3.8-27B/snapshots/0123456789abcdef"
        ),
        "--attempt-tag",
        "qwen38-tp4-comm-20260826-r001",
        "--source-revision",
        "a" * 40,
        "--model-revision",
        "b" * 40,
        "--command-timeout-s",
        "7200",
        "--retry-count",
        "2",
        "--dry-run",
        "--plan-only",
    ])

    assert arguments.ssh_target == "sitian@10.232.195.203"
    assert arguments.remote_root == runner.APPROVED_REMOTE_ROOT
    assert arguments.attempt_tag.endswith("r001")
    assert arguments.source_revision == "a" * 40
    assert arguments.model_revision == "b" * 40
    assert arguments.command_timeout_s == 7200
    assert arguments.retry_count == 2
    assert arguments.dry_run is True
    assert arguments.plan_only is True


def test_main_plan_only_emits_immutable_plan_without_launching(capsys):
    runner = _load()
    events = []

    exit_code = runner.main(
        [
            "--ssh-target",
            "sitian@10.232.195.203",
            "--remote-root",
            runner.APPROVED_REMOTE_ROOT,
            "--model-root",
            (
                f"{runner.APPROVED_REMOTE_ROOT}/models/"
                "Qwen3.8-27B/snapshots/0123456789abcdef"
            ),
            "--attempt-tag",
            "qwen38-tp4-comm-20260826-r001",
            "--source-revision",
            "a" * 40,
            "--model-revision",
            "b" * 40,
            "--command-timeout-s",
            "7200",
            "--retry-count",
            "2",
            "--plan-only",
        ],
        inventory_query=lambda **kwargs: (
            events.append(("inventory", kwargs))
            or _clean_inventory()
        ),
        topology_query=lambda **kwargs: (
            events.append(("topology", kwargs)) or _topology()
        ),
        path_state_query=lambda **kwargs: (
            events.append(("paths", kwargs))
            or {
                "resolved_paths": {
                    "remote_root": runner.APPROVED_REMOTE_ROOT,
                    "model_root": kwargs["model_root"],
                    "attempt_root": (
                        f"{runner.APPROVED_REMOTE_ROOT}/attempts/"
                        "qwen38-tp4-comm-20260826-r001"
                    ),
                },
                "attempt_exists": False,
            }
        ),
        kerberos_query=lambda **kwargs: events.append("kerberos"),
    )

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert payload["classification"] == "PLAN_ONLY"
    assert payload["plan"]["source_revision"] == "a" * 40
    assert payload["plan"]["model_revision"] == "b" * 40
    assert payload["plan"]["benchmark_execution_authorized"] is False
    assert [event[0] for event in events] == [
        "paths",
        "inventory",
        "topology",
    ]


def test_main_checks_kerberos_before_remote_gpu_inventory(capsys):
    runner = _load()
    events = []

    exit_code = runner.main(
        [
            "--model-root",
            (
                f"{runner.APPROVED_REMOTE_ROOT}/models/"
                "Qwen3.8-27B/snapshots/0123456789abcdef"
            ),
            "--attempt-tag",
            "qwen38-tp4-comm-20260826-r001",
            "--source-revision",
            "a" * 40,
            "--model-revision",
            "b" * 40,
            "--dry-run",
        ],
        inventory_query=lambda **kwargs: events.append("inventory"),
        kerberos_query=lambda **kwargs: {
            "classification": "BLOCKED_KERBEROS_TTL",
            "reason": "lifetime is insufficient",
        },
    )

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 2
    assert payload["classification"] == "BLOCKED_KERBEROS_TTL"
    assert events == []


def test_main_rejects_remote_path_escape_before_gpu_inventory():
    runner = _load()
    model_root = (
        f"{runner.APPROVED_REMOTE_ROOT}/models/"
        "Qwen3.8-27B/snapshots/0123456789abcdef"
    )

    with pytest.raises(ValueError, match="symlink escape"):
        runner.main(
            [
                "--model-root",
                model_root,
                "--attempt-tag",
                "qwen38-tp4-comm-20260826-r001",
                "--source-revision",
                "a" * 40,
                "--model-revision",
                "b" * 40,
                "--plan-only",
            ],
            path_state_query=lambda **kwargs: {
                "resolved_paths": {
                    "remote_root": runner.APPROVED_REMOTE_ROOT,
                    "model_root": model_root,
                    "attempt_root": "/tmp/escaped-attempt",
                },
                "attempt_exists": False,
            },
            inventory_query=lambda **kwargs: pytest.fail(
                "unsafe remote path must fail before GPU query"
            ),
        )


def test_main_dry_run_uses_bounded_local_gpu_monitor(capsys):
    runner = _load()
    monitor_calls = []
    inventory_calls = []

    exit_code = runner.main(
        [
            "--model-root",
            (
                f"{runner.APPROVED_REMOTE_ROOT}/models/"
                "Qwen3.8-27B/snapshots/0123456789abcdef"
            ),
            "--attempt-tag",
            "qwen38-tp4-comm-20260826-r001",
            "--source-revision",
            "a" * 40,
            "--model-revision",
            "b" * 40,
            "--gpu-wait-timeout-s",
            "900",
            "--gpu-poll-interval-s",
            "15",
            "--dry-run",
        ],
        inventory_query=lambda **kwargs: (
            inventory_calls.append(kwargs) or _clean_inventory()
        ),
        topology_query=lambda **kwargs: _topology(),
        path_state_query=lambda **kwargs: {
            "resolved_paths": {
                "remote_root": runner.APPROVED_REMOTE_ROOT,
                "model_root": kwargs["model_root"],
                "attempt_root": (
                    f"{runner.APPROVED_REMOTE_ROOT}/attempts/"
                    "qwen38-tp4-comm-20260826-r001"
                ),
            },
            "attempt_exists": False,
        },
        kerberos_query=lambda **kwargs: _ready_kerberos(),
        gpu_monitor=lambda **kwargs: (
            monitor_calls.append(kwargs)
            or {
                "classification": "READY",
                "selected_gpus": _clean_inventory(),
                "samples": [],
            }
        ),
    )

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert payload["classification"] == "DRY_RUN_READY"
    assert monitor_calls[0]["timeout_s"] == 900
    assert monitor_calls[0]["poll_interval_s"] == 15
    assert callable(monitor_calls[0]["query_inventory"])
    assert len(inventory_calls) == 1


def test_main_fails_closed_without_production_worker_adapters(capsys):
    runner = _load()

    exit_code = runner.main(
        [
            "--model-root",
            (
                f"{runner.APPROVED_REMOTE_ROOT}/models/"
                "Qwen3.8-27B/snapshots/0123456789abcdef"
            ),
            "--attempt-tag",
            "qwen38-tp4-comm-20260826-r001",
            "--source-revision",
            "a" * 40,
            "--model-revision",
            "b" * 40,
        ],
        inventory_query=lambda **kwargs: _clean_inventory(),
        topology_query=lambda **kwargs: _topology(),
        path_state_query=lambda **kwargs: {
            "resolved_paths": {
                "remote_root": runner.APPROVED_REMOTE_ROOT,
                "model_root": kwargs["model_root"],
                "attempt_root": (
                    f"{runner.APPROVED_REMOTE_ROOT}/attempts/"
                    "qwen38-tp4-comm-20260826-r001"
                ),
            },
            "attempt_exists": False,
        },
        kerberos_query=lambda **kwargs: _ready_kerberos(),
        gpu_monitor=lambda **kwargs: {
            "classification": "READY",
            "selected_gpus": _clean_inventory(),
            "samples": [],
        },
    )

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 2
    assert payload["classification"] == (
        "EXECUTION_ADAPTER_UNAVAILABLE"
    )
    assert payload["benchmark_execution_authorized"] is False


def test_plan_only_does_not_query_or_launch_remote_work():
    runner = _load()
    events = []

    result = runner.run_attempt(
        plan=runner.build_attempt_plan(**_plan_kwargs(runner)),
        plan_only=True,
        query_inventory=lambda: events.append("query"),
        run_correctness=lambda **kwargs: events.append("correctness"),
        run_case=lambda **kwargs: events.append("case"),
        cleanup_owned=lambda **kwargs: events.append("cleanup"),
    )

    assert result["classification"] == "PLAN_ONLY"
    assert result["benchmark_execution_authorized"] is False
    assert events == []


def test_write_json_atomic_never_leaves_partial_file(tmp_path):
    runner = _load()
    target = tmp_path / "plan.json"

    runner.write_json_atomic(target, {"z": 1, "a": [2, 3]})

    assert target.read_text(encoding="utf-8") == (
        '{"a":[2,3],"z":1}\n'
    )
    assert list(tmp_path.iterdir()) == [target]


def _clean_inventory():
    return [_gpu(index) for index in (2, 4, 6, 7)]


def _ready_kerberos():
    return {
        "classification": "READY",
        "remaining_lifetime_seconds": 7200,
        "minimum_required_lifetime_seconds": 5400,
    }


def test_worker_entry_validates_planned_gpus_without_reselecting():
    runner = _load()
    plan = runner.build_attempt_plan(**_plan_kwargs(runner))
    inventory = [
        _gpu(index) for index in (0, 1, 2, 4, 6, 7)
    ]

    selected = runner._guard_planned_gpus(plan, inventory)

    assert [row["gpu_index"] for row in selected] == [2, 4, 6, 7]


def test_worker_entry_rejects_dirty_planned_gpu_even_with_replacements():
    runner = _load()
    plan = runner.build_attempt_plan(**_plan_kwargs(runner))
    inventory = [
        _gpu(index) for index in (0, 1, 2, 4, 6, 7)
    ]
    inventory[-1]["memory_used_mib"] = 1025

    with pytest.raises(ValueError, match="strict-clean"):
        runner._guard_planned_gpus(plan, inventory)


def _successful_case_result(runner, case, owned_pid):
    result = {
        "classification": "PASS",
        "case_id": case.case_id,
        "owned_pids": [owned_pid],
        "resource_samples": [_clean_inventory()],
        "process_group_destroyed": True,
        "owned_children_remaining": [],
    }
    if case.phase in {"measured", "nsys_replay"}:
        result["decode_time_ms"] = float(
            100 + case.repetition * 10
        )
    if case.phase == "nsys_replay":
        result["sqlite_path"] = (
            f"{runner.APPROVED_REMOTE_ROOT}/attempts/"
            "qwen38-tp4-comm-20260826-r001/nsys/"
            f"{case.case_id}.sqlite"
        )
    return result


def test_run_attempt_executes_correctness_then_all_structured_then_all_replays():
    runner = _load()
    plan = runner.build_attempt_plan(**_plan_kwargs(runner))
    events = []
    cleanup_calls = []
    next_pid = iter(range(10000, 20000))

    def query_inventory():
        events.append("guard")
        return _clean_inventory()

    def run_correctness(**kwargs):
        events.append("correctness")
        return {
            "classification": "PASS",
            "owned_pids": [9001],
            "resource_samples": [_clean_inventory()],
            "process_group_destroyed": True,
            "owned_children_remaining": [],
        }

    def run_case(*, case, **kwargs):
        events.append((case.phase, case.case_id))
        return _successful_case_result(
            runner,
            case,
            next(next_pid),
        )

    result = runner.run_attempt(
        plan=plan,
        plan_only=False,
        kerberos_status=_ready_kerberos(),
        query_inventory=query_inventory,
        run_correctness=run_correctness,
        run_case=run_case,
        cleanup_owned=lambda **kwargs: (
            cleanup_calls.append(kwargs)
            or {
                "classification": "CLEAN",
                "signaled_pids": [],
            }
        ),
    )

    assert result["classification"] == "COMPLETE"
    assert result["correctness"]["classification"] == "PASS"
    phases = [
        event[0] for event in events if isinstance(event, tuple)
    ]
    assert phases == [
        case.phase
        for case in runner.build_workload_cases()
        if case.phase != "nsys_replay"
    ] + ["nsys_replay"] * 25
    assert [
        event[1] for event in events if isinstance(event, tuple)
    ] == [
        case.case_id
        for case in runner.build_workload_cases()
        if case.phase != "nsys_replay"
    ] + [
        case.case_id
        for case in runner.build_workload_cases()
        if case.phase == "nsys_replay"
    ]
    assert events.index("correctness") < next(
        index
        for index, event in enumerate(events)
        if isinstance(event, tuple)
    )
    assert len(result["structured_results"]) == 35
    assert len(result["nsys_results"]) == 25
    assert len({
        row["sqlite_path"] for row in result["nsys_results"]
    }) == 25
    assert sum(
        row["representative"]
        for row in result["nsys_results"]
    ) == 5
    assert len(result["overhead_controls"]) == 25
    assert all(
        row["profiled_decode_time_ms"] == row[
            "unprofiled_decode_time_ms"
        ]
        for row in result["overhead_controls"]
    )
    assert result["cleanup"]["classification"] == "CLEAN"
    assert len(cleanup_calls) == 1
    assert cleanup_calls[0]["normal_control_only"] is True
    assert cleanup_calls[0]["owned_pids"] == frozenset(
        {9001, *range(10000, 10060)}
    )


def test_run_attempt_stops_before_workloads_when_correctness_fails():
    runner = _load()
    events = []

    result = runner.run_attempt(
        plan=runner.build_attempt_plan(**_plan_kwargs(runner)),
        plan_only=False,
        kerberos_status=_ready_kerberos(),
        query_inventory=lambda: _clean_inventory(),
        run_correctness=lambda **kwargs: {
            "classification": "INVALID_CORRECTNESS",
            "owned_pids": [9010],
            "resource_samples": [_clean_inventory()],
            "process_group_destroyed": True,
            "owned_children_remaining": [],
        },
        run_case=lambda **kwargs: events.append("case"),
        cleanup_owned=lambda **kwargs: {
            "classification": "CLEAN",
            "signaled_pids": [],
        },
    )

    assert result["classification"] == "INVALID_CORRECTNESS"
    assert result["preserve_attempt"] is True
    assert events == []


def test_run_attempt_invalidates_on_unrelated_process_and_cleans_only_owned():
    runner = _load()
    plan = runner.build_attempt_plan(**_plan_kwargs(runner))
    observations = 0
    cleanup_calls = []

    def query_inventory():
        nonlocal observations
        observations += 1
        rows = _clean_inventory()
        if observations == 3:
            rows[0]["compute_processes"] = [
                {"pid": 9999, "process_name": "external"},
            ]
        return rows

    result = runner.run_attempt(
        plan=plan,
        plan_only=False,
        kerberos_status=_ready_kerberos(),
        query_inventory=query_inventory,
        run_correctness=lambda **kwargs: {
            "classification": "PASS",
            "owned_pids": [9011],
            "resource_samples": [_clean_inventory()],
            "process_group_destroyed": True,
            "owned_children_remaining": [],
        },
        run_case=lambda **kwargs: pytest.fail(
            "workload must not launch after resource identity loss"
        ),
        cleanup_owned=lambda **kwargs: (
            cleanup_calls.append(kwargs)
            or {
                "classification": "CLEAN",
                "signaled_pids": [9011],
            }
        ),
    )

    assert result["classification"] == "INVALID_RESOURCE_IDENTITY"
    assert result["preserve_attempt"] is True
    assert cleanup_calls == [{
        "owned_pids": frozenset({9011}),
        "normal_control_only": True,
    }]


def test_run_attempt_rejects_unrelated_pid_seen_in_worker_resource_sample():
    runner = _load()
    cleanup_calls = []
    sampled = _clean_inventory()
    sampled[0]["compute_processes"] = [{
        "pid": 9999,
        "process_name": "external",
        "used_memory_mib": 256,
    }]

    result = runner.run_attempt(
        plan=runner.build_attempt_plan(**_plan_kwargs(runner)),
        plan_only=False,
        kerberos_status=_ready_kerberos(),
        query_inventory=lambda: _clean_inventory(),
        run_correctness=lambda **kwargs: {
            "classification": "PASS",
            "owned_pids": [9015],
            "resource_samples": [sampled],
            "process_group_destroyed": True,
            "owned_children_remaining": [],
        },
        run_case=lambda **kwargs: pytest.fail(
            "workload must not run after sampled resource identity loss"
        ),
        cleanup_owned=lambda **kwargs: (
            cleanup_calls.append(kwargs)
            or {
                "classification": "CLEAN",
                "signaled_pids": [9015],
            }
        ),
    )

    assert result["classification"] == "INVALID_RESOURCE_IDENTITY"
    assert cleanup_calls[0]["owned_pids"] == frozenset({9015})


def test_run_attempt_rejects_owned_pid_still_present_after_worker_exit():
    runner = _load()
    observations = 0

    def query_inventory():
        nonlocal observations
        observations += 1
        rows = _clean_inventory()
        if observations == 3:
            rows[0]["compute_processes"] = [{
                "pid": 9018,
                "process_name": "stale-owned-worker",
                "used_memory_mib": 256,
            }]
        return rows

    result = runner.run_attempt(
        plan=runner.build_attempt_plan(**_plan_kwargs(runner)),
        plan_only=False,
        kerberos_status=_ready_kerberos(),
        query_inventory=query_inventory,
        run_correctness=lambda **kwargs: {
            "classification": "PASS",
            "owned_pids": [9018],
            "resource_samples": [_clean_inventory()],
            "process_group_destroyed": True,
            "owned_children_remaining": [],
        },
        run_case=lambda **kwargs: pytest.fail(
            "workload must not run while prior worker remains"
        ),
        cleanup_owned=lambda **kwargs: {
            "classification": "CLEAN",
            "signaled_pids": [9018],
        },
    )

    assert result["classification"] == "INVALID_RESOURCE_IDENTITY"
    assert "strict-clean" in result["reason"]


def test_run_attempt_requires_worker_resource_samples():
    runner = _load()

    result = runner.run_attempt(
        plan=runner.build_attempt_plan(**_plan_kwargs(runner)),
        plan_only=False,
        kerberos_status=_ready_kerberos(),
        query_inventory=lambda: _clean_inventory(),
        run_correctness=lambda **kwargs: {
            "classification": "PASS",
            "owned_pids": [9016],
            "process_group_destroyed": True,
            "owned_children_remaining": [],
        },
        run_case=lambda **kwargs: pytest.fail(
            "workload must not run without runtime resource evidence"
        ),
        cleanup_owned=lambda **kwargs: {
            "classification": "CLEAN",
            "signaled_pids": [9016],
        },
    )

    assert result["classification"] == "FAILED_EXECUTION"
    assert "resource samples" in result["reason"]


def test_run_attempt_rejects_malformed_worker_resource_process():
    runner = _load()
    sampled = _clean_inventory()
    sampled[0]["compute_processes"] = [{
        "pid": 9017,
        "process_name": "owned-worker",
    }]

    result = runner.run_attempt(
        plan=runner.build_attempt_plan(**_plan_kwargs(runner)),
        plan_only=False,
        kerberos_status=_ready_kerberos(),
        query_inventory=lambda: _clean_inventory(),
        run_correctness=lambda **kwargs: {
            "classification": "PASS",
            "owned_pids": [9017],
            "resource_samples": [sampled],
            "process_group_destroyed": True,
            "owned_children_remaining": [],
        },
        run_case=lambda **kwargs: pytest.fail(
            "workload must not run after malformed resource evidence"
        ),
        cleanup_owned=lambda **kwargs: {
            "classification": "CLEAN",
            "signaled_pids": [9017],
        },
    )

    assert result["classification"] == "INVALID_RESOURCE_IDENTITY"
    assert "process telemetry" in result["reason"]


def test_run_attempt_rejects_cleanup_that_signals_unowned_pid():
    runner = _load()

    with pytest.raises(ValueError, match="unowned PID"):
        runner.run_attempt(
            plan=runner.build_attempt_plan(**_plan_kwargs(runner)),
            plan_only=False,
            kerberos_status=_ready_kerberos(),
            query_inventory=lambda: _clean_inventory(),
            run_correctness=lambda **kwargs: {
                "classification": "INVALID_CORRECTNESS",
                "owned_pids": [9012],
                "resource_samples": [_clean_inventory()],
                "process_group_destroyed": True,
                "owned_children_remaining": [],
            },
            run_case=lambda **kwargs: None,
            cleanup_owned=lambda **kwargs: {
                "classification": "CLEAN",
                "signaled_pids": [7777],
            },
        )


def test_run_attempt_marks_incomplete_nsys_export_inconclusive():
    runner = _load()
    next_pid = iter(range(11000, 12000))

    def run_case(*, case, **kwargs):
        result = _successful_case_result(
            runner,
            case,
            next(next_pid),
        )
        if case.phase == "nsys_replay" and case.workload == "Q2":
            result.pop("sqlite_path")
        return result

    result = runner.run_attempt(
        plan=runner.build_attempt_plan(**_plan_kwargs(runner)),
        plan_only=False,
        kerberos_status=_ready_kerberos(),
        query_inventory=lambda: _clean_inventory(),
        run_correctness=lambda **kwargs: {
            "classification": "PASS",
            "owned_pids": [],
            "resource_samples": [_clean_inventory()],
            "process_group_destroyed": True,
            "owned_children_remaining": [],
        },
        run_case=run_case,
        cleanup_owned=lambda **kwargs: {
            "classification": "CLEAN",
            "signaled_pids": [],
        },
    )

    assert result["classification"] == "INCONCLUSIVE_TRACE_COVERAGE"
    assert len(result["nsys_results"]) < 25
    assert result["preserve_attempt"] is True


def test_run_attempt_rejects_nsys_path_with_traversal():
    runner = _load()
    next_pid = iter(range(12000, 13000))

    def run_case(*, case, **kwargs):
        result = _successful_case_result(
            runner,
            case,
            next(next_pid),
        )
        if case.phase == "nsys_replay" and case.case_id.endswith("r0"):
            result["sqlite_path"] = (
                f"{runner.APPROVED_REMOTE_ROOT}/attempts/"
                "qwen38-tp4-comm-20260826-r001/nsys/"
                "../artifacts/escaped.sqlite"
            )
        return result

    result = runner.run_attempt(
        plan=runner.build_attempt_plan(**_plan_kwargs(runner)),
        plan_only=False,
        kerberos_status=_ready_kerberos(),
        query_inventory=lambda: _clean_inventory(),
        run_correctness=lambda **kwargs: {
            "classification": "PASS",
            "owned_pids": [],
            "resource_samples": [_clean_inventory()],
            "process_group_destroyed": True,
            "owned_children_remaining": [],
        },
        run_case=run_case,
        cleanup_owned=lambda **kwargs: {
            "classification": "CLEAN",
            "signaled_pids": [],
        },
    )

    assert result["classification"] == "INCONCLUSIVE_TRACE_COVERAGE"
    assert "SQLite export" in result["reason"]
    assert result["nsys_results"] == []


def test_run_attempt_rejects_nsys_export_without_sqlite_suffix():
    runner = _load()
    next_pid = iter(range(13000, 14000))

    def run_case(*, case, **kwargs):
        result = _successful_case_result(
            runner,
            case,
            next(next_pid),
        )
        if case.phase == "nsys_replay" and case.case_id.endswith("r0"):
            result["sqlite_path"] = (
                f"{runner.APPROVED_REMOTE_ROOT}/attempts/"
                "qwen38-tp4-comm-20260826-r001/nsys/"
                f"{case.case_id}.txt"
            )
        return result

    result = runner.run_attempt(
        plan=runner.build_attempt_plan(**_plan_kwargs(runner)),
        plan_only=False,
        kerberos_status=_ready_kerberos(),
        query_inventory=lambda: _clean_inventory(),
        run_correctness=lambda **kwargs: {
            "classification": "PASS",
            "owned_pids": [],
            "resource_samples": [_clean_inventory()],
            "process_group_destroyed": True,
            "owned_children_remaining": [],
        },
        run_case=run_case,
        cleanup_owned=lambda **kwargs: {
            "classification": "CLEAN",
            "signaled_pids": [],
        },
    )

    assert result["classification"] == "INCONCLUSIVE_TRACE_COVERAGE"
    assert result["nsys_results"] == []


def test_dry_run_checks_admission_but_launches_no_worker():
    runner = _load()
    events = []

    result = runner.run_attempt(
        plan=runner.build_attempt_plan(**_plan_kwargs(runner)),
        plan_only=False,
        dry_run=True,
        kerberos_status=_ready_kerberos(),
        query_inventory=lambda: (
            events.append("query") or _clean_inventory()
        ),
        run_correctness=lambda **kwargs: events.append("correctness"),
        run_case=lambda **kwargs: events.append("case"),
        cleanup_owned=lambda **kwargs: events.append("cleanup"),
    )

    assert result["classification"] == "DRY_RUN_READY"
    assert result["benchmark_execution_authorized"] is False
    assert events == ["query"]


def test_run_attempt_blocks_before_gpu_query_when_kerberos_ttl_is_low():
    runner = _load()
    events = []

    result = runner.run_attempt(
        plan=runner.build_attempt_plan(**_plan_kwargs(runner)),
        plan_only=False,
        kerberos_status={
            "classification": "BLOCKED_KERBEROS_TTL",
            "reason": "lifetime is insufficient",
        },
        query_inventory=lambda: events.append("query"),
        run_correctness=lambda **kwargs: events.append("correctness"),
        run_case=lambda **kwargs: events.append("case"),
        cleanup_owned=lambda **kwargs: events.append("cleanup"),
    )

    assert result["classification"] == "BLOCKED_KERBEROS_TTL"
    assert result["preserve_attempt"] is True
    assert events == []


def test_dry_run_reports_blocked_resources_without_launching():
    runner = _load()
    events = []

    result = runner.run_attempt(
        plan=runner.build_attempt_plan(**_plan_kwargs(runner)),
        plan_only=False,
        dry_run=True,
        kerberos_status=_ready_kerberos(),
        query_inventory=lambda: [_gpu(index) for index in range(3)],
        run_correctness=lambda **kwargs: events.append("correctness"),
        run_case=lambda **kwargs: events.append("case"),
        cleanup_owned=lambda **kwargs: events.append("cleanup"),
    )

    assert result["classification"] == "BLOCKED_RESOURCES"
    assert result["preserve_attempt"] is True
    assert events == []


def test_wait_for_strict_clean_gpus_polls_locally_until_ready():
    runner = _load()
    observations = iter((
        [_gpu(index) for index in range(3)],
        _clean_inventory(),
    ))
    sleeps = []
    clock = iter((0.0, 0.0, 5.0, 5.0))

    result = runner.wait_for_strict_clean_gpus(
        query_inventory=lambda: next(observations),
        timeout_s=10,
        poll_interval_s=5,
        sleep=lambda seconds: sleeps.append(seconds),
        monotonic=lambda: next(clock),
    )

    assert result["classification"] == "READY"
    assert len(result["samples"]) == 2
    assert sleeps == [5]
    assert [row["gpu_index"] for row in result["selected_gpus"]] == [
        2,
        4,
        6,
        7,
    ]


@pytest.mark.parametrize(
    ("timeout_s", "poll_interval_s"),
    [
        (-1, 5),
        (86401, 5),
        (10, 0),
        (10, 86401),
    ],
)
def test_gpu_monitor_rejects_unbounded_policy(
    timeout_s,
    poll_interval_s,
):
    runner = _load()

    with pytest.raises(ValueError, match="monitor policy"):
        runner.wait_for_strict_clean_gpus(
            query_inventory=_clean_inventory,
            timeout_s=timeout_s,
            poll_interval_s=poll_interval_s,
        )


def test_run_attempt_reports_entry_resource_block_without_launching():
    runner = _load()
    events = []

    result = runner.run_attempt(
        plan=runner.build_attempt_plan(**_plan_kwargs(runner)),
        plan_only=False,
        kerberos_status=_ready_kerberos(),
        query_inventory=lambda: [_gpu(index) for index in range(3)],
        run_correctness=lambda **kwargs: events.append("correctness"),
        run_case=lambda **kwargs: events.append("case"),
        cleanup_owned=lambda **kwargs: events.append("cleanup"),
    )

    assert result["classification"] == "BLOCKED_RESOURCES"
    assert result["preserve_attempt"] is True
    assert events == []


def test_run_attempt_rejects_tampered_plan_before_query():
    runner = _load()
    plan = runner.build_attempt_plan(**_plan_kwargs(runner))
    plan["nsys_root"] = "/tmp/escape"
    events = []

    with pytest.raises(ValueError, match="attempt plan"):
        runner.run_attempt(
            plan=plan,
            plan_only=False,
            kerberos_status=_ready_kerberos(),
            query_inventory=lambda: events.append("query"),
            run_correctness=lambda **kwargs: None,
            run_case=lambda **kwargs: None,
            cleanup_owned=lambda **kwargs: None,
        )

    assert events == []


def test_run_attempt_fails_closed_on_worker_cleanup_gap():
    runner = _load()
    cleanup_calls = []

    result = runner.run_attempt(
        plan=runner.build_attempt_plan(**_plan_kwargs(runner)),
        plan_only=False,
        kerberos_status=_ready_kerberos(),
        query_inventory=lambda: _clean_inventory(),
        run_correctness=lambda **kwargs: {
            "classification": "PASS",
            "owned_pids": [9020],
            "resource_samples": [_clean_inventory()],
            "process_group_destroyed": False,
            "owned_children_remaining": [9021],
        },
        run_case=lambda **kwargs: pytest.fail(
            "workload must not run after cleanup gap"
        ),
        cleanup_owned=lambda **kwargs: (
            cleanup_calls.append(kwargs)
            or {
                "classification": "CLEAN",
                "signaled_pids": [9020],
            }
        ),
    )

    assert result["classification"] == "FAILED_WORKER_CLEANUP"
    assert cleanup_calls[0]["owned_pids"] == frozenset({9020})


def test_run_attempt_preserves_failure_receipt_after_worker_exception():
    runner = _load()
    cleanup_calls = []

    result = runner.run_attempt(
        plan=runner.build_attempt_plan(**_plan_kwargs(runner)),
        plan_only=False,
        kerberos_status=_ready_kerberos(),
        query_inventory=lambda: _clean_inventory(),
        run_correctness=lambda **kwargs: (_ for _ in ()).throw(
            RuntimeError("transport dropped")
        ),
        run_case=lambda **kwargs: None,
        cleanup_owned=lambda **kwargs: (
            cleanup_calls.append(kwargs)
            or {
                "classification": "CLEAN",
                "signaled_pids": [],
            }
        ),
    )

    assert result["classification"] == "FAILED_EXECUTION"
    assert "transport dropped" in result["reason"]
    assert cleanup_calls == [{
        "owned_pids": frozenset(),
        "normal_control_only": True,
    }]


def test_failure_cleanup_gap_takes_precedence_and_preserves_cause():
    runner = _load()

    result = runner.run_attempt(
        plan=runner.build_attempt_plan(**_plan_kwargs(runner)),
        plan_only=False,
        kerberos_status=_ready_kerberos(),
        query_inventory=lambda: _clean_inventory(),
        run_correctness=lambda **kwargs: {
            "classification": "INVALID_CORRECTNESS",
            "owned_pids": [9013],
            "resource_samples": [_clean_inventory()],
            "process_group_destroyed": True,
            "owned_children_remaining": [],
        },
        run_case=lambda **kwargs: pytest.fail(
            "workload must not run after correctness failure"
        ),
        cleanup_owned=lambda **kwargs: {
            "classification": "FAILED",
            "signaled_pids": [9013],
            "owned_children_remaining": [9013],
        },
    )

    assert result["classification"] == "FAILED_CLEANUP"
    assert result["prior_classification"] == "INVALID_CORRECTNESS"
    assert result["cleanup"]["owned_children_remaining"] == [9013]


def test_failure_cleanup_exception_is_preserved_as_failed_cleanup():
    runner = _load()

    result = runner.run_attempt(
        plan=runner.build_attempt_plan(**_plan_kwargs(runner)),
        plan_only=False,
        kerberos_status=_ready_kerberos(),
        query_inventory=lambda: _clean_inventory(),
        run_correctness=lambda **kwargs: {
            "classification": "INVALID_CORRECTNESS",
            "owned_pids": [9014],
            "resource_samples": [_clean_inventory()],
            "process_group_destroyed": True,
            "owned_children_remaining": [],
        },
        run_case=lambda **kwargs: pytest.fail(
            "workload must not run after correctness failure"
        ),
        cleanup_owned=lambda **kwargs: (_ for _ in ()).throw(
            RuntimeError("normal control channel failed")
        ),
    )

    assert result["classification"] == "FAILED_CLEANUP"
    assert result["prior_classification"] == "INVALID_CORRECTNESS"
    assert "normal control channel failed" in result["cleanup"]["reason"]


def test_run_attempt_does_not_misclassify_malformed_worker_result_as_gpu_drift():
    runner = _load()

    result = runner.run_attempt(
        plan=runner.build_attempt_plan(**_plan_kwargs(runner)),
        plan_only=False,
        kerberos_status=_ready_kerberos(),
        query_inventory=lambda: _clean_inventory(),
        run_correctness=lambda **kwargs: {
            "classification": "PASS",
            "process_group_destroyed": True,
            "owned_children_remaining": [],
        },
        run_case=lambda **kwargs: pytest.fail(
            "workload must not run after malformed correctness receipt"
        ),
        cleanup_owned=lambda **kwargs: {
            "classification": "CLEAN",
            "signaled_pids": [],
        },
    )

    assert result["classification"] == "FAILED_EXECUTION"
    assert "owned PID inventory" in result["reason"]


def test_run_attempt_cleans_owned_pid_when_later_receipt_validation_fails():
    runner = _load()
    cleanup_calls = []

    result = runner.run_attempt(
        plan=runner.build_attempt_plan(**_plan_kwargs(runner)),
        plan_only=False,
        kerberos_status=_ready_kerberos(),
        query_inventory=lambda: _clean_inventory(),
        run_correctness=lambda **kwargs: {
            "classification": 7,
            "owned_pids": [9090],
            "resource_samples": [_clean_inventory()],
            "process_group_destroyed": True,
            "owned_children_remaining": [],
        },
        run_case=lambda **kwargs: pytest.fail(
            "workload must not run after malformed correctness receipt"
        ),
        cleanup_owned=lambda **kwargs: (
            cleanup_calls.append(kwargs)
            or {
                "classification": "CLEAN",
                "signaled_pids": [9090],
            }
        ),
    )

    assert result["classification"] == "FAILED_EXECUTION"
    assert cleanup_calls == [{
        "owned_pids": frozenset({9090}),
        "normal_control_only": True,
    }]


def test_run_attempt_rejects_tampered_manifest_before_query():
    runner = _load()
    plan = runner.build_attempt_plan(**_plan_kwargs(runner))
    plan["manifests"]["environment"]["dtype"] = "float16"
    events = []

    with pytest.raises(ValueError, match="attempt plan"):
        runner.run_attempt(
            plan=plan,
            plan_only=False,
            kerberos_status=_ready_kerberos(),
            query_inventory=lambda: events.append("query"),
            run_correctness=lambda **kwargs: None,
            run_case=lambda **kwargs: None,
            cleanup_owned=lambda **kwargs: None,
        )

    assert events == []


@pytest.mark.parametrize(
    "manifests",
    [
        None,
        {},
        {"topology": None},
    ],
)
def test_run_attempt_rejects_malformed_manifest_shape(
    manifests,
):
    runner = _load()
    plan = runner.build_attempt_plan(**_plan_kwargs(runner))
    plan["manifests"] = manifests

    with pytest.raises(ValueError, match="attempt plan"):
        runner.run_attempt(
            plan=plan,
            plan_only=True,
            query_inventory=lambda: pytest.fail(
                "malformed plan must fail before GPU query"
            ),
            run_correctness=lambda **kwargs: None,
            run_case=lambda **kwargs: None,
            cleanup_owned=lambda **kwargs: None,
        )


def test_write_attempt_json_rejects_escape_and_uses_atomic_writer(
    tmp_path,
    monkeypatch,
):
    runner = _load()
    plan = runner.build_attempt_plan(**_plan_kwargs(runner))
    local_attempt = tmp_path / "attempt"
    local_attempt.mkdir()
    writes = []
    monkeypatch.setattr(
        runner,
        "write_json_atomic",
        lambda path, payload: writes.append((path, payload)),
    )

    runner.write_attempt_json_atomic(
        plan=plan,
        local_attempt_root=local_attempt,
        relative_path="receipts/result.json",
        payload={"classification": "PASS"},
    )

    assert writes == [(
        local_attempt / "receipts" / "result.json",
        {"classification": "PASS"},
    )]
    with pytest.raises(ValueError, match="relative artifact path"):
        runner.write_attempt_json_atomic(
            plan=plan,
            local_attempt_root=local_attempt,
            relative_path="../escape.json",
            payload={},
        )


def test_parse_nvidia_inventory_uses_memory_used_and_compute_processes():
    runner = _load()
    gpu_csv = "\n".join([
        "2, GPU-a, 1024, 5",
        "4, GPU-b, 0, 0",
        "6, GPU-c, 12, 1",
        "7, GPU-d, 4, 0",
    ])
    process_csv = "GPU-c, 321, python, 12\n"

    rows = runner.parse_nvidia_smi_inventory(gpu_csv, process_csv)

    assert [row["gpu_index"] for row in rows] == [2, 4, 6, 7]
    assert rows[0]["memory_used_mib"] == 1024
    assert rows[2]["compute_processes"] == [{
        "pid": 321,
        "process_name": "python",
        "used_memory_mib": 12,
    }]
    with pytest.raises(ValueError, match="unknown GPU UUID"):
        runner.parse_nvidia_smi_inventory(
            gpu_csv,
            "GPU-unknown, 123, python, 1\n",
        )


def test_build_ssh_argv_quotes_remote_arguments_without_kerberos_mutation():
    runner = _load()
    remote_argv = [
        "python3",
        "-c",
        "print('value with spaces; $(id)')",
    ]

    argv = runner.build_ssh_argv(
        ssh_target="sitian@10.232.195.203",
        remote_argv=remote_argv,
        control_path="/Users/bytedance/.ssh/cm-sitian",
    )

    assert argv[:3] == [
        "ssh",
        "-S",
        "/Users/bytedance/.ssh/cm-sitian",
    ]
    assert argv[-3:-1] == ["sh", "-c"]
    assert shlex.split(argv[-1]) == remote_argv
    assert "kinit" not in argv
    assert "krenew" not in argv


def test_build_ssh_argv_rejects_option_like_target():
    runner = _load()

    with pytest.raises(ValueError, match="SSH target"):
        runner.build_ssh_argv(
            ssh_target="-F@attacker",
            remote_argv=["hostname"],
        )


def test_run_remote_argv_retries_only_transport_failures():
    runner = _load()
    outcomes = iter((
        SimpleNamespace(returncode=255, stdout="", stderr="reset"),
        SimpleNamespace(returncode=0, stdout="ok", stderr=""),
    ))
    calls = []
    sleeps = []

    result = runner.run_remote_argv(
        ssh_target="sitian@10.232.195.203",
        remote_argv=["hostname"],
        control_path="/Users/bytedance/.ssh/cm-sitian",
        timeout_s=30,
        retry_count=2,
        command_runner=lambda argv, **kwargs: (
            calls.append((argv, kwargs)) or next(outcomes)
        ),
        sleep=lambda seconds: sleeps.append(seconds),
    )

    assert result.stdout == "ok"
    assert len(calls) == 2
    assert all(call[1]["timeout"] == 30 for call in calls)
    assert sleeps == [1.0]

    calls.clear()
    result = runner.run_remote_argv(
        ssh_target="sitian@10.232.195.203",
        remote_argv=["false"],
        timeout_s=30,
        retry_count=3,
        command_runner=lambda argv, **kwargs: (
            calls.append(argv)
            or SimpleNamespace(
                returncode=2,
                stdout="",
                stderr="worker failed",
            )
        ),
    )
    assert result.returncode == 2
    assert len(calls) == 1


def test_query_remote_gpu_inventory_parses_single_read_only_probe():
    runner = _load()
    calls = []
    payload = {
        "gpu_csv": "\n".join([
            "2, GPU-a, 0, 0",
            "4, GPU-b, 0, 0",
            "6, GPU-c, 0, 0",
            "7, GPU-d, 0, 0",
        ]),
        "process_csv": "",
    }

    rows = runner.query_remote_gpu_inventory(
        ssh_target="sitian@10.232.195.203",
        control_path="/Users/bytedance/.ssh/cm-sitian",
        timeout_s=30,
        retry_count=2,
        command_runner=lambda argv, **kwargs: (
            calls.append((argv, kwargs))
            or SimpleNamespace(
                returncode=0,
                stdout=json.dumps(payload),
                stderr="",
            )
        ),
    )

    assert len(rows) == 4
    assert len(calls) == 1
    serialized = json.dumps(calls[0][0])
    assert "nvidia-smi" in serialized
    assert "memory.used" in serialized
    assert "kinit" not in serialized
    assert "/tmp" not in serialized


def test_query_remote_gpu_topology_captures_pci_and_interconnect():
    runner = _load()
    calls = []
    payload = {
        "gpu_csv": "\n".join([
            "2, GPU-2, 00000000:02:00.0",
            "4, GPU-4, 00000000:04:00.0",
            "6, GPU-6, 00000000:06:00.0",
            "7, GPU-7, 00000000:07:00.0",
        ]),
        "topology_matrix": "GPU2 GPU4 GPU6 GPU7\\nGPU2 X NV18 NV18 NV18\\n",
    }

    topology = runner.query_remote_gpu_topology(
        ssh_target="sitian@10.232.195.203",
        control_path="/Users/bytedance/.ssh/cm-sitian",
        timeout_s=30,
        retry_count=2,
        command_runner=lambda argv, **kwargs: (
            calls.append((argv, kwargs))
            or SimpleNamespace(
                returncode=0,
                stdout=json.dumps(payload),
                stderr="",
            )
        ),
    )

    assert topology["gpu_rows"][0] == {
        "gpu_index": 2,
        "gpu_uuid": "GPU-2",
        "pci_bus_id": "00000000:02:00.0",
    }
    assert "NV18" in topology["interconnect_matrix"]
    serialized = json.dumps(calls[0][0])
    assert "pci.bus_id" in serialized
    assert "topo" in serialized
    assert all(
        forbidden not in serialized
        for forbidden in ("mkdir", "touch", "kinit", "krenew")
    )


def test_query_remote_path_state_is_read_only_and_complete():
    runner = _load()
    calls = []
    attempt_root = (
        f"{runner.APPROVED_REMOTE_ROOT}/attempts/"
        "qwen38-tp4-comm-20260826-r001"
    )
    model_root = (
        f"{runner.APPROVED_REMOTE_ROOT}/models/"
        "Qwen3.8-27B/snapshots/0123456789abcdef"
    )
    payload = {
        "resolved_paths": {
            "remote_root": runner.APPROVED_REMOTE_ROOT,
            "model_root": model_root,
            "attempt_root": attempt_root,
        },
        "attempt_exists": False,
    }

    result = runner.query_remote_path_state(
        ssh_target="sitian@10.232.195.203",
        remote_root=runner.APPROVED_REMOTE_ROOT,
        model_root=model_root,
        attempt_tag="qwen38-tp4-comm-20260826-r001",
        timeout_s=30,
        retry_count=2,
        command_runner=lambda argv, **kwargs: (
            calls.append((argv, kwargs))
            or SimpleNamespace(
                returncode=0,
                stdout=json.dumps(payload),
                stderr="",
            )
        ),
    )

    assert result == payload
    assert len(calls) == 1
    serialized = json.dumps(calls[0][0])
    assert "realpath" in serialized
    assert "lexists" in serialized
    assert all(
        forbidden not in serialized
        for forbidden in ("mkdir", "touch", "kinit", "krenew")
    )


def test_kerberos_ttl_guard_is_fail_fast_and_never_initializes():
    runner = _load()
    now = datetime(2026, 8, 26, 4, 0, 0, tzinfo=timezone.utc)
    payload = {
        "principal": "sitian@BYTEDANCE.COM",
        "tickets": [{
            "Principal": "krbtgt/BYTEDANCE.COM@BYTEDANCE.COM",
            "Expires": "20260826100000",
        }],
    }

    ready = runner.classify_kerberos_ttl(
        payload,
        now=now,
        minimum_lifetime_seconds=5400,
    )
    assert ready["classification"] == "READY"
    assert ready["remaining_lifetime_seconds"] == 21600

    payload["tickets"][0]["Expires"] = "20260826043000"
    blocked = runner.classify_kerberos_ttl(
        payload,
        now=now,
        minimum_lifetime_seconds=5400,
    )
    assert blocked["classification"] == "BLOCKED_KERBEROS_TTL"

    calls = []
    receipt = runner.query_local_kerberos(
        now=now,
        minimum_lifetime_seconds=5400,
        command_runner=lambda argv, **kwargs: (
            calls.append((argv, kwargs))
            or SimpleNamespace(
                returncode=0,
                stdout=json.dumps({
                    **payload,
                    "tickets": [{
                        "Principal": (
                            "krbtgt/BYTEDANCE.COM@BYTEDANCE.COM"
                        ),
                        "Expires": "20260826100000",
                    }],
                }),
                stderr="",
            )
        ),
    )
    assert receipt["classification"] == "READY"
    assert calls[0][0] == ["klist", "--json"]
    assert all(
        command not in calls[0][0] for command in ("kinit", "krenew")
    )
