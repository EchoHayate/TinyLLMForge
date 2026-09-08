from __future__ import annotations

import inspect
from pathlib import PurePosixPath
from types import SimpleNamespace

import pytest

from tools.run_qwen38_topology_local_tp2_island import (
    APPROVED_REMOTE_ROOT,
    EXPECTED_KERBEROS_PRINCIPAL,
    EXPECTED_KERBEROS_TGT,
    LOCAL_RECEIPT_NAME,
    MINIMUM_KERBEROS_LIFETIME_SECONDS,
    REMOTE_RECEIPT_NAME,
    build_attempt_plan,
    build_remote_worker_commands,
    compact_download_members,
    run_attempt,
    run_ssh_with_retry,
    select_owned_process_groups,
)
import tools.run_qwen38_topology_local_tp2_island as controller_module


def _gpu(index, *, memory=0, utilization=0, processes=()):
    return {
        "gpu_index": index,
        "gpu_uuid": f"GPU-{index}",
        "memory_used_mib": memory,
        "utilization_percent": utilization,
        "compute_processes": list(processes),
    }


def _topology_rows():
    links = {
        (0, 1): "PIX",
        (2, 3): "PXB",
    }
    return [
        {
            "left_rank": left,
            "right_rank": right,
            "link": links.get(tuple(sorted((left, right))), "SYS"),
        }
        for left in range(4)
        for right in range(4)
        if left != right
    ]


def _path_state():
    return {
        "attempt_exists": False,
        "attempt_parent_is_symlink": False,
        "remote_root_is_symlink": False,
    }


def _plan(**overrides):
    arguments = {
        "attempt_tag": (
            "20260908-qwen38-topology-local-tp2-island-stage0-r1"
        ),
        "source_revision": "a" * 40,
        "source_tree_sha256": "b" * 64,
        "selected_gpus": [_gpu(index) for index in range(4)],
        "topology_rows": _topology_rows(),
        "remote_path_state": _path_state(),
        "dist_port": 29683,
    }
    arguments.update(overrides)
    return build_attempt_plan(**arguments)


def _absolute_paths(value):
    paths = []
    if isinstance(value, dict):
        for child in value.values():
            paths.extend(_absolute_paths(child))
    elif isinstance(value, (list, tuple)):
        for child in value:
            paths.extend(_absolute_paths(child))
    elif isinstance(value, str) and value.startswith("/"):
        paths.append(value)
    return paths


def _kerberos():
    return {
        "classification": "READY",
        "principal": EXPECTED_KERBEROS_PRINCIPAL,
        "tgt_principal": EXPECTED_KERBEROS_TGT,
        "remaining_lifetime_seconds": (
            MINIMUM_KERBEROS_LIFETIME_SECONDS + 1
        ),
    }


def test_plan_freezes_safe_paths_four_gpus_and_best_pair_map():
    plan = _plan()
    approved = PurePosixPath(APPROVED_REMOTE_ROOT)
    attempt = PurePosixPath(plan["attempt_root"])

    assert all(
        PurePosixPath(path).is_relative_to(approved)
        for path in _absolute_paths(plan)
    )
    assert plan["pair_groups"] == [[0, 1], [2, 3]]
    assert len(plan["selected_gpus"]) == 4
    assert plan["dist_port"] == 29683
    assert all(
        PurePosixPath(path).is_relative_to(attempt)
        for path in plan["environment"].values()
    )


def test_plan_rejects_existing_attempt_and_non_four_gpu_selection():
    state = _path_state()
    state["attempt_exists"] = True
    with pytest.raises(ValueError, match="fresh"):
        _plan(remote_path_state=state)
    with pytest.raises(ValueError, match="four"):
        _plan(selected_gpus=[_gpu(index) for index in range(3)])


def test_controller_source_contains_no_auth_renewal():
    source = inspect.getsource(controller_module)

    assert "ki" + "nit" not in source
    assert "kre" + "new" not in source


def test_worker_commands_freeze_world_rank_gpu_map_and_pair_groups():
    commands = build_remote_worker_commands(_plan())

    assert len(commands) == 4
    for rank, command in enumerate(commands):
        assert command["environment"]["WORLD_SIZE"] == "4"
        assert command["environment"]["RANK"] == str(rank)
        assert command["environment"]["LOCAL_RANK"] == str(rank)
        assert command["environment"]["CUDA_VISIBLE_DEVICES"] == "0,1,2,3"
        assert command["environment"]["MASTER_PORT"] == "29683"
        assert "--pair-groups" in command["argv"]


def test_ssh_255_retries_only_within_fixed_budget():
    returncodes = iter((255, 255, 0))
    calls = []

    result = run_ssh_with_retry(
        ["ssh", "host", "true"],
        retry_count=2,
        runner=lambda argv, **kwargs: (
            calls.append(list(argv))
            or SimpleNamespace(
                returncode=next(returncodes),
                stdout="",
                stderr="",
            )
        ),
    )

    assert result.returncode == 0
    assert len(calls) == 3
    calls.clear()
    result = run_ssh_with_retry(
        ["ssh", "host", "false"],
        retry_count=5,
        runner=lambda argv, **kwargs: (
            calls.append(list(argv))
            or SimpleNamespace(returncode=7, stdout="", stderr="")
        ),
    )
    assert result.returncode == 7
    assert len(calls) == 1


def test_cleanup_selects_only_registered_exact_tag_process_groups():
    rows = [
        {"pid": 10, "pgid": 10, "attempt": _plan()["attempt_tag"]},
        {"pid": 11, "pgid": 10, "attempt": _plan()["attempt_tag"]},
        {"pid": 20, "pgid": 20, "attempt": "foreign"},
        {"pid": 30, "pgid": 30, "attempt": _plan()["attempt_tag"]},
    ]

    assert select_owned_process_groups(
        rows,
        attempt_tag=_plan()["attempt_tag"],
        registered_pgids={10},
    ) == (10,)


def test_compact_download_excludes_raw_and_runtime():
    members = compact_download_members(_plan())
    encoded = " ".join(members)

    assert "final_bundle" in encoded
    assert "/raw" not in encoded
    assert "/runtime" not in encoded


def test_dry_run_stops_before_remote_write():
    events = []

    result = run_attempt(
        _plan(),
        dry_run=True,
        kerberos_probe=lambda: events.append("kerberos") or _kerberos(),
        gpu_probe=lambda: events.append("gpu") or [
            _gpu(index) for index in range(4)
        ],
        remote_writer=lambda _plan: events.append("write"),
    )

    assert result["classification"] == "DRY_RUN_READY"
    assert result["worker_started"] is False
    assert events == ["kerberos", "gpu"]


def test_changed_process_inventory_blocks_before_worker_launch():
    events = []
    inventories = iter((
        [_gpu(index) for index in range(4)],
        [
            _gpu(0, processes=[{
                "pid": 99,
                "process_name": "foreign",
                "used_memory_mib": 1,
            }]),
            *[_gpu(index) for index in range(1, 4)],
        ],
    ))

    result = run_attempt(
        _plan(),
        dry_run=False,
        kerberos_probe=lambda: _kerberos(),
        gpu_probe=lambda: next(inventories),
        remote_writer=lambda _plan: events.append("write") or {
            "created": True
        },
        worker_runner=lambda _plan: events.append("worker"),
        terminal_writer=lambda receipt: events.append(
            receipt["classification"]
        ),
    )

    assert result["classification"] == "FAILED_CONTROLLER"
    assert events == ["write", "FAILED_CONTROLLER"]


def test_changed_memory_inventory_blocks_before_worker_launch():
    inventories = iter((
        [_gpu(index) for index in range(4)],
        [_gpu(0, memory=1025), *[_gpu(index) for index in range(1, 4)]],
    ))

    result = run_attempt(
        _plan(),
        dry_run=False,
        kerberos_probe=lambda: _kerberos(),
        gpu_probe=lambda: next(inventories),
        remote_writer=lambda _plan: {"created": True},
        worker_runner=lambda _plan: {"classification": "PASS"},
        terminal_writer=lambda _receipt: None,
    )

    assert result["classification"] == "FAILED_CONTROLLER"
    assert result["worker_started"] is False


def test_run_attempt_requires_dual_verifier_agreement():
    events = []
    clean = [_gpu(index) for index in range(4)]

    result = run_attempt(
        _plan(),
        dry_run=False,
        kerberos_probe=lambda: _kerberos(),
        gpu_probe=lambda: clean,
        remote_writer=lambda _plan: events.append("write") or {
            "created": True
        },
        worker_runner=lambda _plan: events.append("worker") or {
            "classification": "PASS"
        },
        assembler=lambda _plan: events.append("assemble") or {
            "classification": "NO_GO_PERFORMANCE"
        },
        remote_verifier=lambda _plan: events.append(
            "remote_verify"
        ) or {"classification": "NO_GO_PERFORMANCE"},
        downloader=lambda _plan: events.append("download") or {
            "downloaded": True
        },
        local_sealer=lambda _plan: events.append("local_seal") or {
            "classification": "NO_GO_PERFORMANCE"
        },
        local_checker=lambda _plan: events.append("local_check") or {
            "classification": "NO_GO_PERFORMANCE"
        },
        terminal_writer=lambda receipt: events.append(
            receipt["classification"]
        ),
    )

    assert result["classification"] == "NO_GO_PERFORMANCE"
    assert events == [
        "write",
        "worker",
        "assemble",
        "remote_verify",
        "download",
        "local_seal",
        "local_check",
        "NO_GO_PERFORMANCE",
    ]


def test_failure_path_writes_terminal_receipt():
    terminal = []

    result = run_attempt(
        _plan(),
        dry_run=False,
        kerberos_probe=lambda: _kerberos(),
        gpu_probe=lambda: [_gpu(index) for index in range(4)],
        remote_writer=lambda _plan: {"created": True},
        worker_runner=lambda _plan: (_ for _ in ()).throw(
            RuntimeError("worker failed")
        ),
        terminal_writer=terminal.append,
    )

    assert result["classification"] == "FAILED_CONTROLLER"
    assert terminal == [result]


def test_receipt_names_are_frozen():
    assert REMOTE_RECEIPT_NAME == "remote_independent_verification.json"
    assert LOCAL_RECEIPT_NAME == "local_independent_verification.json"
