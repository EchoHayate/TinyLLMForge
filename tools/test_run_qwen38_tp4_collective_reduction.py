from __future__ import annotations

import json
from pathlib import PurePosixPath

import pytest

from tools.run_qwen38_tp4_collective_reduction import (
    APPROVED_REMOTE_ROOT,
    build_attempt_plan,
    expected_case_ids,
    main,
    run_attempt,
    select_strict_clean_gpus,
)


def _gpu(index, *, memory=0, utilization=0, processes=()):
    return {
        "gpu_index": index,
        "gpu_uuid": f"GPU-{index}",
        "memory_used_mib": memory,
        "utilization_percent": utilization,
        "compute_processes": list(processes),
    }


def _build_plan(**overrides):
    arguments = {
        "attempt_tag": "20260827-qwen38-tp4-collective-reduction-r1",
        "source_revision": "a" * 40,
        "model_revision": "b" * 40,
        "selected_gpus": [_gpu(index) for index in range(4)],
        "remote_path_state": {
            "attempt_exists": False,
            "attempt_parent_is_symlink": False,
            "remote_root_is_symlink": False,
        },
    }
    arguments.update(overrides)
    return build_attempt_plan(**arguments)


def _worker_result(*, selected_budget=16, case_ids=None):
    if case_ids is None:
        case_ids = expected_case_ids(selected_budget)
    return {
        "classification": "PASS",
        "attempt": "20260827-qwen38-tp4-collective-reduction-r1",
        "source_revision": "a" * 40,
        "selected_budget": selected_budget,
        "owned_pids": [101],
        "cases": [
            {
                "case_id": case_id,
                "classification": "PASS",
            }
            for case_id in case_ids
        ],
        "phase_cleanups": [
            {
                "process_group_destroyed": True,
                "rank_exit_codes": [0, 0, 0, 0],
                "owned_children_remaining": [],
            },
            {
                "process_group_destroyed": True,
                "rank_exit_codes": [0, 0, 0, 0],
                "owned_children_remaining": [],
            },
        ],
    }


def _all_remote_paths(value):
    paths = []
    if isinstance(value, dict):
        for key, child in value.items():
            if (
                isinstance(child, str)
                and ("path" in key or key.endswith("root"))
                and child.startswith("/")
            ):
                paths.append(child)
            paths.extend(_all_remote_paths(child))
    elif isinstance(value, (list, tuple)):
        for child in value:
            paths.extend(_all_remote_paths(child))
    return paths


def test_plan_keeps_every_path_below_approved_remote_root():
    plan = _build_plan()
    paths = _all_remote_paths(plan)

    assert paths
    assert all(
        PurePosixPath(path).is_relative_to(
            PurePosixPath(APPROVED_REMOTE_ROOT)
        )
        for path in paths
    )


def test_plan_uses_the_frozen_qwen38_snapshot_layout():
    plan = _build_plan(model_revision="b" * 40)

    assert plan["model_root"] == (
        f"{APPROVED_REMOTE_ROOT}/models/Qwen3.8-27B/"
        f"snapshots/{'b' * 40}"
    )


def test_plan_contains_no_nsys_or_overlap_mechanism():
    encoded = json.dumps(_build_plan(), sort_keys=True).lower()

    assert "nsys" not in encoded
    assert "async_op" not in encoded
    assert "communication_stream" not in encoded
    assert _build_plan()["overlap_design_authorized"] is False
    assert _build_plan()["async_collectives_authorized"] is False


def test_plan_runs_the_self_selecting_full_worker():
    commands = _build_plan()["remote_commands"]
    worker = next(
        row["argv"]
        for row in commands
        if row["purpose"] == "run qualification worker"
    )

    assert worker[worker.index("--phase") + 1] == "full"
    assert "--selected-budget" not in worker


def test_controller_plan_has_no_auth_or_signal_commands():
    commands = _build_plan()["remote_commands"]
    basenames = {
        PurePosixPath(command["argv"][0]).name
        for command in commands
    }
    assert basenames.isdisjoint({
        "kinit", "krenew", "kill", "pkill", "killall",
    })


@pytest.mark.parametrize(
    "remote_path_state",
    (
        {
            "attempt_exists": True,
            "attempt_parent_is_symlink": False,
            "remote_root_is_symlink": False,
        },
        {
            "attempt_exists": False,
            "attempt_parent_is_symlink": True,
            "remote_root_is_symlink": False,
        },
        {
            "attempt_exists": False,
            "attempt_parent_is_symlink": False,
            "remote_root_is_symlink": True,
        },
    ),
)
def test_plan_rejects_reuse_or_symlink_escape(remote_path_state):
    with pytest.raises(ValueError):
        _build_plan(remote_path_state=remote_path_state)


@pytest.mark.parametrize(
    "inventory",
    (
        [_gpu(0, memory=1025), _gpu(1), _gpu(2), _gpu(3)],
        [_gpu(0, utilization=6), _gpu(1), _gpu(2), _gpu(3)],
        [
            _gpu(0, processes=[{"pid": 9, "process_name": "other"}]),
            _gpu(1),
            _gpu(2),
            _gpu(3),
        ],
    ),
)
def test_plan_requires_four_strict_clean_gpus(inventory):
    with pytest.raises(ValueError, match="four strict-clean GPUs"):
        select_strict_clean_gpus(inventory)


def test_run_attempt_orders_gate_assembly_verification_and_cleanup():
    events = []
    plan = _build_plan()

    result = run_attempt(
        plan,
        plan_only=False,
        dry_run=False,
        kerberos_probe=lambda: events.append("kerberos") or {
            "classification": "PASS",
        },
        gpu_probe=lambda: events.append("gpu") or [
            _gpu(index) for index in range(4)
        ],
        worker_runner=lambda _plan: (
            events.append("worker") or _worker_result()
        ),
        assembler=lambda _plan, _worker: events.append("assemble") or {
            "classification": "GO_SYNC_COLLECTIVE_REDUCTION",
        },
        remote_verifier=lambda _plan: events.append("remote_verify") or {
            "classification": "GO_SYNC_COLLECTIVE_REDUCTION",
        },
        downloader=lambda _plan: events.append("download") or {
            "downloaded": True,
        },
        local_verifier=lambda _plan: events.append("local_verify") or {
            "classification": "GO_SYNC_COLLECTIVE_REDUCTION",
        },
        cleanup_validator=lambda _plan, _worker: (
            events.append("cleanup") or {"complete": True}
        ),
    )

    assert result["classification"] == (
        "GO_SYNC_COLLECTIVE_REDUCTION"
    )
    assert events == [
        "kerberos",
        "gpu",
        "worker",
        "assemble",
        "remote_verify",
        "download",
        "local_verify",
        "cleanup",
    ]


def test_plan_only_and_dry_run_launch_no_worker():
    calls = []
    plan = _build_plan()
    for plan_only, dry_run in ((True, False), (False, True)):
        result = run_attempt(
            plan,
            plan_only=plan_only,
            dry_run=dry_run,
            kerberos_probe=lambda: {"classification": "PASS"},
            gpu_probe=lambda: [_gpu(index) for index in range(4)],
            worker_runner=lambda _plan: calls.append("worker"),
        )
        assert result["worker_started"] is False
    assert calls == []


def test_low_kerberos_ttl_blocks_before_gpu_or_worker():
    events = []

    result = run_attempt(
        _build_plan(),
        plan_only=False,
        kerberos_probe=lambda: {
            "classification": "BLOCKED_KERBEROS_TTL",
        },
        gpu_probe=lambda: events.append("gpu"),
        worker_runner=lambda _plan: events.append("worker"),
    )

    assert result["classification"] == "BLOCKED_KERBEROS"
    assert result["worker_started"] is False
    assert events == []


@pytest.mark.parametrize(
    "inventory",
    (
        [
            {**_gpu(0), "gpu_uuid": "GPU-drift"},
            _gpu(1),
            _gpu(2),
            _gpu(3),
        ],
        [_gpu(0, memory=1025), _gpu(1), _gpu(2), _gpu(3)],
        [_gpu(0, utilization=6), _gpu(1), _gpu(2), _gpu(3)],
        [
            _gpu(0, processes=[{
                "pid": 9,
                "process_name": "other",
                "used_memory_mib": 1,
            }]),
            _gpu(1),
            _gpu(2),
            _gpu(3),
        ],
    ),
)
def test_dirty_or_drifted_entry_inventory_blocks_before_worker(inventory):
    calls = []

    with pytest.raises(ValueError):
        run_attempt(
            _build_plan(),
            plan_only=False,
            kerberos_probe=lambda: {"classification": "PASS"},
            gpu_probe=lambda: inventory,
            worker_runner=lambda _plan: calls.append("worker"),
        )

    assert calls == []


def test_partial_worker_receipts_are_rejected_before_assembly():
    events = []
    incomplete = _worker_result(
        case_ids=expected_case_ids(16)[:-1],
    )

    with pytest.raises(RuntimeError, match="case coverage"):
        run_attempt(
            _build_plan(),
            plan_only=False,
            kerberos_probe=lambda: {"classification": "PASS"},
            gpu_probe=lambda: [_gpu(index) for index in range(4)],
            worker_runner=lambda _plan: incomplete,
            assembler=lambda *_args: events.append("assemble"),
            remote_verifier=lambda *_args: None,
            downloader=lambda *_args: None,
            local_verifier=lambda *_args: None,
            cleanup_validator=lambda *_args: (
                events.append("cleanup") or {"complete": True}
            ),
        )

    assert events == ["cleanup"]


def test_missing_download_and_verifier_disagreement_still_cleanup():
    for downloader, expected_error in (
        (
            lambda _plan: {"downloaded": False},
            "download is incomplete",
        ),
        (
            lambda _plan: {"downloaded": True},
            "classification disagreement",
        ),
    ):
        events = []
        with pytest.raises(RuntimeError, match=expected_error):
            run_attempt(
                _build_plan(),
                plan_only=False,
                kerberos_probe=lambda: {"classification": "PASS"},
                gpu_probe=lambda: [_gpu(index) for index in range(4)],
                worker_runner=lambda _plan: _worker_result(),
                assembler=lambda *_args: {
                    "classification": "GO_SYNC_COLLECTIVE_REDUCTION",
                },
                remote_verifier=lambda _plan: {
                    "classification": "GO_SYNC_COLLECTIVE_REDUCTION",
                },
                downloader=downloader,
                local_verifier=lambda _plan: {
                    "classification": (
                        "NO_GO_NO_REDUCIBLE_COLLECTIVE"
                        if downloader(_plan)["downloaded"]
                        else "GO_SYNC_COLLECTIVE_REDUCTION"
                    ),
                },
                cleanup_validator=lambda *_args: (
                    events.append("cleanup") or {"complete": True}
                ),
            )
        assert events == ["cleanup"]


def test_incomplete_cleanup_overrides_success():
    with pytest.raises(RuntimeError, match="cleanup is incomplete"):
        run_attempt(
            _build_plan(),
            plan_only=False,
            kerberos_probe=lambda: {"classification": "PASS"},
            gpu_probe=lambda: [_gpu(index) for index in range(4)],
            worker_runner=lambda _plan: _worker_result(),
            assembler=lambda *_args: {
                "classification": "GO_SYNC_COLLECTIVE_REDUCTION",
            },
            remote_verifier=lambda _plan: {
                "classification": "GO_SYNC_COLLECTIVE_REDUCTION",
            },
            downloader=lambda _plan: {"downloaded": True},
            local_verifier=lambda _plan: {
                "classification": "GO_SYNC_COLLECTIVE_REDUCTION",
            },
            cleanup_validator=lambda *_args: {
                "complete": False,
                "owned_children_remaining": [101],
            },
        )


def test_cli_dry_run_performs_bounded_preflight_without_worker(capsys):
    events = []
    attempt = "20260827-qwen38-tp4-collective-reduction-r1"
    model_revision = "b" * 40
    model_root = (
        f"{APPROVED_REMOTE_ROOT}/models/Qwen3.8-27B/"
        f"snapshots/{model_revision}"
    )
    attempt_root = f"{APPROVED_REMOTE_ROOT}/attempts/{attempt}"

    return_code = main(
        [
            "--attempt-tag",
            attempt,
            "--source-revision",
            "a" * 40,
            "--model-revision",
            model_revision,
            "--dry-run",
        ],
        kerberos_query=lambda: (
            events.append("kerberos")
            or {"classification": "READY"}
        ),
        path_state_query=lambda **_kwargs: (
            events.append("paths")
            or {
                "resolved_paths": {
                    "remote_root": APPROVED_REMOTE_ROOT,
                    "model_root": model_root,
                    "attempt_root": attempt_root,
                },
                "attempt_exists": False,
            }
        ),
        inventory_query=lambda **_kwargs: (
            events.append("inventory")
            or [_gpu(index) for index in range(4)]
        ),
        gpu_monitor=lambda **kwargs: (
            events.append("monitor")
            or {
                "classification": "READY",
                "selected_gpus": kwargs["query_inventory"](),
            }
        ),
        worker_runner=lambda _plan: events.append("worker"),
    )

    payload = json.loads(capsys.readouterr().out)
    assert return_code == 0
    assert payload["classification"] == "DRY_RUN_READY"
    assert payload["worker_started"] is False
    assert events == ["kerberos", "paths", "monitor", "inventory", "inventory"]
