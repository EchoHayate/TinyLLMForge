from __future__ import annotations

import json
from pathlib import PurePosixPath
from types import SimpleNamespace

import pytest

from tools.run_qwen38_tp4_collective_reduction import (
    APPROVED_REMOTE_ROOT,
    build_source_identity,
    build_attempt_plan,
    expected_case_ids,
    main,
    query_remote_collective_path_state,
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


def _remote_query_state(*, attempt, model_revision):
    model_root = (
        f"{APPROVED_REMOTE_ROOT}/models/Qwen3.8-27B/"
        f"snapshots/{model_revision}"
    )
    return {
        "resolved_paths": {
            "remote_root": APPROVED_REMOTE_ROOT,
            "model_root": model_root,
            "attempt_root": (
                f"{APPROVED_REMOTE_ROOT}/attempts/{attempt}"
            ),
        },
        "attempt_exists": False,
        "remote_root_ready": True,
        "model_manifest": {
            "schema_version": "tinyllmforge.qwen38-model-manifest.v1",
            "repository": "Qwen/Qwen3.8-27B",
            "revision": model_revision,
            "text_profile": {
                "num_hidden_layers": 64,
                "hidden_size": 5120,
                "vocab_size": 248320,
                "dtype": "bfloat16",
            },
        },
        "model_files": {
            "config_readable": True,
            "weight_index_readable": True,
            "weight_shard_count": 18,
            "all_weight_shards_readable": True,
            "snapshot_revision": model_revision,
            "snapshot_revision_matches": True,
        },
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


def test_source_identity_binds_sorted_file_hashes():
    identity = build_source_identity(
        attempt="attempt-r1",
        source_revision="a" * 40,
        source_files={
            "tinyvllm/z.py": "f" * 64,
            "tinyvllm/a.py": "0" * 64,
        },
    )

    assert identity["schema_version"] == (
        "qwen38.tp4-collective-reduction-source.v1"
    )
    assert list(identity["source_files"]) == [
        "tinyvllm/a.py",
        "tinyvllm/z.py",
    ]
    assert len(identity["source_tree_sha256"]) == 64


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


def test_cleanup_failure_does_not_mask_the_original_operation_error():
    with pytest.raises(ValueError, match="case identity is invalid"):
        run_attempt(
            _build_plan(),
            plan_only=False,
            kerberos_probe=lambda: {"classification": "PASS"},
            gpu_probe=lambda: [_gpu(index) for index in range(4)],
            worker_runner=lambda _plan: _worker_result(),
            assembler=lambda *_args: (
                (_ for _ in ()).throw(
                    ValueError("case identity is invalid")
                )
            ),
            remote_verifier=lambda _plan: pytest.fail(
                "verification must not run"
            ),
            downloader=lambda _plan: pytest.fail(
                "download must not run"
            ),
            local_verifier=lambda _plan: pytest.fail(
                "local verification must not run"
            ),
            cleanup_validator=lambda *_args: (
                (_ for _ in ()).throw(
                    FileNotFoundError("cleanup.json")
                )
            ),
        )


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
            or _remote_query_state(
                attempt=attempt,
                model_revision=model_revision,
            )
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


def test_cli_persists_successful_dry_run_preflight_receipts(
    tmp_path,
    capsys,
):
    local_attempt_root = tmp_path / "attempt"
    attempt = "20260827-qwen38-tp4-collective-reduction-r1"
    source_revision = "a" * 40
    model_revision = "b" * 40
    model_root = (
        f"{APPROVED_REMOTE_ROOT}/models/Qwen3.8-27B/"
        f"snapshots/{model_revision}"
    )
    attempt_root = f"{APPROVED_REMOTE_ROOT}/attempts/{attempt}"
    source_identity = build_source_identity(
        attempt=attempt,
        source_revision=source_revision,
        source_files={"tinyvllm/config.py": "c" * 64},
    )
    selected_gpus = [_gpu(index) for index in range(4)]

    return_code = main(
        [
            "--attempt-tag",
            attempt,
            "--source-revision",
            source_revision,
            "--model-revision",
            model_revision,
            "--dry-run",
            "--local-attempt-root",
            str(local_attempt_root),
        ],
        source_identity_builder=lambda **_kwargs: source_identity,
        kerberos_query=lambda: {
            "classification": "READY",
            "principal": "sitian@BYTEDANCE.COM",
            "remaining_lifetime_seconds": 7200,
        },
        path_state_query=lambda **_kwargs: {
            **_remote_query_state(
                attempt=attempt,
                model_revision=model_revision,
            ),
        },
        inventory_query=lambda **_kwargs: selected_gpus,
        gpu_monitor=lambda **_kwargs: {
            "classification": "READY",
            "selected_gpus": selected_gpus,
        },
        worker_runner=lambda _plan: pytest.fail(
            "dry-run must not start a worker"
        ),
    )

    controller_root = local_attempt_root / "controller"
    persisted = {
        path.name: json.loads(path.read_text())
        for path in controller_root.iterdir()
    }
    payload = json.loads(capsys.readouterr().out)

    assert return_code == 0
    assert set(persisted) == {
        "dry_run.json",
        "plan.json",
        "plan_audit.json",
        "source_identity.json",
        "ssh_storage_preflight.json",
        "strict_clean_admission.json",
    }
    assert persisted["source_identity.json"] == source_identity
    assert persisted["plan.json"] == payload["plan"]
    assert persisted["plan_audit.json"] == {
        "schema_version": (
            "qwen38.tp4-collective-reduction-plan-audit.v1"
        ),
        "classification": "PASS",
        "attempt_tag": attempt,
        "source_revision": source_revision,
        "remote_paths_below_approved_root": True,
        "attempt_absent": True,
        "overlap_design_authorized": False,
        "async_collectives_authorized": False,
    }
    assert persisted["ssh_storage_preflight.json"] == {
        "schema_version": (
            "qwen38.tp4-collective-reduction-preflight.v1"
        ),
        "classification": "PASS",
        "kerberos": {
            "classification": "PASS",
            "principal": "sitian@BYTEDANCE.COM",
            "remaining_lifetime_seconds": 7200,
        },
        "resolved_paths": {
            "remote_root": APPROVED_REMOTE_ROOT,
            "model_root": model_root,
            "attempt_root": attempt_root,
        },
        "attempt_exists": False,
        "remote_root_ready": True,
        "model_manifest": {
            "schema_version": "tinyllmforge.qwen38-model-manifest.v1",
            "repository": "Qwen/Qwen3.8-27B",
            "revision": model_revision,
            "text_profile": {
                "num_hidden_layers": 64,
                "hidden_size": 5120,
                "vocab_size": 248320,
                "dtype": "bfloat16",
            },
        },
        "model_files": {
            "config_readable": True,
            "weight_index_readable": True,
            "weight_shard_count": 18,
            "all_weight_shards_readable": True,
            "snapshot_revision": model_revision,
            "snapshot_revision_matches": True,
        },
        "remote_query_performed": True,
        "remote_write_performed": False,
    }
    assert persisted["strict_clean_admission.json"] == {
        "schema_version": (
            "qwen38.tp4-collective-reduction-gpu-admission.v1"
        ),
        "classification": "READY",
        "selected_gpus": selected_gpus,
        "maximum_memory_used_mib": 1024,
        "maximum_utilization_percent": 5,
        "compute_processes_required_empty": True,
        "worker_started": False,
    }
    assert persisted["dry_run.json"] == payload
    assert payload["classification"] == "DRY_RUN_READY"
    assert payload["worker_started"] is False
    assert payload["plan"]["attempt_tag"] == attempt


def test_cli_plan_only_replaces_stale_local_plan_receipts(
    tmp_path,
    capsys,
):
    local_attempt_root = tmp_path / "attempt"
    controller_root = local_attempt_root / "controller"
    controller_root.mkdir(parents=True)
    attempt = "20260827-qwen38-tp4-collective-reduction-r1"
    source_revision = "a" * 40
    model_revision = "b" * 40
    source_identity = build_source_identity(
        attempt=attempt,
        source_revision=source_revision,
        source_files={"tinyvllm/config.py": "c" * 64},
    )
    stale_plan = _build_plan(source_revision="d" * 40)
    (controller_root / "plan.json").write_text(
        json.dumps(stale_plan),
        encoding="utf-8",
    )
    (controller_root / "plan_audit.json").write_text(
        json.dumps(
            {
                "classification": "PASS",
                "source_revision": "d" * 40,
            }
        ),
        encoding="utf-8",
    )

    return_code = main(
        [
            "--attempt-tag",
            attempt,
            "--source-revision",
            source_revision,
            "--model-revision",
            model_revision,
            "--plan-only",
            "--local-attempt-root",
            str(local_attempt_root),
        ],
        source_identity_builder=lambda **_kwargs: source_identity,
        path_state_query=lambda **_kwargs: _remote_query_state(
            attempt=attempt,
            model_revision=model_revision,
        ),
        inventory_query=lambda **_kwargs: [
            _gpu(index) for index in range(4)
        ],
        gpu_monitor=lambda **_kwargs: pytest.fail(
            "plan-only must not monitor GPUs"
        ),
        worker_runner=lambda _plan: pytest.fail(
            "plan-only must not start a worker"
        ),
    )

    payload = json.loads(capsys.readouterr().out)
    persisted_plan = json.loads(
        (controller_root / "plan.json").read_text()
    )
    persisted_audit = json.loads(
        (controller_root / "plan_audit.json").read_text()
    )

    assert return_code == 0
    assert payload["classification"] == "PLAN_ONLY"
    assert persisted_plan == payload["plan"]
    assert persisted_plan["source_revision"] == source_revision
    assert persisted_audit["classification"] == "PASS"
    assert persisted_audit["source_revision"] == source_revision
    assert persisted_audit["attempt_absent"] is True


def test_cli_rejects_missing_model_identity_before_gpu_monitor():
    attempt = "20260827-qwen38-tp4-collective-reduction-r1"
    model_revision = "b" * 40
    model_root = (
        f"{APPROVED_REMOTE_ROOT}/models/Qwen3.8-27B/"
        f"snapshots/{model_revision}"
    )

    with pytest.raises(ValueError, match="model preflight"):
        main(
            [
                "--attempt-tag",
                attempt,
                "--source-revision",
                "a" * 40,
                "--model-revision",
                model_revision,
                "--dry-run",
            ],
            kerberos_query=lambda: {"classification": "READY"},
            path_state_query=lambda **_kwargs: {
                "resolved_paths": {
                    "remote_root": APPROVED_REMOTE_ROOT,
                    "model_root": model_root,
                    "attempt_root": (
                        f"{APPROVED_REMOTE_ROOT}/attempts/{attempt}"
                    ),
                },
                "attempt_exists": False,
            },
            inventory_query=lambda **_kwargs: pytest.fail(
                "GPU inventory must not run"
            ),
            gpu_monitor=lambda **_kwargs: pytest.fail(
                "GPU monitor must not run"
            ),
        )


def test_remote_collective_path_query_builds_verified_model_manifest():
    attempt = "20260827-qwen38-tp4-collective-reduction-r1"
    model_revision = "b" * 40
    expected = _remote_query_state(
        attempt=attempt,
        model_revision=model_revision,
    )
    calls = []

    result = query_remote_collective_path_state(
        ssh_target="sitian@example",
        remote_root=APPROVED_REMOTE_ROOT,
        model_root=expected["resolved_paths"]["model_root"],
        model_revision=model_revision,
        attempt_tag=attempt,
        timeout_s=30,
        retry_count=1,
        command_runner=lambda argv, **kwargs: (
            calls.append((argv, kwargs))
            or SimpleNamespace(
                returncode=0,
                stdout=json.dumps(expected),
                stderr="",
            )
        ),
    )

    assert result == expected
    assert len(calls) == 1
    assert "input" not in calls[0][1]


def test_cli_persists_source_and_blocked_kerberos_receipts(
    tmp_path,
    capsys,
):
    local_attempt_root = tmp_path / "attempt"
    source_identity = build_source_identity(
        attempt="20260827-qwen38-tp4-collective-reduction-r1",
        source_revision="a" * 40,
        source_files={"tinyvllm/config.py": "b" * 64},
    )

    return_code = main(
        [
            "--attempt-tag",
            "20260827-qwen38-tp4-collective-reduction-r1",
            "--source-revision",
            "a" * 40,
            "--model-revision",
            "b" * 40,
            "--dry-run",
            "--local-attempt-root",
            str(local_attempt_root),
        ],
        source_identity_builder=lambda **_kwargs: source_identity,
        kerberos_query=lambda: {
            "classification": "BLOCKED_KERBEROS_TTL",
            "reason": "test",
        },
        path_state_query=lambda **_kwargs: pytest.fail(
            "remote path query must not run"
        ),
        inventory_query=lambda **_kwargs: pytest.fail(
            "GPU query must not run"
        ),
    )

    assert return_code == 2
    assert json.loads(
        (local_attempt_root / "controller/source_identity.json").read_text()
    ) == source_identity
    preflight = json.loads(
        (
            local_attempt_root
            / "controller/ssh_storage_preflight.json"
        ).read_text()
    )
    assert preflight["classification"] == "BLOCKED_KERBEROS"
    assert preflight["remote_query_performed"] is False
    dry_run = json.loads(
        (local_attempt_root / "controller/dry_run.json").read_text()
    )
    assert dry_run["worker_started"] is False
    assert json.loads(capsys.readouterr().out) == dry_run


def test_cli_auto_wires_production_adapter_for_real_execution(
    tmp_path,
    capsys,
):
    attempt = "20260827-qwen38-tp4-collective-reduction-r1"
    source_revision = "a" * 40
    model_revision = "b" * 40
    source_identity = build_source_identity(
        attempt=attempt,
        source_revision=source_revision,
        source_files={"tinyvllm/config.py": "c" * 64},
    )
    selected = [_gpu(index) for index in range(4)]
    factory_calls = []
    events = []

    class FakeAdapter:
        def worker_runner(self, _plan):
            events.append("worker")
            return _worker_result()

        def assembler(self, _plan, _worker):
            events.append("assemble")
            return {"classification": "GO_SYNC_COLLECTIVE_REDUCTION"}

        def remote_verifier(self, _plan):
            events.append("remote")
            return {"classification": "GO_SYNC_COLLECTIVE_REDUCTION"}

        def downloader(self, _plan):
            events.append("download")
            return {"downloaded": True}

        def local_verifier(self, _plan):
            events.append("local")
            return {"classification": "GO_SYNC_COLLECTIVE_REDUCTION"}

        def cleanup_validator(self, _plan, _worker):
            events.append("cleanup")
            return {
                "complete": True,
                "owned_children_remaining": [],
            }

    return_code = main(
        [
            "--attempt-tag",
            attempt,
            "--source-revision",
            source_revision,
            "--model-revision",
            model_revision,
            "--local-attempt-root",
            str(tmp_path / "attempt"),
        ],
        source_identity_builder=lambda **_kwargs: source_identity,
        kerberos_query=lambda: {"classification": "READY"},
        path_state_query=lambda **_kwargs: _remote_query_state(
            attempt=attempt,
            model_revision=model_revision,
        ),
        inventory_query=lambda **_kwargs: selected,
        gpu_monitor=lambda **_kwargs: {
            "classification": "READY",
            "selected_gpus": selected,
        },
        production_adapter_factory=lambda **kwargs: (
            factory_calls.append(kwargs) or FakeAdapter()
        ),
    )

    payload = json.loads(capsys.readouterr().out)
    assert return_code == 0
    assert payload["classification"] == "GO_SYNC_COLLECTIVE_REDUCTION"
    assert events == [
        "worker",
        "assemble",
        "remote",
        "download",
        "local",
        "cleanup",
    ]
    assert len(factory_calls) == 1
    assert factory_calls[0]["source_identity"] == source_identity
    assert factory_calls[0]["model_manifest"]["revision"] == model_revision


def test_existing_attempt_is_allowed_only_for_real_execution_resume(
    tmp_path,
    capsys,
):
    attempt = "20260827-qwen38-tp4-collective-reduction-r1"
    source_revision = "a" * 40
    model_revision = "b" * 40
    source_identity = build_source_identity(
        attempt=attempt,
        source_revision=source_revision,
        source_files={"tinyvllm/config.py": "c" * 64},
    )
    selected = [_gpu(index) for index in range(4)]
    path_state = _remote_query_state(
        attempt=attempt,
        model_revision=model_revision,
    )
    path_state["attempt_exists"] = True
    local_attempt_root = tmp_path / "attempt"
    controller_root = local_attempt_root / "controller"
    controller_root.mkdir(parents=True)
    frozen_plan = build_attempt_plan(
        attempt_tag=attempt,
        source_revision=source_revision,
        model_revision=model_revision,
        selected_gpus=selected,
        remote_path_state={
            "attempt_exists": False,
            "attempt_parent_is_symlink": False,
            "remote_root_is_symlink": False,
        },
    )
    (controller_root / "plan.json").write_text(
        json.dumps(frozen_plan),
        encoding="utf-8",
    )
    (controller_root / "source_identity.json").write_text(
        json.dumps(source_identity),
        encoding="utf-8",
    )

    class ResumeAdapter:
        def worker_runner(self, _plan):
            return _worker_result()

        def assembler(self, _plan, _worker):
            return {"classification": "GO_SYNC_COLLECTIVE_REDUCTION"}

        def remote_verifier(self, _plan):
            return {"classification": "GO_SYNC_COLLECTIVE_REDUCTION"}

        def downloader(self, _plan):
            return {"downloaded": True}

        def local_verifier(self, _plan):
            return {"classification": "GO_SYNC_COLLECTIVE_REDUCTION"}

        def cleanup_validator(self, _plan, _worker):
            return {
                "complete": True,
                "owned_children_remaining": [],
            }

    return_code = main(
        [
            "--attempt-tag",
            attempt,
            "--source-revision",
            source_revision,
            "--model-revision",
            model_revision,
            "--local-attempt-root",
            str(local_attempt_root),
        ],
        source_identity_builder=lambda **_kwargs: source_identity,
        kerberos_query=lambda: {"classification": "READY"},
        path_state_query=lambda **_kwargs: path_state,
        inventory_query=lambda **_kwargs: selected,
        gpu_monitor=lambda **_kwargs: {
            "classification": "READY",
            "selected_gpus": selected,
        },
        production_adapter_factory=lambda **_kwargs: ResumeAdapter(),
    )

    payload = json.loads(capsys.readouterr().out)
    assert return_code == 0
    assert payload["classification"] == "GO_SYNC_COLLECTIVE_REDUCTION"

    with pytest.raises(ValueError, match="already in use"):
        main(
            [
                "--attempt-tag",
                attempt,
                "--source-revision",
                source_revision,
                "--model-revision",
                model_revision,
                "--dry-run",
            ],
            kerberos_query=lambda: {"classification": "READY"},
            path_state_query=lambda **_kwargs: path_state,
            inventory_query=lambda **_kwargs: selected,
            gpu_monitor=lambda **_kwargs: {
                "classification": "READY",
                "selected_gpus": selected,
            },
        )


def test_existing_attempt_never_overwrites_frozen_source_identity(tmp_path):
    attempt = "20260827-qwen38-tp4-collective-reduction-r1"
    frozen_revision = "a" * 40
    requested_revision = "d" * 40
    model_revision = "b" * 40
    selected = [_gpu(index) for index in range(4)]
    local_attempt_root = tmp_path / "attempt"
    controller_root = local_attempt_root / "controller"
    controller_root.mkdir(parents=True)
    frozen_identity = build_source_identity(
        attempt=attempt,
        source_revision=frozen_revision,
        source_files={"tinyvllm/config.py": "c" * 64},
    )
    requested_identity = build_source_identity(
        attempt=attempt,
        source_revision=requested_revision,
        source_files={"tinyvllm/config.py": "e" * 64},
    )
    (controller_root / "source_identity.json").write_text(
        json.dumps(frozen_identity),
        encoding="utf-8",
    )
    (controller_root / "plan.json").write_text(
        json.dumps(build_attempt_plan(
            attempt_tag=attempt,
            source_revision=frozen_revision,
            model_revision=model_revision,
            selected_gpus=selected,
            remote_path_state={
                "attempt_exists": False,
                "attempt_parent_is_symlink": False,
                "remote_root_is_symlink": False,
            },
        )),
        encoding="utf-8",
    )
    path_state = _remote_query_state(
        attempt=attempt,
        model_revision=model_revision,
    )
    path_state["attempt_exists"] = True

    with pytest.raises(
        ValueError,
        match="frozen local source identity mismatch",
    ):
        main(
            [
                "--attempt-tag",
                attempt,
                "--source-revision",
                requested_revision,
                "--model-revision",
                model_revision,
                "--local-attempt-root",
                str(local_attempt_root),
            ],
            source_identity_builder=lambda **_kwargs: requested_identity,
            kerberos_query=lambda: {"classification": "READY"},
            path_state_query=lambda **_kwargs: path_state,
            inventory_query=lambda **_kwargs: selected,
            gpu_monitor=lambda **_kwargs: pytest.fail(
                "must not monitor for an existing attempt"
            ),
        )

    assert json.loads(
        (controller_root / "source_identity.json").read_text()
    ) == frozen_identity
