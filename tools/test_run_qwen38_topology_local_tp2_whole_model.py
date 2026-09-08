from __future__ import annotations

import copy
import inspect
import json
from pathlib import PurePosixPath
from types import SimpleNamespace

import pytest

from tools.run_qwen38_topology_local_tp2_whole_model import (
    APPROVED_REMOTE_ROOT,
    DEFAULT_MODEL_ROOT,
    DEFAULT_REMOTE_PYTHON,
    MINIMUM_KERBEROS_LIFETIME_SECONDS,
    MODEL_REVISION,
    build_remote_correctness_command,
    build_remote_epoch_command,
    build_remote_service_command,
    build_plan,
    run_attempt,
    run_ssh_with_retry,
    select_owned_process_groups,
)
import tools.run_qwen38_topology_local_tp2_whole_model as controller


def _gpu(index, *, memory=0, utilization=0, processes=()):
    return {
        "gpu_index": index,
        "gpu_uuid": f"GPU-{index}",
        "memory_used_mib": memory,
        "utilization_percent": utilization,
        "compute_processes": list(processes),
    }


def _topology():
    links = {(0, 1): "PIX", (2, 3): "PIX"}
    return {
        "rows": [
            {
                "left_rank": left,
                "right_rank": right,
                "link": links.get(tuple(sorted((left, right))), "SYS"),
            }
            for left in range(4)
            for right in range(4)
            if left != right
        ],
    }


def _plan(**overrides):
    kwargs = {
        "attempt_tag": "20260908-qwen38-tp2-whole-model-r1",
        "source_revision": "a" * 40,
        "gpu_inventory": tuple(_gpu(index) for index in range(4)),
        "topology": _topology(),
    }
    kwargs.update(overrides)
    return build_plan(**kwargs)


def _kerberos(ttl=1801):
    return {
        "classification": "READY",
        "principal": "sitian@BYTEDANCE.COM",
        "tgt_principal": "krbtgt/BYTEDANCE.COM@BYTEDANCE.COM",
        "remaining_lifetime_seconds": ttl,
    }


def test_plan_freezes_campaign_and_safe_remote_paths():
    plan = _plan()
    forward = ["P0", "P1", "Q0", "Q1", "Q2"]
    reverse = list(reversed(forward))

    assert MINIMUM_KERBEROS_LIFETIME_SECONDS == 1_800
    assert plan["schema_version"] == (
        "qwen38.topology-local-tp2-whole-model-plan.v1"
    )
    assert plan["model_revision"] == MODEL_REVISION
    assert plan["pair_groups"] == [[0, 1], [2, 3]]
    assert plan["campaign_epochs"] == [
        {"epoch": 0, "arm": "baseline", "workload_order": forward},
        {"epoch": 1, "arm": "candidate", "workload_order": reverse},
        {"epoch": 2, "arm": "candidate", "workload_order": forward},
        {"epoch": 3, "arm": "baseline", "workload_order": reverse},
    ]
    approved = PurePosixPath(APPROVED_REMOTE_ROOT)
    assert all(
        PurePosixPath(path).is_relative_to(approved)
        for key, path in plan.items()
        if key.endswith("_root")
    )


def test_remote_epoch_command_binds_real_worker_cli_and_selected_gpus():
    plan = _plan()

    command = build_remote_epoch_command(
        plan,
        plan["campaign_epochs"][2],
    )

    assert command["argv"][:2] == [
        DEFAULT_REMOTE_PYTHON,
        (
            f"{plan['source_root']}/tools/"
            "qwen38_topology_local_tp2_whole_model_worker.py"
        ),
    ]
    assert command["argv"][2] == "performance-epoch"
    assert command["environment"]["CUDA_VISIBLE_DEVICES"] == "0,1,2,3"
    assert command["environment"]["PYTHONPATH"] == plan["source_root"]
    assert command["argv"][
        command["argv"].index("--model-root") + 1
    ] == DEFAULT_MODEL_ROOT
    assert command["argv"][
        command["argv"].index("--output-root") + 1
    ].startswith(plan["controller_root"] + "/")
    assert command["argv"][
        command["argv"].index("--workload-order") + 1
    ] == "P0,P1,Q0,Q1,Q2"


def test_remote_correctness_and_service_commands_are_attempt_local():
    plan = _plan()

    correctness = build_remote_correctness_command(plan)
    service = build_remote_service_command(plan)

    assert correctness["argv"][2] == "correctness"
    assert service["argv"][2] == "service-control"
    assert correctness["environment"]["CUDA_VISIBLE_DEVICES"] == "0,1,2,3"
    assert service["environment"]["CUDA_VISIBLE_DEVICES"] == "0,1,2,3"
    assert correctness["output_root"].startswith(
        plan["controller_root"] + "/"
    )
    assert service["output_root"].startswith(plan["controller_root"] + "/")
    assert service["argv"][
        service["argv"].index("--pair-devices") + 1
    ] == "0,1;2,3"
    assert service["argv"][
        service["argv"].index("--workloads") + 1
    ] == "Q0,Q1,Q2"


def test_plan_rejects_path_escape_reused_tag_and_insufficient_gpus():
    with pytest.raises(ValueError, match="approved"):
        _plan(remote_root="/tmp")
    with pytest.raises(ValueError, match="fresh"):
        _plan(attempt_exists=True)
    with pytest.raises(ValueError, match="four"):
        _plan(gpu_inventory=tuple(_gpu(index) for index in range(3)))


def test_plan_validation_rejects_nonoptimal_pair_mapping():
    plan = _plan()
    plan["pair_groups"] = [[0, 2], [1, 3]]

    with pytest.raises(ValueError, match="optimal"):
        controller._validate_plan(plan)


def test_kerberos_floor_and_source_model_drift():
    assert controller._validate_kerberos(_kerberos(1799)) is False
    assert controller._validate_kerberos(_kerberos(1800)) is True
    plan = _plan()

    with pytest.raises(ValueError, match="source drift"):
        controller._validate_launch_identity(
            plan,
            source_revision="f" * 40,
            model_revision=MODEL_REVISION,
        )
    with pytest.raises(ValueError, match="model drift"):
        controller._validate_launch_identity(
            plan,
            source_revision=plan["source_revision"],
            model_revision="f" * 40,
        )


def test_ssh_retries_only_transport_255(monkeypatch):
    codes = iter((255, 255, 255, 0))
    sleeps = []
    monkeypatch.setattr(controller.time, "sleep", sleeps.append)

    result = run_ssh_with_retry(
        ["ssh", "host", "true"],
        retry_count=3,
        runner=lambda *_args, **_kwargs: SimpleNamespace(
            returncode=next(codes),
            stdout="",
            stderr="",
        ),
    )

    assert result.returncode == 0
    assert sleeps == [1.0, 2.0, 4.0]
    sleeps.clear()
    result = run_ssh_with_retry(
        ["ssh", "host", "false"],
        retry_count=3,
        runner=lambda *_args, **_kwargs: SimpleNamespace(
            returncode=7,
            stdout="",
            stderr="",
        ),
    )
    assert result.returncode == 7
    assert sleeps == []


def test_owned_cleanup_excludes_foreign_and_unregistered_processes():
    tag = _plan()["attempt_tag"]
    rows = [
        {"pid": 10, "pgid": 10, "attempt_tag": tag},
        {"pid": 11, "pgid": 10, "attempt_tag": tag},
        {"pid": 20, "pgid": 20, "attempt_tag": "foreign"},
        {"pid": 30, "pgid": 30, "attempt_tag": tag},
    ]

    assert select_owned_process_groups(
        rows,
        attempt_tag=tag,
        registered_pgids={10},
    ) == (10,)


def test_dry_run_is_read_only():
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


def test_attempt_runs_all_epochs_control_and_dual_verification():
    events = []
    plan = _plan()

    result = run_attempt(
        plan,
        kerberos_probe=lambda: _kerberos(),
        gpu_probe=lambda: [_gpu(index) for index in range(4)],
        identity_probe=lambda: {
            "source_revision": plan["source_revision"],
            "model_revision": MODEL_REVISION,
        },
        remote_writer=lambda _plan: events.append("write"),
        correctness_runner=lambda: (
            events.append("correctness")
            or {
                "registered_pgids": [90],
                "process_rows": [{
                    "pid": 90,
                    "pgid": 90,
                    "attempt_tag": plan["attempt_tag"],
                }],
                "exit_code": 0,
            }
        ),
        epoch_runner=lambda epoch: (
            events.append(("epoch", epoch["epoch"]))
            or {
                "registered_pgids": [100 + epoch["epoch"]],
                "process_rows": [{
                    "pid": 100 + epoch["epoch"],
                    "pgid": 100 + epoch["epoch"],
                    "attempt_tag": plan["attempt_tag"],
                }],
                "exit_code": 0,
            }
        ),
        service_runner=lambda: (
            events.append("service")
            or {
                "registered_pgids": [200],
                "process_rows": [{
                    "pid": 200,
                    "pgid": 200,
                    "attempt_tag": plan["attempt_tag"],
                }],
                "exit_code": 0,
            }
        ),
        remote_assembler=lambda: events.append("assemble") or {
            "classification": "NO_GO_PERFORMANCE"
        },
        remote_verifier=lambda: events.append("remote_verify") or b"same\n",
        downloader=lambda: events.append("download"),
        local_verifier=lambda: events.append("local_verify") or b"same\n",
    )

    assert result["classification"] == "NO_GO_PERFORMANCE"
    assert result["worker_started"] is True
    assert events == [
        "write",
        "correctness",
        ("epoch", 0),
        ("epoch", 1),
        ("epoch", 2),
        ("epoch", 3),
        "service",
        "assemble",
        "remote_verify",
        "download",
        "local_verify",
    ]
    assert [row["stage"] for row in result["resource_samples"]] == [
        "entry",
        "pre_correctness",
        "post_correctness",
        "pre_epoch_0",
        "post_launch_0",
        "pre_epoch_1",
        "post_launch_1",
        "pre_epoch_2",
        "post_launch_2",
        "pre_epoch_3",
        "post_launch_3",
        "pre_service_control",
        "post_service_control",
        "terminal",
    ]


def test_post_staging_gpu_drift_and_foreign_pid_fail_closed():
    plan = _plan()
    inventories = iter((
        [_gpu(index) for index in range(4)],
        [_gpu(0, memory=2048), *[_gpu(index) for index in range(1, 4)]],
    ))
    with pytest.raises(RuntimeError, match="strict-clean"):
        run_attempt(
            plan,
            kerberos_probe=lambda: _kerberos(),
            gpu_probe=lambda: next(inventories),
            identity_probe=lambda: {
                "source_revision": plan["source_revision"],
                "model_revision": MODEL_REVISION,
            },
            remote_writer=lambda _plan: None,
            epoch_runner=lambda _epoch: pytest.fail("must not launch"),
        )

    with pytest.raises(RuntimeError, match="foreign"):
        controller._validate_launched_processes(
            plan,
            {
                "registered_pgids": [10],
                "process_rows": [{
                    "pid": 99,
                    "pgid": 99,
                    "attempt_tag": "foreign",
                }],
                "exit_code": 0,
            },
        )


def test_controller_contains_no_auth_renewal():
    source = inspect.getsource(controller)

    assert "ki" + "nit" not in source
    assert "kre" + "new" not in source


def test_main_full_run_installs_default_remote_adapters(
    tmp_path,
    monkeypatch,
):
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(_plan()), encoding="utf-8")
    observed = {}

    def adapter_factory(args, plan):
        observed["args"] = args
        observed["plan"] = plan
        return {
            name: (lambda *_args, **_kwargs: None)
            for name in (
                "gpu_probe",
                "identity_probe",
                "remote_writer",
                "correctness_runner",
                "epoch_runner",
                "service_runner",
                "remote_assembler",
                "remote_verifier",
                "downloader",
                "local_verifier",
            )
        }

    def fake_run_attempt(plan, **kwargs):
        observed["callbacks"] = kwargs
        return {
            "classification": "NO_GO_PERFORMANCE",
            "worker_started": True,
        }

    monkeypatch.setattr(controller, "run_attempt", fake_run_attempt)

    assert controller.main(
        ["--plan", str(plan_path)],
        adapter_factory=adapter_factory,
        printer=lambda _value: None,
    ) == 0
    assert observed["plan"]["attempt_tag"] == _plan()["attempt_tag"]
    assert all(
        callable(observed["callbacks"][name])
        for name in (
            "gpu_probe",
            "identity_probe",
            "remote_writer",
            "correctness_runner",
            "epoch_runner",
            "service_runner",
            "remote_assembler",
            "remote_verifier",
            "downloader",
            "local_verifier",
        )
    )
