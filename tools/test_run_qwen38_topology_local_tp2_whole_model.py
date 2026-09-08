from __future__ import annotations

import copy
import inspect
import json
from pathlib import PurePosixPath
import time
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


def _gpu(
    index,
    *,
    memory=0,
    utilization=0,
    power_watts=70.0,
    processes=(),
):
    return {
        "gpu_index": index,
        "gpu_uuid": f"GPU-{index}",
        "memory_used_mib": memory,
        "utilization_percent": utilization,
        "power_watts": power_watts,
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


def _cross_topology():
    links = {(0, 2): "PIX", (1, 3): "PIX"}
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
    assert correctness["argv"][
        correctness["argv"].index("--source-revision") + 1
    ] == plan["source_revision"]


def test_service_command_maps_logical_pairs_to_selected_physical_gpus():
    plan = _plan(
        gpu_inventory=tuple(_gpu(index) for index in (4, 5, 6, 7)),
    )

    service = build_remote_service_command(plan)

    assert service["environment"]["CUDA_VISIBLE_DEVICES"] == "4,5,6,7"
    assert service["argv"][
        service["argv"].index("--pair-devices") + 1
    ] == "4,5;6,7"


def test_plan_reorders_global_ranks_to_make_best_physical_pairs_adjacent():
    plan = _plan(
        gpu_inventory=tuple(_gpu(index) for index in (4, 5, 6, 7)),
        topology=_cross_topology(),
    )

    assert [
        row["gpu_index"] for row in plan["selected_gpus"]
    ] == [4, 6, 5, 7]
    assert plan["pair_groups"] == [[0, 1], [2, 3]]
    assert [
        row["gpu_index"] for row in plan["gpu_rank_mapping"]
    ] == [4, 6, 5, 7]
    service = build_remote_service_command(plan)
    assert service["environment"]["CUDA_VISIBLE_DEVICES"] == "4,6,5,7"
    assert service["argv"][
        service["argv"].index("--pair-devices") + 1
    ] == "4,6;5,7"


def test_plan_validation_rejects_gpu_rank_mapping_drift():
    plan = _plan()
    plan["gpu_rank_mapping"][0]["gpu_index"] = 3

    with pytest.raises(ValueError, match="rank mapping"):
        controller._validate_plan(plan)


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


def test_gpu_power_rows_are_merged_by_index_and_uuid():
    inventory = [_gpu(2), _gpu(5)]
    for row in inventory:
        row.pop("power_watts")

    merged = controller._merge_gpu_power(
        inventory,
        "5, GPU-5, 81.25\n2, GPU-2, 72.5\n",
    )

    assert merged == [
        _gpu(2, power_watts=72.5),
        _gpu(5, power_watts=81.25),
    ]


@pytest.mark.parametrize(
    "power_csv",
    (
        "0, GPU-0, [N/A]\n1, GPU-1, 70\n",
        "0, GPU-drift, 70\n1, GPU-1, 70\n",
        "0, GPU-0, -1\n1, GPU-1, 70\n",
        "0, GPU-0, 70\n",
    ),
)
def test_gpu_power_rows_fail_closed_when_incomplete_or_invalid(power_csv):
    inventory = [_gpu(0), _gpu(1)]
    for row in inventory:
        row.pop("power_watts")

    with pytest.raises(ValueError, match="power"):
        controller._merge_gpu_power(inventory, power_csv)


def test_remote_worker_timeout_reaps_registered_attempt_group(monkeypatch):
    plan = _plan()
    args = SimpleNamespace(command_timeout_s=1)
    calls = []

    def remote_json(_args, argv, timeout_s):
        calls.append((_args, argv, timeout_s))
        if len(calls) == 1:
            raise controller.subprocess.TimeoutExpired(
                cmd=["ssh", "worker"],
                timeout=timeout_s,
            )
        return {
            "classification": "CLEANED",
            "attempt_tag": plan["attempt_tag"],
            "label": "correctness",
            "pgid": 123,
        }

    monkeypatch.setattr(controller, "_remote_json", remote_json)

    with pytest.raises(controller.subprocess.TimeoutExpired):
        controller._run_remote_worker_command(
            args,
            plan,
            {"argv": ["worker"], "environment": {}},
            "correctness",
        )

    assert len(calls) == 2
    assert calls[0][0].retry_count == 0
    cleanup_argv = calls[1][1]
    assert plan["attempt_tag"] in cleanup_argv
    assert "correctness" in cleanup_argv
    assert (
        f"{plan['controller_root']}/owned-processes/"
        "correctness.json"
    ) in cleanup_argv


def test_remote_worker_cleanup_waits_for_delayed_registration(monkeypatch):
    plan = _plan()
    args = SimpleNamespace(
        command_timeout_s=1,
        retry_count=3,
    )
    calls = []

    def remote_json(_args, argv, timeout_s):
        calls.append((_args, argv, timeout_s))
        if len(calls) == 1:
            raise controller.subprocess.TimeoutExpired(
                cmd=["ssh", "worker"],
                timeout=timeout_s,
            )
        return {
            "classification": "CLEANED",
            "attempt_tag": plan["attempt_tag"],
            "label": "correctness",
            "pgid": 123,
        }

    monkeypatch.setattr(controller, "_remote_json", remote_json)

    with pytest.raises(controller.subprocess.TimeoutExpired):
        controller._run_remote_worker_command(
            args,
            plan,
            {"argv": ["worker"], "environment": {}},
            "correctness",
        )

    cleanup_script = calls[1][1][2]
    assert (
        "while not os.path.isfile(path) and "
        "time.monotonic()<registration_deadline:"
    ) in cleanup_script
    assert cleanup_script.index("registration_deadline=") < (
        cleanup_script.index("if not os.path.isfile(path):")
    )


def test_remote_worker_cleanup_binds_pid_generation_and_exact_argv(
    monkeypatch,
):
    plan = _plan()
    args = SimpleNamespace(
        command_timeout_s=1,
        retry_count=3,
    )
    calls = []

    def remote_json(_args, argv, timeout_s):
        calls.append((_args, argv, timeout_s))
        if len(calls) == 1:
            raise controller.subprocess.TimeoutExpired(
                cmd=["ssh", "worker"],
                timeout=timeout_s,
            )
        return {
            "classification": "OWNERSHIP_MISMATCH",
            "attempt_tag": plan["attempt_tag"],
            "label": "correctness",
            "pgid": 123,
        }

    monkeypatch.setattr(controller, "_remote_json", remote_json)

    with pytest.raises(RuntimeError, match="cleanup could not be verified"):
        controller._run_remote_worker_command(
            args,
            plan,
            {"argv": ["worker"], "environment": {}},
            "correctness",
        )

    launch_script = calls[0][1][2]
    cleanup_script = calls[1][1][2]
    assert "'start_time_ticks':start_time_ticks" in launch_script
    assert "'argv':argv" in launch_script
    assert "current_start_time_ticks!=start_time_ticks" in cleanup_script
    assert "current_argv!=registered_argv" in cleanup_script
    assert "os.getpgid(pid)!=pgid" in cleanup_script
    assert cleanup_script.index("OWNERSHIP_MISMATCH") < (
        cleanup_script.index("os.killpg(pgid,signal.SIGTERM)")
    )


def test_download_timeout_reaps_owned_ssh_sender(tmp_path, monkeypatch):
    events = []

    class Pipe:
        def close(self):
            events.append("close_stdout")

        def read(self):
            return b""

    class Sender:
        stdout = Pipe()
        stderr = Pipe()

        def terminate(self):
            events.append("terminate")

        def kill(self):
            events.append("kill")

        def wait(self, timeout=None):
            events.append(("wait", timeout))
            return -15

    sender = Sender()
    monkeypatch.setattr(
        controller.subprocess,
        "Popen",
        lambda *_args, **_kwargs: sender,
    )
    monkeypatch.setattr(
        controller.subprocess,
        "run",
        lambda *_args, **kwargs: (
            (_ for _ in ()).throw(
                controller.subprocess.TimeoutExpired(
                    cmd=["tar"],
                    timeout=kwargs["timeout"],
                )
            )
        ),
    )

    with pytest.raises(controller.subprocess.TimeoutExpired):
        controller._download_final_bundle(
            SimpleNamespace(
                ssh_target="host",
                proxy_host="proxy",
                command_timeout_s=1,
            ),
            _plan(),
            tmp_path / "attempt",
        )

    assert events[:2] == ["close_stdout", "terminate"]
    assert events[2][0] == "wait"
    assert "kill" not in events


def test_download_drains_sender_stderr_without_blocking_read(
    tmp_path,
    monkeypatch,
):
    events = []

    class Pipe:
        def close(self):
            events.append("close_stdout")

        def read(self):
            raise AssertionError("stderr must be drained by communicate")

    class Sender:
        stdout = Pipe()
        stderr = Pipe()
        returncode = 0

        def communicate(self, timeout=None):
            events.append(("communicate", timeout))
            return b"", b""

        def terminate(self):
            events.append("terminate")

        def kill(self):
            events.append("kill")

        def wait(self, timeout=None):
            events.append(("wait", timeout))
            return self.returncode

    sender = Sender()
    monkeypatch.setattr(
        controller.subprocess,
        "Popen",
        lambda *_args, **_kwargs: sender,
    )
    monkeypatch.setattr(
        controller.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(
            returncode=0,
            stderr=b"",
        ),
    )

    controller._download_final_bundle(
        SimpleNamespace(
            ssh_target="host",
            proxy_host="proxy",
            command_timeout_s=1,
        ),
        _plan(),
        tmp_path / "attempt",
    )

    assert events == ["close_stdout", ("communicate", 600)]


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


def test_default_kerberos_probe_uses_whole_model_launch_floor(monkeypatch):
    observed = []
    monkeypatch.setattr(
        controller,
        "query_local_kerberos",
        lambda **kwargs: (
            observed.append(kwargs) or _kerberos()
        ),
    )

    result = run_attempt(
        _plan(),
        dry_run=True,
        gpu_probe=lambda: [_gpu(index) for index in range(4)],
    )

    assert result["classification"] == "DRY_RUN_READY"
    assert observed == [{
        "minimum_lifetime_seconds": 1_800,
    }]


def test_attempt_runs_all_epochs_control_and_dual_verification():
    events = []
    assembled_resource_stages = []
    plan = _plan()

    def launch(event, pgid):
        events.append(event)
        time.sleep(0.005)
        return {
            "registered_pgids": [pgid],
            "process_rows": [{
                "pid": pgid,
                "pgid": pgid,
                "attempt_tag": plan["attempt_tag"],
            }],
            "exit_code": 0,
        }

    def assemble(resources):
        assembled_resource_stages.extend(
            row["stage"] for row in resources
        )
        events.append("assemble")
        return {"classification": "NO_GO_PERFORMANCE"}

    result = run_attempt(
        plan,
        kerberos_probe=lambda: _kerberos(),
        gpu_probe=lambda: [_gpu(index) for index in range(4)],
        identity_probe=lambda: {
            "source_revision": plan["source_revision"],
            "model_revision": MODEL_REVISION,
        },
        remote_writer=lambda _plan: events.append("write"),
        correctness_runner=lambda: launch("correctness", 90),
        epoch_runner=lambda epoch: launch(
            ("epoch", epoch["epoch"]),
            100 + epoch["epoch"],
        ),
        service_runner=lambda: launch("service", 200),
        remote_assembler=assemble,
        remote_verifier=lambda: events.append("remote_verify") or b"same\n",
        downloader=lambda: events.append("download"),
        local_verifier=lambda: events.append("local_verify") or b"same\n",
        runtime_sample_interval_s=0.001,
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
    boundary_stages = [
        row["stage"]
        for row in result["resource_samples"]
        if row["measurement_scope"] == "boundary"
    ]
    assert boundary_stages == [
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
    assert {
        row["run_label"]
        for row in result["resource_samples"]
        if row["measurement_scope"] == "runtime"
    } == {
        "correctness",
        "epoch_0",
        "epoch_1",
        "epoch_2",
        "epoch_3",
        "service_control",
    }
    assert assembled_resource_stages == [
        row["stage"] for row in result["resource_samples"]
    ]


def test_attempt_samples_gpu_telemetry_while_each_worker_is_running():
    plan = _plan()
    worker_running = False

    def gpu_probe():
        nonlocal worker_running
        if worker_running:
            return [
                _gpu(
                    index,
                    memory=4096,
                    utilization=80,
                    power_watts=250.0,
                    processes=({"pid": 1000 + index},),
                )
                for index in range(4)
            ]
        return [_gpu(index) for index in range(4)]

    def launch(pgid):
        nonlocal worker_running
        worker_running = True
        try:
            time.sleep(0.01)
            return {
                "registered_pgids": [pgid],
                "process_rows": [{
                    "pid": pgid,
                    "pgid": pgid,
                    "attempt_tag": plan["attempt_tag"],
                }],
                "exit_code": 0,
            }
        finally:
            worker_running = False

    result = run_attempt(
        plan,
        kerberos_probe=lambda: _kerberos(),
        gpu_probe=gpu_probe,
        identity_probe=lambda: {
            "source_revision": plan["source_revision"],
            "model_revision": MODEL_REVISION,
        },
        remote_writer=lambda _plan: None,
        correctness_runner=lambda: launch(90),
        epoch_runner=lambda epoch: launch(100 + epoch["epoch"]),
        service_runner=lambda: launch(200),
        remote_assembler=lambda _resources: {
            "classification": "NO_GO_PERFORMANCE",
        },
        remote_verifier=lambda: b"same\n",
        downloader=lambda: None,
        local_verifier=lambda: b"same\n",
        runtime_sample_interval_s=0.001,
    )

    runtime_rows = [
        row
        for row in result["resource_samples"]
        if row.get("measurement_scope") == "runtime"
    ]
    assert {
        row["run_label"] for row in runtime_rows
    } == {
        "correctness",
        "epoch_0",
        "epoch_1",
        "epoch_2",
        "epoch_3",
        "service_control",
    }
    assert all(
        gpu["utilization_percent"] == 80
        for row in runtime_rows
        for gpu in row["gpu_inventory"]
    )


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
