#!/usr/bin/env python3
"""Dependency-light tests for the TP4 decode replay controller."""

from __future__ import annotations

import json
from pathlib import Path, PurePosixPath
import subprocess
import sys
import tempfile
from types import SimpleNamespace

import run_tp4_decode_replay as controller
from run_tp4_decode_replay import (
    MODEL_REVISION,
    ProductionAdapter,
    REMOTE_ROOT,
    _remote_driver_source,
    build_plan,
    main,
    monitor_and_run,
    run_attempt,
)


RUN_TAG = "20260831-qwen38-tp4-decode-replay-r1"


def _source(**overrides) -> dict:
    value = {
        "schema_version": "tinyllmforge.tp4-decode-replay-source.v1",
        "run_tag": RUN_TAG,
        "source_revision": "a" * 40,
        "source_tree_sha256": "b" * 64,
        "model_repository": "Qwen/Qwen3.8-27B",
        "model_revision": MODEL_REVISION,
    }
    value.update(overrides)
    return value


def _gpu(index: int, *, uuid=None, memory=0, utilization=0, processes=()):
    return {
        "gpu_index": index,
        "gpu_uuid": uuid or f"GPU-{index:04d}",
        "memory_used_mib": memory,
        "utilization_percent": utilization,
        "compute_processes": list(processes),
    }


def _plan(**overrides):
    arguments = {
        "run_tag": RUN_TAG,
        "source_identity": _source(),
        "selected_gpus": [_gpu(index) for index in range(4)],
    }
    arguments.update(overrides)
    return build_plan(**arguments)


def _expect_error(action, message: str) -> None:
    try:
        action()
    except (RuntimeError, ValueError) as exc:
        assert message in str(exc), str(exc)
    else:
        raise AssertionError(f"expected error containing {message!r}")


def test_plan_freezes_paths_model_and_four_clean_gpus():
    plan = _plan()
    approved = PurePosixPath(REMOTE_ROOT)
    assert plan["remote_root"].startswith(
        "/data00/home/sitian/tinyllmforge-workspaces/"
    )
    assert "/tmp" not in plan["remote_root"]
    assert plan["selected_gpu_indices"] == [0, 1, 2, 3]
    assert plan["world_size"] == 4
    assert plan["model_revision"] == MODEL_REVISION
    assert plan["process_environment"]["PYTHONNOUSERSITE"] == "1"
    assert all(
        PurePosixPath(value).is_relative_to(approved)
        for key, value in plan["paths"].items()
        if key.endswith("_root") or key.endswith("_path")
    )
    assert all(
        PurePosixPath(value).is_relative_to(
            PurePosixPath(plan["paths"]["attempt_root"])
        )
        for value in plan["environment"].values()
    )


def test_plan_rejects_unclean_or_duplicate_gpu_identity():
    _expect_error(
        lambda: _plan(
            selected_gpus=[
                _gpu(0, processes=[{"pid": 9}]),
                _gpu(1),
                _gpu(2),
                _gpu(3),
            ]
        ),
        "four strict-clean GPUs",
    )
    _expect_error(
        lambda: _plan(selected_gpus=[
            _gpu(0, uuid="GPU-duplicate"),
            _gpu(1, uuid="GPU-duplicate"),
            _gpu(2),
            _gpu(3),
        ]),
        "duplicate GPU UUID",
    )


def test_plan_rejects_source_drift_and_unsafe_run_tag():
    _expect_error(
        lambda: _plan(
            source_identity=_source(run_tag="different-tag")
        ),
        "source identity",
    )
    _expect_error(
        lambda: _plan(run_tag="../escape"),
        "run tag",
    )


class _Adapter:
    def __init__(
        self,
        *,
        fail_at=None,
        cleanup_error=None,
        source=None,
    ):
        self.events = []
        self.fail_at = fail_at
        self.cleanup_error = cleanup_error
        self.source = source or _source()

    def _event(self, name, result):
        self.events.append(name)
        if self.fail_at == name:
            raise RuntimeError(f"{name} failed")
        return result

    def freeze_source(self, plan):
        return self._event("source_freeze", self.source)

    def ssh_storage_preflight(self, plan, source):
        assert source == self.source
        return self._event(
            "ssh_storage_preflight",
            {
                "classification": "PASS",
                "attempt_exists": False,
                "remote_root": plan["remote_root"],
            },
        )

    def strict_clean_admission(self, plan, preflight):
        assert preflight["classification"] == "PASS"
        return self._event(
            "strict_clean_admission",
            {
                "classification": "READY",
                "selected_gpus": plan["selected_gpus"],
            },
        )

    def launch(self, plan, admission):
        assert admission["classification"] == "READY"
        return self._event("launch", {"owned_pids": [101, 102, 103, 104]})

    def wait(self, plan, launch):
        assert launch["owned_pids"]
        return self._event("wait", {"exit_code": 0})

    def download(self, plan, wait):
        assert wait["exit_code"] == 0
        return self._event("download", {"downloaded": True})

    def assemble(self, plan, download):
        assert download["downloaded"] is True
        return self._event(
            "assemble",
            {"classification": "GO_STAGE1_JUSTIFIED"},
        )

    def remote_verify(self, plan, assembled):
        assert assembled["classification"] == "GO_STAGE1_JUSTIFIED"
        return self._event(
            "remote_verify",
            {
                "classification": "GO_STAGE1_JUSTIFIED",
                "failed_gates": [],
                "verified_hashes": True,
                "producer_classification_matches": True,
                "summary_matches": True,
                "metrics": {},
            },
        )

    def write_remote_post_verification_manifest(
        self,
        plan,
        remote_verification,
    ):
        assert (
            remote_verification["classification"]
            == "GO_STAGE1_JUSTIFIED"
        )
        return self._event(
            "remote_post_verification_manifest",
            {"verified": True},
        )

    def local_verify(self, plan, assembled):
        assert assembled["classification"] == "GO_STAGE1_JUSTIFIED"
        return self._event(
            "local_frozen_source_verify",
            {
                "classification": "GO_STAGE1_JUSTIFIED",
                "failed_gates": [],
                "verified_hashes": True,
                "producer_classification_matches": True,
                "summary_matches": True,
                "metrics": {},
            },
        )

    def validate_cleanup(self, plan, launch):
        self.events.append("cleanup_validation")
        if self.cleanup_error is not None:
            raise RuntimeError(self.cleanup_error)
        return {"classification": "CLEAN"}


def test_run_attempt_enforces_the_frozen_operation_order():
    adapter = _Adapter()
    result = run_attempt(plan=_plan(), adapter=adapter)
    assert result["classification"] == "GO_STAGE1_JUSTIFIED"
    assert adapter.events == [
        "source_freeze",
        "ssh_storage_preflight",
        "strict_clean_admission",
        "launch",
        "wait",
        "download",
        "assemble",
        "remote_verify",
        "remote_post_verification_manifest",
        "local_frozen_source_verify",
        "cleanup_validation",
    ]


def test_cleanup_always_runs_and_preserves_original_failure():
    adapter = _Adapter(
        fail_at="wait",
        cleanup_error="cleanup also failed",
    )
    try:
        run_attempt(plan=_plan(), adapter=adapter)
    except RuntimeError as exc:
        assert str(exc) == "wait failed"
        assert exc.__cause__ is not None
        assert "cleanup also failed" in str(exc.__cause__)
    else:
        raise AssertionError("operation failure was swallowed")
    assert adapter.events[-1] == "cleanup_validation"


def test_cleanup_failure_overrides_an_otherwise_successful_run():
    adapter = _Adapter(cleanup_error="cleanup failed")
    _expect_error(
        lambda: run_attempt(plan=_plan(), adapter=adapter),
        "cleanup failed",
    )


def test_preflight_rejects_existing_tag_before_launch():
    adapter = _Adapter(fail_at="ssh_storage_preflight")
    _expect_error(
        lambda: run_attempt(plan=_plan(), adapter=adapter),
        "ssh_storage_preflight failed",
    )
    assert "launch" not in adapter.events


def test_run_attempt_rejects_a_verdict_without_verifier_evidence():
    class MissingEvidenceAdapter(_Adapter):
        def remote_verify(self, plan, assembled):
            return self._event(
                "remote_verify",
                {"classification": "GO_STAGE1_JUSTIFIED"},
            )

    adapter = MissingEvidenceAdapter()
    _expect_error(
        lambda: run_attempt(plan=_plan(), adapter=adapter),
        "verification receipt",
    )
    assert adapter.events[-1] == "cleanup_validation"


def test_monitor_and_run_launches_immediately_after_local_admission():
    adapter = _Adapter()
    monitor_calls = []

    result = monitor_and_run(
        run_tag=RUN_TAG,
        gpu_monitor=lambda: (
            monitor_calls.append("poll")
            or {
                "classification": "READY",
                "selected_gpus": [_gpu(index) for index in range(4)],
            }
        ),
        adapter=adapter,
    )

    assert result["classification"] == "GO_STAGE1_JUSTIFIED"
    assert monitor_calls == ["poll"]
    assert adapter.events[0] == "source_freeze"
    assert "launch" in adapter.events


def test_plan_only_performs_no_gpu_query_or_remote_operation():
    with tempfile.TemporaryDirectory() as directory:
        output = Path(directory) / "plan.json"
        return_code = main(
            [
                "plan-only",
                "--run-tag",
                RUN_TAG,
                "--output",
                str(output),
            ],
            source_identity_builder=lambda run_tag: _source(
                run_tag=run_tag
            ),
            gpu_monitor=lambda: (_ for _ in ()).throw(
                AssertionError("plan-only queried GPUs")
            ),
            adapter_factory=lambda: (_ for _ in ()).throw(
                AssertionError("plan-only created remote adapter")
            ),
        )
        payload = json.loads(output.read_text(encoding="utf-8"))
    assert return_code == 0
    assert payload["mode"] == "plan-only"
    assert payload["selected_gpu_indices"] == [0, 1, 2, 3]


def test_controller_supports_direct_script_execution():
    script = Path(__file__).with_name("run_tp4_decode_replay.py")
    result = subprocess.run(
        [sys.executable, str(script), "--help"],
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_remote_upload_retries_transient_ssh_disconnect():
    calls = []
    sleeps = []
    results = iter((
        SimpleNamespace(
            returncode=255,
            stdout=b"",
            stderr=b"Connection closed by UNKNOWN port 65535",
        ),
        SimpleNamespace(returncode=0, stdout=b"", stderr=b""),
    ))

    def command_runner(arguments, **kwargs):
        calls.append((list(arguments), dict(kwargs)))
        return next(results)

    original_run = controller.subprocess.run
    original_sleep = controller.time.sleep
    controller.subprocess.run = command_runner
    controller.time.sleep = sleeps.append
    try:
        with tempfile.TemporaryDirectory() as directory:
            adapter = ProductionAdapter(
                run_tag=RUN_TAG,
                local_attempt_root=Path(directory) / "attempt",
                local_command_runner=command_runner,
                retry_count=3,
            )
            adapter._upload_bytes(
                f"{REMOTE_ROOT}/{RUN_TAG}/raw/source.patch",
                b"payload",
            )
    finally:
        controller.subprocess.run = original_run
        controller.time.sleep = original_sleep

    assert len(calls) == 2
    assert all(call[1]["input"] == b"payload" for call in calls)
    assert sleeps == [1.0]


def test_local_verifier_executes_from_the_frozen_source_revision():
    calls = []

    def command_runner(arguments, **kwargs):
        calls.append((list(arguments), dict(kwargs)))
        if arguments[:3] == ["git", "-C", str(
            Path(controller.__file__).resolve().parents[1]
        )]:
            assert arguments[3:6] == [
                "archive",
                "--format=tar",
                "a" * 40,
            ]
            return SimpleNamespace(
                returncode=0,
                stdout=b"frozen-archive",
                stderr=b"",
            )
        if arguments[:3] == ["tar", "-xf", "-"]:
            assert kwargs["input"] == b"frozen-archive"
            return SimpleNamespace(returncode=0, stdout=b"", stderr=b"")
        assert arguments[0] == sys.executable
        assert arguments[1].endswith(
            "/tools/verify_tp4_decode_replay.py"
        )
        assert arguments[2:] == [
            "--bundle-root",
            str(adapter.local_bundle_root),
        ]
        return SimpleNamespace(
            returncode=0,
            stdout=json.dumps({
                "classification": "NO_GO_PERFORMANCE",
                "failed_gates": ["aggregate_output_throughput_ratio"],
            }),
            stderr="",
        )

    with tempfile.TemporaryDirectory() as directory:
        adapter = ProductionAdapter(
            run_tag=RUN_TAG,
            local_attempt_root=Path(directory) / "attempt",
            local_command_runner=command_runner,
        )
        adapter._source = _source()
        original = controller.verify_bundle
        controller.verify_bundle = lambda root: (_ for _ in ()).throw(
            AssertionError("mutable in-process verifier was used")
        )
        try:
            result = adapter.local_verify(_plan(), {})
        finally:
            controller.verify_bundle = original

    assert result["classification"] == "NO_GO_PERFORMANCE"
    assert len(calls) == 3


def test_remote_driver_never_reuses_a_dynamic_port_across_arms():
    source = _remote_driver_source()
    assert "used_ports = set()" in source
    assert "if port not in used_ports:" in source
    assert "used_ports.add(port)" in source
    assert "worker.create_engine_with_rendezvous_retry(" in source
    assert "engine_config=kwargs" in source
    assert "port_factory=free_port" in source
    assert source.count("engine_factory=engine_factory") == 1


def test_kerberos_guard_covers_the_full_remote_command_window():
    requested_lifetimes = []

    def kerberos_query(**kwargs):
        requested_lifetimes.append(
            kwargs["minimum_lifetime_seconds"]
        )
        return {"classification": "READY"}

    with tempfile.TemporaryDirectory() as directory:
        adapter = ProductionAdapter(
            run_tag=RUN_TAG,
            local_attempt_root=Path(directory) / "attempt",
            command_timeout_s=21_600,
            kerberos_query=kerberos_query,
        )
        adapter._remote = lambda arguments: SimpleNamespace(
            returncode=0,
            stdout=json.dumps({
                "base_ready": True,
                "remote_root_safe": True,
                "attempt_exists": False,
                "model_ready": True,
                "model_revision_matches": True,
                "text_profile": {
                    "num_hidden_layers": 64,
                    "hidden_size": 5120,
                    "vocab_size": 248320,
                    "dtype": "bfloat16",
                },
            }),
            stderr="",
        )
        receipt = adapter.ssh_storage_preflight(
            _plan(),
            _source(),
        )

    assert receipt["classification"] == "PASS"
    assert requested_lifetimes == [22_500]


def test_failed_kerberos_preflight_is_persisted_as_incomplete():
    with tempfile.TemporaryDirectory() as directory:
        adapter = ProductionAdapter(
            run_tag=RUN_TAG,
            local_attempt_root=Path(directory) / "attempt",
            kerberos_query=lambda **kwargs: {
                "classification": "BLOCKED_KERBEROS_TTL",
                "reason": "ticket expired",
            },
        )
        receipt = adapter.ssh_storage_preflight(
            _plan(),
            _source(),
        )
        persisted = json.loads(
            (
                adapter.local_controller_root
                / "ssh_storage_preflight.json"
            ).read_text(encoding="utf-8")
        )

    assert receipt["classification"] == "INCOMPLETE"
    assert receipt["reason"] == "Kerberos TTL preflight failed"
    assert persisted == receipt


def test_readmission_rechecks_the_frozen_gpus_not_the_first_four_clean():
    planned = [_gpu(index) for index in range(4, 8)]
    plan = _plan(selected_gpus=planned)
    observed = [_gpu(0), *planned]
    kerberos_checks = []
    original = controller.query_remote_gpu_inventory
    controller.query_remote_gpu_inventory = lambda **kwargs: observed
    try:
        with tempfile.TemporaryDirectory() as directory:
            adapter = ProductionAdapter(
                run_tag=RUN_TAG,
                local_attempt_root=Path(directory) / "attempt",
                kerberos_query=lambda **kwargs: (
                    kerberos_checks.append(
                        kwargs["minimum_lifetime_seconds"]
                    )
                    or {"classification": "READY"}
                ),
            )
            receipt = adapter.strict_clean_admission(
                plan,
                {"classification": "PASS"},
            )
    finally:
        controller.query_remote_gpu_inventory = original

    assert [
        row["gpu_index"] for row in receipt["selected_gpus"]
    ] == [4, 5, 6, 7]
    assert kerberos_checks == [22_500]


def main_tests() -> None:
    tests = (
        test_plan_freezes_paths_model_and_four_clean_gpus,
        test_plan_rejects_unclean_or_duplicate_gpu_identity,
        test_plan_rejects_source_drift_and_unsafe_run_tag,
        test_run_attempt_enforces_the_frozen_operation_order,
        test_cleanup_always_runs_and_preserves_original_failure,
        test_cleanup_failure_overrides_an_otherwise_successful_run,
        test_preflight_rejects_existing_tag_before_launch,
        test_run_attempt_rejects_a_verdict_without_verifier_evidence,
        test_monitor_and_run_launches_immediately_after_local_admission,
        test_plan_only_performs_no_gpu_query_or_remote_operation,
        test_controller_supports_direct_script_execution,
        test_remote_upload_retries_transient_ssh_disconnect,
        test_local_verifier_executes_from_the_frozen_source_revision,
        test_remote_driver_never_reuses_a_dynamic_port_across_arms,
        test_kerberos_guard_covers_the_full_remote_command_window,
        test_failed_kerberos_preflight_is_persisted_as_incomplete,
        test_readmission_rechecks_the_frozen_gpus_not_the_first_four_clean,
    )
    for test in tests:
        test()
    print(f"{len(tests)} passed")


if __name__ == "__main__":
    main_tests()
