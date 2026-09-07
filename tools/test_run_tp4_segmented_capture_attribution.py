import hashlib
import json
from pathlib import Path
from pathlib import PurePosixPath
from types import SimpleNamespace

import pytest

import run_tp4_segmented_capture_attribution as controller


def _source(run_tag="phase-a1-r61"):
    return {
        "schema_version": controller.SOURCE_SCHEMA,
        "phase": "A1",
        "run_tag": run_tag,
        "source_revision": "1" * 40,
        "source_tree_sha256": "2" * 64,
        "worker_sha256": "3" * 64,
        "verifier_sha256": "4" * 64,
        "model_repository": controller.MODEL_REPOSITORY,
        "model_revision": controller.MODEL_REVISION,
    }


def _gpus():
    return [
        {
            "gpu_index": index,
            "gpu_uuid": f"GPU-{index}",
            "memory_used_mib": 0,
            "utilization_percent": 0,
            "compute_processes": [],
        }
        for index in range(4)
    ]


def _seed(run_tag="phase-a1-r61"):
    return {
        "run_tag": run_tag,
        "admission_mode": "strict_clean",
    }


def _verification(classification="REPAIR_CANDIDATE"):
    return {
        "schema_version": controller.VERIFICATION_SCHEMA,
        "phase": "A1",
        "classification": classification,
        "failed_gates": [],
        "source_revision": "1" * 40,
        "run_tag": "phase-a1-r61",
    }


def _canonical_bytes(value):
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode()


def test_plan_binds_phase_a1_source_workload_controls_and_hashes():
    plan = controller.build_plan(
        run_tag="phase-a1-r61",
        source_identity=_source(),
        selected_gpus=_gpus(),
        admission_mode="strict_clean",
    )

    assert plan["schema_version"] == controller.PLAN_SCHEMA
    assert plan["phase"] == "A1"
    assert plan["source_revision"] == "1" * 40
    assert plan["source_tree_sha256"] == "2" * 64
    assert plan["worker_sha256"] == "3" * 64
    assert plan["verifier_sha256"] == "4" * 64
    assert plan["model_repository"] == controller.MODEL_REPOSITORY
    assert plan["model_revision"] == controller.MODEL_REVISION
    assert plan["dtype"] == "bfloat16"
    assert plan["tensor_parallel_size"] == 4
    assert plan["batch_size"] == 8
    assert plan["prompt_length"] == 256
    assert plan["max_tokens"] == 2
    assert plan["model_length"] == 384
    assert plan["max_segment_ns"] == 1_800_000_000
    assert plan["max_lifecycle_ns"] == 4_500_000_000
    assert plan["max_added_memory_bytes_per_rank"] == 512 * 1024 * 1024
    assert tuple(row["control_id"] for row in plan["controls"]) == (
        controller.EXPECTED_CONTROL_IDS
    )
    assert plan["plan_sha256"] == controller.canonical_sha256(
        plan["controls"][:6]
    )


def test_plan_keeps_every_remote_path_below_large_mounted_root():
    plan = controller.build_plan(
        run_tag="phase-a1-r61",
        source_identity=_source(),
        selected_gpus=_gpus(),
        admission_mode="strict_clean",
    )
    approved = PurePosixPath(controller.REMOTE_BASE)
    attempt = PurePosixPath(plan["paths"]["attempt_root"])

    assert attempt.is_relative_to(approved)
    assert all(
        PurePosixPath(path).is_relative_to(attempt)
        for path in plan["paths"].values()
    )
    assert all(
        PurePosixPath(path).is_relative_to(attempt)
        for path in plan["environment"].values()
    )
    assert all(
        path not in {"/", "/tmp"}
        and not path.startswith("/tmp/")
        and not path.startswith("/root/")
        for path in (
            *plan["paths"].values(),
            *plan["environment"].values(),
        )
    )


@pytest.mark.parametrize("run_tag", ("../escape", "a..b", "/root", "", "a b"))
def test_plan_rejects_unsafe_run_tags(run_tag):
    with pytest.raises(ValueError, match="run tag"):
        controller.build_plan(
            run_tag=run_tag,
            source_identity=_source(run_tag),
            selected_gpus=_gpus(),
            admission_mode="strict_clean",
        )


def test_plan_rejects_non_strict_or_non_unique_gpu_admission():
    with pytest.raises(ValueError, match="strict_clean"):
        controller.build_plan(
            run_tag="phase-a1-r61",
            source_identity=_source(),
            selected_gpus=_gpus(),
            admission_mode="shared_capacity",
        )
    duplicate = _gpus()
    duplicate[3]["gpu_uuid"] = duplicate[2]["gpu_uuid"]
    with pytest.raises(ValueError, match="GPU"):
        controller.build_plan(
            run_tag="phase-a1-r61",
            source_identity=_source(),
            selected_gpus=duplicate,
            admission_mode="strict_clean",
        )
    occupied = _gpus()
    occupied[0]["compute_processes"] = [{"pid": 9}]
    with pytest.raises(ValueError, match="GPU"):
        controller.build_plan(
            run_tag="phase-a1-r61",
            source_identity=_source(),
            selected_gpus=occupied,
            admission_mode="strict_clean",
        )


@pytest.mark.parametrize(
    "mutation",
    (
        {"source_revision": "bad"},
        {"source_tree_sha256": "bad"},
        {"worker_sha256": "bad"},
        {"verifier_sha256": "bad"},
        {"phase": "A2"},
        {"model_revision": "bad"},
    ),
)
def test_plan_rejects_unbound_or_wrong_source_identity(mutation):
    source = _source()
    source.update(mutation)
    with pytest.raises(ValueError, match="source identity"):
        controller.build_plan(
            run_tag="phase-a1-r61",
            source_identity=source,
            selected_gpus=_gpus(),
            admission_mode="strict_clean",
        )


class _Adapter:
    def __init__(self, *, fail_at=None, classification="REPAIR_CANDIDATE"):
        self.fail_at = fail_at
        self.classification = classification
        self.events = []

    def _step(self, name, value):
        self.events.append(name)
        if self.fail_at == name:
            raise RuntimeError(f"{name} failed")
        return value

    def freeze_source(self, seed):
        return self._step("freeze_source", _source(seed["run_tag"]))

    def ssh_storage_preflight(self, seed, source):
        del source
        return self._step(
            "ssh_storage_preflight",
            {
                "classification": "PASS",
                "attempt_exists": False,
                "local_attempt_exists": False,
                "remote_root": controller.REMOTE_ROOT,
            },
        )

    def kerberos_ttl_guard(self, seed, preflight):
        del seed, preflight
        return self._step(
            "kerberos_ttl_guard",
            {
                "classification": "READY",
                "remaining_lifetime_seconds": 30_000,
                "minimum_required_lifetime_seconds": 22_500,
            },
        )

    def gpu_admission(self, seed, preflight):
        del seed, preflight
        return self._step(
            "gpu_admission",
            {
                "classification": "READY",
                "selected_gpus": _gpus(),
                "foreign_processes": [],
            },
        )

    def launch_once(self, plan, admission):
        del plan, admission
        return self._step("launch_once", {"owned_pids": [123]})

    def wait(self, plan, launch):
        del plan, launch
        return self._step("wait", {"exit_code": 0})

    def owned_cleanup(self, plan, launch):
        del plan, launch
        return self._step(
            "owned_cleanup",
            {
                "schema_version": controller.CLEANUP_SCHEMA,
                "run_tag": "phase-a1-r61",
                "classification": "CLEAN",
                "owned_children_remaining": [],
                "rank_rows": [
                    {
                        "rank": rank,
                        "exit_code": 0,
                        "process_group_destroyed": True,
                    }
                    for rank in range(4)
                ],
                "final_exact_tag_scans": [[], [], []],
            },
        )

    def download(self, plan, waited, cleanup):
        del plan, waited, cleanup
        return self._step(
            "download",
            {
                "downloaded": True,
                "bundle_root": "/local/final_bundle",
                "manifest_sha256": "5" * 64,
            },
        )

    def remote_verify(self, plan, downloaded):
        del plan, downloaded
        value = _verification(self.classification)
        return self._step(
            "remote_verify",
            {"result": value, "bytes": _canonical_bytes(value)},
        )

    def local_verify(self, plan, downloaded):
        del plan, downloaded
        value = _verification(self.classification)
        return self._step(
            "local_verify",
            {"result": value, "bytes": _canonical_bytes(value)},
        )

    def validate_verifier_identity(self, plan, remote, local):
        del plan
        return self._step(
            "validate_verifier_identity",
            controller.validate_verifier_identity(remote, local),
        )

    def write_post_verification_manifest(
        self,
        plan,
        downloaded,
        remote,
        local,
        verifier_identity,
        cleanup,
    ):
        del plan, downloaded, remote, local, verifier_identity, cleanup
        return self._step(
            "write_post_verification_manifest",
            {"schema_version": controller.POST_MANIFEST_SCHEMA},
        )

    def final_live_exact_tag_scan(self, plan):
        del plan
        return self._step("final_live_exact_tag_scan", [])


def test_controller_runs_the_exact_one_launch_lifecycle_order():
    adapter = _Adapter()
    result = controller.monitor_and_run(_seed(), adapter)

    assert result["classification"] == "REPAIR_CANDIDATE"
    assert adapter.events == [
        "freeze_source",
        "ssh_storage_preflight",
        "kerberos_ttl_guard",
        "gpu_admission",
        "launch_once",
        "wait",
        "owned_cleanup",
        "download",
        "remote_verify",
        "local_verify",
        "validate_verifier_identity",
        "write_post_verification_manifest",
        "final_live_exact_tag_scan",
    ]


def test_insufficient_kerberos_ttl_stops_before_gpu_or_launch():
    adapter = _Adapter(fail_at="kerberos_ttl_guard")
    with pytest.raises(RuntimeError, match="kerberos_ttl_guard failed"):
        controller.monitor_and_run(_seed(), adapter)
    assert adapter.events == [
        "freeze_source",
        "ssh_storage_preflight",
        "kerberos_ttl_guard",
    ]


def test_cleanup_runs_after_launch_failure_and_primary_error_wins():
    adapter = _Adapter(fail_at="wait")
    with pytest.raises(RuntimeError, match="wait failed"):
        controller.monitor_and_run(_seed(), adapter)
    assert adapter.events == [
        "freeze_source",
        "ssh_storage_preflight",
        "kerberos_ttl_guard",
        "gpu_admission",
        "launch_once",
        "wait",
        "owned_cleanup",
    ]

    class CleanupFailingAdapter(_Adapter):
        def owned_cleanup(self, plan, launch):
            self.events.append("owned_cleanup")
            raise RuntimeError("cleanup failed")

    cleanup_failing = CleanupFailingAdapter(fail_at="wait")
    with pytest.raises(RuntimeError, match="wait failed") as captured:
        controller.monitor_and_run(_seed(), cleanup_failing)
    assert isinstance(captured.value.__cause__, RuntimeError)
    assert str(captured.value.__cause__) == "cleanup failed"


def test_worker_disconnect_never_relaunches():
    adapter = _Adapter(fail_at="wait")
    with pytest.raises(RuntimeError, match="wait failed"):
        controller.monitor_and_run(_seed(), adapter)
    assert adapter.events.count("launch_once") == 1


def test_verifier_outputs_must_be_byte_identical_and_bound():
    value = _verification()
    matching = {"result": value, "bytes": _canonical_bytes(value)}
    identity = controller.validate_verifier_identity(matching, matching)

    assert identity["byte_identical"] is True
    assert identity["sha256"] == hashlib.sha256(
        _canonical_bytes(value)
    ).hexdigest()

    altered = {
        "result": value,
        "bytes": json.dumps(value, sort_keys=True).encode(),
    }
    with pytest.raises(RuntimeError, match="byte-identical"):
        controller.validate_verifier_identity(matching, altered)


def test_phase_a1_rejects_invalid_or_incomplete_terminal_states():
    for classification in ("GO_SEGMENTED_REPAIR", "INCOMPLETE"):
        adapter = _Adapter(classification=classification)
        with pytest.raises(RuntimeError, match=classification):
            controller.monitor_and_run(_seed(), adapter)
        assert adapter.events[-1] == "final_live_exact_tag_scan"


def _write_producer_bundle(root: Path):
    root.mkdir()
    for name in controller.PRODUCER_ARTIFACT_NAMES:
        payload = (
            '{"row":1}\n'
            if name.endswith(".jsonl")
            else '{"value":1}\n'
        )
        (root / name).write_text(payload, encoding="utf-8")


@pytest.mark.parametrize(
    "mutated_name",
    (
        "phase_rows.jsonl",
        "scratch_rows.jsonl",
        "process_receipts.json",
        "diagnosis.json",
    ),
)
def test_pre_verification_manifest_binds_every_producer_artifact(
    tmp_path,
    mutated_name,
):
    root = tmp_path / "bundle"
    _write_producer_bundle(root)

    manifest = controller.write_pre_verification_manifest(root)

    assert manifest["schema_version"] == controller.MANIFEST_SCHEMA
    assert set(manifest["artifacts"]) == set(
        controller.PRODUCER_ARTIFACT_NAMES
    )
    assert "remote_independent_verification.json" not in (
        manifest["artifacts"]
    )
    controller.validate_pre_verification_manifest(root)

    (root / mutated_name).write_text(
        '{"row":2}\n',
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="manifest hash"):
        controller.validate_pre_verification_manifest(root)


def test_post_manifest_binds_pre_manifest_verifiers_cleanup_and_final_scan(
    tmp_path,
):
    root = tmp_path / "bundle"
    _write_producer_bundle(root)
    controller.write_pre_verification_manifest(root)
    remote = {"result": _verification(), "bytes": _canonical_bytes(_verification())}
    local = {"result": _verification(), "bytes": _canonical_bytes(_verification())}
    cleanup = {
        "schema_version": controller.CLEANUP_SCHEMA,
        "run_tag": "phase-a1-r61",
        "classification": "CLEAN",
        "final_exact_tag_scans": [[], [], []],
    }
    cleanup_bytes = _canonical_bytes(cleanup)

    manifest = controller.build_post_verification_manifest(
        bundle_root=root,
        remote_verification=remote,
        local_verification=local,
        cleanup_bytes=cleanup_bytes,
        final_live_exact_tag_scan=[],
    )

    assert manifest["verifier_byte_identical"] is True
    assert manifest["artifacts"]["manifest.json"] == hashlib.sha256(
        (root / "manifest.json").read_bytes()
    ).hexdigest()
    assert manifest["artifacts"]["cleanup_receipt.json"] == hashlib.sha256(
        cleanup_bytes
    ).hexdigest()
    assert manifest["final_live_exact_tag_scan_sha256"] == (
        controller.canonical_sha256([])
    )
    controller.validate_post_verification_manifest(
        manifest,
        bundle_root=root,
        remote_verification=remote,
        local_verification=local,
        cleanup_bytes=cleanup_bytes,
        final_live_exact_tag_scan=[],
    )

    changed = dict(remote)
    changed["bytes"] = remote["bytes"] + b" "
    with pytest.raises(RuntimeError, match="post-verification"):
        controller.validate_post_verification_manifest(
            manifest,
            bundle_root=root,
            remote_verification=changed,
            local_verification=local,
            cleanup_bytes=cleanup_bytes,
            final_live_exact_tag_scan=[],
        )


def test_exact_tag_match_does_not_use_substrings():
    assert controller._matches_exact_tag_identity(
        b"python worker.py\x00",
        b"TINYLLMFORGE_RUN_TAG=phase-a1-r61\x00",
        run_tag="phase-a1-r61",
        attempt_root="/remote/phase-a1-r61",
    )
    assert not controller._matches_exact_tag_identity(
        b"python worker.py\x00",
        b"TINYLLMFORGE_RUN_TAG=phase-a1-r61-extra\x00",
        run_tag="phase-a1-r61",
        attempt_root="/remote/phase-a1-r61",
    )
    assert controller._matches_exact_tag_identity(
        b"python\x00/remote/phase-a1-r61/source/worker.py\x00",
        b"",
        run_tag="phase-a1-r61",
        attempt_root="/remote/phase-a1-r61",
    )
    assert not controller._matches_exact_tag_identity(
        b"python\x00/remote/phase-a1-r61-extra/source/worker.py\x00",
        b"",
        run_tag="phase-a1-r61",
        attempt_root="/remote/phase-a1-r61",
    )


def test_freeze_source_rejects_an_existing_local_attempt(tmp_path, monkeypatch):
    attempt = tmp_path / "existing"
    attempt.mkdir()
    (attempt / "preserve").write_text("yes", encoding="utf-8")
    adapter = controller.ProductionAdapter(
        run_tag="phase-a1-r61",
        local_attempt_root=attempt,
    )
    monkeypatch.setattr(
        controller,
        "_capture_source_identity",
        lambda run_tag: _source(run_tag),
    )

    with pytest.raises(ValueError, match="local attempt"):
        adapter.freeze_source(_seed())

    assert (attempt / "preserve").read_text(encoding="utf-8") == "yes"


@pytest.mark.parametrize(
    ("local_free_bytes", "remote_free_bytes"),
    (
        (0, 8 * 1024**3),
        (1 * 1024**3, 0),
    ),
)
def test_storage_preflight_rejects_insufficient_artifact_space(
    tmp_path,
    monkeypatch,
    local_free_bytes,
    remote_free_bytes,
):
    adapter = object.__new__(controller.ProductionAdapter)
    adapter.local_attempt_root = tmp_path / "attempt"
    adapter.local_controller_root = (
        adapter.local_attempt_root / "controller"
    )
    adapter.local_controller_root.mkdir(parents=True)
    monkeypatch.setattr(
        controller.shutil,
        "disk_usage",
        lambda _path: SimpleNamespace(free=local_free_bytes),
    )
    adapter._remote = lambda _argv: SimpleNamespace(
        stdout=json.dumps({
            "base_ready": True,
            "remote_root_safe": True,
            "attempt_exists": False,
            "model_ready": True,
            "model_revision_matches": True,
            "remote_free_bytes": remote_free_bytes,
            "stale_exact_tag_processes": [],
            "text_profile": {
                "num_hidden_layers": 64,
                "hidden_size": 5120,
                "vocab_size": 248320,
                "dtype": "bfloat16",
            },
        })
    )

    with pytest.raises(ValueError, match="artifact space"):
        adapter.ssh_storage_preflight(_seed(), _source())


def test_storage_preflight_rejects_stale_exact_tag_process(
    tmp_path,
    monkeypatch,
):
    adapter = object.__new__(controller.ProductionAdapter)
    adapter.local_attempt_root = tmp_path / "attempt"
    adapter.local_controller_root = (
        adapter.local_attempt_root / "controller"
    )
    adapter.local_controller_root.mkdir(parents=True)
    monkeypatch.setattr(
        controller.shutil,
        "disk_usage",
        lambda _path: SimpleNamespace(free=1 * 1024**3),
    )
    adapter._remote = lambda _argv: SimpleNamespace(
        stdout=json.dumps({
            "base_ready": True,
            "remote_root_safe": True,
            "attempt_exists": False,
            "model_ready": True,
            "model_revision_matches": True,
            "remote_free_bytes": 8 * 1024**3,
            "stale_exact_tag_processes": [{"pid": 123}],
            "text_profile": {
                "num_hidden_layers": 64,
                "hidden_size": 5120,
                "vocab_size": 248320,
                "dtype": "bfloat16",
            },
        })
    )

    with pytest.raises(ValueError, match="stale exact-tag"):
        adapter.ssh_storage_preflight(_seed(), _source())


def test_diagnosis_is_derived_from_rows_not_hard_coded(tmp_path):
    scratch_rows = [
        {
            "rank": rank,
            "checkpoint": checkpoint,
            "key_diff": {"equal_to_s0": checkpoint != "S3"},
            "value_diff": {"equal_to_s0": checkpoint != "S3"},
        }
        for rank in range(4)
        for checkpoint in ("S0", "S1", "S2", "S3", "S4", "S5", "S6", "S7")
    ]
    phase_rows = [
        {
            "rank": rank,
            "control_id": "stitched_p4_repeat_0",
            "snapshot_and_prepare_ns": 1,
            "graph_object_create_ns": 2,
            "capture_context_enter_ns": 3,
            "capture_body_ns": 100,
            "capture_context_exit_and_instantiate_ns": 5,
            "post_capture_synchronize_ns": 6,
            "post_capture_restore_ns": 7,
            "graph_reset_ns": 8,
            "segment_total_ns": 150,
            "program_lifecycle_ns": 600,
        }
        for rank in range(4)
    ]

    diagnosis = controller.derive_diagnosis(
        phase_rows=phase_rows,
        scratch_rows=scratch_rows,
    )

    assert diagnosis["first_scratch_divergence"] == "S3"
    assert diagnosis["slow_capture_phase"] == "capture_body_ns"
    assert diagnosis["root_cause_kind"] == "capture_scratch_write"
    assert diagnosis["repair_count"] == 1
    assert diagnosis["projected_max_segment_ns"] == 150
    assert diagnosis["projected_lifecycle_ns"] == 600


def test_kerberos_guard_is_separate_and_persists_failure(tmp_path):
    adapter = object.__new__(controller.ProductionAdapter)
    adapter.local_controller_root = tmp_path
    adapter.command_timeout_s = 21_600
    adapter.kerberos_query = lambda **kwargs: {
        "classification": "BLOCKED_KERBEROS_TTL",
        "remaining_lifetime_seconds": 100,
        "minimum_required_lifetime_seconds": kwargs[
            "minimum_lifetime_seconds"
        ],
    }

    receipt = adapter.kerberos_ttl_guard(_seed(), {})

    assert receipt["classification"] == "INCOMPLETE"
    assert receipt["reason"] == "Kerberos TTL preflight failed"
    assert json.loads(
        (tmp_path / "kerberos_ttl_guard.json").read_text()
    ) == receipt


def test_launch_rechecks_kerberos_after_gpu_wait(
    tmp_path,
    monkeypatch,
):
    adapter = controller.ProductionAdapter(
        run_tag="phase-a1-r61",
        local_attempt_root=tmp_path / "attempt",
        kerberos_query=lambda **kwargs: {
            "classification": "BLOCKED_KERBEROS_TTL",
            "remaining_lifetime_seconds": 100,
            "minimum_required_lifetime_seconds": kwargs[
                "minimum_lifetime_seconds"
            ],
        },
    )
    adapter.local_controller_root.mkdir(parents=True)
    adapter._source = _source()
    adapter._admission = {"selected_gpus": _gpus()}
    monkeypatch.setattr(
        controller,
        "query_remote_gpu_inventory",
        lambda **_kwargs: pytest.fail(
            "GPU inventory must not run after TTL rejection"
        ),
    )
    plan = controller.build_plan(
        run_tag="phase-a1-r61",
        source_identity=_source(),
        selected_gpus=_gpus(),
        admission_mode="strict_clean",
    )

    with pytest.raises(RuntimeError, match="Kerberos TTL"):
        adapter.launch_once(
            plan,
            {"selected_gpus": _gpus()},
        )

    assert adapter._process is None


def test_owned_cleanup_reaps_only_exact_tag_rows(tmp_path):
    adapter = object.__new__(controller.ProductionAdapter)
    adapter.local_controller_root = tmp_path
    adapter._cleanup = None
    adapter._process = SimpleNamespace(poll=lambda: 0)
    scans = iter(
        (
            [
                {
                    "pid": 123,
                    "matched_cmdline": True,
                    "matched_environment": True,
                }
            ],
            [],
            [],
            [],
        )
    )
    adapter._scan_exact_tag = lambda plan: next(scans)
    reaped = []
    adapter._reap_exact_tag = (
        lambda plan, rows: reaped.append(rows)
        or {
            "requested_pids": [123],
            "terminated_pids": [123],
            "killed_pids": [],
            "remaining_pids": [],
        }
    )
    plan = controller.build_plan(
        run_tag="phase-a1-r61",
        source_identity=_source(),
        selected_gpus=_gpus(),
        admission_mode="strict_clean",
    )

    cleanup = adapter.owned_cleanup(plan, {"owned_pids": [999]})

    assert reaped == [[{
        "pid": 123,
        "matched_cmdline": True,
        "matched_environment": True,
    }]]
    assert cleanup["classification"] == "CLEAN"
    assert cleanup["final_exact_tag_scans"] == [[], [], []]


def test_main_wires_the_one_launch_controller(tmp_path, capsys):
    adapter = _Adapter()
    exit_code = controller.main(
        [
            "--run-tag",
            "phase-a1-r61",
            "--admission-mode",
            "strict_clean",
            "--local-attempt-root",
            str(tmp_path / "attempt"),
        ],
        adapter_factory=lambda **kwargs: adapter,
    )

    assert exit_code == 0
    assert json.loads(capsys.readouterr().out)["classification"] == (
        "REPAIR_CANDIDATE"
    )
    assert adapter.events.count("launch_once") == 1


def test_prelaunch_failure_is_persisted_without_launch():
    class PersistingAdapter(_Adapter):
        def __init__(self):
            super().__init__(fail_at="kerberos_ttl_guard")
            self.incomplete = None

        def persist_incomplete(self, seed, error):
            self.incomplete = {
                "run_tag": seed["run_tag"],
                "classification": "INCOMPLETE",
                "error": str(error),
            }

    adapter = PersistingAdapter()
    with pytest.raises(RuntimeError, match="kerberos_ttl_guard failed"):
        controller.monitor_and_run(_seed(), adapter)

    assert "gpu_admission" not in adapter.events
    assert "launch_once" not in adapter.events
    assert adapter.incomplete == {
        "run_tag": "phase-a1-r61",
        "classification": "INCOMPLETE",
        "error": "kerberos_ttl_guard failed",
    }


def test_production_incomplete_receipt_is_atomic_and_bounded(tmp_path):
    adapter = object.__new__(controller.ProductionAdapter)
    adapter.local_controller_root = tmp_path / "controller"
    adapter._process = None

    receipt = adapter.persist_incomplete(
        _seed(),
        RuntimeError("blocked"),
    )

    assert receipt["classification"] == "INCOMPLETE"
    assert receipt["launch_started"] is False
    assert json.loads(
        (
            adapter.local_controller_root / "terminal_result.json"
        ).read_text()
    ) == receipt


def test_remote_verifier_preserves_structured_incomplete_output(
    tmp_path,
    monkeypatch,
):
    value = _verification("INCOMPLETE")
    stdout = _canonical_bytes(value).decode()
    monkeypatch.setattr(
        controller,
        "run_remote_argv",
        lambda **kwargs: SimpleNamespace(
            returncode=1,
            stdout=stdout,
            stderr="",
        ),
    )
    adapter = controller.ProductionAdapter(
        run_tag="phase-a1-r61",
        local_attempt_root=tmp_path / "attempt",
    )
    adapter.local_controller_root.mkdir(parents=True)
    plan = controller.build_plan(
        run_tag="phase-a1-r61",
        source_identity=_source(),
        selected_gpus=_gpus(),
        admission_mode="strict_clean",
    )

    result = adapter.remote_verify(
        plan,
        {"downloaded": True},
    )

    assert result["result"]["classification"] == "INCOMPLETE"
    assert result["bytes"] == stdout.encode()


def test_local_verifier_archive_includes_frozen_worker(tmp_path):
    calls = []
    verification = _verification()

    def command_runner(argv, **kwargs):
        calls.append(list(argv))
        if argv[:3] == ["git", "-C", str(Path(
            controller.__file__
        ).resolve().parents[1])]:
            return SimpleNamespace(
                returncode=0,
                stdout=b"archive",
                stderr=b"",
            )
        if argv[:3] == ["tar", "-xf", "-"]:
            return SimpleNamespace(returncode=0, stdout=b"", stderr=b"")
        return SimpleNamespace(
            returncode=0,
            stdout=_canonical_bytes(verification),
            stderr=b"",
        )

    adapter = controller.ProductionAdapter(
        run_tag="phase-a1-r61",
        local_attempt_root=tmp_path / "attempt",
        local_command_runner=command_runner,
    )
    adapter.local_controller_root.mkdir(parents=True)
    plan = controller.build_plan(
        run_tag="phase-a1-r61",
        source_identity=_source(),
        selected_gpus=_gpus(),
        admission_mode="strict_clean",
    )

    adapter.local_verify(plan, {"downloaded": True})

    git_archive = calls[0]
    assert (
        "tools/tp4_segmented_capture_attribution_worker.py"
        in git_archive
    )
