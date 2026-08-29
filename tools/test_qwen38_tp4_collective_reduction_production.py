from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import subprocess
import sys
import tarfile
from types import SimpleNamespace

import pytest

from tools.qwen38_tp4_collective_reduction_production import (
    CollectiveReductionProductionAdapter,
    REMOTE_PYTHON,
    build_committed_source_archive,
    build_gpu_topology_evidence,
    build_stage_payload,
    build_supervisor_argv,
    classify_attempt_state,
    create_production_adapter,
    extract_bounded_archive,
    fetch_remote_bundle,
    launch_remote_supervisor,
    load_remote_json,
    query_remote_attempt_state,
    query_remote_postprocess_state,
    require_byte_identical_verification,
    run_remote_json_command,
    run_remote_bytes,
    stage_remote_attempt,
    validate_launch_inventory,
    validate_postprocess_state,
)
from tools.run_qwen38_tp4_collective_reduction import (
    APPROVED_REMOTE_ROOT,
    build_attempt_plan,
    build_source_identity,
)
from tools.assemble_qwen38_tp4_collective_reduction import (
    PRODUCER_ARTIFACTS,
)


ATTEMPT = "20260827-qwen38-tp4-collective-reduction-r1"
SOURCE_REVISION = "a" * 40
MODEL_REVISION = "b" * 40


def _gpu(index):
    return {
        "gpu_index": index,
        "gpu_uuid": f"GPU-{index}",
        "memory_used_mib": 3,
        "utilization_percent": 0,
        "compute_processes": [],
    }


def _plan():
    return build_attempt_plan(
        attempt_tag=ATTEMPT,
        source_revision=SOURCE_REVISION,
        model_revision=MODEL_REVISION,
        selected_gpus=[_gpu(index) for index in range(4)],
        remote_path_state={
            "attempt_exists": False,
            "attempt_parent_is_symlink": False,
            "remote_root_is_symlink": False,
        },
    )


def _archive_bytes(files):
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w:") as archive:
        for name, payload in files.items():
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            archive.addfile(info, io.BytesIO(payload))
    return stream.getvalue()


def test_committed_archive_must_match_frozen_source_inventory(tmp_path):
    files = {
        "tinyvllm/config.py": b"config\n",
        "tools/worker.py": b"worker\n",
    }
    identity = build_source_identity(
        attempt=ATTEMPT,
        source_revision=SOURCE_REVISION,
        source_files={
            name: hashlib.sha256(payload).hexdigest()
            for name, payload in files.items()
        },
    )
    archive = _archive_bytes(files)

    result = build_committed_source_archive(
        repo_root=tmp_path,
        source_revision=SOURCE_REVISION,
        source_identity=identity,
        command_runner=lambda *args, **kwargs: SimpleNamespace(
            returncode=0,
            stdout=archive,
            stderr=b"",
        ),
    )

    assert result == archive

    mismatched = dict(identity)
    mismatched["source_files"] = dict(identity["source_files"])
    mismatched["source_files"]["tools/worker.py"] = "0" * 64
    with pytest.raises(ValueError, match="source archive inventory"):
        build_committed_source_archive(
            repo_root=tmp_path,
            source_revision=SOURCE_REVISION,
            source_identity=mismatched,
            command_runner=lambda *args, **kwargs: SimpleNamespace(
                returncode=0,
                stdout=archive,
                stderr=b"",
            ),
        )


@pytest.mark.parametrize(
    ("state", "expected"),
    (
        (
            {
                "attempt_exists": False,
                "source_identity": None,
                "launch": None,
                "supervisor_receipt": None,
                "live_exact_tag_pids": [],
            },
            "CREATE",
        ),
        (
            {
                "attempt_exists": True,
                "source_identity": {
                    "attempt": ATTEMPT,
                    "source_revision": SOURCE_REVISION,
                    "source_tree_sha256": "c" * 64,
                },
                "launch": None,
                "supervisor_receipt": None,
                "live_exact_tag_pids": [],
            },
            "LAUNCH",
        ),
        (
            {
                "attempt_exists": True,
                "source_identity": {
                    "attempt": ATTEMPT,
                    "source_revision": SOURCE_REVISION,
                    "source_tree_sha256": "c" * 64,
                },
                "launch": {"pid": 101, "source_revision": SOURCE_REVISION},
                "supervisor_receipt": None,
                "live_exact_tag_pids": [101],
            },
            "MONITOR",
        ),
        (
            {
                "attempt_exists": True,
                "source_identity": {
                    "attempt": ATTEMPT,
                    "source_revision": SOURCE_REVISION,
                    "source_tree_sha256": "c" * 64,
                },
                "launch": {"pid": 101, "source_revision": SOURCE_REVISION},
                "supervisor_receipt": {
                    "classification": "PASS",
                    "source_revision": SOURCE_REVISION,
                },
                "live_exact_tag_pids": [],
            },
            "POSTPROCESS",
        ),
    ),
)
def test_attempt_state_has_only_safe_create_launch_monitor_postprocess(
    state,
    expected,
):
    identity = {
        "attempt": ATTEMPT,
        "source_revision": SOURCE_REVISION,
        "source_tree_sha256": "c" * 64,
    }

    assert classify_attempt_state(
        state,
        source_identity=identity,
    ) == expected


def test_attempt_state_rejects_source_drift_and_orphaned_execution():
    identity = {
        "attempt": ATTEMPT,
        "source_revision": SOURCE_REVISION,
        "source_tree_sha256": "c" * 64,
    }
    drifted = {
        "attempt_exists": True,
        "source_identity": {
            **identity,
            "source_revision": "d" * 40,
        },
        "launch": None,
        "supervisor_receipt": None,
        "live_exact_tag_pids": [],
    }
    orphaned = {
        "attempt_exists": True,
        "source_identity": identity,
        "launch": {"pid": 101, "source_revision": SOURCE_REVISION},
        "supervisor_receipt": None,
        "live_exact_tag_pids": [],
    }

    with pytest.raises(ValueError, match="source identity"):
        classify_attempt_state(drifted, source_identity=identity)
    with pytest.raises(RuntimeError, match="orphaned"):
        classify_attempt_state(orphaned, source_identity=identity)


def test_supervisor_argv_is_bounded_and_contains_no_signal_command():
    plan = _plan()

    argv = build_supervisor_argv(plan, remote_python=REMOTE_PYTHON)

    assert argv[0] == REMOTE_PYTHON
    assert argv[1].endswith(
        "/tools/qwen38_tp4_collective_reduction_supervisor.py"
    )
    assert argv[argv.index("--attempt") + 1] == ATTEMPT
    assert argv[argv.index("--source-revision") + 1] == SOURCE_REVISION
    assert json.loads(
        argv[argv.index("--selected-gpus-json") + 1]
    ) == plan["selected_gpus"]
    encoded = " ".join(argv).lower()
    assert all(
        forbidden not in encoded
        for forbidden in (" kill ", "pkill", "killall", "signal")
    )
    for value in argv:
        if value.startswith(APPROVED_REMOTE_ROOT):
            assert PurePosixPath(value).is_relative_to(
                PurePosixPath(APPROVED_REMOTE_ROOT)
            )


def test_gpu_topology_evidence_binds_rank_order_to_planned_devices():
    plan = _plan()
    topology = {
        "gpu_rows": [
            {
                "gpu_index": index,
                "gpu_uuid": f"GPU-{index}",
                "pci_bus_id": f"00000000:{index + 1:02x}:00.0",
            }
            for index in reversed(range(4))
        ],
        "interconnect_matrix": "GPU0 GPU1 GPU2 GPU3",
    }

    evidence = build_gpu_topology_evidence(plan, topology)

    assert evidence["schema_version"] == (
        "qwen38.tp4-collective-reduction-topology.v1"
    )
    assert evidence["rank_mapping"] == [
        {
            "rank": rank,
            "gpu_index": rank,
            "gpu_uuid": f"GPU-{rank}",
            "pci_bus_id": f"00000000:{rank + 1:02x}:00.0",
        }
        for rank in range(4)
    ]

    topology["gpu_rows"][0]["gpu_uuid"] = "GPU-drift"
    with pytest.raises(ValueError, match="topology"):
        build_gpu_topology_evidence(plan, topology)


def test_launch_inventory_rechecks_strict_clean_frozen_gpus():
    plan = _plan()

    assert validate_launch_inventory(
        plan,
        [_gpu(index) for index in range(4)],
    ) == plan["selected_gpus"]

    for observed in (
        [_gpu(0), _gpu(1), _gpu(2), _gpu(3) | {"memory_used_mib": 1025}],
        [_gpu(0), _gpu(1), _gpu(2), _gpu(3) | {"utilization_percent": 6}],
        [
            _gpu(0),
            _gpu(1),
            _gpu(2),
            _gpu(3) | {
                "compute_processes": [{
                    "pid": 999,
                    "process_name": "foreign",
                    "used_memory_mib": 1,
                }],
            },
        ],
        [
            _gpu(0),
            _gpu(1),
            _gpu(2),
            _gpu(3) | {"gpu_uuid": "GPU-drift"},
        ],
    ):
        with pytest.raises(ValueError, match="strict-clean"):
            validate_launch_inventory(plan, observed)


def test_bounded_download_extracts_exact_regular_file_inventory(tmp_path):
    archive = _archive_bytes({
        "final_bundle/classification.json": b'{"classification":"GO"}\n',
        "controller/remote-independent-verification.json": b"{}\n",
    })
    destination = tmp_path / "download"

    inventory = extract_bounded_archive(
        archive,
        destination=destination,
        expected_files={
            "final_bundle/classification.json",
            "controller/remote-independent-verification.json",
        },
    )

    assert set(inventory) == {
        "final_bundle/classification.json",
        "controller/remote-independent-verification.json",
    }
    assert (
        destination / "final_bundle/classification.json"
    ).is_file()


def test_bounded_download_rejects_path_traversal_and_extra_files(tmp_path):
    traversal = _archive_bytes({"../escape": b"x"})
    with pytest.raises(ValueError, match="unsafe"):
        extract_bounded_archive(
            traversal,
            destination=tmp_path / "traversal",
            expected_files={"final_bundle/classification.json"},
        )

    extra = _archive_bytes({
        "final_bundle/classification.json": b"{}\n",
        "raw/model.bin": b"x",
    })
    with pytest.raises(ValueError, match="inventory"):
        extract_bounded_archive(
            extra,
            destination=tmp_path / "extra",
            expected_files={"final_bundle/classification.json"},
        )


def test_bounded_download_is_idempotent_only_for_identical_bytes(tmp_path):
    destination = tmp_path / "download"
    expected = {"final_bundle/classification.json"}
    archive = _archive_bytes({
        "final_bundle/classification.json": b'{"value":1}\n',
    })

    first = extract_bounded_archive(
        archive,
        destination=destination,
        expected_files=expected,
    )
    second = extract_bounded_archive(
        archive,
        destination=destination,
        expected_files=expected,
    )

    assert second == first
    changed = _archive_bytes({
        "final_bundle/classification.json": b'{"value":2}\n',
    })
    with pytest.raises(ValueError, match="existing download"):
        extract_bounded_archive(
            changed,
            destination=destination,
            expected_files=expected,
        )


def test_remote_and_local_verification_must_be_byte_identical():
    payload = b'{"classification":"GO","status":"PASS"}\n'

    parsed = require_byte_identical_verification(payload, payload)

    assert parsed["status"] == "PASS"
    with pytest.raises(RuntimeError, match="byte-identical"):
        require_byte_identical_verification(
            payload,
            b'{"status":"PASS","classification":"GO"}\n',
        )


def test_production_adapter_runs_one_resumable_supervisor_and_full_chain(
    tmp_path,
):
    plan = _plan()
    source_identity = {
        "schema_version": (
            "qwen38.tp4-collective-reduction-source.v1"
        ),
        "attempt": ATTEMPT,
        "source_revision": SOURCE_REVISION,
        "source_tree_sha256": "c" * 64,
        "source_files": {"tools/worker.py": "d" * 64},
        "source_archive_paths": ["tinyvllm", "tools"],
    }
    model_manifest = {
        "schema_version": "tinyllmforge.qwen38-model-manifest.v1",
        "repository": "Qwen/Qwen3.8-27B",
        "revision": MODEL_REVISION,
        "text_profile": {
            "num_hidden_layers": 64,
            "hidden_size": 5120,
            "vocab_size": 248320,
            "dtype": "bfloat16",
        },
    }
    topology = {
        "gpu_rows": [
            {
                "gpu_index": index,
                "gpu_uuid": f"GPU-{index}",
                "pci_bus_id": f"00000000:{index + 1:02x}:00.0",
            }
            for index in range(4)
        ],
        "interconnect_matrix": "GPU0 GPU1 GPU2 GPU3",
    }
    identity_summary = {
        key: source_identity[key]
        for key in ("attempt", "source_revision", "source_tree_sha256")
    }
    states = iter((
        {
            "attempt_exists": False,
            "source_identity": None,
            "launch": None,
            "supervisor_receipt": None,
            "live_exact_tag_pids": [],
        },
        {
            "attempt_exists": True,
            "source_identity": identity_summary,
            "launch": None,
            "supervisor_receipt": None,
            "live_exact_tag_pids": [],
        },
        {
            "attempt_exists": True,
            "source_identity": identity_summary,
            "launch": {"pid": 101, "source_revision": SOURCE_REVISION},
            "supervisor_receipt": None,
            "live_exact_tag_pids": [101],
        },
        {
            "attempt_exists": True,
            "source_identity": identity_summary,
            "launch": {"pid": 101, "source_revision": SOURCE_REVISION},
            "supervisor_receipt": {
                "classification": "PASS",
                "source_revision": SOURCE_REVISION,
            },
            "live_exact_tag_pids": [],
        },
    ))
    events = []
    worker = {
        "classification": "PASS",
        "attempt": ATTEMPT,
        "source_revision": SOURCE_REVISION,
        "selected_budget": None,
        "owned_pids": [101],
        "cases": [],
        "phase_cleanups": [],
    }
    remote_verification = (
        b'{"producer_classification":"INCONCLUSIVE_PROFILER_OVERHEAD",'
        b'"reconstructed_classification":"INCONCLUSIVE_PROFILER_OVERHEAD",'
        b'"status":"PASS"}\n'
    )
    bundle_files = {
        f"final_bundle/{name}": (
            remote_verification
            if name == "independent_verification.json"
            else b"{}\n"
        )
        for name in (
            *PRODUCER_ARTIFACTS,
            "independent_verification.json",
        )
    }
    cleanup = {
        "schema_version": (
            "qwen38.tp4-collective-reduction-cleanup.v1"
        ),
        "complete": True,
        "process_group_destroyed": True,
        "owned_children_remaining": [],
        "exact_tag_scans": [[], [], []],
    }
    bundle_files["final_bundle/cleanup.json"] = (
        json.dumps(cleanup, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode()
    manifest = {"schema_version": "manifest", "artifacts": {}}
    bundle_files["final_bundle/manifest.sha256"] = (
        json.dumps(manifest, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode()

    def local_verifier(bundle_root):
        events.append("local_verify")
        (bundle_root / "independent_verification.json").write_bytes(
            remote_verification
        )
        return json.loads(remote_verification)

    adapter = CollectiveReductionProductionAdapter(
        plan=plan,
        source_identity=source_identity,
        model_manifest=model_manifest,
        repo_root=tmp_path,
        local_attempt_root=tmp_path / "local-attempt",
        topology_query=lambda: topology,
        archive_builder=lambda **_kwargs: b"archive",
        attempt_state_query=lambda: next(states),
        attempt_stager=lambda **_kwargs: events.append("stage") or {
            "classification": "STAGED",
        },
        supervisor_launcher=lambda _argv: events.append("launch") or {
            "pid": 101,
            "source_revision": SOURCE_REVISION,
        },
        launch_guard=lambda: events.append("guard") or {
            "classification": "READY",
        },
        remote_json_loader=lambda path: (
            worker
            if path.endswith("/worker.json")
            else cleanup
        ),
        postprocess_state_query=lambda: {
            "producer": None,
            "verification": None,
        },
        remote_command_runner=lambda argv: (
            events.append(Path(argv[1]).name)
            or (
                {
                    "classification": (
                        "INCONCLUSIVE_PROFILER_OVERHEAD"
                    )
                }
                if "assemble_" in argv[1]
                else json.loads(remote_verification)
            )
        ),
        bundle_fetcher=lambda _plan, _names: _archive_bytes(bundle_files),
        local_verifier_runner=local_verifier,
        sleep=lambda _seconds: events.append("poll"),
    )

    assert adapter.worker_runner(plan) == worker
    producer = adapter.assembler(plan, worker)
    remote = adapter.remote_verifier(plan)
    download = adapter.downloader(plan)
    local = adapter.local_verifier(plan)
    cleanup_result = adapter.cleanup_validator(plan, worker)

    assert producer["classification"] == (
        "INCONCLUSIVE_PROFILER_OVERHEAD"
    )
    assert remote["classification"] == (
        "INCONCLUSIVE_PROFILER_OVERHEAD"
    )
    assert download["downloaded"] is True
    assert local["classification"] == (
        "INCONCLUSIVE_PROFILER_OVERHEAD"
    )
    assert cleanup_result["complete"] is True
    assert events == [
        "stage",
        "guard",
        "launch",
        "poll",
        "assemble_qwen38_tp4_collective_reduction.py",
        "verify_qwen38_tp4_collective_reduction.py",
        "local_verify",
    ]
    controller = tmp_path / "local-attempt" / "controller"
    assert (
        controller / "producer_result.json"
    ).is_file()
    assert (
        controller / "remote-independent-verification.json"
    ).read_bytes() == remote_verification
    assert (
        controller / "local-independent-verification.json"
    ).read_bytes() == remote_verification
    assert (
        controller / "remote-post-verification-manifest.json"
    ).is_file()


def test_adapter_resume_reuses_completed_remote_postprocessing(tmp_path):
    plan = _plan()
    identity = {
        "attempt": ATTEMPT,
        "source_revision": SOURCE_REVISION,
        "source_tree_sha256": "c" * 64,
    }
    verification = {
        "status": "PASS",
        "producer_classification": "NO_GO_NO_REDUCIBLE_COLLECTIVE",
        "reconstructed_classification": (
            "NO_GO_NO_REDUCIBLE_COLLECTIVE"
        ),
    }
    remote_commands = []
    adapter = CollectiveReductionProductionAdapter(
        plan=plan,
        source_identity=identity,
        model_manifest={"revision": MODEL_REVISION},
        repo_root=tmp_path,
        local_attempt_root=tmp_path / "attempt",
        topology_query=lambda: pytest.fail("must not query topology"),
        launch_guard=lambda: pytest.fail("must not launch"),
        archive_builder=lambda **_kwargs: pytest.fail("must not archive"),
        attempt_state_query=lambda: {
            "attempt_exists": True,
            "source_identity": identity,
            "launch": {"pid": 101, "source_revision": SOURCE_REVISION},
            "supervisor_receipt": {
                "classification": "PASS",
                "source_revision": SOURCE_REVISION,
            },
            "live_exact_tag_pids": [],
        },
        attempt_stager=lambda **_kwargs: pytest.fail("must not stage"),
        supervisor_launcher=lambda _argv: pytest.fail("must not launch"),
        remote_json_loader=lambda path: (
            {
                "classification": "PASS",
                "attempt": ATTEMPT,
                "source_revision": SOURCE_REVISION,
            }
            if path.endswith("/worker.json")
            else pytest.fail("unexpected remote JSON load")
        ),
        postprocess_state_query=lambda: {
            "producer": {
                "classification": "NO_GO_NO_REDUCIBLE_COLLECTIVE",
            },
            "verification": verification,
        },
        remote_command_runner=lambda argv: remote_commands.append(argv),
        bundle_fetcher=lambda *_args: pytest.fail("not reached"),
        local_verifier_runner=lambda *_args: pytest.fail("not reached"),
        sleep=lambda _seconds: None,
    )

    worker = adapter.worker_runner(plan)
    producer = adapter.assembler(plan, worker)
    remote = adapter.remote_verifier(plan)

    assert producer["classification"] == (
        "NO_GO_NO_REDUCIBLE_COLLECTIVE"
    )
    assert remote["classification"] == (
        "NO_GO_NO_REDUCIBLE_COLLECTIVE"
    )
    assert remote_commands == []


def test_binary_ssh_transport_retries_only_connection_loss():
    calls = []
    results = iter((
        SimpleNamespace(returncode=255, stdout=b"", stderr=b"closed"),
        SimpleNamespace(returncode=0, stdout=b"payload", stderr=b""),
    ))

    result = run_remote_bytes(
        ssh_target="sitian@example",
        remote_argv=["python3", "-c", "print('ok')"],
        input_bytes=b"input",
        timeout_s=30,
        retry_count=2,
        control_path="/tmp/test-control",
        command_runner=lambda argv, **kwargs: (
            calls.append((argv, kwargs)) or next(results)
        ),
        sleep=lambda _seconds: None,
    )

    assert result.stdout == b"payload"
    assert len(calls) == 2
    assert all(call[1]["input"] == b"input" for call in calls)
    assert all(call[1]["text"] is False for call in calls)


def test_stage_payload_and_remote_stage_are_source_bound():
    plan = _plan()
    source_identity = {
        "schema_version": (
            "qwen38.tp4-collective-reduction-source.v1"
        ),
        "attempt": ATTEMPT,
        "source_revision": SOURCE_REVISION,
        "source_tree_sha256": "c" * 64,
        "source_files": {"tools/worker.py": "d" * 64},
        "source_archive_paths": ["tinyvllm", "tools"],
    }
    model_manifest = {
        "schema_version": "tinyllmforge.qwen38-model-manifest.v1",
        "repository": "Qwen/Qwen3.8-27B",
        "revision": MODEL_REVISION,
        "text_profile": {},
    }
    topology = {
        "schema_version": (
            "qwen38.tp4-collective-reduction-topology.v1"
        ),
        "rank_mapping": [],
        "interconnect_matrix": "GPU0 GPU1 GPU2 GPU3",
    }
    payload = build_stage_payload(
        plan=plan,
        archive=b"archive",
        source_identity=source_identity,
        model_manifest=model_manifest,
        gpu_topology=topology,
    )
    metadata_size = int.from_bytes(payload[:8], "big")
    metadata = json.loads(payload[8:8 + metadata_size])

    assert metadata["source_identity"] == source_identity
    assert metadata["plan"] == plan
    assert payload[8 + metadata_size:] == b"archive"

    calls = []
    result = stage_remote_attempt(
        plan=plan,
        source_identity=source_identity,
        model_manifest=model_manifest,
        gpu_topology=topology,
        archive=b"archive",
        remote_runner=lambda **kwargs: (
            calls.append(kwargs)
            or SimpleNamespace(
                returncode=0,
                stdout=b'{"classification":"STAGED"}\n',
                stderr=b"",
            )
        ),
        ssh_target="sitian@example",
        timeout_s=30,
        retry_count=1,
    )

    assert result["classification"] == "STAGED"
    assert calls[0]["input_bytes"] == payload
    encoded = " ".join(calls[0]["remote_argv"])
    assert "/tmp" not in encoded
    assert plan["attempt_root"] in encoded
    assert APPROVED_REMOTE_ROOT in encoded


def test_remote_attempt_state_query_is_read_only_and_parses_receipts():
    expected = {
        "attempt_exists": True,
        "source_identity": {
            "attempt": ATTEMPT,
            "source_revision": SOURCE_REVISION,
            "source_tree_sha256": "c" * 64,
        },
        "launch": {"pid": 101, "source_revision": SOURCE_REVISION},
        "supervisor_receipt": None,
        "live_exact_tag_pids": [101],
    }
    calls = []

    result = query_remote_attempt_state(
        plan=_plan(),
        remote_runner=lambda **kwargs: (
            calls.append(kwargs)
            or SimpleNamespace(
                returncode=0,
                stdout=(
                    json.dumps(expected, sort_keys=True) + "\n"
                ).encode(),
                stderr=b"",
            )
        ),
        ssh_target="sitian@example",
        timeout_s=30,
        retry_count=1,
    )

    assert result == expected
    assert calls[0]["input_bytes"] is None
    script = calls[0]["remote_argv"][2]
    assert all(
        token not in script
        for token in (
            ".write_",
            ".mkdir",
            "os.remove",
            "unlink(",
            "replace(",
        )
    )


def test_remote_postprocess_state_query_is_read_only_and_parses_results():
    expected = {
        "producer": {
            "classification": "NO_GO_NO_REDUCIBLE_COLLECTIVE",
        },
        "verification": {
            "status": "PASS",
            "producer_classification": "NO_GO_NO_REDUCIBLE_COLLECTIVE",
            "reconstructed_classification": (
                "NO_GO_NO_REDUCIBLE_COLLECTIVE"
            ),
        },
    }
    calls = []

    result = query_remote_postprocess_state(
        plan=_plan(),
        remote_runner=lambda **kwargs: (
            calls.append(kwargs)
            or SimpleNamespace(
                returncode=0,
                stdout=(
                    json.dumps(expected, sort_keys=True) + "\n"
                ).encode(),
                stderr=b"",
            )
        ),
        ssh_target="sitian@example",
        timeout_s=30,
        retry_count=1,
    )

    assert result == expected
    assert calls[0]["input_bytes"] is None
    script = calls[0]["remote_argv"][2]
    assert "classification.json" in script
    assert "independent_verification.json" in script
    assert "manifest.sha256" in script
    assert "hashlib.sha256" in script
    assert all(
        token not in script
        for token in (
            ".write_",
            ".mkdir",
            "os.remove",
            "unlink(",
            "replace(",
        )
    )


def test_remote_postprocess_query_rejects_stale_verifier_manifest(tmp_path):
    bundle = tmp_path / "final_bundle"
    bundle.mkdir()
    plan = _plan()
    plan["bundle_root"] = str(bundle)

    def local_remote_runner(**kwargs):
        argv = list(kwargs["remote_argv"])
        argv[0] = sys.executable
        return subprocess.run(
            argv,
            input=kwargs["input_bytes"],
            capture_output=True,
            check=False,
        )

    empty = query_remote_postprocess_state(
        plan=plan,
        remote_runner=local_remote_runner,
        ssh_target="unused",
        timeout_s=30,
        retry_count=1,
    )
    assert empty == {"producer": None, "verification": None}

    producer = {
        "classification": "NO_GO_NO_REDUCIBLE_COLLECTIVE",
    }
    for name in PRODUCER_ARTIFACTS:
        if name == "manifest.sha256":
            continue
        payload = (
            json.dumps(producer, sort_keys=True, separators=(",", ":"))
            + "\n"
            if name == "classification.json"
            else "{}\n"
        ).encode()
        (bundle / name).write_bytes(payload)

    def write_manifest():
        artifacts = {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in bundle.iterdir()
            if path.is_file() and path.name != "manifest.sha256"
        }
        (bundle / "manifest.sha256").write_text(
            json.dumps(
                {
                    "schema_version": (
                        "qwen38.tp4-collective-reduction-manifest.v1"
                    ),
                    "artifacts": artifacts,
                },
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n",
            encoding="utf-8",
        )

    write_manifest()
    producer_only = query_remote_postprocess_state(
        plan=plan,
        remote_runner=local_remote_runner,
        ssh_target="unused",
        timeout_s=30,
        retry_count=1,
    )
    assert producer_only == {
        "producer": producer,
        "verification": None,
    }

    verification = {
        "status": "PASS",
        "producer_classification": producer["classification"],
        "reconstructed_classification": producer["classification"],
    }
    (bundle / "independent_verification.json").write_text(
        json.dumps(verification, sort_keys=True, separators=(",", ":"))
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="manifest is stale"):
        query_remote_postprocess_state(
            plan=plan,
            remote_runner=local_remote_runner,
            ssh_target="unused",
            timeout_s=30,
            retry_count=1,
        )

    write_manifest()
    complete = query_remote_postprocess_state(
        plan=plan,
        remote_runner=local_remote_runner,
        ssh_target="unused",
        timeout_s=30,
        retry_count=1,
    )
    assert complete == {
        "producer": producer,
        "verification": verification,
    }


def test_postprocess_state_rejects_partial_or_stale_verification():
    producer = {
        "classification": "NO_GO_NO_REDUCIBLE_COLLECTIVE",
    }
    verification = {
        "status": "PASS",
        "producer_classification": "NO_GO_NO_REDUCIBLE_COLLECTIVE",
        "reconstructed_classification": (
            "NO_GO_NO_REDUCIBLE_COLLECTIVE"
        ),
    }

    assert validate_postprocess_state({
        "producer": producer,
        "verification": verification,
    }) == {
        "producer": producer,
        "verification": verification,
    }
    with pytest.raises(RuntimeError, match="without producer"):
        validate_postprocess_state({
            "producer": None,
            "verification": verification,
        })
    with pytest.raises(RuntimeError, match="classification mismatch"):
        validate_postprocess_state({
            "producer": producer,
            "verification": {
                **verification,
                "producer_classification": (
                    "INCONCLUSIVE_PROFILER_OVERHEAD"
                ),
            },
        })


def test_remote_supervisor_launch_is_detached_and_idempotent():
    plan = _plan()
    calls = []
    expected = {
        "classification": "LAUNCHED",
        "pid": 101,
        "source_revision": SOURCE_REVISION,
    }

    result = launch_remote_supervisor(
        plan=plan,
        supervisor_argv=build_supervisor_argv(plan),
        remote_runner=lambda **kwargs: (
            calls.append(kwargs)
            or SimpleNamespace(
                returncode=0,
                stdout=(json.dumps(expected) + "\n").encode(),
                stderr=b"",
            )
        ),
        ssh_target="sitian@example",
        timeout_s=30,
        retry_count=1,
    )

    assert result == expected
    script = calls[0]["remote_argv"][2]
    assert "start_new_session=True" in script
    assert all(
        token not in script
        for token in ("os.kill", ".terminate(", ".kill(", "signal.")
    )
    assert "'TORCH_EXTENSIONS_DIR'" in script
    assert "'CUDA_CACHE_PATH'" in script
    assert "/tmp" not in script


def test_remote_json_helpers_and_exact_bundle_fetch():
    plan = _plan()
    calls = []
    responses = iter((
        b'{"value":1}\n',
        b'noise\n{"classification":"PASS"}\n',
        b"tar-bytes",
    ))

    def remote_runner(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(
            returncode=0,
            stdout=next(responses),
            stderr=b"",
        )

    loaded = load_remote_json(
        path=f"{plan['attempt_root']}/worker.json",
        remote_runner=remote_runner,
        ssh_target="sitian@example",
        timeout_s=30,
        retry_count=1,
    )
    command = run_remote_json_command(
        remote_argv=[REMOTE_PYTHON, "-c", "print('{}')"],
        remote_runner=remote_runner,
        ssh_target="sitian@example",
        timeout_s=30,
        retry_count=1,
    )
    archive = fetch_remote_bundle(
        plan=plan,
        names={
            "final_bundle/classification.json",
            "final_bundle/manifest.sha256",
        },
        remote_runner=remote_runner,
        ssh_target="sitian@example",
        timeout_s=30,
        retry_count=1,
    )

    assert loaded == {"value": 1}
    assert command == {"classification": "PASS"}
    assert archive == b"tar-bytes"
    fetch_script = calls[2]["remote_argv"][2]
    assert "is_symlink" in fetch_script
    assert "/tmp" not in fetch_script


def test_production_adapter_gives_postprocess_commands_extended_timeout(
    tmp_path,
):
    plan = _plan()
    observed_timeouts = []

    def command_runner(
        _argv,
        *,
        input,
        text,
        capture_output,
        check,
        timeout,
    ):
        assert input is None
        assert text is False
        assert capture_output is True
        assert check is False
        observed_timeouts.append(timeout)
        return SimpleNamespace(
            returncode=0,
            stdout=(
                b'{"classification":"INCONCLUSIVE_PROFILER_OVERHEAD"}\n'
            ),
            stderr=b"",
        )

    adapter = create_production_adapter(
        plan=plan,
        source_identity={
            "attempt": ATTEMPT,
            "source_revision": SOURCE_REVISION,
            "source_tree_sha256": "c" * 64,
        },
        model_manifest={"revision": MODEL_REVISION},
        repo_root=tmp_path,
        local_attempt_root=tmp_path / "attempt",
        ssh_target="sitian@example",
        control_path=None,
        command_timeout_s=120,
        retry_count=1,
        command_runner=command_runner,
    )
    adapter.postprocess_state = {
        "producer": None,
        "verification": None,
    }

    result = adapter.assembler(plan, {"classification": "PASS"})

    assert result["classification"] == (
        "INCONCLUSIVE_PROFILER_OVERHEAD"
    )
    assert observed_timeouts == [600]
