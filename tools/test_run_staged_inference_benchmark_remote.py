"""Dependency-light tests for the staged benchmark remote runner.

Run:
    python3 tools/test_run_staged_inference_benchmark_remote.py
"""

from __future__ import annotations

import hashlib
import io
import json
import os
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
import subprocess
import sys
import tarfile
from tempfile import TemporaryDirectory
import types
from datetime import datetime
from zoneinfo import ZoneInfo


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools import run_staged_inference_benchmark_remote as remote


def _gpu_row(
    index: int,
    *,
    memory_used_mib: int = 0,
    utilization_percent: int = 0,
    compute_processes: list[dict] | None = None,
) -> dict:
    return {
        "index": index,
        "uuid": f"GPU-{index}",
        "name": "NVIDIA H100 80GB HBM3",
        "memory_used_mib": memory_used_mib,
        "utilization_percent": utilization_percent,
        "compute_processes": (
            [] if compute_processes is None else compute_processes
        ),
    }


def _completed(
    stdout: str = "",
    *,
    returncode: int = 0,
    stderr: str = "",
) -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess(
        args=[],
        returncode=returncode,
        stdout=stdout,
        stderr=stderr,
    )


def test_remote_paths_are_all_below_approved_root():
    paths = remote.remote_paths("stage1-prefix-r1")
    assert paths == {
        "staging": (
            remote.APPROVED_ROOT
            + "/staged-benchmark/staging/stage1-prefix-r1"
        ),
        "primary": (
            remote.APPROVED_ROOT
            + "/staged-benchmark/runs/stage1-prefix-r1"
        ),
        "controller": (
            remote.APPROVED_ROOT
            + "/staged-benchmark/controller-verification/"
            "stage1-prefix-r1"
        ),
    }
    for path in paths.values():
        assert path.startswith(remote.APPROVED_ROOT + "/")
        assert "/tmp" not in path
        assert "/private/tmp" not in path
        assert "/data00/home/sitian/tllm/TinyLLMForge" not in path


def test_direct_script_entrypoint_adds_repo_root_to_sys_path():
    code = "\n".join((
        "import runpy",
        (
            "runpy.run_path("
            f"{str(Path(remote.__file__).resolve())!r})"
        ),
        "import tools.staged_inference_benchmark_gate",
    ))
    result = subprocess.run(
        [sys.executable, "-I", "-c", code],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_run_tag_rejects_paths_and_noncanonical_text():
    for value in (
        "",
        ".",
        "..",
        "../escape",
        "nested/tag",
        "white space",
        "-leading",
        "a" * 129,
    ):
        try:
            remote.validate_run_tag(value)
        except ValueError:
            pass
        else:
            raise AssertionError(f"unsafe run tag accepted: {value!r}")
    assert remote.validate_run_tag("stage1-prefix-r1") == "stage1-prefix-r1"


def test_gpu_admission_is_strict_and_non_destructive():
    rows = [
        _gpu_row(0, memory_used_mib=1024, utilization_percent=5),
        _gpu_row(1, memory_used_mib=1025),
        _gpu_row(2, utilization_percent=6),
        _gpu_row(
            3,
            compute_processes=[
                {"pid": 123, "process_name": "python"},
            ],
        ),
    ]
    assert remote.strict_clean_gpus(rows) == [rows[0]]
    source = Path(remote.__file__).read_text(encoding="utf-8")
    for forbidden in (
        "pkill",
        "killall",
        "nvidia-smi --gpu-reset",
        "rsync",
    ):
        assert forbidden not in source


def test_stage1_requires_exactly_one_clean_gpu():
    selected = remote.select_admitted_gpus(
        [_gpu_row(0), _gpu_row(1, memory_used_mib=2048)],
        model_tier="qwen3-0.6b",
        capacity_receipt=None,
    )
    assert [row["index"] for row in selected] == [0]
    try:
        remote.select_admitted_gpus(
            [_gpu_row(0, memory_used_mib=2048)],
            model_tier="qwen3-0.6b",
            capacity_receipt=None,
        )
    except ValueError as error:
        assert "one strict-clean GPU" in str(error)
    else:
        raise AssertionError("Stage 1 accepted no clean GPU")


def test_stage2_requires_explicit_capacity_receipt():
    rows = [_gpu_row(index) for index in range(4)]
    try:
        remote.select_admitted_gpus(
            rows,
            model_tier="qwen3-8b",
            capacity_receipt=None,
        )
    except ValueError as error:
        assert "capacity preflight" in str(error)
    else:
        raise AssertionError("Stage 2 accepted implicit GPU capacity")
    receipt = {
        "schema_version": 1,
        "model_tier": "qwen3-8b",
        "required_gpu_count": 4,
        "status": "PASS",
    }
    selected = remote.select_admitted_gpus(
        rows,
        model_tier="qwen3-8b",
        capacity_receipt=receipt,
    )
    assert [row["index"] for row in selected] == [0, 1, 2, 3]
    for unsupported_count in (2, 3):
        invalid = dict(receipt)
        invalid["required_gpu_count"] = unsupported_count
        try:
            remote.select_admitted_gpus(
                rows,
                model_tier="qwen3-8b",
                capacity_receipt=invalid,
            )
        except ValueError as error:
            assert "GPU count" in str(error)
        else:
            raise AssertionError(
                "unsupported Stage 2 GPU count was accepted"
            )


def test_ownership_change_records_concrete_conflict():
    baseline = [_gpu_row(0)]
    observed = [
        _gpu_row(
            0,
            memory_used_mib=2048,
            utilization_percent=20,
            compute_processes=[
                {"pid": 456, "process_name": "python"},
            ],
        )
    ]
    conflict = remote.gpu_ownership_conflict(
        baseline_rows=baseline,
        observed_rows=observed,
        selected_gpu_indices=[0],
        owned_pids=set(),
        phase="before-case",
    )
    assert conflict == {
        "schema_version": 1,
        "phase": "before-case",
        "selected_gpu_indices": [0],
        "conflicts": [
            {
                "gpu_index": 0,
                "gpu_uuid": "GPU-0",
                "pid": 456,
                "process_name": "python",
            }
        ],
        "status": "CONFLICT",
    }


def test_ownership_check_accepts_only_owned_compute_processes():
    baseline = [_gpu_row(0)]
    observed = [
        _gpu_row(
            0,
            memory_used_mib=2048,
            utilization_percent=20,
            compute_processes=[
                {"pid": 456, "process_name": "python"},
            ],
        )
    ]
    assert remote.gpu_ownership_conflict(
        baseline_rows=baseline,
        observed_rows=observed,
        selected_gpu_indices=[0],
        owned_pids={456},
        phase="after-case",
    ) is None


def test_ownership_sampling_closes_child_creation_race():
    inventories = iter(({123}, {123, 456}))
    original_inventory = remote._owned_process_group_pids
    original_query = remote._query_gpu_rows_local
    try:
        remote._owned_process_group_pids = (
            lambda process_group_id: set(next(inventories))
        )
        remote._query_gpu_rows_local = lambda: [
            _gpu_row(
                0,
                compute_processes=[
                    {"pid": 456, "process_name": "python"},
                ],
            )
        ]
        rows, owned = remote._sample_owned_gpu_rows(
            process_group_id=123,
            owned_pids={123},
        )
    finally:
        remote._owned_process_group_pids = original_inventory
        remote._query_gpu_rows_local = original_query
    assert rows[0]["compute_processes"][0]["pid"] == 456
    assert owned == {123, 456}


def test_owned_process_group_cleanup_escalates_until_group_is_empty():
    signals = []
    inventories = iter([{456}, set()])

    class Process:
        def wait(self, timeout=None):
            return 0

        def poll(self):
            return 0

    original_killpg = remote.os.killpg
    original_inventory = remote._owned_process_group_pids
    original_sleep = remote.time.sleep
    original_monotonic = remote.time.monotonic
    ticks = iter([0.0, 6.0])
    try:
        remote.os.killpg = lambda pgid, sig: signals.append((pgid, sig))
        remote._owned_process_group_pids = lambda pgid: next(inventories)
        remote.time.sleep = lambda _: None
        remote.time.monotonic = lambda: next(ticks)
        remote._terminate_owned_process_group(Process(), 456)
    finally:
        remote.os.killpg = original_killpg
        remote._owned_process_group_pids = original_inventory
        remote.time.sleep = original_sleep
        remote.time.monotonic = original_monotonic
    assert signals == [
        (456, remote.signal.SIGTERM),
        (456, remote.signal.SIGKILL),
    ]


def test_kerberos_validation_never_refreshes_ticket():
    calls = []

    def runner(command, **kwargs):
        calls.append((command, kwargs))
        return _completed(
            "Credentials cache: FILE:/Users/bytedance/krb5cc_sitian\n"
            "        Principal: sitian@BYTEDANCE.COM\n\n"
            "  Issued                Expires               Principal\n"
            "Aug 21 20:00:00 2026  Aug 21 23:00:00 2026  "
            "krbtgt/BYTEDANCE.COM@BYTEDANCE.COM\n"
        )

    receipt = remote.validate_kerberos(
        environ={
            "KRB5CCNAME": remote.KRB5CCNAME,
        },
        command_runner=runner,
        now=lambda: datetime(
            2026,
            8,
            21,
            21,
            0,
            tzinfo=ZoneInfo("Asia/Shanghai"),
        ),
    )
    assert receipt["status"] == "PASS"
    assert receipt["remaining_lifetime_seconds"] == 7200
    assert calls[0][0] == ["klist"]
    assert calls[0][1]["env"]["KRB5CCNAME"] == remote.KRB5CCNAME
    source = Path(remote.__file__).read_text(encoding="utf-8")
    assert "kinit" not in source


def test_kerberos_validation_fails_before_short_ticket_run():
    def runner(command, **kwargs):
        return _completed(
            "Credentials cache: FILE:/Users/bytedance/krb5cc_sitian\n"
            "        Principal: sitian@BYTEDANCE.COM\n\n"
            "  Issued                Expires               Principal\n"
            "Aug 21 20:00:00 2026  Aug 21 21:30:00 2026  "
            "krbtgt/BYTEDANCE.COM@BYTEDANCE.COM\n"
        )

    try:
        remote.validate_kerberos(
            environ={"KRB5CCNAME": remote.KRB5CCNAME},
            command_runner=runner,
            now=lambda: datetime(
                2026,
                8,
                21,
                21,
                0,
                tzinfo=ZoneInfo("Asia/Shanghai"),
            ),
            minimum_lifetime_seconds=5400,
        )
    except ValueError as error:
        assert "remaining lifetime" in str(error)
    else:
        raise AssertionError("short Kerberos ticket was accepted")


def test_source_head_must_equal_tracking_ref():
    outputs = iter([
        _completed("a" * 40 + "\n"),
        _completed("b" * 40 + "\n"),
    ])

    def runner(command, **kwargs):
        return next(outputs)

    try:
        remote.require_pushed_head(
            REPO_ROOT,
            command_runner=runner,
        )
    except ValueError as error:
        assert "origin/feat/kv-sparse-attention" in str(error)
    else:
        raise AssertionError("mismatched pushed source was accepted")


def test_remote_gpu_query_preserves_process_identity():
    rows = [
        _gpu_row(
            2,
            memory_used_mib=900,
            utilization_percent=3,
            compute_processes=[
                {"pid": 789, "process_name": "python"},
            ],
        ),
    ]

    def runner(command, **kwargs):
        return _completed(json.dumps(rows))

    assert remote.query_remote_gpu_rows(command_runner=runner) == rows


def test_remote_requirement_probe_is_model_specific_and_space_bounded():
    payload = {
        "python": {
            "path": remote.REMOTE_PYTHON,
            "is_file": True,
            "is_executable": True,
        },
        "model": {
            "path": remote.MODEL_PATHS["qwen3-0.6b"],
            "is_dir": True,
            "config_path": (
                remote.MODEL_PATHS["qwen3-0.6b"] + "/config.json"
            ),
            "config_is_file": True,
        },
        "approved_root": {
            "path": remote.APPROVED_ROOT,
            "is_dir": True,
            "free_bytes": 2 * remote.MINIMUM_REMOTE_FREE_BYTES,
        },
    }

    def runner(command, **kwargs):
        return _completed(json.dumps(payload))

    assert remote.probe_remote_requirements(
        "qwen3-0.6b",
        command_runner=runner,
    ) == payload


def test_preflight_receipt_binds_source_paths_and_selected_gpu():
    rows = [_gpu_row(0)]
    paths = remote.remote_paths("prefix-stage1-r1")
    receipt = remote.build_preflight_receipt(
        gate_name="prefix",
        model_tier="qwen3-0.6b",
        run_tag="prefix-stage1-r1",
        source_commit="a" * 40,
        kerberos_receipt={
            "schema_version": 1,
            "status": "PASS",
            "cache": remote.KRB5CCNAME,
            "principal": "sitian@BYTEDANCE.COM",
            "tgt_principal": (
                "krbtgt/BYTEDANCE.COM@BYTEDANCE.COM"
            ),
            "expires_at": "2026-08-21T23:00:00+08:00",
            "minimum_required_lifetime_seconds": 5400,
            "remaining_lifetime_seconds": 7200,
        },
        remote_requirements={
            "python": {"path": remote.REMOTE_PYTHON},
            "model": {
                "path": remote.MODEL_PATHS["qwen3-0.6b"],
            },
            "approved_root": {
                "path": remote.APPROVED_ROOT,
                "free_bytes": 10**12,
            },
        },
        gpu_rows=rows,
        selected_rows=rows,
        paths=paths,
        capacity_receipt=None,
    )
    assert receipt["status"] == "READY"
    assert receipt["source_commit"] == "a" * 40
    assert receipt["selected_gpu_indices"] == [0]
    assert receipt["selected_gpu_uuids"] == ["GPU-0"]
    assert receipt["remote_paths"] == paths


def test_execution_spec_binds_preflight_source_and_runtime_paths():
    run_tag = "prefix-execute-r1"
    paths = remote.remote_paths(run_tag)
    preflight = {
        "schema_version": 1,
        "status": "READY",
        "gate": "prefix",
        "model_tier": "qwen3-0.6b",
        "run_tag": run_tag,
        "source_commit": "a" * 40,
        "selected_gpu_indices": [2],
        "selected_gpu_uuids": ["GPU-2"],
        "remote_paths": paths,
    }
    source = {
        "schema_version": 1,
        "base_commit": "a" * 40,
        "local_head": "a" * 40,
        "tracking_head": "a" * 40,
        "dirty": False,
        "tree_sha256": "b" * 64,
        "owned_roots": ["tools/example.py"],
    }
    spec = remote.build_execution_spec(
        preflight=preflight,
        source_evidence=source,
        promotion=None,
    )
    assert spec["remote_paths"] == paths
    assert spec["runtime_environment"] == (
        remote.remote_runtime_environment(paths["primary"])
    )
    assert spec["source_evidence"] == source
    assert spec["promotion"] is None


def test_run_preflight_samples_gpu_twice_and_writes_exclusively():
    with TemporaryDirectory() as temporary:
        output_dir = Path(temporary) / "preflight"
        gpu_calls = []

        def gpu_query(**kwargs):
            gpu_calls.append(True)
            return [_gpu_row(0)]

        receipt = remote.run_preflight(
            repo_root=REPO_ROOT,
            gate_name="prefix",
            model_tier="qwen3-0.6b",
            run_tag="prefix-preflight-r1",
            output_dir=output_dir,
            capacity_receipt=None,
            kerberos_validator=lambda **kwargs: {
                "schema_version": 1,
                "status": "PASS",
                "cache": remote.KRB5CCNAME,
                "principal": "sitian@BYTEDANCE.COM",
                "tgt_principal": (
                    "krbtgt/BYTEDANCE.COM@BYTEDANCE.COM"
                ),
                "expires_at": "2026-08-21T23:00:00+08:00",
                "minimum_required_lifetime_seconds": 5400,
                "remaining_lifetime_seconds": 7200,
            },
            source_validator=lambda *args, **kwargs: "a" * 40,
            destination_validator=lambda *args, **kwargs: None,
            requirement_probe=lambda *args, **kwargs: {
                "python": {"path": remote.REMOTE_PYTHON},
                "model": {
                    "path": remote.MODEL_PATHS["qwen3-0.6b"],
                },
                "approved_root": {
                    "path": remote.APPROVED_ROOT,
                    "free_bytes": 10**12,
                },
            },
            gpu_query=gpu_query,
        )
        assert len(gpu_calls) == 2
        assert receipt["status"] == "READY"
        assert receipt["gpu_rows_before"] == [_gpu_row(0)]
        assert receipt["gpu_rows_after"] == [_gpu_row(0)]
        assert json.loads(
            (output_dir / "preflight.json").read_text(encoding="utf-8")
        ) == receipt
        try:
            remote.run_preflight(
                repo_root=REPO_ROOT,
                gate_name="prefix",
                model_tier="qwen3-0.6b",
                run_tag="prefix-preflight-r1",
                output_dir=output_dir,
                capacity_receipt=None,
                kerberos_validator=lambda **kwargs: {},
                source_validator=lambda *args, **kwargs: "a" * 40,
                destination_validator=lambda *args, **kwargs: None,
                requirement_probe=lambda *args, **kwargs: {},
                gpu_query=gpu_query,
            )
        except ValueError as error:
            assert "already exists" in str(error)
        else:
            raise AssertionError("preflight receipt was overwritten")


def test_exclusive_json_is_atomically_published_after_complete_write():
    with TemporaryDirectory() as temporary:
        destination = Path(temporary) / "receipt.json"
        publications = []
        original_link = remote.os.link
        try:
            def publish(source, target):
                source_path = Path(source)
                target_path = Path(target)
                assert not target_path.exists()
                assert json.loads(
                    source_path.read_text(encoding="utf-8")
                ) == {"status": "PASS"}
                publications.append((source_path, target_path))
                original_link(source, target)

            remote.os.link = publish
            remote._write_json_exclusive(
                destination,
                {"status": "PASS"},
            )
        finally:
            remote.os.link = original_link
        assert publications
        assert json.loads(
            destination.read_text(encoding="utf-8")
        ) == {"status": "PASS"}
        assert not list(destination.parent.glob("*.tmp-*"))


def test_remote_destination_preflight_rejects_any_existing_path():
    paths = remote.remote_paths("immutable-r1")

    def runner(command, **kwargs):
        assert command[-1].startswith("python3 -c ")
        return _completed(json.dumps({
            paths["staging"]: False,
            paths["primary"]: True,
            paths["controller"]: False,
        }))

    try:
        remote.require_remote_destinations_absent(
            paths,
            command_runner=runner,
        )
    except ValueError as error:
        assert paths["primary"] in str(error)
    else:
        raise AssertionError("existing immutable run path was accepted")


def test_runtime_environment_stays_under_run_root():
    primary = remote.remote_paths("env-r1")["primary"]
    environment = remote.remote_runtime_environment(primary)
    assert environment == {
        "TMPDIR": primary + "/tmp",
        "TEMP": primary + "/tmp",
        "TMP": primary + "/tmp",
        "PYTHONPYCACHEPREFIX": primary + "/pycache",
        "HF_HOME": primary + "/hf-home",
        "TORCH_EXTENSIONS_DIR": primary + "/torch-extensions",
    }
    assert all(
        value.startswith(primary + "/")
        for value in environment.values()
    )


def test_download_ranges_are_bounded_and_complete():
    assert list(remote.iter_download_ranges(0, chunk_size=4)) == []
    assert list(remote.iter_download_ranges(10, chunk_size=4)) == [
        (0, 4),
        (4, 4),
        (8, 2),
    ]
    for size, chunk_size in ((-1, 4), (1, 0)):
        try:
            list(remote.iter_download_ranges(size, chunk_size=chunk_size))
        except ValueError:
            pass
        else:
            raise AssertionError("invalid chunk bounds were accepted")


def test_deterministic_tar_has_stable_regular_file_metadata():
    with TemporaryDirectory() as temporary:
        root = Path(temporary) / "source"
        (root / "nested").mkdir(parents=True)
        (root / "b.txt").write_bytes(b"b")
        (root / "nested" / "a.txt").write_bytes(b"a")
        first = remote.build_deterministic_tar(root, prefix="source")
        second = remote.build_deterministic_tar(root, prefix="source")
        assert first == second
        with tarfile.open(fileobj=io.BytesIO(first), mode="r:") as archive:
            members = archive.getmembers()
        assert [member.name for member in members] == [
            "source/b.txt",
            "source/nested/a.txt",
        ]
        assert all(member.isfile() for member in members)
        assert all(member.mtime == 0 for member in members)
        assert all(member.uid == 0 and member.gid == 0 for member in members)


def test_deterministic_tar_rejects_symlinks():
    with TemporaryDirectory() as temporary:
        root = Path(temporary) / "source"
        root.mkdir()
        (root / "target").write_text("x", encoding="utf-8")
        (root / "link").symlink_to("target")
        try:
            remote.build_deterministic_tar(root, prefix="source")
        except ValueError as error:
            assert "symlink" in str(error)
        else:
            raise AssertionError("source symlink was archived")


def test_download_member_rejects_links_and_path_escape():
    for name, is_file, is_link in (
        ("../escape", True, False),
        ("/absolute", True, False),
        ("nested/../../escape", True, False),
        ("safe/link", False, True),
        ("safe\\windows", True, False),
    ):
        try:
            remote.validate_download_member(
                name,
                is_file=is_file,
                is_link=is_link,
            )
        except ValueError:
            pass
        else:
            raise AssertionError(f"unsafe member accepted: {name}")
    assert (
        remote.validate_download_member(
            "cases/case-0/result.json",
            is_file=True,
            is_link=False,
        )
        == "cases/case-0/result.json"
    )


def test_download_inventory_requires_canonical_chunk_hashes():
    payload = b"abcdefghij"
    inventory = {
        "schema_version": 1,
        "root": remote.APPROVED_ROOT + "/staged-benchmark/runs/run-r1",
        "files": [
            {
                "path": "summary.json",
                "size_bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
                "chunks": [
                    {
                        "offset": 0,
                        "length": 4,
                        "sha256": hashlib.sha256(payload[:4]).hexdigest(),
                    },
                    {
                        "offset": 4,
                        "length": 4,
                        "sha256": hashlib.sha256(payload[4:8]).hexdigest(),
                    },
                    {
                        "offset": 8,
                        "length": 2,
                        "sha256": hashlib.sha256(payload[8:]).hexdigest(),
                    },
                ],
            }
        ],
    }
    assert remote.validate_download_inventory(
        inventory,
        expected_root=inventory["root"],
        chunk_size=4,
    ) == inventory["files"]
    inventory["files"][0]["chunks"][1]["offset"] = 5
    try:
        remote.validate_download_inventory(
            inventory,
            expected_root=inventory["root"],
            chunk_size=4,
        )
    except ValueError as error:
        assert "chunk" in str(error)
    else:
        raise AssertionError("noncanonical download chunks were accepted")


def test_download_chunk_hash_and_size_are_verified():
    payload = b"artifact bytes"
    remote_path = remote.APPROVED_ROOT + "/staged-benchmark/file"

    def runner(command, **kwargs):
        return _completed(payload)

    actual = remote.download_chunk(
        remote_path,
        offset=0,
        length=len(payload),
        expected_sha256=hashlib.sha256(payload).hexdigest(),
        command_runner=runner,
    )
    assert actual == payload
    try:
        remote.download_chunk(
            remote_path,
            offset=0,
            length=len(payload),
            expected_sha256="0" * 64,
            command_runner=runner,
        )
    except ValueError as error:
        assert "sha256" in str(error)
    else:
        raise AssertionError("corrupt download chunk was accepted")


def test_failed_tree_download_leaves_no_final_or_partial_destination():
    with TemporaryDirectory() as temporary:
        target = Path(temporary) / "download"
        payload = b"artifact"
        inventory = [{
            "path": "summary.json",
            "size_bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
            "chunks": [{
                "offset": 0,
                "length": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }],
        }]
        original_inventory = remote.fetch_remote_inventory
        original_chunk = remote.download_chunk
        try:
            remote.fetch_remote_inventory = (
                lambda *args, **kwargs: inventory
            )

            def fail_chunk(*args, **kwargs):
                raise RuntimeError("transport interrupted")

            remote.download_chunk = fail_chunk
            try:
                remote.download_remote_tree(
                    remote.APPROVED_ROOT + "/staged-benchmark/runs/r1",
                    target,
                    retries=1,
                )
            except RuntimeError:
                pass
            else:
                raise AssertionError("failed download was accepted")
        finally:
            remote.fetch_remote_inventory = original_inventory
            remote.download_chunk = original_chunk
        assert not target.exists()
        assert not target.with_name(target.name + ".partial").exists()


def test_complete_existing_download_is_reused_after_hash_verification():
    with TemporaryDirectory() as temporary:
        target = Path(temporary) / "download"
        target.mkdir()
        payload = b"artifact"
        (target / "summary.json").write_bytes(payload)
        inventory = [{
            "path": "summary.json",
            "size_bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
            "chunks": [{
                "offset": 0,
                "length": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }],
        }]
        assert remote.verify_downloaded_tree(
            target,
            inventory,
        ) == inventory


def test_ownership_samples_record_normal_rows_not_only_conflicts():
    with TemporaryDirectory() as temporary:
        path = Path(temporary) / "samples.jsonl"
        rows = [_gpu_row(2)]
        remote.append_ownership_sample(
            path,
            phase="case-0:running",
            rows=rows,
            owned_pids={123},
        )
        sample = json.loads(path.read_text(encoding="utf-8"))
        assert sample["phase"] == "case-0:running"
        assert sample["gpu_rows"] == rows
        assert sample["owned_pids"] == [123]
        assert isinstance(sample["sampled_at_unix_ns"], int)


def test_verification_receipts_must_match_exactly():
    receipt = {
        "status": "PASS",
        "run_manifest_sha256": "1" * 64,
        "primary_summary_sha256": "2" * 64,
        "controller_summary_sha256": "2" * 64,
        "classification": "PREFIX_CACHE_GO",
    }
    assert remote.compare_verification_receipts(
        receipt,
        dict(receipt),
    ) == {
        "schema_version": 1,
        "status": "PASS",
        "receipt_sha256": hashlib.sha256(
            json.dumps(
                receipt,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest(),
    }
    changed = dict(receipt)
    changed["classification"] = "PREFIX_CACHE_NO_GO"
    try:
        remote.compare_verification_receipts(receipt, changed)
    except ValueError as error:
        assert "disagree" in str(error)
    else:
        raise AssertionError("disagreeing verifier receipts were accepted")


def test_cli_requires_only_command_specific_qwen3_8b_evidence():
    preflight = remote.parse_args([
        "preflight",
        "--gate",
        "prefix",
        "--model-tier",
        "qwen3-8b",
        "--run-tag",
        "stage2-preflight-r1",
        "--promotion-prefix-run",
        "prefix",
        "--promotion-chunked-run",
        "chunked",
        "--capacity-receipt",
        "capacity.json",
    ])
    assert preflight.promotion_prefix_run == Path("prefix")
    execute = remote.parse_args([
        "execute",
        "--gate",
        "prefix",
        "--model-tier",
        "qwen3-8b",
        "--run-tag",
        "stage2-execute-r1",
        "--promotion-prefix-run",
        "prefix",
        "--promotion-chunked-run",
        "chunked",
    ])
    assert execute.capacity_receipt is None
    for command in ("download-only", "verify-local"):
        args = remote.parse_args([
            command,
            "--gate",
            "prefix",
            "--model-tier",
            "qwen3-8b",
            "--run-tag",
            f"stage2-{command}-r1",
        ])
        assert args.command == command
    try:
        with io.StringIO() as stderr, redirect_stderr(stderr):
            remote.parse_args([
                "execute",
                "--gate",
                "prefix",
                "--model-tier",
                "qwen3-8b",
                "--run-tag",
                "stage2-r1",
            ])
    except SystemExit:
        pass
    else:
        raise AssertionError("8B execute accepted missing promotion evidence")
    args = remote.parse_args([
        "preflight",
        "--gate",
        "prefix",
        "--model-tier",
        "qwen3-0.6b",
        "--run-tag",
        "stage1-r1",
    ])
    assert args.command == "preflight"


def test_qwen3_8b_preflight_validates_promotion_before_gpu_admission():
    with TemporaryDirectory() as temporary:
        capacity = Path(temporary) / "capacity.json"
        capacity.write_text("{}", encoding="utf-8")
        calls = []
        original_promotion = remote._load_promotion
        original_preflight = remote.run_preflight
        try:
            remote._load_promotion = (
                lambda args: calls.append("promotion")
                or {"winner": "prefix"}
            )
            remote.run_preflight = (
                lambda **kwargs: calls.append("preflight")
                or {"status": "READY"}
            )
            with io.StringIO() as stdout, redirect_stdout(stdout):
                assert remote.main([
                    "preflight",
                    "--gate",
                    "prefix",
                    "--model-tier",
                    "qwen3-8b",
                    "--run-tag",
                    "stage2-preflight-order-r1",
                    "--promotion-prefix-run",
                    "prefix",
                    "--promotion-chunked-run",
                    "chunked",
                    "--capacity-receipt",
                    str(capacity),
                    "--local-run-dir",
                    str(Path(temporary) / "local"),
                ]) == 0
        finally:
            remote._load_promotion = original_promotion
            remote.run_preflight = original_preflight
    assert calls == ["promotion", "preflight"]


def test_stage2_promotion_requires_matching_remote_and_local_verifiers():
    with TemporaryDirectory() as temporary:
        root = Path(temporary)
        primary = root / "primary"
        controller = root / "controller"
        local = root / "local-verification"
        primary.mkdir()
        controller.mkdir()
        summary = {"classification": "PREFIX_CACHE_GO"}
        summary_bytes = (
            json.dumps(summary, sort_keys=True, indent=2) + "\n"
        ).encode("utf-8")
        manifest_bytes = b'{"status":"FINALIZED"}\n'
        (primary / "summary.json").write_bytes(summary_bytes)
        (primary / "run_manifest.json").write_bytes(manifest_bytes)
        (controller / "summary.json").write_bytes(summary_bytes)
        receipt = {
            "status": "PASS",
            "run_manifest_sha256": "1" * 64,
            "primary_summary_sha256": hashlib.sha256(
                summary_bytes
            ).hexdigest(),
            "controller_summary_sha256": hashlib.sha256(
                summary_bytes
            ).hexdigest(),
            "classification": "PREFIX_CACHE_GO",
        }
        (controller / "verification_receipt.json").write_text(
            json.dumps(receipt),
            encoding="utf-8",
        )
        try:
            remote._load_stage1_promotion_bundle(root)
        except ValueError as error:
            assert "local verification" in str(error)
        else:
            raise AssertionError(
                "remote-only Stage-1 verification was accepted"
            )
        local.mkdir()
        (local / "verification_receipt.json").write_text(
            json.dumps(receipt),
            encoding="utf-8",
        )
        comparison = remote.compare_verification_receipts(
            receipt,
            receipt,
        )
        (local / "receipt_comparison.json").write_text(
            json.dumps(comparison),
            encoding="utf-8",
        )
        try:
            remote._load_stage1_promotion_bundle(root)
        except ValueError as error:
            assert "manifest" in str(error)
        else:
            raise AssertionError(
                "promotion accepted an unbound run manifest"
            )
        receipt["run_manifest_sha256"] = hashlib.sha256(
            manifest_bytes
        ).hexdigest()
        for path in (
            controller / "verification_receipt.json",
            local / "verification_receipt.json",
        ):
            path.write_text(json.dumps(receipt), encoding="utf-8")
        comparison = remote.compare_verification_receipts(
            receipt,
            receipt,
        )
        (local / "receipt_comparison.json").write_text(
            json.dumps(comparison),
            encoding="utf-8",
        )
        loaded_summary, loaded_receipt = (
            remote._load_stage1_promotion_bundle(root)
        )
        assert loaded_summary == summary
        assert loaded_receipt["status"] == "PASS"


def test_remote_execution_is_detached_then_polled():
    spec = {
        "run_tag": "detached-r1",
        "remote_paths": remote.remote_paths("detached-r1"),
    }
    responses = iter([
        _completed(json.dumps({
            "schema_version": 1,
            "status": "STARTED",
            "pid": 12345,
        })),
        _completed(json.dumps({
            "schema_version": 1,
            "done": False,
            "alive": True,
        })),
        _completed(json.dumps({
            "schema_version": 1,
            "done": True,
            "alive": False,
            "result": {
                "schema_version": 1,
                "status": "PASS",
                "run_tag": "detached-r1",
            },
        })),
    ])
    calls = []

    def runner(command, **kwargs):
        calls.append((command, kwargs))
        return next(responses)

    ticks = iter([0.0, 1.0, 2.0])
    result = remote.launch_remote_execution(
        spec=spec,
        command_runner=runner,
        sleep=lambda _: None,
        monotonic=lambda: next(ticks),
        timeout_seconds=10,
    )
    assert result["status"] == "PASS"
    assert len(calls) == 3
    source = Path(remote.__file__).read_text(encoding="utf-8")
    assert "start_new_session=True" in source


def test_orchestration_error_still_downloads_partial_artifacts():
    calls = []

    def uploader(**kwargs):
        return {"status": "PASS"}

    def launcher(**kwargs):
        raise RuntimeError("poll connection closed")

    def downloader(**kwargs):
        calls.append("download")
        return {
            "primary": {"path": "/local/primary", "file_count": 3},
            "controller": None,
        }

    final = remote.execute_and_collect(
        run_tag="partial-r1",
        output_dir=Path("/local"),
        payload=b"payload",
        spec={
            "run_tag": "partial-r1",
            "remote_paths": remote.remote_paths("partial-r1"),
        },
        uploader=uploader,
        launcher=launcher,
        downloader=downloader,
        kerberos_validator=lambda: {"status": "PASS"},
        local_verifier=lambda **kwargs: {
            "status": "PASS",
        },
    )
    assert calls == ["download"]
    assert final["status"] == "FAILED"
    assert final["execution"] is None
    assert final["orchestration_error"]["error_type"] == "RuntimeError"
    assert final["downloaded"]["primary"]["file_count"] == 3


def test_execute_revalidates_kerberos_before_upload_and_still_collects():
    calls = []

    def reject_short_ticket():
        calls.append("kerberos")
        raise ValueError("remaining lifetime is below minimum")

    def uploader(**kwargs):
        calls.append("upload")
        return {"status": "PASS"}

    def downloader(**kwargs):
        calls.append("download")
        return {"primary": None, "controller": None}

    final = remote.execute_and_collect(
        run_tag="short-ticket-r1",
        output_dir=Path("/local"),
        payload=b"payload",
        spec={
            "run_tag": "short-ticket-r1",
            "remote_paths": remote.remote_paths("short-ticket-r1"),
        },
        uploader=uploader,
        launcher=lambda **kwargs: {"status": "PASS"},
        downloader=downloader,
        kerberos_validator=reject_short_ticket,
        local_verifier=lambda **kwargs: {"status": "PASS"},
    )
    assert calls == ["kerberos", "download"]
    assert final["status"] == "FAILED"
    assert final["orchestration_error"]["error_type"] == "ValueError"


def main() -> int:
    tests = [
        value
        for name, value in sorted(globals().items())
        if name.startswith("test_") and isinstance(value, types.FunctionType)
    ]
    for test in tests:
        test()
    print("staged inference benchmark remote runner tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
