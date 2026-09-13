from __future__ import annotations

import io
import json
from pathlib import Path
import subprocess
import sys
import tarfile
from types import SimpleNamespace

import pytest

from tools import run_slo_cohort_burst_remote as remote


def _gpu(
    index: int,
    *,
    memory: int = 0,
    utilization: int = 0,
    processes=None,
) -> dict:
    return {
        "index": index,
        "uuid": f"GPU-{index}",
        "name": "NVIDIA A100 80GB PCIe",
        "memory_used_mib": memory,
        "utilization_percent": utilization,
        "compute_processes": [] if processes is None else processes,
    }


def test_direct_script_entrypoint_can_import_tools() -> None:
    result = subprocess.run(
        [sys.executable, str(Path(remote.__file__)), "--help"],
        cwd=Path(remote.__file__).resolve().parents[1],
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr


def test_remote_paths_are_confined_to_approved_large_mount() -> None:
    paths = remote.build_remote_paths("20260913-stage0-r1")
    prefix = (
        "/data00/home/sitian/tinyllmforge-workspaces/"
        "command-timeline-20260818/slo-cohort-burst/"
    )
    assert set(paths) == {"staging", "primary", "controller"}
    assert all(path.startswith(prefix) for path in paths.values())
    assert all("/tmp/" not in path for path in paths.values())


@pytest.mark.parametrize(
    "tag",
    ("", "../escape", "/absolute", "white space", "a" * 129),
)
def test_remote_paths_reject_invalid_run_tags(tag: str) -> None:
    with pytest.raises(ValueError, match="run tag"):
        remote.build_remote_paths(tag)


def test_remote_runtime_prelude_redirects_every_cache_to_large_mount() -> None:
    paths = remote.build_remote_paths("20260913-stage0-r1")
    source = paths["staging"] + "/source"
    prelude = remote.build_remote_runtime_prelude(
        source=source,
        gpu_index=3,
        dist_port=23456,
    )
    runtime = paths["staging"] + "/runtime"
    for name in (
        "TMPDIR",
        "TMP",
        "TEMP",
        "PYTHONPYCACHEPREFIX",
        "XDG_CACHE_HOME",
        "HF_HOME",
        "TORCH_EXTENSIONS_DIR",
    ):
        assert f"export {name}=" in prelude
    assert runtime in prelude
    assert "CUDA_VISIBLE_DEVICES=3" in prelude
    assert "MASTER_PORT=23456" in prelude
    assert "export TMPDIR=/tmp" not in prelude


def test_kerberos_guard_is_short_enough_for_fast_gpu_claim() -> None:
    assert remote.MINIMUM_KERBEROS_LIFETIME_SECONDS == 1_800


def test_strict_clean_a100_requires_zero_processes_and_low_usage() -> None:
    rows = [
        _gpu(0),
        _gpu(1, memory=1_025),
        _gpu(2, utilization=6),
        _gpu(3, processes=[{"pid": 7, "process_name": "python"}]),
        {**_gpu(4), "name": "NVIDIA H100 80GB HBM3"},
    ]

    assert remote.strict_clean_a100s(rows) == [rows[0]]


def test_committed_source_archive_contains_only_qualification_sources(
    tmp_path: Path,
) -> None:
    for relative in remote.SOURCE_FILES:
        path = tmp_path / relative
        if Path(relative).suffix:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(relative + "\n")
        else:
            path.mkdir(parents=True, exist_ok=True)
            (path / "__init__.py").write_text(relative + "\n")
    unrelated = tmp_path / "experiments/raw.trace"
    unrelated.parent.mkdir(parents=True)
    unrelated.write_text("large\n")
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(
        ["git", "add", "--", "."],
        cwd=tmp_path,
        check=True,
    )
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=Remote Test",
            "-c",
            "user.email=remote@example.invalid",
            "commit",
            "-qm",
            "fixture",
        ],
        cwd=tmp_path,
        check=True,
    )
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        check=True,
    ).stdout.strip()

    payload = remote.committed_source_archive(tmp_path, commit)

    with tarfile.open(fileobj=io.BytesIO(payload), mode="r:") as bundle:
        names = {member.name for member in bundle.getmembers()}
    assert "source/experiments/raw.trace" not in names
    assert all(
        any(
            name == "source/" + relative
            or name.startswith("source/" + relative + "/")
            for name in names
        )
        for relative in remote.SOURCE_FILES
    )


def test_worker_plan_runs_profiler_and_remote_verifier_mounted_only() -> None:
    paths = remote.build_remote_paths("20260913-stage0-plan-r1")
    plan = remote.build_worker_plan(
        paths=paths,
        run_tag="20260913-stage0-plan-r1",
        source_commit="a" * 40,
        gpu=_gpu(2),
    )
    joined = "\n".join(plan["commands"])

    assert "tools.profile_slo_cohort_burst_ceiling" in joined
    assert "--mode run" in joined
    assert "--mode verify" in joined
    assert "CUDA_VISIBLE_DEVICES=2" in joined
    assert paths["primary"] in joined
    assert paths["controller"] in joined
    assert "export TMPDIR=/tmp" not in joined
    assert "cd /tmp/" not in joined
    assert "/private/tmp" not in joined
    assert "TinyLLMForge-adaptive-ngram" not in joined
    assert "pkill" not in joined
    assert "killall" not in joined


@pytest.mark.parametrize(
    ("name", "expected"),
    (
        ("raw_rows.jsonl", True),
        ("cost_table.json", True),
        ("ceiling_summary.json", True),
        ("source_manifest.json", True),
        ("remote_verify.json", True),
        ("runner.log", True),
        ("runtime/hf-cache/blob", False),
        ("raw/nsys.sqlite", False),
    ),
)
def test_compact_artifact_allowlist(name: str, expected: bool) -> None:
    assert remote.is_compact_artifact(name) is expected


def test_resume_receipt_requires_exact_source_and_terminal_hashes() -> None:
    paths = remote.build_remote_paths("20260913-stage0-resume-r1")
    receipt = {
        "schema_version": "slo-cohort-burst.remote-resume.v1",
        "status": "COMPLETE",
        "run_tag": "20260913-stage0-resume-r1",
        "source_commit": "a" * 40,
        "remote_paths": paths,
        "artifact_sha256": {
            name: str(index) * 64
            for index, name in enumerate(
                sorted(remote.REQUIRED_TERMINAL_FILES),
                start=1,
            )
        },
    }

    assert remote.validate_resume_receipt(
        receipt,
        run_tag="20260913-stage0-resume-r1",
        source_commit="a" * 40,
        paths=paths,
    )["status"] == "COMPLETE"

    mutated = json.loads(json.dumps(receipt))
    mutated["source_commit"] = "b" * 40
    with pytest.raises(ValueError, match="source identity"):
        remote.validate_resume_receipt(
            mutated,
            run_tag="20260913-stage0-resume-r1",
            source_commit="a" * 40,
            paths=paths,
        )


def test_wait_for_clean_a100_retries_transient_remote_failure(
    monkeypatch,
) -> None:
    calls = []
    selected = _gpu(3)
    monkeypatch.setattr(
        remote,
        "validate_kerberos",
        lambda **_kwargs: calls.append("kerberos"),
    )

    def query():
        calls.append("query")
        if calls.count("query") == 1:
            raise RuntimeError("transient SSH failure")
        return [selected]

    monkeypatch.setattr(remote.base, "query_remote_gpu_rows", query)
    monkeypatch.setattr(remote.time, "sleep", lambda _seconds: None)
    monotonic = iter((0.0, 0.0, 0.0)).__next__
    monkeypatch.setattr(remote.time, "monotonic", monotonic)

    inventory, gpu = remote.wait_for_clean_a100(
        timeout_seconds=60,
        poll_interval_seconds=1,
    )

    assert inventory == [selected]
    assert gpu == selected
    assert calls == ["kerberos", "query", "kerberos", "query"]


def test_worker_plan_seals_terminal_hashes_for_immutable_resume() -> None:
    paths = remote.build_remote_paths("20260913-stage0-seal-r1")
    plan = remote.build_worker_plan(
        paths=paths,
        run_tag="20260913-stage0-seal-r1",
        source_commit="a" * 40,
        gpu=_gpu(1),
    )
    joined = "\n".join(plan["commands"])

    assert paths["controller"] in joined
    assert "resume.json" in joined
    assert "slo-cohort-burst.remote-resume.v1" in joined
    assert all(name in joined for name in remote.REQUIRED_TERMINAL_FILES)


def _write_compact_bundle(path: Path) -> dict[str, str]:
    path.mkdir(parents=True)
    for name in remote.REQUIRED_TERMINAL_FILES:
        (path / name).write_text("{}\n", encoding="utf-8")
    (path / "runner.log").write_text("complete\n", encoding="utf-8")
    return {
        name: __import__("hashlib").sha256(
            (path / name).read_bytes()
        ).hexdigest()
        for name in remote.REQUIRED_TERMINAL_FILES
    }


def test_controller_fresh_run_executes_source_exact_flow(
    monkeypatch,
    tmp_path: Path,
) -> None:
    calls = []
    paths = remote.build_remote_paths("20260913-stage0-fresh-r1")
    selected = _gpu(2)
    receipt = {
        "schema_version": "slo-cohort-burst.remote-resume.v1",
        "status": "COMPLETE",
        "run_tag": "20260913-stage0-fresh-r1",
        "source_commit": "a" * 40,
        "remote_paths": paths,
        "artifact_sha256": {},
    }
    monkeypatch.setattr(
        remote,
        "require_pushed_head",
        lambda _root: calls.append("head") or "a" * 40,
    )
    monkeypatch.setattr(
        remote,
        "validate_kerberos",
        lambda **_kwargs: calls.append("kerberos") or {"status": "PASS"},
    )
    resume_results = iter((None, receipt))

    def probe_resume(**_kwargs):
        calls.append("resume")
        return next(resume_results)

    monkeypatch.setattr(remote, "probe_resume_receipt", probe_resume)
    monkeypatch.setattr(
        remote,
        "committed_source_archive",
        lambda *_args: calls.append("archive") or b"tar",
    )
    monkeypatch.setattr(
        remote,
        "upload_source_archive",
        lambda **_kwargs: calls.append("upload") or paths["staging"] + "/source",
    )
    monkeypatch.setattr(
        remote,
        "wait_for_clean_a100",
        lambda **_kwargs: calls.append("wait") or ([selected], selected),
    )
    monkeypatch.setattr(
        remote,
        "validate_selected_gpu_still_clean",
        lambda gpu: calls.append("recheck") or gpu,
    )
    monkeypatch.setattr(
        remote,
        "run_worker_plan",
        lambda _plan: calls.append("worker") or {"status": "COMPLETE"},
    )

    def download(**_kwargs):
        calls.append("download")
        destination = tmp_path / "20260913-stage0-fresh-r1"
        receipt["artifact_sha256"] = _write_compact_bundle(
            destination
        )
        return destination

    monkeypatch.setattr(remote, "download_compact_bundle", download)
    monkeypatch.setattr(
        remote,
        "verify_local_bundle",
        lambda _path: calls.append("verify") or {
            "verified": True,
            "classification": "CONTINUE_RUNTIME",
        },
    )
    monkeypatch.setattr(
        remote,
        "write_local_controller_receipt",
        lambda **_kwargs: calls.append("receipt") or (
            tmp_path / "controller.json"
        ),
    )
    args = SimpleNamespace(
        stage="ceiling",
        tag="20260913-stage0-fresh-r1",
        source_commit=None,
        local_artifact_root=str(tmp_path),
        gpu_timeout_seconds=60,
        poll_interval_seconds=1,
    )

    result = remote.run_controller(args)

    assert result["status"] == "COMPLETE"
    assert result["resumed"] is False
    assert result["classification"] == "CONTINUE_RUNTIME"
    assert calls == [
        "head",
        "kerberos",
        "resume",
        "archive",
        "upload",
        "wait",
        "kerberos",
        "recheck",
        "worker",
        "resume",
        "download",
        "verify",
        "receipt",
    ]


def test_controller_valid_resume_skips_upload_gpu_and_worker(
    monkeypatch,
    tmp_path: Path,
) -> None:
    calls = []
    tag = "20260913-stage0-resume-r2"
    paths = remote.build_remote_paths(tag)
    receipt = {
        "schema_version": "slo-cohort-burst.remote-resume.v1",
        "status": "COMPLETE",
        "run_tag": tag,
        "source_commit": "a" * 40,
        "remote_paths": paths,
        "artifact_sha256": {},
    }
    monkeypatch.setattr(
        remote,
        "require_pushed_head",
        lambda _root: "a" * 40,
    )
    monkeypatch.setattr(
        remote,
        "validate_kerberos",
        lambda **_kwargs: {"status": "PASS"},
    )
    monkeypatch.setattr(
        remote,
        "probe_resume_receipt",
        lambda **_kwargs: calls.append("resume") or receipt,
    )
    monkeypatch.setattr(
        remote,
        "committed_source_archive",
        lambda *_args: pytest.fail("archive must be skipped"),
    )
    monkeypatch.setattr(
        remote,
        "wait_for_clean_a100",
        lambda **_kwargs: pytest.fail("GPU wait must be skipped"),
    )
    monkeypatch.setattr(
        remote,
        "run_worker_plan",
        lambda _plan: pytest.fail("worker must be skipped"),
    )

    def download(**_kwargs):
        calls.append("download")
        destination = tmp_path / tag
        receipt["artifact_sha256"] = _write_compact_bundle(
            destination
        )
        return destination

    monkeypatch.setattr(remote, "download_compact_bundle", download)
    monkeypatch.setattr(
        remote,
        "verify_local_bundle",
        lambda _path: calls.append("verify") or {
            "verified": True,
            "classification": "NO_GO_CEILING",
        },
    )
    monkeypatch.setattr(
        remote,
        "write_local_controller_receipt",
        lambda **_kwargs: tmp_path / "controller.json",
    )

    result = remote.run_controller(SimpleNamespace(
        stage="ceiling",
        tag=tag,
        source_commit=None,
        local_artifact_root=str(tmp_path),
        gpu_timeout_seconds=60,
        poll_interval_seconds=1,
    ))

    assert result["resumed"] is True
    assert result["classification"] == "NO_GO_CEILING"
    assert calls == ["resume", "download", "verify"]
