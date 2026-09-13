from __future__ import annotations

import io
import json
from pathlib import Path
import subprocess
import sys
import tarfile

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
