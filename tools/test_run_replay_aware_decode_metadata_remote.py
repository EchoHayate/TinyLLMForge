"""Safety contracts for the replay-aware metadata remote controller."""

from __future__ import annotations

from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
from zoneinfo import ZoneInfo

import pytest

_REPO_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from tools import run_replay_aware_decode_metadata_remote as remote


def _gpu_row(
    index: int,
    *,
    memory_used_mib: int = 0,
    utilization_percent: int = 0,
    compute_processes=None,
) -> dict:
    return {
        "index": index,
        "uuid": f"GPU-{index}",
        "name": "NVIDIA H100 80GB HBM3",
        "memory_used_mib": memory_used_mib,
        "utilization_percent": utilization_percent,
        "compute_processes": (
            []
            if compute_processes is None
            else compute_processes
        ),
    }


def test_remote_paths_are_confined_to_approved_root():
    paths = remote.remote_paths(
        "20260822-qwen3-06b-replay-meta-r1"
    )
    assert set(paths) == {
        "staging",
        "primary",
        "controller",
    }
    assert all(
        path.startswith(remote.APPROVED_ROOT + "/")
        for path in paths.values()
    )
    assert all("/tmp" not in path for path in paths.values())
    assert all(
        "/private/tmp" not in path
        for path in paths.values()
    )
    assert all(
        "/data00/home/sitian/tllm/TinyLLMForge"
        not in path
        for path in paths.values()
    )


def test_gpu_admission_requires_strict_clean_state():
    rows = [
        _gpu_row(
            0,
            memory_used_mib=1024,
            utilization_percent=5,
        ),
        _gpu_row(1, memory_used_mib=1025),
        _gpu_row(2, utilization_percent=6),
        _gpu_row(
            3,
            compute_processes=[
                {
                    "pid": 123,
                    "process_name": "python",
                }
            ],
        ),
    ]
    assert remote.strict_clean_gpus(rows) == [rows[0]]


def test_run_tag_and_destinations_are_immutable():
    with TemporaryDirectory() as tmp:
        local_root = Path(tmp)
        remote.ensure_local_destination_absent(
            local_root,
            "fresh-tag",
        )
        (local_root / "used-tag").mkdir()
        with pytest.raises(
            ValueError,
            match="local run tag already exists",
        ):
            remote.ensure_local_destination_absent(
                local_root,
                "used-tag",
            )
    for tag in (
        "",
        "../escape",
        "nested/tag",
        "-leading",
        "white space",
    ):
        with pytest.raises(ValueError):
            remote.remote_paths(tag)


def test_kerberos_guard_rejects_ttl_below_5400():
    now = datetime(
        2026,
        8,
        22,
        12,
        0,
        0,
        tzinfo=ZoneInfo("Asia/Shanghai"),
    )
    output = "\n".join((
        "Credentials cache: "
        "FILE:/Users/bytedance/krb5cc_sitian",
        "Principal: sitian@BYTESEE.NET",
        (
            "Aug 22 11:00:00 2026  "
            "Aug 22 13:00:00 2026  "
            "krbtgt/BYTESEE.NET@BYTESEE.NET"
        ),
    ))

    def runner(*_args, **_kwargs):
        return subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout=output,
            stderr="",
        )

    with pytest.raises(
        ValueError,
        match="remaining lifetime",
    ):
        remote.validate_kerberos(
            command_runner=runner,
            now=lambda: now,
        )


def test_source_commit_must_match_pushed_head():
    commit = "a" * 40
    assert remote.validate_source_commit(
        commit,
        pushed_head=commit,
    ) == commit
    with pytest.raises(
        ValueError,
        match="requested source commit",
    ):
        remote.validate_source_commit(
            "b" * 40,
            pushed_head=commit,
        )


def test_remote_requirement_receipt_requires_python_and_model():
    valid = {
        "python": {
            "path": remote.REMOTE_PYTHON,
            "is_file": True,
            "is_executable": True,
        },
        "model": {
            "path": remote.MODEL_PATH,
            "is_dir": True,
            "config_is_file": True,
        },
        "approved_root": {
            "path": remote.APPROVED_ROOT,
            "is_dir": True,
            "free_bytes": 100 * 1024**3,
        },
    }
    assert remote.validate_remote_requirements(valid) == valid
    for section, field in (
        ("python", "is_file"),
        ("python", "is_executable"),
        ("model", "is_dir"),
        ("model", "config_is_file"),
    ):
        invalid = json.loads(json.dumps(valid))
        invalid[section][field] = False
        with pytest.raises(
            ValueError,
            match="remote requirements",
        ):
            remote.validate_remote_requirements(invalid)


def test_download_inventory_requires_terminal_bundle():
    required = [
        {"path": name}
        for name in remote.REQUIRED_PRIMARY_FILES
    ]
    remote.validate_terminal_download_inventory(required)
    with pytest.raises(
        ValueError,
        match="download is incomplete",
    ):
        remote.validate_terminal_download_inventory(
            required[:-1]
        )


def test_remote_preflight_uses_pinned_python():
    commands = remote.preflight_commands()
    assert commands
    assert all(
        command.startswith(remote.REMOTE_PYTHON + " ")
        for command in commands
    )
    assert any(
        "tools/test_chunked_prefill.py" in command
        for command in commands
    )


def test_failed_chunked_download_preserves_partial_evidence():
    with TemporaryDirectory() as tmp:
        destination = Path(tmp) / "download"
        original_inventory = remote.base.fetch_remote_inventory
        original_chunk = remote.base.download_chunk
        calls = []
        try:
            remote.base.fetch_remote_inventory = (
                lambda *_args, **_kwargs: [{
                    "path": "evidence.bin",
                    "size_bytes": 4,
                    "sha256": (
                        "9f64a747e1b97f131fabb6b447296c9b"
                        "6f0201e79fb3c5356e6c77e89b6a806a"
                    ),
                    "chunks": [
                        {
                            "offset": 0,
                            "length": 2,
                            "sha256": (
                                "a12871fee210fb8619291eaea194581c"
                                "bd2531e4b23759d225f6806923f63222"
                            ),
                        },
                        {
                            "offset": 2,
                            "length": 2,
                            "sha256": "0" * 64,
                        },
                    ],
                }]
            )

            def fake_chunk(
                _path,
                *,
                offset,
                **_kwargs,
            ):
                calls.append(offset)
                if offset == 0:
                    return b"\x00\x01"
                raise RuntimeError("network interruption")

            remote.base.download_chunk = fake_chunk
            with pytest.raises(
                RuntimeError,
                match="artifact chunk download failed",
            ):
                remote.download_remote_tree_preserving_partial(
                    remote.APPROVED_ROOT + "/fixture",
                    destination,
                    retries=1,
                )
        finally:
            remote.base.fetch_remote_inventory = original_inventory
            remote.base.download_chunk = original_chunk

        partial = destination.with_name(
            destination.name + ".partial"
        )
        assert calls == [0, 2]
        assert partial.is_dir()
        assert (partial / "evidence.bin").read_bytes() == (
            b"\x00\x01"
        )


def test_controller_source_has_no_destructive_gpu_commands():
    source = Path(remote.__file__).read_text(
        encoding="utf-8"
    )
    for forbidden in (
        "pkill",
        "killall",
        "nvidia-smi --gpu-reset",
        "rm -rf",
    ):
        assert forbidden not in source


def main() -> None:
    test_remote_paths_are_confined_to_approved_root()
    test_gpu_admission_requires_strict_clean_state()
    test_run_tag_and_destinations_are_immutable()
    test_kerberos_guard_rejects_ttl_below_5400()
    test_source_commit_must_match_pushed_head()
    test_remote_requirement_receipt_requires_python_and_model()
    test_download_inventory_requires_terminal_bundle()
    test_remote_preflight_uses_pinned_python()
    test_failed_chunked_download_preserves_partial_evidence()
    test_controller_source_has_no_destructive_gpu_commands()
    print("replay-aware metadata remote tests passed")


if __name__ == "__main__":
    main()
