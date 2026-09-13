from __future__ import annotations

import pytest

from tools import run_slo_cohort_burst_remote as remote


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

