from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from tools.cross_engine_k8_environment import (
    build_environment_manifest,
    build_model_inventory,
    candidate_releases,
    compatibility_decision,
)


def _pypi_fixture():
    return {
        "releases": {
            "0.10.1": [{"yanked": False}],
            "0.10.2": [{"yanked": False}],
            "0.11.0rc1": [{"yanked": False}],
            "0.9.9": [{"yanked": True}],
            "garbage": [{"yanked": False}],
        }
    }


def test_candidate_releases_are_stable_and_newest_first():
    releases = candidate_releases(_pypi_fixture())

    assert [item.version for item in releases] == ["0.10.2", "0.10.1"]
    assert releases[0].requirement == "vllm==0.10.2"


def test_first_complete_compatible_release_is_frozen():
    decision = compatibility_decision([
        {"version": "0.10.2", "compatible": False, "reason": "driver"},
        {
            "version": "0.10.1",
            "compatible": True,
            "smoke_output_tokens": 128,
            "public_multi_step": False,
        },
    ])

    assert decision["selected_version"] == "0.10.1"
    assert decision["classification"] == "COMPATIBLE"
    assert decision["multi_step_status"] == (
        "VLLM_MULTI_STEP_NOT_PUBLICLY_AVAILABLE"
    )


def test_compatible_probe_requires_exact_smoke_output_length():
    decision = compatibility_decision([
        {
            "version": "0.10.2",
            "compatible": True,
            "smoke_output_tokens": 127,
            "public_multi_step": True,
        },
    ])

    assert decision["classification"] == "INCOMPLETE_VLLM_COMPATIBILITY"


def test_no_release_yields_incomplete_not_source_patch():
    decision = compatibility_decision([
        {"version": "0.10.2", "compatible": False, "reason": "model"},
    ])

    assert decision["classification"] == "INCOMPLETE_VLLM_COMPATIBILITY"
    assert decision["source_patch_allowed"] is False
    assert decision["selected_version"] is None


def test_model_inventory_is_sorted_and_hashes_files(tmp_path):
    (tmp_path / "sub").mkdir()
    (tmp_path / "z.bin").write_bytes(b"z")
    (tmp_path / "sub" / "a.json").write_bytes(b"a")

    inventory = build_model_inventory(
        tmp_path,
        expected_root=tmp_path.resolve(),
    )

    assert [item["path"] for item in inventory["files"]] == [
        "sub/a.json",
        "z.bin",
    ]
    assert inventory["logical_bytes"] == 2
    assert inventory["files"][0]["sha256"] == hashlib.sha256(b"a").hexdigest()
    assert len(inventory["inventory_sha256"]) == 64


def test_model_inventory_rejects_symlink(tmp_path):
    target = tmp_path / "target"
    target.write_bytes(b"x")
    (tmp_path / "link").symlink_to(target)

    with pytest.raises(ValueError, match="symlink"):
        build_model_inventory(
            tmp_path,
            expected_root=tmp_path.resolve(),
        )


def test_environment_manifest_binds_source_model_and_packages():
    manifest = build_environment_manifest(
        tinyllmforge_probe={
            "python_version": "3.10.14",
            "packages": {"torch": "2.6.0"},
        },
        vllm_probe={
            "version": "0.10.1",
            "python_version": "3.10.14",
            "packages": {"vllm": "0.10.1"},
            "public_multi_step": False,
        },
        model_inventory={"inventory_sha256": "b" * 64},
        source_revision="a" * 40,
    )

    assert manifest["source_revision"] == "a" * 40
    assert manifest["model_inventory_sha256"] == "b" * 64
    assert manifest["vllm"]["version"] == "0.10.1"
    assert len(manifest["manifest_sha256"]) == 64


@pytest.mark.parametrize("source", ("a" * 39, "g" * 40))
def test_environment_manifest_rejects_bad_source_revision(source):
    with pytest.raises(ValueError, match="source_revision"):
        build_environment_manifest(
            tinyllmforge_probe={},
            vllm_probe={},
            model_inventory={"inventory_sha256": "b" * 64},
            source_revision=source,
        )
