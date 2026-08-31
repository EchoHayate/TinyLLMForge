from __future__ import annotations

import argparse

import pytest

from tools.run_quantized_draft_int4_microgate import (
    APPROVED_REMOTE_ROOT,
    build_remote_commands,
    build_run_plan,
    classify_preflight,
    download_inventory,
    run_controller,
)
import tools.run_quantized_draft_int4_microgate as controller


def _ready_payload():
    return {
        "python_exists": True,
        "draft_model_exists": True,
        "remote_root_exists": True,
        "exact_tag_exists": False,
        "exact_tag_processes": [],
        "gpus": [
            {
                "index": index,
                "uuid": f"GPU-{index}",
                "name": "NVIDIA A100-SXM4-80GB",
                "memory_used_mib": 0,
                "utilization_percent": 0,
                "compute_process_count": 0,
            }
            for index in range(8)
        ],
    }


def _args(tmp_path, **overrides):
    values = {
        "run_tag": "fixture-r1",
        "repo_root": tmp_path,
        "local_artifact_root": tmp_path / "artifacts",
        "dry_run": False,
        "execute": True,
        "poll_seconds": 1,
        "max_wait_seconds": 1,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def test_plan_rejects_any_remote_path_outside_approved_root():
    with pytest.raises(ValueError, match="approved remote root"):
        build_run_plan(
            run_tag="fixture-r1",
            source_revision="a" * 40,
            remote_root="/tmp/not-approved",
        )


def test_plan_uses_only_approved_remote_paths():
    plan = build_run_plan(
        run_tag="fixture-r1",
        source_revision="a" * 40,
    )

    assert plan["remote_run"].startswith(APPROVED_REMOTE_ROOT + "/")
    assert plan["remote_source"].startswith(APPROVED_REMOTE_ROOT + "/")
    assert plan["remote_cache"].startswith(APPROVED_REMOTE_ROOT + "/")


def test_preflight_requires_one_clean_a100_and_does_not_require_four():
    payload = _ready_payload()
    for row in payload["gpus"][1:]:
        row["compute_process_count"] = 2

    result = classify_preflight(payload)

    assert result["status"] == "READY"
    assert result["selected_gpu_index"] == 0


def test_preflight_never_selects_an_occupied_gpu():
    payload = _ready_payload()
    for row in payload["gpus"]:
        row["compute_process_count"] = 1

    assert classify_preflight(payload)["status"] == "WAIT_GPU"


def test_preflight_rejects_non_a100_or_existing_tag():
    payload = _ready_payload()
    for row in payload["gpus"]:
        row["name"] = "NVIDIA H100 80GB HBM3"
    assert classify_preflight(payload)["status"] == "WAIT_GPU"

    payload = _ready_payload()
    payload["exact_tag_exists"] = True
    assert classify_preflight(payload)["status"] == "INCONCLUSIVE_ENVIRONMENT"


def test_low_kerberos_ttl_fails_before_source_upload(
    tmp_path,
    monkeypatch,
):
    calls = []
    monkeypatch.setattr(
        controller,
        "_validate_local_source",
        lambda *args, **kwargs: "a" * 40,
    )
    monkeypatch.setattr(
        controller,
        "_kerberos_preflight",
        lambda *args, **kwargs: {
            "status": "INCONCLUSIVE_ENVIRONMENT",
            "remaining_lifetime_seconds": 5399,
        },
    )
    monkeypatch.setattr(
        controller,
        "_upload_source",
        lambda *args, **kwargs: calls.append("upload"),
    )

    result = run_controller(_args(tmp_path))

    assert result != 0
    assert calls == []


def test_controller_does_not_emit_kinit_krenew_or_remote_tmp():
    commands = build_remote_commands(
        build_run_plan(
            run_tag="fixture-r1",
            source_revision="a" * 40,
        ),
        selected_gpu_index=3,
    )
    encoded = "\n".join(commands)

    assert "kinit" not in encoded
    assert "krenew" not in encoded
    assert "/tmp" not in encoded
    assert "CUDA_VISIBLE_DEVICES=3" in encoded


def test_ssh_commands_reuse_the_existing_control_master():
    command = controller._ssh(["hostname"])

    assert command[:3] == [
        "ssh",
        "-S",
        "/tmp/ssh-sitian-10.232.195.203",
    ]


def test_download_inventory_contains_only_compact_artifacts():
    assert download_inventory("fixture-r1") == (
        "controller",
        "final_bundle",
    )
