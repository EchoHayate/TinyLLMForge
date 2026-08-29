from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from tools.cross_engine_k8_contract import (
    GIB,
    HARD_STOP_BYTES,
    LOCAL_ALLOWLIST,
    MODEL_PATH,
    REMOTE_ROOT,
    REQUIRED_CACHE_VARIABLES,
    CampaignPaths,
    cache_environment,
    classify_allocated_bytes,
    parse_klist_lifetime,
    require_kerberos_coverage,
    validate_attempt_tag,
    validate_local_allowlist,
)


def test_campaign_paths_accept_owned_attempt_path():
    paths = CampaignPaths.create(
        remote_root=REMOTE_ROOT,
        model_path=MODEL_PATH,
    )

    accepted = paths.require_owned_remote(
        f"{REMOTE_ROOT}/attempts/20260829-cross-engine-k8-qwen3-06b-r1"
    )

    assert accepted.as_posix().startswith(f"{REMOTE_ROOT}/attempts/")


@pytest.mark.parametrize(
    "forbidden",
    (
        "/",
        "/tmp/run",
        "/data00/home/sitian/TinyLLMForge/run",
        f"{REMOTE_ROOT}/shared/model-copy",
        f"{MODEL_PATH}/config.json",
    ),
)
def test_campaign_paths_reject_forbidden_or_model_paths(forbidden):
    paths = CampaignPaths.create(
        remote_root=REMOTE_ROOT,
        model_path=MODEL_PATH,
    )

    with pytest.raises(ValueError):
        paths.require_owned_remote(forbidden)


def test_campaign_paths_reject_noncanonical_segments():
    paths = CampaignPaths.create(
        remote_root=REMOTE_ROOT,
        model_path=MODEL_PATH,
    )

    with pytest.raises(ValueError, match="canonical"):
        paths.require_owned_remote(f"{REMOTE_ROOT}/attempts/../escape")


def test_cache_environment_redirects_every_cache_without_home():
    env = cache_environment(
        CampaignPaths.create(
            remote_root=REMOTE_ROOT,
            model_path=MODEL_PATH,
        )
    )

    assert set(env) == set(REQUIRED_CACHE_VARIABLES)
    assert "HOME" not in env
    assert all(value.startswith(REMOTE_ROOT + "/") for value in env.values())


@pytest.mark.parametrize(
    ("bytes_used", "expected"),
    (
        (16 * GIB - 1, "OK"),
        (16 * GIB, "WARNING"),
        (20 * GIB - 1, "WARNING"),
        (20 * GIB, "HARD_STOP"),
        (HARD_STOP_BYTES + 1, "HARD_STOP"),
    ),
)
def test_storage_boundaries(bytes_used, expected):
    assert classify_allocated_bytes(bytes_used) == expected


def test_storage_rejects_negative_or_boolean_values():
    for invalid in (-1, True):
        with pytest.raises(ValueError, match="bytes_used"):
            classify_allocated_bytes(invalid)


def test_parse_klist_lifetime_uses_latest_tgt_expiry():
    now = datetime(2026, 8, 29, 18, 0, tzinfo=timezone.utc)
    text = """
Credentials cache: FILE:/Users/bytedance/krb5cc_sitian
        Principal: sitian@BYTEDANCE.COM

  Issued                Expires               Principal
Aug 29 17:43:26 2026  Aug 30 03:43:21 2026  krbtgt/BYTEDANCE.COM@BYTEDANCE.COM
"""

    lifetime = parse_klist_lifetime(text, now=now)

    assert lifetime == timedelta(hours=9, minutes=43, seconds=21)


def test_parse_klist_rejects_missing_tgt():
    with pytest.raises(RuntimeError, match="KERBEROS_TGT_NOT_FOUND"):
        parse_klist_lifetime(
            "Credentials cache: FILE:/x\n",
            now=datetime(2026, 8, 29, tzinfo=timezone.utc),
        )


def test_kerberos_guard_requires_estimate_plus_thirty_minutes():
    with pytest.raises(RuntimeError, match="KERBEROS_TTL_INSUFFICIENT"):
        require_kerberos_coverage(
            lifetime=timedelta(hours=2),
            estimated=timedelta(hours=1, minutes=31),
            margin=timedelta(minutes=30),
        )


def test_kerberos_guard_accepts_exact_boundary():
    require_kerberos_coverage(
        lifetime=timedelta(hours=2),
        estimated=timedelta(hours=1, minutes=30),
        margin=timedelta(minutes=30),
    )


@pytest.mark.parametrize(
    "tag",
    (
        "20260829-cross-engine-k8-qwen3-06b-r1",
        "20260829-cross-engine-k8-qwen3-06b-r19",
    ),
)
def test_attempt_tag_accepts_frozen_shape(tag):
    assert validate_attempt_tag(tag) == tag


@pytest.mark.parametrize(
    "tag",
    (
        "20260829-cross-engine-k8-qwen3-06b-r0",
        "20260830-cross-engine-k8-qwen3-06b-r1",
        "../20260829-cross-engine-k8-qwen3-06b-r1",
    ),
)
def test_attempt_tag_rejects_non_frozen_shape(tag):
    with pytest.raises(ValueError, match="attempt tag"):
        validate_attempt_tag(tag)


def test_local_allowlist_accepts_exact_files_and_reports_size(tmp_path):
    for name in LOCAL_ALLOWLIST:
        (tmp_path / name).write_text("{}\n", encoding="utf-8")

    result = validate_local_allowlist(tmp_path)

    assert result["files"] == list(LOCAL_ALLOWLIST)
    assert result["total_bytes"] == 3 * len(LOCAL_ALLOWLIST)


def test_local_allowlist_rejects_extra_file(tmp_path):
    for name in LOCAL_ALLOWLIST:
        (tmp_path / name).write_text("{}\n", encoding="utf-8")
    (tmp_path / "worker.log").write_text("large log", encoding="utf-8")

    with pytest.raises(RuntimeError, match="LOCAL_ALLOWLIST_VIOLATION"):
        validate_local_allowlist(tmp_path)


def test_local_allowlist_rejects_oversize_file(tmp_path, monkeypatch):
    for name in LOCAL_ALLOWLIST:
        (tmp_path / name).write_text("{}\n", encoding="utf-8")
    real_stat = Path.stat

    def fake_stat(path):
        result = real_stat(path)
        if path.name == LOCAL_ALLOWLIST[0]:
            values = list(result)
            values[6] = 50 * 1024**2
            return type(result)(values)
        return result

    monkeypatch.setattr(Path, "stat", fake_stat)

    with pytest.raises(RuntimeError, match="LOCAL_STORAGE_HARD_STOP"):
        validate_local_allowlist(tmp_path)
