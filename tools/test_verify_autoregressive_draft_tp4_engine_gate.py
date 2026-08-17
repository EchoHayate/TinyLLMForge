from __future__ import annotations

import importlib.util
import io
import json
from pathlib import Path
import tarfile

import pytest

from tools.autoregressive_draft_tp4_engine_gate import (
    DEFAULT_SOURCE_FILES,
    build_source_archive,
    hash_source_files,
    publish_authority_bundle,
    safe_extract_source_archive,
    sha256_file,
    source_tree_sha256,
)
from tools.test_autoregressive_draft_tp4_engine_gate import _payload


REPO_ROOT = Path(__file__).resolve().parents[1]
VERIFIER_PATH = (
    REPO_ROOT
    / "tools"
    / "verify_autoregressive_draft_tp4_engine_gate.py"
)


def _load_verifier():
    assert VERIFIER_PATH.is_file()
    spec = importlib.util.spec_from_file_location(
        "verify_autoregressive_draft_tp4_engine_gate_test",
        VERIFIER_PATH,
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_json(path, value):
    path.write_text(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


def _write_verifier_run(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    result_path = run_dir / "result.json"
    _write_json(result_path, _payload())
    archive_path = run_dir / "source.tar"
    build_source_archive(
        REPO_ROOT,
        archive_path,
        DEFAULT_SOURCE_FILES,
    )
    source_hashes = hash_source_files(
        REPO_ROOT,
        DEFAULT_SOURCE_FILES,
    )
    _write_json(
        run_dir / "source_manifest.json",
        {
            "schema_version": 2,
            "source_tree_sha256": source_tree_sha256(
                REPO_ROOT,
                DEFAULT_SOURCE_FILES,
            ),
            "source_files": source_hashes,
            "artifacts": {
                "result.json": sha256_file(result_path),
                "source.tar": sha256_file(archive_path),
            },
        },
    )
    return run_dir


def test_source_inventory_is_sorted_explicit_and_complete():
    assert DEFAULT_SOURCE_FILES == tuple(
        sorted(DEFAULT_SOURCE_FILES)
    )
    assert len(DEFAULT_SOURCE_FILES) == len(
        set(DEFAULT_SOURCE_FILES)
    )
    assert (
        "tools/verify_autoregressive_draft_tp4_engine_gate.py"
        in DEFAULT_SOURCE_FILES
    )
    assert (
        "tinyvllm/engine/autoregressive_draft_executor.py"
        in DEFAULT_SOURCE_FILES
    )
    for name in DEFAULT_SOURCE_FILES:
        path = REPO_ROOT / name
        assert path.is_file()
        assert not path.is_symlink()


def test_source_archive_is_byte_deterministic_and_normalized(tmp_path):
    first = tmp_path / "first.tar"
    second = tmp_path / "second.tar"

    build_source_archive(
        REPO_ROOT,
        first,
        DEFAULT_SOURCE_FILES,
    )
    build_source_archive(
        REPO_ROOT,
        second,
        DEFAULT_SOURCE_FILES,
    )

    assert first.read_bytes() == second.read_bytes()
    with tarfile.open(first, "r:") as archive:
        members = archive.getmembers()
        assert tuple(member.name for member in members) == (
            DEFAULT_SOURCE_FILES
        )
        for member in members:
            assert member.isfile()
            assert member.mode == 0o644
            assert member.uid == 0
            assert member.gid == 0
            assert member.uname == ""
            assert member.gname == ""
            assert member.mtime == 0


def test_safe_extract_validates_hashes_before_writing(tmp_path):
    archive_path = tmp_path / "source.tar"
    build_source_archive(
        REPO_ROOT,
        archive_path,
        DEFAULT_SOURCE_FILES,
    )
    destination = tmp_path / "source"
    hashes = hash_source_files(
        REPO_ROOT,
        DEFAULT_SOURCE_FILES,
    )

    extracted = safe_extract_source_archive(
        archive_path,
        destination,
        hashes,
    )

    assert extracted == DEFAULT_SOURCE_FILES
    assert hash_source_files(
        destination,
        DEFAULT_SOURCE_FILES,
    ) == hashes


def _write_single_member_archive(path, names):
    with tarfile.open(path, "w:") as archive:
        for name in names:
            payload = f"{name}\n".encode()
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            info.mode = 0o644
            info.uid = 0
            info.gid = 0
            info.uname = ""
            info.gname = ""
            info.mtime = 0
            archive.addfile(info, io.BytesIO(payload))


@pytest.mark.parametrize(
    ("name", "type_name"),
    (
        ("../escape.py", "regular"),
        ("tools/link.py", "symlink"),
    ),
)
def test_safe_extract_rejects_unsafe_members(
    tmp_path,
    name,
    type_name,
):
    archive_path = tmp_path / "unsafe.tar"
    with tarfile.open(archive_path, "w:") as archive:
        info = tarfile.TarInfo(name)
        if type_name == "symlink":
            info.type = tarfile.SYMTYPE
            info.linkname = "target.py"
            archive.addfile(info)
        else:
            payload = b"x = 1\n"
            info.size = len(payload)
            archive.addfile(info, io.BytesIO(payload))

    with pytest.raises(ValueError, match="unsafe source archive"):
        safe_extract_source_archive(
            archive_path,
            tmp_path / "source",
            {name: "0" * 64},
        )

    assert not (tmp_path / "source").exists()


@pytest.mark.parametrize(
    "archive_names",
    (
        (),
        ("safe.py", "safe.py"),
        ("other.py",),
    ),
)
def test_safe_extract_rejects_missing_duplicate_or_unexpected_members(
    tmp_path,
    archive_names,
):
    archive_path = tmp_path / "invalid.tar"
    _write_single_member_archive(
        archive_path,
        archive_names,
    )

    with pytest.raises(
        ValueError,
        match="unsafe source archive inventory",
    ):
        safe_extract_source_archive(
            archive_path,
            tmp_path / "source",
            {"safe.py": "0" * 64},
        )

    assert not (tmp_path / "source").exists()


def test_safe_extract_rejects_payload_hash_before_writing(tmp_path):
    archive_path = tmp_path / "invalid.tar"
    _write_single_member_archive(
        archive_path,
        ("safe.py",),
    )

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        safe_extract_source_archive(
            archive_path,
            tmp_path / "source",
            {"safe.py": "0" * 64},
        )

    assert not (tmp_path / "source").exists()


def test_independent_verifier_passes_bound_bundle(tmp_path):
    verifier = _load_verifier()
    run_dir = _write_verifier_run(tmp_path)

    receipt = verifier.verify_run(run_dir, REPO_ROOT)

    assert receipt["classification"] == "PASS"
    assert receipt["failures"] == []
    assert receipt["source_tree_sha256"] == source_tree_sha256(
        REPO_ROOT,
        DEFAULT_SOURCE_FILES,
    )


def test_independent_verifier_rejects_result_tamper(tmp_path):
    verifier = _load_verifier()
    run_dir = _write_verifier_run(tmp_path)
    result = json.loads((run_dir / "result.json").read_text())
    result["gate_pass"] = False
    _write_json(run_dir / "result.json", result)

    receipt = verifier.verify_run(run_dir, REPO_ROOT)

    assert receipt["classification"] == "FAIL"
    assert any(
        "gate pass" in failure
        or "result artifact SHA-256" in failure
        for failure in receipt["failures"]
    )


def test_publish_bundle_is_atomic_and_archived_verifier_bound(tmp_path):
    output_dir = tmp_path / "authority"

    receipt = publish_authority_bundle(
        _payload(),
        output_dir,
        source_root=REPO_ROOT,
    )

    assert receipt["classification"] == "PASS"
    assert receipt["receipts_match"] is True
    assert sorted(path.name for path in output_dir.iterdir()) == [
        "result.json",
        "source.tar",
        "source_manifest.json",
        "verify.json",
    ]
    with pytest.raises(FileExistsError, match="already exists"):
        publish_authority_bundle(
            _payload(),
            output_dir,
            source_root=REPO_ROOT,
        )


def test_publish_bundle_preserves_failed_receipt_mismatch(tmp_path):
    output_dir = tmp_path / "authority"
    calls = []
    current_receipt = {
        "schema_version": 1,
        "classification": "PASS",
        "failures": [],
        "result_sha256": "a" * 64,
        "source_tree_sha256": "b" * 64,
    }

    def verifier_runner(verifier_path, run_dir, source_root):
        calls.append((
            Path(verifier_path),
            Path(run_dir),
            Path(source_root),
        ))
        if len(calls) == 1:
            return current_receipt
        return {
            **current_receipt,
            "source_tree_sha256": "c" * 64,
        }

    with pytest.raises(
        RuntimeError,
        match="archived-source independent verification failed",
    ):
        publish_authority_bundle(
            _payload(),
            output_dir,
            source_root=REPO_ROOT,
            verification_runner=verifier_runner,
        )

    assert not output_dir.exists()
    failed_dir = tmp_path / "authority.failed"
    assert failed_dir.is_dir()
    assert (failed_dir / "failure.json").is_file()
    assert (failed_dir / "verify.json").is_file()
    assert calls[0][0] == (
        REPO_ROOT
        / "tools"
        / "verify_autoregressive_draft_tp4_engine_gate.py"
    )
    assert calls[1][0] != calls[0][0]
    assert calls[1][0].name == calls[0][0].name

