from __future__ import annotations

import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path


THIS_DIR = Path(__file__).resolve().parent
AUDIT_PATH = THIS_DIR / "source_audit.py"
SPEC = importlib.util.spec_from_file_location(
    "source_audit_under_test",
    os.fspath(AUDIT_PATH),
)
audit = importlib.util.module_from_spec(SPEC)
sys.modules["source_audit_under_test"] = audit
SPEC.loader.exec_module(audit)

OWNED_ROOTS = (
    "tinyvllm",
    "tools/profile_ngram_commit.py",
    "tools/speculation_router_gate.py",
)
IGNORED_PREFIXES = (
    "experiments/speculation_router",
)


def _run(command: list[str], cwd: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        command,
        cwd=cwd,
        text=True,
        capture_output=True,
        check=True,
    )


def _source_repo() -> tuple[tempfile.TemporaryDirectory, Path]:
    temporary = tempfile.TemporaryDirectory()
    root = Path(temporary.name)
    (root / ".gitignore").write_text(
        "__pycache__/\n",
        encoding="utf-8",
    )
    (root / "tinyvllm").mkdir()
    (root / "tinyvllm" / "__init__.py").write_text(
        "VALUE = 1\n",
        encoding="utf-8",
    )
    (root / "tools").mkdir()
    (root / "tools" / "profile_ngram_commit.py").write_text(
        "PROFILE = 1\n",
        encoding="utf-8",
    )
    (root / "tools" / "speculation_router_gate.py").write_text(
        "GATE = 1\n",
        encoding="utf-8",
    )
    _run(["git", "init"], root)
    _run(["git", "config", "user.name", "Gate Test"], root)
    _run(["git", "config", "user.email", "gate@example.invalid"], root)
    _run(["git", "config", "gc.auto", "0"], root)
    _run(["git", "config", "maintenance.auto", "false"], root)
    _run(["git", "add", "."], root)
    _run(["git", "commit", "-m", "base"], root)
    return temporary, root


def _cleanup_repo(
    temporary: tempfile.TemporaryDirectory,
    root: Path,
) -> None:
    for attempt in range(10):
        if (root / ".git").exists():
            subprocess.run(
                ["git", "maintenance", "stop"],
                cwd=root,
                text=True,
                capture_output=True,
                check=False,
            )
            try:
                shutil.rmtree(root / ".git")
            except FileNotFoundError:
                pass
            except OSError:
                pass
        try:
            temporary.cleanup()
            return
        except OSError:
            if attempt == 9:
                raise
            time.sleep(0.05)


def _build(root: Path, out_dir: Path) -> dict:
    return audit.build_source_evidence(
        root,
        out_dir,
        owned_roots=OWNED_ROOTS,
        ignored_untracked_prefixes=IGNORED_PREFIXES,
    )


def test_generic_source_evidence_reconstructs_dirty_tree():
    temporary, root = _source_repo()
    try:
        target = root / "tools" / "profile_ngram_commit.py"
        target.write_text("PROFILE = 2\n", encoding="utf-8")
        out_dir = root / "snapshot"
        evidence = _build(root, out_dir)
        assert evidence["owned_roots"] == list(OWNED_ROOTS)
        assert evidence["dirty"] is True
        assert evidence["tree_sha256"] == audit.source_tree_sha256(
            evidence["files"]
        )
        reconstructed = root / "reconstructed"
        audit.reconstruct_source_snapshot(
            root,
            reconstructed,
            evidence,
            out_dir / "source.patch",
            expected_owned_roots=OWNED_ROOTS,
        )
        audit.validate_source_snapshot(
            reconstructed,
            evidence,
            out_dir / "source.patch",
            expected_owned_roots=OWNED_ROOTS,
        )
        assert (
            reconstructed / "tools" / "profile_ngram_commit.py"
        ).read_text(encoding="utf-8") == "PROFILE = 2\n"
    finally:
        _cleanup_repo(temporary, root)


def test_generic_source_evidence_ignores_only_configured_artifacts():
    temporary, root = _source_repo()
    try:
        ignored = (
            root
            / "experiments"
            / "speculation_router"
            / "run"
            / "summary.json"
        )
        ignored.parent.mkdir(parents=True)
        ignored.write_text("{}\n", encoding="utf-8")
        evidence = _build(root, root / "snapshot")
        assert evidence["dirty"] is False

        unrelated = root / "experiments" / "other" / "summary.json"
        unrelated.parent.mkdir(parents=True)
        unrelated.write_text("{}\n", encoding="utf-8")
        try:
            _build(root, root / "snapshot-two")
        except ValueError as exc:
            assert "untracked path outside owned source boundary" in str(exc)
        else:
            raise AssertionError("unrelated untracked artifact must fail")
    finally:
        _cleanup_repo(temporary, root)


def test_generic_source_evidence_rejects_untracked_owned_file():
    temporary, root = _source_repo()
    try:
        (root / "tinyvllm" / "untracked.py").write_text(
            "UNTRACKED = True\n",
            encoding="utf-8",
        )
        try:
            _build(root, root / "snapshot")
        except ValueError as exc:
            assert "untracked owned source" in str(exc)
        else:
            raise AssertionError("untracked owned source must fail")
    finally:
        _cleanup_repo(temporary, root)


def test_generic_source_evidence_excludes_ignored_files_under_owned_root():
    temporary, root = _source_repo()
    try:
        cache = root / "tinyvllm" / "__pycache__" / "module.pyc"
        cache.parent.mkdir()
        cache.write_bytes(b"ignored bytecode")
        evidence = _build(root, root / "snapshot")
        assert [
            record["path"] for record in evidence["files"]
        ] == [
            "tinyvllm/__init__.py",
            "tools/profile_ngram_commit.py",
            "tools/speculation_router_gate.py",
        ]
    finally:
        _cleanup_repo(temporary, root)


def test_generic_snapshot_rejects_patch_and_file_tampering():
    temporary, root = _source_repo()
    try:
        target = root / "tools" / "profile_ngram_commit.py"
        target.write_text("PROFILE = 2\n", encoding="utf-8")
        out_dir = root / "snapshot"
        evidence = _build(root, out_dir)
        patch = out_dir / "source.patch"
        original_patch = patch.read_bytes()
        patch.write_bytes(original_patch + b"x")
        try:
            audit.validate_source_snapshot(
                out_dir / "source",
                evidence,
                patch,
                expected_owned_roots=OWNED_ROOTS,
            )
        except ValueError as exc:
            assert "patch size mismatch" in str(exc)
        else:
            raise AssertionError("changed patch must fail")

        patch.write_bytes(original_patch)
        source_file = (
            out_dir / "source" / "tinyvllm" / "__init__.py"
        )
        source_file.write_text("VALUE = 3\n", encoding="utf-8")
        try:
            audit.validate_source_snapshot(
                out_dir / "source",
                evidence,
                patch,
                expected_owned_roots=OWNED_ROOTS,
            )
        except ValueError as exc:
            assert "source file hash mismatch" in str(exc)
        else:
            raise AssertionError("changed source file must fail")
    finally:
        _cleanup_repo(temporary, root)


def test_generic_source_evidence_is_json_serializable():
    temporary, root = _source_repo()
    try:
        evidence = _build(root, root / "snapshot")
        json.dumps(evidence, sort_keys=True)
    finally:
        _cleanup_repo(temporary, root)


def main():
    test_generic_source_evidence_reconstructs_dirty_tree()
    test_generic_source_evidence_ignores_only_configured_artifacts()
    test_generic_source_evidence_rejects_untracked_owned_file()
    test_generic_source_evidence_excludes_ignored_files_under_owned_root()
    test_generic_snapshot_rejects_patch_and_file_tampering()
    test_generic_source_evidence_is_json_serializable()
    print("source audit tests passed")


if __name__ == "__main__":
    main()
