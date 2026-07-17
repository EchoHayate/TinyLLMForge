from __future__ import annotations

import hashlib
import io
import json
import os
import re
import shutil
import subprocess
import tarfile
import tempfile
from pathlib import Path


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _git(
    repo_root: Path,
    *args: str,
    input_bytes: bytes | None = None,
) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", *args],
        cwd=repo_root,
        input=input_bytes,
        capture_output=True,
        check=False,
    )


def _checked_git(
    repo_root: Path,
    *args: str,
    input_bytes: bytes | None = None,
) -> bytes:
    result = _git(repo_root, *args, input_bytes=input_bytes)
    if result.returncode != 0:
        stderr = result.stderr.decode(
            "utf-8",
            errors="replace",
        ).strip()
        raise ValueError(
            f"git {' '.join(args)} failed with "
            f"{result.returncode}: {stderr}"
        )
    return result.stdout


def _matches_root(relative_path: str, roots: tuple[str, ...]) -> bool:
    normalized = Path(relative_path).as_posix().lstrip("./")
    return any(
        normalized == root or normalized.startswith(root + "/")
        for root in roots
    )


def _git_path_set(repo_root: Path, *args: str) -> set[str]:
    output = _checked_git(repo_root, *args)
    return {
        value.decode("utf-8", errors="surrogateescape")
        for value in output.split(b"\0")
        if value
    }


def expand_owned_source_paths(
    repo_root: Path,
    owned_roots: tuple[str, ...],
    ignored_untracked_prefixes: tuple[str, ...] = (),
) -> tuple[str, ...]:
    del ignored_untracked_prefixes
    repo_root = repo_root.resolve()
    paths = []
    for owned_root in owned_roots:
        root = repo_root / owned_root
        if not root.exists():
            raise ValueError(
                f"missing owned source path: {owned_root}"
            )
        if root.is_symlink():
            raise ValueError(
                f"owned source path is a symlink: {owned_root}"
            )
        candidates = root.rglob("*") if root.is_dir() else (root,)
        for candidate in candidates:
            if candidate.is_symlink():
                raise ValueError(
                    "owned source contains a symlink: "
                    f"{candidate.relative_to(repo_root).as_posix()}"
                )
            if candidate.is_file():
                paths.append(
                    candidate.relative_to(repo_root).as_posix()
                )
            elif candidate.exists() and not candidate.is_dir():
                raise ValueError(
                    "owned source contains a non-regular path: "
                    f"{candidate.relative_to(repo_root).as_posix()}"
                )
    if len(paths) != len(set(paths)):
        raise ValueError("owned source roots overlap")
    return tuple(sorted(paths))


def hash_source_tree(
    source_root: Path,
    relative_paths: tuple[str, ...],
) -> list[dict]:
    source_root = source_root.resolve()
    files = []
    for relative_path in sorted(relative_paths):
        path = source_root / relative_path
        if path.is_symlink() or not path.is_file():
            raise ValueError(
                f"source path is not a regular file: {relative_path}"
            )
        payload = path.read_bytes()
        files.append({
            "path": relative_path,
            "size_bytes": len(payload),
            "sha256": sha256_bytes(payload),
        })
    return files


def source_tree_sha256(files: list[dict]) -> str:
    canonical = json.dumps(
        files,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return sha256_bytes(canonical)


def _validate_sha256(value, label: str) -> None:
    if (
        not isinstance(value, str)
        or re.fullmatch(r"[0-9a-f]{64}", value) is None
    ):
        raise ValueError(f"invalid {label}")


def validate_source_snapshot(
    source_root: Path,
    evidence: dict,
    patch_path: Path,
    *,
    expected_owned_roots: tuple[str, ...],
) -> dict:
    if evidence.get("schema_version") != 1:
        raise ValueError("unsupported source evidence schema")
    base_commit = evidence.get("base_commit")
    if (
        not isinstance(base_commit, str)
        or re.fullmatch(r"[0-9a-f]{40}", base_commit) is None
    ):
        raise ValueError("invalid source base commit")
    _validate_sha256(
        evidence.get("patch_sha256"),
        "patch sha256",
    )
    _validate_sha256(
        evidence.get("tree_sha256"),
        "source tree sha256",
    )
    if evidence.get("owned_roots") != list(expected_owned_roots):
        raise ValueError("owned source roots mismatch")

    patch_payload = patch_path.read_bytes()
    if len(patch_payload) != evidence.get("patch_size_bytes"):
        raise ValueError("patch size mismatch")
    if sha256_bytes(patch_payload) != evidence["patch_sha256"]:
        raise ValueError("patch hash mismatch")

    expected_files = evidence.get("files")
    if not isinstance(expected_files, list):
        raise ValueError("source evidence files must be a list")
    expected_paths = []
    for record in expected_files:
        if not isinstance(record, dict):
            raise ValueError("invalid source file record")
        relative_path = record.get("path")
        if (
            not isinstance(relative_path, str)
            or not relative_path
            or Path(relative_path).is_absolute()
            or ".." in Path(relative_path).parts
            or not _matches_root(
                relative_path,
                expected_owned_roots,
            )
        ):
            raise ValueError("invalid source file path")
        if (
            not isinstance(record.get("size_bytes"), int)
            or record["size_bytes"] < 0
        ):
            raise ValueError("invalid source file size")
        _validate_sha256(
            record.get("sha256"),
            "source file sha256",
        )
        expected_paths.append(relative_path)
    if (
        expected_paths != sorted(expected_paths)
        or len(expected_paths) != len(set(expected_paths))
    ):
        raise ValueError(
            "source file records must be unique and sorted"
        )

    source_root = source_root.resolve()
    actual_paths = []
    for path in source_root.rglob("*"):
        if path.is_symlink():
            raise ValueError("source snapshot contains a symlink")
        if path.is_file():
            actual_paths.append(
                path.relative_to(source_root).as_posix()
            )
        elif path.exists() and not path.is_dir():
            raise ValueError(
                "source snapshot contains a non-regular path"
            )
    actual_paths.sort()
    if actual_paths != expected_paths:
        raise ValueError("source path set mismatch")

    actual_files = hash_source_tree(
        source_root,
        tuple(actual_paths),
    )
    for expected, actual in zip(expected_files, actual_files):
        if (
            expected["size_bytes"] != actual["size_bytes"]
            or expected["sha256"] != actual["sha256"]
        ):
            raise ValueError(
                f"source file hash mismatch: {expected['path']}"
            )
    actual_tree_sha256 = source_tree_sha256(actual_files)
    if actual_tree_sha256 != evidence["tree_sha256"]:
        raise ValueError("source tree hash mismatch")
    return {
        "valid": True,
        "source_tree_sha256": actual_tree_sha256,
        "file_count": len(actual_files),
    }


def reconstruct_source_snapshot(
    repo_root: Path,
    source_root: Path,
    evidence: dict,
    patch_path: Path,
    *,
    expected_owned_roots: tuple[str, ...],
) -> None:
    repo_root = repo_root.resolve()
    source_root = source_root.resolve()
    if source_root.exists():
        if any(source_root.iterdir()):
            raise ValueError(
                "source reconstruction destination is not empty"
            )
    else:
        source_root.mkdir(parents=True)

    archive = _checked_git(
        repo_root,
        "archive",
        "--format=tar",
        evidence["base_commit"],
        "--",
        *expected_owned_roots,
    )
    with tarfile.open(
        fileobj=io.BytesIO(archive),
        mode="r:",
    ) as tar:
        members = tar.getmembers()
        for member in members:
            member_path = Path(member.name)
            if (
                member_path.is_absolute()
                or ".." in member_path.parts
            ):
                raise ValueError(
                    "git archive contains an unsafe path"
                )
            if member.issym() or member.islnk():
                raise ValueError("git archive contains a link")
        tar.extractall(source_root, members=members)

    patch_payload = patch_path.read_bytes()
    if patch_payload:
        environment = os.environ.copy()
        environment["GIT_CEILING_DIRECTORIES"] = str(
            source_root.parent
        )
        result = subprocess.run(
            [
                "git",
                "apply",
                "--binary",
                "--whitespace=nowarn",
                "-",
            ],
            cwd=source_root,
            env=environment,
            input=patch_payload,
            capture_output=True,
            check=False,
        )
        if result.returncode != 0:
            stderr = result.stderr.decode(
                "utf-8",
                errors="replace",
            ).strip()
            raise ValueError(
                "source patch reconstruction failed with "
                f"{result.returncode}: {stderr}"
            )
    validate_source_snapshot(
        source_root,
        evidence,
        patch_path,
        expected_owned_roots=expected_owned_roots,
    )


def build_source_evidence(
    repo_root: Path,
    out_dir: Path,
    *,
    owned_roots: tuple[str, ...],
    ignored_untracked_prefixes: tuple[str, ...] = (),
) -> dict:
    repo_root = repo_root.resolve()
    out_dir = out_dir.resolve()
    if out_dir.exists():
        if not out_dir.is_dir() or any(out_dir.iterdir()):
            raise ValueError(
                "source evidence output directory is not empty"
            )
    else:
        out_dir.mkdir(parents=True)
    base_commit = _checked_git(
        repo_root,
        "rev-parse",
        "HEAD",
    ).decode().strip()
    if re.fullmatch(r"[0-9a-f]{40}", base_commit) is None:
        raise ValueError(
            "git HEAD did not resolve to a full commit"
        )

    changed_paths = _git_path_set(
        repo_root,
        "diff",
        "--name-only",
        "-z",
        base_commit,
        "--",
    )
    outside_changes = sorted(
        path
        for path in changed_paths
        if not _matches_root(path, owned_roots)
    )
    if outside_changes:
        raise ValueError(
            "changed path outside owned source boundary: "
            + ", ".join(outside_changes)
        )
    untracked_paths = _git_path_set(
        repo_root,
        "ls-files",
        "--others",
        "--exclude-standard",
        "-z",
    )
    untracked_owned = sorted(
        path
        for path in untracked_paths
        if _matches_root(path, owned_roots)
    )
    if untracked_owned:
        raise ValueError(
            "untracked owned source: "
            + ", ".join(untracked_owned)
        )
    untracked_outside = sorted(
        path
        for path in untracked_paths
        if not _matches_root(path, owned_roots)
        and not _matches_root(
            path,
            ignored_untracked_prefixes,
        )
    )
    if untracked_outside:
        raise ValueError(
            "untracked path outside owned source boundary: "
            + ", ".join(untracked_outside)
        )

    relative_paths = expand_owned_source_paths(
        repo_root,
        owned_roots,
    )
    staged_source = out_dir / "source"
    for relative_path in relative_paths:
        destination = staged_source / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(
            repo_root / relative_path,
            destination,
        )

    patch_payload = _checked_git(
        repo_root,
        "diff",
        "--binary",
        "--no-ext-diff",
        base_commit,
        "--",
        *owned_roots,
    )
    patch_path = out_dir / "source.patch"
    patch_path.write_bytes(patch_payload)
    files = hash_source_tree(
        staged_source,
        relative_paths,
    )
    evidence = {
        "schema_version": 1,
        "base_commit": base_commit,
        "dirty": bool(patch_payload),
        "patch_path": "source.patch",
        "patch_sha256": sha256_bytes(patch_payload),
        "patch_size_bytes": len(patch_payload),
        "owned_roots": list(owned_roots),
        "files": files,
        "tree_sha256": source_tree_sha256(files),
    }
    (out_dir / "source_evidence.json").write_text(
        json.dumps(
            evidence,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    validate_source_snapshot(
        staged_source,
        evidence,
        patch_path,
        expected_owned_roots=owned_roots,
    )
    with tempfile.TemporaryDirectory() as temporary:
        reconstruct_source_snapshot(
            repo_root,
            Path(temporary) / "source",
            evidence,
            patch_path,
            expected_owned_roots=owned_roots,
        )
    return evidence
