"""Local source-bound authorization for real checkpoint-load work."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys


THIS_DIR = Path(__file__).resolve().parent
CONTRACT_PATH = THIS_DIR / "qwen35_real_checkpoint_load_contract.py"
CLAIM_BOUNDARY = (
    "Authorization permits worker implementation only; worker execution, "
    "checkpoint payload access, inference speed, cache or memory reduction, "
    "compression safety, quality retention, and native execution remain "
    "unproven."
)


def _load_contract():
    spec = importlib.util.spec_from_file_location(
        "qwen35_real_checkpoint_load_contract_for_authorization",
        os.fspath(CONTRACT_PATH),
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


contract = _load_contract()


def _read_json(path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid {Path(path).name}: {exc}") from exc


def _source_tree_sha256(hashes):
    payload = json.dumps(
        hashes,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def current_source_hashes(repo_root, owned_source_files):
    root = Path(repo_root)
    result = {}
    for relative in owned_source_files:
        path = root / relative
        if not path.is_file():
            raise ValueError(f"missing owned source file: {relative}")
        digest = hashlib.sha256()
        with path.open("rb") as source:
            for block in iter(lambda: source.read(1 << 20), b""):
                digest.update(block)
        result[relative] = digest.hexdigest()
    return result


def _git_lines(repo_root, *arguments):
    completed = subprocess.run(
        ["git", *arguments],
        cwd=repo_root,
        check=False,
        text=True,
        capture_output=True,
    )
    if completed.returncode != 0:
        detail = completed.stderr or completed.stdout
        raise ValueError(
            f"git {' '.join(arguments)} failed: {detail.strip()}"
        )
    return {
        line for line in completed.stdout.splitlines() if line
    }


def inspect_git_state(repo_root, owned_source_files):
    root = Path(repo_root)
    branch = subprocess.run(
        ["git", "branch", "--show-current"],
        cwd=root,
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()
    tracked = _git_lines(
        root,
        "ls-files",
        "--",
        *owned_source_files,
    )
    staged = _git_lines(
        root,
        "diff",
        "--cached",
        "--name-only",
        "--",
        *owned_source_files,
    )
    unstaged = _git_lines(
        root,
        "diff",
        "--name-only",
        "--",
        *owned_source_files,
    )
    untracked = _git_lines(
        root,
        "ls-files",
        "--others",
        "--exclude-standard",
        "--",
        *owned_source_files,
    )
    return {
        "branch": branch,
        "commit": commit,
        "tracked": tracked,
        "staged": staged,
        "unstaged": unstaged,
        "untracked": untracked,
    }


def _blocked_result(run_tag, checks, reasons, **fields):
    return {
        "schema_version": contract.SCHEMA_VERSION,
        "decision": "BLOCKED",
        "run_tag": run_tag,
        "checks": checks,
        "reasons": reasons,
        "worker_implementation_authorized": False,
        "worker_execution_authorized": False,
        "claim_boundary": CLAIM_BOUNDARY,
        **fields,
    }


def authorize_run(
    run_dir,
    *,
    owned_source_files,
    current_hash_function=current_source_hashes,
    git_state_function=inspect_git_state,
):
    destination = Path(run_dir)
    run_tag = destination.name
    repo_root = destination
    for _ in range(3):
        repo_root = repo_root.parent
    checks = {}
    reasons = []
    try:
        preflight = _read_json(destination / "preflight.json")
        source = _read_json(destination / "source_manifest.json")
    except ValueError as exc:
        return _blocked_result(run_tag, checks, [str(exc)])
    try:
        contract.validate_preflight(preflight)
        checks["preflight_valid"] = True
    except (TypeError, ValueError) as exc:
        checks["preflight_valid"] = False
        reasons.append(f"preflight validation failed: {exc}")
    checks["preflight_ready"] = preflight.get("status") == "READY"
    if not checks["preflight_ready"]:
        reasons.append("preflight is not READY")
    checks["source_schema"] = (
        source.get("schema_version") == contract.SCHEMA_VERSION
    )
    if not checks["source_schema"]:
        reasons.append("source manifest schema mismatch")
    checks["remote_target"] = (
        source.get("remote_target") == contract.REMOTE_TARGET
    )
    if not checks["remote_target"]:
        reasons.append("source manifest remote target mismatch")
    local_hashes = source.get("local_file_sha256")
    remote_hashes = source.get("remote_file_sha256")
    preflight_local = preflight.get("source_file_sha256")
    preflight_remote = preflight.get("remote_source_file_sha256")
    expected_paths = set(owned_source_files)
    checks["source_hash_maps"] = (
        isinstance(local_hashes, dict)
        and set(local_hashes) == expected_paths
        and local_hashes == remote_hashes
        and local_hashes == preflight_local
        and local_hashes == preflight_remote
    )
    if not checks["source_hash_maps"]:
        reasons.append("source hash maps disagree")
    recorded_tree = source.get("source_tree_sha256")
    checks["recorded_source_tree"] = (
        isinstance(local_hashes, dict)
        and recorded_tree == _source_tree_sha256(local_hashes)
        and recorded_tree == preflight.get("source_tree_sha256")
    )
    if not checks["recorded_source_tree"]:
        reasons.append("recorded source tree SHA256 mismatch")
    try:
        current_hashes = current_hash_function(
            repo_root,
            tuple(owned_source_files),
        )
        current_tree = _source_tree_sha256(current_hashes)
        checks["current_source_hashes"] = (
            isinstance(local_hashes, dict)
            and current_hashes == local_hashes
        )
        if not checks["current_source_hashes"]:
            reasons.append("current owned source hashes differ")
    except (OSError, TypeError, ValueError) as exc:
        current_hashes = {}
        current_tree = None
        checks["current_source_hashes"] = False
        reasons.append(f"current source hashing failed: {exc}")
    try:
        git_state = git_state_function(
            repo_root,
            tuple(owned_source_files),
        )
    except (OSError, subprocess.SubprocessError, ValueError) as exc:
        git_state = {}
        reasons.append(f"Git provenance inspection failed: {exc}")
    current_branch = git_state.get("branch")
    current_commit = git_state.get("commit")
    checks["branch_match"] = current_branch == source.get("branch")
    if not checks["branch_match"]:
        reasons.append("current branch differs")
    checks["commit_match"] = current_commit == source.get("commit")
    if not checks["commit_match"]:
        reasons.append("current commit differs")
    tracked = set(git_state.get("tracked", ()))
    staged = set(git_state.get("staged", ()))
    unstaged = set(git_state.get("unstaged", ()))
    untracked = set(git_state.get("untracked", ()))
    checks["owned_source_tracked"] = tracked == expected_paths
    if not checks["owned_source_tracked"]:
        reasons.append("owned source is not tracked")
    checks["owned_source_staged_clean"] = not (staged & expected_paths)
    if not checks["owned_source_staged_clean"]:
        reasons.append("owned source has staged changes")
    checks["owned_source_unstaged_clean"] = not (
        unstaged & expected_paths
    )
    if not checks["owned_source_unstaged_clean"]:
        reasons.append("owned source has unstaged changes")
    checks["owned_source_untracked_clean"] = not (
        untracked & expected_paths
    )
    if not checks["owned_source_untracked_clean"]:
        reasons.append("owned source is untracked")
    fields = {
        "source_tree_sha256": recorded_tree,
        "current_source_tree_sha256": current_tree,
        "current_branch": current_branch,
        "current_commit": current_commit,
    }
    if reasons or not checks or not all(checks.values()):
        return _blocked_result(run_tag, checks, reasons, **fields)
    return {
        "schema_version": contract.SCHEMA_VERSION,
        "decision": "AUTHORIZED",
        "run_tag": run_tag,
        "checks": checks,
        "reasons": [],
        "worker_implementation_authorized": True,
        "worker_execution_authorized": False,
        "claim_boundary": CLAIM_BOUNDARY,
        **fields,
    }
