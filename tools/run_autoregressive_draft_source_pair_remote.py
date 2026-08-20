#!/usr/bin/env python3
"""Run a source-version paired autoregressive-draft campaign remotely."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import re
import shutil
import stat
import subprocess
import sys


TOOLS_ROOT = Path(__file__).resolve().parent
if str(TOOLS_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOLS_ROOT))

import run_autoregressive_draft_command_timeline_remote as child_runner
from autoregressive_draft_source_pair_gate import (
    BASELINE_REVISION,
    build_source_pair_artifact,
    canonical_json_bytes,
    expected_source_pair_schedule,
)


REMOTE_TASK_ROOT = child_runner.REMOTE_TASK_ROOT


def _child_tag(run_tag: str, source: str) -> str:
    tag = child_runner.validate_run_tag(run_tag)
    if source not in ("baseline", "candidate"):
        raise ValueError("source must be baseline or candidate")
    return child_runner.validate_run_tag(f"{tag}-{source}")


def source_pair_paths(run_tag: str) -> dict[str, str]:
    tag = child_runner.validate_run_tag(run_tag)
    baseline_tag = _child_tag(tag, "baseline")
    candidate_tag = _child_tag(tag, "candidate")
    return {
        "parent_primary": (
            f"{REMOTE_TASK_ROOT}/source-pair-runs/{tag}"
        ),
        "parent_controller": (
            f"{REMOTE_TASK_ROOT}/source-pair-controller-verification/{tag}"
        ),
        "baseline_primary": child_runner.primary_run_path(baseline_tag),
        "baseline_controller": child_runner.controller_run_path(
            baseline_tag
        ),
        "candidate_primary": child_runner.primary_run_path(candidate_tag),
        "candidate_controller": child_runner.controller_run_path(
            candidate_tag
        ),
    }


def build_execution_plan(run_tag: str) -> list[dict]:
    child_runner.validate_run_tag(run_tag)
    rows = []
    for pair in expected_source_pair_schedule():
        for source in (pair.first_source, pair.second_source):
            rows.append({
                "scope": "child",
                "source": source,
                "action": "epoch",
                "pair_index": pair.pair_index,
                "cuda_mode": pair.cuda_mode,
            })
    for source in ("baseline", "candidate"):
        rows.append({
            "scope": "child",
            "source": source,
            "action": "finalize",
        })
    for action in (
        "assemble",
        "pre-manifest-verify",
        "manifest",
        "primary-verify",
        "controller-copy",
        "controller-verify",
        "compare-receipts",
    ):
        rows.append({
            "scope": "parent",
            "source": "candidate",
            "action": action,
        })
    return rows


def _run_checked(
    command: list[str],
    *,
    command_runner,
    cwd: Path,
    text: bool,
):
    result = command_runner(
        command,
        cwd=cwd,
        text=text,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"command failed: {' '.join(command)}")
    return result


def export_git_revision_archive(
    repo_root: Path,
    revision: str,
    *,
    command_runner=subprocess.run,
) -> bytes:
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("revision must be a full lowercase Git object")
    root = Path(repo_root)
    listing = _run_checked(
        ["git", "ls-tree", "-r", "--name-only", revision],
        command_runner=command_runner,
        cwd=root,
        text=True,
    )
    tracked = {
        line.strip()
        for line in listing.stdout.splitlines()
        if line.strip()
    }
    selected = []
    for configured in child_runner.SOURCE_PATHS:
        prefix = configured.rstrip("/")
        if (
            configured in tracked
            or prefix in tracked
            or any(path.startswith(f"{prefix}/") for path in tracked)
        ):
            selected.append(configured)
    if not selected:
        raise ValueError("Git object source inventory is empty")
    result = _run_checked(
        [
            "git",
            "archive",
            "--format=tar",
            "--prefix=source/",
            revision,
            "--",
            *selected,
        ],
        command_runner=command_runner,
        cwd=root,
        text=False,
    )
    if not isinstance(result.stdout, bytes) or not result.stdout:
        raise ValueError("Git object archive payload is invalid")
    return result.stdout


def build_parent_preflight_script(run_tag: str) -> str:
    paths = source_pair_paths(run_tag)
    return "\n".join((
        "import json,pathlib,subprocess",
        f"paths={json.dumps(paths, sort_keys=True)!r}",
        "paths=json.loads(paths)",
        "gpu=subprocess.run([",
        " 'nvidia-smi',",
        " '--query-gpu=index,uuid,memory.used,utilization.gpu',",
        " '--format=csv,noheader,nounits'],",
        " capture_output=True,text=True,check=False)",
        "apps=subprocess.run([",
        " 'nvidia-smi',",
        " '--query-compute-apps=pid,gpu_uuid,process_name',",
        " '--format=csv,noheader,nounits'],",
        " capture_output=True,text=True,check=False)",
        "if gpu.returncode or apps.returncode: raise SystemExit(3)",
        "processes={}",
        "for line in apps.stdout.splitlines():",
        " fields=[part.strip() for part in line.split(',')]",
        " if len(fields)==3:",
        "  processes.setdefault(fields[1],[]).append(",
        "   {'pid':int(fields[0]),'process_name':fields[2]})",
        "rows=[]",
        "for line in gpu.stdout.splitlines():",
        " fields=[part.strip() for part in line.split(',')]",
        " if len(fields)!=4: raise SystemExit(4)",
        " rows.append({'index':int(fields[0]),'uuid':fields[1],",
        "  'memory_used_mib':int(fields[2]),",
        "  'utilization_percent':int(fields[3]),",
        "  'compute_processes':processes.get(fields[1],[])})",
        "idle=[row for row in rows",
        " if row['memory_used_mib']<=1024",
        " and row['utilization_percent']<=5",
        " and not row['compute_processes']]",
        "print(json.dumps({'path_exists':{",
        " key:pathlib.Path(value).exists()",
        " for key,value in paths.items()},",
        " 'gpu_rows':idle},sort_keys=True,separators=(',',':')))",
    ))


def run_preflight(
    *,
    run_tag: str,
    command_runner=subprocess.run,
    now=None,
    repo_root: Path | None = None,
) -> dict:
    tag = child_runner.validate_run_tag(run_tag)
    kerberos = child_runner._local_kerberos_preflight(
        command_runner=command_runner,
        now=now,
    )
    if kerberos.get("status") != "READY":
        return kerberos
    root = (
        Path(__file__).resolve().parents[1]
        if repo_root is None
        else Path(repo_root)
    )
    candidate_revision = child_runner._local_source_commit(
        repo_root=root,
        command_runner=command_runner,
    )
    result = child_runner._run_remote_command(
        [
            "bash",
            "-lc",
            (
                f"{child_runner.REMOTE_PYTHON} - <<'PY'\n"
                f"{build_parent_preflight_script(tag)}\n"
                "PY"
            ),
        ],
        command_runner=command_runner,
        context="source-pair read-only preflight",
        now=now,
        kerberos_status=kerberos,
        text=True,
        capture_output=True,
    )
    try:
        payload = json.loads(result.stdout)
    except (TypeError, json.JSONDecodeError) as error:
        raise RuntimeError(
            "source-pair preflight returned invalid JSON"
        ) from error
    if not isinstance(payload, dict):
        raise RuntimeError("source-pair preflight payload is invalid")
    path_exists = payload.get("path_exists")
    paths = source_pair_paths(tag)
    if (
        not isinstance(path_exists, dict)
        or set(path_exists) != set(paths)
        or any(value is not False for value in path_exists.values())
    ):
        raise ValueError("source-pair remote destination already exists")
    gpu_rows = payload.get("gpu_rows")
    try:
        gpu = child_runner.classify_gpu_preflight(gpu_rows)
    except ValueError as error:
        return {
            "status": "INCONCLUSIVE_ENVIRONMENT",
            "reason": str(error),
            "available_idle_gpu_count": (
                len(gpu_rows) if isinstance(gpu_rows, list) else 0
            ),
            "gpu_indices": [],
            "gpu_uuids": [],
            "baseline_revision": BASELINE_REVISION,
            "candidate_revision": candidate_revision,
            "local_kerberos": kerberos,
            **paths,
        }
    return {
        **gpu,
        "baseline_revision": BASELINE_REVISION,
        "candidate_revision": candidate_revision,
        "local_kerberos": kerberos,
        **paths,
    }


def _parent_remote_arguments(
    action: str,
    arguments: list[str],
) -> list[str]:
    if not arguments:
        raise ValueError("parent remote action requires a run tag")
    tag = child_runner.validate_run_tag(arguments[0])
    candidate_source = (
        f"{child_runner.primary_run_path(_child_tag(tag, 'candidate'))}"
        "/source"
    )
    script = " ".join((
        f"SOURCE_PAIR_ACTION={action}",
        "exec",
        child_runner.shlex.quote(child_runner.REMOTE_PYTHON),
        child_runner.shlex.quote(
            f"{candidate_source}/tools/{Path(__file__).name}"
        ),
        "_remote-action",
        child_runner.shlex.quote(action),
        *(child_runner.shlex.quote(value) for value in arguments),
    ))
    return ["bash", "-lc", script]


def _run_parent_remote_action(
    action: str,
    arguments: list[str],
    *,
    command_runner=subprocess.run,
    context: str,
    now=None,
    allow_failure: bool = False,
    **kwargs,
):
    return child_runner._run_remote_command(
        _parent_remote_arguments(action, arguments),
        command_runner=command_runner,
        context=context,
        now=now,
        allow_failure=allow_failure,
        **kwargs,
    )


def _preserve_partial_children(
    run_tag: str,
    sources: list[str],
    *,
    command_runner,
    now,
) -> None:
    for source in sources:
        tag = _child_tag(run_tag, source)
        child_runner._run_remote_action(
            "partial-copy",
            [tag],
            command_runner=command_runner,
            context=f"preserve partial {source} child evidence",
            now=now,
            allow_failure=True,
            text=True,
            capture_output=True,
        )


def _run_child_member(
    *,
    run_tag: str,
    source: str,
    pair_index: int,
    cuda_mode: str,
    preflight: dict,
    command_runner,
    now,
    target_model: str,
    draft_model: str,
) -> tuple[bool, str]:
    child_tag = _child_tag(run_tag, source)
    schedule = child_runner.build_epoch_schedule()
    if pair_index < 0 or pair_index >= len(schedule):
        raise ValueError("pair index is invalid")
    _, expected_mode, position = schedule[pair_index]
    if cuda_mode != expected_mode:
        raise ValueError("source-pair CUDA mode does not match child epoch")
    gpu_indices = preflight["gpu_indices"]
    gpu_uuids = preflight["gpu_uuids"]
    before_result = child_runner._run_remote_action(
        "inventory-before",
        [child_tag, str(pair_index)],
        command_runner=command_runner,
        context=f"{source} inventory before pair {pair_index}",
        now=now,
        allow_failure=True,
        text=True,
        capture_output=True,
    )
    if before_result.returncode != 0:
        return False, "inventory-before"
    try:
        before = child_runner._inventory_rows_from_result(before_result)
    except (RuntimeError, ValueError):
        return False, "inventory-before"
    if not child_runner._matches_frozen_gpu_inventory(
        before,
        gpu_indices=gpu_indices,
        gpu_uuids=gpu_uuids,
    ):
        return False, "inventory-before"
    worker_result = child_runner._run_remote_action(
        "epoch",
        [
            child_tag,
            str(pair_index),
            cuda_mode,
            position,
            json.dumps(gpu_indices),
            json.dumps(gpu_uuids),
            target_model,
            draft_model,
        ],
        command_runner=command_runner,
        context=f"{source} epoch for pair {pair_index}",
        now=now,
        allow_failure=True,
        text=True,
        capture_output=True,
    )
    if worker_result.returncode != 0:
        return False, "epoch"
    after_result = child_runner._run_remote_action(
        "inventory-after",
        [child_tag, str(pair_index)],
        command_runner=command_runner,
        context=f"{source} inventory after pair {pair_index}",
        now=now,
        allow_failure=True,
        text=True,
        capture_output=True,
    )
    if after_result.returncode != 0:
        return False, "inventory-after"
    try:
        after = child_runner._inventory_rows_from_result(after_result)
    except (RuntimeError, ValueError):
        return False, "inventory-after"
    if (
        not child_runner._same_gpu_inventory(before, after)
        or not child_runner._matches_frozen_gpu_inventory(
            after,
            gpu_indices=gpu_indices,
            gpu_uuids=gpu_uuids,
        )
    ):
        return False, "inventory-after"
    return True, ""


def _finalize_child(
    *,
    run_tag: str,
    source: str,
    command_runner,
    now,
) -> tuple[bool, str]:
    child_tag = _child_tag(run_tag, source)
    for action, context in (
        ("assemble", "canonical assembly"),
        ("pre-manifest-verify", "pre-manifest verification"),
        ("manifest", "manifest creation"),
        ("primary-verify", "primary verification"),
        ("controller-copy", "controller copy"),
        ("controller-verify", "controller verification"),
        ("compare-receipts", "receipt comparison"),
    ):
        result = child_runner._run_remote_action(
            action,
            [child_tag],
            command_runner=command_runner,
            context=f"{source} child {context}",
            now=now,
            allow_failure=True,
            text=True,
            capture_output=True,
        )
        if result.returncode != 0:
            return False, action
    return True, ""


def run_campaign(
    *,
    run_tag: str,
    command_runner=subprocess.run,
    now=None,
    repo_root: Path | None = None,
    target_model: str = child_runner.DEFAULT_TARGET_MODEL,
    draft_model: str = child_runner.DEFAULT_DRAFT_MODEL,
) -> dict:
    tag = child_runner.validate_run_tag(run_tag)
    preflight = run_preflight(
        run_tag=tag,
        command_runner=command_runner,
        now=now,
        repo_root=repo_root,
    )
    if preflight.get("status") != "READY":
        return preflight
    root = (
        Path(__file__).resolve().parents[1]
        if repo_root is None
        else Path(repo_root)
    )
    revisions = {
        "baseline": preflight["baseline_revision"],
        "candidate": preflight["candidate_revision"],
    }
    archives = {
        source: export_git_revision_archive(
            root,
            revision,
            command_runner=command_runner,
        )
        for source, revision in revisions.items()
    }
    prepared = []
    for source in ("baseline", "candidate"):
        child_tag = _child_tag(tag, source)
        child_preflight = copy.deepcopy(preflight)
        child_preflight.update({
            "source_commit": revisions[source],
            "primary_run": child_runner.primary_run_path(child_tag),
            "controller_run": child_runner.controller_run_path(child_tag),
        })
        payload = child_runner._encode_prepare_payload(
            source_archive=archives[source],
            source_patch=b"",
        )
        result = child_runner._run_remote_action(
            "prepare",
            [
                child_tag,
                revisions[source],
                json.dumps(child_preflight, sort_keys=True),
            ],
            command_runner=command_runner,
            context=f"prepare {source} source child",
            now=now,
            allow_failure=True,
            input=payload,
            capture_output=True,
        )
        if result.returncode != 0:
            _preserve_partial_children(
                tag,
                list(dict.fromkeys([*prepared, source])),
                command_runner=command_runner,
                now=now,
            )
            return {
                "status": "FAILED",
                "failed_scope": "prepare",
                "failed_source": source,
                "prepared_sources": prepared,
                **source_pair_paths(tag),
            }
        prepared.append(source)
    completed_members = []
    for pair in expected_source_pair_schedule():
        for source in (pair.first_source, pair.second_source):
            passed, failed_action = _run_child_member(
                run_tag=tag,
                source=source,
                pair_index=pair.pair_index,
                cuda_mode=pair.cuda_mode,
                preflight=preflight,
                command_runner=command_runner,
                now=now,
                target_model=target_model,
                draft_model=draft_model,
            )
            if not passed:
                _preserve_partial_children(
                    tag,
                    prepared,
                    command_runner=command_runner,
                    now=now,
                )
                return {
                    "status": "FAILED",
                    "failed_scope": "pair-member",
                    "failed_action": failed_action,
                    "failed_source": source,
                    "failed_pair_index": pair.pair_index,
                    "completed_members": completed_members,
                    **source_pair_paths(tag),
                }
            completed_members.append({
                "pair_index": pair.pair_index,
                "cuda_mode": pair.cuda_mode,
                "source": source,
            })
    finalized = []
    for source in ("baseline", "candidate"):
        passed, failed_action = _finalize_child(
            run_tag=tag,
            source=source,
            command_runner=command_runner,
            now=now,
        )
        if not passed:
            unfinished = [
                candidate
                for candidate in prepared
                if candidate not in finalized
            ]
            _preserve_partial_children(
                tag,
                unfinished,
                command_runner=command_runner,
                now=now,
            )
            return {
                "status": "FAILED",
                "failed_scope": "child-finalization",
                "failed_action": failed_action,
                "failed_source": source,
                "finalized_sources": finalized,
                **source_pair_paths(tag),
            }
        finalized.append(source)
    for action in (
        "assemble",
        "pre-manifest-verify",
        "manifest",
        "primary-verify",
        "controller-copy",
        "controller-verify",
        "compare-receipts",
    ):
        result = _run_parent_remote_action(
            action,
            [tag],
            command_runner=command_runner,
            context=f"parent source-pair {action}",
            now=now,
            allow_failure=True,
            text=True,
            capture_output=True,
        )
        if result.returncode != 0:
            if action != "controller-copy":
                _run_parent_remote_action(
                    "partial-copy",
                    [tag],
                    command_runner=command_runner,
                    context="preserve partial parent evidence",
                    now=now,
                    allow_failure=True,
                    text=True,
                    capture_output=True,
                )
            return {
                "status": "FAILED",
                "failed_scope": "parent-finalization",
                "failed_action": action,
                "completed_members": completed_members,
                **source_pair_paths(tag),
            }
    return {
        "status": "PASS",
        "baseline_revision": revisions["baseline"],
        "candidate_revision": revisions["candidate"],
        "completed_members": completed_members,
        **source_pair_paths(tag),
    }


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json_exclusive(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(canonical_json_bytes(payload))


def _parent_bundle_inputs(root: Path) -> dict:
    return {
        "artifact_path": root / "source-pair.json",
        "baseline_artifact_path": (
            root / "children" / "baseline" / "command-timeline.json"
        ),
        "candidate_artifact_path": (
            root / "children" / "candidate" / "command-timeline.json"
        ),
        "baseline_manifest_path": (
            root / "children" / "baseline" / "manifest.sha256"
        ),
        "candidate_manifest_path": (
            root / "children" / "candidate" / "manifest.sha256"
        ),
        "baseline_receipt_paths": {
            "remote": (
                root / "children" / "baseline" / "verify.remote.json"
            ),
            "local": (
                root / "children" / "baseline" / "verify.local.json"
            ),
        },
        "candidate_receipt_paths": {
            "remote": (
                root / "children" / "candidate" / "verify.remote.json"
            ),
            "local": (
                root / "children" / "candidate" / "verify.local.json"
            ),
        },
    }


def _remote_parent_assemble(tag: str) -> int:
    paths = source_pair_paths(tag)
    primary = Path(paths["parent_primary"])
    controller = Path(paths["parent_controller"])
    if primary.exists() or controller.exists():
        raise ValueError("parent source-pair destination already exists")
    primary.mkdir(parents=True, exist_ok=False)
    for source in ("baseline", "candidate"):
        destination = primary / "children" / source
        destination.mkdir(parents=True, exist_ok=False)
        source_primary = Path(paths[f"{source}_primary"])
        source_controller = Path(paths[f"{source}_controller"])
        copies = {
            "command-timeline.json": (
                source_primary / "command-timeline.json"
            ),
            "manifest.sha256": source_primary / "manifest.sha256",
            "verify.remote.json": (
                source_primary
                / "verify.command-timeline.remote.json"
            ),
            "verify.local.json": (
                source_controller
                / "verify.command-timeline.local.json"
            ),
        }
        for name, source_path in copies.items():
            if not source_path.is_file():
                raise ValueError(
                    f"finalized {source} child input is missing"
                )
            shutil.copy2(source_path, destination / name)
    inputs = _parent_bundle_inputs(primary)
    baseline_artifact = json.loads(
        inputs["baseline_artifact_path"].read_text(encoding="utf-8")
    )
    candidate_artifact = json.loads(
        inputs["candidate_artifact_path"].read_text(encoding="utf-8")
    )
    baseline_receipts = {
        location: json.loads(path.read_text(encoding="utf-8"))
        for location, path in inputs[
            "baseline_receipt_paths"
        ].items()
    }
    candidate_receipts = {
        location: json.loads(path.read_text(encoding="utf-8"))
        for location, path in inputs[
            "candidate_receipt_paths"
        ].items()
    }
    artifact = build_source_pair_artifact(
        run_tag=tag,
        baseline_artifact=baseline_artifact,
        candidate_artifact=candidate_artifact,
        baseline_manifest_sha256=_sha256_path(
            inputs["baseline_manifest_path"]
        ),
        candidate_manifest_sha256=_sha256_path(
            inputs["candidate_manifest_path"]
        ),
        baseline_verifier_receipts=baseline_receipts,
        candidate_verifier_receipts=candidate_receipts,
    )
    _write_json_exclusive(inputs["artifact_path"], artifact)
    _write_json_exclusive(primary / "result.json", {
        "artifact_sha256": _sha256_path(inputs["artifact_path"]),
        "classification": artifact["classification"],
        "performance_improvement_established": artifact[
            "performance_improvement_established"
        ],
    })
    return 0


def _verify_parent(
    root: Path,
    *,
    manifest: bool,
    receipt_path: Path | None,
    verification_location: str,
) -> dict:
    from verify_autoregressive_draft_source_pair_gate import (
        verify_source_pair_gate,
    )

    inputs = _parent_bundle_inputs(root)
    receipt = verify_source_pair_gate(
        **inputs,
        manifest_path=(root / "manifest.sha256" if manifest else None),
    )
    receipt["verification_location"] = verification_location
    if receipt_path is not None:
        _write_json_exclusive(receipt_path, receipt)
    return receipt


def _remote_parent_manifest(tag: str) -> int:
    primary = Path(source_pair_paths(tag)["parent_primary"])
    rows = []
    detached = {
        "manifest.sha256",
        "verify.source-pair.remote.json",
        "verify.source-pair.remote.log",
        "verify.source-pair.local.json",
        "verify.source-pair.local.log",
    }
    for path in sorted(primary.rglob("*")):
        relative = path.relative_to(primary).as_posix()
        info = path.lstat()
        if stat.S_ISLNK(info.st_mode):
            raise ValueError("parent manifest contains a symlink")
        if stat.S_ISDIR(info.st_mode):
            continue
        if not stat.S_ISREG(info.st_mode):
            raise ValueError("parent manifest entry is not regular")
        if relative in detached:
            continue
        rows.append(f"{_sha256_path(path)}  {relative}")
    manifest = primary / "manifest.sha256"
    with manifest.open("x", encoding="utf-8") as handle:
        handle.write("\n".join(rows) + "\n")
    return 0


def _remote_parent_copy(tag: str, *, partial: bool) -> int:
    paths = source_pair_paths(tag)
    primary = Path(paths["parent_primary"])
    controller = Path(paths["parent_controller"])
    if controller.exists():
        raise ValueError("parent controller destination already exists")
    shutil.copytree(primary, controller, symlinks=False)
    if partial:
        _write_json_exclusive(
            controller / "partial-evidence.json",
            {"partial": True, "source": str(primary)},
        )
    return 0


def _normalize_parent_receipt(receipt: dict) -> dict:
    volatile = {
        "verified_at_utc",
        "verification_location",
        "artifact_path",
    }
    return {
        key: value for key, value in receipt.items() if key not in volatile
    }


def _remote_parent_action(action: str, arguments: list[str]) -> int:
    if len(arguments) != 1:
        raise ValueError("parent remote action arguments are invalid")
    tag = child_runner.validate_run_tag(arguments[0])
    paths = source_pair_paths(tag)
    primary = Path(paths["parent_primary"])
    controller = Path(paths["parent_controller"])
    if action == "assemble":
        return _remote_parent_assemble(tag)
    if action == "pre-manifest-verify":
        _verify_parent(
            primary,
            manifest=False,
            receipt_path=None,
            verification_location="remote",
        )
        return 0
    if action == "manifest":
        return _remote_parent_manifest(tag)
    if action == "primary-verify":
        _verify_parent(
            primary,
            manifest=True,
            receipt_path=(
                primary / "verify.source-pair.remote.json"
            ),
            verification_location="remote",
        )
        return 0
    if action == "controller-copy":
        return _remote_parent_copy(tag, partial=False)
    if action == "partial-copy":
        return _remote_parent_copy(tag, partial=True)
    if action == "controller-verify":
        _verify_parent(
            controller,
            manifest=True,
            receipt_path=(
                controller / "verify.source-pair.local.json"
            ),
            verification_location="local",
        )
        return 0
    if action == "compare-receipts":
        remote = json.loads(
            (
                primary / "verify.source-pair.remote.json"
            ).read_text(encoding="utf-8")
        )
        local = json.loads(
            (
                controller / "verify.source-pair.local.json"
            ).read_text(encoding="utf-8")
        )
        if canonical_json_bytes(
            _normalize_parent_receipt(remote)
        ) != canonical_json_bytes(_normalize_parent_receipt(local)):
            raise ValueError("parent verification receipts differ")
        return 0
    raise ValueError("parent remote action is invalid")


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("preflight", "execute"):
        subparser = subparsers.add_parser(command)
        subparser.add_argument("--run-tag", required=True)
    return parser.parse_args(argv)


def main(argv=None) -> int:
    arguments = sys.argv[1:] if argv is None else list(argv)
    if arguments and arguments[0] == "_remote-action":
        if len(arguments) < 2:
            raise ValueError("parent remote action is missing")
        return _remote_parent_action(arguments[1], arguments[2:])
    args = parse_args(arguments)
    result = (
        run_campaign(run_tag=args.run_tag)
        if args.command == "execute"
        else run_preflight(run_tag=args.run_tag)
    )
    sys.stdout.write(json.dumps(result, sort_keys=True) + "\n")
    return (
        0
        if result.get("status") in ("READY", "PASS")
        else 2
    )


if __name__ == "__main__":
    raise SystemExit(main())
