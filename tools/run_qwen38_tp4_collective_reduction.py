#!/usr/bin/env python3
"""Safely orchestrate the Qwen3.8 TP4 collective-reduction gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import subprocess
from typing import Callable

from tools.run_qwen38_tp4_communication_profile import (
    DEFAULT_COMMAND_TIMEOUT_S,
    DEFAULT_GPU_POLL_INTERVAL_S,
    DEFAULT_GPU_WAIT_TIMEOUT_S,
    DEFAULT_RETRY_COUNT,
    DEFAULT_SSH_TARGET,
    MAX_GPU_MEMORY_USED_MIB,
    MAX_GPU_UTILIZATION_PERCENT,
    build_ssh_argv,
    classify_kerberos_ttl,
    parse_nvidia_smi_inventory,
    query_local_kerberos,
    query_remote_gpu_inventory,
    query_remote_gpu_topology,
    run_remote_argv,
    select_strict_clean_gpus,
    validate_selected_gpu_processes,
    wait_for_strict_clean_gpus,
    write_attempt_json_atomic,
    write_json_atomic,
)
from tools.qwen38_tp4_collective_reduction_worker import (
    build_collective_reduction_cases,
    collective_reduction_case_id,
)
from tools.qwen38_collective_reduction import EXPECTED_PROFILE


APPROVED_REMOTE_ROOT = (
    "/data00/home/sitian/tinyllmforge-workspaces/"
    "command-timeline-20260818"
)
PLAN_SCHEMA_VERSION = "qwen38.tp4-collective-reduction-plan.v1"
EVENT_BUDGETS = (0, 8, 16, 32)
ATTEMPT_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
REVISION_PATTERN = re.compile(r"^[0-9a-f]{40}$")
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
SOURCE_ARCHIVE_PATHS = ("tinyvllm", "tools")
MODEL_REPOSITORY = "Qwen/Qwen3.8-27B"
MODEL_MANIFEST_SCHEMA = "tinyllmforge.qwen38-model-manifest.v1"


def _path_below(path: str, root: str) -> bool:
    candidate = PurePosixPath(path)
    approved = PurePosixPath(root)
    return candidate.is_absolute() and candidate.is_relative_to(approved)


def _validate_revision(value, name):
    if not isinstance(value, str) or not REVISION_PATTERN.fullmatch(value):
        raise ValueError(f"{name} must be a lowercase 40-character SHA")
    return value


def build_source_identity(
    *,
    attempt,
    source_revision,
    source_files,
):
    if (
        not isinstance(attempt, str)
        or not ATTEMPT_PATTERN.fullmatch(attempt)
        or ".." in attempt
    ):
        raise ValueError("attempt is invalid")
    source_revision = _validate_revision(
        source_revision,
        "source_revision",
    )
    if not isinstance(source_files, dict) or not source_files:
        raise ValueError("source file inventory is invalid")
    normalized = {}
    for relative, digest in sorted(source_files.items()):
        if not isinstance(relative, str):
            raise ValueError("source file inventory is invalid")
        path = PurePosixPath(relative)
        if (
            path.is_absolute()
            or ".." in path.parts
            or path.as_posix() != relative
            or not isinstance(digest, str)
            or not SHA256_PATTERN.fullmatch(digest)
        ):
            raise ValueError("source file inventory is invalid")
        normalized[relative] = digest
    canonical = json.dumps(
        normalized,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return {
        "schema_version": (
            "qwen38.tp4-collective-reduction-source.v1"
        ),
        "attempt": attempt,
        "source_revision": source_revision,
        "source_tree_sha256": hashlib.sha256(canonical).hexdigest(),
        "source_files": normalized,
        "source_archive_paths": list(SOURCE_ARCHIVE_PATHS),
    }


def capture_source_identity(
    *,
    attempt,
    source_revision,
    repo_root=None,
    command_runner=subprocess.run,
):
    if not callable(command_runner):
        raise ValueError("source command runner is invalid")
    root = (
        Path(__file__).resolve().parents[1]
        if repo_root is None
        else Path(repo_root).resolve()
    )

    def run_git(arguments):
        result = command_runner(
            ["git", "-C", str(root), *arguments],
            text=True,
            capture_output=True,
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError(
                result.stderr or "source identity git command failed"
            )
        return result.stdout

    head = run_git(["rev-parse", "HEAD"]).strip()
    if head != source_revision:
        raise ValueError("source revision does not match local HEAD")
    dirty = run_git([
        "status",
        "--porcelain=v1",
        "--untracked-files=no",
        "--",
        *SOURCE_ARCHIVE_PATHS,
    ])
    if dirty:
        raise ValueError("source archive scope has tracked changes")
    raw_files = run_git([
        "ls-files",
        "-z",
        "--",
        *SOURCE_ARCHIVE_PATHS,
    ])
    source_files = {}
    for relative in raw_files.split("\0"):
        if not relative:
            continue
        path = root / relative
        payload = (
            os.readlink(path).encode("utf-8")
            if path.is_symlink()
            else path.read_bytes()
        )
        source_files[relative] = hashlib.sha256(payload).hexdigest()
    return build_source_identity(
        attempt=attempt,
        source_revision=source_revision,
        source_files=source_files,
    )


def query_remote_collective_path_state(
    *,
    ssh_target,
    remote_root,
    model_root,
    model_revision,
    attempt_tag,
    timeout_s,
    retry_count,
    control_path=None,
    command_runner=subprocess.run,
):
    if remote_root != APPROVED_REMOTE_ROOT:
        raise ValueError("remote_root is not approved")
    model_revision = _validate_revision(
        model_revision,
        "model_revision",
    )
    if (
        not _path_below(model_root, remote_root)
        or PurePosixPath(model_root).name != model_revision
        or not isinstance(attempt_tag, str)
        or not ATTEMPT_PATTERN.fullmatch(attempt_tag)
        or ".." in attempt_tag
    ):
        raise ValueError("remote model or attempt path is invalid")
    attempt_root = f"{remote_root}/attempts/{attempt_tag}"
    script = "\n".join([
        "import json,os,sys",
        (
            "remote_root,model_root,attempt_root,model_revision="
            "sys.argv[1:]"
        ),
        "config_path=os.path.join(model_root,'config.json')",
        (
            "index_path=os.path.join("
            "model_root,'model.safetensors.index.json')"
        ),
        "config_readable=os.path.isfile(config_path) and os.access(config_path,os.R_OK)",
        "index_readable=os.path.isfile(index_path) and os.access(index_path,os.R_OK)",
        "config={}",
        "index={}",
        "if config_readable:",
        "  with open(config_path,encoding='utf-8') as handle: config=json.load(handle)",
        "if index_readable:",
        "  with open(index_path,encoding='utf-8') as handle: index=json.load(handle)",
        "text=config.get('text_config',{}) if isinstance(config,dict) else {}",
        "weight_map=index.get('weight_map',{}) if isinstance(index,dict) else {}",
        (
            "shards=sorted(set(weight_map.values())) "
            "if isinstance(weight_map,dict) else []"
        ),
        (
            "all_shards=bool(shards) and all("
            "os.path.isfile(os.path.join(model_root,name)) and "
            "os.access(os.path.join(model_root,name),os.R_OK) "
            "for name in shards)"
        ),
        "snapshot_revision=os.path.basename(os.path.realpath(model_root))",
        "print(json.dumps({",
        "'resolved_paths':{",
        "'remote_root':os.path.realpath(remote_root),",
        "'model_root':os.path.realpath(model_root),",
        "'attempt_root':os.path.realpath(attempt_root),",
        "},",
        "'attempt_exists':os.path.lexists(attempt_root),",
        (
            "'remote_root_ready':os.path.isdir(remote_root) "
            "and os.access(remote_root,os.R_OK|os.W_OK|os.X_OK),"
        ),
        "'model_manifest':{",
        f"'schema_version':{MODEL_MANIFEST_SCHEMA!r},",
        f"'repository':{MODEL_REPOSITORY!r},",
        "'revision':model_revision,",
        "'text_profile':{",
        "'num_hidden_layers':text.get('num_hidden_layers'),",
        "'hidden_size':text.get('hidden_size'),",
        "'vocab_size':text.get('vocab_size'),",
        "'dtype':text.get('dtype'),",
        "},",
        "},",
        "'model_files':{",
        "'config_readable':config_readable,",
        "'weight_index_readable':index_readable,",
        "'weight_shard_count':len(shards),",
        "'all_weight_shards_readable':all_shards,",
        "'snapshot_revision':snapshot_revision,",
        (
            "'snapshot_revision_matches':"
            "snapshot_revision==model_revision,"
        ),
        "},",
        "},sort_keys=True))",
    ])
    result = run_remote_argv(
        ssh_target=ssh_target,
        remote_argv=[
            "python3",
            "-c",
            script,
            remote_root,
            model_root,
            attempt_root,
            model_revision,
        ],
        control_path=control_path,
        timeout_s=timeout_s,
        retry_count=retry_count,
        command_runner=command_runner,
    )
    if result.returncode != 0:
        raise RuntimeError(
            getattr(result, "stderr", "")
            or "remote collective path preflight failed"
        )
    try:
        payload = json.loads(result.stdout)
    except (TypeError, json.JSONDecodeError) as error:
        raise ValueError(
            "remote collective path preflight JSON is invalid"
        ) from error
    if not isinstance(payload, dict):
        raise ValueError(
            "remote collective path preflight JSON is invalid"
        )
    return payload


def _validate_collective_path_state(
    path_state,
    *,
    remote_root,
    model_root,
    model_revision,
    attempt_root,
    allow_existing_attempt=False,
):
    if type(allow_existing_attempt) is not bool:
        raise ValueError("attempt resume policy is invalid")
    if (
        not isinstance(path_state, dict)
        or set(path_state) != {
            "resolved_paths",
            "attempt_exists",
            "remote_root_ready",
            "model_manifest",
            "model_files",
        }
        or not isinstance(path_state.get("resolved_paths"), dict)
        or set(path_state["resolved_paths"])
        != {"remote_root", "model_root", "attempt_root"}
    ):
        raise ValueError("remote model preflight result is invalid")
    resolved = path_state["resolved_paths"]
    manifest = path_state["model_manifest"]
    files = path_state["model_files"]
    if (
        type(path_state["attempt_exists"]) is not bool
        or path_state["remote_root_ready"] is not True
        or not isinstance(manifest, dict)
        or manifest.get("schema_version") != MODEL_MANIFEST_SCHEMA
        or manifest.get("repository") != MODEL_REPOSITORY
        or manifest.get("revision") != model_revision
        or manifest.get("text_profile") != EXPECTED_PROFILE
        or not isinstance(files, dict)
        or files.get("config_readable") is not True
        or files.get("weight_index_readable") is not True
        or type(files.get("weight_shard_count")) is not int
        or files["weight_shard_count"] <= 0
        or files.get("all_weight_shards_readable") is not True
        or files.get("snapshot_revision") != model_revision
        or files.get("snapshot_revision_matches") is not True
    ):
        raise ValueError("remote model preflight result is invalid")
    if (
        resolved.get("remote_root") != remote_root
        or resolved.get("model_root") != model_root
        or resolved.get("attempt_root") != attempt_root
    ):
        raise ValueError("remote path preflight result is invalid")
    if path_state["attempt_exists"] and not allow_existing_attempt:
        raise ValueError("attempt tag is already in use")
    return {
        "attempt_exists": path_state["attempt_exists"],
        "attempt_parent_is_symlink": False,
        "remote_root_is_symlink": False,
    }


def _validate_remote_path_state(
    state,
    *,
    allow_existing_attempt=False,
):
    expected = {
        "attempt_exists",
        "attempt_parent_is_symlink",
        "remote_root_is_symlink",
    }
    if (
        not isinstance(state, dict)
        or set(state) != expected
        or any(type(state[name]) is not bool for name in expected)
    ):
        raise ValueError("remote path state is invalid")
    if state["attempt_exists"] and not allow_existing_attempt:
        raise ValueError("attempt tag is already in use")
    if (
        state["attempt_parent_is_symlink"]
        or state["remote_root_is_symlink"]
    ):
        raise ValueError("remote symlink escape detected")


def _validate_selected_gpus(selected_gpus):
    selected = select_strict_clean_gpus(list(selected_gpus))
    if len(selected) != 4:
        raise ValueError("four strict-clean GPUs are required")
    return tuple(dict(row) for row in selected)


def build_attempt_plan(
    *,
    attempt_tag,
    source_revision,
    model_revision,
    selected_gpus,
    remote_path_state,
    remote_root=APPROVED_REMOTE_ROOT,
    allow_existing_attempt=False,
):
    if remote_root != APPROVED_REMOTE_ROOT:
        raise ValueError("remote_root is not approved")
    if (
        not isinstance(attempt_tag, str)
        or not ATTEMPT_PATTERN.fullmatch(attempt_tag)
        or ".." in attempt_tag
    ):
        raise ValueError("attempt_tag is unsafe")
    source_revision = _validate_revision(
        source_revision,
        "source_revision",
    )
    model_revision = _validate_revision(
        model_revision,
        "model_revision",
    )
    if type(allow_existing_attempt) is not bool:
        raise ValueError("attempt resume policy is invalid")
    _validate_remote_path_state(
        remote_path_state,
        allow_existing_attempt=allow_existing_attempt,
    )
    selected = _validate_selected_gpus(selected_gpus)

    attempt_root = f"{remote_root}/attempts/{attempt_tag}"
    paths = {
        "attempt_root": attempt_root,
        "source_root": f"{attempt_root}/source",
        "model_root": (
            f"{remote_root}/models/Qwen3.8-27B/"
            f"snapshots/{model_revision}"
        ),
        "case_root": f"{attempt_root}/cases",
        "bundle_root": f"{attempt_root}/final_bundle",
        "controller_log_path": (
            f"{attempt_root}/controller/controller.jsonl"
        ),
    }
    if not all(
        _path_below(path, remote_root)
        for path in paths.values()
    ):
        raise ValueError("planned remote path escapes approved root")
    worker_argv = [
        "python3",
        f"{paths['source_root']}/"
        "tools/qwen38_tp4_collective_reduction_worker.py",
        "--attempt",
        attempt_tag,
        "--source-revision",
        source_revision,
        "--model-root",
        paths["model_root"],
        "--output",
        f"{attempt_root}/worker.json",
        "--output-dir",
        paths["case_root"],
        "--phase",
        "full",
    ]
    return {
        "schema_version": PLAN_SCHEMA_VERSION,
        "attempt_tag": attempt_tag,
        "source_revision": source_revision,
        "model_revision": model_revision,
        "remote_root": remote_root,
        **paths,
        "selected_gpus": [dict(row) for row in selected],
        "event_budgets": list(EVENT_BUDGETS),
        "median_overhead_ceiling": 0.03,
        "maximum_overhead_ceiling": 0.05,
        "minimum_lower_bound_opportunity": 0.05,
        "overlap_design_authorized": False,
        "async_collectives_authorized": False,
        "resume_existing_attempt": (
            remote_path_state["attempt_exists"]
        ),
        "remote_commands": [
            {
                "purpose": "create attempt directories",
                "argv": [
                    "mkdir",
                    "-p",
                    f"{attempt_root}/controller",
                    paths["case_root"],
                    paths["bundle_root"],
                ],
            },
            {
                "purpose": "run qualification worker",
                "argv": worker_argv,
            },
            {
                "purpose": "assemble terminal bundle",
                "argv": [
                    "python3",
                    f"{paths['source_root']}/"
                    "tools/assemble_qwen38_tp4_collective_reduction.py",
                    "--attempt-root",
                    attempt_root,
                ],
            },
            {
                "purpose": "verify terminal bundle",
                "argv": [
                    "python3",
                    f"{paths['source_root']}/"
                    "tools/verify_qwen38_tp4_collective_reduction.py",
                    "--bundle-root",
                    paths["bundle_root"],
                ],
            },
        ],
    }


def _validate_plan(plan):
    if (
        not isinstance(plan, dict)
        or plan.get("schema_version") != PLAN_SCHEMA_VERSION
        or plan.get("remote_root") != APPROVED_REMOTE_ROOT
        or plan.get("overlap_design_authorized") is not False
        or plan.get("async_collectives_authorized") is not False
        or plan.get("event_budgets") != list(EVENT_BUDGETS)
        or type(plan.get("resume_existing_attempt")) is not bool
    ):
        raise ValueError("collective reduction plan is invalid")
    for key in (
        "attempt_root",
        "source_root",
        "model_root",
        "case_root",
        "bundle_root",
        "controller_log_path",
    ):
        if not _path_below(plan.get(key, ""), APPROVED_REMOTE_ROOT):
            raise ValueError("collective reduction plan path is invalid")
    selected = _validate_selected_gpus(plan.get("selected_gpus", ()))
    return selected


def expected_case_ids(selected_budget):
    matrix = build_collective_reduction_cases(
        selected_budget=16 if selected_budget is None else selected_budget
    )
    rows = list(matrix["calibration"])
    if selected_budget is not None:
        rows.extend(matrix["terminal"])
    return tuple(
        collective_reduction_case_id(**{
            key: row[key]
            for key in (
                "campaign_phase",
                "workload",
                "phase",
                "repetition",
                "budget",
            )
        })
        for row in rows
    )


def _validate_worker_result(plan, worker):
    if (
        not isinstance(worker, dict)
        or worker.get("classification") != "PASS"
        or worker.get("attempt") != plan["attempt_tag"]
        or worker.get("source_revision") != plan["source_revision"]
    ):
        raise RuntimeError("collective reduction worker failed")
    selected_budget = worker.get("selected_budget")
    if selected_budget is not None and selected_budget not in EVENT_BUDGETS[1:]:
        raise RuntimeError("collective reduction budget is invalid")
    rows = worker.get("cases")
    if not isinstance(rows, list):
        raise RuntimeError("collective reduction case coverage is invalid")
    case_ids = []
    for row in rows:
        if (
            not isinstance(row, dict)
            or not isinstance(row.get("case_id"), str)
            or row.get("classification") != "PASS"
        ):
            raise RuntimeError(
                "collective reduction case coverage is invalid"
            )
        case_ids.append(row["case_id"])
    if (
        len(case_ids) != len(set(case_ids))
        or tuple(case_ids) != expected_case_ids(selected_budget)
    ):
        raise RuntimeError(
            "collective reduction case coverage is incomplete"
        )
    phase_cleanups = worker.get("phase_cleanups")
    expected_cleanup_count = 1 if selected_budget is None else 2
    if (
        not isinstance(phase_cleanups, list)
        or len(phase_cleanups) != expected_cleanup_count
        or any(
            not isinstance(receipt, dict)
            or receipt.get("process_group_destroyed") is not True
            or receipt.get("rank_exit_codes") != [0, 0, 0, 0]
            or receipt.get("owned_children_remaining") != []
            for receipt in phase_cleanups
        )
    ):
        raise RuntimeError("collective reduction worker cleanup is incomplete")
    owned_pids = worker.get("owned_pids")
    if (
        not isinstance(owned_pids, list)
        or any(type(pid) is not int or pid <= 0 for pid in owned_pids)
        or len(owned_pids) != len(set(owned_pids))
    ):
        raise RuntimeError("collective reduction worker ownership is invalid")
    return dict(worker)


def run_attempt(
    plan,
    *,
    plan_only,
    dry_run=False,
    kerberos_probe=None,
    gpu_probe=None,
    worker_runner=None,
    assembler=None,
    remote_verifier=None,
    downloader=None,
    local_verifier=None,
    cleanup_validator=None,
):
    if not isinstance(plan_only, bool) or not isinstance(dry_run, bool):
        raise ValueError("plan_only and dry_run must be booleans")
    selected = _validate_plan(plan)
    if plan_only:
        return {
            "classification": "PLAN_ONLY",
            "worker_started": False,
            "plan": plan,
        }
    if not callable(kerberos_probe) or not callable(gpu_probe):
        raise RuntimeError("resource probes are required")
    kerberos = kerberos_probe()
    if kerberos.get("classification") != "PASS":
        return {
            "classification": "BLOCKED_KERBEROS",
            "worker_started": False,
            "kerberos": kerberos,
        }
    if not plan["resume_existing_attempt"]:
        observed = list(gpu_probe())
        select_strict_clean_gpus(observed)
        validate_selected_gpu_processes(
            selected=selected,
            observed=observed,
            owned_pids=set(),
        )
    if dry_run:
        return {
            "classification": "DRY_RUN_READY",
            "worker_started": False,
            "kerberos": kerberos,
            "selected_gpus": [dict(row) for row in selected],
            "plan": plan,
        }
    required = {
        "worker_runner": worker_runner,
        "assembler": assembler,
        "remote_verifier": remote_verifier,
        "downloader": downloader,
        "local_verifier": local_verifier,
        "cleanup_validator": cleanup_validator,
    }
    missing = [
        name for name, callback in required.items()
        if not callable(callback)
    ]
    if missing:
        raise RuntimeError(
            "execution adapters are missing: " + ", ".join(missing)
        )

    worker = worker_runner(plan)
    operation_error = None
    producer = None
    remote = None
    local = None
    try:
        worker = _validate_worker_result(plan, worker)
        producer = assembler(plan, worker)
        remote = remote_verifier(plan)
        download = downloader(plan)
        if (
            not isinstance(download, dict)
            or download.get("downloaded") is not True
        ):
            raise RuntimeError(
                "collective reduction download is incomplete"
            )
        local = local_verifier(plan)
        if not all(
            isinstance(row, dict)
            for row in (producer, remote, local)
        ):
            raise RuntimeError(
                "producer/verifier result is invalid"
            )
    except Exception as error:
        operation_error = error
    cleanup = cleanup_validator(plan, worker)
    if (
        not isinstance(cleanup, dict)
        or cleanup.get("complete") is not True
        or cleanup.get("owned_children_remaining", []) != []
    ):
        raise RuntimeError(
            "collective reduction cleanup is incomplete"
        ) from operation_error
    if operation_error is not None:
        raise operation_error
    assert producer is not None
    assert remote is not None
    assert local is not None
    classifications = {
        producer.get("classification"),
        remote.get("classification"),
        local.get("classification"),
    }
    if len(classifications) != 1 or None in classifications:
        raise RuntimeError(
            "producer/verifier classification disagreement"
        )
    return {
        "classification": classifications.pop(),
        "worker_started": True,
        "producer": producer,
        "remote_verification": remote,
        "local_verification": local,
        "cleanup": cleanup,
    }


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ssh-target", default=DEFAULT_SSH_TARGET)
    parser.add_argument("--remote-root", default=APPROVED_REMOTE_ROOT)
    parser.add_argument("--attempt-tag", required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--control-path")
    parser.add_argument(
        "--command-timeout-s",
        type=int,
        default=DEFAULT_COMMAND_TIMEOUT_S,
    )
    parser.add_argument(
        "--retry-count",
        type=int,
        default=DEFAULT_RETRY_COUNT,
    )
    parser.add_argument(
        "--gpu-wait-timeout-s",
        type=int,
        default=DEFAULT_GPU_WAIT_TIMEOUT_S,
    )
    parser.add_argument(
        "--gpu-poll-interval-s",
        type=int,
        default=DEFAULT_GPU_POLL_INTERVAL_S,
    )
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--local-attempt-root", type=Path)
    return parser


def main(
    argv=None,
    *,
    inventory_query: Callable[..., list[dict]] = (
        query_remote_gpu_inventory
    ),
    path_state_query: Callable[..., dict] = (
        query_remote_collective_path_state
    ),
    kerberos_query: Callable[..., dict] = query_local_kerberos,
    gpu_monitor: Callable[..., dict] = wait_for_strict_clean_gpus,
    source_identity_builder: Callable[..., dict] = (
        capture_source_identity
    ),
    worker_runner=None,
    assembler=None,
    remote_verifier=None,
    downloader=None,
    local_verifier=None,
    cleanup_validator=None,
    production_adapter_factory=None,
):
    parser = build_parser()
    args = parser.parse_args(argv)
    if not all(
        callable(callback)
        for callback in (
            inventory_query,
            path_state_query,
            kerberos_query,
            gpu_monitor,
        )
    ):
        raise ValueError("CLI query dependency is invalid")
    local_controller_root = None
    source_identity = None
    source_identity_path = None
    if args.local_attempt_root is not None:
        if not callable(source_identity_builder):
            raise ValueError("source identity dependency is invalid")
        local_controller_root = (
            args.local_attempt_root.resolve() / "controller"
        )
        source_identity = source_identity_builder(
            attempt=args.attempt_tag,
            source_revision=args.source_revision,
        )
        source_identity_path = (
            local_controller_root / "source_identity.json"
        )

    model_root = (
        f"{args.remote_root}/models/Qwen3.8-27B/"
        f"snapshots/{args.model_revision}"
    )
    attempt_root = (
        f"{args.remote_root}/attempts/{args.attempt_tag}"
    )

    def query_inventory():
        return inventory_query(
            ssh_target=args.ssh_target,
            control_path=args.control_path,
            timeout_s=args.command_timeout_s,
            retry_count=args.retry_count,
        )

    kerberos = {"classification": "PASS"}
    if not args.plan_only:
        raw_kerberos = kerberos_query()
        if (
            not isinstance(raw_kerberos, dict)
            or raw_kerberos.get("classification")
            not in {"PASS", "READY"}
        ):
            result = {
                "classification": "BLOCKED_KERBEROS",
                "worker_started": False,
                "kerberos": raw_kerberos,
            }
            if local_controller_root is not None:
                if source_identity_path.exists():
                    if (
                        source_identity_path.is_symlink()
                        or not source_identity_path.is_file()
                    ):
                        raise ValueError(
                            "frozen local source identity is invalid"
                        )
                    try:
                        frozen_source_identity = json.loads(
                            source_identity_path.read_text(
                                encoding="utf-8"
                            )
                        )
                    except json.JSONDecodeError as error:
                        raise ValueError(
                            "frozen local source identity is invalid"
                        ) from error
                    if frozen_source_identity != source_identity:
                        raise ValueError(
                            "frozen local source identity mismatch"
                        )
                else:
                    write_json_atomic(
                        source_identity_path,
                        source_identity,
                    )
                write_json_atomic(
                    local_controller_root
                    / "ssh_storage_preflight.json",
                    {
                        "classification": "BLOCKED_KERBEROS",
                        "kerberos": raw_kerberos,
                        "remote_query_performed": False,
                        "remote_write_performed": False,
                    },
                )
                write_json_atomic(
                    local_controller_root / "dry_run.json",
                    result,
                )
            print(json.dumps(result, indent=2, sort_keys=True))
            return 2
        kerberos = dict(raw_kerberos)
        kerberos["classification"] = "PASS"

    path_state = path_state_query(
        ssh_target=args.ssh_target,
        remote_root=args.remote_root,
        model_root=model_root,
        model_revision=args.model_revision,
        attempt_tag=args.attempt_tag,
        control_path=args.control_path,
        timeout_s=args.command_timeout_s,
        retry_count=args.retry_count,
    )
    remote_path_state = _validate_collective_path_state(
        path_state,
        remote_root=args.remote_root,
        model_root=model_root,
        model_revision=args.model_revision,
        attempt_root=attempt_root,
        allow_existing_attempt=(
            not args.plan_only and not args.dry_run
        ),
    )
    if (
        not path_state["attempt_exists"]
        and source_identity_path is not None
    ):
        write_json_atomic(source_identity_path, source_identity)

    if path_state["attempt_exists"]:
        if local_controller_root is None:
            raise ValueError(
                "existing attempt resume requires local attempt root"
            )
        if (
            source_identity_path is None
            or not source_identity_path.is_file()
            or source_identity_path.is_symlink()
        ):
            raise ValueError(
                "existing attempt resume requires frozen source identity"
            )
        try:
            frozen_source_identity = json.loads(
                source_identity_path.read_text(encoding="utf-8")
            )
        except json.JSONDecodeError as error:
            raise ValueError(
                "frozen local source identity is invalid"
            ) from error
        if frozen_source_identity != source_identity:
            raise ValueError("frozen local source identity mismatch")
        plan_path = local_controller_root / "plan.json"
        if not plan_path.is_file() or plan_path.is_symlink():
            raise ValueError(
                "existing attempt resume requires frozen local plan"
            )
        try:
            plan = json.loads(plan_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            raise ValueError("frozen local plan is invalid") from error
        plan = dict(plan)
        plan["resume_existing_attempt"] = True
        _validate_plan(plan)
        if (
            plan.get("attempt_tag") != args.attempt_tag
            or plan.get("source_revision") != args.source_revision
            or plan.get("model_revision") != args.model_revision
            or plan.get("remote_root") != args.remote_root
        ):
            raise ValueError("frozen local plan identity mismatch")
        selected = tuple(
            dict(row) for row in plan["selected_gpus"]
        )
    elif args.plan_only:
        selected = select_strict_clean_gpus(query_inventory())
    else:
        monitor = gpu_monitor(
            query_inventory=query_inventory,
            timeout_s=args.gpu_wait_timeout_s,
            poll_interval_s=args.gpu_poll_interval_s,
        )
        if (
            not isinstance(monitor, dict)
            or monitor.get("classification") != "READY"
            or not isinstance(monitor.get("selected_gpus"), list)
        ):
            result = {
                "classification": "BLOCKED_RESOURCES",
                "worker_started": False,
                "monitor": monitor,
            }
            print(json.dumps(result, indent=2, sort_keys=True))
            return 2
        selected = _validate_selected_gpus(
            monitor["selected_gpus"]
        )

    if not path_state["attempt_exists"]:
        plan = build_attempt_plan(
            attempt_tag=args.attempt_tag,
            source_revision=args.source_revision,
            model_revision=args.model_revision,
            selected_gpus=selected,
            remote_path_state=remote_path_state,
            remote_root=args.remote_root,
        )
    if (
        local_controller_root is not None
        and not args.plan_only
        and not args.dry_run
        and not path_state["attempt_exists"]
    ):
        write_json_atomic(local_controller_root / "plan.json", plan)
    if not args.plan_only and not args.dry_run:
        adapters = [
            worker_runner,
            assembler,
            remote_verifier,
            downloader,
            local_verifier,
            cleanup_validator,
        ]
        if not any(callable(callback) for callback in adapters):
            if (
                local_controller_root is not None
                and source_identity is not None
            ):
                if production_adapter_factory is None:
                    from tools.qwen38_tp4_collective_reduction_production import (
                        create_production_adapter,
                    )

                    production_adapter_factory = (
                        create_production_adapter
                    )
                adapter = production_adapter_factory(
                    plan=plan,
                    source_identity=source_identity,
                    model_manifest=dict(path_state["model_manifest"]),
                    repo_root=Path(__file__).resolve().parents[1],
                    local_attempt_root=args.local_attempt_root,
                    ssh_target=args.ssh_target,
                    control_path=args.control_path,
                    command_timeout_s=args.command_timeout_s,
                    retry_count=args.retry_count,
                )
                adapters = [
                    adapter.worker_runner,
                    adapter.assembler,
                    adapter.remote_verifier,
                    adapter.downloader,
                    adapter.local_verifier,
                    adapter.cleanup_validator,
                ]
                (
                    worker_runner,
                    assembler,
                    remote_verifier,
                    downloader,
                    local_verifier,
                    cleanup_validator,
                ) = adapters
        if not all(callable(callback) for callback in adapters):
            result = {
                "classification": "EXECUTION_ADAPTER_UNAVAILABLE",
                "worker_started": False,
                "plan": plan,
            }
            print(json.dumps(result, indent=2, sort_keys=True))
            return 2
    result = run_attempt(
        plan,
        plan_only=args.plan_only,
        dry_run=args.dry_run,
        kerberos_probe=lambda: kerberos,
        gpu_probe=query_inventory,
        worker_runner=worker_runner,
        assembler=assembler,
        remote_verifier=remote_verifier,
        downloader=downloader,
        local_verifier=local_verifier,
        cleanup_validator=cleanup_validator,
    )
    if local_controller_root is not None and args.dry_run:
        write_json_atomic(
            local_controller_root / "plan.json",
            plan,
        )
        write_json_atomic(
            local_controller_root / "plan_audit.json",
            {
                "schema_version": (
                    "qwen38.tp4-collective-reduction-plan-audit.v1"
                ),
                "classification": "PASS",
                "attempt_tag": plan["attempt_tag"],
                "source_revision": plan["source_revision"],
                "remote_paths_below_approved_root": all(
                    _path_below(plan[key], APPROVED_REMOTE_ROOT)
                    for key in (
                        "attempt_root",
                        "source_root",
                        "model_root",
                        "case_root",
                        "bundle_root",
                        "controller_log_path",
                    )
                ),
                "attempt_absent": not path_state["attempt_exists"],
                "overlap_design_authorized": (
                    plan["overlap_design_authorized"]
                ),
                "async_collectives_authorized": (
                    plan["async_collectives_authorized"]
                ),
            },
        )
        write_json_atomic(
            local_controller_root / "ssh_storage_preflight.json",
            {
                "schema_version": (
                    "qwen38.tp4-collective-reduction-preflight.v1"
                ),
                "classification": "PASS",
                "kerberos": kerberos,
                "resolved_paths": dict(path_state["resolved_paths"]),
                "attempt_exists": path_state["attempt_exists"],
                "remote_root_ready": path_state["remote_root_ready"],
                "model_manifest": dict(path_state["model_manifest"]),
                "model_files": dict(path_state["model_files"]),
                "remote_query_performed": True,
                "remote_write_performed": False,
            },
        )
        write_json_atomic(
            local_controller_root / "strict_clean_admission.json",
            {
                "schema_version": (
                    "qwen38.tp4-collective-reduction-gpu-admission.v1"
                ),
                "classification": "READY",
                "selected_gpus": [dict(row) for row in selected],
                "maximum_memory_used_mib": MAX_GPU_MEMORY_USED_MIB,
                "maximum_utilization_percent": (
                    MAX_GPU_UTILIZATION_PERCENT
                ),
                "compute_processes_required_empty": True,
                "worker_started": False,
            },
        )
        write_json_atomic(
            local_controller_root / "dry_run.json",
            result,
        )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["classification"] in {
        "PLAN_ONLY",
        "DRY_RUN_READY",
        "GO_SYNC_COLLECTIVE_REDUCTION",
        "NO_GO_NO_REDUCIBLE_COLLECTIVE",
        "INCONCLUSIVE_PROFILER_OVERHEAD",
        "INCONCLUSIVE_INCOMPLETE_COVERAGE",
    } else 2


if __name__ == "__main__":
    raise SystemExit(main())
