#!/usr/bin/env python3
"""Safely orchestrate the Qwen3.8 TP4 collective-reduction gate."""

from __future__ import annotations

import argparse
import json
from pathlib import PurePosixPath
import re
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
    query_remote_path_state,
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


APPROVED_REMOTE_ROOT = (
    "/data00/home/sitian/tinyllmforge-workspaces/"
    "command-timeline-20260818"
)
PLAN_SCHEMA_VERSION = "qwen38.tp4-collective-reduction-plan.v1"
EVENT_BUDGETS = (0, 8, 16, 32)
ATTEMPT_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
REVISION_PATTERN = re.compile(r"^[0-9a-f]{40}$")


def _path_below(path: str, root: str) -> bool:
    candidate = PurePosixPath(path)
    approved = PurePosixPath(root)
    return candidate.is_absolute() and candidate.is_relative_to(approved)


def _validate_revision(value, name):
    if not isinstance(value, str) or not REVISION_PATTERN.fullmatch(value):
        raise ValueError(f"{name} must be a lowercase 40-character SHA")
    return value


def _validate_remote_path_state(state):
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
    if state["attempt_exists"]:
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
    _validate_remote_path_state(remote_path_state)
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
    return parser


def main(
    argv=None,
    *,
    inventory_query: Callable[..., list[dict]] = (
        query_remote_gpu_inventory
    ),
    path_state_query: Callable[..., dict] = query_remote_path_state,
    kerberos_query: Callable[..., dict] = query_local_kerberos,
    gpu_monitor: Callable[..., dict] = wait_for_strict_clean_gpus,
    worker_runner=None,
    assembler=None,
    remote_verifier=None,
    downloader=None,
    local_verifier=None,
    cleanup_validator=None,
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
            print(json.dumps(result, indent=2, sort_keys=True))
            return 2
        kerberos = dict(raw_kerberos)
        kerberos["classification"] = "PASS"

    path_state = path_state_query(
        ssh_target=args.ssh_target,
        remote_root=args.remote_root,
        model_root=model_root,
        attempt_tag=args.attempt_tag,
        control_path=args.control_path,
        timeout_s=args.command_timeout_s,
        retry_count=args.retry_count,
    )
    if (
        not isinstance(path_state, dict)
        or set(path_state) != {"resolved_paths", "attempt_exists"}
        or not isinstance(path_state.get("resolved_paths"), dict)
        or set(path_state["resolved_paths"])
        != {"remote_root", "model_root", "attempt_root"}
    ):
        raise ValueError("remote path preflight result is invalid")
    resolved = path_state["resolved_paths"]
    remote_path_state = {
        "attempt_exists": path_state["attempt_exists"],
        "attempt_parent_is_symlink": (
            resolved["attempt_root"] != attempt_root
        ),
        "remote_root_is_symlink": (
            resolved["remote_root"] != args.remote_root
            or resolved["model_root"] != model_root
        ),
    }

    if args.plan_only:
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

    plan = build_attempt_plan(
        attempt_tag=args.attempt_tag,
        source_revision=args.source_revision,
        model_revision=args.model_revision,
        selected_gpus=selected,
        remote_path_state=remote_path_state,
        remote_root=args.remote_root,
    )
    if not args.plan_only and not args.dry_run:
        adapters = (
            worker_runner,
            assembler,
            remote_verifier,
            downloader,
            local_verifier,
            cleanup_validator,
        )
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
