#!/usr/bin/env python3
"""Orchestrate the immutable Qwen3.8 TP2 whole-model gate."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import subprocess
import time

if __package__:
    from tools.run_qwen38_topology_local_tp2_island import (
        _select_best_pair_groups,
        query_local_kerberos,
        select_strict_clean_gpus,
    )
else:
    from run_qwen38_topology_local_tp2_island import (
        _select_best_pair_groups,
        query_local_kerberos,
        select_strict_clean_gpus,
    )


APPROVED_REMOTE_ROOT = (
    "/data00/home/sitian/tinyllmforge-workspaces/"
    "command-timeline-20260818"
)
MODEL_REPOSITORY = "Qwen/Qwen3.8-27B"
MODEL_REVISION = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
MINIMUM_KERBEROS_LIFETIME_SECONDS = 1_800
EXPECTED_KERBEROS_PRINCIPAL = "sitian@BYTEDANCE.COM"
EXPECTED_KERBEROS_TGT = "krbtgt/BYTEDANCE.COM@BYTEDANCE.COM"
PLAN_SCHEMA = "qwen38.topology-local-tp2-whole-model-plan.v1"
ATTEMPT_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
HEX40_PATTERN = re.compile(r"^[0-9a-f]{40}$")
HEX64_PATTERN = re.compile(r"^[0-9a-f]{64}$")
WORKLOADS = {
    "P0": ["causal", 256, 128, 1],
    "P1": ["causal", 2048, 128, 1],
    "Q0": ["online", 256, 128, 4],
    "Q1": ["online", 256, 128, 8],
    "Q2": ["online", 2048, 128, 4],
}


def _below(path, root):
    candidate = PurePosixPath(path)
    parent = PurePosixPath(root)
    return (
        candidate.is_absolute()
        and candidate != parent
        and candidate.is_relative_to(parent)
    )


def _strict_inventory(rows):
    try:
        selected = select_strict_clean_gpus(list(rows))
    except ValueError as error:
        raise ValueError("exactly four strict-clean GPUs are required") from error
    if len(selected) != 4:
        raise ValueError("exactly four strict-clean GPUs are required")
    return tuple(dict(row) for row in selected)


def build_plan(
    *,
    attempt_tag: str,
    source_revision: str,
    gpu_inventory: tuple[dict, ...],
    topology,
    source_tree_sha256: str | None = None,
    remote_root: str = APPROVED_REMOTE_ROOT,
    attempt_exists: bool = False,
) -> dict:
    if remote_root != APPROVED_REMOTE_ROOT:
        raise ValueError("remote root is not approved")
    if (
        not isinstance(attempt_tag, str)
        or not ATTEMPT_PATTERN.fullmatch(attempt_tag)
        or ".." in attempt_tag
    ):
        raise ValueError("attempt tag is invalid")
    if attempt_exists:
        raise ValueError("attempt tag must be fresh")
    if (
        not isinstance(source_revision, str)
        or not HEX40_PATTERN.fullmatch(source_revision)
    ):
        raise ValueError("source revision is invalid")
    if source_tree_sha256 is None:
        source_tree_sha256 = hashlib.sha256(
            source_revision.encode("ascii")
        ).hexdigest()
    if not HEX64_PATTERN.fullmatch(str(source_tree_sha256)):
        raise ValueError("source tree SHA-256 is invalid")
    selected = _strict_inventory(gpu_inventory)
    topology_rows = topology.get("rows") if isinstance(topology, dict) else None
    if not isinstance(topology_rows, list):
        raise ValueError("topology inventory is invalid")
    pair_groups = [
        list(pair) for pair in _select_best_pair_groups(topology_rows)
    ]
    attempt_root = f"{remote_root}/attempts/{attempt_tag}"
    runtime_root = f"{attempt_root}/runtime"
    paths = {
        "attempt_root": attempt_root,
        "source_root": f"{attempt_root}/source",
        "raw_root": f"{attempt_root}/raw",
        "bundle_root": f"{attempt_root}/final_bundle",
        "controller_root": f"{attempt_root}/controller",
    }
    environment = {
        "TMPDIR": f"{runtime_root}/tmp",
        "XDG_CACHE_HOME": f"{runtime_root}/cache/xdg",
        "TORCH_EXTENSIONS_DIR": f"{runtime_root}/cache/torch-extensions",
        "CUDA_CACHE_PATH": f"{runtime_root}/cache/cuda",
    }
    if not all(
        _below(path, remote_root)
        for path in (*paths.values(), *environment.values())
    ):
        raise ValueError("planned path escapes approved remote root")
    forward = list(WORKLOADS)
    reverse = list(reversed(forward))
    return {
        "schema_version": PLAN_SCHEMA,
        "attempt_tag": attempt_tag,
        "source_revision": source_revision,
        "source_tree_sha256": source_tree_sha256,
        "model_repository": MODEL_REPOSITORY,
        "model_revision": MODEL_REVISION,
        "remote_root": remote_root,
        **paths,
        "environment": environment,
        "selected_gpus": [dict(row) for row in selected],
        "topology": {"rows": [dict(row) for row in topology_rows]},
        "gpu_rank_mapping": [
            {
                "rank": rank,
                "gpu_index": row["gpu_index"],
                "gpu_uuid": row["gpu_uuid"],
            }
            for rank, row in enumerate(selected)
        ],
        "pair_groups": pair_groups,
        "campaign_epochs": [
            {"epoch": 0, "arm": "baseline", "workload_order": forward},
            {"epoch": 1, "arm": "candidate", "workload_order": reverse},
            {"epoch": 2, "arm": "candidate", "workload_order": forward},
            {"epoch": 3, "arm": "baseline", "workload_order": reverse},
        ],
        "workloads": WORKLOADS,
        "thresholds": {
            "minimum_aggregate_median_tpot_improvement_percent": 5.0,
            "minimum_improving_workloads": 4,
            "maximum_tpot_p99_regression_ratio": 1.02,
            "maximum_ttft_regression_ratio": 1.02,
            "minimum_throughput_ratio": 0.98,
            "minimum_improving_pairs_per_workload": 7,
            "maximum_break_even_output_tokens": 32,
            "maximum_steady_increment_bytes_per_rank": 1920 * 1024**2,
            "maximum_peak_allocated_ratio": 0.98,
        },
    }


def _validate_plan(plan):
    if (
        not isinstance(plan, dict)
        or plan.get("schema_version") != PLAN_SCHEMA
        or plan.get("remote_root") != APPROVED_REMOTE_ROOT
        or not HEX40_PATTERN.fullmatch(
            str(plan.get("source_revision", ""))
        )
        or not HEX64_PATTERN.fullmatch(
            str(plan.get("source_tree_sha256", ""))
        )
        or plan.get("model_repository") != MODEL_REPOSITORY
        or plan.get("model_revision") != MODEL_REVISION
    ):
        raise ValueError("whole-model controller plan is invalid")
    for key in (
        "attempt_root",
        "source_root",
        "raw_root",
        "bundle_root",
        "controller_root",
    ):
        if not _below(plan.get(key, ""), APPROVED_REMOTE_ROOT):
            raise ValueError("controller path escapes approved remote root")
    selected = _strict_inventory(tuple(plan.get("selected_gpus", ())))
    expected_pairs = [
        list(pair)
        for pair in _select_best_pair_groups(plan["topology"]["rows"])
    ]
    if plan.get("pair_groups") != expected_pairs:
        raise ValueError("pair mapping is not topology-optimal")
    forward = list(WORKLOADS)
    reverse = list(reversed(forward))
    if plan.get("campaign_epochs") != [
        {"epoch": 0, "arm": "baseline", "workload_order": forward},
        {"epoch": 1, "arm": "candidate", "workload_order": reverse},
        {"epoch": 2, "arm": "candidate", "workload_order": forward},
        {"epoch": 3, "arm": "baseline", "workload_order": reverse},
    ]:
        raise ValueError("campaign epoch plan is invalid")
    return selected


def _validate_kerberos(receipt):
    return (
        isinstance(receipt, dict)
        and receipt.get("classification") == "READY"
        and receipt.get("principal") == EXPECTED_KERBEROS_PRINCIPAL
        and receipt.get("tgt_principal") == EXPECTED_KERBEROS_TGT
        and isinstance(receipt.get("remaining_lifetime_seconds"), int)
        and not isinstance(receipt.get("remaining_lifetime_seconds"), bool)
        and receipt["remaining_lifetime_seconds"]
        >= MINIMUM_KERBEROS_LIFETIME_SECONDS
    )


def _validate_launch_identity(
    plan,
    *,
    source_revision,
    model_revision,
):
    if source_revision != plan["source_revision"]:
        raise ValueError("source drift detected before launch")
    if model_revision != plan["model_revision"]:
        raise ValueError("model drift detected before launch")
    return True


def _validate_gpu_identity(plan, rows, *, require_clean):
    try:
        selected = _strict_inventory(rows) if require_clean else tuple(rows)
    except ValueError as error:
        raise RuntimeError(
            "four strict-clean GPUs are unavailable"
        ) from error
    expected = [
        (row["gpu_index"], row["gpu_uuid"])
        for row in plan["selected_gpus"]
    ]
    observed = [
        (row.get("gpu_index"), row.get("gpu_uuid"))
        for row in selected
    ]
    if observed != expected:
        raise RuntimeError("GPU rank identity drift")
    return [dict(row) for row in rows]


def select_owned_process_groups(
    process_rows,
    *,
    attempt_tag,
    registered_pgids,
):
    if not isinstance(registered_pgids, set) or any(
        not isinstance(value, int) or value <= 0
        for value in registered_pgids
    ):
        raise ValueError("registered process groups are invalid")
    owned = {
        row.get("pgid")
        for row in process_rows
        if (
            isinstance(row, dict)
            and row.get("attempt_tag") == attempt_tag
            and row.get("pgid") in registered_pgids
        )
    }
    return tuple(sorted(owned))


def _validate_launched_processes(plan, launch):
    if (
        not isinstance(launch, dict)
        or launch.get("exit_code") != 0
        or not isinstance(launch.get("registered_pgids"), list)
        or not isinstance(launch.get("process_rows"), list)
    ):
        raise RuntimeError("attempt-owned launch evidence is incomplete")
    registered = set(launch["registered_pgids"])
    for row in launch["process_rows"]:
        if (
            row.get("attempt_tag") != plan["attempt_tag"]
            or row.get("pgid") not in registered
        ):
            raise RuntimeError("foreign process appeared after launch")
    owned = select_owned_process_groups(
        launch["process_rows"],
        attempt_tag=plan["attempt_tag"],
        registered_pgids=registered,
    )
    if owned != tuple(sorted(registered)):
        raise RuntimeError("owned process registration is incomplete")
    return owned


def run_ssh_with_retry(
    argv,
    *,
    retry_count,
    runner=subprocess.run,
    timeout_s=None,
    input_text=None,
):
    if not isinstance(retry_count, int) or retry_count < 0:
        raise ValueError("retry_count is invalid")
    result = None
    for attempt in range(retry_count + 1):
        result = runner(
            argv,
            text=True,
            capture_output=True,
            check=False,
            timeout=timeout_s,
            input=input_text,
        )
        if result.returncode != 255:
            return result
        if attempt < retry_count:
            time.sleep((1.0, 2.0, 4.0)[min(attempt, 2)])
    return result


def _required_callback(name):
    def missing(*_args, **_kwargs):
        raise RuntimeError(f"{name} callback is required for launch")

    return missing


def run_attempt(
    plan,
    *,
    dry_run=False,
    check_only=False,
    kerberos_probe=query_local_kerberos,
    gpu_probe=None,
    identity_probe=None,
    remote_writer=None,
    epoch_runner=None,
    service_runner=None,
    remote_assembler=None,
    remote_verifier=None,
    downloader=None,
    local_verifier=None,
):
    _validate_plan(plan)
    gpu_probe = gpu_probe or _required_callback("gpu_probe")
    identity_probe = identity_probe or (
        lambda: {
            "source_revision": plan["source_revision"],
            "model_revision": plan["model_revision"],
        }
    )
    entry_kerberos = kerberos_probe()
    if not _validate_kerberos(entry_kerberos):
        raise RuntimeError("Kerberos launch lifetime is below 1800 seconds")
    entry_gpu = gpu_probe()
    _validate_gpu_identity(plan, entry_gpu, require_clean=True)
    if dry_run or check_only:
        return {
            "classification": (
                "DRY_RUN_READY" if dry_run else "CHECK_ONLY_READY"
            ),
            "worker_started": False,
            "resource_samples": [{
                "stage": "entry",
                "gpu_inventory": entry_gpu,
            }],
        }

    remote_writer = remote_writer or _required_callback("remote_writer")
    epoch_runner = epoch_runner or _required_callback("epoch_runner")
    service_runner = service_runner or _required_callback("service_runner")
    remote_assembler = (
        remote_assembler or _required_callback("remote_assembler")
    )
    remote_verifier = (
        remote_verifier or _required_callback("remote_verifier")
    )
    downloader = downloader or _required_callback("downloader")
    local_verifier = (
        local_verifier or _required_callback("local_verifier")
    )
    resources = [{
        "stage": "entry",
        "gpu_inventory": entry_gpu,
        "kerberos": entry_kerberos,
    }]
    remote_writer(plan)
    worker_started = False
    for epoch in plan["campaign_epochs"]:
        kerberos = kerberos_probe()
        if not _validate_kerberos(kerberos):
            raise RuntimeError(
                "Kerberos launch lifetime is below 1800 seconds"
            )
        identity = identity_probe()
        _validate_launch_identity(plan, **identity)
        inventory = gpu_probe()
        _validate_gpu_identity(plan, inventory, require_clean=True)
        resources.append({
            "stage": f"pre_epoch_{epoch['epoch']}",
            "gpu_inventory": inventory,
            "kerberos": kerberos,
        })
        launch = epoch_runner(dict(epoch))
        worker_started = True
        _validate_launched_processes(plan, launch)
        resources.append({
            "stage": f"post_launch_{epoch['epoch']}",
            "process_rows": launch["process_rows"],
        })

    kerberos = kerberos_probe()
    if not _validate_kerberos(kerberos):
        raise RuntimeError("Kerberos launch lifetime is below 1800 seconds")
    identity = identity_probe()
    _validate_launch_identity(plan, **identity)
    inventory = gpu_probe()
    _validate_gpu_identity(plan, inventory, require_clean=True)
    resources.append({
        "stage": "pre_service_control",
        "gpu_inventory": inventory,
        "kerberos": kerberos,
    })
    service = service_runner()
    _validate_launched_processes(plan, service)
    resources.append({
        "stage": "post_service_control",
        "process_rows": service["process_rows"],
    })
    producer = remote_assembler()
    remote_receipt = remote_verifier()
    downloader()
    local_receipt = local_verifier()
    if remote_receipt != local_receipt:
        raise RuntimeError(
            "remote and local verifier semantic bytes differ"
        )
    resources.append({
        "stage": "terminal",
        "gpu_inventory": gpu_probe(),
    })
    return {
        "classification": producer["classification"],
        "worker_started": worker_started,
        "resource_samples": resources,
        "remote_local_receipts_match": True,
    }


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--gpu-wait-timeout-s", type=float, default=21600)
    parser.add_argument("--gpu-poll-interval-s", type=float, default=15)
    args = parser.parse_args(argv)
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    result = run_attempt(
        plan,
        dry_run=args.dry_run,
        check_only=args.check_only,
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
