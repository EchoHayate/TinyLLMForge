#!/usr/bin/env python3
"""Orchestrate the immutable Qwen3.8 TP2 whole-model gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys
import threading
import time

if __package__:
    from tools.run_qwen38_topology_local_tp2_island import (
        DEFAULT_COMMAND_TIMEOUT_S,
        DEFAULT_PROXY_HOST,
        DEFAULT_RETRY_COUNT,
        DEFAULT_SSH_TARGET,
        _create_remote_attempt,
        _query_remote_inventory,
        _remote_json,
        _remote_run,
        _ssh_argv,
        _upload_json,
        _select_best_pair_groups,
        query_local_kerberos,
        select_strict_clean_gpus,
    )
else:
    from run_qwen38_topology_local_tp2_island import (
        DEFAULT_COMMAND_TIMEOUT_S,
        DEFAULT_PROXY_HOST,
        DEFAULT_RETRY_COUNT,
        DEFAULT_SSH_TARGET,
        _create_remote_attempt,
        _query_remote_inventory,
        _remote_json,
        _remote_run,
        _ssh_argv,
        _upload_json,
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
DEFAULT_REMOTE_PYTHON = "/data00/home/sitian/tllm/env/bin/python"
DEFAULT_MODEL_ROOT = (
    f"{APPROVED_REMOTE_ROOT}/models/Qwen3.8-27B/snapshots/"
    f"{MODEL_REVISION}"
)
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


def _merge_gpu_power(inventory, power_csv):
    power_by_identity = {}
    for line in str(power_csv).splitlines():
        if not line.strip():
            continue
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 3:
            raise ValueError("GPU power evidence is invalid")
        try:
            gpu_index = int(fields[0])
            power_watts = float(fields[2])
        except (TypeError, ValueError) as error:
            raise ValueError("GPU power evidence is invalid") from error
        identity = (gpu_index, fields[1])
        if (
            identity in power_by_identity
            or not fields[1]
            or not math.isfinite(power_watts)
            or power_watts < 0
        ):
            raise ValueError("GPU power evidence is invalid")
        power_by_identity[identity] = power_watts
    expected = {
        (row.get("gpu_index"), row.get("gpu_uuid"))
        for row in inventory
        if isinstance(row, dict)
    }
    if (
        len(expected) != len(inventory)
        or set(power_by_identity) != expected
    ):
        raise ValueError("GPU power evidence is incomplete")
    return [
        {
            **row,
            "power_watts": power_by_identity[(
                row["gpu_index"],
                row["gpu_uuid"],
            )],
        }
        for row in inventory
    ]


def _query_remote_inventory_with_power(args):
    inventory = _query_remote_inventory(args)
    result = _remote_run(
        ssh_target=args.ssh_target,
        remote_argv=[
            "nvidia-smi",
            "--query-gpu=index,uuid,power.draw",
            "--format=csv,noheader,nounits",
        ],
        proxy_host=args.proxy_host,
        retry_count=args.retry_count,
        timeout_s=args.command_timeout_s,
    )
    return _merge_gpu_power(inventory, result.stdout)


def _below(path, root):
    candidate = PurePosixPath(path)
    parent = PurePosixPath(root)
    return (
        candidate.is_absolute()
        and candidate != parent
        and candidate.is_relative_to(parent)
    )


def _strict_inventory(rows, *, preserve_order=False):
    rows = tuple(rows)
    base_fields = (
        "gpu_index",
        "gpu_uuid",
        "memory_used_mib",
        "utilization_percent",
        "compute_processes",
    )
    normalized = []
    originals = {}
    for row in rows:
        if (
            not isinstance(row, dict)
            or any(field not in row for field in base_fields)
            or set(row) - {*base_fields, "power_watts"}
        ):
            raise ValueError("exactly four strict-clean GPUs are required")
        power_watts = row.get("power_watts")
        if (
            "power_watts" in row
            and (
                isinstance(power_watts, bool)
                or not isinstance(power_watts, (int, float))
                or not math.isfinite(float(power_watts))
                or power_watts < 0
            )
        ):
            raise ValueError("exactly four strict-clean GPUs are required")
        base = {field: row[field] for field in base_fields}
        identity = (base["gpu_index"], base["gpu_uuid"])
        if identity in originals:
            raise ValueError("exactly four strict-clean GPUs are required")
        normalized.append(base)
        originals[identity] = dict(row)
    try:
        selected = select_strict_clean_gpus(normalized)
    except ValueError as error:
        raise ValueError("exactly four strict-clean GPUs are required") from error
    if len(selected) != 4:
        raise ValueError("exactly four strict-clean GPUs are required")
    selected = tuple(
        originals[(row["gpu_index"], row["gpu_uuid"])]
        for row in selected
    )
    if preserve_order:
        if (
            len(rows) != 4
            or {row["gpu_index"] for row in rows}
            != {row["gpu_index"] for row in selected}
        ):
            raise ValueError("exactly four strict-clean GPUs are required")
        return tuple(dict(row) for row in rows)
    return tuple(dict(row) for row in selected)


def _rank_topology_for_adjacent_pairs(selected, topology_rows):
    matching = _select_best_pair_groups(topology_rows)
    old_rank_order = tuple(
        rank for pair in matching for rank in pair
    )
    if set(old_rank_order) != set(range(4)):
        raise ValueError("topology matching is invalid")
    old_to_new = {
        old_rank: new_rank
        for new_rank, old_rank in enumerate(old_rank_order)
    }
    reordered_selected = tuple(
        dict(selected[old_rank]) for old_rank in old_rank_order
    )
    reordered_topology = sorted(
        (
            {
                **row,
                "left_rank": old_to_new[row["left_rank"]],
                "right_rank": old_to_new[row["right_rank"]],
            }
            for row in topology_rows
        ),
        key=lambda row: (row["left_rank"], row["right_rank"]),
    )
    if _select_best_pair_groups(reordered_topology) != (
        (0, 1),
        (2, 3),
    ):
        raise ValueError("topology matching could not be rank-normalized")
    return reordered_selected, reordered_topology


def build_plan(
    *,
    attempt_tag: str,
    source_revision: str,
    gpu_inventory: tuple[dict, ...],
    topology,
    source_tree_sha256: str | None = None,
    remote_root: str = APPROVED_REMOTE_ROOT,
    model_root: str = DEFAULT_MODEL_ROOT,
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
    if not _below(model_root, "/data00/home/sitian"):
        raise ValueError("model root is invalid")
    selected = _strict_inventory(gpu_inventory)
    topology_rows = topology.get("rows") if isinstance(topology, dict) else None
    if not isinstance(topology_rows, list):
        raise ValueError("topology inventory is invalid")
    selected, topology_rows = _rank_topology_for_adjacent_pairs(
        selected,
        topology_rows,
    )
    pair_groups = [[0, 1], [2, 3]]
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
        "model_root": model_root,
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
        or not _below(plan.get("model_root", ""), "/data00/home/sitian")
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
    selected = _strict_inventory(
        tuple(plan.get("selected_gpus", ())),
        preserve_order=True,
    )
    expected_pairs = [
        list(pair)
        for pair in _select_best_pair_groups(plan["topology"]["rows"])
    ]
    if (
        expected_pairs != [[0, 1], [2, 3]]
        or plan.get("pair_groups") != expected_pairs
    ):
        raise ValueError("pair mapping is not topology-optimal")
    expected_mapping = [
        {
            "rank": rank,
            "gpu_index": row["gpu_index"],
            "gpu_uuid": row["gpu_uuid"],
        }
        for rank, row in enumerate(selected)
    ]
    if plan.get("gpu_rank_mapping") != expected_mapping:
        raise ValueError("GPU rank mapping is invalid")
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


def build_remote_epoch_command(
    plan,
    epoch,
    *,
    python_path: str = DEFAULT_REMOTE_PYTHON,
) -> dict:
    selected = _validate_plan(plan)
    if not _below(python_path, "/data00/home/sitian"):
        raise ValueError("remote Python path is invalid")
    if epoch not in plan["campaign_epochs"]:
        raise ValueError("epoch is not part of the immutable plan")
    output_root = (
        f"{plan['controller_root']}/worker_outputs/"
        f"epoch-{epoch['epoch']}"
    )
    if not _below(output_root, plan["attempt_root"]):
        raise ValueError("worker output path escapes attempt root")
    visible = ",".join(
        str(row["gpu_index"]) for row in selected
    )
    return {
        "argv": [
            python_path,
            (
                f"{plan['source_root']}/tools/"
                "qwen38_topology_local_tp2_whole_model_worker.py"
            ),
            "performance-epoch",
            "--model-root",
            plan["model_root"],
            "--output-root",
            output_root,
            "--epoch",
            str(epoch["epoch"]),
            "--arm",
            epoch["arm"],
            "--workload-order",
            ",".join(epoch["workload_order"]),
            "--source-revision",
            plan["source_revision"],
            "--model-revision",
            plan["model_revision"],
        ],
        "environment": {
            "CUDA_VISIBLE_DEVICES": visible,
            "PYTHONPATH": plan["source_root"],
            "PYTHONNOUSERSITE": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
            **plan["environment"],
        },
        "output_root": output_root,
    }


def _base_worker_command(plan, mode, output_name, python_path):
    selected = _validate_plan(plan)
    if not _below(python_path, "/data00/home/sitian"):
        raise ValueError("remote Python path is invalid")
    output_root = (
        f"{plan['controller_root']}/worker_outputs/{output_name}"
    )
    if not _below(output_root, plan["attempt_root"]):
        raise ValueError("worker output path escapes attempt root")
    visible = ",".join(
        str(row["gpu_index"]) for row in selected
    )
    return {
        "argv": [
            python_path,
            (
                f"{plan['source_root']}/tools/"
                "qwen38_topology_local_tp2_whole_model_worker.py"
            ),
            mode,
            "--model-root",
            plan["model_root"],
            "--output-root",
            output_root,
        ],
        "environment": {
            "CUDA_VISIBLE_DEVICES": visible,
            "PYTHONPATH": plan["source_root"],
            "PYTHONNOUSERSITE": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
            **plan["environment"],
        },
        "output_root": output_root,
    }


def build_remote_correctness_command(
    plan,
    *,
    python_path: str = DEFAULT_REMOTE_PYTHON,
) -> dict:
    command = _base_worker_command(
        plan,
        "correctness",
        "correctness",
        python_path,
    )
    command["argv"].extend([
        "--source-revision",
        plan["source_revision"],
        "--model-revision",
        plan["model_revision"],
    ])
    return command


def build_remote_service_command(
    plan,
    *,
    python_path: str = DEFAULT_REMOTE_PYTHON,
) -> dict:
    command = _base_worker_command(
        plan,
        "service-control",
        "service-control",
        python_path,
    )
    physical_by_rank = {
        row["rank"]: row["gpu_index"]
        for row in plan["gpu_rank_mapping"]
    }
    command["argv"].extend([
        "--pair-devices",
        ";".join(
            ",".join(
                str(physical_by_rank[rank]) for rank in pair
            )
            for pair in plan["pair_groups"]
        ),
        "--workloads",
        "Q0,Q1,Q2",
    ])
    return command


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
    expected = [
        (row["gpu_index"], row["gpu_uuid"])
        for row in plan["selected_gpus"]
    ]
    inventory = {
        row.get("gpu_index"): row
        for row in rows
        if isinstance(row, dict)
    }
    selected_rows = tuple(
        inventory.get(gpu_index) for gpu_index, _gpu_uuid in expected
    )
    if any(row is None for row in selected_rows):
        raise RuntimeError("GPU rank identity drift")
    if require_clean:
        try:
            selected = _strict_inventory(
                selected_rows,
                preserve_order=True,
            )
        except ValueError as error:
            raise RuntimeError(
                "four strict-clean GPUs are unavailable"
            ) from error
    else:
        selected = selected_rows
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


def _run_with_runtime_resource_sampling(
    runner,
    *,
    gpu_probe,
    plan,
    run_label,
    sample_interval_s,
):
    if (
        isinstance(sample_interval_s, bool)
        or not isinstance(sample_interval_s, (int, float))
        or not math.isfinite(float(sample_interval_s))
        or sample_interval_s <= 0
    ):
        raise ValueError("runtime resource sample interval is invalid")
    started = threading.Event()
    done = threading.Event()
    outcome = {}

    def invoke():
        try:
            started.set()
            outcome["result"] = runner()
        except BaseException as error:
            outcome["error"] = error
        finally:
            done.set()

    thread = threading.Thread(
        target=invoke,
        name=f"qwen38-resource-sampler-{run_label}",
    )
    thread.start()
    started.wait()
    samples = []
    sample_index = 0
    sample_error = None
    while not done.wait(float(sample_interval_s)):
        try:
            inventory = gpu_probe()
            _validate_gpu_identity(plan, inventory, require_clean=False)
            if done.is_set():
                break
            samples.append({
                "stage": f"runtime_{run_label}_{sample_index:04d}",
                "measurement_scope": "runtime",
                "run_label": run_label,
                "sample_index": sample_index,
                "gpu_inventory": inventory,
            })
            sample_index += 1
        except BaseException as error:
            sample_error = error
            break
    thread.join()
    if "error" in outcome:
        raise outcome["error"]
    if sample_error is not None:
        raise RuntimeError(
            f"runtime GPU telemetry failed for {run_label}"
        ) from sample_error
    if not samples:
        raise RuntimeError(
            f"runtime GPU telemetry unavailable for {run_label}"
        )
    return outcome["result"], samples


def run_attempt(
    plan,
    *,
    dry_run=False,
    check_only=False,
    kerberos_probe=None,
    gpu_probe=None,
    identity_probe=None,
    remote_writer=None,
    correctness_runner=None,
    epoch_runner=None,
    service_runner=None,
    remote_assembler=None,
    remote_verifier=None,
    downloader=None,
    local_verifier=None,
    runtime_sample_interval_s=5.0,
):
    _validate_plan(plan)
    kerberos_probe = kerberos_probe or (
        lambda: query_local_kerberos(
            minimum_lifetime_seconds=(
                MINIMUM_KERBEROS_LIFETIME_SECONDS
            )
        )
    )
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
    correctness_runner = (
        correctness_runner or _required_callback("correctness_runner")
    )
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
        "measurement_scope": "boundary",
        "gpu_inventory": entry_gpu,
        "kerberos": entry_kerberos,
    }]
    remote_writer(plan)
    worker_started = False
    kerberos = kerberos_probe()
    if not _validate_kerberos(kerberos):
        raise RuntimeError("Kerberos launch lifetime is below 1800 seconds")
    identity = identity_probe()
    _validate_launch_identity(plan, **identity)
    inventory = gpu_probe()
    _validate_gpu_identity(plan, inventory, require_clean=True)
    resources.append({
        "stage": "pre_correctness",
        "measurement_scope": "boundary",
        "gpu_inventory": inventory,
        "kerberos": kerberos,
    })
    correctness, runtime_samples = _run_with_runtime_resource_sampling(
        correctness_runner,
        gpu_probe=gpu_probe,
        plan=plan,
        run_label="correctness",
        sample_interval_s=runtime_sample_interval_s,
    )
    resources.extend(runtime_samples)
    worker_started = True
    _validate_launched_processes(plan, correctness)
    post_correctness_gpu = gpu_probe()
    _validate_gpu_identity(
        plan,
        post_correctness_gpu,
        require_clean=True,
    )
    resources.append({
        "stage": "post_correctness",
        "measurement_scope": "boundary",
        "process_rows": correctness["process_rows"],
        "gpu_inventory": post_correctness_gpu,
    })
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
            "measurement_scope": "boundary",
            "gpu_inventory": inventory,
            "kerberos": kerberos,
        })
        launch, runtime_samples = _run_with_runtime_resource_sampling(
            lambda epoch=epoch: epoch_runner(dict(epoch)),
            gpu_probe=gpu_probe,
            plan=plan,
            run_label=f"epoch_{epoch['epoch']}",
            sample_interval_s=runtime_sample_interval_s,
        )
        resources.extend(runtime_samples)
        worker_started = True
        _validate_launched_processes(plan, launch)
        post_launch_gpu = gpu_probe()
        _validate_gpu_identity(
            plan,
            post_launch_gpu,
            require_clean=True,
        )
        resources.append({
            "stage": f"post_launch_{epoch['epoch']}",
            "measurement_scope": "boundary",
            "process_rows": launch["process_rows"],
            "gpu_inventory": post_launch_gpu,
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
        "measurement_scope": "boundary",
        "gpu_inventory": inventory,
        "kerberos": kerberos,
    })
    service, runtime_samples = _run_with_runtime_resource_sampling(
        service_runner,
        gpu_probe=gpu_probe,
        plan=plan,
        run_label="service_control",
        sample_interval_s=runtime_sample_interval_s,
    )
    resources.extend(runtime_samples)
    _validate_launched_processes(plan, service)
    post_service_gpu = gpu_probe()
    _validate_gpu_identity(
        plan,
        post_service_gpu,
        require_clean=True,
    )
    resources.append({
        "stage": "post_service_control",
        "measurement_scope": "boundary",
        "process_rows": service["process_rows"],
        "gpu_inventory": post_service_gpu,
    })
    terminal_gpu = gpu_probe()
    _validate_gpu_identity(plan, terminal_gpu, require_clean=True)
    resources.append({
        "stage": "terminal",
        "measurement_scope": "boundary",
        "gpu_inventory": terminal_gpu,
    })
    producer = remote_assembler(tuple(resources))
    remote_receipt = remote_verifier()
    downloader()
    local_receipt = local_verifier()
    if remote_receipt != local_receipt:
        raise RuntimeError(
            "remote and local verifier semantic bytes differ"
        )
    return {
        "classification": producer["classification"],
        "worker_started": worker_started,
        "resource_samples": resources,
        "remote_local_receipts_match": True,
    }


def _run_remote_worker_command(args, plan, command, label):
    log_root = f"{plan['controller_root']}/logs"
    registration = (
        f"{plan['controller_root']}/owned-processes/{label}.json"
    )
    script = "\n".join([
        "import json,os,subprocess,sys",
        "argv=json.loads(sys.argv[1]); env=json.loads(sys.argv[2])",
        "attempt,label,log_root,registration=sys.argv[3:]",
        "os.makedirs(log_root,exist_ok=True)",
        "os.makedirs(os.path.dirname(registration),exist_ok=True)",
        "stdout_path=os.path.join(log_root,label+'.stdout.log')",
        "stderr_path=os.path.join(log_root,label+'.stderr.log')",
        "with open(stdout_path,'w') as out, open(stderr_path,'w') as err:",
        " p=subprocess.Popen(argv,env={**os.environ,**env},stdout=out,"
        "stderr=err,start_new_session=True)",
        " pgid=os.getpgid(p.pid)",
        " stat=open('/proc/'+str(p.pid)+'/stat',encoding='utf-8').read()",
        " start_time_ticks=int(stat.rsplit(')',1)[1].split()[19])",
        " payload={'attempt_tag':attempt,'label':label,'pid':p.pid,"
        "'pgid':pgid,'start_time_ticks':start_time_ticks,'argv':argv}",
        " temporary=registration+'.tmp-'+str(os.getpid())",
        " with open(temporary,'w',encoding='utf-8') as stream:",
        "  json.dump(payload,stream,sort_keys=True); stream.flush();"
        " os.fsync(stream.fileno())",
        " os.replace(temporary,registration)",
        " code=p.wait()",
        "print(json.dumps({'exit_code':code,'registered_pgids':[pgid],"
        "'process_rows':[{'pid':p.pid,'pgid':pgid,"
        "'attempt_tag':attempt}]} ,sort_keys=True))",
    ])
    launch_args = argparse.Namespace(**vars(args))
    launch_args.retry_count = 0
    try:
        return _remote_json(
            launch_args,
            [
                "python3",
                "-c",
                script,
                json.dumps(command["argv"]),
                json.dumps(command["environment"]),
                plan["attempt_tag"],
                label,
                log_root,
                registration,
            ],
            max(args.command_timeout_s, 7200),
        )
    except BaseException as launch_error:
        cleanup_script = "\n".join([
            "import json,os,signal,sys,time",
            "path,attempt,label=sys.argv[1:]",
            "registration_deadline=time.monotonic()+10.0",
            "while not os.path.isfile(path) and "
            "time.monotonic()<registration_deadline:",
            " time.sleep(0.1)",
            "if not os.path.isfile(path):",
            " print(json.dumps({'classification':'NOT_REGISTERED',"
            "'attempt_tag':attempt,'label':label},sort_keys=True));"
            " raise SystemExit(0)",
            "payload=json.load(open(path,encoding='utf-8'))",
            "if payload.get('attempt_tag')!=attempt or "
            "payload.get('label')!=label:",
            " raise RuntimeError('worker registration identity mismatch')",
            "pid=payload.get('pid')",
            "pgid=payload.get('pgid')",
            "start_time_ticks=payload.get('start_time_ticks')",
            "registered_argv=payload.get('argv')",
            "if type(pid) is not int or pid<=0 or type(pgid) is not int "
            "or pgid<=0 or type(start_time_ticks) is not int or "
            "not isinstance(registered_argv,list) or not registered_argv:",
            " raise RuntimeError('worker registration identity is invalid')",
            "classification='ALREADY_EXITED'",
            "proc_root='/proc/'+str(pid)",
            "if os.path.isdir(proc_root):",
            " current_stat=open(proc_root+'/stat',encoding='utf-8').read()",
            " current_start_time_ticks=int("
            "current_stat.rsplit(')',1)[1].split()[19])",
            " current_argv=[part.decode('utf-8') for part in "
            "open(proc_root+'/cmdline','rb').read().split(b'\\0') if part]",
            " if current_start_time_ticks!=start_time_ticks or "
            "current_argv!=registered_argv or os.getpgid(pid)!=pgid:",
            "  classification='OWNERSHIP_MISMATCH'",
            " else:",
            "  try:",
            "   os.killpg(pgid,signal.SIGTERM); classification='CLEANED'",
            "  except ProcessLookupError:",
            "   classification='ALREADY_EXITED'",
            "deadline=time.monotonic()+10.0",
            "while classification=='CLEANED' and time.monotonic()<deadline:",
            " try: os.killpg(pgid,0)",
            " except ProcessLookupError: break",
            " time.sleep(0.1)",
            "else:",
            " if classification=='CLEANED':",
            "  try: os.killpg(pgid,signal.SIGKILL)",
            "  except ProcessLookupError: pass",
            "print(json.dumps({'classification':classification,"
            "'attempt_tag':attempt,'label':label,'pgid':pgid},"
            "sort_keys=True))",
        ])
        try:
            cleanup = _remote_json(
                args,
                [
                    "python3",
                    "-c",
                    cleanup_script,
                    registration,
                    plan["attempt_tag"],
                    label,
                ],
                max(args.command_timeout_s, 30),
            )
            if cleanup.get("classification") not in {
                "CLEANED",
                "ALREADY_EXITED",
                "NOT_REGISTERED",
            }:
                raise RuntimeError(
                    "remote worker cleanup receipt is invalid"
                )
        except Exception as cleanup_error:
            raise RuntimeError(
                "remote worker launch failed and cleanup could not "
                f"be verified: {cleanup_error}"
            ) from launch_error
        raise


def _reap_local_process(process, *, timeout_s=5.0):
    try:
        process.terminate()
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        try:
            process.kill()
        except ProcessLookupError:
            return
        process.wait(timeout=timeout_s)


def _download_final_bundle(args, plan, local_attempt_root):
    local_root = Path(local_attempt_root).resolve()
    bundle = local_root / "final_bundle"
    if bundle.exists():
        raise ValueError("local final bundle must be fresh")
    local_root.mkdir(parents=True, exist_ok=True)
    sender = subprocess.Popen(
        _ssh_argv(
            args.ssh_target,
            ["tar", "-cf", "-", "-C", plan["attempt_root"], "final_bundle"],
            args.proxy_host,
        ),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    timeout_s = max(args.command_timeout_s, 600)
    sender_stdout = sender.stdout
    sender_stdout_closed = False
    try:
        receiver = subprocess.run(
            ["tar", "-xf", "-", "-C", str(local_root)],
            stdin=sender_stdout,
            capture_output=True,
            check=False,
            timeout=timeout_s,
        )
        if sender_stdout is not None:
            sender_stdout.close()
            sender_stdout_closed = True
        sender.stdout = None
        _sender_output, sender_error = sender.communicate(
            timeout=timeout_s
        )
        sender_code = sender.returncode
    except BaseException:
        if sender_stdout is not None and not sender_stdout_closed:
            sender_stdout.close()
        sender.stdout = None
        _reap_local_process(sender)
        raise
    if sender_code != 0 or receiver.returncode != 0:
        raise RuntimeError(
            sender_error.decode(errors="replace")
            or receiver.stderr.decode(errors="replace")
            or "compact final-bundle download failed"
        )


def build_default_adapters(args, plan):
    repo_root = Path(__file__).resolve().parents[1]
    local_attempt = (
        args.local_attempt_root.resolve()
        if args.local_attempt_root is not None
        else repo_root / "artifacts" / (
            "qwen38_topology_local_tp2_whole_model"
        ) / plan["attempt_tag"]
    )

    def inventory():
        return _query_remote_inventory_with_power(args)

    def identity():
        script = "\n".join([
            "import json,os,sys",
            "source,model=sys.argv[1:]",
            "payload=json.load(open(source,encoding='utf-8'))",
            "print(json.dumps({'source_revision':payload['source_revision'],"
            "'model_revision':os.path.basename(os.path.realpath(model))},"
            "sort_keys=True))",
        ])
        return _remote_json(
            args,
            [
                "python3", "-c", script,
                f"{plan['controller_root']}/source_identity.json",
                plan["model_root"],
            ],
            args.command_timeout_s,
        )

    def write_remote(current):
        return _create_remote_attempt(
            args,
            current,
            {
                "source_revision": current["source_revision"],
                "source_tree_sha256": current["source_tree_sha256"],
                "model_repository": current["model_repository"],
                "model_revision": current["model_revision"],
            },
        )

    def run_command(command, label):
        return _run_remote_worker_command(args, plan, command, label)

    def assemble(resource_samples):
        resource_path = (
            f"{plan['controller_root']}/resource-samples.json"
        )
        _upload_json(args, list(resource_samples), resource_path)
        _remote_json(
            args,
            [
                args.remote_python,
                f"{plan['source_root']}/tools/"
                "qwen38_topology_local_tp2_whole_model_worker.py",
                "finalize-artifacts",
                "--plan",
                f"{plan['controller_root']}/plan.json",
                "--worker-output-root",
                f"{plan['controller_root']}/worker_outputs",
                "--resource-samples",
                resource_path,
                "--raw-root",
                plan["raw_root"],
            ],
            max(args.command_timeout_s, 600),
        )
        return _remote_json(
            args,
            [
                args.remote_python,
                f"{plan['source_root']}/tools/"
                "assemble_qwen38_topology_local_tp2_whole_model.py",
                "--attempt-root",
                plan["raw_root"],
                "--output-root",
                plan["bundle_root"],
            ],
            max(args.command_timeout_s, 600),
        )

    remote_receipt = (
        f"{plan['controller_root']}/remote-verification.json"
    )

    def verify_remote():
        _remote_run(
            ssh_target=args.ssh_target,
            remote_argv=[
                args.remote_python,
                f"{plan['source_root']}/tools/"
                "verify_qwen38_topology_local_tp2_whole_model.py",
                "--bundle",
                plan["bundle_root"],
                "--output",
                remote_receipt,
            ],
            proxy_host=args.proxy_host,
            retry_count=args.retry_count,
            timeout_s=max(args.command_timeout_s, 600),
        )
        return _remote_run(
            ssh_target=args.ssh_target,
            remote_argv=["cat", remote_receipt],
            proxy_host=args.proxy_host,
            retry_count=args.retry_count,
            timeout_s=args.command_timeout_s,
        ).stdout.encode()

    def download():
        _download_final_bundle(args, plan, local_attempt)

    def verify_local():
        output = local_attempt / "local-verification.json"
        result = subprocess.run(
            [
                sys.executable,
                str(repo_root / "tools" /
                    "verify_qwen38_topology_local_tp2_whole_model.py"),
                "--bundle",
                str(local_attempt / "final_bundle"),
                "--output",
                str(output),
            ],
            text=True,
            capture_output=True,
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError(result.stderr or "local verifier failed")
        return output.read_bytes()

    return {
        "gpu_probe": inventory,
        "identity_probe": identity,
        "remote_writer": write_remote,
        "correctness_runner": lambda: run_command(
            build_remote_correctness_command(
                plan, python_path=args.remote_python
            ),
            "correctness",
        ),
        "epoch_runner": lambda epoch: run_command(
            build_remote_epoch_command(
                plan, epoch, python_path=args.remote_python
            ),
            f"epoch-{epoch['epoch']}",
        ),
        "service_runner": lambda: run_command(
            build_remote_service_command(
                plan, python_path=args.remote_python
            ),
            "service-control",
        ),
        "remote_assembler": assemble,
        "remote_verifier": verify_remote,
        "downloader": download,
        "local_verifier": verify_local,
    }


def main(
    argv=None,
    *,
    adapter_factory=build_default_adapters,
    printer=print,
):
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--gpu-wait-timeout-s", type=float, default=21600)
    parser.add_argument("--gpu-poll-interval-s", type=float, default=15)
    parser.add_argument("--remote-python", default=DEFAULT_REMOTE_PYTHON)
    parser.add_argument("--ssh-target", default=DEFAULT_SSH_TARGET)
    parser.add_argument("--proxy-host", default=DEFAULT_PROXY_HOST)
    parser.add_argument("--retry-count", type=int, default=DEFAULT_RETRY_COUNT)
    parser.add_argument(
        "--command-timeout-s",
        type=int,
        default=DEFAULT_COMMAND_TIMEOUT_S,
    )
    parser.add_argument("--local-attempt-root", type=Path)
    args = parser.parse_args(argv)
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    adapters = (
        {}
        if args.dry_run or args.check_only
        else adapter_factory(args, plan)
    )
    result = run_attempt(
        plan,
        dry_run=args.dry_run,
        check_only=args.check_only,
        **adapters,
    )
    printer(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
