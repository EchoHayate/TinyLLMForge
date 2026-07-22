#!/usr/bin/env python3
"""Run source-bound exact CUDA Graph production gates on the remote A100."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import random
import shlex
import shutil
import socket
import subprocess
import sys
import tarfile
import tempfile
import time
from dataclasses import asdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
SSH_TARGET = "sitian@10.232.195.203"
SSH_CONTROL_PATH = "/tmp/ssh-sitian-10.232.195.203"
REMOTE_PYTHON = (
    "/data00/home/sitian/sitian-workspace01/tllm/env/bin/python"
)
REMOTE_MODEL = (
    "/data00/home/sitian/sitian-workspace01/.ms_cache/"
    "Qwen/Qwen3-0___6B"
)
DIAGNOSTIC_VERIFIER_PYTHON = Path(
    "/Users/bytedance/Desktop/RL_local_mirror/.venv/bin/python"
)
REMOTE_RUN_ROOT = (
    "/data00/home/sitian/sitian-workspace01/tllm/"
    "exact-cuda-production-runs"
)
CUDA_VISIBLE_DEVICES = "0"
DIST_PORT_ENV = "TINYVLLM_DIST_PORT"
MASTER_PORT_ENV = "MASTER_PORT"
OUTPUT_ROOT = ROOT / "experiments" / "cuda_graph"
MODES = (
    "preflight",
    "local-contracts",
    "correctness-smoke",
    "correctness-canonical",
    "arrival-smoke",
    "arrival-canonical",
    "download-only",
    "verify-only",
)
OWNED_SOURCE_ROOTS = (
    "tinyvllm",
    "tools/source_audit.py",
    "tools/arrival_load_gate.py",
    "tools/multi_sequence_cuda_graph_contract.py",
    "tools/verify_multi_sequence_cuda_graph_production.py",
    "tools/run_multi_sequence_cuda_graph_production_gate_remote.py",
    "tools/run_multi_sequence_cuda_graph_diagnostic_remote.py",
    "tools/diagnose_multi_sequence_cuda_graph.py",
    "tools/verify_multi_sequence_cuda_graph_diagnostic.py",
    "tools/test_multi_sequence_cuda_graph_gate.py",
    "tools/test_model_runner_spec_verify.py",
)
IGNORED_UNTRACKED_PREFIXES = ("experiments",)
SSH_OPTIONS = (
    "-o",
    "ControlMaster=auto",
    "-o",
    f"ControlPath={SSH_CONTROL_PATH}",
    "-o",
    "ControlPersist=600",
)


def _load_tool(name: str, filename: str):
    path = TOOLS / filename
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load tool: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


contract = _load_tool(
    "exact_cuda_production_contract",
    "multi_sequence_cuda_graph_contract.py",
)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    temporary.write_bytes(contract.canonical_json_bytes(value) + b"\n")
    temporary.replace(path)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    temporary.write_bytes(
        b"".join(
            contract.canonical_json_bytes(row) + b"\n"
            for row in rows
        )
    )
    temporary.replace(path)


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict]:
    payload = path.read_bytes()
    if payload and not payload.endswith(b"\n"):
        raise ValueError(f"{path.name} lacks final newline")
    return [json.loads(line) for line in payload.splitlines()]


def _safe_run_tag(run_tag: str) -> str:
    if (
        not run_tag
        or any(
            character
            not in (
                "abcdefghijklmnopqrstuvwxyz"
                "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
                "0123456789-_"
            )
            for character in run_tag
        )
    ):
        raise ValueError("run tag contains unsupported characters")
    return run_tag


def _default_run_tag(mode: str) -> str:
    stamp = time.strftime("%Y%m%d-%H%M%S", time.localtime())
    entropy = hashlib.sha256(os.urandom(16)).hexdigest()[:8]
    return f"qwen3-06b-exact-cuda-{mode}-{stamp}-{entropy}"


def _remote_run_dir(run_tag: str) -> str:
    return f"{REMOTE_RUN_ROOT}/{_safe_run_tag(run_tag)}"


def _ssh_argv(remote_argv: list[str]) -> list[str]:
    remote_command = " ".join(shlex.quote(value) for value in remote_argv)
    return [
        "ssh",
        *SSH_OPTIONS,
        SSH_TARGET,
        "bash",
        "-lc",
        shlex.quote(remote_command),
    ]


def _run_remote(
    remote_argv: list[str],
    *,
    input_bytes: bytes | None = None,
    check: bool = True,
) -> subprocess.CompletedProcess:
    result = subprocess.run(
        _ssh_argv(remote_argv),
        input=input_bytes,
        capture_output=True,
        check=False,
    )
    if check and result.returncode != 0:
        raise RuntimeError(
            result.stderr.decode("utf-8", errors="replace").strip()
            or f"remote command exited {result.returncode}"
        )
    return result


def allocate_port_pair() -> tuple[int, int]:
    handles = []
    ports = []
    try:
        while len(ports) < 2:
            handle = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            socket.socket.bind(handle, ("127.0.0.1", 0))
            handles.append(handle)
            port = int(handle.getsockname()[1])
            if port not in ports:
                ports.append(port)
    finally:
        for handle in handles:
            handle.close()
    return ports[0], ports[1]


def build_worker_command(
    *,
    remote_source: str,
    output_dir: str,
    worker_kind: str,
    source_sha256: str,
    dist_port: int,
    master_port: int,
    case_id: str,
    policy: str,
    workload: str,
    repetition: int,
    warmup: bool,
    visible_blocks: int,
) -> dict:
    if dist_port == master_port:
        raise ValueError("worker ports must be distinct")
    argv = [
        REMOTE_PYTHON,
        "tools/run_multi_sequence_cuda_graph_production_gate_remote.py",
        "local-contracts",
        "--worker-kind",
        worker_kind,
        "--output-dir",
        output_dir,
        "--source-sha256",
        source_sha256,
        "--case-id",
        case_id,
        "--policy",
        policy,
        "--workload",
        workload,
        "--repetition",
        str(repetition),
        "--num-kvcache-blocks",
        str(visible_blocks),
    ]
    if warmup:
        argv.append("--warmup")
    return {
        "cwd": remote_source,
        "env": {
            "CUDA_VISIBLE_DEVICES": CUDA_VISIBLE_DEVICES,
            DIST_PORT_ENV: str(dist_port),
            MASTER_PORT_ENV: str(master_port),
            "PYTHONPATH": remote_source,
            "PYTHONDONTWRITEBYTECODE": "1",
            "TINYVLLM_SOURCE_SHA256": source_sha256,
        },
        "argv": argv,
    }


def build_budget_fallback_worker_command(
    *,
    remote_source: str,
    output_dir: str,
    source_sha256: str,
    dist_port: int,
    master_port: int,
    reason: str,
    visible_blocks: int,
) -> dict:
    if reason not in contract.BUDGET_FALLBACK_REASONS:
        raise ValueError(f"unknown budget fallback reason: {reason}")
    if dist_port == master_port:
        raise ValueError("worker ports must be distinct")
    argv = [
        REMOTE_PYTHON,
        "tools/run_multi_sequence_cuda_graph_production_gate_remote.py",
        "local-contracts",
        "--worker-kind",
        "budget-fallback",
        "--budget-fallback-reason",
        reason,
        "--output-dir",
        output_dir,
        "--source-sha256",
        source_sha256,
        "--case-id",
        f"budget-fallback:{reason}",
        "--num-kvcache-blocks",
        str(visible_blocks),
    ]
    return {
        "cwd": remote_source,
        "env": {
            "CUDA_VISIBLE_DEVICES": CUDA_VISIBLE_DEVICES,
            DIST_PORT_ENV: str(dist_port),
            MASTER_PORT_ENV: str(master_port),
            "PYTHONPATH": remote_source,
            "PYTHONDONTWRITEBYTECODE": "1",
            "TINYVLLM_SOURCE_SHA256": source_sha256,
        },
        "argv": argv,
    }


def build_paired_capacity_contract(candidate_capacity: dict) -> dict:
    required = (
        "physical_blocks",
        "scratch_blocks",
        "scheduler_visible_blocks",
        "block_size",
    )
    if any(
        isinstance(candidate_capacity.get(field), bool)
        or not isinstance(candidate_capacity.get(field), int)
        or candidate_capacity[field] <= 0
        for field in required
    ):
        raise ValueError("candidate capacity evidence is incomplete")
    visible = candidate_capacity["scheduler_visible_blocks"]
    physical = candidate_capacity["physical_blocks"]
    scratch = candidate_capacity["scratch_blocks"]
    if physical != visible + scratch:
        raise ValueError("candidate capacity is not visible plus scratch")
    baseline = {
        "physical_blocks": visible,
        "scratch_blocks": 0,
        "scheduler_visible_blocks": visible,
        "block_size": candidate_capacity["block_size"],
    }
    candidate = dict(candidate_capacity)
    return {
        "baseline_config": {
            "multi_sequence_cuda_graphs": False,
            "num_kvcache_blocks": visible,
        },
        "candidate_config": {
            "multi_sequence_cuda_graphs": True,
            "num_kvcache_blocks": visible,
        },
        "capacity": {
            "baseline": baseline,
            "candidate": candidate,
        },
    }


def build_correctness_plan(*, canonical: bool) -> dict:
    return {
        "canonical": bool(canonical),
        "allowlisted_batches": [2, 4, 8],
        "fallback_batches": [3, 5, 7, 9],
        "stable_page_widths": [1, 2, 3],
        "page_transition_tokens": [255, 256, 257],
        "minimum_observations": 3,
        "requires_post_capture_replay": True,
        "required_budget_fallbacks": [
            "entry_limit",
            "static_byte_budget",
            "reserved_byte_budget",
            "single_capture_budget",
            "total_capture_budget",
            "scratch_unavailable",
            "capture_failed",
            "identity_drift",
        ],
    }


def build_arrival_workload(
    *,
    workload: str,
    repetition: int,
    warmup: bool,
) -> dict:
    if workload not in contract.PRODUCTION_WORKLOADS:
        raise ValueError(f"unknown production workload: {workload}")
    if repetition < 0:
        raise ValueError("repetition must be non-negative")
    seed = int(
        hashlib.sha256(
            f"{workload}:{repetition}:{int(warmup)}".encode()
        ).hexdigest()[:16],
        16,
    )
    randomizer = random.Random(seed)
    request_count = 4 if warmup else 12
    if workload == "long_prompt_pressure":
        prompt_tokens = 1536
    elif workload == "page_width_transition":
        prompt_tokens = 255
    elif workload == "long_decode":
        prompt_tokens = 64
    else:
        prompt_tokens = 32
    output_tokens = (
        64
        if workload == "long_decode"
        else 16
        if workload in {
            "stable_exact_reuse",
            "mixed_allowlist_and_fallback",
            "page_width_transition",
            "burst_arrivals",
            "long_prompt_pressure",
        }
        else 8
    )
    offsets = []
    current = 0
    for index in range(request_count):
        if workload == "burst_arrivals":
            current = (index // 4) * 2_000_000
        elif workload == "near_stable_service_rate":
            current += randomizer.randint(4_000_000, 8_000_000)
        else:
            current += randomizer.randint(500_000, 2_000_000)
        offsets.append(current)
    requests = [
        {
            "request_id": f"{workload}-r{repetition}-q{index}",
            "arrival_offset_ns": offset,
            "prompt_token_ids": [
                100 + ((index + token_index) % 1000)
                for token_index in range(prompt_tokens)
            ],
            "prompt_token_count": prompt_tokens,
            "requested_output_tokens": output_tokens,
            "sampling": {
                "temperature": 0.0,
                "max_tokens": output_tokens,
                "ignore_eos": True,
            },
        }
        for index, offset in enumerate(offsets)
    ]
    return {
        "schema_version": 1,
        "workload": workload,
        "repetition": repetition,
        "warmup": bool(warmup),
        "seed": seed,
        "requests": requests,
    }


def build_correctness_workload(
    *,
    workload: str,
    repetition: int,
    warmup: bool,
) -> dict:
    batch_by_workload = {
        "stable_exact_reuse": 2,
        "mixed_allowlist_and_fallback": 3,
        "page_width_transition": 4,
        "short_capture_cold_cost": 5,
        "long_decode": 7,
        "burst_arrivals": 8,
        "near_stable_service_rate": 9,
        "long_prompt_pressure": 4,
    }
    batch_size = batch_by_workload[workload]
    prompt_tokens = (
        255
        if workload == "page_width_transition"
        else 513
        if workload == "long_prompt_pressure"
        else 32
    )
    output_tokens = 16 if batch_size in (2, 4, 8) else 4
    return {
        "schema_version": 1,
        "workload": workload,
        "repetition": repetition,
        "warmup": bool(warmup),
        "seed": int(
            hashlib.sha256(
                f"correctness:{workload}:{repetition}:{int(warmup)}".encode()
            ).hexdigest()[:16],
            16,
        ),
        "requests": [
            {
                "request_id": (
                    f"correctness-{workload}-r{repetition}-q{index}"
                ),
                "arrival_offset_ns": 0,
                "prompt_token_ids": [
                    100 + ((index + token_index) % 1000)
                    for token_index in range(prompt_tokens)
                ],
                "prompt_token_count": prompt_tokens,
                "requested_output_tokens": output_tokens,
                "sampling": {
                    "temperature": 0.0,
                    "max_tokens": output_tokens,
                    "ignore_eos": True,
                },
            }
            for index in range(batch_size)
        ],
    }


def production_matrix_for_mode(
    mode: str,
) -> tuple[contract.ProductionCase, ...]:
    if mode not in {
        "correctness-smoke",
        "correctness-canonical",
        "arrival-smoke",
        "arrival-canonical",
    }:
        raise ValueError(f"mode does not execute cases: {mode}")
    if mode.endswith("-smoke"):
        return contract.build_production_smoke_matrix()
    return contract.build_production_matrix()


def build_case_plan(mode: str) -> list[dict]:
    worker_kind = (
        "correctness" if mode.startswith("correctness-") else "arrival"
    )
    matrix = production_matrix_for_mode(mode)
    return [
        {
            "case_id": case.case_id,
            "workload": case.workload,
            "policy": case.policy,
            "repetition": case.repetition,
            "warmup": case.warmup,
            "policy_order": case.policy_order,
            "paired_order": list(case.paired_order),
            "worker_kind": worker_kind,
        }
        for case in matrix
    ]


def build_budget_fallback_plan(mode: str) -> list[dict]:
    if mode in {"arrival-smoke", "arrival-canonical"}:
        return []
    if mode not in {"correctness-smoke", "correctness-canonical"}:
        raise ValueError(f"mode does not execute budget fallbacks: {mode}")
    return [
        {
            "case_id": f"budget-fallback:{reason}",
            "reason": reason,
            "worker_kind": "budget-fallback",
        }
        for reason in contract.BUDGET_FALLBACK_REASONS
    ]


def build_canonical_diagnostic_command(*, run_tag: str) -> list[str]:
    return [
        sys.executable,
        "tools/run_multi_sequence_cuda_graph_diagnostic_remote.py",
        "heuristic-exact-width-canonical",
        "--run-tag",
        f"{run_tag}-diagnostic",
        "--output-root",
        str(OUTPUT_ROOT),
        "--verifier-python",
        str(DIAGNOSTIC_VERIFIER_PYTHON),
    ]


def load_canonical_diagnostic_binding(
    diagnostic_dir: Path,
    *,
    expected_source_sha256: str,
) -> dict:
    diagnostic_dir = Path(diagnostic_dir)
    source = _read_json(diagnostic_dir / "source_evidence.json")
    manifest = _read_json(diagnostic_dir / "manifest.json")
    verification = _read_json(
        diagnostic_dir / "independent-verification" / "summary.json"
    )
    source_sha256 = source.get("tree_sha256")
    if (
        source_sha256 != expected_source_sha256
        or manifest.get("source_tree_sha256") != expected_source_sha256
    ):
        raise ValueError("canonical diagnostic source mismatch")
    case_ids = manifest.get("case_ids")
    if manifest.get("canonical") is not True or not isinstance(
        case_ids,
        list,
    ) or len(case_ids) != 315:
        raise ValueError("canonical diagnostic is not 315-case complete")
    expected = {
        "classification": "EXACT_REPLAY_CORRECT",
        "rounded_classification": "ROUNDED_REPLAY_CORRUPT",
        "legacy_compatibility": "LEGACY_COMPATIBLE",
        "policy_integrity": "POLICY_EXACT",
    }
    for field, value in expected.items():
        if verification.get(field) != value:
            raise ValueError(
                f"canonical diagnostic {field} mismatch"
            )
    if verification.get("case_count") != 315:
        raise ValueError("canonical diagnostic case_count mismatch")
    if verification.get("failures") != []:
        raise ValueError("canonical diagnostic contains failures")
    return {
        "required": True,
        "run_tag": diagnostic_dir.name,
        "source_tree_sha256": source_sha256,
        "case_count": 315,
        "classifications": expected,
    }


def _nearest_rank(values: list[float], percentile: float) -> float:
    if not values:
        raise ValueError("percentile requires samples")
    ordered = sorted(float(value) for value in values)
    rank = max(1, math.ceil(percentile * len(ordered)))
    return ordered[rank - 1]


def _single_or_combined_sha256(values: list[str]) -> str:
    if not values:
        raise ValueError("hash evidence is empty")
    if len(values) == 1:
        return values[0]
    return contract.canonical_json_sha256(values)


def _case_summary(worker_result: dict, production_case) -> dict:
    request_rows = worker_result["request_rows"]
    model_step = worker_result["model_step_rows"][0]
    memory_rows = worker_result["memory_rows"]
    duration_ns = float(model_step["measurement_duration_ns"])
    decode_duration_ns = float(model_step["decode_duration_ns"])
    decoded_tokens = float(model_step["decoded_tokens"])
    itl_values = [
        float(value)
        for row in request_rows
        for value in row["itl_ns"]
    ]
    dispatch_contract_rows = [
        {
            "dispatch": row.get("dispatch"),
            "graph_identity_sha256": row.get(
                "graph_identity_sha256"
            ),
            "rebuilt_identity_sha256": row.get(
                "graph_identity_sha256"
            ),
            "page_table_width": row.get("page_table_width"),
            "active_batch_size": row.get("active_batch_size"),
            "fallback_reason": row.get("fallback_reason"),
            "cache_state": row.get("cache_state"),
        }
        for row in worker_result["dispatch_rows"]
        if (
            row.get("dispatch") == "graph"
            or (
                row.get("dispatch") == "eager"
                and row.get("fallback_reason") is not None
            )
        )
    ]
    graph_hits = sum(
        row.get("dispatch") == "graph"
        for row in worker_result["dispatch_rows"]
    )
    return {
        "case_id": production_case.case_id,
        "workload": production_case.workload,
        "policy": production_case.policy,
        "repetition": production_case.repetition,
        "warmup": production_case.warmup,
        "policy_order": production_case.policy_order,
        "paired_order": list(production_case.paired_order),
        "status": "PASS",
        "output_match": True,
        "capacity_snapshot": {
            "scheduler_visible_blocks": worker_result["capacity"][
                "scheduler_visible_blocks"
            ],
        },
        "request_throughput_rps": (
            len(request_rows) / (duration_ns / 1_000_000_000.0)
        ),
        "decode_throughput_tps": (
            decoded_tokens
            / (decode_duration_ns / 1_000_000_000.0)
        ),
        "p95_itl_ns": _nearest_rank(itl_values, 0.95),
        "p99_itl_ns": _nearest_rank(itl_values, 0.99),
        "peak_reserved_bytes": max(
            int(row["reserved_bytes"]) for row in memory_rows
        ),
        "initialization_duration_ns": int(
            model_step["initialization_duration_ns"]
        ),
        "dispatch_events": dispatch_contract_rows,
        "capture_events": list(worker_result["capture_rows"]),
        "replay_after_rejection": any(
            row.get("dispatch") == "graph"
            and row.get("cache_state") != "ready"
            for row in worker_result["dispatch_rows"]
        ),
        "graph_hits": graph_hits,
        "graph_eligible_steps": int(
            model_step["graph_eligible_steps"]
        ),
    }


def pair_worker_results(
    baseline: dict,
    candidate: dict,
    *,
    matrix_by_id: dict,
    correctness_by_case: dict | None = None,
) -> dict:
    if baseline.get("policy") != "baseline":
        raise ValueError("baseline worker result has wrong policy")
    if candidate.get("policy") != "candidate":
        raise ValueError("candidate worker result has wrong policy")
    pair_fields = ("workload", "repetition", "warmup")
    if any(
        baseline.get(field) != candidate.get(field)
        for field in pair_fields
    ):
        raise ValueError("worker results are not a policy pair")
    source_sha256 = {
        row.get("source_sha256")
        for result in (baseline, candidate)
        for evidence_name in (
            "request_rows",
            "model_step_rows",
            "memory_rows",
        )
        for row in result[evidence_name]
    }
    if len(source_sha256) != 1:
        raise ValueError("worker pair has mixed source identity")
    source_sha = source_sha256.pop()
    if correctness_by_case is None:
        output_match = (
            baseline["output_token_ids"]
            == candidate["output_token_ids"]
        )
        logits_match = (
            baseline["logit_step_sha256"]
            == candidate["logit_step_sha256"]
        )
        kv_match = (
            baseline["live_slot_kv_step_sha256"]
            == candidate["live_slot_kv_step_sha256"]
        )
        baseline_kv_sha = _single_or_combined_sha256(
            baseline["live_slot_kv_step_sha256"]
        )
        candidate_kv_sha = _single_or_combined_sha256(
            candidate["live_slot_kv_step_sha256"]
        )
        correctness_rows = []
        for result, live_slot_kv_sha in (
            (baseline, baseline_kv_sha),
            (candidate, candidate_kv_sha),
        ):
            correctness_rows.append({
                "row_id": f"{result['case_id']}:correctness",
                "case_id": result["case_id"],
                "source_sha256": source_sha,
                "output_token_ids": result["output_token_ids"],
                "reference_token_ids": baseline["output_token_ids"],
                "logits_close": logits_match,
                "live_slot_kv_sha256": live_slot_kv_sha,
                "reference_live_slot_kv_sha256": baseline_kv_sha,
            })
    else:
        correctness_rows = [
            correctness_by_case[baseline["case_id"]],
            correctness_by_case[candidate["case_id"]],
        ]
        output_match = all(
            row["output_token_ids"] == row["reference_token_ids"]
            for row in correctness_rows
        )
        logits_match = all(
            row["logits_close"] is True
            for row in correctness_rows
        )
        kv_match = all(
            row["live_slot_kv_sha256"]
            == row["reference_live_slot_kv_sha256"]
            for row in correctness_rows
        )
    summaries = []
    for result in (baseline, candidate):
        case = matrix_by_id[result["case_id"]]
        summary = _case_summary(result, case)
        summary["output_match"] = (
            output_match and logits_match and kv_match
        )
        summaries.append(summary)
    return {
        "correctness_rows": correctness_rows,
        "case_summaries": summaries,
        "dispatch_rows": [
            row
            for result in (baseline, candidate)
            for row in result["dispatch_rows"]
        ],
        "capture_rows": [
            row
            for result in (baseline, candidate)
            for row in result["capture_rows"]
        ],
        "request_rows": [
            row
            for result in (baseline, candidate)
            for row in result["request_rows"]
        ],
        "model_step_rows": [
            row
            for result in (baseline, candidate)
            for row in result["model_step_rows"]
        ],
        "memory_rows": [
            row
            for result in (baseline, candidate)
            for row in result["memory_rows"]
        ],
    }


def classify_production_run(
    mode: str,
    case_summaries: list[dict],
) -> dict:
    if mode in {"correctness-smoke", "arrival-smoke"}:
        return {
            "classification": "NON_AUTHORITATIVE_SMOKE",
            "failures": [],
            "metrics": {},
            "thresholds": dict(contract.PRODUCTION_THRESHOLDS),
        }
    return contract.classify_production_gate(
        case_summaries,
        producer_summary={"classification": "GO"},
        independent_summary={"classification": "GO"},
    )


def merge_budget_fallback_worker_results(
    plan: list[dict],
    worker_results: dict[str, dict],
) -> dict:
    aggregate = {
        "budget_fallback_rows": [],
        "dispatch_rows": [],
        "capture_rows": [],
        "correctness_rows": [],
        "case_summaries": [],
    }
    seen_row_ids = set()
    for case in plan:
        case_id = case["case_id"]
        result = worker_results.get(case_id)
        if result is None:
            raise ValueError(
                f"missing budget fallback worker: {case_id}"
            )
        budget_row = result.get("budget_fallback_row")
        if (
            not isinstance(budget_row, dict)
            or budget_row.get("case_id") != case_id
            or budget_row.get("reason") != case["reason"]
        ):
            raise ValueError(
                f"invalid budget fallback worker result: {case_id}"
            )
        row_groups = {
            "budget_fallback_rows": [budget_row],
            "dispatch_rows": result.get("dispatch_rows"),
            "capture_rows": result.get("capture_rows"),
            "correctness_rows": result.get("correctness_rows"),
        }
        for name, rows in row_groups.items():
            if not isinstance(rows, list):
                raise ValueError(
                    f"{case_id}: missing {name} evidence"
                )
            for row in rows:
                if (
                    not isinstance(row, dict)
                    or row.get("case_id") != case_id
                    or not isinstance(row.get("row_id"), str)
                    or not row["row_id"]
                ):
                    raise ValueError(
                        f"{case_id}: invalid {name} row"
                    )
                if row["row_id"] in seen_row_ids:
                    raise ValueError(
                        f"duplicate budget fallback row_id: "
                        f"{row['row_id']}"
                    )
                seen_row_ids.add(row["row_id"])
                aggregate[name].append(row)
    return aggregate


def resolve_budget_fallback_aggregate(
    *,
    mode: str,
    worker_results: dict[str, dict],
    correctness_binding: dict | None,
) -> dict:
    plan = build_budget_fallback_plan(mode)
    if plan:
        return merge_budget_fallback_worker_results(
            plan,
            worker_results,
        )
    if correctness_binding is None:
        raise ValueError(
            "arrival mode requires a correctness binding"
        )
    aggregate = correctness_binding.get("budget_fallback")
    if not isinstance(aggregate, dict):
        raise ValueError(
            "correctness binding lacks budget fallback evidence"
        )
    return aggregate


def merge_budget_fallback_evidence(
    aggregate: dict,
    budget_fallback: dict,
) -> list[dict]:
    if budget_fallback.get("case_summaries") != []:
        raise ValueError(
            "budget fallback workers cannot contribute performance rows"
        )
    for target_name, source_name in (
        ("dispatch_rows", "dispatch_rows"),
        ("capture_rows", "capture_rows"),
        ("correctness_rows", "correctness_rows"),
    ):
        rows = budget_fallback.get(source_name)
        if not isinstance(rows, list):
            raise ValueError(
                f"budget fallback evidence lacks {source_name}"
            )
        aggregate[target_name].extend(rows)
    budget_rows = budget_fallback.get("budget_fallback_rows")
    if not isinstance(budget_rows, list):
        raise ValueError(
            "budget fallback evidence lacks budget_fallback_rows"
        )
    return budget_rows


def write_budget_fallback_artifact(
    run_dir: Path,
    rows: list[dict],
) -> str:
    path = run_dir / "budget_fallback_rows.jsonl"
    _write_jsonl(path, rows)
    return contract.sha256_file(path)


def _engine_capacity(engine) -> dict:
    runner = engine.model_runner
    visible = int(runner.config.num_kvcache_blocks)
    physical = int(runner._physical_num_kvcache_blocks)
    scratch = len(runner._exact_graph_scratch_block_ids)
    return {
        "physical_blocks": physical,
        "scratch_blocks": scratch,
        "scheduler_visible_blocks": visible,
        "block_size": int(runner.block_size),
    }


def _base_engine_config(
    *,
    feature_enabled: bool,
    num_kvcache_blocks: int,
) -> dict:
    config = {
        "enforce_eager": False,
        "multi_sequence_cuda_graphs": feature_enabled,
        "max_num_seqs": 16,
        "max_num_batched_tokens": 4096,
        "max_num_prefill_tokens_per_step": 256,
    }
    if num_kvcache_blocks > 0:
        config["num_kvcache_blocks"] = int(num_kvcache_blocks)
    return config


def _tensor_sha256(tensor) -> str:
    detached = tensor.detach().cpu().contiguous()
    return hashlib.sha256(
        detached.view(__import__("torch").uint8).numpy().tobytes()
    ).hexdigest()


def _live_kv_sha256(engine) -> str:
    import torch

    block_ids = sorted(
        int(block_id)
        for block_id in engine.scheduler.block_manager.used_block_ids
    )
    if not block_ids:
        return hashlib.sha256(b"empty-live-kv").hexdigest()
    selected = engine.model_runner.kv_cache[
        :,
        :,
        torch.tensor(
            block_ids,
            device=engine.model_runner.kv_cache.device,
            dtype=torch.long,
        ),
    ]
    return _tensor_sha256(selected)


def _identity_fields_for_event(engine, event: dict) -> dict | None:
    if (
        event.get("graph_identity_sha256") is None
        or event.get("page_table_width") is None
        or event.get("effective_num_splits") is None
    ):
        return None
    from tinyvllm.engine.flash_attn_split_policy import (
        FlashAttentionSplitInputs,
        build_flash_attn_263_graph_identity,
    )
    import flash_attn
    import torch

    runner = engine.model_runner
    hf_config = runner.config.hf_config
    identity = build_flash_attn_263_graph_identity(
        graph_batch_size=int(event["active_batch_size"]),
        inputs=FlashAttentionSplitInputs(
            batch_size=int(event["active_batch_size"]),
            num_query_heads=int(
                hf_config.num_attention_heads // runner.world_size
            ),
            num_kv_heads=int(
                hf_config.num_key_value_heads // runner.world_size
            ),
            head_dim=int(hf_config.head_dim),
            page_block_size=int(runner.block_size),
            page_table_width=int(event["page_table_width"]),
            max_seqlen_q=1,
            multi_processor_count=int(
                torch.cuda.get_device_properties(
                    runner.kv_cache.device
                ).multi_processor_count
            ),
        ),
        flash_attn_version=str(flash_attn.__version__),
        require_exact_batch=True,
    )
    if identity.sha256 != event["graph_identity_sha256"]:
        raise RuntimeError("runtime graph identity cannot be rebuilt")
    if identity.effective_num_splits != int(
        event["effective_num_splits"]
    ):
        raise RuntimeError("runtime split identity drift")
    return asdict(identity)


def _capacity_worker(args) -> dict:
    from tinyvllm.engine.llm_engine import LLMEngine

    engine = LLMEngine(
        REMOTE_MODEL,
        **_base_engine_config(
            feature_enabled=True,
            num_kvcache_blocks=args.num_kvcache_blocks,
        ),
    )
    try:
        capacity = _engine_capacity(engine)
    finally:
        engine.exit()
    return {
        "schema_version": 1,
        "source_sha256": args.source_sha256,
        **capacity,
    }


def _run_engine_workload(args) -> dict:
    from tinyvllm.engine.llm_engine import LLMEngine
    from tinyvllm.sampling_params import SamplingParams

    feature_enabled = args.policy == "candidate"
    initialization_started_ns = time.monotonic_ns()
    engine = LLMEngine(
        REMOTE_MODEL,
        **_base_engine_config(
            feature_enabled=feature_enabled,
            num_kvcache_blocks=args.num_kvcache_blocks,
        ),
    )
    initialization_duration_ns = (
        time.monotonic_ns() - initialization_started_ns
    )
    workload = (
        build_correctness_workload(
            workload=args.workload,
            repetition=args.repetition,
            warmup=args.warmup,
        )
        if args.worker_kind == "correctness"
        else build_arrival_workload(
            workload=args.workload,
            repetition=args.repetition,
            warmup=args.warmup,
        )
    )
    request_rows = []
    dispatch_rows = []
    capture_rows = []
    model_step_rows = []
    memory_rows = []
    logit_step_sha256 = []
    live_slot_kv_step_sha256 = []
    original_run_model = engine.model_runner.run_model

    if args.worker_kind == "correctness":
        def observed_run_model(*run_args, **run_kwargs):
            result = original_run_model(*run_args, **run_kwargs)
            logits = result[0] if isinstance(result, tuple) else result
            logit_step_sha256.append(_tensor_sha256(logits))
            return result

        engine.model_runner.run_model = observed_run_model
    try:
        start_ns = time.monotonic_ns()
        pending = list(enumerate(workload["requests"]))
        sequence_to_request = {}
        token_timestamps = {}
        completed = {}
        actual_arrival_ns = {}
        step_index = 0
        decode_duration_ns = 0
        decoded_tokens = 0
        graph_eligible_steps = 0
        last_dispatch_step_id = None
        while pending or not engine.is_finished():
            elapsed_ns = time.monotonic_ns() - start_ns
            while (
                pending
                and pending[0][1]["arrival_offset_ns"] <= elapsed_ns
            ):
                request_index, request = pending.pop(0)
                engine.add_request(
                    request["prompt_token_ids"],
                    SamplingParams(
                        temperature=0.0,
                        max_tokens=request[
                            "requested_output_tokens"
                        ],
                        ignore_eos=True,
                    ),
                )
                seq = engine.scheduler.waiting[-1]
                sequence_to_request[int(seq.seq_id)] = request_index
                actual_arrival_ns[request_index] = time.monotonic_ns()
            if engine.is_finished():
                next_offset_ns = pending[0][1]["arrival_offset_ns"]
                sleep_ns = max(
                    0,
                    start_ns + next_offset_ns - time.monotonic_ns(),
                )
                if sleep_ns:
                    time.sleep(sleep_ns / 1_000_000_000.0)
                continue
            step_started_ns = time.monotonic_ns()
            outputs, num_tokens = engine.step()
            step_ended_ns = time.monotonic_ns()
            if num_tokens < 0:
                decode_duration_ns += step_ended_ns - step_started_ns
                decoded_tokens += -num_tokens
            event = (
                engine.model_runner.cuda_graph_dispatch_observation()
            )
            if (
                event is not None
                and feature_enabled
                and event["step_id"] != last_dispatch_step_id
            ):
                last_dispatch_step_id = event["step_id"]
                graph_eligible_steps += int(
                    event["mode"] == "decode"
                    and event["active_batch_size"] in (2, 4, 8)
                )
                identity_fields = _identity_fields_for_event(
                    engine,
                    event,
                )
                dispatch_row = {
                    "row_id": f"{args.case_id}:dispatch:{step_index}",
                    "case_id": args.case_id,
                    "source_sha256": args.source_sha256,
                    **event,
                }
                if identity_fields is not None:
                    dispatch_row["identity_fields"] = identity_fields
                    dispatch_rows.append(dispatch_row)
                    if (
                        event["capture_attempted"]
                        and event["capture_duration_ns"] > 0
                        and event["graph_identity_sha256"]
                        in engine.model_runner.exact_cuda_graph_cache.summary()[
                            "ready_entries"
                        ]
                    ):
                        capture_rows.append({
                            "row_id": (
                                f"{args.case_id}:capture:{step_index}"
                            ),
                            "case_id": args.case_id,
                            "step_id": event["step_id"],
                            "source_sha256": args.source_sha256,
                            "graph_identity_sha256": event[
                                "graph_identity_sha256"
                            ],
                            "identity_fields": identity_fields,
                            "observation_count": event[
                                "observation_count"
                            ],
                            "status": "ready",
                            "fallback_reason": None,
                            "capture_duration_ns": event[
                                "capture_duration_ns"
                            ],
                            "static_bytes": event[
                                "capture_static_bytes"
                            ],
                            "reserved_delta_bytes": event[
                                "capture_reserved_delta_bytes"
                            ],
                            "budget_overshoot": False,
                        })
            observation = engine.last_step_observation or {}
            step_end_observed_ns = int(
                observation.get("step_end_ns") or step_ended_ns
            )
            for seq_id_text, tokens in observation.get(
                "new_completion_tokens_by_seq",
                {},
            ).items():
                seq_id = int(seq_id_text)
                request_index = sequence_to_request.get(seq_id)
                if request_index is None:
                    continue
                token_timestamps.setdefault(request_index, []).extend(
                    [step_end_observed_ns] * len(tokens)
                )
            memory_rows.append({
                "row_id": f"{args.case_id}:memory:{step_index}",
                "case_id": args.case_id,
                "source_sha256": args.source_sha256,
                "reserved_bytes": int(
                    observation.get("memory", {}).get(
                        "cuda_reserved_bytes",
                        __import__("torch").cuda.memory_reserved(),
                    )
                ),
            })
            for seq_id, tokens in outputs:
                request_index = sequence_to_request[int(seq_id)]
                completed[request_index] = list(tokens)
            if args.worker_kind == "correctness":
                live_slot_kv_step_sha256.append(
                    _live_kv_sha256(engine)
                )
            step_index += 1
        end_ns = time.monotonic_ns()
        ordered_outputs = [
            completed[request_index]
            for request_index in range(len(workload["requests"]))
        ]
        for request_index, request in enumerate(workload["requests"]):
            tokens = ordered_outputs[request_index]
            timestamps = token_timestamps.get(request_index, [])
            arrival_ns = actual_arrival_ns[request_index]
            if not timestamps:
                raise RuntimeError("request completed without token timestamps")
            itl_ns = [max(1, timestamps[0] - arrival_ns)]
            itl_ns.extend(
                max(1, current - previous)
                for previous, current in zip(
                    timestamps,
                    timestamps[1:],
                )
            )
            request_rows.append({
                "row_id": f"{args.case_id}:request:{request_index}",
                "case_id": args.case_id,
                "request_id": request["request_id"],
                "source_sha256": args.source_sha256,
                "scheduled_arrival_ns": request["arrival_offset_ns"],
                "actual_arrival_ns": arrival_ns,
                "token_timestamps_ns": timestamps,
                "output_token_ids": tokens,
                "itl_ns": itl_ns,
            })
        model_step_rows.append({
            "row_id": f"{args.case_id}:model-step",
            "case_id": args.case_id,
            "source_sha256": args.source_sha256,
            "measurement_duration_ns": max(1, end_ns - start_ns),
            "decode_duration_ns": max(1, decode_duration_ns),
            "decoded_tokens": max(1, decoded_tokens),
            "initialization_duration_ns": max(
                1, initialization_duration_ns
            ),
            "graph_eligible_steps": graph_eligible_steps,
        })
        return {
            "case_id": args.case_id,
            "policy": args.policy,
            "workload": args.workload,
            "repetition": args.repetition,
            "warmup": args.warmup,
            "capacity": _engine_capacity(engine),
            "request_rows": request_rows,
            "dispatch_rows": dispatch_rows,
            "capture_rows": capture_rows,
            "model_step_rows": model_step_rows,
            "memory_rows": memory_rows,
            "output_token_ids": ordered_outputs,
            "logit_step_sha256": logit_step_sha256,
            "live_slot_kv_step_sha256": live_slot_kv_step_sha256,
            "pid": os.getpid(),
        }
    finally:
        engine.exit()


def _worker_main(args) -> int:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.worker_kind == "capacity":
        result = _capacity_worker(args)
        _write_json(output_dir / "capacity.json", result)
    elif args.worker_kind in {"arrival", "correctness"}:
        result = _run_engine_workload(args)
        _write_json(output_dir / "worker_result.json", result)
        _write_jsonl(
            output_dir / "dispatch_events.jsonl",
            result["dispatch_rows"],
        )
        _write_jsonl(
            output_dir / "capture_events.jsonl",
            result["capture_rows"],
        )
        _write_jsonl(
            output_dir / "request_metrics.jsonl",
            result["request_rows"],
        )
        _write_jsonl(
            output_dir / "model_step_metrics.jsonl",
            result["model_step_rows"],
        )
        _write_jsonl(
            output_dir / "memory_trace.jsonl",
            result["memory_rows"],
        )
    else:
        raise ValueError(f"unknown worker kind: {args.worker_kind}")
    return 0


def _create_source_snapshot(run_dir: Path) -> tuple[Path, dict]:
    source_audit = _load_tool(
        "exact_cuda_production_source_audit",
        "source_audit.py",
    )
    staging = run_dir / "staging"
    if staging.exists():
        shutil.rmtree(staging)
    evidence = source_audit.build_source_evidence(
        ROOT,
        staging,
        owned_roots=OWNED_SOURCE_ROOTS,
        ignored_untracked_prefixes=IGNORED_UNTRACKED_PREFIXES,
    )
    source_audit.validate_source_snapshot(
        staging / "source",
        evidence,
        staging / "source.patch",
        expected_owned_roots=OWNED_SOURCE_ROOTS,
    )
    return staging, evidence


def _stream_snapshot(staging: Path, remote_dir: str) -> None:
    _run_remote(["mkdir", "-p", remote_dir])
    producer = subprocess.Popen(
        ["tar", "-C", str(staging), "-czf", "-", "."],
        stdout=subprocess.PIPE,
    )
    if producer.stdout is None:
        raise RuntimeError("could not open tar stream")
    consumer = subprocess.run(
        _ssh_argv(["tar", "-xzf", "-", "-C", remote_dir]),
        stdin=producer.stdout,
        capture_output=True,
        check=False,
    )
    producer.stdout.close()
    producer_returncode = producer.wait()
    if producer_returncode != 0 or consumer.returncode != 0:
        raise RuntimeError(
            consumer.stderr.decode(
                "utf-8",
                errors="replace",
            ).strip()
            or "source snapshot stream failed"
        )


def _remote_validate_source(remote_dir: str) -> None:
    script = (
        "import json\n"
        "from pathlib import Path\n"
        "from tools import source_audit\n"
        f"root=Path({remote_dir!r})\n"
        "e=json.loads((root/'source_evidence.json').read_text())\n"
        "source_audit.validate_source_snapshot("
        "root/'source',e,root/'source.patch',"
        f"expected_owned_roots={OWNED_SOURCE_ROOTS!r})\n"
    )
    _run_remote([
        "env",
        "PYTHONDONTWRITEBYTECODE=1",
        f"PYTHONPATH={remote_dir}/source",
        REMOTE_PYTHON,
        "-c",
        script,
    ])


def _gpu_occupancy() -> list[dict]:
    result = _run_remote([
        "nvidia-smi",
        "--query-compute-apps=pid,process_name,used_memory",
        "--format=csv,noheader,nounits",
        "-i",
        CUDA_VISIBLE_DEVICES,
    ], check=False)
    rows = []
    for line in result.stdout.decode().splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) == 3 and parts[0].isdigit():
            rows.append({
                "pid": int(parts[0]),
                "process_name": parts[1],
                "used_memory_mib": int(parts[2]),
            })
    return rows


class GpuOccupancyError(RuntimeError):
    def __init__(self, *, stage: str, occupancy: list[dict]):
        self.stage = stage
        self.occupancy = occupancy
        super().__init__(
            f"GPU 0 is occupied {stage}: {occupancy}"
        )


def _run_preflight(remote_dir: str) -> None:
    source = f"{remote_dir}/source"
    commands = (
        [REMOTE_PYTHON, "tools/test_model_runner_spec_verify.py"],
        [REMOTE_PYTHON, "tools/test_multi_sequence_cuda_graph_gate.py"],
        [
            REMOTE_PYTHON,
            "-m",
            "py_compile",
            "tinyvllm/engine/model_runner.py",
            "tools/verify_multi_sequence_cuda_graph_production.py",
            "tools/run_multi_sequence_cuda_graph_production_gate_remote.py",
        ],
    )
    for command in commands:
        _run_remote([
            "env",
            "PYTHONDONTWRITEBYTECODE=1",
            f"PYTHONPATH={source}",
            "bash",
            "-lc",
            "cd "
            + shlex.quote(source)
            + " && "
            + " ".join(shlex.quote(value) for value in command),
        ])


def _download_remote_tree(remote_dir: str, local_dir: Path) -> None:
    if local_dir.exists():
        raise ValueError(f"local run directory exists: {local_dir}")
    local_dir.parent.mkdir(parents=True, exist_ok=True)
    archive = local_dir.with_suffix(".tar.partial")
    with archive.open("wb") as handle:
        result = subprocess.run(
            _ssh_argv([
                "tar",
                "-C",
                str(Path(remote_dir).parent),
                "-cf",
                "-",
                Path(remote_dir).name,
            ]),
            stdout=handle,
            stderr=subprocess.PIPE,
            check=False,
        )
    if result.returncode != 0:
        archive.unlink(missing_ok=True)
        raise RuntimeError(
            result.stderr.decode("utf-8", errors="replace")
        )
    partial = local_dir.with_name(local_dir.name + ".partial")
    partial.mkdir()
    with tarfile.open(archive, "r:") as tar:
        for member in tar.getmembers():
            path = Path(member.name)
            if path.is_absolute() or ".." in path.parts:
                raise ValueError("unsafe remote artifact path")
        tar.extractall(partial)
    extracted = partial / Path(remote_dir).name
    extracted.replace(local_dir)
    partial.rmdir()
    archive.unlink()


def _verify_local(run_dir: Path) -> dict:
    verifier = _load_tool(
        "exact_cuda_production_verifier_for_runner",
        "verify_multi_sequence_cuda_graph_production.py",
    )
    return verifier.verify_run(run_dir, write_report=True)


def _remote_allocate_port_pair() -> tuple[int, int]:
    script = (
        "import json,socket\n"
        "handles=[]\n"
        "try:\n"
        "  ports=[]\n"
        "  while len(ports)<2:\n"
        "    handle=socket.socket(socket.AF_INET,socket.SOCK_STREAM)\n"
        "    handle.bind(('127.0.0.1',0));handles.append(handle)\n"
        "    port=handle.getsockname()[1]\n"
        "    if port not in ports:ports.append(port)\n"
        "  print(json.dumps(ports))\n"
        "finally:\n"
        "  [handle.close() for handle in handles]\n"
    )
    result = _run_remote([REMOTE_PYTHON, "-c", script])
    ports = json.loads(result.stdout.decode("utf-8"))
    return int(ports[0]), int(ports[1])


def _copy_source_artifacts(staging: Path, run_dir: Path) -> None:
    for name in ("source_evidence.json", "source.patch"):
        shutil.copyfile(staging / name, run_dir / name)


def _source_file_hashes(source_evidence: dict) -> dict:
    return {
        row["path"]: row["sha256"]
        for row in source_evidence["files"]
    }


def _production_environment(source_sha256: str) -> dict:
    script = (
        "import json,torch,flash_attn\n"
        "p=torch.cuda.get_device_properties(0)\n"
        "print(json.dumps({"
        "'torch':torch.__version__,"
        "'cuda':torch.version.cuda,"
        "'flash_attn':flash_attn.__version__,"
        "'gpu':p.name,"
        "'multi_processor_count':p.multi_processor_count,"
        "'bf16_supported':torch.cuda.is_bf16_supported()"
        "},sort_keys=True))\n"
    )
    result = _run_remote([
        "env",
        f"CUDA_VISIBLE_DEVICES={CUDA_VISIBLE_DEVICES}",
        REMOTE_PYTHON,
        "-c",
        script,
    ])
    return {
        "source_tree_sha256": source_sha256,
        "model": {
            "path": REMOTE_MODEL,
            "dtype": "bfloat16",
            "local_files_only": True,
        },
        **json.loads(result.stdout.decode("utf-8")),
    }


def _load_correctness_binding(
    run_tag: str,
    source_sha256: str,
) -> dict:
    run_dir = OUTPUT_ROOT / _safe_run_tag(run_tag)
    manifest = _read_json(run_dir / "manifest.json")
    verification = _read_json(
        run_dir / "independent_verification.json"
    )
    if (
        verification.get("classification") != "GO"
        or verification.get("failures") not in (None, [])
    ):
        raise ValueError(
            "correctness independent verification is not GO"
        )
    rows = _read_jsonl(run_dir / "correctness_rows.jsonl")
    budget_fallback_rows = _read_jsonl(
        run_dir / "budget_fallback_rows.jsonl"
    )
    dispatch_rows = _read_jsonl(run_dir / "dispatch_events.jsonl")
    capture_rows = _read_jsonl(run_dir / "capture_events.jsonl")
    diagnostic_binding = _read_json(
        run_dir / "diagnostic_binding.json"
    )
    required_budget_fallbacks = len(
        contract.BUDGET_FALLBACK_REASONS
    )
    if (
        verification.get("budget_fallback_required")
        != required_budget_fallbacks
        or verification.get("budget_fallback_verified")
        != required_budget_fallbacks
        or verification.get("budget_fallback_reasons")
        != list(contract.BUDGET_FALLBACK_REASONS)
    ):
        raise ValueError(
            "correctness binding lacks verified budget fallback evidence"
        )
    if manifest["source_tree_sha256"] != source_sha256:
        raise ValueError("correctness binding source mismatch")
    budget_case_ids = {
        row["case_id"] for row in budget_fallback_rows
    }
    production_rows = [
        row for row in rows
        if row.get("case_id") not in budget_case_ids
    ]
    if len(production_rows) != len(
        contract.build_production_matrix()
    ):
        raise ValueError("correctness binding is not canonical-complete")
    if any(
        row["output_token_ids"] != row["reference_token_ids"]
        or row["logits_close"] is not True
        or row["live_slot_kv_sha256"]
        != row["reference_live_slot_kv_sha256"]
        for row in production_rows
    ):
        raise ValueError("correctness binding contains a mismatch")
    if (
        diagnostic_binding.get("required") is not True
        or diagnostic_binding.get("source_tree_sha256")
        != source_sha256
    ):
        raise ValueError(
            "correctness binding lacks canonical diagnostic evidence"
        )
    return {
        "run_tag": run_tag,
        "verification": verification,
        "rows": {
            row["case_id"]: row for row in production_rows
        },
        "budget_fallback": {
            "budget_fallback_rows": budget_fallback_rows,
            "dispatch_rows": [
                row for row in dispatch_rows
                if row.get("case_id") in budget_case_ids
            ],
            "capture_rows": [
                row for row in capture_rows
                if row.get("case_id") in budget_case_ids
            ],
            "correctness_rows": [
                row for row in rows
                if row.get("case_id") in budget_case_ids
            ],
            "case_summaries": [],
        },
        "diagnostic_binding": diagnostic_binding,
    }


def _run_canonical_diagnostic(
    *,
    run_tag: str,
    source_sha256: str,
) -> dict:
    occupancy = _gpu_occupancy()
    if occupancy:
        raise GpuOccupancyError(
            stage="before_canonical_diagnostic",
            occupancy=occupancy,
        )
    command = build_canonical_diagnostic_command(run_tag=run_tag)
    result = subprocess.run(
        command,
        cwd=ROOT,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        incomplete_path = (
            OUTPUT_ROOT
            / f"{run_tag}-diagnostic"
            / "incomplete.json"
        )
        if incomplete_path.is_file():
            incomplete = _read_json(incomplete_path)
            if (
                incomplete.get("failure_reason")
                == "unrelated_gpu_occupancy"
            ):
                raise GpuOccupancyError(
                    stage=str(incomplete.get("stage")),
                    occupancy=list(incomplete.get("occupancy", [])),
                )
        raise RuntimeError(
            result.stderr.decode("utf-8", errors="replace")
            or result.stdout.decode("utf-8", errors="replace")
            or "canonical diagnostic failed"
        )
    occupancy = _gpu_occupancy()
    if occupancy:
        raise GpuOccupancyError(
            stage="after_canonical_diagnostic",
            occupancy=occupancy,
        )
    return load_canonical_diagnostic_binding(
        OUTPUT_ROOT / f"{run_tag}-diagnostic",
        expected_source_sha256=source_sha256,
    )


def _write_artifact_hashes_after_go(run_dir: Path) -> None:
    names = tuple(contract.PRODUCTION_ARTIFACT_FILES)
    if any(not (run_dir / name).is_file() for name in names):
        raise ValueError("production artifact set is incomplete")
    _write_json(
        run_dir / "artifact_hashes.json",
        {
            name: contract.sha256_file(run_dir / name)
            for name in names
        },
    )


def _run_remote_worker(
    remote_dir: str,
    *,
    command: dict,
) -> subprocess.CompletedProcess:
    occupancy_before = _gpu_occupancy()
    if occupancy_before:
        raise GpuOccupancyError(
            stage="before_worker",
            occupancy=occupancy_before,
        )
    env_args = [
        f"{name}={value}"
        for name, value in command["env"].items()
    ]
    result = _run_remote([
        "env",
        *env_args,
        "bash",
        "-lc",
        "cd "
        + shlex.quote(command["cwd"])
        + " && "
        + " ".join(
            shlex.quote(value) for value in command["argv"]
        ),
    ], check=False)
    occupancy_after = _gpu_occupancy()
    if occupancy_after:
        raise GpuOccupancyError(
            stage="after_worker",
            occupancy=occupancy_after,
        )
    return result


def _run_worker_with_retry(
    *,
    remote_source: str,
    remote_output: str,
    case: dict,
    source_sha256: str,
    visible_blocks: int,
    used_ports: set[int],
) -> tuple[dict, dict]:
    for attempt in range(1, 4):
        dist_port, master_port = _remote_allocate_port_pair()
        if (
            dist_port == master_port
            or dist_port in used_ports
            or master_port in used_ports
        ):
            continue
        if case["worker_kind"] == "budget-fallback":
            command = build_budget_fallback_worker_command(
                remote_source=remote_source,
                output_dir=remote_output,
                source_sha256=source_sha256,
                dist_port=dist_port,
                master_port=master_port,
                reason=case["reason"],
                visible_blocks=visible_blocks,
            )
        else:
            command = build_worker_command(
                remote_source=remote_source,
                output_dir=remote_output,
                worker_kind=case["worker_kind"],
                source_sha256=source_sha256,
                dist_port=dist_port,
                master_port=master_port,
                case_id=case["case_id"],
                policy=case["policy"],
                workload=case["workload"],
                repetition=case["repetition"],
                warmup=case["warmup"],
                visible_blocks=visible_blocks,
            )
        result = _run_remote_worker(
            str(Path(remote_output).parent),
            command=command,
        )
        stdout = result.stdout.decode("utf-8", errors="replace")
        stderr = result.stderr.decode("utf-8", errors="replace")
        if result.returncode == 0:
            used_ports.update((dist_port, master_port))
            return command, {
                "tinyvllm_dist_port": dist_port,
                "master_port": master_port,
                "attempts": attempt,
                "stdout": stdout,
                "stderr": stderr,
            }
        if (
            "EADDRINUSE" not in stderr
            and "Address already in use" not in stderr
        ):
            raise RuntimeError(stderr or stdout or "remote worker failed")
    raise RuntimeError("remote worker exhausted EADDRINUSE retries")


def _write_production_artifacts(
    *,
    mode: str,
    run_dir: Path,
    run_tag: str,
    source_evidence: dict,
    environment: dict,
    paired_capacity: dict,
    case_plan: list[dict],
    worker_results: dict[str, dict],
    budget_fallback: dict,
    process_rows: list[dict],
    ports: dict,
    correctness_binding: dict | None,
    diagnostic_binding: dict,
) -> dict:
    matrix = production_matrix_for_mode(mode)
    matrix_by_id = {case.case_id: case for case in matrix}
    aggregate = {
        "dispatch_rows": [],
        "capture_rows": [],
        "request_rows": [],
        "model_step_rows": [],
        "memory_rows": [],
        "correctness_rows": [],
        "case_summaries": [],
    }
    seen_pairs = set()
    for case in matrix:
        pair_key = (case.workload, case.repetition, case.warmup)
        if pair_key in seen_pairs:
            continue
        seen_pairs.add(pair_key)
        baseline = next(
            value
            for value in matrix
            if (
                value.workload,
                value.repetition,
                value.warmup,
                value.policy,
            ) == (*pair_key, "baseline")
        )
        candidate = next(
            value
            for value in matrix
            if (
                value.workload,
                value.repetition,
                value.warmup,
                value.policy,
            ) == (*pair_key, "candidate")
        )
        if (
            baseline.case_id not in worker_results
            or candidate.case_id not in worker_results
        ):
            continue
        paired = pair_worker_results(
            worker_results[baseline.case_id],
            worker_results[candidate.case_id],
            matrix_by_id=matrix_by_id,
            correctness_by_case=(
                None
                if correctness_binding is None
                else correctness_binding["rows"]
            ),
        )
        for name in aggregate:
            aggregate[name].extend(paired[name])

    matrix_order = {
        case.case_id: index for index, case in enumerate(matrix)
    }
    aggregate["case_summaries"].sort(
        key=lambda row: matrix_order[row["case_id"]]
    )
    aggregate["correctness_rows"].sort(
        key=lambda row: matrix_order[row["case_id"]]
    )
    budget_fallback_rows = merge_budget_fallback_evidence(
        aggregate,
        budget_fallback,
    )

    for filename, rows in (
        ("dispatch_events.jsonl", aggregate["dispatch_rows"]),
        ("capture_events.jsonl", aggregate["capture_rows"]),
        ("request_metrics.jsonl", aggregate["request_rows"]),
        ("model_step_metrics.jsonl", aggregate["model_step_rows"]),
        ("memory_trace.jsonl", aggregate["memory_rows"]),
        ("correctness_rows.jsonl", aggregate["correctness_rows"]),
    ):
        _write_jsonl(run_dir / filename, rows)
    budget_fallback_sha256 = write_budget_fallback_artifact(
        run_dir,
        budget_fallback_rows,
    )
    _write_json(
        run_dir / "case_summaries.json",
        aggregate["case_summaries"],
    )
    producer_summary = classify_production_run(
        mode,
        aggregate["case_summaries"],
    )
    _write_json(run_dir / "summary.json", producer_summary)
    _write_json(run_dir / "environment.json", environment)
    _write_json(
        run_dir / "diagnostic_binding.json",
        diagnostic_binding,
    )

    policy_configs = {
        "baseline": paired_capacity["baseline_config"],
        "candidate": paired_capacity["candidate_config"],
    }
    manifest = {
        "schema_version": 1,
        "mode": mode,
        "run_tag": run_tag,
        "source_tree_sha256": source_evidence["tree_sha256"],
        "copied_file_sha256": _source_file_hashes(source_evidence),
        "model_sha256": contract.canonical_json_sha256(
            environment["model"]
        ),
        "config_sha256": contract.canonical_json_sha256(
            policy_configs
        ),
        "commands": [
            "python",
            "tools/run_multi_sequence_cuda_graph_production_gate_remote.py",
            "local-contracts",
        ],
        "workload_sha256": contract.canonical_json_sha256([
            {
                "case_id": case.case_id,
                "workload": case.workload,
                "policy": case.policy,
                "repetition": case.repetition,
                "warmup": case.warmup,
                "policy_order": case.policy_order,
                "paired_order": list(case.paired_order),
            }
            for case in matrix
        ]),
        "arrival_sha256": contract.canonical_json_sha256([
            {
                "row_id": row["row_id"],
                "case_id": row["case_id"],
                "request_id": row["request_id"],
                "scheduled_arrival_ns": row["scheduled_arrival_ns"],
            }
            for row in aggregate["request_rows"]
        ]),
        "paired_policy_order": [
            {
                "workload": case.workload,
                "repetition": case.repetition,
                "paired_order": list(case.paired_order),
            }
            for case in matrix
            if case.policy_order == 0
        ],
        "processes": process_rows,
        "ports": ports,
        "policy_configs": policy_configs,
        "capacity": paired_capacity["capacity"],
        "thresholds": dict(contract.PRODUCTION_THRESHOLDS),
        "case_ids": [case["case_id"] for case in case_plan],
        "diagnostic_binding_sha256": (
            contract.sha256_file(run_dir / "diagnostic_binding.json")
        ),
        "budget_fallback_sha256": budget_fallback_sha256,
    }
    _write_json(run_dir / "manifest.json", manifest)
    hashed_names = (
        "environment.json",
        "diagnostic_binding.json",
        "dispatch_events.jsonl",
        "capture_events.jsonl",
        "request_metrics.jsonl",
        "model_step_metrics.jsonl",
        "memory_trace.jsonl",
        "correctness_rows.jsonl",
        "budget_fallback_rows.jsonl",
        "case_summaries.json",
        "summary.json",
    )
    _write_json(
        run_dir / "source_manifest.json",
        {
            "source_tree_sha256": source_evidence["tree_sha256"],
            "files": _source_file_hashes(source_evidence),
            "artifact_sha256": {
                name: contract.sha256_file(run_dir / name)
                for name in hashed_names
            },
        },
    )
    _write_json(
        run_dir / "independent_verification.json",
        {"classification": "PENDING"},
    )
    (run_dir / "report.md").write_text(
        "# Pending independent verification\n",
        encoding="utf-8",
    )
    if correctness_binding is not None:
        _write_json(
            run_dir / "correctness_binding.json",
            {
                "run_tag": correctness_binding["run_tag"],
                "verification": correctness_binding["verification"],
                "diagnostic_binding": correctness_binding[
                    "diagnostic_binding"
                ],
            },
        )
    return producer_summary


def _orchestrate(
    mode: str,
    run_tag: str,
    *,
    diagnostic_run_tag: str | None = None,
) -> Path:
    run_dir = OUTPUT_ROOT / run_tag
    if run_dir.exists():
        raise ValueError(f"run directory already exists: {run_dir}")
    run_dir.mkdir(parents=True)
    staging, source_evidence = _create_source_snapshot(run_dir)
    remote_dir = _remote_run_dir(run_tag)
    _stream_snapshot(staging, remote_dir)
    _remote_validate_source(remote_dir)
    _run_preflight(remote_dir)
    _copy_source_artifacts(staging, run_dir)
    if mode == "preflight":
        _write_json(run_dir / "source_manifest.json", source_evidence)
        return run_dir

    diagnostic_binding = {
        "required": False,
        "run_tag": None,
        "source_tree_sha256": source_evidence["tree_sha256"],
        "case_count": 0,
        "classifications": None,
    }
    if mode == "correctness-canonical":
        diagnostic_binding = _run_canonical_diagnostic(
            run_tag=run_tag,
            source_sha256=source_evidence["tree_sha256"],
        )

    capacity_output = f"{remote_dir}/capacity"
    capacity_dist_port, capacity_master_port = (
        _remote_allocate_port_pair()
    )
    command = build_worker_command(
        remote_source=f"{remote_dir}/source",
        output_dir=capacity_output,
        worker_kind="capacity",
        source_sha256=source_evidence["tree_sha256"],
        dist_port=capacity_dist_port,
        master_port=capacity_master_port,
        case_id="capacity",
        policy="candidate",
        workload="stable_exact_reuse",
        repetition=0,
        warmup=True,
        visible_blocks=0,
    )
    result = _run_remote_worker(remote_dir, command=command)
    if result.returncode != 0:
        raise RuntimeError(
            result.stderr.decode("utf-8", errors="replace")
        )
    _download_remote_tree(capacity_output, run_dir / "capacity")
    capacity = _read_json(run_dir / "capacity" / "capacity.json")
    paired = build_paired_capacity_contract(capacity)
    _write_json(run_dir / "capacity.json", paired)
    correctness_binding = (
        _load_correctness_binding(
            diagnostic_run_tag,
            source_evidence["tree_sha256"],
        )
        if mode.startswith("arrival-")
        else None
    )
    if correctness_binding is not None:
        diagnostic_binding = correctness_binding[
            "diagnostic_binding"
        ]
    case_plan = build_case_plan(mode)
    worker_results = {}
    process_rows = []
    ports = {}
    used_ports = {capacity_dist_port, capacity_master_port}
    remote_source = f"{remote_dir}/source"
    for case in case_plan:
        remote_output = (
            f"{remote_dir}/cases/{case['case_id']}/output"
        )
        _run_remote(["mkdir", "-p", remote_output])
        worker_command, execution = _run_worker_with_retry(
            remote_source=remote_source,
            remote_output=remote_output,
            case=case,
            source_sha256=source_evidence["tree_sha256"],
            visible_blocks=paired["candidate_config"][
                "num_kvcache_blocks"
            ],
            used_ports=used_ports,
        )
        local_case_dir = run_dir / "cases" / case["case_id"]
        _download_remote_tree(remote_output, local_case_dir)
        worker_result = _read_json(
            local_case_dir / "worker_result.json"
        )
        worker_results[case["case_id"]] = worker_result
        ports[case["case_id"]] = {
            "tinyvllm_dist_port": execution[
                "tinyvllm_dist_port"
            ],
            "master_port": execution["master_port"],
        }
        process_rows.append({
            "case_id": case["case_id"],
            "pid": worker_result["pid"],
            "command": [
                "python",
                "tools/run_multi_sequence_cuda_graph_production_gate_remote.py",
                "local-contracts",
                "--case-id",
                case["case_id"],
            ],
            "tinyvllm_dist_port": execution[
                "tinyvllm_dist_port"
            ],
            "master_port": execution["master_port"],
            "source_sha256": source_evidence["tree_sha256"],
        })
        _write_json(
            local_case_dir / "execution.json",
            {
                "command": worker_command,
                **execution,
            },
        )
    budget_fallback_plan = build_budget_fallback_plan(mode)
    budget_fallback_worker_results = {}
    for case in budget_fallback_plan:
        remote_output = (
            f"{remote_dir}/budget-fallback/{case['reason']}/output"
        )
        _run_remote(["mkdir", "-p", remote_output])
        worker_command, execution = _run_worker_with_retry(
            remote_source=remote_source,
            remote_output=remote_output,
            case=case,
            source_sha256=source_evidence["tree_sha256"],
            visible_blocks=paired["candidate_config"][
                "num_kvcache_blocks"
            ],
            used_ports=used_ports,
        )
        local_case_dir = (
            run_dir / "budget-fallback" / case["reason"]
        )
        _download_remote_tree(remote_output, local_case_dir)
        worker_result = _read_json(
            local_case_dir / "worker_result.json"
        )
        budget_fallback_worker_results[case["case_id"]] = (
            worker_result
        )
        _write_json(
            local_case_dir / "execution.json",
            {
                "command": worker_command,
                **execution,
            },
        )
    budget_fallback = resolve_budget_fallback_aggregate(
        mode=mode,
        worker_results=budget_fallback_worker_results,
        correctness_binding=correctness_binding,
    )
    environment = _production_environment(
        source_evidence["tree_sha256"]
    )
    _write_production_artifacts(
        mode=mode,
        run_dir=run_dir,
        run_tag=run_tag,
        source_evidence=source_evidence,
        environment=environment,
        paired_capacity=paired,
        case_plan=case_plan,
        worker_results=worker_results,
        budget_fallback=budget_fallback,
        process_rows=process_rows,
        ports=ports,
        correctness_binding=correctness_binding,
        diagnostic_binding=diagnostic_binding,
    )
    verification = _verify_local(run_dir)
    if verification["classification"] == "GO":
        _write_artifact_hashes_after_go(run_dir)
    return run_dir


def _parse_args(argv: list[str] | None = None):
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=MODES)
    parser.add_argument("--run-tag")
    parser.add_argument("--diagnostic-run-tag")
    parser.add_argument(
        "--worker-kind",
        choices=(
            "capacity",
            "correctness",
            "arrival",
            "budget-fallback",
        ),
    )
    parser.add_argument(
        "--budget-fallback-reason",
        choices=contract.BUDGET_FALLBACK_REASONS,
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--source-sha256")
    parser.add_argument("--case-id", default="case")
    parser.add_argument(
        "--policy",
        choices=("baseline", "candidate"),
        default="candidate",
    )
    parser.add_argument(
        "--workload",
        choices=contract.PRODUCTION_WORKLOADS,
        default="stable_exact_reuse",
    )
    parser.add_argument("--repetition", type=int, default=0)
    parser.add_argument("--warmup", action="store_true")
    parser.add_argument("--num-kvcache-blocks", type=int, default=0)
    args = parser.parse_args(argv)
    if args.mode in {"download-only", "verify-only"} and not args.run_tag:
        parser.error(f"{args.mode} requires --run-tag")
    if args.mode.startswith("arrival-") and not args.diagnostic_run_tag:
        parser.error(
            f"{args.mode} requires --diagnostic-run-tag"
        )
    if args.mode == "local-contracts" and args.worker_kind is not None:
        missing = [
            name
            for name, value in (
                ("--worker-kind", args.worker_kind),
                ("--output-dir", args.output_dir),
                ("--source-sha256", args.source_sha256),
            )
            if value is None
        ]
        if missing:
            parser.error(
                "local-contracts worker requires " + ", ".join(missing)
            )
        if (
            args.worker_kind == "budget-fallback"
            and args.budget_fallback_reason is None
        ):
            parser.error(
                "budget-fallback worker requires "
                "--budget-fallback-reason"
            )
        if (
            args.worker_kind != "budget-fallback"
            and args.budget_fallback_reason is not None
        ):
            parser.error(
                "--budget-fallback-reason requires "
                "--worker-kind budget-fallback"
            )
    return args


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.mode == "local-contracts" and args.worker_kind is not None:
        return _worker_main(args)
    if args.mode == "local-contracts":
        commands = (
            [sys.executable, "tools/test_model_runner_spec_verify.py"],
            [sys.executable, "tools/test_multi_sequence_cuda_graph_gate.py"],
        )
        for command in commands:
            result = subprocess.run(
                command,
                cwd=ROOT,
                check=False,
            )
            if result.returncode != 0:
                return result.returncode
        return 0
    run_tag = args.run_tag or _default_run_tag(args.mode)
    if args.mode == "verify-only":
        result = _verify_local(OUTPUT_ROOT / run_tag)
        print(json.dumps(result, indent=2, sort_keys=True))
        return (
            0
            if result["classification"]
            in {"GO", "NON_AUTHORITATIVE_SMOKE"}
            else 1
        )
    if args.mode == "download-only":
        _download_remote_tree(
            _remote_run_dir(run_tag),
            OUTPUT_ROOT / run_tag,
        )
        return 0
    try:
        run_dir = _orchestrate(
            args.mode,
            run_tag,
            diagnostic_run_tag=args.diagnostic_run_tag,
        )
    except GpuOccupancyError as exc:
        run_dir = OUTPUT_ROOT / run_tag
        run_dir.mkdir(parents=True, exist_ok=True)
        _write_json(
            run_dir / "incomplete.json",
            {
                "classification": "INCOMPLETE",
                "failure_reason": "unrelated_gpu_occupancy",
                "gpu": int(CUDA_VISIBLE_DEVICES),
                "occupancy": exc.occupancy,
                "stage": exc.stage,
            },
        )
        print(run_dir)
        return 1
    print(run_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
