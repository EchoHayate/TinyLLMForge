"""Source-bound primary orchestrator for staged inference benchmarks."""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import re
import socket
import statistics
import subprocess
import sys
import time


TOOLS_ROOT = Path(__file__).resolve().parent
REPO_ROOT = TOOLS_ROOT.parent
if str(TOOLS_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOLS_ROOT))

import source_audit
import staged_inference_benchmark_contract as contract


OWNED_SOURCE_ROOTS = (
    "tools/arrival_load_driver.py",
    "tools/profile_prefix_cache.py",
    "tools/run_prefix_cache_gate_remote.sh",
    "tools/source_audit.py",
    "tools/staged_inference_benchmark_contract.py",
    "tools/staged_inference_benchmark_gate.py",
    "tools/staged_inference_benchmark_worker.py",
    "tools/test_arrival_load_driver.py",
    "tools/test_chunked_prefill.py",
    "tools/test_profile_prefix_cache.py",
    "tools/test_staged_inference_benchmark_contract.py",
    "tools/test_staged_inference_benchmark_gate.py",
    "tools/test_staged_inference_benchmark_worker.py",
    "tinyvllm",
)
IGNORED_UNTRACKED_PREFIXES = (
    "artifacts",
    "experiments",
)
SERVICE_CLASS_BUCKETS = (
    "short__short",
    "short__long",
    "medium__short",
    "medium__long",
    "long__short",
    "long__long",
)
PRIMARY_ARTIFACTS = (
    "run_manifest.json",
    "resolved_config.json",
    "workload_manifest.jsonl",
    "request_timeline.jsonl",
    "scheduler_trace.jsonl",
    "cache_trace.jsonl",
    "memory_trace.jsonl",
    "case_rows.jsonl",
    "summary.json",
    "report.md",
    "primary_verification_receipt.json",
    "manifest.sha256",
)
PREFIX_PROFILE_POLICY = {
    "mode": "full",
    "shared_prefix_tokens": "256,1024,2048",
    "batch_prefix_tokens": "1024,2048",
    "batch_size": 8,
    "suffix_tokens": 64,
    "repetitions": 7,
    "warmup_repetitions": 2,
    "max_model_len": 4096,
    "max_num_batched_tokens": 8192,
    "max_num_seqs": 8,
    "gpu_memory_utilization": 0.5,
    "enforce_eager": True,
}
PREFIX_ENGINE_LIMITS = {
    field: PREFIX_PROFILE_POLICY[field]
    for field in (
        "max_model_len",
        "max_num_batched_tokens",
        "max_num_seqs",
    )
}
PROMOTION_SELECTION_RULE = {
    "primary_benefit": "larger normalized primary benefit",
    "worst_protected_regression": "smaller ratio wins",
    "peak_cuda_reserved_regression": "smaller ratio wins",
    "exact_tie": "prefix",
}


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_path(path: Path) -> str:
    return _sha256_bytes(Path(path).read_bytes())


def _canonical_sha256(value: object) -> str:
    return contract.canonical_json_sha256(value)


def _atomic_write_text(path: Path, text: str) -> None:
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("x", encoding="utf-8") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def _atomic_write_json(path: Path, value: object) -> None:
    _atomic_write_text(
        path,
        json.dumps(
            value,
            sort_keys=True,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n",
    )


def _atomic_write_jsonl(path: Path, rows: list[dict]) -> None:
    _atomic_write_text(
        path,
        "".join(
            json.dumps(
                row,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            )
            + "\n"
            for row in rows
        ),
    )


def _load_json(path: Path) -> dict:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain an object")
    return value


def _load_jsonl(path: Path) -> list[dict]:
    rows = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.endswith("\n"):
                raise ValueError(
                    f"truncated JSONL line {line_number}: {path}"
                )
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(
                    f"JSONL row {line_number} must be an object: {path}"
                )
            rows.append(value)
    return rows


def _validate_sha256(value, label: str) -> str:
    if (
        not isinstance(value, str)
        or re.fullmatch(r"[0-9a-f]{64}", value) is None
    ):
        raise ValueError(f"invalid {label}")
    return value


def _validate_run_tag(run_tag: str) -> str:
    if (
        not isinstance(run_tag, str)
        or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", run_tag)
        is None
    ):
        raise ValueError("invalid run_tag")
    return run_tag


def _validate_source_evidence(evidence: dict) -> dict:
    if not isinstance(evidence, dict):
        raise ValueError("source_evidence must be an object")
    if evidence.get("schema_version") != 1:
        raise ValueError("unsupported source evidence schema")
    if (
        not isinstance(evidence.get("base_commit"), str)
        or re.fullmatch(r"[0-9a-f]{40}", evidence["base_commit"]) is None
    ):
        raise ValueError("invalid source commit")
    for field in ("local_head", "tracking_head"):
        if (
            not isinstance(evidence.get(field), str)
            or re.fullmatch(r"[0-9a-f]{40}", evidence[field]) is None
        ):
            raise ValueError(f"invalid source {field}")
    if evidence["base_commit"] != evidence["local_head"]:
        raise ValueError("source base commit must equal local HEAD")
    if evidence["base_commit"] != evidence["tracking_head"]:
        raise ValueError("source base commit must equal tracking HEAD")
    if evidence.get("dirty") is not False:
        raise ValueError("owned source paths must be clean")
    _validate_sha256(evidence.get("tree_sha256"), "source tree sha256")
    if evidence.get("owned_roots") != list(OWNED_SOURCE_ROOTS):
        raise ValueError("owned source roots mismatch")
    return json.loads(json.dumps(evidence))


def _validate_environment_evidence(
    evidence: dict,
    *,
    expected_model_tier: str,
    expected_engine_limits: dict,
) -> dict:
    if not isinstance(evidence, dict):
        raise ValueError("environment_evidence must be an object")
    if evidence.get("model_tier") != expected_model_tier:
        raise ValueError("environment model tier mismatch")
    for field in (
        "python_version",
        "torch_version",
        "cuda_version",
        "checkpoint_identifier",
        "model_path",
    ):
        if not isinstance(evidence.get(field), str) or not evidence[field]:
            raise ValueError(f"invalid environment field: {field}")
    _validate_sha256(
        evidence.get("model_config_sha256"),
        "model config sha256",
    )
    limits = evidence.get("engine_limits")
    if not isinstance(limits, dict) or any(
        isinstance(limits.get(field), bool)
        or not isinstance(limits.get(field), int)
        or limits[field] <= 0
        for field in (
            "max_model_len",
            "max_num_batched_tokens",
            "max_num_seqs",
        )
    ):
        raise ValueError("invalid engine limits")
    if limits != expected_engine_limits:
        raise ValueError("environment engine limits mismatch")
    inventory = evidence.get("gpu_inventory")
    selected = evidence.get("selected_gpu_indices")
    if (
        not isinstance(inventory, list)
        or len(inventory) not in (1, 4)
        or not isinstance(selected, list)
        or len(selected) != len(inventory)
    ):
        raise ValueError("invalid selected GPU inventory")
    indices = []
    uuids = []
    for row in inventory:
        if not isinstance(row, dict):
            raise ValueError("invalid GPU inventory row")
        index = row.get("index")
        uuid = row.get("uuid")
        name = row.get("name")
        if (
            isinstance(index, bool)
            or not isinstance(index, int)
            or index < 0
            or not isinstance(uuid, str)
            or not uuid
            or not isinstance(name, str)
            or not name
        ):
            raise ValueError("invalid GPU inventory row")
        indices.append(index)
        uuids.append(uuid)
    if (
        selected != indices
        or len(indices) != len(set(indices))
        or len(uuids) != len(set(uuids))
    ):
        raise ValueError("selected GPU inventory mismatch")
    return json.loads(json.dumps(evidence))


def _git_output(repo_root: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", *arguments],
        cwd=Path(repo_root),
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        raise ValueError(
            f"git {' '.join(arguments)} failed: {result.stderr.strip()}"
        )
    return result.stdout.strip()


def collect_source_evidence(
    *,
    repo_root: Path,
    output_dir: Path,
) -> dict:
    root = Path(repo_root).resolve()
    local_head = _git_output(root, "rev-parse", "HEAD")
    tracking_head = _git_output(
        root,
        "rev-parse",
        "origin/feat/kv-sparse-attention",
    )
    if local_head != tracking_head:
        raise ValueError(
            "local HEAD must equal origin/feat/kv-sparse-attention"
        )
    evidence = source_audit.build_source_evidence(
        root,
        output_dir,
        owned_roots=OWNED_SOURCE_ROOTS,
        ignored_untracked_prefixes=IGNORED_UNTRACKED_PREFIXES,
    )
    if evidence.get("dirty") is not False:
        raise ValueError("owned source paths must be clean")
    evidence["local_head"] = local_head
    evidence["tracking_head"] = tracking_head
    _atomic_write_json(
        Path(output_dir) / "source_evidence.json",
        evidence,
    )
    return evidence


def _prefix_workload(model_tier: str) -> list[dict]:
    return contract.build_prefix_case_matrix(model_tier=model_tier)


def _policy_identity(gate_name: str) -> dict:
    if gate_name == "prefix":
        return dict(PREFIX_PROFILE_POLICY)
    if gate_name == "chunked":
        return {
            "policies": contract.CHUNKED_POLICIES,
            "policy_order": {
                str(repetition): (
                    ["OFF", "FAIR_CHUNKED"]
                    if repetition % 2 == 0
                    else ["FAIR_CHUNKED", "OFF"]
                )
                for repetition in range(5)
            },
        }
    raise ValueError(f"unsupported gate: {gate_name!r}")


def _case_specs(
    *,
    gate_name: str,
    model_tier: str,
    model_path: str,
) -> list[dict]:
    if gate_name == "prefix":
        return [{
            "case_id": f"prefix_full__{model_tier}",
            "gate": "prefix",
            "model_tier": model_tier,
            "profile_args": {
                "model": model_path,
                **PREFIX_PROFILE_POLICY,
            },
        }]
    if gate_name == "chunked":
        return [
            {
                **row,
                "model": model_path,
                "drain_timeout_ns": 180_000_000_000,
            }
            for row in contract.build_chunked_case_matrix(
                model_tier=model_tier
            )
        ]
    raise ValueError(f"unsupported gate: {gate_name!r}")


def _promotion_record(
    *,
    gate_name: str,
    model_tier: str,
    promotion: dict | None,
) -> dict | None:
    if model_tier != "qwen3-8b":
        if promotion is not None:
            raise ValueError("promotion is only valid for qwen3-8b")
        return None
    if not isinstance(promotion, dict):
        raise ValueError("qwen3-8b requires Stage-1 promotion evidence")
    prefix = promotion.get("prefix_summary")
    chunked = promotion.get("chunked_summary")
    if not isinstance(prefix, dict) or not isinstance(chunked, dict):
        raise ValueError("promotion summaries must be objects")
    prefix_sha256 = _canonical_sha256(prefix)
    chunked_sha256 = _canonical_sha256(chunked)
    for name, expected in (
        ("prefix", prefix_sha256),
        ("chunked", chunked_sha256),
    ):
        receipt = promotion.get(f"{name}_verification_receipt")
        if (
            not isinstance(receipt, dict)
            or receipt.get("status") != "PASS"
            or receipt.get("primary_summary_sha256") != expected
            or receipt.get("controller_summary_sha256") != expected
        ):
            raise ValueError(
                f"{name} independent verification receipt is invalid"
            )
    selection = contract.select_stage2_winner(prefix, chunked)
    selected = selection.get("winner")
    if selected is None:
        raise ValueError("neither Stage-1 summary is GO")
    if promotion.get("winner") != selected or gate_name != selected:
        raise ValueError("selected winner differs from frozen selection")
    return {
        "winner": selected,
        "prefix_summary_sha256": prefix_sha256,
        "chunked_summary_sha256": chunked_sha256,
        "selection_rule": dict(PROMOTION_SELECTION_RULE),
    }


def initialize_run(
    *,
    run_dir: Path,
    run_tag: str,
    gate_name: str,
    model_tier: str,
    source_evidence: dict,
    environment_evidence: dict,
    promotion: dict | None = None,
) -> dict:
    destination = Path(run_dir)
    if destination.exists():
        raise ValueError(f"run directory already exists: {destination}")
    tag = _validate_run_tag(run_tag)
    if gate_name not in {"prefix", "chunked"}:
        raise ValueError(f"unsupported gate: {gate_name!r}")
    if model_tier not in {"qwen3-0.6b", "qwen3-8b"}:
        raise ValueError(f"unsupported model tier: {model_tier!r}")
    source = _validate_source_evidence(source_evidence)
    environment = _validate_environment_evidence(
        environment_evidence,
        expected_model_tier=model_tier,
        expected_engine_limits=(
            PREFIX_ENGINE_LIMITS
            if gate_name == "prefix"
            else contract.CHUNKED_ENGINE_CONFIG
        ),
    )
    promotion_record = _promotion_record(
        gate_name=gate_name,
        model_tier=model_tier,
        promotion=promotion,
    )
    workload = (
        _prefix_workload(model_tier)
        if gate_name == "prefix"
        else contract.build_chunked_workload()
    )
    policy = _policy_identity(gate_name)
    cases = _case_specs(
        gate_name=gate_name,
        model_tier=model_tier,
        model_path=environment["model_path"],
    )
    destination.mkdir(parents=True)
    case_specs_dir = destination / "case_specs"
    case_specs_dir.mkdir()
    for case in cases:
        _atomic_write_json(
            case_specs_dir / f"{case['case_id']}.json",
            case,
        )
    _atomic_write_jsonl(
        destination / "workload_manifest.jsonl",
        workload,
    )
    resolved_config = {
        "gate": gate_name,
        "model_tier": model_tier,
        "model_path": environment["model_path"],
        "policy": policy,
        "cases": cases,
        "environment": environment,
    }
    _atomic_write_json(
        destination / "resolved_config.json",
        resolved_config,
    )
    manifest = {
        "schema_version": 1,
        "status": "INITIALIZED",
        "run_tag": tag,
        "gate": gate_name,
        "model_tier": model_tier,
        "source_commit": source["base_commit"],
        "source_tree_sha256": source["tree_sha256"],
        "environment_sha256": _canonical_sha256(environment),
        "workload_sha256": _canonical_sha256(workload),
        "policy_sha256": _canonical_sha256(policy),
        "source_evidence": source,
        "environment_evidence": environment,
        "case_order": [case["case_id"] for case in cases],
        "case_specs": {
            case["case_id"]: f"case_specs/{case['case_id']}.json"
            for case in cases
        },
    }
    if promotion_record is not None:
        manifest["promotion"] = promotion_record
    _atomic_write_json(destination / "run_manifest.json", manifest)
    return manifest


def _validate_initialized_run_identity(
    root: Path,
    manifest: dict,
) -> None:
    if manifest.get("schema_version") != 1:
        raise ValueError("unsupported run manifest schema")
    gate_name = manifest.get("gate")
    model_tier = manifest.get("model_tier")
    if gate_name not in {"prefix", "chunked"}:
        raise ValueError("unsupported manifest gate")
    if model_tier not in {"qwen3-0.6b", "qwen3-8b"}:
        raise ValueError("unsupported manifest model tier")
    source = _validate_source_evidence(manifest.get("source_evidence"))
    if (
        manifest.get("source_commit") != source["base_commit"]
        or manifest.get("source_tree_sha256") != source["tree_sha256"]
    ):
        raise ValueError("source identity mismatch")
    environment = _validate_environment_evidence(
        manifest.get("environment_evidence"),
        expected_model_tier=model_tier,
        expected_engine_limits=(
            PREFIX_ENGINE_LIMITS
            if gate_name == "prefix"
            else contract.CHUNKED_ENGINE_CONFIG
        ),
    )
    if (
        manifest.get("environment_sha256")
        != _canonical_sha256(environment)
    ):
        raise ValueError("environment identity mismatch")
    workload = _load_jsonl(root / "workload_manifest.jsonl")
    expected_workload = (
        _prefix_workload(model_tier)
        if gate_name == "prefix"
        else contract.build_chunked_workload()
    )
    if (
        workload != expected_workload
        or manifest.get("workload_sha256")
        != _canonical_sha256(workload)
    ):
        raise ValueError("workload identity mismatch")
    policy = _policy_identity(gate_name)
    if manifest.get("policy_sha256") != _canonical_sha256(policy):
        raise ValueError("policy identity mismatch")
    expected_cases = _case_specs(
        gate_name=gate_name,
        model_tier=model_tier,
        model_path=environment["model_path"],
    )
    expected_order = [case["case_id"] for case in expected_cases]
    expected_paths = {
        case["case_id"]: f"case_specs/{case['case_id']}.json"
        for case in expected_cases
    }
    if (
        manifest.get("case_order") != expected_order
        or manifest.get("case_specs") != expected_paths
    ):
        raise ValueError("case matrix identity mismatch")
    loaded_cases = [
        _load_json(root / expected_paths[case_id])
        for case_id in expected_order
    ]
    if loaded_cases != expected_cases:
        raise ValueError("case specification identity mismatch")
    resolved = _load_json(root / "resolved_config.json")
    if resolved != {
        "gate": gate_name,
        "model_tier": model_tier,
        "model_path": environment["model_path"],
        "policy": policy,
        "cases": expected_cases,
        "environment": environment,
    }:
        raise ValueError("resolved configuration identity mismatch")


def _allocate_port() -> int:
    handle = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        handle.bind(("127.0.0.1", 0))
        return int(handle.getsockname()[1])
    finally:
        handle.close()


def _used_case_ports(run_dir: Path) -> set[int]:
    used = set()
    cases_root = Path(run_dir) / "cases"
    if not cases_root.is_dir():
        return used
    for receipt_path in cases_root.glob("*/process.json"):
        receipt = _load_json(receipt_path)
        for field in ("master_port", "distributed_port"):
            value = receipt.get(field)
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value <= 0
            ):
                raise ValueError(
                    f"invalid prior case port in {receipt_path}"
                )
            used.add(value)
    return used


def _allocate_unique_case_ports(
    run_dir: Path,
    port_allocator,
) -> tuple[int, int]:
    used = _used_case_ports(run_dir)
    selected = []
    for _ in range(256):
        value = port_allocator()
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or value <= 0
        ):
            raise ValueError(
                "case ports must be unique positive integers"
            )
        if value in used or value in selected:
            continue
        selected.append(value)
        if len(selected) == 2:
            return selected[0], selected[1]
    raise ValueError("unable to allocate unique case ports")


def launch_case(
    run_dir: Path,
    case_id: str,
    *,
    python_bin: str = sys.executable,
    process_runner=subprocess.run,
    port_allocator=_allocate_port,
) -> dict:
    root = Path(run_dir)
    manifest = _load_json(root / "run_manifest.json")
    if manifest.get("status") != "INITIALIZED":
        raise ValueError("cases may only launch for initialized runs")
    _validate_initialized_run_identity(root, manifest)
    if case_id not in manifest.get("case_order", []):
        raise ValueError(f"unknown case_id: {case_id!r}")
    case_dir = root / "cases" / case_id
    if case_dir.exists():
        raise ValueError(f"case directory already exists: {case_id}")
    master_port, distributed_port = _allocate_unique_case_ports(
        root,
        port_allocator,
    )
    case_dir.mkdir(parents=True)
    output_dir = case_dir / "output"
    spec_path = root / manifest["case_specs"][case_id]
    command = [
        python_bin,
        str(TOOLS_ROOT / "staged_inference_benchmark_worker.py"),
        "--spec",
        str(spec_path),
    ]
    if manifest["gate"] == "chunked":
        command.extend((
            "--workload-manifest",
            str(root / "workload_manifest.jsonl"),
        ))
    command.extend(("--output-dir", str(output_dir)))
    environment = os.environ.copy()
    environment["MASTER_PORT"] = str(master_port)
    environment["TINYLLMFORGE_DIST_PORT"] = str(distributed_port)
    started_ns = time.time_ns()
    result = process_runner(
        command,
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
        env=environment,
    )
    finished_ns = time.time_ns()
    stdout = result.stdout if isinstance(result.stdout, str) else ""
    stderr = result.stderr if isinstance(result.stderr, str) else ""
    returncode = result.returncode
    if isinstance(returncode, bool) or not isinstance(returncode, int):
        raise ValueError("worker returncode is invalid")
    receipt = {
        "schema_version": 1,
        "case_id": case_id,
        "command": command,
        "cwd": str(REPO_ROOT),
        "master_port": master_port,
        "distributed_port": distributed_port,
        "started_unix_ns": started_ns,
        "finished_unix_ns": finished_ns,
        "duration_ns": finished_ns - started_ns,
        "returncode": returncode,
    }
    _atomic_write_text(case_dir / "stdout.log", stdout)
    _atomic_write_text(case_dir / "stderr.log", stderr)
    _atomic_write_text(case_dir / "exitcode", f"{returncode}\n")
    _atomic_write_json(case_dir / "process.json", receipt)
    return receipt


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        raise ValueError("percentile requires at least one value")
    ordered = sorted(float(value) for value in values)
    index = max(
        0,
        min(
            len(ordered) - 1,
            math.ceil(percentile * len(ordered)) - 1,
        ),
    )
    return ordered[index]


def _case_identity(manifest: dict, case_id: str) -> dict:
    return _load_json(
        Path(manifest["_run_dir"]) / manifest["case_specs"][case_id]
    )


def _prefix_raw_identity(row: dict) -> tuple[str, str, int]:
    if not isinstance(row, dict) or row.get("schema_version") != 2:
        raise ValueError("prefix raw artifacts have invalid schema")
    shape = row.get("shape")
    state = row.get("state")
    repetition = row.get("repetition")
    if (
        shape not in {
            "single-256",
            "single-1024",
            "single-2048",
            "batch8-1024",
            "batch8-2048",
        }
        or state not in {"cold", "warm", "cache_cleared"}
        or isinstance(repetition, bool)
        or not isinstance(repetition, int)
        or repetition not in range(7)
        or row.get("warmup") is not False
        or row.get("case_id")
        != f"{shape}__{state}__r{repetition}"
    ):
        raise ValueError("prefix raw artifacts have invalid identity")
    return shape, state, repetition


def _prefix_state_summary(
    rows: list[dict],
    *,
    cache_by_identity: dict[tuple[str, str, int], dict],
    memory_by_identity: dict[tuple[str, str, int], dict],
) -> dict:
    elapsed_ms = []
    cached_tokens = []
    query_tokens = []
    model_batches = []
    logit_max_abs = []
    logit_mean_abs = []
    logit_argmax_match = []
    retained_blocks = []
    retained_bytes = []
    cache_clear_host_ns = []
    peak_cuda_reserved_bytes = []
    exact_outputs = []
    for row in rows:
        identity = _prefix_raw_identity(row)
        cache = cache_by_identity[identity]
        memory = memory_by_identity[identity]
        for field in (
            "cached_prompt_tokens",
            "executed_query_tokens",
            "retained_logical_kv_bytes",
        ):
            if row.get(field) != cache.get(field):
                raise ValueError(
                    f"Prefix performance/cache evidence mismatch: {field}"
                )
        if (
            row.get("retained_logical_kv_bytes")
            != memory.get("retained_logical_kv_bytes")
            or row.get("cuda_peak_reserved_bytes")
            != memory.get("cuda_peak_reserved_bytes")
        ):
            raise ValueError("Prefix performance/memory evidence mismatch")
        logit = row.get("logit")
        if not isinstance(logit, dict):
            raise ValueError("Prefix performance row lacks logit evidence")
        ttft_ns = row.get("ttft_ns")
        if (
            isinstance(ttft_ns, bool)
            or not isinstance(ttft_ns, int)
            or ttft_ns <= 0
        ):
            raise ValueError("Prefix performance row has invalid TTFT")
        elapsed_ms.append(ttft_ns / 1_000_000.0)
        cached_tokens.append(int(row["cached_prompt_tokens"]))
        query_tokens.append(int(row["executed_query_tokens"]))
        model_batches.append(int(row["model_batches"]))
        logit_max_abs.append(float(logit["max_abs"]))
        logit_mean_abs.append(float(logit["mean_abs"]))
        logit_argmax_match.append(logit.get("argmax_match") is True)
        retained_blocks.append(int(cache["retained_reusable_blocks"]))
        retained_bytes.append(int(cache["retained_logical_kv_bytes"]))
        cache_clear_host_ns.append(int(cache["cache_clear_host_ns"]))
        peak_cuda_reserved_bytes.append(
            int(memory["cuda_peak_reserved_bytes"])
        )
        exact_outputs.append(row.get("correct") is True)
    return {
        "samples": len(rows),
        "median_elapsed_ms": statistics.median(elapsed_ms),
        "p95_elapsed_ms": _percentile(elapsed_ms, 0.95),
        "median_cached_prompt_tokens": statistics.median(cached_tokens),
        "median_executed_query_tokens": statistics.median(query_tokens),
        "median_model_batches": statistics.median(model_batches),
        "peak_cuda_reserved_bytes": max(peak_cuda_reserved_bytes),
        "exact_outputs": all(exact_outputs),
        "logit_argmax_match": all(logit_argmax_match),
        "logit_max_abs": max(logit_max_abs),
        "logit_mean_abs": max(logit_mean_abs),
        "_retained_reusable_blocks": max(retained_blocks),
        "_retained_logical_kv_bytes": max(retained_bytes),
        "_median_cache_clear_host_ms": (
            statistics.median(cache_clear_host_ns) / 1_000_000.0
        ),
    }


def _rebuild_prefix_bundle(
    merged: dict[str, list[dict]],
    *,
    artifact_complete: bool,
) -> dict:
    performance = merged["scheduler_trace.jsonl"]
    cache_by_identity = {
        _prefix_raw_identity(row): row
        for row in merged["cache_trace.jsonl"]
    }
    memory_by_identity = {
        _prefix_raw_identity(row): row
        for row in merged["memory_trace.jsonl"]
    }
    families = {"single": {}, "batch": {}}
    for shape in (
        "single-256",
        "single-1024",
        "single-2048",
        "batch8-1024",
        "batch8-2048",
    ):
        family = "single" if shape.startswith("single-") else "batch"
        prefix_tokens = int(shape.rsplit("-", 1)[1])
        batch_size = 1 if family == "single" else 8
        shape_rows = [row for row in performance if row["shape"] == shape]
        if any(
            row.get("shared_prefix_tokens") != prefix_tokens
            or row.get("suffix_tokens") != 64
            or row.get("batch_size") != batch_size
            for row in shape_rows
        ):
            raise ValueError("Prefix raw shape evidence mismatch")
        states = {}
        for state in ("cold", "warm", "cache_cleared"):
            state_rows = [
                row for row in shape_rows if row["state"] == state
            ]
            states[state] = _prefix_state_summary(
                state_rows,
                cache_by_identity=cache_by_identity,
                memory_by_identity=memory_by_identity,
            )
        retained_blocks = max(
            state.pop("_retained_reusable_blocks")
            for state in states.values()
        )
        retained_bytes = max(
            state.pop("_retained_logical_kv_bytes")
            for state in states.values()
        )
        clear_host_ms = max(
            state.pop("_median_cache_clear_host_ms")
            for state in states.values()
        )
        families[family][str(prefix_tokens)] = {
            "prefix_tokens": prefix_tokens,
            "suffix_tokens": 64,
            "batch_size": batch_size,
            "expected_reusable_tokens": prefix_tokens * batch_size,
            **states,
            "retained_reusable_blocks": retained_blocks,
            "retained_logical_kv_bytes": retained_bytes,
            "median_cache_clear_host_ms": clear_host_ms,
        }
    return {
        "artifact_complete": artifact_complete,
        "single": families["single"],
        "batch": families["batch"],
    }


def _validate_prefix_raw_artifacts(
    merged: dict[str, list[dict]],
) -> dict:
    correctness = merged["request_timeline.jsonl"]
    expected_correctness_cases = {
        "cpu_collision_and_lifecycle_preflight",
        "repeat_255",
        "repeat_256",
        "repeat_257",
        "repeat_512",
        "repeat_513",
        "same_batch_p_q_p_first",
        "same_batch_p_q_p_middle",
        "same_batch_p_q_p",
        "shared_prefix_different_suffix",
        "cache_cleared",
    }
    observed_correctness_cases = [
        row.get("case")
        for row in correctness
        if isinstance(row, dict)
    ]
    if (
        len(observed_correctness_cases) != len(expected_correctness_cases)
        or set(observed_correctness_cases) != expected_correctness_cases
        or any(row.get("correct") is not True for row in correctness)
    ):
        raise ValueError(
            "prefix raw artifacts lack complete correctness evidence"
        )
    expected = {
        (shape, state, repetition)
        for shape in (
            "single-256",
            "single-1024",
            "single-2048",
            "batch8-1024",
            "batch8-2048",
        )
        for state in ("cold", "warm", "cache_cleared")
        for repetition in range(7)
    }
    identities = {}
    for filename in (
        "scheduler_trace.jsonl",
        "cache_trace.jsonl",
        "memory_trace.jsonl",
    ):
        rows = merged[filename]
        values = [_prefix_raw_identity(row) for row in rows]
        if len(values) != len(expected) or set(values) != expected:
            raise ValueError(
                f"prefix raw artifacts are incomplete: {filename}"
            )
        identities[filename] = values
    if not (
        identities["scheduler_trace.jsonl"]
        == identities["cache_trace.jsonl"]
        == identities["memory_trace.jsonl"]
    ):
        raise ValueError("prefix raw artifacts are not aligned")
    return _rebuild_prefix_bundle(
        merged,
        artifact_complete=True,
    )


def _prefix_finalization(root: Path, manifest: dict):
    case_id = manifest["case_order"][0]
    case_dir = root / "cases" / case_id
    process = _load_json(case_dir / "process.json")
    if process.get("returncode") != 0:
        raise ValueError("prefix worker did not exit successfully")
    output = case_dir / "output"
    worker_summary = _load_json(output / "summary.json")
    raw = worker_summary.get("staged_contract_bundle")
    if not isinstance(raw, dict):
        raise ValueError("prefix worker summary lacks contract bundle")
    mapping = {
        "request_timeline.jsonl": "prefix_correctness_rows.jsonl",
        "scheduler_trace.jsonl": "prefix_performance_rows.jsonl",
        "cache_trace.jsonl": "prefix_cache_rows.jsonl",
        "memory_trace.jsonl": "prefix_memory_rows.jsonl",
    }
    merged = {}
    for destination, source in mapping.items():
        source_path = output / source
        merged[destination] = (
            _load_jsonl(source_path) if source_path.is_file() else []
        )
    rebuilt_raw = _validate_prefix_raw_artifacts(merged)
    if raw != rebuilt_raw:
        raise ValueError(
            "worker Prefix summary does not match raw Prefix evidence"
        )
    summary = contract.classify_prefix_bundle(rebuilt_raw)
    case_rows = merged["scheduler_trace.jsonl"]
    return summary, merged, case_rows


def _measured_lifecycle(rows: list[dict]) -> list[dict]:
    return [row for row in rows if row.get("warmup") is False]


def _validate_chunked_timeline(
    timeline: list[dict],
    workload: list[dict],
) -> None:
    expected_by_id = {
        row.get("request_id"): row
        for row in workload
        if isinstance(row, dict)
        and isinstance(row.get("request_id"), str)
    }
    observed_ids = [
        row.get("request_id")
        for row in timeline
        if isinstance(row, dict)
    ]
    if (
        len(expected_by_id) != len(workload)
        or len(timeline) != len(workload)
        or len(observed_ids) != len(timeline)
        or len(set(observed_ids)) != len(observed_ids)
        or set(observed_ids) != set(expected_by_id)
    ):
        raise ValueError("chunked request identity mismatch")
    for row in timeline:
        expected = expected_by_id[row["request_id"]]
        if (
            row.get("warmup") is not expected.get("warmup")
            or row.get("phase") != expected.get("phase")
            or row.get("prompt_token_count")
            != expected.get("prompt_tokens")
            or row.get("requested_output_tokens")
            != expected.get("requested_output_tokens")
            or row.get("service_time_bucket")
            != expected.get("service_time_bucket")
            or row.get("starvation_deadline_ns")
            != expected.get("starvation_deadline_ns")
        ):
            raise ValueError("chunked request identity mismatch")


def _chunked_case_metrics(
    *,
    case: dict,
    timeline: list[dict],
    workload: list[dict],
    memory: list[dict],
    case_result: dict,
) -> tuple[dict, dict[str, list[int]]]:
    _validate_chunked_timeline(timeline, workload)
    measured = _measured_lifecycle(timeline)
    by_request = {
        row["request_id"]: list(row.get("output_token_ids", []))
        for row in measured
        if isinstance(row.get("request_id"), str)
    }
    complete_rows = [
        row for row in measured
        if row.get("first_scheduled_ns") is not None
        and row.get("first_token_ns") is not None
        and row.get("completion_ns") is not None
        and row.get("finish_reason") == "length"
        and len(row.get("output_token_ids", []))
        == row.get("requested_output_tokens")
    ]
    ttft_short = [
        row["first_token_ns"] - row["scheduled_arrival_ns"]
        for row in complete_rows
        if row.get("service_time_bucket", "").startswith("short__")
    ]
    short_itl = []
    all_itl = []
    for row in complete_rows:
        timestamps = row.get("token_timestamps_ns", [])
        gaps = [
            right - left
            for left, right in zip(timestamps, timestamps[1:])
        ]
        all_itl.extend(gaps)
        if row.get("service_time_bucket", "").startswith("short__"):
            short_itl.extend(gaps)
    completions = {
        bucket: [
            row["completion_ns"] - row["scheduled_arrival_ns"]
            for row in complete_rows
            if row.get("service_time_bucket") == bucket
        ]
        for bucket in SERVICE_CLASS_BUCKETS
    }
    long_completion = [
        row["completion_ns"] - row["scheduled_arrival_ns"]
        for row in complete_rows
        if row.get("service_time_bucket", "").startswith("long__")
    ]
    if complete_rows:
        start_ns = min(
            row["scheduled_arrival_ns"] for row in complete_rows
        )
        finish_ns = max(row["completion_ns"] for row in complete_rows)
        duration_s = (finish_ns - start_ns) / 1_000_000_000
    else:
        duration_s = 0.0
    result_error = case_result.get("error_type")
    expected_measured = 96
    unfinished = expected_measured - len(complete_rows)
    metrics = {
        "case_id": case["case_id"],
        "policy": case["policy"],
        "repetition": case["repetition"],
        "short_p99_ttft_ns": _percentile(ttft_short, 0.99),
        "short_p99_itl_ns": _percentile(short_itl, 0.99),
        "maximum_decode_gap_ns": max(all_itl),
        "service_class_p95_completion_ns": {
            bucket: _percentile(values, 0.95)
            for bucket, values in completions.items()
        },
        "long_p95_completion_ns": _percentile(
            long_completion,
            0.95,
        ),
        "request_throughput_rps": (
            len(complete_rows) / duration_s if duration_s > 0 else 0.0
        ),
        "output_token_throughput_tps": (
            sum(
                len(row["output_token_ids"])
                for row in complete_rows
            )
            / duration_s
            if duration_s > 0
            else 0.0
        ),
        "peak_cuda_reserved_bytes": max(
            float(row.get("cuda_peak_reserved_bytes", 0))
            for row in memory
        ),
        "exact_outputs": True,
        "complete_lifecycle": (
            case_result.get("status") == "PASS"
            and len(complete_rows) == expected_measured
        ),
        "dropped_requests": max(
            0,
            expected_measured - len(measured),
        ),
        "rejected_requests": int(result_error == "admission_error"),
        "truncated_requests": sum(
            row.get("completion_ns") is not None
            and len(row.get("output_token_ids", []))
            < row.get("requested_output_tokens", 0)
            for row in measured
        ),
        "unfinished_requests": max(0, unfinished),
        "starved_requests": int(result_error == "starved_request"),
    }
    return metrics, by_request


def _chunked_finalization(root: Path, manifest: dict):
    merged = {
        "request_timeline.jsonl": [],
        "scheduler_trace.jsonl": [],
        "cache_trace.jsonl": [],
        "memory_trace.jsonl": [],
    }
    metrics_by_repetition: dict[int, dict[str, dict]] = {}
    outputs_by_repetition: dict[int, dict[str, dict[str, list[int]]]] = {}
    artifact_complete = True
    case_rows = []
    workload = _load_jsonl(root / "workload_manifest.jsonl")
    for case_id in manifest["case_order"]:
        case = _load_json(root / manifest["case_specs"][case_id])
        case_dir = root / "cases" / case_id
        process = _load_json(case_dir / "process.json")
        output = case_dir / "output"
        timeline = _load_jsonl(output / "request_timeline.jsonl")
        scheduler = _load_jsonl(output / "scheduler_trace.jsonl")
        memory = _load_jsonl(output / "memory_trace.jsonl")
        case_result = _load_json(output / "case_result.json")
        if process.get("returncode") != 0:
            artifact_complete = False
        identity = {
            "case_id": case_id,
            "policy": case["policy"],
            "repetition": case["repetition"],
        }
        for row in timeline:
            merged["request_timeline.jsonl"].append(
                {**identity, **row}
            )
        for row in scheduler:
            merged["scheduler_trace.jsonl"].append(
                {**identity, **row}
            )
        for row in memory:
            merged["memory_trace.jsonl"].append(
                {**identity, **row}
            )
        try:
            metrics, outputs = _chunked_case_metrics(
                case=case,
                timeline=timeline,
                workload=workload,
                memory=memory,
                case_result=case_result,
            )
        except (KeyError, TypeError, ValueError):
            artifact_complete = False
            raise
        case_rows.append(metrics)
        repetition = case["repetition"]
        metrics_by_repetition.setdefault(repetition, {})[
            case["policy"]
        ] = metrics
        outputs_by_repetition.setdefault(repetition, {})[
            case["policy"]
        ] = outputs
    repetitions = []
    for repetition in range(5):
        policies = metrics_by_repetition.get(repetition, {})
        outputs = outputs_by_repetition.get(repetition, {})
        if set(policies) != {"OFF", "FAIR_CHUNKED"}:
            artifact_complete = False
        else:
            exact = outputs["OFF"] == outputs["FAIR_CHUNKED"]
            policies["OFF"]["exact_outputs"] = exact
            policies["FAIR_CHUNKED"]["exact_outputs"] = exact
        repetitions.append({
            "repetition": repetition,
            **policies,
        })
    raw = {
        "artifact_complete": artifact_complete,
        "repetitions": repetitions,
    }
    return contract.classify_chunked_bundle(raw), merged, case_rows


def _render_report(manifest: dict, summary: dict) -> str:
    benefit = summary.get("benefit", {})
    cost = summary.get("cost", {})
    return "\n".join((
        f"# {manifest['run_tag']}",
        "",
        f"- Gate: `{manifest['gate']}`",
        f"- Model tier: `{manifest['model_tier']}`",
        f"- Classification: `{summary.get('classification')}`",
        "",
        "| Benefit | Cost |",
        "| --- | --- |",
        (
            f"| `{json.dumps(benefit, sort_keys=True)}` "
            f"| `{json.dumps(cost, sort_keys=True)}` |"
        ),
        "",
    ))


def finalize_run(run_dir: Path) -> dict:
    root = Path(run_dir)
    manifest = _load_json(root / "run_manifest.json")
    if manifest.get("status") == "FINALIZED":
        raise ValueError("run is already finalized")
    if manifest.get("status") != "INITIALIZED":
        raise ValueError("run must be initialized before finalization")
    _validate_initialized_run_identity(root, manifest)
    manifest["_run_dir"] = str(root)
    if manifest["gate"] == "prefix":
        summary, merged, case_rows = _prefix_finalization(
            root,
            manifest,
        )
    elif manifest["gate"] == "chunked":
        summary, merged, case_rows = _chunked_finalization(
            root,
            manifest,
        )
    else:
        raise ValueError("unsupported manifest gate")
    manifest.pop("_run_dir", None)
    manifest["status"] = "FINALIZED"
    manifest["classification"] = summary["classification"]
    _atomic_write_json(root / "run_manifest.json", manifest)
    for filename, rows in merged.items():
        _atomic_write_jsonl(root / filename, rows)
    _atomic_write_jsonl(root / "case_rows.jsonl", case_rows)
    _atomic_write_json(root / "summary.json", summary)
    _atomic_write_text(root / "report.md", _render_report(manifest, summary))
    receipt = {
        "schema_version": 1,
        "status": "PASS",
        "classification": summary["classification"],
        "case_count": len(manifest["case_order"]),
        "source_tree_sha256": manifest["source_tree_sha256"],
        "environment_sha256": manifest["environment_sha256"],
        "workload_sha256": manifest["workload_sha256"],
        "policy_sha256": manifest["policy_sha256"],
    }
    _atomic_write_json(
        root / "primary_verification_receipt.json",
        receipt,
    )
    _atomic_write_text(
        root / "manifest.sha256",
        _sha256_path(root / "run_manifest.json") + "\n",
    )
    missing = [
        filename
        for filename in PRIMARY_ARTIFACTS
        if not (root / filename).is_file()
    ]
    if missing:
        raise ValueError(
            "cannot finalize missing artifacts: " + ", ".join(missing)
        )
    hashes = {}
    for path in sorted(root.rglob("*")):
        if (
            not path.is_file()
            or path.name == "artifact_hashes.json"
            or path.name.endswith(".tmp")
        ):
            continue
        relative = path.relative_to(root).as_posix()
        hashes[relative] = _sha256_path(path)
    _atomic_write_json(root / "artifact_hashes.json", hashes)
    return summary
