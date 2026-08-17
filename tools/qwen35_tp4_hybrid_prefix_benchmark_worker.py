from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time

import qwen35_tp4_hybrid_prefix_benchmark_contract as contract
import qwen35_tp4_decode_internal_profile as decode_profile_contract


MODEL_FINGERPRINT = contract.MODEL_MANIFEST_SHA256
HYBRID_PREFIX_MAX_ENTRIES = 16
HYBRID_PREFIX_MAX_BYTES = 2 * 1024**3
HYBRID_PREFIX_TIMEOUT_S = 120.0
SCHEDULER_VISIBLE_KV_BLOCKS = 64
KV_BLOCK_SIZE = 256
MAX_MODEL_LEN = 4352
MAX_NUM_BATCHED_TOKENS = 17408
MAX_NUM_SEQS = 8
COMPLETION_MARKER = "QWEN35_TP4_BENCHMARK_WORKER_COMPLETE"


def _require_policy(policy):
    if policy not in contract.POLICIES:
        raise ValueError(f"unsupported policy: {policy}")
    return policy


def _require_workload(workload):
    if workload not in contract.WORKLOADS:
        raise ValueError(f"unsupported workload: {workload}")
    return workload


def _require_sha256(value, label):
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(
            character not in "0123456789abcdef"
            for character in value
        )
    ):
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return value


def _non_empty_path(value):
    if not isinstance(value, str) or not value.strip():
        raise argparse.ArgumentTypeError(
            "capture directory must be a non-empty path"
        )
    return Path(value)


def build_engine_configuration(
    policy,
    case,
    *,
    workload_payload=None,
    recurrent_calibration_capture_dir=None,
    capture_identity_fields=None,
    expected_capture_identity_fields=None,
):
    policy = _require_policy(policy)
    case = _canonical_case(case)
    workload = _require_workload(case.workload)
    workload_payload = (
        copy.deepcopy(workload_payload)
        if workload_payload is not None
        else contract.workload_payload(workload)
    )
    if (
        not isinstance(workload_payload, dict)
        or not isinstance(workload_payload.get("spec"), dict)
    ):
        raise ValueError("benchmark workload payload mismatch")
    workload_spec = workload_payload["spec"]
    common_engine = {
        "tensor_parallel_size": contract.WORLD_SIZE,
        "num_kvcache_blocks": SCHEDULER_VISIBLE_KV_BLOCKS,
        "kvcache_block_size": KV_BLOCK_SIZE,
        "enforce_eager": True,
        "max_model_len": MAX_MODEL_LEN,
        "max_num_batched_tokens": MAX_NUM_BATCHED_TOKENS,
        "max_num_seqs": MAX_NUM_SEQS,
    }
    configuration = {
        "schema_version": contract.SCHEMA_VERSION,
        "policy": policy,
        "tensor_parallel_size": contract.WORLD_SIZE,
        "num_kvcache_blocks": SCHEDULER_VISIBLE_KV_BLOCKS,
        "kvcache_block_size": KV_BLOCK_SIZE,
        "engine": common_engine,
        "sampling": {
            "temperature": 0.0,
            "ignore_eos": True,
            "max_tokens": workload_spec["generated_tokens"],
        },
        "workload": {
            "name": workload,
            **workload_spec,
        },
    }
    if policy == "recompute":
        configuration["hybrid_prefix"] = {
            "enabled": False,
            "representation": "none",
        }
    else:
        configuration["hybrid_prefix"] = {
            "enabled": True,
            "representation": "exact_full_fidelity",
            "max_entries": HYBRID_PREFIX_MAX_ENTRIES,
            "max_bytes": HYBRID_PREFIX_MAX_BYTES,
            "timeout_s": HYBRID_PREFIX_TIMEOUT_S,
        }
    if recurrent_calibration_capture_dir is not None:
        if (
            policy != "exact_restore"
            or case.phase != "correctness"
            or case.repetition != 0
        ):
            raise ValueError(
                "recurrent capture requires exact_restore correctness r0"
            )
        capture_root = Path(recurrent_calibration_capture_dir)
        if capture_root.exists() and not capture_root.is_dir():
            raise ValueError(
                "recurrent capture root must be a directory"
            )
        expected_fields = {
            "model_manifest_sha256",
            "source_tree_sha256",
            "workload_manifest_sha256",
            "world_size",
            "workload_ids",
        }
        if (
            not isinstance(capture_identity_fields, dict)
            or set(capture_identity_fields) != expected_fields
            or not isinstance(expected_capture_identity_fields, dict)
            or set(expected_capture_identity_fields) != expected_fields
        ):
            raise ValueError(
                "recurrent capture identity fields mismatch"
            )
        if capture_identity_fields != expected_capture_identity_fields:
            raise ValueError(
                "recurrent capture identity does not match authority"
            )
        for name in (
            "model_manifest_sha256",
            "source_tree_sha256",
            "workload_manifest_sha256",
        ):
            _require_sha256(
                capture_identity_fields[name],
                f"recurrent capture {name}",
            )
        if (
            type(capture_identity_fields["world_size"]) is not int
            or type(
                expected_capture_identity_fields["world_size"]
            ) is not int
            or capture_identity_fields["world_size"]
            != contract.WORLD_SIZE
            or capture_identity_fields["workload_ids"]
            != list(contract.WORKLOADS)
        ):
            raise ValueError(
                "recurrent capture identity inventory mismatch"
            )
        configuration["recurrent_calibration_capture"] = {
            "capture_root": str(capture_root.resolve()),
            **capture_identity_fields,
        }
    return configuration


def _atomic_write(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="wb",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".partial",
        delete=False,
    ) as handle:
        temporary_path = Path(handle.name)
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    temporary_path.replace(path)


def _atomic_write_json(path, value):
    _atomic_write(
        path,
        json.dumps(
            value,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8") + b"\n",
    )


def _atomic_write_jsonl(path, rows):
    _atomic_write(
        path,
        b"".join(
            contract.canonical_json_bytes(row) + b"\n"
            for row in rows
        ),
    )


def _sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_runtime_artifacts(
    *,
    model_dir,
    model_manifest_path,
    expected_model_manifest_sha256,
    correctness_prerequisites_path,
    expected_correctness_prerequisites_sha256,
    workload_manifest_path=None,
    expected_workload_manifest_sha256=None,
):
    model_dir = Path(model_dir)
    model_manifest_path = Path(model_manifest_path)
    correctness_prerequisites_path = Path(
        correctness_prerequisites_path
    )
    expected_model_manifest_sha256 = _require_sha256(
        expected_model_manifest_sha256,
        "model manifest",
    )
    if (
        expected_model_manifest_sha256
        != contract.MODEL_MANIFEST_SHA256
    ):
        raise ValueError("canonical model manifest SHA mismatch")
    expected_correctness_prerequisites_sha256 = _require_sha256(
        expected_correctness_prerequisites_sha256,
        "correctness prerequisites",
    )
    if not model_dir.is_dir():
        raise ValueError("model directory is missing")
    if not model_manifest_path.is_file():
        raise ValueError("model manifest is missing")
    if not correctness_prerequisites_path.is_file():
        raise ValueError("correctness prerequisites are missing")
    model_manifest_sha256 = _sha256_file(model_manifest_path)
    if model_manifest_sha256 != expected_model_manifest_sha256:
        raise ValueError("model manifest SHA mismatch")
    correctness_prerequisites_sha256 = _sha256_file(
        correctness_prerequisites_path
    )
    if (
        correctness_prerequisites_sha256
        != expected_correctness_prerequisites_sha256
    ):
        raise ValueError("correctness prerequisites SHA mismatch")
    try:
        model_manifest = json.loads(
            model_manifest_path.read_text(encoding="utf-8")
        )
        prerequisites = json.loads(
            correctness_prerequisites_path.read_text(encoding="utf-8")
        )
    except json.JSONDecodeError as error:
        raise ValueError("runtime authorization JSON is invalid") from error
    if not isinstance(model_manifest, dict):
        raise ValueError("model manifest is invalid")
    prerequisite_status = contract.validate_prerequisites(
        correctness_prerequisites_path
    )
    if not prerequisite_status.authorized:
        detail = "; ".join(prerequisite_status.reasons)
        raise ValueError(
            "correctness prerequisites are not PASS"
            + (f": {detail}" if detail else "")
        )
    if workload_manifest_path is None:
        raise ValueError("workload manifest is missing")
    workload_manifest_path = Path(workload_manifest_path)
    expected_workload_manifest_sha256 = _require_sha256(
        expected_workload_manifest_sha256,
        "workload manifest",
    )
    canonical_workload_sha256 = (
        contract.canonical_json_file_sha256(
            contract.workload_manifest_payload()
        )
    )
    if expected_workload_manifest_sha256 != canonical_workload_sha256:
        raise ValueError("canonical workload manifest SHA mismatch")
    if not workload_manifest_path.is_file():
        raise ValueError("workload manifest is missing")
    workload_manifest_sha256 = _sha256_file(
        workload_manifest_path
    )
    if workload_manifest_sha256 != expected_workload_manifest_sha256:
        raise ValueError("workload manifest SHA mismatch")
    try:
        workload_manifest = json.loads(
            workload_manifest_path.read_text(encoding="utf-8")
        )
    except json.JSONDecodeError as error:
        raise ValueError("workload manifest JSON is invalid") from error
    if workload_manifest != contract.workload_manifest_payload():
        raise ValueError("workload manifest content mismatch")
    manifest_model_dir = model_manifest.get(
        "remote_model_dir",
        model_manifest.get("local_path"),
    )
    if (
        not isinstance(manifest_model_dir, str)
        or Path(manifest_model_dir).resolve()
        != model_dir.resolve()
    ):
        raise ValueError("model directory does not match manifest")
    return {
        "model_dir": str(model_dir),
        "model_manifest_path": str(model_manifest_path),
        "model_manifest_sha256": model_manifest_sha256,
        "correctness_prerequisites_path": str(
            correctness_prerequisites_path
        ),
        "correctness_prerequisites_sha256": (
            correctness_prerequisites_sha256
        ),
        "workload_manifest_path": str(workload_manifest_path),
        "workload_manifest_sha256": workload_manifest_sha256,
    }


def _phase_repetitions():
    return (
        ("warmup", range(contract.WARMUP_REPETITIONS)),
        (
            "correctness",
            range(contract.CORRECTNESS_REPETITIONS),
        ),
        ("measured", range(contract.MEASURED_REPETITIONS)),
    )


def _case_id(workload, policy, phase, repetition):
    return f"{workload}__{phase}__r{repetition}__{policy}"


def _canonical_case(case):
    fields = (
        "case_id",
        "workload",
        "policy",
        "phase",
        "repetition",
    )
    try:
        identity = tuple(getattr(case, name) for name in fields)
    except AttributeError as error:
        raise ValueError("benchmark case is not canonical") from error
    for candidate in contract.build_case_matrix():
        if tuple(getattr(candidate, name) for name in fields) == identity:
            return candidate
    if (
        case.workload == "w2_long_reuse"
        and case.policy in contract.POLICIES
        and case.phase == "nsys_replay"
        and case.repetition in range(5)
        and case.case_id == _case_id(
            case.workload,
            case.policy,
            case.phase,
            case.repetition,
        )
    ):
        return case
    raise ValueError("benchmark case is not canonical")


def _zero_cache_snapshot():
    return {
        "current_entries": 0,
        "current_bytes": 0,
        "current_logical_bytes": 0,
        "deduplicated_bytes": 0,
        "peak_entries": 0,
        "peak_bytes": 0,
        "hits": 0,
        "misses": 0,
        "evictions": 0,
        "validation_failures": 0,
        "failed_restores": 0,
    }


def aggregate_tp4_cache_snapshots(rows):
    if not isinstance(rows, (tuple, list)):
        raise ValueError("TP4 cache snapshot rows must be a list")
    expected_fields = {"rank", *_zero_cache_snapshot()}
    by_rank = {}
    for row in rows:
        if (
            not isinstance(row, dict)
            or not expected_fields.issubset(row)
        ):
            raise ValueError("TP4 cache snapshot schema mismatch")
        rank = row["rank"]
        if (
            isinstance(rank, bool)
            or not isinstance(rank, int)
            or rank < 0
            or rank in by_rank
        ):
            raise ValueError("TP4 cache rank inventory mismatch")
        normalized = {}
        for name in _zero_cache_snapshot():
            value = row[name]
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 0
            ):
                raise ValueError(
                    f"TP4 cache counter is invalid: {name}"
                )
            normalized[name] = value
        if (
            normalized["deduplicated_bytes"]
            != (
                normalized["current_logical_bytes"]
                - normalized["current_bytes"]
            )
        ):
            raise ValueError("TP4 cache accounting mismatch")
        by_rank[rank] = normalized
    if tuple(sorted(by_rank)) != tuple(range(contract.WORLD_SIZE)):
        raise ValueError("TP4 cache rank inventory mismatch")

    byte_fields = (
        "current_bytes",
        "current_logical_bytes",
        "deduplicated_bytes",
        "peak_bytes",
    )
    parity_fields = tuple(
        name for name in _zero_cache_snapshot()
        if name not in byte_fields
    )
    reference = by_rank[0]
    for name in parity_fields:
        if any(
            by_rank[rank][name] != reference[name]
            for rank in range(1, contract.WORLD_SIZE)
        ):
            raise ValueError(
                f"TP4 cache counter parity mismatch: {name}"
            )
    result = {
        name: sum(by_rank[rank][name] for rank in by_rank)
        for name in byte_fields
    }
    result.update({
        name: reference[name]
        for name in parity_fields
    })
    return {
        name: result[name]
        for name in _zero_cache_snapshot()
    }


def aggregate_tp4_memory_snapshots(rows):
    if not isinstance(rows, (tuple, list)):
        raise ValueError("TP4 memory snapshot rows must be a list")
    memory_fields = (
        "cuda_allocated_bytes",
        "cuda_reserved_bytes",
        "cuda_peak_allocated_bytes",
        "cuda_peak_reserved_bytes",
        "kv_capacity_bytes",
    )
    expected_fields = {"rank", *memory_fields}
    by_rank = {}
    for row in rows:
        if not isinstance(row, dict) or set(row) != expected_fields:
            raise ValueError("TP4 memory snapshot schema mismatch")
        rank = row["rank"]
        if (
            isinstance(rank, bool)
            or not isinstance(rank, int)
            or rank < 0
            or rank in by_rank
        ):
            raise ValueError("TP4 memory rank inventory mismatch")
        normalized = {}
        for name in memory_fields:
            value = row[name]
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 0
            ):
                raise ValueError(
                    f"TP4 memory counter is invalid: {name}"
                )
            normalized[name] = value
        by_rank[rank] = normalized
    if tuple(sorted(by_rank)) != tuple(range(contract.WORLD_SIZE)):
        raise ValueError("TP4 memory rank inventory mismatch")
    return {
        name: sum(
            by_rank[rank][name]
            for rank in range(contract.WORLD_SIZE)
        )
        for name in memory_fields
    }


def _normalize_cache_snapshot(policy, engine):
    if policy == "recompute":
        return _zero_cache_snapshot()
    tp4_snapshot = getattr(
        engine,
        "qwen35_hybrid_prefix_cache_snapshots",
        None,
    )
    if callable(tp4_snapshot):
        snapshot = aggregate_tp4_cache_snapshots(
            tp4_snapshot(timeout_s=HYBRID_PREFIX_TIMEOUT_S)
        )
    else:
        legacy_snapshot = getattr(
            engine,
            "hybrid_prefix_cache_snapshot",
            None,
        )
        if not callable(legacy_snapshot):
            raise ValueError(
                "hybrid-prefix cache snapshot transport is missing"
            )
        snapshot = legacy_snapshot()
    required = set(_zero_cache_snapshot())
    if not isinstance(snapshot, dict) or set(snapshot) != required:
        raise ValueError("hybrid-prefix cache snapshot schema mismatch")
    normalized = {}
    for name in sorted(required):
        value = snapshot[name]
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or value < 0
        ):
            raise ValueError(
                f"hybrid-prefix cache counter is invalid: {name}"
            )
        normalized[name] = value
    if (
        normalized["current_bytes"]
        > normalized["current_logical_bytes"]
        or normalized["deduplicated_bytes"]
        != (
            normalized["current_logical_bytes"]
            - normalized["current_bytes"]
        )
    ):
        raise ValueError("hybrid-prefix cache accounting mismatch")
    return normalized


def _write_logits(
    output_dir,
    *,
    case_id,
    request_id,
    final_logits,
):
    if final_logits is None:
        return None, None
    relative = (
        Path("logits")
        / f"{case_id}__{request_id}.json"
    )
    data = contract.canonical_json_bytes(final_logits) + b"\n"
    _atomic_write(Path(output_dir) / relative, data)
    return relative.as_posix(), _sha256_bytes(data)


def _case_row(
    *,
    request,
    request_index,
    policy,
    workload,
    phase,
    repetition,
    output_dir,
    source_tree_sha256,
    model_manifest_sha256,
    workload_manifest_sha256,
    correctness_prerequisites_sha256,
):
    case_id = _case_id(
        workload,
        policy,
        phase,
        repetition,
    )
    request_id = request["request_id"]
    output_token_ids = list(request["output_token_ids"])
    logits_path, logits_sha256 = _write_logits(
        output_dir,
        case_id=case_id,
        request_id=request_id,
        final_logits=request.get("final_logits"),
    )
    row = {
        "row_id": f"{case_id}__request-{request_index}",
        "case_id": case_id,
        "policy": policy,
        "workload": workload,
        "phase": phase,
        "repetition": repetition,
        "request_id": request_id,
        "source_tree_sha256": source_tree_sha256,
        "model_manifest_sha256": model_manifest_sha256,
        "workload_manifest_sha256": workload_manifest_sha256,
        "correctness_prerequisites_sha256": (
            correctness_prerequisites_sha256
        ),
        "prompt_tokens": int(request["prompt_tokens"]),
        "reused_kv_tokens": int(request["reused_kv_tokens"]),
        "restored_hybrid_state": bool(
            request["restored_hybrid_state"]
        ),
        "executed_prefill_tokens": int(
            request["executed_prefill_tokens"]
        ),
        "generated_tokens": int(request["generated_tokens"]),
        "ttft_ns": int(request["ttft_ns"]),
        "e2e_ns": int(request["e2e_ns"]),
        "decode_step_ns": [
            int(value) for value in request["decode_step_ns"]
        ],
        "output_token_ids": output_token_ids,
        "output_token_ids_sha256": (
            contract.canonical_json_sha256(output_token_ids)
        ),
        "final_logits_path": logits_path,
        "final_logits_sha256": logits_sha256,
    }
    if set(row) != set(contract.CASE_ROW_FIELDS):
        raise RuntimeError("case row schema mismatch")
    return row


def validate_benchmark_requests(
    *,
    workload,
    policy,
    requests,
    workload_payload=None,
):
    workload = _require_workload(workload)
    policy = _require_policy(policy)
    payload = (
        workload_payload
        if workload_payload is not None
        else contract.workload_payload(workload)
    )
    if (
        not isinstance(payload, dict)
        or not isinstance(payload.get("spec"), dict)
    ):
        raise ValueError("benchmark workload payload mismatch")
    spec = payload["spec"]
    if (
        not isinstance(requests, list)
        or len(requests) != spec["continuations"]
    ):
        raise ValueError("benchmark request count mismatch")
    required = {
        "request_id",
        "prompt_tokens",
        "reused_kv_tokens",
        "restored_hybrid_state",
        "executed_prefill_tokens",
        "generated_tokens",
        "ttft_ns",
        "e2e_ns",
        "decode_step_ns",
        "output_token_ids",
    }
    allowed = required | {"final_logits"}
    expected_restore = (
        policy == "exact_restore"
        and workload in {
            "w1_medium_reuse",
            "w2_long_reuse",
            "w3_batched_fanout",
        }
    )
    prompt_tokens = (
        spec["shared_prefix_tokens"] + spec["suffix_tokens"]
    )
    expected_reused = (
        spec["shared_prefix_tokens"] if expected_restore else 0
    )
    expected_prefill = (
        spec["suffix_tokens"] if expected_restore else prompt_tokens
    )
    request_ids = set()
    for request in requests:
        if (
            not isinstance(request, dict)
            or not required.issubset(request)
            or not set(request).issubset(allowed)
        ):
            raise ValueError("benchmark request schema mismatch")
        request_id = request["request_id"]
        if (
            not isinstance(request_id, str)
            or not request_id
            or request_id in request_ids
        ):
            raise ValueError("benchmark request_id is invalid")
        request_ids.add(request_id)
        restored = request["restored_hybrid_state"]
        if (
            not isinstance(restored, bool)
            or restored is not expected_restore
        ):
            raise ValueError(
                "benchmark restored_hybrid_state mismatch"
            )
        expected_integers = {
            "prompt_tokens": prompt_tokens,
            "reused_kv_tokens": expected_reused,
            "executed_prefill_tokens": expected_prefill,
            "generated_tokens": spec["generated_tokens"],
        }
        for name, expected in expected_integers.items():
            value = request[name]
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value != expected
            ):
                raise ValueError(f"benchmark {name} mismatch")
        output_token_ids = request["output_token_ids"]
        if (
            not isinstance(output_token_ids, list)
            or len(output_token_ids) != spec["generated_tokens"]
            or any(
                isinstance(token_id, bool)
                or not isinstance(token_id, int)
                or token_id < 0
                for token_id in output_token_ids
            )
        ):
            raise ValueError(
                "benchmark output_token_ids mismatch"
            )
        decode_step_ns = request["decode_step_ns"]
        if (
            not isinstance(decode_step_ns, list)
            or len(decode_step_ns) != max(
                spec["generated_tokens"] - 1,
                0,
            )
            or any(
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 0
                for value in decode_step_ns
            )
        ):
            raise ValueError("benchmark decode_step_ns mismatch")
        for name in ("ttft_ns", "e2e_ns"):
            value = request[name]
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 0
            ):
                raise ValueError(f"benchmark {name} is invalid")
        if request["e2e_ns"] < request["ttft_ns"]:
            raise ValueError("benchmark timing order mismatch")
    return requests


def _process_row(
    *,
    engine,
    policy,
    workload,
    phase,
    repetition,
    initialization_ns,
):
    tp4_memory_snapshot = getattr(engine, "memory_snapshots", None)
    if callable(tp4_memory_snapshot):
        memory = aggregate_tp4_memory_snapshots(
            tp4_memory_snapshot(timeout_s=HYBRID_PREFIX_TIMEOUT_S)
        )
    else:
        memory = engine.memory_snapshot()
    capacity = engine.capacity_snapshot()
    cache = _normalize_cache_snapshot(policy, engine)
    row = {
        "case_id": _case_id(
            workload,
            policy,
            phase,
            repetition,
        ),
        "policy": policy,
        "workload": workload,
        "phase": phase,
        "repetition": repetition,
        "initialization_ns": int(initialization_ns),
        "cuda_allocated_bytes": int(
            memory["cuda_allocated_bytes"]
        ),
        "cuda_reserved_bytes": int(
            memory["cuda_reserved_bytes"]
        ),
        "cuda_peak_allocated_bytes": int(
            memory["cuda_peak_allocated_bytes"]
        ),
        "cuda_peak_reserved_bytes": int(
            memory["cuda_peak_reserved_bytes"]
        ),
        "kv_capacity_bytes": int(memory["kv_capacity_bytes"]),
        "scheduler_visible_kv_blocks": int(
            capacity["num_kvcache_blocks"]
        ),
        "hybrid_cache_current_entries": cache["current_entries"],
        "hybrid_cache_current_bytes": cache["current_bytes"],
        "hybrid_cache_current_logical_bytes": cache[
            "current_logical_bytes"
        ],
        "hybrid_cache_deduplicated_bytes": cache[
            "deduplicated_bytes"
        ],
        "hybrid_cache_peak_entries": cache["peak_entries"],
        "hybrid_cache_peak_bytes": cache["peak_bytes"],
        "hybrid_cache_hits": cache["hits"],
        "hybrid_cache_misses": cache["misses"],
        "hybrid_cache_evictions": cache["evictions"],
        "hybrid_cache_validation_failures": cache[
            "validation_failures"
        ],
        "hybrid_cache_failed_restores": cache[
            "failed_restores"
        ],
    }
    if set(row) != set(contract.PROCESS_ROW_FIELDS):
        raise RuntimeError("process row schema mismatch")
    return row


def run_benchmark_case(
    *,
    case,
    output_dir,
    engine_factory,
    clock_ns,
    cuda_sync,
    reset_peak_memory,
    source_tree_sha256=None,
    model_manifest_sha256=contract.MODEL_MANIFEST_SHA256,
    workload_manifest_sha256=None,
    correctness_prerequisites_sha256=None,
    recurrent_calibration_capture_dir=None,
    profiling=False,
    generated_tokens_override=None,
    decode_internal_profile=False,
):
    case = _canonical_case(case)
    canonical_workload_payload = contract.workload_payload(
        case.workload
    )
    canonical_generated_tokens = canonical_workload_payload[
        "spec"
    ]["generated_tokens"]
    if generated_tokens_override is not None:
        if not profiling:
            raise ValueError(
                "generated-token override requires profiling"
            )
        if case.workload != "w2_long_reuse":
            raise ValueError(
                "generated-token override requires w2_long_reuse"
            )
        if (
            isinstance(generated_tokens_override, bool)
            or not isinstance(generated_tokens_override, int)
            or generated_tokens_override <= 0
        ):
            raise ValueError(
                "generated-token override must be a positive integer"
            )
        if generated_tokens_override > canonical_generated_tokens:
            raise ValueError(
                "generated-token override exceeds canonical "
                "generated tokens"
            )
    effective_generated_tokens = (
        generated_tokens_override
        if generated_tokens_override is not None
        else canonical_generated_tokens
    )
    if decode_internal_profile and (
        not profiling
        or case.workload != "w2_long_reuse"
        or effective_generated_tokens != 8
    ):
        raise ValueError(
            "decode internal profile requires profiled "
            "eight-token w2_long_reuse"
        )
    effective_workload_payload = copy.deepcopy(
        canonical_workload_payload
    )
    effective_workload_payload["spec"]["generated_tokens"] = (
        effective_generated_tokens
    )
    source_tree_sha256 = _require_sha256(
        source_tree_sha256,
        "benchmark source tree",
    )
    model_manifest_sha256 = _require_sha256(
        model_manifest_sha256,
        "model manifest",
    )
    workload_manifest_sha256 = _require_sha256(
        workload_manifest_sha256
        or contract.canonical_json_file_sha256(
            contract.workload_manifest_payload()
        ),
        "workload manifest",
    )
    correctness_prerequisites_sha256 = _require_sha256(
        correctness_prerequisites_sha256,
        "correctness prerequisites",
    )
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    capture_identity_fields = {
        "model_manifest_sha256": model_manifest_sha256,
        "source_tree_sha256": source_tree_sha256,
        "workload_manifest_sha256": workload_manifest_sha256,
        "world_size": contract.WORLD_SIZE,
        "workload_ids": list(contract.WORKLOADS),
    }
    configuration = build_engine_configuration(
        case.policy,
        case,
        workload_payload=effective_workload_payload,
        recurrent_calibration_capture_dir=(
            recurrent_calibration_capture_dir
        ),
        capture_identity_fields=(
            capture_identity_fields
            if recurrent_calibration_capture_dir is not None
            else None
        ),
        expected_capture_identity_fields=(
            capture_identity_fields
            if recurrent_calibration_capture_dir is not None
            else None
        ),
    )
    if profiling:
        configuration["profiling"] = {
            "enabled": True,
            **(
                {"decode_internal": True}
                if decode_internal_profile
                else {}
            ),
        }
    engine = None
    try:
        initialization_start = clock_ns()
        engine = engine_factory(configuration)
        cuda_sync()
        initialization_ns = clock_ns() - initialization_start
        if case.policy == "exact_restore":
            hybrid = configuration["hybrid_prefix"]
            engine.configure_qwen35_hybrid_prefix_publication_runtime(
                model_fingerprint=model_manifest_sha256,
                max_entries=hybrid["max_entries"],
                max_bytes=hybrid["max_bytes"],
                timeout_s=hybrid["timeout_s"],
            )
        if case.phase == "measured":
            cuda_sync()
            reset_peak_memory()
        cuda_sync()
        payload = engine.run_benchmark_workload(
            workload=case.workload,
            workload_spec=effective_workload_payload,
            phase=case.phase,
            repetition=case.repetition,
            policy=case.policy,
        )
        cuda_sync()
        requests = payload.get("requests")
        requests = validate_benchmark_requests(
            workload=case.workload,
            policy=case.policy,
            requests=requests,
            workload_payload=effective_workload_payload,
        )
        case_rows = [
            _case_row(
                request=request,
                request_index=request_index,
                policy=case.policy,
                workload=case.workload,
                phase=case.phase,
                repetition=case.repetition,
                output_dir=output_dir,
                source_tree_sha256=source_tree_sha256,
                model_manifest_sha256=model_manifest_sha256,
                workload_manifest_sha256=workload_manifest_sha256,
                correctness_prerequisites_sha256=(
                    correctness_prerequisites_sha256
                ),
            )
            for request_index, request in enumerate(requests)
        ]
        process_rows = [_process_row(
            engine=engine,
            policy=case.policy,
            workload=case.workload,
            phase=case.phase,
            repetition=case.repetition,
            initialization_ns=initialization_ns,
        )]
        _atomic_write_jsonl(
            output_dir / "case_rows.jsonl",
            case_rows,
        )
        _atomic_write_jsonl(
            output_dir / "process_rows.jsonl",
            process_rows,
        )
        if profiling:
            profile_snapshot = getattr(
                engine,
                "profile_snapshot",
                None,
            )
            if not callable(profile_snapshot):
                raise RuntimeError(
                    "profiling engine lacks profile_snapshot"
                )
            profile = profile_snapshot()
            if (
                not isinstance(profile, dict)
                or profile.get("enabled") is not True
                or not isinstance(profile.get("events"), list)
                or not isinstance(profile.get("requests"), list)
            ):
                raise RuntimeError(
                    "profiling engine snapshot is invalid"
                )
            _atomic_write_json(
                output_dir / "profile.json",
                {
                    "schema_version": (
                        "qwen35.tp4-w2-restore-profile-case.v1"
                    ),
                    "case_id": case.case_id,
                    "policy": case.policy,
                    "workload": case.workload,
                    "phase": case.phase,
                    "repetition": case.repetition,
                    "variant": (
                        "short_output"
                        if generated_tokens_override is not None
                        else "canonical_output"
                    ),
                    "canonical_generated_tokens": (
                        canonical_generated_tokens
                    ),
                    "generated_tokens": (
                        effective_generated_tokens
                    ),
                    "events": profile["events"],
                    "requests": profile["requests"],
                },
            )
            if decode_internal_profile:
                decode_internal = profile.get("decode_internal")
                if (
                    not isinstance(decode_internal, dict)
                    or decode_internal.get("enabled") is not True
                    or decode_internal.get("rank_inventory")
                    != [0, 1, 2, 3]
                    or not isinstance(
                        decode_internal.get("ranks"),
                        list,
                    )
                ):
                    raise RuntimeError(
                        "decode internal profiling snapshot is invalid"
                    )
                decode_payload = {
                    "schema_version": (
                        decode_profile_contract.SCHEMA_VERSION
                    ),
                    "variant": "decode_internal",
                    "resource_policy": "shared-low-utilization",
                    "exclusive": False,
                    "source_tree_sha256": source_tree_sha256,
                    "workload_manifest_sha256": (
                        workload_manifest_sha256
                    ),
                    "case_id": case.case_id,
                    "workload": case.workload,
                    "policy": case.policy,
                    "phase": case.phase,
                    "repetition": case.repetition,
                    "generated_tokens": effective_generated_tokens,
                    "units": "nanoseconds",
                    "rank_inventory": [0, 1, 2, 3],
                    "finalization_status": "complete",
                    "ranks": decode_internal["ranks"],
                }
                decode_profile_contract.validate_decode_profile(
                    decode_payload
                )
                _atomic_write_json(
                    output_dir / "decode_profile.json",
                    decode_payload,
                )
        summary = {
            "schema_version": contract.SCHEMA_VERSION,
            "complete": True,
            "case_id": case.case_id,
            "case_rows": len(case_rows),
            "process_rows": len(process_rows),
        }
        _atomic_write_json(output_dir / "summary.json", summary)
        return summary
    except Exception as error:
        _atomic_write_json(
            output_dir / "failure.json",
            {
                "schema_version": contract.SCHEMA_VERSION,
                "complete": False,
                "case_id": case.case_id,
                "error_type": type(error).__name__,
                "error": str(error),
            },
        )
        raise
    finally:
        if engine is not None:
            engine.close()


def run_policy_workload(
    *,
    policy,
    workload,
    output_dir,
    engine_factory,
    clock_ns,
    cuda_sync,
    reset_peak_memory,
    source_tree_sha256=None,
    model_manifest_sha256=contract.MODEL_MANIFEST_SHA256,
    workload_manifest_sha256=None,
    correctness_prerequisites_sha256=None,
):
    policy = _require_policy(policy)
    workload = _require_workload(workload)
    source_tree_sha256 = _require_sha256(
        source_tree_sha256,
        "benchmark source tree",
    )
    model_manifest_sha256 = _require_sha256(
        model_manifest_sha256,
        "model manifest",
    )
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    workload_manifest_sha256 = (
        workload_manifest_sha256
        or contract.canonical_json_file_sha256(
            contract.workload_manifest_payload()
        )
    )
    correctness_prerequisites_sha256 = (
        correctness_prerequisites_sha256
        or "0" * 64
    )
    case = next(
        case
        for case in contract.build_case_matrix()
        if case.workload == workload and case.policy == policy
    )
    configuration = build_engine_configuration(policy, case)
    engine = None
    case_rows = []
    process_rows = []
    try:
        initialization_start = clock_ns()
        engine = engine_factory(configuration)
        cuda_sync()
        initialization_ns = clock_ns() - initialization_start
        if policy == "exact_restore":
            hybrid = configuration["hybrid_prefix"]
            engine.configure_qwen35_hybrid_prefix_publication_runtime(
                model_fingerprint=model_manifest_sha256,
                max_entries=hybrid["max_entries"],
                max_bytes=hybrid["max_bytes"],
                timeout_s=hybrid["timeout_s"],
            )
        measured_started = False
        for phase, repetitions in _phase_repetitions():
            for repetition in repetitions:
                if phase == "measured" and not measured_started:
                    cuda_sync()
                    reset_peak_memory()
                    measured_started = True
                cuda_sync()
                payload = engine.run_benchmark_workload(
                    workload=workload,
                    workload_spec=contract.workload_payload(workload),
                    phase=phase,
                    repetition=repetition,
                    policy=policy,
                )
                cuda_sync()
                requests = payload.get("requests")
                requests = validate_benchmark_requests(
                    workload=workload,
                    policy=policy,
                    requests=requests,
                )
                for request_index, request in enumerate(requests):
                    case_rows.append(_case_row(
                        request=request,
                        request_index=request_index,
                        policy=policy,
                        workload=workload,
                        phase=phase,
                        repetition=repetition,
                        output_dir=output_dir,
                        source_tree_sha256=source_tree_sha256,
                        model_manifest_sha256=model_manifest_sha256,
                        workload_manifest_sha256=(
                            workload_manifest_sha256
                        ),
                        correctness_prerequisites_sha256=(
                            correctness_prerequisites_sha256
                        ),
                    ))
                process_rows.append(_process_row(
                    engine=engine,
                    policy=policy,
                    workload=workload,
                    phase=phase,
                    repetition=repetition,
                    initialization_ns=initialization_ns,
                ))
        _atomic_write_jsonl(
            output_dir / "case_rows.jsonl",
            case_rows,
        )
        _atomic_write_jsonl(
            output_dir / "process_rows.jsonl",
            process_rows,
        )
        summary = {
            "schema_version": contract.SCHEMA_VERSION,
            "complete": True,
            "policy": policy,
            "workload": workload,
            "case_rows": len(case_rows),
            "process_rows": len(process_rows),
        }
        _atomic_write_json(output_dir / "summary.json", summary)
        return summary
    except Exception as error:
        _atomic_write_json(
            output_dir / "failure.json",
            {
                "schema_version": contract.SCHEMA_VERSION,
                "complete": False,
                "policy": policy,
                "workload": workload,
                "error_type": type(error).__name__,
                "error": str(error),
            },
        )
        raise
    finally:
        if engine is not None:
            engine.close()


def _default_runtime_loader(configuration, authorized):
    module_name = (
        "qwen35_tp4_hybrid_prefix_benchmark_engine_adapter"
    )
    module = __import__(module_name)
    return module.BenchmarkEngineAdapter(
        configuration,
        authorized,
    )


def _default_cuda_sync():
    import torch
    torch.cuda.synchronize()


def _default_reset_peak_memory():
    import torch
    torch.cuda.reset_peak_memory_stats()


def _case_from_arguments(args):
    if args.phase == "nsys_replay":
        if not (
            args.policy in contract.POLICIES
            and args.workload == "w2_long_reuse"
            and isinstance(args.repetition, int)
            and not isinstance(args.repetition, bool)
            and args.repetition in range(5)
            and args.profile is True
            and args.generated_tokens_override == 8
            and args.decode_internal_profile is True
        ):
            raise ValueError(
                "nsys replay requires the eight-token w2 "
                "decode internal profile"
            )
        return contract.BenchmarkCase(
            case_id=(
                f"{args.workload}__nsys_replay__"
                f"r{args.repetition}__{args.policy}"
            ),
            workload=args.workload,
            policy=args.policy,
            phase=args.phase,
            repetition=args.repetition,
        )
    for case in contract.build_case_matrix():
        if (
            case.policy == args.policy
            and case.workload == args.workload
            and case.phase == args.phase
            and case.repetition == args.repetition
        ):
            return case
    raise ValueError("benchmark case is not canonical")


def main(
    argv=None,
    *,
    runtime_loader=_default_runtime_loader,
    clock_ns=time.monotonic_ns,
    cuda_sync=_default_cuda_sync,
    reset_peak_memory=_default_reset_peak_memory,
):
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", required=True)
    parser.add_argument("--workload", required=True)
    parser.add_argument("--phase", required=True)
    parser.add_argument("--repetition", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-tree-sha256", required=True)
    parser.add_argument("--model-manifest-sha256", required=True)
    parser.add_argument("--prerequisites-sha256", required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--model-manifest", type=Path, required=True)
    parser.add_argument(
        "--correctness-prerequisites",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--workload-manifest",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--workload-manifest-sha256",
        required=True,
    )
    parser.add_argument(
        "--recurrent-calibration-capture-dir",
        type=_non_empty_path,
    )
    parser.add_argument(
        "--profile",
        action="store_true",
    )
    parser.add_argument(
        "--generated-tokens-override",
        type=int,
    )
    parser.add_argument(
        "--decode-internal-profile",
        action="store_true",
    )
    args = parser.parse_args(argv)
    case = _case_from_arguments(args)
    source_tree_sha256 = _require_sha256(
        args.source_tree_sha256,
        "benchmark source tree",
    )
    authorized = validate_runtime_artifacts(
        model_dir=args.model_dir,
        model_manifest_path=args.model_manifest,
        expected_model_manifest_sha256=(
            args.model_manifest_sha256
        ),
        correctness_prerequisites_path=(
            args.correctness_prerequisites
        ),
        expected_correctness_prerequisites_sha256=(
            args.prerequisites_sha256
        ),
        workload_manifest_path=args.workload_manifest,
        expected_workload_manifest_sha256=(
            args.workload_manifest_sha256
        ),
    )

    def engine_factory(configuration):
        return runtime_loader(configuration, authorized)

    run_benchmark_case(
        case=case,
        output_dir=args.output_dir,
        engine_factory=engine_factory,
        clock_ns=clock_ns,
        cuda_sync=cuda_sync,
        reset_peak_memory=reset_peak_memory,
        source_tree_sha256=source_tree_sha256,
        model_manifest_sha256=authorized[
            "model_manifest_sha256"
        ],
        workload_manifest_sha256=authorized[
            "workload_manifest_sha256"
        ],
        correctness_prerequisites_sha256=authorized[
            "correctness_prerequisites_sha256"
        ],
        recurrent_calibration_capture_dir=(
            args.recurrent_calibration_capture_dir
        ),
        profiling=args.profile,
        generated_tokens_override=(
            args.generated_tokens_override
        ),
        decode_internal_profile=args.decode_internal_profile,
    )
    print(COMPLETION_MARKER)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
