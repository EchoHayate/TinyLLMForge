"""Reproducible evidence gate for the native multi-token verifier."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import socket
import statistics
import subprocess
import sys
import time
from pathlib import Path


POLICIES = (
    "baseline",
    "legacy_rematerialize",
    "native",
    "oracle",
)

REQUIRED_ARTIFACTS = (
    "manifest.json",
    "capability.json",
    "case_rows.json",
    "event_rows.json",
    "summary.json",
    "report.md",
)

THRESHOLDS = {
    "k1_max_regression_fraction": 0.01,
    "min_continuation_steps": 16,
    "native_k_gt_1_must_beat_legacy": True,
    "target_forward_reduction_must_equal_removed_replay": True,
}

DTYPE_TOLERANCES = {
    "torch.float16": {
        "logits_rtol": 2e-3,
        "logits_atol": 2e-3,
        "kv_rtol": 2e-3,
        "kv_atol": 2e-3,
    },
    "torch.bfloat16": {
        "logits_rtol": 8e-3,
        "logits_atol": 8e-3,
        "kv_rtol": 8e-3,
        "kv_atol": 8e-3,
    },
}

CLAIM_BOUNDARIES = (
    "profiler-owned only",
    "strict greedy decoding only",
    "single sequence only",
    "linear drafts only",
    "eager execution only",
    "FP16/BF16 KV only",
    "no production batch throughput claim",
    "no ragged or tree verification claim",
    "no non-greedy equivalence claim",
    "no CUDA graph support claim",
    "no KV offload support claim",
    "no quantized KV support claim",
    "no memory reduction claim",
    "no production GO claim",
)

_PROMPT = (
    "Repeat the sequence alpha beta gamma while preserving exact spacing: "
    "alpha beta gamma alpha beta gamma."
)
_EOS_PROMPT = (
    "<|im_start|>user\n"
    "Reply with exactly OK and then stop.<|im_end|>\n"
    "<|im_start|>assistant\n"
)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_json(value) -> str:
    return sha256_text(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _case(
    case_id: str,
    *,
    draft_len: int,
    acceptance_case: str,
    history_len: int,
    block_case: str,
    eos_case: bool = False,
    output_budget_case: bool = False,
    prompt: str = _PROMPT,
) -> dict:
    return {
        "case_id": case_id,
        "prompt": prompt,
        "prompt_sha256": sha256_text(prompt),
        "history_len": history_len,
        "draft_len": draft_len,
        "acceptance_case": acceptance_case,
        "eos_case": eos_case,
        "output_budget_case": output_budget_case,
        "block_case": block_case,
        "block_size": 256,
        "continuation_steps": 16,
        "draft_construction": acceptance_case,
    }


CASE_MATRIX = (
    _case(
        "k1-full-current",
        draft_len=1,
        acceptance_case="full",
        history_len=52,
        block_case="current_block",
    ),
    _case(
        "k4-zero-current",
        draft_len=4,
        acceptance_case="zero",
        history_len=52,
        block_case="current_block",
    ),
    _case(
        "k4-one-current",
        draft_len=4,
        acceptance_case="one",
        history_len=52,
        block_case="current_block",
    ),
    _case(
        "k4-partial-boundary",
        draft_len=4,
        acceptance_case="partial",
        history_len=255,
        block_case="one_new_block",
    ),
    _case(
        "k4-full-boundary",
        draft_len=4,
        acceptance_case="full",
        history_len=255,
        block_case="one_new_block",
    ),
    _case(
        "k8-full-boundary",
        draft_len=8,
        acceptance_case="full",
        history_len=255,
        block_case="one_new_block",
    ),
    _case(
        "k8-eos-real-history",
        draft_len=8,
        acceptance_case="partial",
        history_len=255,
        block_case="real_eos_history",
        eos_case=True,
        prompt=_EOS_PROMPT,
    ),
    _case(
        "k8-budget-boundary",
        draft_len=8,
        acceptance_case="full",
        history_len=255,
        block_case="one_new_block",
        output_budget_case=True,
    ),
    _case(
        "k16-full-multiblock-context",
        draft_len=16,
        acceptance_case="full",
        history_len=511,
        block_case="multi_block_context",
    ),
)


def build_capability_specs(bf16_supported: bool) -> list[dict]:
    dtypes = ["torch.float16"]
    if bf16_supported:
        dtypes.append("torch.bfloat16")
    return [
        {
            "dtype": dtype,
            "query_len": query_len,
            "block_case": block_case,
            "gqa": True,
        }
        for dtype in dtypes
        for query_len in (1, 3, 7, 15)
        for block_case in ("one_block", "cross_block")
    ]


def build_manifest(
    source_commit: str,
    source_dirty: bool,
    model_path: str,
    model_identifier: str,
    host: str,
    python_bin: str,
    torch_version: str,
    cuda_version: str,
    flash_attn_version: str,
    gpu_name: str,
    bf16_supported: bool,
    run_tag: str,
) -> dict:
    return {
        "schema_version": 1,
        "run_tag": str(run_tag),
        "created_unix_s": time.time(),
        "source_commit": str(source_commit),
        "source_dirty": bool(source_dirty),
        "model_path": str(model_path),
        "model_identifier": str(model_identifier),
        "host": str(host),
        "python_bin": str(python_bin),
        "torch_version": str(torch_version),
        "cuda_version": str(cuda_version),
        "flash_attn_version": str(flash_attn_version),
        "gpu_name": str(gpu_name),
        "bf16_supported": bool(bf16_supported),
        "dtype_tolerances": DTYPE_TOLERANCES,
        "thresholds": THRESHOLDS,
        "case_matrix": list(CASE_MATRIX),
        "case_matrix_sha256": sha256_json(CASE_MATRIX),
        "policies": list(POLICIES),
        "required_artifacts": list(REQUIRED_ARTIFACTS),
        "claim_boundaries": list(CLAIM_BOUNDARIES),
        "classification_on_success": "READY_FOR_PERFORMANCE_GATE",
        "process_port_pairs": [],
        "prompt_history_hashes": {
            case["case_id"]: {
                "prompt_sha256": case["prompt_sha256"],
                "history_contract_sha256": sha256_json({
                    "history_len": case["history_len"],
                    "draft_len": case["draft_len"],
                    "acceptance_case": case["acceptance_case"],
                }),
            }
            for case in CASE_MATRIX
        },
        "artifact_hashes": {},
    }


def _capability_tolerance(dtype_name: str) -> dict:
    return DTYPE_TOLERANCES[dtype_name]


def _run_capability_spec(spec: dict) -> dict:
    import torch
    from flash_attn import flash_attn_with_kvcache

    dtype = getattr(torch, spec["dtype"].split(".")[-1])
    query_len = int(spec["query_len"])
    block_size = 256
    prefix_len = (
        32
        if spec["block_case"] == "one_block"
        else block_size - 2
    )
    total_len = prefix_len + query_len
    num_blocks = (total_len + block_size - 1) // block_size
    num_heads = 4
    num_kv_heads = 2
    head_dim = 32
    generator = torch.Generator(device="cuda")
    generator.manual_seed(
        20260716
        + query_len * 17
        + (1000 if dtype == torch.bfloat16 else 0)
        + (100 if spec["block_case"] == "cross_block" else 0)
    )
    q = torch.randn(
        1,
        query_len,
        num_heads,
        head_dim,
        device="cuda",
        dtype=dtype,
        generator=generator,
    )
    dense_k = torch.randn(
        total_len,
        num_kv_heads,
        head_dim,
        device="cuda",
        dtype=dtype,
        generator=generator,
    )
    dense_v = torch.randn(
        total_len,
        num_kv_heads,
        head_dim,
        device="cuda",
        dtype=dtype,
        generator=generator,
    )
    k_cache = torch.zeros(
        num_blocks,
        block_size,
        num_kv_heads,
        head_dim,
        device="cuda",
        dtype=dtype,
    )
    v_cache = torch.zeros_like(k_cache)
    for position in range(total_len):
        block_id = position // block_size
        offset = position % block_size
        k_cache[block_id, offset].copy_(dense_k[position])
        v_cache[block_id, offset].copy_(dense_v[position])
    block_table = torch.arange(
        num_blocks,
        device="cuda",
        dtype=torch.int32,
    ).unsqueeze(0)
    scale = head_dim ** -0.5
    native = flash_attn_with_kvcache(
        q,
        k_cache,
        v_cache,
        cache_seqlens=torch.tensor(
            [total_len],
            device="cuda",
            dtype=torch.int32,
        ),
        block_table=block_table,
        softmax_scale=scale,
        causal=True,
    )
    oracle_rows = []
    for query_index in range(query_len):
        oracle_rows.append(
            flash_attn_with_kvcache(
                q[:, query_index:query_index + 1],
                k_cache,
                v_cache,
                cache_seqlens=torch.tensor(
                    [prefix_len + query_index + 1],
                    device="cuda",
                    dtype=torch.int32,
                ),
                block_table=block_table,
                softmax_scale=scale,
                causal=True,
            )
        )
    oracle = torch.cat(oracle_rows, dim=1)
    tolerance = _capability_tolerance(spec["dtype"])
    output_match = torch.allclose(
        native,
        oracle,
        rtol=tolerance["logits_rtol"],
        atol=tolerance["logits_atol"],
    )
    max_output_abs_error = float(
        (native.float() - oracle.float()).abs().max().item()
    )

    future_row_masked = True
    if query_len > 1:
        perturbed_k = k_cache.clone()
        perturbed_v = v_cache.clone()
        final_position = total_len - 1
        final_block = final_position // block_size
        final_offset = final_position % block_size
        perturbed_k[final_block, final_offset].add_(7)
        perturbed_v[final_block, final_offset].sub_(5)
        perturbed = flash_attn_with_kvcache(
            q,
            perturbed_k,
            perturbed_v,
            cache_seqlens=torch.tensor(
                [total_len],
                device="cuda",
                dtype=torch.int32,
            ),
            block_table=block_table,
            softmax_scale=scale,
            causal=True,
        )
        future_row_masked = torch.allclose(
            native[:, :-1],
            perturbed[:, :-1],
            rtol=tolerance["logits_rtol"],
            atol=tolerance["logits_atol"],
        )
    kv_match = (
        torch.equal(
            k_cache.reshape(-1, num_kv_heads, head_dim)[:total_len],
            dense_k,
        )
        and torch.equal(
            v_cache.reshape(-1, num_kv_heads, head_dim)[:total_len],
            dense_v,
        )
    )
    finite = bool(
        torch.isfinite(native).all()
        and torch.isfinite(oracle).all()
        and torch.isfinite(k_cache).all()
        and torch.isfinite(v_cache).all()
    )
    return {
        **spec,
        "output_match": bool(output_match),
        "kv_match": bool(kv_match),
        "future_row_masked": bool(future_row_masked),
        "finite": finite,
        "max_output_abs_error": max_output_abs_error,
        "tolerance": tolerance,
    }


def run_capability_matrix(out_path: Path) -> dict:
    import torch

    rows = []
    errors = []
    bf16_supported = bool(torch.cuda.is_bf16_supported())
    for spec in build_capability_specs(bf16_supported):
        try:
            rows.append(_run_capability_spec(spec))
        except Exception as exc:
            errors.append({
                **spec,
                "error_type": type(exc).__name__,
                "error": str(exc),
            })
    status = "PASS"
    if errors or any(
        not all(
            row[field]
            for field in (
                "output_match",
                "kv_match",
                "future_row_masked",
                "finite",
            )
        )
        for row in rows
    ):
        status = "INCOMPLETE"
    payload = {
        "status": status,
        "rows": rows,
        "errors": errors,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "gpu_name": torch.cuda.get_device_name(0),
        "bf16_supported": bf16_supported,
    }
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )
    return payload


def _row_key(row: dict) -> tuple[str, str]:
    return str(row.get("case_id")), str(row.get("policy"))


def _expected_row_keys() -> set[tuple[str, str]]:
    return {
        (case["case_id"], policy)
        for case in CASE_MATRIX
        for policy in POLICIES
    }


def _required_comparison_fields() -> tuple[str, ...]:
    return (
        "status",
        "target_token_match",
        "accepted_prefix_match",
        "metadata_match",
        "continuation_token_match",
        "continuation_steps",
        "finite",
        "max_logit_abs_error",
        "max_kv_abs_error",
        "logits_within_tolerance",
        "kv_within_tolerance",
    )


def _median(values) -> float:
    values = [float(value) for value in values]
    if not values:
        raise ValueError("median requires evidence")
    return float(statistics.median(values))


def _allocate_port_pair() -> tuple[int, int]:
    sockets = []
    ports = []
    try:
        for _ in range(2):
            handle = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            handle.bind(("127.0.0.1", 0))
            sockets.append(handle)
            ports.append(int(handle.getsockname()[1]))
    finally:
        for handle in sockets:
            handle.close()
    return ports[0], ports[1]


def _case_process(
    *,
    python_bin: str,
    model_path: str,
    policy: str,
    case: dict,
    out_path: Path,
    log_dir: Path,
) -> tuple[dict | None, dict]:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    case_path = out_path.with_suffix(".case.json")
    case_path.write_text(
        json.dumps(case, indent=2, sort_keys=True) + "\n"
    )
    last_process = None
    for attempt in range(1, 4):
        dist_port, master_port = _allocate_port_pair()
        stdout_path = log_dir / (
            f"{case['case_id']}.{policy}.attempt{attempt}.stdout.log"
        )
        stderr_path = log_dir / (
            f"{case['case_id']}.{policy}.attempt{attempt}.stderr.log"
        )
        command = [
            str(python_bin),
            str(Path(__file__).with_name("native_verifier_oracle.py")),
            "run-case",
            "--policy",
            policy,
            "--case-json",
            str(case_path),
            "--out",
            str(out_path),
            "--model",
            str(model_path),
            "--continuation-steps",
            str(case["continuation_steps"]),
        ]
        environment = os.environ.copy()
        environment["TINYVLLM_DIST_PORT"] = str(dist_port)
        environment["MASTER_PORT"] = str(master_port)
        t0 = time.perf_counter()
        completed = subprocess.run(
            command,
            text=True,
            capture_output=True,
            env=environment,
        )
        elapsed_s = time.perf_counter() - t0
        stdout_path.write_text(completed.stdout)
        stderr_path.write_text(completed.stderr)
        last_process = {
            "returncode": int(completed.returncode),
            "command": command,
            "tinyvllm_dist_port": dist_port,
            "master_port": master_port,
            "stdout_path": str(stdout_path),
            "stderr_path": str(stderr_path),
            "elapsed_s": elapsed_s,
            "attempt": attempt,
        }
        combined = completed.stdout + "\n" + completed.stderr
        retryable = completed.returncode != 0 and (
            "EADDRINUSE" in combined
            or "address already in use" in combined.lower()
        )
        if completed.returncode == 0:
            payload = json.loads(out_path.read_text())
            return payload, last_process
        if not retryable:
            break
    return None, last_process or {
        "returncode": 1,
        "tinyvllm_dist_port": -1,
        "master_port": -1,
    }


def _materialize_case(
    case_spec: dict,
    probe: dict,
    source_commit: str,
    source_dirty: bool,
) -> dict:
    from native_verifier_oracle import construct_draft_tokens

    targets = [int(token_id) for token_id in probe["target_tokens"]]
    history_len = int(case_spec["history_len"])
    draft_tokens = construct_draft_tokens(
        targets,
        acceptance_case=case_spec["acceptance_case"],
        vocab_size=int(probe["vocab_size"]),
    )
    prompt_tokens = int(probe["prompt_token_count"])
    if case_spec["eos_case"]:
        eos_token_id = int(probe["eos_token_id"])
        history_tokens = [
            int(token_id) for token_id in probe["history_tokens"]
        ]
        draft_len = int(case_spec["draft_len"])
        eos_indices = [
            index
            for index, token_id in enumerate(history_tokens)
            if token_id == eos_token_id
            and index - draft_len + 1 >= prompt_tokens
        ]
        if not eos_indices:
            raise ValueError(
                f"{case_spec['case_id']} probe history has no usable real EOS"
            )
        eos_index = eos_indices[-1]
        history_len = eos_index - draft_len + 1
        draft_tokens = history_tokens[
            history_len:history_len + draft_len
        ]
    completion_at_history = history_len - prompt_tokens
    max_tokens = (
        completion_at_history + 2
        if case_spec["output_budget_case"]
        else completion_at_history
        + len(draft_tokens)
        + int(case_spec["continuation_steps"])
        + 4
    )
    return {
        **case_spec,
        "history_len": history_len,
        "draft_tokens": draft_tokens,
        "max_tokens": max_tokens,
        "ignore_eos": not bool(case_spec["eos_case"]),
        "source_commit": source_commit,
        "source_dirty": source_dirty,
    }


def _normalize_case_row(
    payload: dict | None,
    process: dict,
    case_id: str,
    policy: str,
    source_commit: str,
    source_dirty: bool,
) -> dict:
    if payload is None:
        return {
            "case_id": case_id,
            "policy": policy,
            "status": "INCOMPLETE",
            "source_commit": source_commit,
            "source_dirty": source_dirty,
            "process": process,
        }
    row = dict(payload)
    row["process"] = process
    row["source_commit"] = source_commit
    row["source_dirty"] = source_dirty
    return row


def _event_from_rows(
    native_row: dict,
    legacy_row: dict,
) -> dict | None:
    native_event = native_row.get("event")
    legacy_event = legacy_row.get("event")
    if not isinstance(native_event, dict) or not isinstance(
        legacy_event,
        dict,
    ):
        return None
    return {
        "case_id": native_row["case_id"],
        "policy": "native",
        "draft_len": int(native_event["draft_len"]),
        "accepted_count": int(native_event["accepted_count"]),
        "zero_accept_included_in_throughput": True,
        "accepted_kv_rematerialization": native_event[
            "accepted_kv_rematerialization"
        ],
        "accepted_kv_copy_calls": native_event[
            "accepted_kv_copy_calls"
        ],
        "accepted_kv_replay_calls": native_event[
            "accepted_kv_replay_calls"
        ],
        "target_forward_count": native_event["target_forward_count"],
        "legacy_decode_replay_calls": legacy_event[
            "accepted_kv_rematerialization"
        ]["decode_calls"],
        "legacy_total_target_forwards": (
            int(legacy_event["target_forward_count"])
            + int(
                legacy_event["accepted_kv_rematerialization"][
                    "decode_calls"
                ]
            )
        ),
        "native_total_target_forwards": int(
            native_event["target_forward_count"]
        ),
        "verifier_commit_ms": float(
            native_event["timing_ms"]["verify_commit_total_ms"]
        ),
        "legacy_verifier_commit_ms": float(
            legacy_event["timing_ms"]["verify_commit_total_ms"]
        ),
        "eos_truncated": bool(native_event["eos_truncated"]),
        "output_budget_truncated": bool(
            native_event["output_budget_truncated"]
        ),
    }


def run_gate(
    *,
    out_dir: Path,
    python_bin: str,
    model_path: str,
    source_commit: str,
    source_dirty: bool,
    host: str,
    run_tag: str,
    preflight_path: Path,
) -> dict:
    from native_verifier_oracle import compare_native_and_oracle

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_dir = out_dir / "logs"
    raw_dir = out_dir / "raw"
    preflight = json.loads(Path(preflight_path).read_text())
    capability_path = out_dir / "capability.json"
    capability = json.loads(capability_path.read_text())
    manifest = build_manifest(
        source_commit=source_commit,
        source_dirty=source_dirty,
        model_path=model_path,
        model_identifier=preflight["model_identifier"],
        host=host,
        python_bin=python_bin,
        torch_version=preflight["torch"],
        cuda_version=preflight["cuda"],
        flash_attn_version=preflight["flash_attn"],
        gpu_name=preflight["gpu"],
        bf16_supported=preflight["bf16_supported"],
        run_tag=run_tag,
    )
    case_rows = []
    event_rows = []
    for case_spec in CASE_MATRIX:
        probe_case = {
            **case_spec,
            "draft_tokens": [0] * int(case_spec["draft_len"]),
            "max_tokens": 2048,
            "ignore_eos": True,
            "source_commit": source_commit,
            "source_dirty": source_dirty,
        }
        probe, probe_process = _case_process(
            python_bin=python_bin,
            model_path=model_path,
            policy="probe",
            case=probe_case,
            out_path=raw_dir / f"{case_spec['case_id']}.probe.json",
            log_dir=log_dir,
        )
        manifest["process_port_pairs"].append({
            "case_id": case_spec["case_id"],
            "policy": "probe",
            "tinyvllm_dist_port": probe_process[
                "tinyvllm_dist_port"
            ],
            "master_port": probe_process["master_port"],
        })
        try:
            if probe is None:
                raise RuntimeError("probe process failed")
            case = _materialize_case(
                case_spec,
                probe,
                source_commit,
                source_dirty,
            )
        except Exception as exc:
            for policy in POLICIES:
                case_rows.append({
                    "case_id": case_spec["case_id"],
                    "policy": policy,
                    "status": "INCOMPLETE",
                    "source_commit": source_commit,
                    "source_dirty": source_dirty,
                    "reason": str(exc),
                    "process": probe_process,
                })
            continue

        rows_by_policy = {}
        for policy in POLICIES:
            payload, process = _case_process(
                python_bin=python_bin,
                model_path=model_path,
                policy=policy,
                case=case,
                out_path=raw_dir / (
                    f"{case_spec['case_id']}.{policy}.json"
                ),
                log_dir=log_dir,
            )
            manifest["process_port_pairs"].append({
                "case_id": case_spec["case_id"],
                "policy": policy,
                "tinyvllm_dist_port": process[
                    "tinyvllm_dist_port"
                ],
                "master_port": process["master_port"],
            })
            row = _normalize_case_row(
                payload,
                process,
                case_spec["case_id"],
                policy,
                source_commit,
                source_dirty,
            )
            rows_by_policy[policy] = row
            case_rows.append(row)
        if (
            rows_by_policy["native"].get("status") == "PASS"
            and rows_by_policy["oracle"].get("status") == "PASS"
        ):
            rows_by_policy["oracle"]["comparison"] = (
                compare_native_and_oracle(
                    rows_by_policy["native"],
                    rows_by_policy["oracle"],
                )
            )
        event = _event_from_rows(
            rows_by_policy["native"],
            rows_by_policy["legacy_rematerialize"],
        )
        if event is not None:
            event_rows.append(event)

    summary = classify_gate(
        manifest,
        capability,
        case_rows,
        event_rows,
    )
    report = render_report(
        manifest,
        capability,
        case_rows,
        event_rows,
        summary,
    )
    payloads = {
        "capability.json": capability,
        "case_rows.json": case_rows,
        "event_rows.json": event_rows,
        "summary.json": summary,
    }
    for name, payload in payloads.items():
        (out_dir / name).write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n"
        )
    (out_dir / "report.md").write_text(report)
    manifest["artifact_hashes"] = {
        name: sha256_file(out_dir / name)
        for name in (
            "capability.json",
            "case_rows.json",
            "event_rows.json",
            "summary.json",
            "report.md",
        )
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    return summary


def _incomplete(reasons: list[str], **extra) -> dict:
    return {
        "classification": "INCOMPLETE",
        "reasons": reasons,
        "exactness_pass": False,
        "replay_elimination_pass": False,
        "performance_direction_pass": False,
        "memory_is_diagnostic_only": True,
        **extra,
    }


def _no_go(reasons: list[str], **extra) -> dict:
    return {
        "classification": "NO_GO",
        "reasons": reasons,
        "exactness_pass": not any(
            "mismatch" in reason
            or "exactness" in reason
            or "non-finite" in reason
            for reason in reasons
        ),
        "replay_elimination_pass": not any(
            term in reason for reason in reasons
            for term in ("replay", "copy", "rematerialization")
        ),
        "performance_direction_pass": False,
        "memory_is_diagnostic_only": True,
        **extra,
    }


def classify_gate(
    manifest: dict,
    capability: dict,
    case_rows: list[dict],
    event_rows: list[dict],
) -> dict:
    structural = []
    if manifest.get("source_dirty") is not False:
        structural.append("canonical source_dirty must be false")
    if manifest.get("source_commit") in (None, ""):
        structural.append("source commit is missing")

    observed_keys = [_row_key(row) for row in case_rows]
    expected_keys = _expected_row_keys()
    if len(observed_keys) != len(set(observed_keys)):
        structural.append("duplicate policy/case rows")
    missing_keys = sorted(expected_keys - set(observed_keys))
    extra_keys = sorted(set(observed_keys) - expected_keys)
    if missing_keys:
        structural.append(f"missing policy/case rows: {missing_keys}")
    if extra_keys:
        structural.append(f"unexpected policy/case rows: {extra_keys}")
    for row in case_rows:
        process = row.get("process")
        if not isinstance(process, dict):
            structural.append(f"{_row_key(row)} missing process evidence")
            continue
        if process.get("returncode") != 0:
            structural.append(f"{_row_key(row)} process failed")
        for field in ("tinyvllm_dist_port", "master_port"):
            if not isinstance(process.get(field), int):
                structural.append(
                    f"{_row_key(row)} missing dynamic {field}"
                )
        if row.get("source_commit") != manifest.get("source_commit"):
            structural.append(f"{_row_key(row)} source commit mismatch")
        if row.get("source_dirty") is not False:
            structural.append(f"{_row_key(row)} source_dirty evidence")

    native_events = [
        event for event in event_rows
        if event.get("policy") == "native"
    ]
    native_event_ids = [
        str(event.get("case_id")) for event in native_events
    ]
    expected_case_ids = {case["case_id"] for case in CASE_MATRIX}
    if len(native_event_ids) != len(set(native_event_ids)):
        structural.append("duplicate native event rows")
    if set(native_event_ids) != expected_case_ids:
        structural.append("missing or unexpected native event rows")
    for event in native_events:
        for field in (
            "accepted_kv_rematerialization",
            "accepted_kv_copy_calls",
            "accepted_kv_replay_calls",
            "target_forward_count",
            "legacy_decode_replay_calls",
            "legacy_total_target_forwards",
            "native_total_target_forwards",
            "verifier_commit_ms",
            "legacy_verifier_commit_ms",
        ):
            if field not in event:
                structural.append(
                    f"{event.get('case_id')} native event missing {field}"
                )
        rematerialization = event.get(
            "accepted_kv_rematerialization",
            {},
        )
        for field in (
            "decode_calls",
            "rematerialized_tokens",
            "elapsed_ms",
        ):
            if field not in rematerialization:
                structural.append(
                    f"{event.get('case_id')} rematerialization missing {field}"
                )
    if structural:
        return _incomplete(structural)

    if capability.get("status") != "PASS":
        return _incomplete(["FlashAttention capability is unavailable"])
    capability_rows = capability.get("rows")
    if not isinstance(capability_rows, list):
        return _incomplete(["capability rows are missing"])
    required_dtypes = {"torch.float16"}
    if manifest.get("bf16_supported"):
        required_dtypes.add("torch.bfloat16")
    required_capability = {
        (dtype, query_len, block_case)
        for dtype in required_dtypes
        for query_len in (1, 3, 7, 15)
        for block_case in ("one_block", "cross_block")
    }
    observed_capability = {
        (
            row.get("dtype"),
            row.get("query_len"),
            row.get("block_case"),
        )
        for row in capability_rows
        if all(
            row.get(field) is True
            for field in (
                "gqa",
                "output_match",
                "kv_match",
                "future_row_masked",
                "finite",
            )
        )
    }
    if not required_capability <= observed_capability:
        return _incomplete(["capability matrix is incomplete or divergent"])

    by_key = {_row_key(row): row for row in case_rows}
    semantic_reasons = []
    for case in CASE_MATRIX:
        case_id = case["case_id"]
        rows = {
            policy: by_key[(case_id, policy)]
            for policy in POLICIES
        }
        for policy, row in rows.items():
            if row.get("status") == "INCOMPLETE":
                return _incomplete(
                    [f"{case_id}/{policy} row is INCOMPLETE"]
                )
            if row.get("status") != "PASS":
                semantic_reasons.append(
                    f"{case_id}/{policy} semantic failure"
                )
        comparison = rows["oracle"].get("comparison")
        if not isinstance(comparison, dict):
            return _incomplete([f"{case_id} comparison is missing"])
        missing_comparison = [
            field for field in _required_comparison_fields()
            if field not in comparison
        ]
        if missing_comparison:
            return _incomplete([
                f"{case_id} comparison missing {missing_comparison}"
            ])
        if comparison["status"] == "INCOMPLETE":
            return _incomplete(
                [f"{case_id} comparison is INCOMPLETE"]
            )
        if comparison["status"] != "PASS":
            semantic_reasons.extend(
                comparison.get("reasons")
                or [f"{case_id} native/oracle exactness failure"]
            )
        for field in (
            "target_token_match",
            "accepted_prefix_match",
            "metadata_match",
            "continuation_token_match",
            "finite",
            "logits_within_tolerance",
            "kv_within_tolerance",
        ):
            if comparison.get(field) is not True:
                semantic_reasons.append(
                    f"{case_id} comparison {field} mismatch"
                )
        if comparison["continuation_steps"] < THRESHOLDS[
            "min_continuation_steps"
        ]:
            semantic_reasons.append(
                f"{case_id} continuation exactness below 16"
            )

        baseline_hash = rows["baseline"].get("output_token_sha256")
        continuation_hash = rows["baseline"].get(
            "continuation_token_sha256"
        )
        if not baseline_hash or not continuation_hash:
            return _incomplete(
                [f"{case_id} baseline token hashes are missing"]
            )
        for policy in ("legacy_rematerialize", "native", "oracle"):
            if rows[policy].get("output_token_sha256") != baseline_hash:
                semantic_reasons.append(
                    f"{case_id}/{policy} output token mismatch"
                )
            if (
                rows[policy].get("continuation_token_sha256")
                != continuation_hash
            ):
                semantic_reasons.append(
                    f"{case_id}/{policy} continuation token mismatch"
                )
        if (
            rows["native"].get("accepted_tokens")
            != rows["oracle"].get("accepted_tokens")
        ):
            semantic_reasons.append(
                f"{case_id} accepted prefix mismatch"
            )
        if (
            rows["native"].get("sequence_token_sha256")
            != rows["oracle"].get("sequence_token_sha256")
            or rows["native"].get("block_table_after")
            != rows["oracle"].get("block_table_after")
        ):
            semantic_reasons.append(
                f"{case_id} committed lifecycle mismatch"
            )
    if semantic_reasons:
        return _no_go(sorted(set(semantic_reasons)))

    replay_reasons = []
    for event in native_events:
        case_id = event["case_id"]
        rematerialization = event["accepted_kv_rematerialization"]
        if (
            rematerialization["decode_calls"] != 0
            or rematerialization["rematerialized_tokens"]
            or float(rematerialization["elapsed_ms"]) != 0.0
        ):
            replay_reasons.append(
                f"{case_id} accepted KV rematerialization remains"
            )
        if event["accepted_kv_copy_calls"] != 0:
            replay_reasons.append(f"{case_id} accepted KV copy remains")
        if event["accepted_kv_replay_calls"] != 0:
            replay_reasons.append(f"{case_id} accepted KV replay remains")
        removed = (
            int(event["legacy_total_target_forwards"])
            - int(event["native_total_target_forwards"])
        )
        if removed != int(event["legacy_decode_replay_calls"]):
            replay_reasons.append(
                f"{case_id} target forward reduction does not equal "
                "removed replay calls"
            )
        if (
            int(event["accepted_count"]) == 0
            and event.get("zero_accept_included_in_throughput") is not True
        ):
            replay_reasons.append(
                f"{case_id} zero-accept event excluded from throughput"
            )
    if replay_reasons:
        return _no_go(replay_reasons)

    performance_missing = []
    for row in case_rows:
        for field in (
            "elapsed_s",
            "output_tokens",
            "output_tokens_per_s",
            "max_allocated_gpu_memory_bytes",
        ):
            value = row.get(field)
            if not isinstance(value, (int, float)) or not math.isfinite(
                float(value)
            ):
                performance_missing.append(
                    f"{_row_key(row)} missing performance {field}"
                )
    for event in native_events:
        for field in (
            "verifier_commit_ms",
            "legacy_verifier_commit_ms",
        ):
            value = event.get(field)
            if not isinstance(value, (int, float)) or not math.isfinite(
                float(value)
            ):
                performance_missing.append(
                    f"{event['case_id']} missing {field}"
                )
    if performance_missing:
        return _incomplete(performance_missing)

    performance_reasons = []
    k1_ratios = []
    for case in CASE_MATRIX:
        if case["draft_len"] != 1:
            continue
        baseline = by_key[(case["case_id"], "baseline")]
        native = by_key[(case["case_id"], "native")]
        k1_ratios.append(
            float(native["elapsed_s"]) / float(baseline["elapsed_s"])
        )
    k1_regression = max(k1_ratios) - 1.0
    if k1_regression > THRESHOLDS["k1_max_regression_fraction"]:
        performance_reasons.append(
            f"K=1 native regression {k1_regression:.6f} exceeds 1%"
        )

    accepted_k_gt_1 = [
        event for event in native_events
        if int(event["draft_len"]) > 1
        and int(event["accepted_count"]) > 0
    ]
    if not accepted_k_gt_1:
        return _incomplete(
            ["no accepted K>1 performance evidence"]
        )
    native_median = _median(
        event["verifier_commit_ms"]
        for event in accepted_k_gt_1
    )
    legacy_median = _median(
        event["legacy_verifier_commit_ms"]
        for event in accepted_k_gt_1
    )
    if native_median >= legacy_median:
        performance_reasons.append(
            "accepted K>1 native verifier-plus-commit median "
            "does not beat legacy"
        )
    if performance_reasons:
        return _no_go(
            performance_reasons,
            exactness_pass=True,
            replay_elimination_pass=True,
            k1_regression_fraction=k1_regression,
            native_k_gt_1_median_ms=native_median,
            legacy_k_gt_1_median_ms=legacy_median,
        )

    return {
        "classification": "READY_FOR_PERFORMANCE_GATE",
        "reasons": [],
        "exactness_pass": True,
        "replay_elimination_pass": True,
        "performance_direction_pass": True,
        "memory_is_diagnostic_only": True,
        "observed_case_rows": len(case_rows),
        "observed_native_events": len(native_events),
        "k1_regression_fraction": k1_regression,
        "native_k_gt_1_median_ms": native_median,
        "legacy_k_gt_1_median_ms": legacy_median,
        "max_allocated_gpu_memory_bytes": max(
            int(row["max_allocated_gpu_memory_bytes"])
            for row in case_rows
        ),
    }


def render_report(
    manifest: dict,
    capability: dict,
    case_rows: list[dict],
    event_rows: list[dict],
    summary: dict,
) -> str:
    lines = [
        "# Native Multi-Token Verifier Gate",
        "",
        "## Environment",
        "",
        "| Field | Value |",
        "| --- | --- |",
    ]
    for field in (
        "run_tag",
        "source_commit",
        "source_dirty",
        "model_identifier",
        "model_path",
        "host",
        "python_bin",
        "torch_version",
        "cuda_version",
        "flash_attn_version",
        "gpu_name",
    ):
        lines.append(f"| {field} | {manifest.get(field)} |")
    lines.extend([
        "",
        "## Capability",
        "",
        f"- Status: `{capability.get('status')}`",
        f"- Rows: `{len(capability.get('rows', []))}`",
        "- Required query lengths: `Q in {1,3,7,15}`",
        "",
        "## Exactness",
        "",
        "| Case | Oracle | Continuation | Max Logit Error | Max KV Error |",
        "| --- | --- | ---: | ---: | ---: |",
    ])
    oracle_rows = {
        row["case_id"]: row
        for row in case_rows
        if row.get("policy") == "oracle"
    }
    for case in CASE_MATRIX:
        comparison = oracle_rows.get(
            case["case_id"],
            {},
        ).get("comparison", {})
        lines.append(
            "| {case} | {status} | {steps} | {logit} | {kv} |".format(
                case=case["case_id"],
                status=comparison.get("status", "missing"),
                steps=comparison.get("continuation_steps", "missing"),
                logit=comparison.get("max_logit_abs_error", "missing"),
                kv=comparison.get("max_kv_abs_error", "missing"),
            )
        )
    lines.extend([
        "",
        "## Performance Direction",
        "",
        f"- K=1 regression fraction: `{summary.get('k1_regression_fraction')}`",
        "- Native accepted K>1 verifier-plus-commit median ms: "
        f"`{summary.get('native_k_gt_1_median_ms')}`",
        "- Legacy accepted K>1 verifier-plus-commit median ms: "
        f"`{summary.get('legacy_k_gt_1_median_ms')}`",
        "- Zero-accept events are included in end-to-end throughput.",
        "- Maximum allocated GPU memory is diagnostic only: "
        f"`{summary.get('max_allocated_gpu_memory_bytes')}` bytes.",
        "",
        "## Classification",
        "",
        f"Classification: `{summary['classification']}`",
        "",
        "Reasons:",
    ])
    if summary.get("reasons"):
        lines.extend(f"- {reason}" for reason in summary["reasons"])
    else:
        lines.append("- none")
    lines.extend([
        "",
        "## Non-Claims",
        "",
    ])
    lines.extend(f"- {boundary}" for boundary in CLAIM_BOUNDARIES)
    return "\n".join(lines) + "\n"


def _validate_manifest(manifest: dict) -> None:
    expected = {
        "dtype_tolerances": DTYPE_TOLERANCES,
        "thresholds": THRESHOLDS,
        "case_matrix": list(CASE_MATRIX),
        "case_matrix_sha256": sha256_json(CASE_MATRIX),
        "policies": list(POLICIES),
        "required_artifacts": list(REQUIRED_ARTIFACTS),
        "claim_boundaries": list(CLAIM_BOUNDARIES),
        "classification_on_success": "READY_FOR_PERFORMANCE_GATE",
    }
    for field, value in expected.items():
        if manifest.get(field) != value:
            raise ValueError(f"manifest {field} drift")
    if manifest.get("source_dirty") is not False:
        raise ValueError("manifest source_dirty must be false")


def verify_artifacts(out_dir: Path) -> dict:
    out_dir = Path(out_dir)
    for name in REQUIRED_ARTIFACTS:
        if not (out_dir / name).is_file():
            raise ValueError(f"missing artifact: {name}")
    manifest = json.loads((out_dir / "manifest.json").read_text())
    _validate_manifest(manifest)
    artifact_hashes = manifest.get("artifact_hashes")
    if not isinstance(artifact_hashes, dict):
        raise ValueError("manifest artifact_hashes are missing")
    for name in REQUIRED_ARTIFACTS:
        if name == "manifest.json":
            continue
        expected_hash = artifact_hashes.get(name)
        if not expected_hash:
            raise ValueError(f"missing SHA-256 for {name}")
        actual_hash = sha256_file(out_dir / name)
        if actual_hash != expected_hash:
            raise ValueError(f"SHA-256 mismatch for {name}")

    capability = json.loads((out_dir / "capability.json").read_text())
    case_rows = json.loads((out_dir / "case_rows.json").read_text())
    event_rows = json.loads((out_dir / "event_rows.json").read_text())
    recorded_summary = json.loads((out_dir / "summary.json").read_text())
    computed_summary = classify_gate(
        manifest,
        capability,
        case_rows,
        event_rows,
    )
    if recorded_summary != computed_summary:
        raise ValueError("summary.json differs from recomputed gate")
    expected_report = render_report(
        manifest,
        capability,
        case_rows,
        event_rows,
        computed_summary,
    )
    if (out_dir / "report.md").read_text() != expected_report:
        raise ValueError("report classification or contents differ")
    return computed_summary


def _parse_args():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--out-dir", required=True)
    run_parser.add_argument("--python-bin", required=True)
    run_parser.add_argument("--model-path", required=True)
    run_parser.add_argument("--source-commit", required=True)
    run_parser.add_argument("--source-dirty", action="store_true")
    run_parser.add_argument("--host", required=True)
    run_parser.add_argument("--run-tag", required=True)
    run_parser.add_argument("--preflight", required=True)
    capability_parser = subparsers.add_parser("capability")
    capability_parser.add_argument("--out", required=True)
    verify_parser = subparsers.add_parser("verify")
    verify_parser.add_argument("--out-dir", required=True)
    report_parser = subparsers.add_parser("render-report")
    report_parser.add_argument("--out-dir", required=True)
    return parser.parse_args()


def main():
    args = _parse_args()
    if args.command == "capability":
        result = run_capability_matrix(Path(args.out))
        print(json.dumps(result, indent=2, sort_keys=True))
        if result["status"] != "PASS":
            raise SystemExit(2)
        return
    out_dir = Path(args.out_dir)
    if args.command == "run":
        result = run_gate(
            out_dir=out_dir,
            python_bin=args.python_bin,
            model_path=args.model_path,
            source_commit=args.source_commit,
            source_dirty=args.source_dirty,
            host=args.host,
            run_tag=args.run_tag,
            preflight_path=Path(args.preflight),
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        if result["classification"] == "INCOMPLETE":
            raise SystemExit(2)
        if result["classification"] == "NO_GO":
            raise SystemExit(3)
        return
    if args.command == "verify":
        result = verify_artifacts(out_dir)
        print(json.dumps(result, indent=2, sort_keys=True))
        return
    if args.command == "render-report":
        manifest = json.loads((out_dir / "manifest.json").read_text())
        capability = json.loads((out_dir / "capability.json").read_text())
        case_rows = json.loads((out_dir / "case_rows.json").read_text())
        event_rows = json.loads((out_dir / "event_rows.json").read_text())
        summary = classify_gate(
            manifest,
            capability,
            case_rows,
            event_rows,
        )
        print(
            render_report(
                manifest,
                capability,
                case_rows,
                event_rows,
                summary,
            ),
            end="",
        )
        return
    raise AssertionError(args.command)


if __name__ == "__main__":
    main()
