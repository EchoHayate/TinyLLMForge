"""Strict comparison helpers for native and serialized-decode verifier runs."""

from __future__ import annotations

import argparse
import atexit
import json
import math
import os
import sys
from copy import copy
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable


@dataclass(frozen=True)
class DTypeTolerance:
    logits_rtol: float
    logits_atol: float
    kv_rtol: float
    kv_atol: float


_TOLERANCES = {
    "torch.float16": DTypeTolerance(
        logits_rtol=2e-3,
        logits_atol=2e-3,
        kv_rtol=2e-3,
        kv_atol=2e-3,
    ),
    "torch.bfloat16": DTypeTolerance(
        logits_rtol=8e-3,
        logits_atol=8e-3,
        kv_rtol=8e-3,
        kv_atol=8e-3,
    ),
}

_REQUIRED_FIELDS = (
    "dtype",
    "target_tokens",
    "accepted_tokens",
    "sequence_tokens_after",
    "block_table_after",
    "continuation_tokens",
    "logits",
    "kv",
    "continuation_logits",
    "continuation_kv",
    "finite",
)


def dtype_tolerance(dtype_name: str) -> DTypeTolerance:
    try:
        return _TOLERANCES[str(dtype_name)]
    except KeyError as exc:
        raise ValueError(
            f"unsupported verifier comparison dtype: {dtype_name}"
        ) from exc


def _flatten_numbers(value) -> list[float]:
    if isinstance(value, dict):
        flattened = []
        for key in sorted(value):
            flattened.extend(_flatten_numbers(value[key]))
        return flattened
    if isinstance(value, (list, tuple)):
        flattened = []
        for item in value:
            flattened.extend(_flatten_numbers(item))
        return flattened
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("numeric comparison payload contains non-number")
    return [float(value)]


def _all_finite(value) -> bool:
    try:
        numbers = _flatten_numbers(value)
    except ValueError:
        return False
    return all(math.isfinite(number) for number in numbers)


def _validate_case(case: dict) -> dict:
    required = (
        "case_id",
        "prompt",
        "history_len",
        "draft_tokens",
        "max_tokens",
        "ignore_eos",
    )
    missing = [field for field in required if field not in case]
    if missing:
        raise ValueError(
            "oracle case is missing required field: " + ", ".join(missing)
        )
    normalized = dict(case)
    normalized["case_id"] = str(case["case_id"])
    normalized["prompt"] = str(case["prompt"])
    normalized["history_len"] = int(case["history_len"])
    normalized["draft_tokens"] = [
        int(token_id) for token_id in case["draft_tokens"]
    ]
    normalized["max_tokens"] = int(case["max_tokens"])
    normalized["ignore_eos"] = bool(case["ignore_eos"])
    if not normalized["case_id"]:
        raise ValueError("oracle case_id must not be empty")
    if normalized["history_len"] < 1:
        raise ValueError("oracle history_len must be positive")
    if not normalized["draft_tokens"]:
        raise ValueError("oracle draft_tokens must not be empty")
    if normalized["max_tokens"] < 1:
        raise ValueError("oracle max_tokens must be positive")
    return normalized


def build_case_payload(evidence: dict) -> dict:
    payload = dict(evidence)
    missing = [
        field for field in _REQUIRED_FIELDS
        if field != "finite" and field not in payload
    ]
    if missing:
        raise ValueError(
            "oracle evidence is missing required field: "
            + ", ".join(missing)
        )
    tolerance = dtype_tolerance(payload["dtype"])
    numeric_fields = (
        "logits",
        "kv",
        "continuation_logits",
        "continuation_kv",
    )
    payload["finite"] = all(
        _all_finite(payload.get(field, []))
        for field in numeric_fields
    )
    payload["tolerance"] = asdict(tolerance)
    return payload


def run_case(
    *,
    policy: str,
    case: dict,
    out_path: Path,
    model: str,
    continuation_steps: int,
    backend: Callable[..., dict] | None = None,
) -> dict:
    if policy not in ("native", "oracle"):
        raise ValueError(f"unsupported verifier oracle policy: {policy}")
    if continuation_steps < 16:
        raise ValueError("continuation_steps must be at least 16")
    normalized_case = _validate_case(case)
    if backend is None:
        backend = _run_tinyvllm_case
    payload = backend(
        policy=policy,
        case=normalized_case,
        model=str(model),
        continuation_steps=int(continuation_steps),
    )
    required_identity = {
        "policy": policy,
        "case_id": normalized_case["case_id"],
    }
    for field, expected in required_identity.items():
        if payload.get(field) != expected:
            raise ValueError(
                f"oracle backend returned invalid {field}: "
                f"expected {expected!r}, got {payload.get(field)!r}"
            )
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )
    return payload


def _load_profile_module():
    tools_dir = os.path.dirname(os.path.abspath(__file__))
    if tools_dir not in sys.path:
        sys.path.insert(0, tools_dir)
    import profile_ngram_commit

    return profile_ngram_commit


def _physical_slot(seq, logical_position: int) -> int:
    block_index = int(logical_position) // int(seq.block_size)
    offset = int(logical_position) % int(seq.block_size)
    return int(seq.block_table[block_index]) * int(seq.block_size) + offset


def _tensor_to_float_list(tensor):
    return tensor.detach().to(device="cpu").float().tolist()


def _snapshot_to_lists(snapshot: dict) -> dict:
    return {
        name: _tensor_to_float_list(tensor)
        for name, tensor in snapshot.items()
    }


def _append_kv_rows(total: dict[str, list], snapshot: dict) -> None:
    for name in ("keys", "values"):
        rows = snapshot.get(name)
        if rows is None:
            raise ValueError(f"KV snapshot is missing {name}")
        total[name].extend(rows)


def _run_decode_evidence_step(llm, seq) -> dict:
    from tinyvllm.utils.context import reset_context

    block_manager = llm.scheduler.block_manager
    if not block_manager.can_append(seq):
        raise RuntimeError("decode evidence step has no append capacity")
    block_manager.may_append(seq)
    logical_slot = len(seq) - 1
    physical_slot = _physical_slot(seq, logical_slot)
    try:
        input_ids, positions = llm.model_runner.prepare_decode([seq])
        logits = llm.model_runner.run_model(
            input_ids,
            positions,
            is_prefill=False,
            execution_mode="decode",
        )
        token_id = int(logits.argmax(dim=-1).tolist()[0])
        kv = llm.model_runner.snapshot_kv_slots([physical_slot])
        return {
            "token_id": token_id,
            "logits": _tensor_to_float_list(logits),
            "kv": _snapshot_to_lists(kv),
            "physical_slot": physical_slot,
        }
    finally:
        reset_context()


def _advance_to_history(llm, seq, history_len: int) -> None:
    while len(seq) < history_len:
        llm.step()
        if seq.is_finished and len(seq) < history_len:
            raise RuntimeError(
                "request finished before the requested verifier history"
            )
    if len(seq) != history_len:
        raise RuntimeError(
            f"history length overshot: expected {history_len}, got {len(seq)}"
        )


def _truncate_accepted_tokens(llm, seq, draft_tokens, target_tokens):
    profile = _load_profile_module()
    accepted_count = profile.count_accepted_prefix(
        draft_tokens,
        target_tokens,
    )
    accepted_tokens = list(draft_tokens[:accepted_count])
    if not seq.ignore_eos and llm.scheduler.eos in accepted_tokens:
        accepted_tokens = accepted_tokens[
            :accepted_tokens.index(llm.scheduler.eos) + 1
        ]
    remaining_budget = max(
        0,
        seq.max_tokens - seq.num_completion_tokens,
    )
    return accepted_tokens[:remaining_budget]


def _run_serialized_oracle_verify(llm, seq, draft_tokens: list[int]) -> dict:
    from tinyvllm.utils.context import reset_context

    block_manager = llm.scheduler.block_manager
    reserved_blocks = block_manager.reserve_append_blocks(
        seq,
        len(draft_tokens),
    )
    owned_blocks = list(reserved_blocks)
    try:
        first_target = int(
            llm.model_runner.run([seq], is_prefill=False)[0]
        )
        proxy = copy(seq)
        proxy.token_ids = list(seq.token_ids)
        proxy.block_table = list(seq.block_table) + list(reserved_blocks)
        proxy.num_tokens = len(proxy.token_ids)
        proxy.last_token = proxy.token_ids[-1]

        tail_targets = []
        logits_rows = []
        physical_slots = []
        kv_rows = {"keys": [], "values": []}
        for token_id in draft_tokens[:-1]:
            proxy.append_token(int(token_id))
            logical_slot = len(proxy) - 1
            physical_slot = _physical_slot(proxy, logical_slot)
            try:
                input_ids, positions = llm.model_runner.prepare_decode([proxy])
                logits = llm.model_runner.run_model(
                    input_ids,
                    positions,
                    is_prefill=False,
                    execution_mode="decode",
                )
                tail_targets.append(
                    int(logits.argmax(dim=-1).tolist()[0])
                )
                logits_rows.extend(_tensor_to_float_list(logits))
                snapshot = _snapshot_to_lists(
                    llm.model_runner.snapshot_kv_slots([physical_slot])
                )
                _append_kv_rows(kv_rows, snapshot)
                physical_slots.append(physical_slot)
            finally:
                reset_context()

        target_tokens = [first_target] + tail_targets
        accepted_tokens = _truncate_accepted_tokens(
            llm,
            seq,
            draft_tokens,
            target_tokens,
        )
        block_manager.commit_accepted_tokens(
            seq,
            accepted_tokens,
            reserved_blocks,
        )
        owned_blocks = []
        return {
            "target_tokens": target_tokens,
            "accepted_tokens": accepted_tokens,
            "logits": logits_rows,
            "kv": kv_rows,
            "physical_slots": physical_slots,
        }
    except Exception:
        block_manager.release_reserved_blocks(owned_blocks)
        raise
    finally:
        reset_context()


def _run_continuation(llm, seq, continuation_steps: int) -> dict:
    if seq.is_finished:
        raise RuntimeError(
            "verifier case finished before continuation evidence"
        )
    tokens = []
    logits = []
    kv = []
    physical_slots = []
    for _ in range(continuation_steps):
        step = _run_decode_evidence_step(llm, seq)
        seq.append_token(step["token_id"])
        tokens.append(step["token_id"])
        logits.append(step["logits"])
        kv.append(step["kv"])
        physical_slots.append(step["physical_slot"])
    return {
        "tokens": tokens,
        "logits": logits,
        "kv": kv,
        "physical_slots": physical_slots,
    }


def _run_tinyvllm_case(
    *,
    policy: str,
    case: dict,
    model: str,
    continuation_steps: int,
) -> dict:
    from tinyvllm import LLM, SamplingParams

    profile = _load_profile_module()
    prompt_tokens = None
    llm = LLM(
        model,
        enforce_eager=True,
        tensor_parallel_size=1,
        max_model_len=int(case.get("max_model_len", 4096)),
        gpu_memory_utilization=float(
            case.get("gpu_memory_utilization", 0.8)
        ),
        max_num_seqs=1,
        kv_quant_bits=0,
        kv_offload_mvp0=False,
        quest_top_k_blocks=-1,
    )
    try:
        prompt_tokens = llm.tokenizer.encode(case["prompt"])
        required_completion = (
            case["history_len"]
            - len(prompt_tokens)
            + len(case["draft_tokens"])
            + continuation_steps
            + 2
        )
        engine_max_tokens = max(
            int(case["max_tokens"]),
            required_completion,
        )
        sampling_params = SamplingParams(
            temperature=0.0,
            ignore_eos=True,
            max_tokens=engine_max_tokens,
        )
        llm.add_request(case["prompt"], sampling_params)
        seq = llm.scheduler.waiting[-1]
        _advance_to_history(llm, seq, case["history_len"])
        seq.max_tokens = int(case["max_tokens"])
        seq.ignore_eos = bool(case["ignore_eos"])

        if policy == "native":
            event = profile.verify_and_commit_block(
                llm,
                seq,
                case["draft_tokens"],
                draft_source="oracle-case",
                verifier_mode="native",
                capture_oracle_evidence=True,
                defer_finish_for_oracle_evidence=True,
            )
            verify = {
                "target_tokens": list(event["target_tokens"]),
                "accepted_tokens": list(event["accepted_tokens"]),
                "logits": event["oracle_evidence"]["logits"],
                "kv": event["oracle_evidence"]["kv"],
                "physical_slots": event[
                    "oracle_evidence"
                ]["physical_slots"],
                "event": event,
            }
        else:
            verify = _run_serialized_oracle_verify(
                llm,
                seq,
                case["draft_tokens"],
            )

        sequence_tokens_after = list(seq.token_ids)
        block_table_after = list(seq.block_table)
        continuation = _run_continuation(
            llm,
            seq,
            continuation_steps,
        )
        dtype_name = str(llm.model_runner.kv_cache.dtype)
        return build_case_payload({
            "case_id": case["case_id"],
            "policy": policy,
            "model": str(model),
            "prompt_token_count": len(prompt_tokens),
            "history_len": int(case["history_len"]),
            "draft_tokens": list(case["draft_tokens"]),
            "dtype": dtype_name,
            "target_tokens": verify["target_tokens"],
            "accepted_tokens": verify["accepted_tokens"],
            "sequence_tokens_after": sequence_tokens_after,
            "block_table_after": block_table_after,
            "continuation_tokens": continuation["tokens"],
            "logits": verify["logits"],
            "kv": verify["kv"],
            "physical_slots": verify["physical_slots"],
            "continuation_logits": continuation["logits"],
            "continuation_kv": continuation["kv"],
            "continuation_physical_slots": continuation[
                "physical_slots"
            ],
            "event": verify.get("event"),
        })
    finally:
        atexit.unregister(llm.exit)
        llm.exit()


def _numeric_comparison(
    native_value,
    oracle_value,
    *,
    rtol: float,
    atol: float,
) -> tuple[float, bool]:
    native_flat = _flatten_numbers(native_value)
    oracle_flat = _flatten_numbers(oracle_value)
    if len(native_flat) != len(oracle_flat):
        return math.inf, False
    max_abs_error = 0.0
    within_tolerance = True
    for native_number, oracle_number in zip(native_flat, oracle_flat):
        abs_error = abs(native_number - oracle_number)
        max_abs_error = max(max_abs_error, abs_error)
        allowed = atol + rtol * abs(oracle_number)
        if abs_error > allowed:
            within_tolerance = False
    return max_abs_error, within_tolerance


def compare_native_and_oracle(native: dict, oracle: dict) -> dict:
    reasons = []
    missing = []
    for label, payload in (("native", native), ("oracle", oracle)):
        for field in _REQUIRED_FIELDS:
            if field not in payload:
                missing.append(f"missing {label} field: {field}")
    if missing:
        return {
            "status": "INCOMPLETE",
            "reasons": missing,
            "target_token_match": False,
            "accepted_prefix_match": False,
            "metadata_match": False,
            "continuation_token_match": False,
            "continuation_steps": 0,
            "finite": False,
            "max_logit_abs_error": math.inf,
            "max_kv_abs_error": math.inf,
            "logits_within_tolerance": False,
            "kv_within_tolerance": False,
        }

    if native["dtype"] != oracle["dtype"]:
        return {
            "status": "INCOMPLETE",
            "reasons": ["native and oracle dtype mismatch"],
            "target_token_match": False,
            "accepted_prefix_match": False,
            "metadata_match": False,
            "continuation_token_match": False,
            "continuation_steps": 0,
            "finite": False,
            "max_logit_abs_error": math.inf,
            "max_kv_abs_error": math.inf,
            "logits_within_tolerance": False,
            "kv_within_tolerance": False,
        }

    tolerance = dtype_tolerance(native["dtype"])
    target_token_match = (
        native["target_tokens"] == oracle["target_tokens"]
    )
    accepted_prefix_match = (
        native["accepted_tokens"] == oracle["accepted_tokens"]
    )
    metadata_match = (
        native["sequence_tokens_after"]
        == oracle["sequence_tokens_after"]
        and native["block_table_after"] == oracle["block_table_after"]
    )
    continuation_token_match = (
        native["continuation_tokens"] == oracle["continuation_tokens"]
    )
    continuation_steps = min(
        len(native["continuation_tokens"]),
        len(oracle["continuation_tokens"]),
    )
    finite = bool(native["finite"]) and bool(oracle["finite"])
    max_logit_abs_error, logits_within_tolerance = _numeric_comparison(
        {
            "verify": native["logits"],
            "continuation": native["continuation_logits"],
        },
        {
            "verify": oracle["logits"],
            "continuation": oracle["continuation_logits"],
        },
        rtol=tolerance.logits_rtol,
        atol=tolerance.logits_atol,
    )
    max_kv_abs_error, kv_within_tolerance = _numeric_comparison(
        {
            "verify": native["kv"],
            "continuation": native["continuation_kv"],
        },
        {
            "verify": oracle["kv"],
            "continuation": oracle["continuation_kv"],
        },
        rtol=tolerance.kv_rtol,
        atol=tolerance.kv_atol,
    )

    if not target_token_match:
        reasons.append("target token mismatch")
    if not accepted_prefix_match:
        reasons.append("accepted prefix mismatch")
    if not metadata_match:
        reasons.append("committed metadata mismatch")
    if not continuation_token_match:
        reasons.append("continuation token mismatch")
    if not finite:
        reasons.append("non-finite logits or KV")
    if not logits_within_tolerance:
        reasons.append("logits exceed tolerance")
    if not kv_within_tolerance:
        reasons.append("KV exceeds tolerance")

    status = "PASS"
    if reasons:
        status = "NO_GO"
    if continuation_steps < 16:
        reasons.append("continuation coverage below 16 steps")
        status = "INCOMPLETE" if status == "PASS" else status

    return {
        "status": status,
        "reasons": reasons,
        "target_token_match": target_token_match,
        "accepted_prefix_match": accepted_prefix_match,
        "metadata_match": metadata_match,
        "continuation_token_match": continuation_token_match,
        "continuation_steps": continuation_steps,
        "finite": finite,
        "max_logit_abs_error": max_logit_abs_error,
        "max_kv_abs_error": max_kv_abs_error,
        "logits_within_tolerance": logits_within_tolerance,
        "kv_within_tolerance": kv_within_tolerance,
    }


def _parse_args():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    run_parser = subparsers.add_parser("run-case")
    run_parser.add_argument(
        "--policy",
        required=True,
        choices=["native", "oracle"],
    )
    run_parser.add_argument("--case-json", required=True)
    run_parser.add_argument("--out", required=True)
    run_parser.add_argument("--model", required=True)
    run_parser.add_argument("--continuation-steps", type=int, default=16)
    compare_parser = subparsers.add_parser("compare")
    compare_parser.add_argument("--native", required=True)
    compare_parser.add_argument("--oracle", required=True)
    compare_parser.add_argument("--out", required=True)
    return parser.parse_args()


def main():
    args = _parse_args()
    if args.command == "run-case":
        case = json.loads(Path(args.case_json).read_text())
        payload = run_case(
            policy=args.policy,
            case=case,
            out_path=Path(args.out),
            model=args.model,
            continuation_steps=args.continuation_steps,
        )
        print(json.dumps(payload, indent=2, sort_keys=True))
        return
    native = json.loads(Path(args.native).read_text())
    oracle = json.loads(Path(args.oracle).read_text())
    comparison = compare_native_and_oracle(native, oracle)
    Path(args.out).write_text(
        json.dumps(comparison, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(comparison, indent=2, sort_keys=True))
    if comparison["status"] != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
