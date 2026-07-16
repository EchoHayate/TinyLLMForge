from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import subprocess
import sys
import time
from copy import deepcopy
from pathlib import Path


def make_token_prompt(length: int, offset: int = 0) -> list[int]:
    return [100 + ((index + offset) % 1000) for index in range(length)]


def expected_reusable_tokens(prompt_tokens: int, block_size: int) -> int:
    if prompt_tokens <= 1:
        return 0
    return ((prompt_tokens - 1) // block_size) * block_size


def expected_shared_reusable_tokens(
    shared_prefix_tokens: int,
    prompt_tokens: int,
    block_size: int,
) -> int:
    shared_full_blocks = (shared_prefix_tokens // block_size) * block_size
    return min(
        shared_full_blocks,
        expected_reusable_tokens(prompt_tokens, block_size),
    )


def parse_int_list(raw: str) -> list[int]:
    return [int(item.strip()) for item in raw.split(",") if item.strip()]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_manifest(repo_root: Path, source_files: list[str], args: dict) -> dict:
    return {
        "args": args,
        "source_sha256": {
            relative: sha256_file(repo_root / relative) for relative in source_files
        },
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--mode",
        choices=["correctness", "performance", "full"],
        default="full",
    )
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--shared-prefix-tokens", default="256,1024,2048")
    parser.add_argument("--suffix-tokens", type=int, default=64)
    parser.add_argument("--repetitions", type=int, default=7)
    parser.add_argument("--warmup-repetitions", type=int, default=2)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--max-num-batched-tokens", type=int, default=8192)
    parser.add_argument("--max-num-seqs", type=int, default=8)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.5)
    parser.add_argument("--enforce-eager", action="store_true", default=False)
    return parser.parse_args()


def compare_logits(reference, candidate) -> dict:
    delta = (reference - candidate).abs()
    max_abs = float(delta.max())
    mean_abs = float(delta.mean())
    reference_argmax = int(reference.argmax())
    candidate_argmax = int(candidate.argmax())
    argmax_match = reference_argmax == candidate_argmax
    return {
        "max_abs": max_abs,
        "mean_abs": mean_abs,
        "reference_argmax": reference_argmax,
        "candidate_argmax": candidate_argmax,
        "argmax_match": argmax_match,
        "within_tolerance": (
            argmax_match and max_abs <= 0.25 and mean_abs <= 0.05
        ),
    }


def summarize_case_rows(rows: list[dict]) -> dict:
    return {
        "samples": len(rows),
        "median_ttft_ms": statistics.median(
            float(row["ttft_ms"]) for row in rows
        ),
        "min_ttft_ms": min(float(row["ttft_ms"]) for row in rows),
        "max_ttft_ms": max(float(row["ttft_ms"]) for row in rows),
        "median_query_tokens": statistics.median(
            int(row["query_tokens"]) for row in rows
        ),
        "median_cached_tokens": statistics.median(
            int(row["cached_tokens"]) for row in rows
        ),
        "all_correct": all(bool(row["correct"]) for row in rows),
    }


def decide_gate(correctness_rows: list[dict], performance_cases: list[dict]) -> dict:
    reasons = []
    failed = [row["case"] for row in correctness_rows if not row["correct"]]
    if failed:
        reasons.append("correctness failures: " + ", ".join(failed))
    for case in performance_cases:
        prefix = int(case["shared_prefix_tokens"])
        cold = float(case["cold"]["median_ttft_ms"])
        warm = float(case["warm"]["median_ttft_ms"])
        improvement = (cold - warm) / cold if cold > 0 else 0.0
        case["warm_ttft_improvement_fraction"] = improvement
        if not case["all_correct"]:
            reasons.append(f"{prefix}: incorrect performance sample")
        if prefix >= 1024 and improvement < 0.20:
            reasons.append(f"{prefix}: warm median TTFT improvement below 20%")
        if warm > cold * 1.05:
            reasons.append(f"{prefix}: warm median TTFT regression exceeds 5%")
        if int(case["warm_median_cached_tokens"]) != int(
            case["expected_reusable_tokens"]
        ):
            reasons.append(f"{prefix}: cached-token accounting mismatch")
        saved_queries = int(case["cold_median_query_tokens"]) - int(
            case["warm_median_query_tokens"]
        )
        if saved_queries != int(case["expected_reusable_tokens"]):
            reasons.append(f"{prefix}: executed prefill-token reduction mismatch")
    return {"decision": "NO_GO" if reasons else "GO", "reasons": reasons}


def audit_artifact_payloads(
    correctness_rows: list[dict],
    performance_rows: list[dict],
    summary: dict,
    repetitions: int,
    block_size: int = 256,
) -> list[str]:
    errors = []
    if summary.get("correctness_rows") != correctness_rows:
        errors.append("summary correctness rows do not match correctness_rows.json")

    stored_cases = summary.get("performance_cases", [])
    stored_by_prefix = {
        int(case["shared_prefix_tokens"]): case for case in stored_cases
    }
    raw_prefixes = {
        int(row["shared_prefix_tokens"]) for row in performance_rows
    }
    if raw_prefixes != set(stored_by_prefix):
        errors.append(
            "raw rows and summary performance prefixes differ: "
            f"raw={sorted(raw_prefixes)} summary={sorted(stored_by_prefix)}"
        )

    recomputed_cases = []
    for prefix in sorted(raw_prefixes):
        prefix_rows = [
            row
            for row in performance_rows
            if int(row["shared_prefix_tokens"]) == prefix
        ]
        state_rows = {
            state: [row for row in prefix_rows if row["state"] == state]
            for state in ("cold", "warm", "cache_cleared")
        }
        for state, rows in state_rows.items():
            if len(rows) != repetitions:
                errors.append(
                    f"{prefix} {state} raw samples {len(rows)} != {repetitions}"
                )
        if any(not rows for rows in state_rows.values()):
            continue

        stored = stored_by_prefix.get(prefix)
        if stored is None:
            continue
        suffix_values = {
            int(row["suffix_tokens"])
            for row in prefix_rows
            if "suffix_tokens" in row
        }
        if len(suffix_values) != 1:
            errors.append(
                f"{prefix} raw rows have inconsistent suffix tokens: "
                f"{sorted(suffix_values)}"
            )
            continue
        suffix_tokens = suffix_values.pop()
        expected_reusable = expected_shared_reusable_tokens(
            prefix,
            prefix + suffix_tokens,
            block_size,
        )
        if int(stored["expected_reusable_tokens"]) != expected_reusable:
            errors.append(
                f"{prefix} expected reusable tokens "
                f"{stored['expected_reusable_tokens']} != {expected_reusable}"
            )
        summaries = {
            state: summarize_case_rows(rows)
            for state, rows in state_rows.items()
        }
        recomputed = {
            "shared_prefix_tokens": prefix,
            "suffix_tokens": suffix_tokens,
            "expected_reusable_tokens": expected_reusable,
            "cold": summaries["cold"],
            "warm": summaries["warm"],
            "cache_cleared": summaries["cache_cleared"],
            "cold_median_query_tokens": summaries["cold"][
                "median_query_tokens"
            ],
            "warm_median_query_tokens": summaries["warm"][
                "median_query_tokens"
            ],
            "warm_median_cached_tokens": summaries["warm"][
                "median_cached_tokens"
            ],
            "all_correct": all(row["correct"] for row in prefix_rows),
        }
        recomputed_cases.append(recomputed)

    recomputed_decision = decide_gate(
        deepcopy(correctness_rows),
        recomputed_cases,
    )
    if recomputed_cases != stored_cases:
        errors.append("summary performance cases do not match raw rows")
    if recomputed_decision != summary.get("decision"):
        errors.append("summary decision does not match recomputed gate")
    return errors


def cuda_sync():
    import torch

    if torch.cuda.is_available():
        torch.cuda.synchronize()


def clone_logits_for_capture(logits):
    return logits.detach().float().clone()


def materialize_captured_logits(captures):
    return captures[0].cpu() if captures else None


def adjusted_ttft_ms(raw_ttft_ms: float, capture_overhead_ms: float) -> float:
    return max(0.0, float(raw_ttft_ms) - float(capture_overhead_ms))


def schedule_and_run_prefill(llm, prompts, capture_logits=True):
    from tinyvllm import SamplingParams

    params = SamplingParams(temperature=0.0, max_tokens=1, ignore_eos=True)
    for prompt in prompts:
        llm.add_request(prompt, params)
    scheduled = llm.scheduler.schedule()
    if len(scheduled) == 4:
        seqs, is_prefill, do_sample, batch_kind = scheduled
    else:
        seqs, is_prefill, do_sample = scheduled
        batch_kind = None
    assert is_prefill and do_sample
    metadata = [
        {
            "seq_id": seq.seq_id,
            "prompt_tokens": len(seq),
            "cached_tokens": int(seq.num_cached_tokens),
            "chunk_start": int(seq.prefill_chunk_start),
            "chunk_end": int(seq.prefill_chunk_end),
            "query_tokens": int(
                seq.prefill_chunk_end - seq.prefill_chunk_start
            ),
            "block_table": list(seq.block_table),
        }
        for seq in seqs
    ]
    assert all(row["query_tokens"] > 0 for row in metadata)

    captures = []
    capture_events = []
    original_forward = llm.model_runner.sampler.forward
    if capture_logits:

        def capture_forward(logits, temperatures):
            if logits.is_cuda:
                import torch

                capture_start = torch.cuda.Event(enable_timing=True)
                capture_end = torch.cuda.Event(enable_timing=True)
                capture_start.record()
                captures.append(clone_logits_for_capture(logits))
                capture_end.record()
                capture_events.append((capture_start, capture_end))
            else:
                captures.append(clone_logits_for_capture(logits))
            return original_forward(logits, temperatures)

        llm.model_runner.sampler.forward = capture_forward
    try:
        cuda_sync()
        start = time.perf_counter()
        token_ids = llm.model_runner.call(
            "run",
            seqs,
            is_prefill,
            do_sample,
            batch_kind,
        )
        cuda_sync()
        raw_ttft_ms = (time.perf_counter() - start) * 1000.0
    finally:
        llm.model_runner.sampler.forward = original_forward
    capture_overhead_ms = sum(
        float(start_event.elapsed_time(end_event))
        for start_event, end_event in capture_events
    )
    ttft_ms = adjusted_ttft_ms(raw_ttft_ms, capture_overhead_ms)
    logits = materialize_captured_logits(captures)
    llm.scheduler.postprocess(
        seqs,
        token_ids,
        is_prefill,
        do_sample,
        batch_kind,
    )
    return {
        "metadata": metadata,
        "token_ids": [int(token_id) for token_id in token_ids],
        "decoded": [
            llm.tokenizer.decode([int(token_id)]) for token_id in token_ids
        ],
        "logits": logits,
        "ttft_ms": ttft_ms,
        "raw_ttft_ms": raw_ttft_ms,
        "capture_overhead_ms": capture_overhead_ms,
    }


def _comparison_row(case_name, state, result, index, reference):
    comparison = compare_logits(reference["logits"][index], result["logits"][index])
    token_id = result["token_ids"][index]
    decoded = result["decoded"][index]
    reference_token_id = reference["token_ids"][index]
    reference_decoded = reference["decoded"][index]
    metadata = result["metadata"][index]
    return {
        "case": case_name,
        "state": state,
        "prompt_tokens": metadata["prompt_tokens"],
        "cached_tokens": metadata["cached_tokens"],
        "query_tokens": metadata["query_tokens"],
        "token_id": token_id,
        "decoded": decoded,
        "logit_diff": comparison,
        "correct": (
            token_id == reference_token_id
            and decoded == reference_decoded
            and comparison["within_tolerance"]
        ),
    }


def _single_comparison_row(case_name, state, result, reference):
    comparison = compare_logits(reference["logits"][0], result["logits"][0])
    metadata = result["metadata"][0]
    token_id = result["token_ids"][0]
    decoded = result["decoded"][0]
    return {
        "case": case_name,
        "state": state,
        "prompt_tokens": metadata["prompt_tokens"],
        "cached_tokens": metadata["cached_tokens"],
        "query_tokens": metadata["query_tokens"],
        "token_id": token_id,
        "decoded": decoded,
        "logit_diff": comparison,
        "correct": (
            token_id == reference["token_ids"][0]
            and decoded == reference["decoded"][0]
            and comparison["within_tolerance"]
        ),
    }


def _run_cpu_preflight(repo_root: Path) -> dict:
    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    command = [sys.executable, "tools/test_chunked_prefill.py"]
    completed = subprocess.run(
        command,
        cwd=repo_root,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    return {
        "case": "cpu_collision_and_lifecycle_preflight",
        "state": "preflight",
        "command": command,
        "returncode": completed.returncode,
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
        "correct": completed.returncode == 0,
    }


def run_correctness_cases(llm, repo_root: Path) -> list[dict]:
    block_manager = llm.scheduler.block_manager
    block_size = block_manager.block_size
    rows = [_run_cpu_preflight(repo_root)]

    for prompt_tokens in (255, 256, 257, 512, 513):
        prompt = make_token_prompt(prompt_tokens, 1000 + prompt_tokens)
        block_manager.clear_reusable_cache()
        cold = schedule_and_run_prefill(llm, [prompt])
        warm = schedule_and_run_prefill(llm, [prompt])
        row = _single_comparison_row(
            f"repeat_{prompt_tokens}",
            "warm",
            warm,
            cold,
        )
        expected_cached = expected_reusable_tokens(prompt_tokens, block_size)
        row["expected_reusable_tokens"] = expected_cached
        row["correct"] = (
            row["correct"]
            and row["cached_tokens"] == expected_cached
            and row["query_tokens"] == prompt_tokens - expected_cached
        )
        rows.append(row)

    prompt_tokens = 513
    prompt_p = make_token_prompt(prompt_tokens, 3000)
    independent_p = None
    independent_q = None
    prompt_q = None
    for attempt in range(32):
        candidate_q = make_token_prompt(prompt_tokens, 4000 + attempt * 137)
        block_manager.clear_reusable_cache()
        candidate_p_result = schedule_and_run_prefill(llm, [prompt_p])
        block_manager.clear_reusable_cache()
        candidate_q_result = schedule_and_run_prefill(llm, [candidate_q])
        difference = compare_logits(
            candidate_p_result["logits"][0],
            candidate_q_result["logits"][0],
        )
        if difference["max_abs"] > 1.0:
            independent_p = candidate_p_result
            independent_q = candidate_q_result
            prompt_q = candidate_q
            break
    if independent_p is None or independent_q is None or prompt_q is None:
        raise RuntimeError("failed to find distinct deterministic P/Q prompts")

    block_manager.clear_reusable_cache()
    batch = schedule_and_run_prefill(llm, [prompt_p, prompt_q, prompt_p])
    batch_q_delta = compare_logits(batch["logits"][2], batch["logits"][1])
    p_first = _comparison_row(
        "same_batch_p_q_p_first",
        "same_batch",
        batch,
        0,
        {
            **independent_p,
            "logits": independent_p["logits"].repeat(3, 1),
            "token_ids": independent_p["token_ids"] * 3,
            "decoded": independent_p["decoded"] * 3,
        },
    )
    q_middle = _comparison_row(
        "same_batch_p_q_p_middle",
        "same_batch",
        batch,
        1,
        {
            **independent_q,
            "logits": independent_q["logits"].repeat(3, 1),
            "token_ids": independent_q["token_ids"] * 3,
            "decoded": independent_q["decoded"] * 3,
        },
    )
    p_third = _comparison_row(
        "same_batch_p_q_p",
        "same_batch",
        batch,
        2,
        {
            **independent_p,
            "logits": independent_p["logits"].repeat(3, 1),
            "token_ids": independent_p["token_ids"] * 3,
            "decoded": independent_p["decoded"] * 3,
        },
    )
    p_third["batch_q_logit_diff"] = batch_q_delta
    for row in (p_first, q_middle, p_third):
        row["correct"] = (
            row["correct"]
            and row["cached_tokens"] == 0
            and row["query_tokens"] == prompt_tokens
        )
    p_third["correct"] = p_third["correct"] and batch_q_delta["max_abs"] > 0.0
    rows.extend([p_first, q_middle, p_third])

    shared_prefix = make_token_prompt(block_size, 6000)
    producer = shared_prefix + make_token_prompt(64, 6211)
    consumer = shared_prefix + make_token_prompt(64, 6523)
    block_manager.clear_reusable_cache()
    cold = schedule_and_run_prefill(llm, [consumer])
    block_manager.clear_reusable_cache()
    schedule_and_run_prefill(llm, [producer])
    warm = schedule_and_run_prefill(llm, [consumer])
    warm_row = _single_comparison_row(
        "shared_prefix_different_suffix",
        "warm",
        warm,
        cold,
    )
    warm_row["expected_reusable_tokens"] = block_size
    warm_row["correct"] = (
        warm_row["correct"]
        and warm_row["cached_tokens"] == block_size
        and warm_row["query_tokens"] == len(consumer) - block_size
    )
    rows.append(warm_row)

    block_manager.clear_reusable_cache()
    schedule_and_run_prefill(llm, [producer])
    block_manager.clear_reusable_cache()
    cleared = schedule_and_run_prefill(llm, [consumer])
    cleared_row = _single_comparison_row(
        "cache_cleared",
        "cache_cleared",
        cleared,
        cold,
    )
    cleared_row["expected_reusable_tokens"] = 0
    cleared_row["correct"] = (
        cleared_row["correct"]
        and cleared_row["cached_tokens"] == 0
        and cleared_row["query_tokens"] == len(consumer)
    )
    rows.append(cleared_row)
    return rows


def _performance_row(state, result, reference, repetition):
    row = _single_comparison_row(
        f"performance_{state}",
        state,
        result,
        reference,
    )
    row["repetition"] = repetition
    row["ttft_ms"] = result["ttft_ms"]
    row["raw_ttft_ms"] = result["raw_ttft_ms"]
    row["capture_overhead_ms"] = result["capture_overhead_ms"]
    return row


def run_performance_cases(
    llm,
    shared_prefix_tokens: list[int],
    suffix_tokens: int,
    repetitions: int,
    warmup_repetitions: int,
) -> tuple[list[dict], list[dict]]:
    block_manager = llm.scheduler.block_manager
    raw_rows = []
    performance_cases = []
    total_repetitions = repetitions + warmup_repetitions
    for prefix_tokens in shared_prefix_tokens:
        case_rows = []
        for repetition in range(total_repetitions):
            base_offset = 10000 + prefix_tokens * 11 + repetition * 97
            prefix = make_token_prompt(prefix_tokens, base_offset)
            producer = prefix + make_token_prompt(
                suffix_tokens,
                base_offset + 211,
            )
            consumer = prefix + make_token_prompt(
                suffix_tokens,
                base_offset + 523,
            )

            block_manager.clear_reusable_cache()
            cold = schedule_and_run_prefill(llm, [consumer])

            block_manager.clear_reusable_cache()
            schedule_and_run_prefill(llm, [producer])
            warm = schedule_and_run_prefill(llm, [consumer])

            block_manager.clear_reusable_cache()
            schedule_and_run_prefill(llm, [producer])
            block_manager.clear_reusable_cache()
            cleared = schedule_and_run_prefill(llm, [consumer])

            if repetition < warmup_repetitions:
                continue
            measured_repetition = repetition - warmup_repetitions
            rows = [
                _performance_row("cold", cold, cold, measured_repetition),
                _performance_row("warm", warm, cold, measured_repetition),
                _performance_row(
                    "cache_cleared",
                    cleared,
                    cold,
                    measured_repetition,
                ),
            ]
            for row in rows:
                row["shared_prefix_tokens"] = prefix_tokens
                row["suffix_tokens"] = suffix_tokens
            case_rows.extend(rows)
            raw_rows.extend(rows)

        summaries = {
            state: summarize_case_rows(
                [row for row in case_rows if row["state"] == state]
            )
            for state in ("cold", "warm", "cache_cleared")
        }
        expected_cached = expected_shared_reusable_tokens(
            prefix_tokens,
            prefix_tokens + suffix_tokens,
            block_manager.block_size,
        )
        performance_cases.append(
            {
                "shared_prefix_tokens": prefix_tokens,
                "suffix_tokens": suffix_tokens,
                "expected_reusable_tokens": expected_cached,
                "cold": summaries["cold"],
                "warm": summaries["warm"],
                "cache_cleared": summaries["cache_cleared"],
                "cold_median_query_tokens": summaries["cold"][
                    "median_query_tokens"
                ],
                "warm_median_query_tokens": summaries["warm"][
                    "median_query_tokens"
                ],
                "warm_median_cached_tokens": summaries["warm"][
                    "median_cached_tokens"
                ],
                "all_correct": all(row["correct"] for row in case_rows),
            }
        )
    return raw_rows, performance_cases


def _write_json(path: Path, payload):
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n"
    )


def render_report(manifest, correctness_rows, performance_cases, decision) -> str:
    lines = [
        "# Prefix Cache Correctness and TTFT Gate",
        "",
        f"**Decision:** `{decision['decision']}`",
        "",
        "## Source Hashes",
        "",
        "| File | SHA-256 |",
        "| --- | --- |",
    ]
    for source, digest in manifest["source_sha256"].items():
        lines.append(f"| `{source}` | `{digest}` |")

    lines.extend(
        [
            "",
            "## Correctness",
            "",
            "| Case | State | Prompt | Cached | Query | Correct |",
            "| --- | --- | ---: | ---: | ---: | --- |",
        ]
    )
    for row in correctness_rows:
        lines.append(
            "| {case} | {state} | {prompt} | {cached} | {query} | {correct} |".format(
                case=row["case"],
                state=row.get("state", ""),
                prompt=row.get("prompt_tokens", ""),
                cached=row.get("cached_tokens", ""),
                query=row.get("query_tokens", ""),
                correct=row["correct"],
            )
        )

    lines.extend(
        [
            "",
            "## TTFT",
            "",
            "| Shared Prefix | Cold Median ms | Warm Median ms | Cleared Median ms | "
            "Warm Cached | Cold Query | Warm Query | Improvement | Correct |",
            "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for case in performance_cases:
        lines.append(
            "| {prefix} | {cold:.3f} | {warm:.3f} | {cleared:.3f} | "
            "{cached} | {cold_query} | {warm_query} | {improvement:.2%} | "
            "{correct} |".format(
                prefix=case["shared_prefix_tokens"],
                cold=case["cold"]["median_ttft_ms"],
                warm=case["warm"]["median_ttft_ms"],
                cleared=case["cache_cleared"]["median_ttft_ms"],
                cached=case["warm_median_cached_tokens"],
                cold_query=case["cold_median_query_tokens"],
                warm_query=case["warm_median_query_tokens"],
                improvement=case.get("warm_ttft_improvement_fraction", 0.0),
                correct=case["all_correct"],
            )
        )

    lines.extend(["", "## Rejection Reasons", ""])
    if decision["reasons"]:
        lines.extend(f"- {reason}" for reason in decision["reasons"])
    else:
        lines.append("- None.")
    lines.extend(
        [
            "",
            "## Claim Boundaries",
            "",
            "- This gate covers prompt prefill and time-to-first-token only.",
            "- It does not claim decode acceleration.",
            "- Logical prefix reuse is not a claim of lower physical KV allocation "
            "or greater KV capacity.",
            "- Correctness requires exact greedy token/text equality plus full-logit "
            "argmax and numeric tolerance checks.",
            "",
        ]
    )
    return "\n".join(lines)


def run_profile(args) -> dict:
    from tinyvllm import LLM

    repo_root = Path(__file__).resolve().parents[1]
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    source_files = [
        "tinyvllm/engine/block_manager.py",
        "tinyvllm/engine/scheduler.py",
        "tools/profile_prefix_cache.py",
        "tools/test_profile_prefix_cache.py",
    ]
    manifest = build_manifest(repo_root, source_files, vars(args))
    _write_json(out_dir / "manifest.json", manifest)

    llm = LLM(
        args.model,
        enforce_eager=args.enforce_eager,
        max_model_len=args.max_model_len,
        max_num_batched_tokens=args.max_num_batched_tokens,
        max_num_seqs=args.max_num_seqs,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )
    correctness_rows = []
    performance_rows = []
    performance_cases = []
    if args.mode in ("correctness", "full"):
        correctness_rows = run_correctness_cases(llm, repo_root)
    if args.mode in ("performance", "full"):
        performance_rows, performance_cases = run_performance_cases(
            llm,
            parse_int_list(args.shared_prefix_tokens),
            args.suffix_tokens,
            args.repetitions,
            args.warmup_repetitions,
        )
    decision = decide_gate(correctness_rows, performance_cases)
    if args.mode != "full":
        decision["decision"] = "NO_GO"
        decision["reasons"].append(
            f"{args.mode}-only run does not satisfy the full gate"
        )
    summary = {
        "correctness_rows": correctness_rows,
        "performance_cases": performance_cases,
        "decision": decision,
    }
    _write_json(out_dir / "correctness_rows.json", correctness_rows)
    _write_json(out_dir / "performance_rows.json", performance_rows)
    _write_json(out_dir / "summary.json", summary)
    (out_dir / "report.md").write_text(
        render_report(manifest, correctness_rows, performance_cases, decision)
    )
    return summary


def main():
    args = parse_args()
    summary = run_profile(args)
    print(json.dumps(summary["decision"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
