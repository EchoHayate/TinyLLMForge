"""Pure contracts for source-bound P5 step-cost calibration."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time


INT64_MAX = (1 << 63) - 1
CANONICAL_MAX_NUM_SEQS = 512
CANONICAL_MAX_PREFILL_TOKENS = 128
DECODE_CONTEXT_TOKENS = {
    "short": 64,
    "medium": 512,
    "long": 1536,
}


def canonical_json_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _is_int(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def build_required_shapes(
    *,
    max_num_seqs: int,
    max_prefill_tokens: int,
) -> list[dict]:
    if (
        not _is_int(max_num_seqs)
        or not _is_int(max_prefill_tokens)
        or max_num_seqs <= 0
        or max_prefill_tokens != CANONICAL_MAX_PREFILL_TOKENS
    ):
        raise ValueError("unsupported P5 calibration limits")
    decode_counts = sorted({
        count for count in (1, 8, 32, max_num_seqs)
        if count <= max_num_seqs
    })
    shapes = []
    for context_class in ("short", "medium", "long"):
        for decode_rows in decode_counts:
            shapes.append({
                "shape_id": f"decode-{context_class}-d{decode_rows}",
                "kind": "decode",
                "context_class": context_class,
                "decode_rows": decode_rows,
                "prefill_rows": 0,
                "prefill_tokens": 0,
                "warmup_iterations": 1,
                "measured_iterations": 7,
            })
    for prefill_tokens in (16, 32, 64, 128):
        prefill_rows = sorted({
            1,
            min(8, prefill_tokens // 16),
        })
        for row_count in prefill_rows:
            mixed_decode_counts = sorted({
                count
                for count in (
                    1,
                    8,
                    32,
                    max_num_seqs - row_count,
                )
                if 0 < count <= max_num_seqs - row_count
            })
            for decode_rows in mixed_decode_counts:
                shapes.append({
                    "shape_id": (
                        f"mixed-p{prefill_tokens}-r{row_count}"
                        f"-d{decode_rows}"
                    ),
                    "kind": "mixed",
                    "context_class": "mixed",
                    "decode_rows": decode_rows,
                    "prefill_rows": row_count,
                    "prefill_tokens": prefill_tokens,
                    "warmup_iterations": 1,
                    "measured_iterations": 7,
                })
    return shapes


def nearest_rank_p99_ns(values: list[int]) -> int:
    if len(values) < 7:
        raise ValueError("cost calibration requires at least seven samples")
    if any(not _is_int(value) or value <= 0 for value in values):
        raise ValueError("invalid calibration duration")
    ordered = sorted(values)
    return ordered[math.ceil(len(ordered) * 0.99) - 1]


def inflate_duration_ns(measured_p99_ns: int) -> int:
    if not _is_int(measured_p99_ns) or measured_p99_ns <= 0:
        raise ValueError("invalid measured p99 duration")
    inflated = (measured_p99_ns * 5 + 3) // 4
    if inflated > INT64_MAX:
        raise OverflowError("calibration inflation overflows int64")
    return inflated


def _shape_contract(shape: dict) -> dict:
    fields = (
        "shape_id",
        "kind",
        "context_class",
        "decode_rows",
        "prefill_rows",
        "prefill_tokens",
        "warmup_iterations",
        "measured_iterations",
    )
    if not isinstance(shape, dict) or any(
        field not in shape for field in fields
    ):
        raise ValueError("invalid calibration shape")
    return {field: shape[field] for field in fields}


def _token_prompt(length: int, offset: int) -> list[int]:
    if not _is_int(length) or length <= 0:
        raise ValueError("invalid calibration prompt length")
    return [100 + ((offset + index) % 1000) for index in range(length)]


def _prefill_row_lengths(
    *,
    prefill_tokens: int,
    prefill_rows: int,
) -> list[int]:
    if (
        not _is_int(prefill_tokens)
        or not _is_int(prefill_rows)
        or prefill_tokens <= 0
        or prefill_rows <= 0
        or prefill_rows > prefill_tokens
    ):
        raise ValueError("invalid mixed calibration shape")
    quotient, remainder = divmod(prefill_tokens, prefill_rows)
    return [
        quotient + (1 if index < remainder else 0)
        for index in range(prefill_rows)
    ]


def _observed_batch_shape(observation: object) -> dict:
    if not isinstance(observation, dict):
        raise ValueError("calibration step missing observation")
    scheduled = observation.get("scheduled")
    if not isinstance(scheduled, list):
        raise ValueError("calibration step missing scheduled rows")
    decode_rows = 0
    prefill_rows = 0
    prefill_tokens = 0
    for row in scheduled:
        if not isinstance(row, dict):
            raise ValueError("invalid scheduled calibration row")
        if row.get("is_decode") is True:
            decode_rows += 1
            continue
        start = row.get("prefill_chunk_start")
        end = row.get("prefill_chunk_end")
        if (
            not _is_int(start)
            or not _is_int(end)
            or start < 0
            or end <= start
        ):
            raise ValueError("invalid scheduled prefill chunk")
        prefill_rows += 1
        prefill_tokens += end - start
    return {
        "decode_rows": decode_rows,
        "prefill_rows": prefill_rows,
        "prefill_tokens": prefill_tokens,
    }


def _assert_observed_shape(shape: dict, observation: object) -> None:
    observed = _observed_batch_shape(observation)
    expected = {
        "decode_rows": shape["decode_rows"],
        "prefill_rows": shape["prefill_rows"],
        "prefill_tokens": shape["prefill_tokens"],
    }
    if observed != expected:
        raise ValueError(
            f"calibration batch shape mismatch: "
            f"expected={expected} observed={observed}"
        )


def _prime_decode_rows(
    shape: dict,
    *,
    engine,
    sampling_params_factory,
) -> None:
    context_tokens = DECODE_CONTEXT_TOKENS.get(shape["context_class"])
    if context_tokens is None:
        if shape["kind"] != "mixed":
            raise ValueError("invalid decode context class")
        context_tokens = DECODE_CONTEXT_TOKENS["medium"]
    target_steps = (
        shape["warmup_iterations"]
        + shape["measured_iterations"]
    )
    sampling_params = sampling_params_factory(
        max_tokens=target_steps + 2
    )
    for row_index in range(shape["decode_rows"]):
        engine.add_request(
            _token_prompt(
                context_tokens,
                10_000 + row_index * 17,
            ),
            sampling_params,
        )
    max_prime_steps = (
        shape["decode_rows"] * context_tokens
        // CANONICAL_MAX_PREFILL_TOKENS
        + shape["decode_rows"]
        + 16
    )
    for _ in range(max_prime_steps):
        engine.step()
        try:
            _assert_observed_shape({
                **shape,
                "prefill_rows": 0,
                "prefill_tokens": 0,
            }, engine.last_step_observation)
        except ValueError:
            continue
        return
    raise ValueError("decode calibration rows did not become runnable")


def execute_calibration_shape(
    shape: dict,
    *,
    engine,
    sampling_params_factory,
    synchronize,
    clock_ns=time.monotonic_ns,
) -> list[dict]:
    contract = _shape_contract(shape)
    _prime_decode_rows(
        contract,
        engine=engine,
        sampling_params_factory=sampling_params_factory,
    )
    prefill_lengths = []
    if contract["kind"] == "mixed":
        prefill_lengths = _prefill_row_lengths(
            prefill_tokens=contract["prefill_tokens"],
            prefill_rows=contract["prefill_rows"],
        )
    elif contract["kind"] != "decode":
        raise ValueError("invalid calibration shape kind")

    rows = []
    total_iterations = (
        contract["warmup_iterations"]
        + contract["measured_iterations"]
    )
    for raw_iteration in range(total_iterations):
        if prefill_lengths:
            prefill_sampling = sampling_params_factory(max_tokens=1)
            for row_index, prompt_length in enumerate(prefill_lengths):
                engine.add_request(
                    _token_prompt(
                        prompt_length,
                        100_000
                        + raw_iteration * 1_000
                        + row_index * 31,
                    ),
                    prefill_sampling,
                )
        synchronize()
        started_ns = clock_ns()
        engine.step()
        synchronize()
        ended_ns = clock_ns()
        duration_ns = ended_ns - started_ns
        if not _is_int(duration_ns) or duration_ns <= 0:
            raise ValueError("non-positive synchronous duration")
        _assert_observed_shape(
            contract,
            engine.last_step_observation,
        )
        if raw_iteration < contract["warmup_iterations"]:
            continue
        rows.append({
            **contract,
            "iteration": (
                raw_iteration - contract["warmup_iterations"]
            ),
            "duration_ns": duration_ns,
        })
    return rows


def orchestrate_calibration_shapes(
    required_shapes: list[dict],
    *,
    allocate_port_pair,
    launch_shape,
) -> list[dict]:
    if not isinstance(required_shapes, list) or not required_shapes:
        raise ValueError("calibration requires shapes")
    rows = []
    used_ports = set()
    seen_shape_ids = set()
    for shape in required_shapes:
        contract = _shape_contract(shape)
        shape_id = contract["shape_id"]
        if shape_id in seen_shape_ids:
            raise ValueError("duplicate required calibration shape")
        seen_shape_ids.add(shape_id)
        tinyvllm_dist_port, master_port = allocate_port_pair()
        ports = (tinyvllm_dist_port, master_port)
        if any(
            not _is_int(port)
            or port <= 0
            or port > 65_535
            or port in used_ports
            for port in ports
        ) or tinyvllm_dist_port == master_port:
            raise ValueError("invalid or reused calibration port")
        used_ports.update(ports)
        shape_rows = launch_shape(
            shape=contract,
            tinyvllm_dist_port=tinyvllm_dist_port,
            master_port=master_port,
        )
        if not isinstance(shape_rows, list):
            raise ValueError("calibration shape launch returned no rows")
        expected_iterations = set(
            range(contract["measured_iterations"])
        )
        observed_iterations = set()
        for row in shape_rows:
            if (
                not isinstance(row, dict)
                or _shape_contract(row) != contract
            ):
                raise ValueError("calibration launch changed shape")
            iteration = row.get("iteration")
            duration_ns = row.get("duration_ns")
            if (
                not _is_int(iteration)
                or iteration in observed_iterations
                or not _is_int(duration_ns)
                or duration_ns <= 0
            ):
                raise ValueError("invalid calibration launch row")
            observed_iterations.add(iteration)
        if observed_iterations != expected_iterations:
            raise ValueError("incomplete calibration launch rows")
        rows.extend(shape_rows)
    return rows


def recompute_cost_envelope(
    rows: list[dict],
    *,
    required_shapes: list[dict] | None = None,
) -> dict:
    if required_shapes is None:
        required_shapes = build_required_shapes(
            max_num_seqs=CANONICAL_MAX_NUM_SEQS,
            max_prefill_tokens=CANONICAL_MAX_PREFILL_TOKENS,
        )
    expected_by_id = {}
    for shape in required_shapes:
        contract = _shape_contract(shape)
        shape_id = contract["shape_id"]
        if shape_id in expected_by_id:
            raise ValueError("duplicate required calibration shape")
        expected_by_id[shape_id] = contract

    if not isinstance(rows, list):
        raise ValueError("calibration rows must be a list")
    grouped = {}
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("invalid calibration row")
        shape_id = row.get("shape_id")
        if shape_id not in expected_by_id:
            raise ValueError("unexpected calibration shape")
        if _shape_contract(row) != expected_by_id[shape_id]:
            raise ValueError("inconsistent calibration shape")
        iteration = row.get("iteration")
        if not _is_int(iteration) or iteration < 0:
            raise ValueError("invalid calibration iteration")
        shape_rows = grouped.setdefault(shape_id, {})
        if iteration in shape_rows:
            raise ValueError("duplicate calibration iteration")
        shape_rows[iteration] = row

    missing = sorted(set(expected_by_id) - set(grouped))
    if missing:
        raise ValueError(
            "missing required calibration shapes: " + ", ".join(missing)
        )

    shape_summaries = []
    for shape_id in sorted(expected_by_id):
        shape = expected_by_id[shape_id]
        shape_rows = grouped[shape_id]
        expected_iterations = shape["measured_iterations"]
        if set(shape_rows) != set(range(expected_iterations)):
            if len(shape_rows) < 7:
                raise ValueError(
                    "cost calibration requires at least seven samples"
                )
            raise ValueError("invalid calibration iteration set")
        durations = [
            shape_rows[index].get("duration_ns")
            for index in range(expected_iterations)
        ]
        measured_p99_ns = nearest_rank_p99_ns(durations)
        shape_summaries.append({
            **shape,
            "measured_duration_ns": durations,
            "measured_p99_ns": measured_p99_ns,
            "inflated_duration_ns": inflate_duration_ns(
                measured_p99_ns
            ),
        })

    decode_points = [
        row for row in shape_summaries if row["kind"] == "decode"
    ]
    mixed_points = [
        row for row in shape_summaries if row["kind"] == "mixed"
    ]
    if not decode_points or not mixed_points:
        raise ValueError("incomplete calibration shape classes")
    intercept = max(
        row["inflated_duration_ns"] for row in decode_points
    )
    slope = 1
    for point in mixed_points:
        tokens = point["prefill_tokens"]
        if not _is_int(tokens) or tokens <= 0:
            raise ValueError("infeasible calibration shape")
        excess = max(0, point["inflated_duration_ns"] - intercept)
        required = max(1, (excess + tokens - 1) // tokens)
        slope = max(slope, required)

    max_tokens = max(
        row["prefill_tokens"] for row in mixed_points
    )
    if (
        intercept > INT64_MAX
        or slope > (INT64_MAX - intercept) // max_tokens
    ):
        raise OverflowError("cost envelope overflows int64")
    for point in mixed_points:
        predicted = intercept + point["prefill_tokens"] * slope
        if predicted < point["inflated_duration_ns"]:
            raise ValueError("cost envelope does not dominate point")
    return {
        "shape_summaries": shape_summaries,
        "cost_intercept_ns": intercept,
        "cost_per_prefill_token_ns": slope,
    }


def build_cost_calibration_summary(
    *,
    source_tree_sha256: str,
    environment_sha256: str,
    engine_config_sha256: str,
    required_shapes: list[dict],
    raw_rows: list[dict],
) -> dict:
    identities = (
        source_tree_sha256,
        environment_sha256,
        engine_config_sha256,
    )
    if any(
        not isinstance(value, str) or len(value) != 64
        for value in identities
    ):
        raise ValueError("invalid cost calibration identity")
    envelope = recompute_cost_envelope(
        raw_rows,
        required_shapes=required_shapes,
    )
    return {
        "status": "PASS",
        "source_tree_sha256": source_tree_sha256,
        "environment_sha256": environment_sha256,
        "engine_config_sha256": engine_config_sha256,
        "required_shape_sha256": canonical_json_sha256(
            required_shapes
        ),
        "raw_rows_sha256": canonical_json_sha256(raw_rows),
        "cost_intercept_ns": envelope["cost_intercept_ns"],
        "cost_per_prefill_token_ns": envelope[
            "cost_per_prefill_token_ns"
        ],
        "envelope_sha256": canonical_json_sha256(envelope),
    }


def _cuda_synchronize() -> None:
    import torch
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def run_shape_process(
    *,
    shape: dict,
    model_path: str,
    engine_config: dict,
) -> list[dict]:
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from tinyvllm import LLM, SamplingParams

    engine = LLM(str(model_path), **dict(engine_config))
    return execute_calibration_shape(
        shape,
        engine=engine,
        sampling_params_factory=lambda max_tokens: SamplingParams(
            temperature=0.0,
            max_tokens=max_tokens,
            ignore_eos=True,
        ),
        synchronize=_cuda_synchronize,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Execute one isolated P5 cost-calibration shape",
    )
    parser.add_argument("--shape-json", type=Path, required=True)
    parser.add_argument("--engine-config-json", type=Path, required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    shape = json.loads(args.shape_json.read_text(encoding="utf-8"))
    engine_config = json.loads(
        args.engine_config_json.read_text(encoding="utf-8")
    )
    rows = run_shape_process(
        shape=shape,
        model_path=args.model_path,
        engine_config=engine_config,
    )
    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output_jsonl.with_name(
        args.output_jsonl.name + ".tmp"
    )
    temporary.write_text(
        "".join(
            json.dumps(
                row,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            ) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )
    temporary.replace(args.output_jsonl)
    return 0


if __name__ == "__main__":
    os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
    raise SystemExit(main())
