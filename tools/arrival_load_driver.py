"""Inline synchronous driver for one frozen arrival-load case."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import traceback
from collections import deque
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

P5_DECISION_FIELDS = (
    "decision_now_ns",
    "target_gap_ns",
    "reserve_ns",
    "oldest_decode_seq_id",
    "oldest_decode_progress_ns",
    "oldest_decode_age_ns",
    "remaining_slack_ns",
    "cost_intercept_ns",
    "cost_per_prefill_token_ns",
    "candidate_chunk_tokens",
    "predicted_step_ns",
    "selected_chunk_tokens",
    "actual_prefill_tokens",
    "scheduled_decode_seq_ids",
    "demand_state_before",
    "demand_state_after",
    "suppression_reason",
    "clock_invalid",
    "clock_invalid_reason",
)
P5_POSTPROCESS_FIELDS = (
    "step_end_ns",
    "actual_step_duration_ns",
    "decode_progress_updates",
    "finished_progress_entries_removed",
)


class DriverError(RuntimeError):
    def __init__(self, error_type: str, message: str):
        super().__init__(message)
        self.error_type = error_type


class AppendOnlyJsonl:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.handle = self.path.open(
            "a",
            encoding="utf-8",
            buffering=1,
        )

    def append(self, row: dict):
        self.handle.write(
            json.dumps(
                row,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            )
            + "\n"
        )
        self.handle.flush()
        os.fsync(self.handle.fileno())

    def close(self):
        self.handle.close()


def _atomic_write_json(path: Path, value: dict):
    temporary = path.with_name(path.name + ".tmp")
    payload = json.dumps(
        value,
        sort_keys=True,
        indent=2,
        ensure_ascii=False,
        allow_nan=False,
    ) + "\n"
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _write_text(path: Path, text: str):
    with path.open("w", encoding="utf-8") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())


def _validate_case_spec(case_spec: dict):
    if not isinstance(case_spec, dict):
        raise DriverError(
            "invalid_case_spec",
            "case_spec must be an object",
        )
    drain_timeout_ns = case_spec.get("drain_timeout_ns")
    if (
        isinstance(drain_timeout_ns, bool)
        or not isinstance(drain_timeout_ns, int)
        or drain_timeout_ns <= 0
    ):
        raise DriverError(
            "invalid_case_spec",
            "drain_timeout_ns must be a positive integer",
        )


def _validate_workload(workload_rows: list[dict]):
    if not isinstance(workload_rows, list) or not workload_rows:
        raise DriverError(
            "invalid_workload_manifest",
            "workload manifest must contain requests",
        )
    request_ids = set()
    previous_key = None
    for index, row in enumerate(workload_rows):
        if not isinstance(row, dict):
            raise DriverError(
                "invalid_workload_manifest",
                f"workload row {index} must be an object",
            )
        request_id = row.get("request_id")
        if not isinstance(request_id, str) or not request_id:
            raise DriverError(
                "invalid_workload_manifest",
                f"invalid request_id at row {index}",
            )
        if request_id in request_ids:
            raise DriverError(
                "invalid_workload_manifest",
                f"duplicate request_id: {request_id}",
            )
        request_ids.add(request_id)
        arrival_offset_ns = row.get("arrival_offset_ns")
        if (
            isinstance(arrival_offset_ns, bool)
            or not isinstance(arrival_offset_ns, int)
            or arrival_offset_ns < 0
        ):
            raise DriverError(
                "invalid_workload_manifest",
                f"invalid arrival_offset_ns for {request_id}",
            )
        key = (arrival_offset_ns, request_id)
        if previous_key is not None and key < previous_key:
            raise DriverError(
                "invalid_workload_manifest",
                "workload rows must be in arrival order",
            )
        previous_key = key
        prompt_token_ids = row.get("prompt_token_ids")
        if (
            not isinstance(prompt_token_ids, list)
            or not prompt_token_ids
            or any(
                isinstance(token_id, bool)
                or not isinstance(token_id, int)
                for token_id in prompt_token_ids
            )
        ):
            raise DriverError(
                "invalid_workload_manifest",
                f"invalid prompt_token_ids for {request_id}",
            )
        if row.get("prompt_token_count") != len(prompt_token_ids):
            raise DriverError(
                "invalid_workload_manifest",
                f"prompt_token_count mismatch for {request_id}",
            )
        requested_output_tokens = row.get(
            "requested_output_tokens"
        )
        if (
            isinstance(requested_output_tokens, bool)
            or not isinstance(requested_output_tokens, int)
            or requested_output_tokens <= 0
        ):
            raise DriverError(
                "invalid_workload_manifest",
                f"invalid requested_output_tokens for {request_id}",
            )
        sampling = row.get("sampling")
        if not isinstance(sampling, dict):
            raise DriverError(
                "invalid_workload_manifest",
                f"missing sampling contract for {request_id}",
            )
        if sampling.get("max_tokens") != requested_output_tokens:
            raise DriverError(
                "invalid_workload_manifest",
                f"sampling max_tokens mismatch for {request_id}",
            )
        if "warmup" in row and not isinstance(row["warmup"], bool):
            raise DriverError(
                "invalid_workload_manifest",
                f"warmup must be boolean for {request_id}",
            )
        for field in ("phase", "service_time_bucket"):
            if field in row and (
                not isinstance(row[field], str) or not row[field]
            ):
                raise DriverError(
                    "invalid_workload_manifest",
                    f"{field} must be a non-empty string for {request_id}",
                )
        if "starvation_deadline_ns" in row:
            starvation_deadline_ns = row["starvation_deadline_ns"]
            if (
                isinstance(starvation_deadline_ns, bool)
                or not isinstance(starvation_deadline_ns, int)
                or starvation_deadline_ns <= 0
            ):
                raise DriverError(
                    "invalid_workload_manifest",
                    "starvation_deadline_ns must be a positive integer "
                    f"for {request_id}",
                )


def _sampling_params(request: dict):
    from tinyvllm.sampling_params import SamplingParams

    sampling = request["sampling"]
    return SamplingParams(
        temperature=float(sampling["temperature"]),
        max_tokens=int(sampling["max_tokens"]),
        ignore_eos=bool(sampling["ignore_eos"]),
    )


def _initial_lifecycle(
    request: dict,
    *,
    seq_id: int,
    scheduled_arrival_ns: int,
    actual_arrival_ns: int,
) -> dict:
    return {
        "request_id": request["request_id"],
        "seq_id": seq_id,
        "scheduled_arrival_ns": scheduled_arrival_ns,
        "actual_arrival_ns": actual_arrival_ns,
        "first_scheduled_ns": None,
        "first_token_ns": None,
        "token_timestamps_ns": [],
        "completion_ns": None,
        "output_token_ids": [],
        "prompt_token_count": request["prompt_token_count"],
        "requested_output_tokens": request[
            "requested_output_tokens"
        ],
        "warmup": request.get("warmup", False),
        "phase": request.get("phase"),
        "service_time_bucket": request.get("service_time_bucket"),
        "starvation_deadline_ns": request.get(
            "starvation_deadline_ns"
        ),
        "finish_reason": None,
        "error": None,
    }


def _lookup_request_id(
    request_id_by_seq: dict[int, str],
    seq_id,
) -> str:
    if (
        isinstance(seq_id, bool)
        or not isinstance(seq_id, int)
        or seq_id not in request_id_by_seq
    ):
        raise DriverError(
            "unexpected_sequence_event",
            f"unexpected sequence event: {seq_id}",
        )
    return request_id_by_seq[seq_id]


def _memory_row(observation: dict, step_index: int, timestamp_ns: int):
    queue_after = observation.get("queue_after")
    memory = observation.get("memory")
    if not isinstance(queue_after, dict) or not isinstance(memory, dict):
        raise DriverError(
            "malformed_step_observation",
            "step observation requires queue_after and memory",
        )
    if not _is_int(timestamp_ns) or timestamp_ns < 0:
        raise DriverError(
            "malformed_step_observation",
            "memory timestamp_ns must be a non-negative integer",
        )
    for field, value in memory.items():
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or value < 0
        ):
            raise DriverError(
                "malformed_step_observation",
                f"{field} must be finite and non-negative",
            )
    block_fields = (
        "free_kv_blocks",
        "used_kv_blocks",
        "total_kv_blocks",
        "kv_block_size_tokens",
    )
    missing = [
        key for key in block_fields
        if key not in queue_after
    ]
    if missing:
        raise DriverError(
            "malformed_step_observation",
            "missing queue_after fields: " + ", ".join(missing),
        )
    total_kv_blocks = queue_after["total_kv_blocks"]
    kv_capacity_bytes = memory.get("kv_capacity_bytes")
    if (
        isinstance(total_kv_blocks, bool)
        or not isinstance(total_kv_blocks, int)
        or total_kv_blocks <= 0
        or isinstance(kv_capacity_bytes, bool)
        or not isinstance(kv_capacity_bytes, int)
        or kv_capacity_bytes < 0
        or kv_capacity_bytes % total_kv_blocks != 0
    ):
        raise DriverError(
            "malformed_step_observation",
            "kv_capacity_bytes must divide evenly across total_kv_blocks",
        )
    return {
        "step_index": step_index,
        "timestamp_ns": timestamp_ns,
        **memory,
        **{key: queue_after[key] for key in block_fields},
        "kv_block_bytes": kv_capacity_bytes // total_kv_blocks,
    }


def _is_int(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _validate_timestamp_ns(value, field: str) -> int:
    if not _is_int(value) or value < 0:
        raise DriverError(
            "malformed_step_observation",
            f"{field} must be a non-negative integer",
        )
    return value


def _validate_int_list(observation: dict, field: str):
    value = observation[field]
    if not isinstance(value, list) or any(
        not _is_int(item) for item in value
    ):
        raise DriverError(
            "malformed_step_observation",
            f"{field} must be a list of integers",
        )


def _validate_p5_observation(observation: dict):
    required_fields = P5_DECISION_FIELDS + P5_POSTPROCESS_FIELDS
    missing = [
        field for field in required_fields
        if field not in observation
    ]
    if missing:
        raise DriverError(
            "malformed_step_observation",
            "missing P5 observation fields: " + ", ".join(missing),
        )

    required_int_fields = (
        "decision_now_ns",
        "target_gap_ns",
        "reserve_ns",
        "cost_intercept_ns",
        "cost_per_prefill_token_ns",
        "actual_prefill_tokens",
        "step_end_ns",
        "actual_step_duration_ns",
    )
    nullable_int_fields = (
        "oldest_decode_seq_id",
        "oldest_decode_progress_ns",
        "oldest_decode_age_ns",
        "remaining_slack_ns",
        "predicted_step_ns",
        "selected_chunk_tokens",
    )
    for field in required_int_fields:
        if not _is_int(observation[field]):
            raise DriverError(
                "malformed_step_observation",
                f"{field} must be an integer",
            )
    for field in nullable_int_fields:
        value = observation[field]
        if value is not None and not _is_int(value):
            raise DriverError(
                "malformed_step_observation",
                f"{field} must be an integer or null",
            )

    for field in (
        "candidate_chunk_tokens",
        "scheduled_decode_seq_ids",
        "finished_progress_entries_removed",
    ):
        _validate_int_list(observation, field)
    for field in ("demand_state_before", "demand_state_after"):
        if not isinstance(observation[field], str):
            raise DriverError(
                "malformed_step_observation",
                f"{field} must be a string",
            )
    for field in ("suppression_reason", "clock_invalid_reason"):
        value = observation[field]
        if value is not None and not isinstance(value, str):
            raise DriverError(
                "malformed_step_observation",
                f"{field} must be a string or null",
            )
    if not isinstance(observation["clock_invalid"], bool):
        raise DriverError(
            "malformed_step_observation",
            "clock_invalid must be a boolean",
        )

    progress_updates = observation["decode_progress_updates"]
    if not isinstance(progress_updates, dict) or any(
        not isinstance(seq_id, str) or not _is_int(timestamp_ns)
        for seq_id, timestamp_ns in progress_updates.items()
    ):
        raise DriverError(
            "malformed_step_observation",
            "decode_progress_updates must map strings to integers",
        )


def run_case(
    *,
    case_spec: dict,
    workload_rows: list[dict],
    engine_factory,
    clock_ns,
    output_dir: Path,
) -> dict:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = output_dir / "stdout.log"
    stderr_path = output_dir / "stderr.log"
    stdout_path.touch()
    stderr_path.touch()
    timeline_writer = AppendOnlyJsonl(
        output_dir / "request_timeline.jsonl"
    )
    scheduler_writer = AppendOnlyJsonl(
        output_dir / "scheduler_trace.jsonl"
    )
    memory_writer = AppendOnlyJsonl(
        output_dir / "memory_trace.jsonl"
    )

    result = {
        "case_id": (
            case_spec.get("case_id")
            if isinstance(case_spec, dict)
            else None
        ),
        "status": "INCOMPLETE",
        "error_type": None,
        "error": None,
        "request_count": (
            len(workload_rows)
            if isinstance(workload_rows, list)
            else 0
        ),
        "completed_request_count": 0,
        "step_count": 0,
    }
    lifecycle_by_request: dict[str, dict] = {}
    request_id_by_seq: dict[int, str] = {}
    epoch_ns = None
    exitcode = 1

    try:
        _validate_case_spec(case_spec)
        _validate_workload(workload_rows)
        pending_requests = deque(workload_rows)
        engine = engine_factory(case_spec)
        epoch_ns = clock_ns()
        deadline_ns = (
            epoch_ns
            + workload_rows[-1]["arrival_offset_ns"]
            + case_spec["drain_timeout_ns"]
        )
        step_index = 0

        while pending_requests or not engine.is_finished():
            now_ns = clock_ns()
            if now_ns >= deadline_ns:
                raise DriverError(
                    "drain_timeout",
                    "case exceeded drain_timeout_ns",
                )

            while (
                pending_requests
                and (
                    epoch_ns
                    + pending_requests[0]["arrival_offset_ns"]
                    <= now_ns
                )
            ):
                request = pending_requests.popleft()
                scheduled_ns = (
                    epoch_ns + request["arrival_offset_ns"]
                )
                actual_ns = clock_ns()
                before_ids = {
                    seq.seq_id
                    for seq in engine.scheduler.waiting
                }
                try:
                    engine.add_request(
                        request["prompt_token_ids"],
                        _sampling_params(request),
                    )
                except Exception as exc:
                    raise DriverError(
                        "admission_error",
                        f"{request['request_id']}: {exc}",
                    ) from exc
                appended = [
                    seq for seq in engine.scheduler.waiting
                    if seq.seq_id not in before_ids
                ]
                if len(appended) != 1:
                    raise DriverError(
                        "ambiguous_sequence_binding",
                        "request admission must append exactly one "
                        f"new waiting sequence; got {len(appended)}",
                    )
                seq_id = appended[0].seq_id
                if seq_id in request_id_by_seq:
                    raise DriverError(
                        "ambiguous_sequence_binding",
                        f"duplicate sequence binding: {seq_id}",
                    )
                request_id = request["request_id"]
                request_id_by_seq[seq_id] = request_id
                lifecycle_by_request[request_id] = (
                    _initial_lifecycle(
                        request,
                        seq_id=seq_id,
                        scheduled_arrival_ns=scheduled_ns,
                        actual_arrival_ns=actual_ns,
                    )
                )
                now_ns = clock_ns()

            if not engine.is_finished():
                driver_step_start_ns = clock_ns()
                outputs, num_tokens = engine.step()
                driver_step_end_ns = clock_ns()
                observation = dict(
                    engine.last_step_observation or {}
                )
                observation.update({
                    "step_index": step_index,
                    "driver_step_start_ns": driver_step_start_ns,
                    "driver_step_end_ns": driver_step_end_ns,
                    "num_tokens_returned": num_tokens,
                })
                if case_spec["policy"] == "P5":
                    _validate_p5_observation(observation)
                token_event_ns = _validate_timestamp_ns(
                    observation.get(
                        "step_end_ns",
                        driver_step_end_ns,
                    ),
                    "step_end_ns",
                )
                memory_row = _memory_row(
                    observation,
                    step_index,
                    driver_step_end_ns,
                )
                scheduler_writer.append(observation)
                memory_writer.append(memory_row)

                scheduled_rows = observation.get("scheduled")
                if not isinstance(scheduled_rows, list):
                    raise DriverError(
                        "malformed_step_observation",
                        "scheduled must be a list",
                    )
                for scheduled in scheduled_rows:
                    if not isinstance(scheduled, dict):
                        raise DriverError(
                            "malformed_step_observation",
                            "scheduled row must be an object",
                        )
                    request_id = _lookup_request_id(
                        request_id_by_seq,
                        scheduled.get("seq_id"),
                    )
                    lifecycle = lifecycle_by_request[request_id]
                    if lifecycle["first_scheduled_ns"] is None:
                        lifecycle["first_scheduled_ns"] = (
                            driver_step_start_ns
                        )

                token_deltas = observation.get(
                    "new_completion_tokens_by_seq"
                )
                if not isinstance(token_deltas, dict):
                    raise DriverError(
                        "malformed_step_observation",
                        "new_completion_tokens_by_seq must be an object",
                    )
                for raw_seq_id, delta in token_deltas.items():
                    try:
                        seq_id = int(raw_seq_id)
                    except (TypeError, ValueError) as exc:
                        raise DriverError(
                            "unexpected_sequence_event",
                            f"invalid sequence id: {raw_seq_id}",
                        ) from exc
                    request_id = _lookup_request_id(
                        request_id_by_seq,
                        seq_id,
                    )
                    if not isinstance(delta, list) or any(
                        isinstance(token_id, bool)
                        or not isinstance(token_id, int)
                        for token_id in delta
                    ):
                        raise DriverError(
                            "malformed_step_observation",
                            f"invalid token delta for sequence {seq_id}",
                        )
                    lifecycle = lifecycle_by_request[request_id]
                    if delta:
                        if lifecycle["first_token_ns"] is None:
                            lifecycle["first_token_ns"] = token_event_ns
                        lifecycle["token_timestamps_ns"].extend(
                            [token_event_ns] * len(delta)
                        )
                        lifecycle["output_token_ids"].extend(delta)

                if not isinstance(outputs, list):
                    raise DriverError(
                        "malformed_engine_output",
                        "engine outputs must be a list",
                    )
                for output in outputs:
                    if (
                        not isinstance(output, tuple)
                        or len(output) != 2
                    ):
                        raise DriverError(
                            "malformed_engine_output",
                            "engine output must be (seq_id, token_ids)",
                        )
                    seq_id, token_ids = output
                    request_id = _lookup_request_id(
                        request_id_by_seq,
                        seq_id,
                    )
                    lifecycle = lifecycle_by_request[request_id]
                    if list(token_ids) != lifecycle[
                        "output_token_ids"
                    ]:
                        raise DriverError(
                            "token_delta_mismatch",
                            f"output/delta mismatch for sequence {seq_id}",
                        )
                    if (
                        len(token_ids)
                        != lifecycle["requested_output_tokens"]
                    ):
                        raise DriverError(
                            "token_delta_mismatch",
                            f"output token count mismatch for sequence "
                            f"{seq_id}",
                        )
                    lifecycle["completion_ns"] = token_event_ns
                    lifecycle["finish_reason"] = "length"
                step_index += 1
                result["step_count"] = step_index
            elif pending_requests:
                remaining_ns = (
                    epoch_ns
                    + pending_requests[0]["arrival_offset_ns"]
                    - clock_ns()
                )
                if remaining_ns > 0:
                    time.sleep(
                        min(
                            remaining_ns / 1_000_000_000,
                            0.001,
                        )
                    )

        expected_request_ids = [
            row["request_id"] for row in workload_rows
        ]
        if set(lifecycle_by_request) != set(expected_request_ids):
            raise DriverError(
                "request_set_mismatch",
                "not every manifest request has one lifecycle record",
            )
        for request_id in expected_request_ids:
            lifecycle = lifecycle_by_request[request_id]
            starvation_deadline_ns = lifecycle.get(
                "starvation_deadline_ns"
            )
            if (
                starvation_deadline_ns is not None
                and lifecycle["first_scheduled_ns"] is not None
                and (
                    lifecycle["first_scheduled_ns"]
                    - lifecycle["scheduled_arrival_ns"]
                    > starvation_deadline_ns
                )
            ):
                raise DriverError(
                    "starved_request",
                    f"request exceeded starvation deadline: {request_id}",
                )
            if (
                lifecycle["first_scheduled_ns"] is None
                or lifecycle["first_token_ns"] is None
                or lifecycle["completion_ns"] is None
                or lifecycle["finish_reason"] != "length"
                or len(lifecycle["output_token_ids"])
                != lifecycle["requested_output_tokens"]
            ):
                raise DriverError(
                    "incomplete_request",
                    f"incomplete lifecycle for {request_id}",
                )
        result["completed_request_count"] = len(
            expected_request_ids
        )
        result["status"] = "PASS"
        exitcode = 0
    except DriverError as exc:
        result["error_type"] = exc.error_type
        result["error"] = str(exc)
        with stderr_path.open(
            "a",
            encoding="utf-8",
        ) as stderr_handle:
            stderr_handle.write(
                f"{exc.error_type}: {exc}\n"
            )
            stderr_handle.flush()
            os.fsync(stderr_handle.fileno())
    except Exception as exc:
        result["error_type"] = "driver_exception"
        result["error"] = str(exc)
        with stderr_path.open(
            "a",
            encoding="utf-8",
        ) as stderr_handle:
            traceback.print_exc(file=stderr_handle)
            stderr_handle.flush()
            os.fsync(stderr_handle.fileno())
    finally:
        if result["status"] != "PASS":
            for request in (
                workload_rows
                if isinstance(workload_rows, list)
                else []
            ):
                request_id = request.get("request_id")
                lifecycle = lifecycle_by_request.get(request_id)
                if lifecycle is not None:
                    lifecycle["error"] = result["error"]
        if isinstance(workload_rows, list):
            for request in workload_rows:
                request_id = request.get("request_id")
                lifecycle = lifecycle_by_request.get(request_id)
                if lifecycle is not None:
                    timeline_writer.append(lifecycle)
        timeline_writer.close()
        scheduler_writer.close()
        memory_writer.close()
        _atomic_write_json(
            output_dir / "case_result.json",
            result,
        )
        _write_text(
            output_dir / "exitcode",
            f"{exitcode}\n",
        )
    return result


def _load_json(path: Path):
    return json.loads(Path(path).read_text())


def _load_jsonl(path: Path):
    rows = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.endswith("\n"):
                raise ValueError(
                    f"truncated JSONL line {line_number}: {path}"
                )
            rows.append(json.loads(line))
    return rows


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Run one TinyLLMForge arrival-load case",
    )
    parser.add_argument(
        "--case-spec",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--workload-manifest",
        type=Path,
        required=True,
    )
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv)
    case_spec = _load_json(args.case_spec)
    workload_rows = _load_jsonl(args.workload_manifest)

    def engine_factory(spec):
        from tinyvllm.engine.llm_engine import LLMEngine

        return LLMEngine(
            args.model,
            **spec.get("resolved_config", {}),
        )

    result = run_case(
        case_spec=case_spec,
        workload_rows=workload_rows,
        engine_factory=engine_factory,
        clock_ns=time.monotonic_ns,
        output_dir=args.output_dir,
    )
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
