"""Dependency-light tests for the inline arrival-load driver."""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import tempfile
import types
from collections import deque
from pathlib import Path
from types import SimpleNamespace


_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
tinyvllm_package = types.ModuleType("tinyvllm")
tinyvllm_package.__path__ = [str(_REPO_ROOT / "tinyvllm")]
sys.modules.setdefault("tinyvllm", tinyvllm_package)
sampling_spec = importlib.util.spec_from_file_location(
    "tinyvllm.sampling_params",
    _REPO_ROOT / "tinyvllm" / "sampling_params.py",
)
sampling_module = importlib.util.module_from_spec(sampling_spec)
sys.modules["tinyvllm.sampling_params"] = sampling_module
sampling_spec.loader.exec_module(sampling_module)
_DRIVER_PATH = _THIS_DIR / "arrival_load_driver.py"
assert _DRIVER_PATH.is_file(), (
    "tools/arrival_load_driver.py is missing"
)
_SPEC = importlib.util.spec_from_file_location(
    "arrival_load_driver_under_test",
    _DRIVER_PATH,
)
driver = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(driver)


class IncrementingClock:
    def __init__(self, start_ns: int = 100, step_ns: int = 10):
        self.value = start_ns - step_ns
        self.step_ns = step_ns

    def __call__(self):
        self.value += self.step_ns
        return self.value


class FakeSequence:
    def __init__(self, seq_id: int):
        self.seq_id = seq_id


class FakeEngine:
    def __init__(self, case_spec):
        del case_spec
        self.scheduler = SimpleNamespace(waiting=deque())
        self.last_step_observation = None
        self.next_seq_id = 10
        self.active = None
        self.progress = {}

    def add_request(self, prompt_token_ids, sampling_params):
        del prompt_token_ids
        seq = FakeSequence(self.next_seq_id)
        self.next_seq_id += 1
        seq.requested_tokens = sampling_params.max_tokens
        self.progress[seq.seq_id] = []
        self.scheduler.waiting.append(seq)

    def is_finished(self):
        return self.active is None and not self.scheduler.waiting

    def step(self):
        if self.active is None:
            self.active = self.scheduler.waiting.popleft()
        seq = self.active
        produced = self.progress[seq.seq_id]
        if seq.requested_tokens == 3 and not produced:
            delta = [101]
        elif seq.requested_tokens == 3:
            delta = [102, 103]
        else:
            delta = [201]
        produced.extend(delta)
        outputs = []
        if len(produced) == seq.requested_tokens:
            outputs = [(seq.seq_id, list(produced))]
            self.active = None
        self.last_step_observation = {
            "policy_branch": "decode_fallback",
            "batch_kind": None,
            "is_prefill": False,
            "do_sample": True,
            "scheduled": [{
                "seq_id": seq.seq_id,
                "is_decode": True,
                "do_sample": True,
                "prefill_chunk_start": 0,
                "prefill_chunk_end": 0,
                "prefill_chunk_final": False,
            }],
            "queue_before": {"waiting_seq_ids": []},
            "queue_after": {
                "waiting_seq_ids": [
                    waiting.seq_id
                    for waiting in self.scheduler.waiting
                ],
                "prefilling_seq_ids": [],
                "running_seq_ids": (
                    [] if self.active is None else [self.active.seq_id]
                ),
                "free_kv_blocks": 7,
                "used_kv_blocks": 1,
                "total_kv_blocks": 8,
                "kv_block_size_tokens": 4,
                "consecutive_prefill_chunks": 0,
            },
            "new_completion_tokens_by_seq": {
                seq.seq_id: delta,
            },
            "finished_seq_ids": (
                [seq.seq_id] if outputs else []
            ),
            "memory": {
                "cuda_allocated_bytes": 1000,
                "cuda_reserved_bytes": 1200,
                "cuda_peak_allocated_bytes": 1400,
                "cuda_peak_reserved_bytes": 1600,
                "kv_capacity_bytes": 800,
            },
        }
        return outputs, -1


class StuckFakeEngine(FakeEngine):
    def step(self):
        if self.active is None:
            self.active = self.scheduler.waiting.popleft()
        seq = self.active
        self.last_step_observation = {
            "policy_branch": "decode_fallback",
            "batch_kind": None,
            "is_prefill": False,
            "do_sample": True,
            "scheduled": [{"seq_id": seq.seq_id}],
            "queue_before": {},
            "queue_after": {
                "waiting_seq_ids": [],
                "prefilling_seq_ids": [],
                "running_seq_ids": [seq.seq_id],
                "free_kv_blocks": 7,
                "used_kv_blocks": 1,
                "total_kv_blocks": 8,
                "kv_block_size_tokens": 4,
                "consecutive_prefill_chunks": 0,
            },
            "new_completion_tokens_by_seq": {
                seq.seq_id: [],
            },
            "finished_seq_ids": [],
            "memory": {
                "cuda_allocated_bytes": 1000,
                "cuda_reserved_bytes": 1200,
                "cuda_peak_allocated_bytes": 1400,
                "cuda_peak_reserved_bytes": 1600,
                "kv_capacity_bytes": 800,
            },
        }
        return [], -1


class AdmissionFailureEngine(FakeEngine):
    def add_request(self, prompt_token_ids, sampling_params):
        del prompt_token_ids, sampling_params
        raise ValueError("synthetic admission failure")


class AmbiguousBindingEngine(FakeEngine):
    def add_request(self, prompt_token_ids, sampling_params):
        super().add_request(prompt_token_ids, sampling_params)
        self.scheduler.waiting.append(
            FakeSequence(self.next_seq_id)
        )
        self.next_seq_id += 1


class UnexpectedSequenceEngine(FakeEngine):
    def step(self):
        outputs, num_tokens = super().step()
        self.last_step_observation["scheduled"][0]["seq_id"] = 999
        return outputs, num_tokens


class TokenMismatchEngine(FakeEngine):
    def step(self):
        outputs, num_tokens = super().step()
        if outputs:
            seq_id, token_ids = outputs[0]
            outputs = [(seq_id, token_ids + [999])]
        return outputs, num_tokens


def _case_spec(**overrides):
    spec = {
        "case_id": "steady-P0-r0",
        "scenario": "steady_moderate",
        "policy": "P0",
        "repetition": 0,
        "drain_timeout_ns": 10_000,
        "resolved_config": {},
    }
    spec.update(overrides)
    return spec


def _request(
    request_id: str,
    arrival_offset_ns: int,
    requested_output_tokens: int,
):
    return {
        "request_id": request_id,
        "scenario": "steady_moderate",
        "arrival_offset_ns": arrival_offset_ns,
        "prompt_token_ids": [1, 2, 3],
        "prompt_token_count": 3,
        "requested_output_tokens": requested_output_tokens,
        "sampling": {
            "temperature": 0.0,
            "ignore_eos": True,
            "max_tokens": requested_output_tokens,
        },
    }


def _workload():
    return [
        _request("multi", 0, 3),
        _request("single", 35, 1),
    ]


def _jsonl(path: Path):
    return [
        json.loads(line)
        for line in path.read_text().splitlines()
        if line
    ]


def _run(engine_factory=FakeEngine, **case_overrides):
    temporary = tempfile.TemporaryDirectory()
    output_dir = Path(temporary.name)
    result = driver.run_case(
        case_spec=_case_spec(**case_overrides),
        workload_rows=_workload(),
        engine_factory=engine_factory,
        clock_ns=IncrementingClock(),
        output_dir=output_dir,
    )
    return temporary, output_dir, result


def test_driver_binds_new_waiting_sequence_and_accounts_injection_lag():
    temporary, output_dir, result = _run()
    try:
        rows = _jsonl(output_dir / "request_timeline.jsonl")
        assert result["status"] == "PASS"
        assert [row["request_id"] for row in rows] == [
            "multi",
            "single",
        ]
        assert rows[0]["scheduled_arrival_ns"] < (
            rows[0]["actual_arrival_ns"]
        )
        assert len({row["seq_id"] for row in rows}) == len(rows)
        assert all(row["finish_reason"] == "length" for row in rows)
        assert all(row["error"] is None for row in rows)
        assert (output_dir / "exitcode").read_text() == "0\n"
    finally:
        temporary.cleanup()


def test_driver_records_multiple_tokens_at_one_step_timestamp():
    temporary, output_dir, result = _run()
    try:
        assert result["status"] == "PASS"
        row = next(
            row for row in _jsonl(
                output_dir / "request_timeline.jsonl"
            )
            if row["request_id"] == "multi"
        )
        assert row["output_token_ids"] == [101, 102, 103]
        assert (
            row["token_timestamps_ns"][-2]
            == row["token_timestamps_ns"][-1]
        )
        assert row["completion_ns"] == row["token_timestamps_ns"][-1]
    finally:
        temporary.cleanup()


def test_driver_watchdog_preserves_partial_append_only_evidence():
    temporary, output_dir, result = _run(
        StuckFakeEngine,
        drain_timeout_ns=120,
    )
    try:
        assert result["status"] == "INCOMPLETE"
        assert result["error_type"] == "drain_timeout"
        assert (output_dir / "scheduler_trace.jsonl").read_bytes()
        assert (output_dir / "memory_trace.jsonl").read_bytes()
        assert (output_dir / "request_timeline.jsonl").read_bytes()
        assert (output_dir / "exitcode").read_text() == "1\n"
    finally:
        temporary.cleanup()


def test_driver_drain_timeout_starts_after_final_scheduled_arrival():
    with tempfile.TemporaryDirectory() as temporary:
        output_dir = Path(temporary)
        result = driver.run_case(
            case_spec=_case_spec(drain_timeout_ns=50),
            workload_rows=[
                _request("late", 200, 1),
            ],
            engine_factory=FakeEngine,
            clock_ns=IncrementingClock(
                start_ns=100,
                step_ns=10,
            ),
            output_dir=output_dir,
        )
        assert result["status"] == "PASS"
        timeline = _jsonl(
            output_dir / "request_timeline.jsonl"
        )
        assert timeline[0]["scheduled_arrival_ns"] == 300
        assert timeline[0]["actual_arrival_ns"] >= 300


def test_driver_fails_closed_on_admission_exception():
    temporary, output_dir, result = _run(AdmissionFailureEngine)
    try:
        assert result["status"] == "INCOMPLETE"
        assert result["error_type"] == "admission_error"
        assert "synthetic admission failure" in result["error"]
        assert (output_dir / "stderr.log").read_bytes()
    finally:
        temporary.cleanup()


def test_driver_rejects_ambiguous_new_waiting_binding():
    temporary, output_dir, result = _run(AmbiguousBindingEngine)
    try:
        assert result["status"] == "INCOMPLETE"
        assert result["error_type"] == "ambiguous_sequence_binding"
        assert "exactly one" in result["error"]
    finally:
        temporary.cleanup()


def test_driver_rejects_unexpected_sequence_event():
    temporary, output_dir, result = _run(UnexpectedSequenceEngine)
    try:
        assert result["status"] == "INCOMPLETE"
        assert result["error_type"] == "unexpected_sequence_event"
        assert "999" in result["error"]
    finally:
        temporary.cleanup()


def test_driver_rejects_token_count_delta_mismatch():
    temporary, output_dir, result = _run(TokenMismatchEngine)
    try:
        assert result["status"] == "INCOMPLETE"
        assert result["error_type"] == "token_delta_mismatch"
    finally:
        temporary.cleanup()


def test_driver_rejects_malformed_manifest_order_before_engine_start():
    with tempfile.TemporaryDirectory() as temporary:
        output_dir = Path(temporary)
        workload = [
            _request("late", 10, 1),
            _request("early", 0, 1),
        ]
        result = driver.run_case(
            case_spec=_case_spec(),
            workload_rows=workload,
            engine_factory=lambda case_spec: (
                _ for _ in ()
            ).throw(AssertionError("engine must not start")),
            clock_ns=IncrementingClock(),
            output_dir=output_dir,
        )
        assert result["status"] == "INCOMPLETE"
        assert result["error_type"] == "invalid_workload_manifest"
        assert "arrival order" in result["error"]


def test_driver_rejects_duplicate_request_ids():
    with tempfile.TemporaryDirectory() as temporary:
        output_dir = Path(temporary)
        request = _request("duplicate", 0, 1)
        result = driver.run_case(
            case_spec=_case_spec(),
            workload_rows=[request, dict(request)],
            engine_factory=FakeEngine,
            clock_ns=IncrementingClock(),
            output_dir=output_dir,
        )
        assert result["status"] == "INCOMPLETE"
        assert result["error_type"] == "invalid_workload_manifest"
        assert "duplicate request_id" in result["error"]


def test_driver_jsonl_files_preserve_final_newline():
    temporary, output_dir, result = _run()
    try:
        assert result["status"] == "PASS"
        for filename in (
            "request_timeline.jsonl",
            "scheduler_trace.jsonl",
            "memory_trace.jsonl",
        ):
            payload = (output_dir / filename).read_bytes()
            assert payload
            assert payload.endswith(b"\n")
            for line in payload.splitlines():
                json.loads(line)
    finally:
        temporary.cleanup()


def main():
    test_driver_binds_new_waiting_sequence_and_accounts_injection_lag()
    test_driver_records_multiple_tokens_at_one_step_timestamp()
    test_driver_watchdog_preserves_partial_append_only_evidence()
    test_driver_drain_timeout_starts_after_final_scheduled_arrival()
    test_driver_fails_closed_on_admission_exception()
    test_driver_rejects_ambiguous_new_waiting_binding()
    test_driver_rejects_unexpected_sequence_event()
    test_driver_rejects_token_count_delta_mismatch()
    test_driver_rejects_malformed_manifest_order_before_engine_start()
    test_driver_rejects_duplicate_request_ids()
    test_driver_jsonl_files_preserve_final_newline()
    print("arrival load driver tests passed")


if __name__ == "__main__":
    main()
