"""Dependency-light tests for staged inference benchmark workers.

Run:
    python3 tools/test_staged_inference_benchmark_worker.py
"""

from __future__ import annotations

from collections import Counter, deque
import importlib.util
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import types


TOOLS_ROOT = Path(__file__).resolve().parent
REPO_ROOT = TOOLS_ROOT.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

tinyvllm_package = types.ModuleType("tinyvllm")
tinyvllm_package.__path__ = [str(REPO_ROOT / "tinyvllm")]
sys.modules.setdefault("tinyvllm", tinyvllm_package)
sampling_spec = importlib.util.spec_from_file_location(
    "tinyvllm.sampling_params",
    REPO_ROOT / "tinyvllm" / "sampling_params.py",
)
sampling_module = importlib.util.module_from_spec(sampling_spec)
sys.modules["tinyvllm.sampling_params"] = sampling_module
sampling_spec.loader.exec_module(sampling_module)

from tools import staged_inference_benchmark_contract as contract
from tools import staged_inference_benchmark_worker as worker


class JumpClock:
    def __init__(self):
        self.first = True

    def __call__(self):
        if self.first:
            self.first = False
            return 0
        return 10_000_000_000


class BatchCompletingEngine:
    def __init__(self, case_spec):
        self.case_spec = case_spec
        self.scheduler = types.SimpleNamespace(waiting=deque())
        self.last_step_observation = None
        self.next_seq_id = 1
        self.prompt_lengths = []

    def add_request(self, prompt_token_ids, sampling_params):
        self.prompt_lengths.append(len(prompt_token_ids))
        sequence = types.SimpleNamespace(
            seq_id=self.next_seq_id,
            requested_tokens=sampling_params.max_tokens,
        )
        self.next_seq_id += 1
        self.scheduler.waiting.append(sequence)

    def is_finished(self):
        return not self.scheduler.waiting

    def step(self):
        sequences = list(self.scheduler.waiting)
        self.scheduler.waiting.clear()
        deltas = {
            sequence.seq_id: [
                sequence.seq_id * 1000 + index
                for index in range(sequence.requested_tokens)
            ]
            for sequence in sequences
        }
        self.last_step_observation = {
            "policy_branch": self.case_spec["policy"],
            "batch_kind": "mixed",
            "is_prefill": True,
            "do_sample": True,
            "scheduled": [
                {
                    "seq_id": sequence.seq_id,
                    "is_decode": False,
                    "do_sample": True,
                    "prefill_chunk_start": 0,
                    "prefill_chunk_end": 1,
                    "prefill_chunk_final": True,
                }
                for sequence in sequences
            ],
            "queue_before": {
                "waiting_seq_ids": [
                    sequence.seq_id for sequence in sequences
                ],
            },
            "queue_after": {
                "waiting_seq_ids": [],
                "prefilling_seq_ids": [],
                "running_seq_ids": [],
                "free_kv_blocks": 7,
                "used_kv_blocks": 1,
                "total_kv_blocks": 8,
                "kv_block_size_tokens": 256,
                "consecutive_prefill_chunks": 0,
            },
            "new_completion_tokens_by_seq": deltas,
            "finished_seq_ids": [
                sequence.seq_id for sequence in sequences
            ],
            "memory": {
                "cuda_allocated_bytes": 1000,
                "cuda_reserved_bytes": 1200,
                "cuda_peak_allocated_bytes": 1400,
                "cuda_peak_reserved_bytes": 1600,
                "kv_capacity_bytes": 8192,
            },
        }
        return [
            (sequence.seq_id, deltas[sequence.seq_id])
            for sequence in sequences
        ], sum(len(tokens) for tokens in deltas.values())


class MissingTokenEvidenceEngine(BatchCompletingEngine):
    def step(self):
        outputs, num_tokens = super().step()
        self.last_step_observation["new_completion_tokens_by_seq"] = {}
        return outputs, num_tokens


def _chunked_spec(policy: str) -> dict:
    case = next(
        row
        for row in contract.build_chunked_case_matrix(
            model_tier="qwen3-0.6b"
        )
        if row["policy"] == policy
    )
    return {
        **case,
        "drain_timeout_ns": 180_000_000_000,
        "workload_rows": contract.build_chunked_workload(),
    }


def _read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def _assert_raises(error_type, message: str, function):
    try:
        function()
    except error_type as error:
        assert message in str(error)
    else:
        raise AssertionError(f"expected {error_type.__name__}")


def test_chunked_worker_passes_only_frozen_policy_fields():
    seen = {}

    def engine_factory(spec):
        seen.update(spec["engine_config"])
        return BatchCompletingEngine(spec)

    with TemporaryDirectory() as temporary:
        result = worker.run_worker(
            _chunked_spec("FAIR_CHUNKED"),
            Path(temporary) / "case",
            engine_factory=engine_factory,
            clock_ns=JumpClock(),
        )

    assert result["status"] == "PASS"
    assert seen == contract.CHUNKED_POLICIES["FAIR_CHUNKED"]
    assert seen["max_num_prefill_tokens_per_step"] == 128
    assert seen["chunked_prefill_decode_first"] is False
    assert seen["chunked_prefill_max_consecutive_chunks"] == 2
    assert seen["chunked_prefill_mixed_batch"] is False
    assert seen["chunked_prefill_adaptive_mixed"] is False
    assert seen["chunked_prefill_slo_mixed"] is False


def test_worker_keeps_warmup_lifecycle_but_excludes_warmup_metrics():
    with TemporaryDirectory() as temporary:
        output_dir = Path(temporary) / "case"
        result = worker.run_worker(
            _chunked_spec("OFF"),
            output_dir,
            engine_factory=BatchCompletingEngine,
            clock_ns=JumpClock(),
        )
        lifecycle = _read_jsonl(output_dir / "request_timeline.jsonl")
        persisted_result = json.loads(
            (output_dir / "case_result.json").read_text(encoding="utf-8")
        )

    assert result["status"] == "PASS"
    assert result["lifecycle_requests"] == 104
    assert result["measured_requests"] == 96
    assert persisted_result == result
    assert Counter(row["warmup"] for row in lifecycle) == {
        True: 8,
        False: 96,
    }
    measured_mix = Counter(
        row["prompt_token_count"]
        for row in lifecycle
        if not row["warmup"]
    )
    assert measured_mix == {64: 58, 512: 24, 4096: 14}


def test_chunked_worker_rejects_unexpected_policy_fields_before_engine_start():
    spec = _chunked_spec("OFF")
    spec["engine_config"]["unexpected_policy_field"] = True
    with TemporaryDirectory() as temporary:
        _assert_raises(
            ValueError,
            "engine_config",
            lambda: worker.run_worker(
                spec,
                Path(temporary) / "case",
                engine_factory=lambda unused: (_ for _ in ()).throw(
                    AssertionError("engine must not start")
                ),
                clock_ns=JumpClock(),
            ),
        )


def test_chunked_worker_rejects_duplicate_request_ids():
    spec = _chunked_spec("OFF")
    spec["workload_rows"][1]["request_id"] = (
        spec["workload_rows"][0]["request_id"]
    )
    spec["workload_sha256"] = contract.canonical_json_sha256(
        spec["workload_rows"]
    )
    with TemporaryDirectory() as temporary:
        _assert_raises(
            ValueError,
            "request ids",
            lambda: worker.run_worker(
                spec,
                Path(temporary) / "case",
                engine_factory=BatchCompletingEngine,
                clock_ns=JumpClock(),
            ),
        )


def test_chunked_worker_fails_closed_on_missing_token_timestamps():
    with TemporaryDirectory() as temporary:
        result = worker.run_worker(
            _chunked_spec("OFF"),
            Path(temporary) / "case",
            engine_factory=MissingTokenEvidenceEngine,
            clock_ns=JumpClock(),
        )

    assert result["status"] == "INCOMPLETE"
    assert result["error_type"] == "token_delta_mismatch"


def test_prefix_worker_delegates_to_existing_profiler():
    captured = {}
    spec = {
        "case_id": "prefix_full__qwen3-0.6b",
        "gate": "prefix",
        "model_tier": "qwen3-0.6b",
        "profile_args": {
            "model": "/models/qwen3-0.6b",
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
        },
    }

    def prefix_runner(args):
        captured.update(vars(args))
        return {"staged_decision": {"classification": "PREFIX_CACHE_GO"}}

    with TemporaryDirectory() as temporary:
        output_dir = Path(temporary) / "case"
        result = worker.run_worker(
            spec,
            output_dir,
            prefix_runner=prefix_runner,
        )

    assert result["status"] == "PASS"
    assert captured["out_dir"] == str(output_dir)
    assert captured["model"] == "/models/qwen3-0.6b"
    assert captured["enforce_eager"] is True
    assert result["summary"]["staged_decision"]["classification"] == (
        "PREFIX_CACHE_GO"
    )


def test_worker_cli_binds_frozen_chunked_workload_jsonl():
    spec = _chunked_spec("OFF")
    workload = spec.pop("workload_rows")
    with TemporaryDirectory() as temporary:
        root = Path(temporary)
        spec_path = root / "case.json"
        workload_path = root / "workload.jsonl"
        spec_path.write_text(
            json.dumps(spec, sort_keys=True),
            encoding="utf-8",
        )
        workload_path.write_text(
            "".join(
                json.dumps(row, sort_keys=True) + "\n"
                for row in workload
            ),
            encoding="utf-8",
        )
        loaded = worker._load_worker_spec(
            spec_path,
            workload_path=workload_path,
        )

    assert loaded["workload_rows"] == workload
    assert loaded["workload_sha256"] == contract.canonical_json_sha256(
        workload
    )


def test_worker_exit_code_accepts_complete_starvation_evidence():
    result = {
        "status": "INCOMPLETE",
        "error_type": "starved_request",
        "request_count": 104,
        "completed_request_count": 104,
        "lifecycle_requests": 104,
    }
    assert worker._worker_exit_code(result) == 0
    assert worker._worker_exit_code({
        **result,
        "completed_request_count": 103,
    }) == 1
    assert worker._worker_exit_code({
        **result,
        "error_type": "drain_timeout",
    }) == 1


def main():
    test_chunked_worker_passes_only_frozen_policy_fields()
    test_worker_keeps_warmup_lifecycle_but_excludes_warmup_metrics()
    test_chunked_worker_rejects_unexpected_policy_fields_before_engine_start()
    test_chunked_worker_rejects_duplicate_request_ids()
    test_chunked_worker_fails_closed_on_missing_token_timestamps()
    test_prefix_worker_delegates_to_existing_profiler()
    test_worker_cli_binds_frozen_chunked_workload_jsonl()
    test_worker_exit_code_accepts_complete_starvation_evidence()
    print("staged inference benchmark worker tests passed")


if __name__ == "__main__":
    main()
