from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "tools/qwen38_tp4_communication_profile_worker.py"


def _load():
    assert MODULE_PATH.is_file(), (
        "Qwen3.8 TP4 communication profile worker is missing"
    )
    spec = importlib.util.spec_from_file_location(
        "qwen38_tp4_communication_profile_worker_under_test",
        MODULE_PATH,
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class _FakeEngine:

    def __init__(self, output_tokens):
        self.output_tokens = output_tokens
        self.added = []
        self.steps = 0
        self.last_step_observation = None
        self.model_runner = SimpleNamespace(
            rank=0,
            world_size=4,
        )
        self.configured = []
        self.reset_calls = 0
        self.closed = False

    def add_request(self, prompt, sampling_params):
        self.added.append((list(prompt), sampling_params))

    def configure_decode_internal_profile(
        self,
        enabled,
        profile_label,
        *,
        timeout_s,
    ):
        self.added = []
        self.steps = 0
        self.configured.append((enabled, profile_label, timeout_s))
        return {"enabled": enabled, "rank_inventory": [0, 1, 2, 3]}

    def reset_peak_memory_stats(self, *, timeout_s):
        self.reset_calls += 1
        return tuple(
            {
                "rank": rank,
                "cuda_peak_allocated_bytes": 100 + rank,
                "cuda_peak_reserved_bytes": 200 + rank,
            }
            for rank in range(4)
        )

    def is_finished(self):
        return self.steps >= self.output_tokens

    def step(self):
        now = 1_000_000 + self.steps * 100_000
        token_deltas = {
            seq_id: [10 + self.steps]
            for seq_id in range(len(self.added))
        }
        self.last_step_observation = {
            "step_end_ns": now,
            "scheduled": [
                {
                    "seq_id": seq_id,
                    "is_decode": self.steps > 0,
                    "do_sample": True,
                    "prefill_chunk_start": 0,
                    "prefill_chunk_end": len(prompt),
                }
                for seq_id, (prompt, _) in enumerate(self.added)
            ],
            "new_completion_tokens_by_seq": token_deltas,
        }
        self.steps += 1
        outputs = (
            [
                (
                    seq_id,
                    [10 + offset for offset in range(self.output_tokens)],
                )
                for seq_id in range(len(self.added))
            ]
            if self.is_finished()
            else []
        )
        return outputs, len(self.added)

    def finalize_decode_internal_profile(self, *, timeout_s):
        assert timeout_s == 30.0
        return {
            "enabled": True,
            "rank_inventory": [0, 1, 2, 3],
            "ranks": [
                {
                    "rank": rank,
                    "enabled": True,
                    "finalization_status": "complete",
                    "steps": [{
                        "rank": rank,
                        "step_index": 1,
                        "is_decode": True,
                        "decode_ordinal": 0,
                        "request_set_sha256": "a" * 64,
                        "cuda_ns": 10_000,
                    }],
                    "layers": [{
                        "rank": rank,
                        "step_index": 1,
                        "decode_ordinal": 0,
                        "request_set_sha256": "a" * 64,
                        "layer_index": 0,
                        "layer_role": "full_attention",
                        "cuda_ns": 9_000,
                    }],
                    "operations": [{
                        "rank": rank,
                        "step_index": 1,
                        "decode_ordinal": 0,
                        "request_set_sha256": "a" * 64,
                        "layer_index": 0,
                        "layer_role": "full_attention",
                        "operation_ordinal": 0,
                        "operation_class": "gemm",
                        "operation_name": "qkv_projection",
                        "cuda_ns": 4_000,
                    }],
                    "collectives": [],
                }
                for rank in range(4)
            ],
        }

    def memory_snapshots(self, *, timeout_s):
        assert timeout_s == 30.0
        return tuple(
            {
                "rank": rank,
                "cuda_peak_allocated_bytes": 300 + rank,
                "cuda_peak_reserved_bytes": 400 + rank,
            }
            for rank in range(4)
        )

    def exit(self):
        self.closed = True
        return {
            "process_group_destroyed": True,
            "rank_exit_codes": [0, 0, 0, 0],
            "owned_children_remaining": [],
            "rank_cleanup_receipts": [
                {
                    "rank": rank,
                    "process_group_destroyed": True,
                }
                for rank in range(4)
            ],
        }


def test_build_request_specs_freezes_exact_length_and_concurrency():
    worker = _load()

    requests = worker.build_request_specs(
        prompt_tokens=2048,
        output_tokens=128,
        concurrency=4,
    )

    assert len(requests) == 4
    assert all(len(row["prompt_token_ids"]) == 2048 for row in requests)
    assert all(row["output_tokens"] == 128 for row in requests)
    assert len({
        tuple(row["prompt_token_ids"])
        for row in requests
    }) == 4
    assert all(
        min(row["prompt_token_ids"]) >= 11
        and max(row["prompt_token_ids"]) < 2048
        for row in requests
    )


def test_run_profile_case_profiles_all_ranks_and_closes_engine():
    worker = _load()
    engine = _FakeEngine(output_tokens=128)
    ticks = iter(range(0, 100_000_000, 100_000))

    result = worker.run_profile_case(
        attempt="attempt-r1",
        workload="Q0",
        workload_family="online",
        phase="measured",
        repetition=3,
        prompt_tokens=256,
        output_tokens=128,
        concurrency=4,
        model_root=Path("/model"),
        timeout_s=30.0,
        engine_factory=lambda *_args, **_kwargs: engine,
        sampling_params_factory=lambda **kwargs: dict(kwargs),
        clock_ns=lambda: next(ticks),
        reset_sequence_ids=lambda: None,
    )

    assert result["classification"] == "PASS"
    assert result["case_id"] == "Q0__measured__r3"
    assert result["rank_inventory"] == [0, 1, 2, 3]
    assert result["profile"]["rank_inventory"] == [0, 1, 2, 3]
    assert "operations" not in result["profile"]["ranks"][0]
    assert len(result["requests"]) == 4
    assert all(row["generated_tokens"] == 128 for row in result["requests"])
    assert all(row["ttft_ns"] >= 0 for row in result["requests"])
    assert all(row["e2e_ns"] >= row["ttft_ns"] for row in result["requests"])
    assert len(result["memory"]) == 4
    assert result["cleanup"]["process_group_destroyed"] is True
    assert result["cleanup"]["owned_children_remaining"] == []
    assert engine.configured == [
        (
            True,
            "attempt=attempt-r1/workload=Q0/repetition=3",
            30.0,
        )
    ]
    assert engine.reset_calls == 1
    assert engine.closed is True


def test_run_profile_campaign_reuses_one_engine_for_all_structured_cases():
    worker = _load()
    engine = _FakeEngine(output_tokens=128)
    factory_calls = []
    resets = []
    ticks = iter(range(0, 1_000_000_000, 100_000))
    cases = [
        {
            "workload": "P0",
            "workload_family": "causal",
            "phase": "warmup",
            "repetition": repetition,
            "prompt_tokens": 256,
            "output_tokens": 128,
            "concurrency": 1,
        }
        for repetition in (0, 1)
    ]

    result = worker.run_profile_campaign(
        attempt="attempt-r1",
        cases=cases,
        model_root=Path("/model"),
        timeout_s=30.0,
        engine_factory=lambda *_args, **_kwargs: (
            factory_calls.append(_kwargs) or engine
        ),
        sampling_params_factory=lambda **kwargs: dict(kwargs),
        clock_ns=lambda: next(ticks),
        reset_sequence_ids=lambda: resets.append(True),
    )

    assert result["classification"] == "PASS"
    assert [row["case_id"] for row in result["cases"]] == [
        "P0__warmup__r0",
        "P0__warmup__r1",
    ]
    assert len(factory_calls) == 1
    assert len(resets) == 2
    assert engine.reset_calls == 2
    assert engine.closed is True
    assert result["cleanup"]["process_group_destroyed"] is True


def test_run_profile_campaign_streams_full_cases_and_retains_receipts_only():
    worker = _load()
    engine = _FakeEngine(output_tokens=128)
    streamed = []
    cases = [{
        "workload": "P0",
        "workload_family": "causal",
        "phase": "warmup",
        "repetition": 0,
        "prompt_tokens": 256,
        "output_tokens": 128,
        "concurrency": 1,
    }]

    result = worker.run_profile_campaign(
        attempt="attempt-r1",
        cases=cases,
        model_root=Path("/model"),
        timeout_s=30.0,
        engine_factory=lambda *_args, **_kwargs: engine,
        sampling_params_factory=lambda **kwargs: dict(kwargs),
        clock_ns=iter(range(0, 1_000_000_000, 100_000)).__next__,
        reset_sequence_ids=lambda: None,
        case_sink=lambda row: streamed.append(row),
        retain_case_profiles=False,
    )

    assert len(streamed) == 1
    assert "profile" in streamed[0]
    assert result["cases"] == [{
        "classification": "PASS",
        "case_id": "P0__warmup__r0",
        "decode_time_ns": streamed[0]["decode_time_ns"],
    }]


def test_main_prints_bounded_receipt_instead_of_full_profile(
    tmp_path,
    capsys,
    monkeypatch,
):
    worker = _load()
    output = tmp_path / "case.json"
    result = {
        "classification": "PASS",
        "case_id": "Q0__warmup__r0",
        "profile": {"payload": "must-only-exist-in-artifact"},
    }
    monkeypatch.setattr(
        worker,
        "run_profile_case",
        lambda **_kwargs: result,
    )

    returncode = worker.main([
        "--attempt",
        "attempt-r1",
        "--workload",
        "Q0",
        "--phase",
        "warmup",
        "--repetition",
        "0",
        "--model-root",
        "/model",
        "--output",
        str(output),
    ])

    assert returncode == 0
    assert json.loads(output.read_text(encoding="utf-8")) == result
    assert json.loads(capsys.readouterr().out) == {
        "case_id": "Q0__warmup__r0",
        "classification": "PASS",
        "output": str(output),
    }


def test_main_structured_campaign_streams_35_case_artifacts(
    tmp_path,
    capsys,
    monkeypatch,
):
    worker = _load()
    output = tmp_path / "campaign.json"
    case_dir = tmp_path / "cases"
    observed = {}

    def fake_campaign(**kwargs):
        observed.update(kwargs)
        full = {
            "classification": "PASS",
            "case_id": "P0__warmup__r0",
            "decode_time_ns": 123,
            "profile": {"large": True},
        }
        kwargs["case_sink"](full)
        return {
            "schema_version": worker.WORKER_SCHEMA,
            "classification": "PASS",
            "attempt": kwargs["attempt"],
            "cases": [{
                "classification": "PASS",
                "case_id": full["case_id"],
                "decode_time_ns": 123,
            }],
            "cleanup": {
                "process_group_destroyed": True,
                "owned_children_remaining": [],
            },
        }

    monkeypatch.setattr(worker, "run_profile_campaign", fake_campaign)

    returncode = worker.main([
        "--attempt",
        "attempt-r1",
        "--structured-campaign",
        "--model-root",
        "/model",
        "--output-dir",
        str(case_dir),
        "--output",
        str(output),
    ])

    assert returncode == 0
    assert len(observed["cases"]) == 35
    assert observed["retain_case_profiles"] is False
    assert json.loads(
        (case_dir / "P0__warmup__r0.json").read_text(encoding="utf-8")
    )["profile"] == {"large": True}
    assert json.loads(output.read_text(encoding="utf-8"))["classification"] == (
        "PASS"
    )
    assert json.loads(capsys.readouterr().out) == {
        "case_count": 1,
        "classification": "PASS",
        "output": str(output),
        "output_dir": str(case_dir),
    }


def test_compact_profile_preserves_operation_and_per_collective_bytes():
    worker = _load()
    ranks = []
    for rank in range(4):
        common = {
            "rank": rank,
            "step_index": 1,
            "decode_ordinal": 0,
            "request_set_sha256": "a" * 64,
        }
        ranks.append({
            "rank": rank,
            "enabled": True,
            "finalization_status": "complete",
            "steps": [
                common | {
                    "is_decode": True,
                    "cuda_ns": 10_000,
                },
            ],
            "layers": [
                common | {
                    "layer_index": 0,
                    "layer_role": "full_attention",
                    "cuda_ns": 9_000,
                },
            ],
            "operations": [
                common | {
                    "layer_index": 0,
                    "layer_role": "full_attention",
                    "operation_ordinal": 0,
                    "operation_class": "gemm",
                    "operation_name": "qkv_projection",
                    "cuda_ns": 4_000,
                },
                common | {
                    "layer_index": 0,
                    "layer_role": "full_attention",
                    "operation_ordinal": 1,
                    "operation_class": "collective",
                    "operation_name": "row_parallel_all_reduce",
                    "cuda_ns": 2_000,
                },
                common | {
                    "layer_index": 0,
                    "layer_role": "full_attention",
                    "operation_ordinal": 2,
                    "operation_class": "collective",
                    "operation_name": "row_parallel_all_reduce",
                    "cuda_ns": 1_000,
                },
            ],
            "collectives": [
                common | {
                    "layer_index": 0,
                    "layer_role": "full_attention",
                    "operation_ordinal": 1,
                    "operation": "row_parallel_all_reduce",
                    "tensor_shape": [2, 8],
                    "tensor_dtype": "torch.bfloat16",
                    "cuda_ns": 2_000,
                },
                common | {
                    "layer_index": 0,
                    "layer_role": "full_attention",
                    "operation_ordinal": 2,
                    "operation": "row_parallel_all_reduce",
                    "tensor_shape": [2, 4],
                    "tensor_dtype": "torch.float32",
                    "cuda_ns": 1_000,
                },
            ],
        })
    profile = {
        "enabled": True,
        "rank_inventory": [0, 1, 2, 3],
        "ranks": ranks,
    }

    compact = worker.compact_internal_profile(profile)

    layer = compact["ranks"][0]["steps"][0]["layers"][0]
    assert layer["operation_inventory"] == [
        [0, "gemm", "qkv_projection"],
        [1, "collective", "row_parallel_all_reduce"],
        [2, "collective", "row_parallel_all_reduce"],
    ]
    assert layer["collective_byte_inventory"] == [
        [1, 32],
        [2, 32],
    ]
    assert layer["collective_count"] == 2
    assert layer["collective_bytes"] == 64
    assert "operations" not in compact["ranks"][0]
    assert "collectives" not in compact["ranks"][0]
