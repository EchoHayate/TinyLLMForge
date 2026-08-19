from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
GATE_PATH = (
    ROOT / "tools" / "autoregressive_draft_performance_gate.py"
)
WORKER_PATH = (
    ROOT / "tools" / "autoregressive_draft_performance_worker.py"
)
VERIFY_PATH = (
    ROOT
    / "tools"
    / "verify_autoregressive_draft_performance_gate.py"
)
DIAGNOSTIC_PATH = (
    ROOT
    / "tools"
    / "autoregressive_draft_b4_timing_diagnostic.py"
)
DIAGNOSTIC_VERIFY_PATH = (
    ROOT
    / "tools"
    / "verify_autoregressive_draft_b4_timing_diagnostic.py"
)
DIAGNOSTIC_REMOTE_SCRIPT_PATH = (
    ROOT
    / "tools"
    / "run_autoregressive_draft_b4_timing_diagnostic_remote.sh"
)
REMOTE_SCRIPT_PATH = (
    ROOT
    / "tools"
    / "run_autoregressive_draft_performance_gate_remote.sh"
)


def _load_module(name: str, path: Path):
    assert path.exists(), f"missing module: {path}"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _gate():
    return _load_module(
        "autoregressive_draft_performance_gate_test_module",
        GATE_PATH,
    )


def _worker_module():
    _load_module(
        "autoregressive_draft_performance_gate",
        GATE_PATH,
    )
    return _load_module(
        "autoregressive_draft_performance_worker_test_module",
        WORKER_PATH,
    )


def _verifier():
    _load_module(
        "autoregressive_draft_performance_gate",
        GATE_PATH,
    )
    return _load_module(
        "verify_autoregressive_draft_performance_gate_test_module",
        VERIFY_PATH,
    )


def _diagnostic():
    _load_module(
        "autoregressive_draft_performance_gate",
        GATE_PATH,
    )
    return _load_module(
        "autoregressive_draft_b4_timing_diagnostic_test_module",
        DIAGNOSTIC_PATH,
    )


def _diagnostic_verifier():
    _load_module(
        "autoregressive_draft_performance_gate",
        GATE_PATH,
    )
    _load_module(
        "autoregressive_draft_b4_timing_diagnostic",
        DIAGNOSTIC_PATH,
    )
    return _load_module(
        "verify_autoregressive_draft_b4_timing_diagnostic_test_module",
        DIAGNOSTIC_VERIFY_PATH,
    )


def _run(*, policy: str, batch_size: int, repeat: int) -> dict:
    learned = policy == "learned"
    ttft_s = 0.040 + repeat * 0.001
    tpot_s = (0.010 if learned else 0.012) + repeat * 0.0001
    batch_elapsed_s = (
        0.22 if learned else 0.26
    ) + repeat * 0.002
    proposed_tokens = batch_size * 24 if learned else 0
    accepted_draft_tokens = batch_size * 12 if learned else 0
    stage_keys = (
        "first_target_batch_ms",
        "draft_proposal_ms",
        "reserve_blocks_ms",
        "tail_batch_ms",
        "kv_materialize_ms",
        "accept_sample_ms",
        "commit_metadata_ms",
    )
    stage_steps = (
        [
            {
                "step_index": step_index,
                "timing_ms": {
                    key: (
                        (stage_index + 1) * 0.1
                        + step_index * 0.01
                        + repeat * 0.001
                    )
                    for stage_index, key in enumerate(stage_keys)
                },
            }
            for step_index in range(2)
        ]
        if learned
        else []
    )
    stage_totals = {
        key: sum(
            row["timing_ms"][key] for row in stage_steps
        )
        for key in stage_keys
    }
    executor_keys = (
        "prompt_bootstrap",
        "proposal_forward",
        "proposal_finalize",
    )
    proposal_detail_keys = (
        "setup",
        "backend_submit",
        "selection_collective",
        "decode_authority",
        "token_readback",
        "materialize_register",
    )
    proposal_detail_bases = {
        "setup": 0.2,
        "backend_submit": 0.3,
        "selection_collective": 0.4,
        "decode_authority": 0.1,
        "token_readback": 0.3,
        "materialize_register": 0.2,
    }
    executor_ranks = [
        {
            "rank": rank,
            **{
                key: (
                    (key_index + 1) * 1.0
                    + rank * 0.1
                    + repeat * 0.01
                    if learned
                    else 0.0
                )
                for key_index, key in enumerate(executor_keys)
            },
        }
        for rank in range(4)
    ]
    proposal_detail_ranks = [
        {
            "rank": rank,
            **{
                key: (
                    proposal_detail_bases[key]
                    + rank * 0.01
                    + repeat * 0.001
                    if learned
                    else 0.0
                )
                for key in proposal_detail_keys
            },
        }
        for rank in range(4)
    ]
    proposal_detail_max = {
        key: max(row[key] for row in proposal_detail_ranks)
        for key in proposal_detail_keys
    }
    critical_rank = max(
        range(4),
        key=lambda rank: executor_ranks[rank]["proposal_forward"],
    )
    critical_rank_detail = {
        key: proposal_detail_ranks[critical_rank][key]
        for key in proposal_detail_keys
    }
    proposal_detail_sum = sum(critical_rank_detail.values())
    proposal_forward_residual = (
        executor_ranks[critical_rank]["proposal_forward"]
        - proposal_detail_sum
    )
    per_request = [
        {
            "sequence_id": sequence_id,
            "output_tokens": 16,
            "ttft_s": ttft_s + sequence_id * 0.0001,
            "tpot_s": tpot_s + sequence_id * 0.00001,
            "completion_latency_s": (
                ttft_s
                + sequence_id * 0.0001
                + (tpot_s + sequence_id * 0.00001) * 15
            ),
        }
        for sequence_id in range(batch_size)
    ]
    proposal_kv_ranks = [
        {
            "rank": rank,
            "h2d_entries": 0,
            "h2d_bytes": 0,
            "d2h_entries": 0,
            "d2h_bytes": 0,
        }
        for rank in range(4)
    ]
    memory_ranks = [
        {
            "rank": rank,
            "peak_allocated_bytes": (
                1_000_000 + rank * 1_000 + repeat * 100
            ),
            "peak_reserved_bytes": (
                1_200_000 + rank * 1_000 + repeat * 100
            ),
        }
        for rank in range(4)
    ]
    return {
        "repeat": repeat,
        "outputs": [
            list(range(sequence_id, sequence_id + 16))
            for sequence_id in range(batch_size)
        ],
        "timing": {
            "request_count": batch_size,
            "total_output_tokens": batch_size * 16,
            "batch_elapsed_s": batch_elapsed_s,
            "batch_token_throughput_tps": (
                batch_size * 16 / batch_elapsed_s
            ),
            "request_throughput_rps": (
                batch_size / batch_elapsed_s
            ),
            "per_request": per_request,
        },
        "runtime": {
            "proposed_tokens": proposed_tokens,
            "accepted_draft_tokens": accepted_draft_tokens,
            "acceptance_rate": (
                accepted_draft_tokens / proposed_tokens
                if proposed_tokens
                else 0.0
            ),
            "stage_timing": {
                "step_count": len(stage_steps),
                "steps": stage_steps,
                "totals_ms": stage_totals,
            },
            "draft_executor_timing": {
                "ranks": executor_ranks,
                "max_rank_ms": {
                    key: max(row[key] for row in executor_ranks)
                    for key in executor_keys
                },
            },
            "draft_executor_proposal_detail": {
                "ranks": proposal_detail_ranks,
                "max_rank_ms": proposal_detail_max,
                "critical_rank": critical_rank,
                "critical_rank_ms": critical_rank_detail,
                "detail_sum_ms": proposal_detail_sum,
                "residual_ms": proposal_forward_residual,
            },
        },
        "memory": {
            "ranks": memory_ranks,
            "peak_allocated_bytes": max(
                row["peak_allocated_bytes"]
                for row in memory_ranks
            ),
            "peak_reserved_bytes": max(
                row["peak_reserved_bytes"]
                for row in memory_ranks
            ),
        },
        "proposal_kv": {
            "ranks": proposal_kv_ranks,
            "totals": {
                "h2d_entries": 0,
                "h2d_bytes": 0,
                "d2h_entries": 0,
                "d2h_bytes": 0,
            },
        },
    }


def _worker(policy: str, batch_size: int) -> dict:
    return {
        "policy": policy,
        "batch_size": batch_size,
        "prompt_rows": [
            {
                "prompt_index": prompt_index,
                "token_ids": [prompt_index + 1] * 256,
                "token_count": 256,
                "sha256": hashlib.sha256(
                    json.dumps(
                        [prompt_index + 1] * 256,
                        separators=(",", ":"),
                    ).encode("utf-8")
                ).hexdigest(),
            }
            for prompt_index in range(batch_size)
        ],
        "warmup_runs": [
            _run(
                policy=policy,
                batch_size=batch_size,
                repeat=-1,
            )
        ],
        "measured_runs": [
            _run(
                policy=policy,
                batch_size=batch_size,
                repeat=repeat,
            )
            for repeat in range(3)
        ],
        "target_checkpoint_identifier": "Qwen3-1.7B",
        "draft_checkpoint_identifier": (
            "Qwen3-0.6B" if policy == "learned" else None
        ),
        "tokenizer_identifier": "Qwen3-1.7B",
        "dtype": "bfloat16",
        "tensor_parallel_size": 4,
        "proposal_kv_allocator": "direct",
        "proposal_slot_capacity": batch_size * (256 + 16 + 4),
    }


def _workers() -> list[dict]:
    return [
        _worker(policy, batch_size)
        for policy in ("target", "learned")
        for batch_size in (1, 4)
    ]


def _diagnostic_worker(policy: str) -> dict:
    worker = _worker(policy, 4)
    worker["warmup_runs"] = [
        _run(policy=policy, batch_size=4, repeat=-2),
        _run(policy=policy, batch_size=4, repeat=-1),
    ]
    worker["measured_runs"] = [
        _run(policy=policy, batch_size=4, repeat=repeat)
        for repeat in range(8)
    ]
    return worker


def _tamper_target_proposal_detail(rows: list[dict]) -> None:
    detail = rows[0]["measured_runs"][0]["runtime"][
        "draft_executor_proposal_detail"
    ]
    detail["ranks"][0]["setup"] = 0.1
    detail["max_rank_ms"]["setup"] = 0.1
    detail["critical_rank_ms"]["setup"] = 0.1
    detail["detail_sum_ms"] = 0.1
    detail["residual_ms"] = -0.1


def _environment() -> dict:
    return {
        "target_model_path": "/models/Qwen3-1.7B",
        "draft_model_path": "/models/Qwen3-0.6B",
        "device_names": ["A100", "A100", "A100", "A100"],
        "python_version": "3.11.9",
        "torch_version": "2.7.1",
        "command": ["python", "gate.py"],
    }


def _artifact() -> dict:
    return _gate().build_performance_artifact(
        worker_results=_workers(),
        environment=_environment(),
        source_files={"tools/source.py": "a" * 64},
    )


def test_artifact_freezes_four_cells_raw_distributions_and_pilot_scope():
    artifact = _artifact()

    receipt = _gate().validate_performance_artifact(artifact)

    assert artifact["schema_version"] == 3
    assert artifact["classification"] == "PILOT_ONLY"
    assert artifact["direction"] == "POSITIVE"
    assert artifact["batch_directions"] == {
        "1": "IMPROVED",
        "4": "IMPROVED",
    }
    assert set(artifact["cells"]) == {
        "target:b1",
        "learned:b1",
        "target:b4",
        "learned:b4",
    }
    assert all(
        len(cell["measured_runs"]) == 3
        for cell in artifact["cells"].values()
    )
    aggregate = artifact["cells"]["learned:b4"]["aggregate"]
    assert set(aggregate) == {
        "ttft_s",
        "tpot_s",
        "e2e_s",
        "output_throughput_tps",
        "peak_allocated_bytes",
        "peak_reserved_bytes",
        "proposal_kv_h2d_bytes",
        "proposal_kv_d2h_bytes",
        "proposed_tokens",
        "accepted_draft_tokens",
        "acceptance_rate",
        "stage_first_target_batch_ms",
        "stage_draft_proposal_ms",
        "stage_reserve_blocks_ms",
        "stage_tail_batch_ms",
        "stage_kv_materialize_ms",
        "stage_accept_sample_ms",
        "stage_commit_metadata_ms",
        "executor_prompt_bootstrap_ms",
        "executor_proposal_forward_ms",
        "executor_proposal_finalize_ms",
        "executor_detail_setup_ms",
        "executor_detail_backend_submit_ms",
        "executor_detail_selection_collective_ms",
        "executor_detail_decode_authority_ms",
        "executor_detail_token_readback_ms",
        "executor_detail_materialize_register_ms",
        "executor_detail_sum_ms",
        "executor_detail_residual_ms",
    }
    assert all(row["count"] == 3 for row in aggregate.values())
    assert aggregate["accepted_draft_tokens"]["median"] > 0
    assert receipt["status"] == "PASS"
    assert receipt["classification"] == "PILOT_ONLY"


@pytest.mark.parametrize(
    "mutate,match",
    (
        (lambda rows: rows.pop(), "four"),
        (
            lambda rows: rows[0]["measured_runs"].pop(),
            "three measured",
        ),
        (
            lambda rows: rows[2]["measured_runs"][1][
                "outputs"
            ][0].__setitem__(0, 999),
            "parity",
        ),
        (
            lambda rows: rows[2]["measured_runs"][0][
                "runtime"
            ].__setitem__("accepted_draft_tokens", 0),
            "accepted",
        ),
        (
            lambda rows: rows[0]["measured_runs"][0][
                "memory"
            ]["ranks"].pop(),
            "memory",
        ),
        (
            lambda rows: rows[2]["measured_runs"][0][
                "proposal_kv"
            ]["ranks"].pop(),
            "Proposal-KV",
        ),
        (
            lambda rows: rows[2]["measured_runs"][0][
                "runtime"
            ]["stage_timing"]["totals_ms"].__setitem__(
                "tail_batch_ms",
                999.0,
            ),
            "stage timing",
        ),
        (
            lambda rows: rows[2]["measured_runs"][0][
                "runtime"
            ]["draft_executor_timing"]["max_rank_ms"].__setitem__(
                "proposal_forward",
                999.0,
            ),
            "executor timing",
        ),
        (
            lambda rows: rows[2]["measured_runs"][0][
                "runtime"
            ]["draft_executor_proposal_detail"][
                "max_rank_ms"
            ].__setitem__(
                "backend_submit",
                999.0,
            ),
            "proposal detail",
        ),
        (
            _tamper_target_proposal_detail,
            "residual",
        ),
    ),
)
def test_artifact_rejects_missing_raw_or_correctness_evidence(
    mutate,
    match,
):
    rows = _workers()
    mutate(rows)

    with pytest.raises(ValueError, match=match):
        _gate().build_performance_artifact(
            worker_results=rows,
            environment=_environment(),
            source_files={"tools/source.py": "a" * 64},
        )


def test_validation_recomputes_aggregate_and_rejects_tampering():
    artifact = _artifact()
    artifact["cells"]["learned:b1"]["aggregate"]["tpot_s"][
        "median"
    ] += 0.1

    with pytest.raises(ValueError, match="aggregate"):
        _gate().validate_performance_artifact(artifact)


def test_artifact_rejects_negative_proposal_forward_residual():
    rows = _workers()
    detail = rows[2]["measured_runs"][0]["runtime"][
        "draft_executor_proposal_detail"
    ]
    detail["ranks"][3]["backend_submit"] += 10.0
    detail["max_rank_ms"]["backend_submit"] += 10.0
    detail["critical_rank_ms"]["backend_submit"] += 10.0
    detail["detail_sum_ms"] += 10.0
    detail["residual_ms"] -= 10.0

    with pytest.raises(ValueError, match="residual"):
        _gate().build_performance_artifact(
            worker_results=rows,
            environment=_environment(),
            source_files={"tools/source.py": "a" * 64},
        )


def test_artifact_requires_positive_learned_core_proposal_detail():
    rows = _workers()
    detail = rows[2]["measured_runs"][0]["runtime"][
        "draft_executor_proposal_detail"
    ]
    for key in (
        "backend_submit",
        "selection_collective",
        "decode_authority",
        "token_readback",
    ):
        for row in detail["ranks"]:
            row[key] = 0.0
        detail["max_rank_ms"][key] = 0.0
        detail["critical_rank_ms"][key] = 0.0
    detail["detail_sum_ms"] = sum(
        detail["critical_rank_ms"].values()
    )
    detail["residual_ms"] = (
        rows[2]["measured_runs"][0]["runtime"][
            "draft_executor_timing"
        ]["max_rank_ms"]["proposal_forward"]
        - detail["detail_sum_ms"]
    )

    with pytest.raises(ValueError, match="lacks executor proposal detail"):
        _gate().build_performance_artifact(
            worker_results=rows,
            environment=_environment(),
            source_files={"tools/source.py": "a" * 64},
        )


def test_validation_rejects_proposal_detail_aggregate_tampering():
    artifact = _artifact()
    artifact["cells"]["learned:b1"]["aggregate"][
        "executor_detail_sum_ms"
    ]["median"] += 0.1

    with pytest.raises(ValueError, match="aggregate"):
        _gate().validate_performance_artifact(artifact)


def test_validation_rejects_non_pilot_classification():
    artifact = _artifact()
    artifact["classification"] = "PASS"

    with pytest.raises(ValueError, match="PILOT_ONLY"):
        _gate().validate_performance_artifact(artifact)


def test_source_hashing_and_atomic_json_writer(tmp_path):
    source = tmp_path / "source.py"
    source.write_text("source\n", encoding="utf-8")

    hashes = _gate().hash_source_files(
        repo_root=tmp_path,
        source_files=("source.py",),
    )
    output = tmp_path / "nested" / "result.json"
    _gate().write_json_atomic(output, {"hashes": hashes})

    assert hashes == {
        "source.py": hashlib.sha256(source.read_bytes()).hexdigest()
    }
    assert json.loads(output.read_text(encoding="utf-8")) == {
        "hashes": hashes
    }
    assert not output.with_suffix(".json.tmp").exists()


class _FakeTokenizer:
    name_or_path = "Qwen3-1.7B"

    def encode(self, text, add_special_tokens=False):
        assert add_special_tokens is False
        return [ord(character) % 97 + 1 for character in text]


def _authority_rank(rank: int, *, offset: int) -> dict:
    return {
        "rank": rank,
        "world_size": 4,
        "checkpoint_identity": {
            "target": {"composite_sha256": "1" * 64},
            "draft": {"composite_sha256": "2" * 64},
        },
        "executor": {
            "last_logical_authority_sha256": "a" * 64,
            "proposal_kv_lifecycle": {
                "active_transaction_count": 0,
            },
            "cuda_graph": {
                "capture_attempts": 1,
                "captures": 1,
                "replays": 4,
                "quarantines": 0,
                "fallback_pre_replay": 0,
                "quarantined": {},
                "quarantine_details": {},
                "ready_entries": ("b" * 64,),
                "static_bytes": 55_000,
                "reserved_bytes": 700_000_000,
                "total_capture_ns": 1_500_000_000,
            },
            "timing_ms": {
                "prompt_bootstrap": offset + rank * 0.1,
                "proposal_forward": offset * 2 + rank * 0.1,
                "proposal_finalize": offset * 3 + rank * 0.1,
            },
            "proposal_forward_detail_ms": {
                "setup": offset * 0.2 + rank * 0.01,
                "backend_submit": offset * 0.3 + rank * 0.01,
                "selection_collective": offset * 0.4 + rank * 0.01,
                "decode_authority": offset * 0.1 + rank * 0.01,
                "token_readback": offset * 0.3 + rank * 0.01,
                "materialize_register": offset * 0.2 + rank * 0.01,
            },
            "backend": {
                "proposal_kv_cache": {
                    "entry_allocator": {
                        "allocator_mode": "direct",
                        "h2d_entry_count": offset + rank,
                        "h2d_bytes": (offset + rank) * 10,
                        "d2h_entry_count": offset + rank,
                        "d2h_bytes": (offset + rank) * 5,
                    },
                },
            },
        },
    }


class _FakeWorkerEngine:
    def __init__(self):
        self.tokenizer = _FakeTokenizer()
        self.config = type(
            "Config",
            (),
            {"dtype": "bfloat16"},
        )()
        self.events = []
        self.pending = {}
        self.step_index = 0
        self.next_sequence_id = 0
        self.last_step_observation = None
        self.authority_offset = 0
        self.command_timeline_configurations = 0
        self.decode_profile_configurations = 0
        self.command_timeline_repeat = None
        self.command_timeline_request_set_sha256 = None
        self.last_command_timeline_repeat = None
        self.last_command_timeline_request_set_sha256 = None

    def is_finished(self):
        return not self.pending

    def clear_reusable_prefix_cache(self):
        self.events.append("clear")

    def reset_peak_memory_stats(self, *, timeout_s):
        assert timeout_s == 60.0
        self.events.append("reset")
        return tuple(
            {
                "rank": rank,
                "cuda_allocated_bytes": 100 + rank,
                "cuda_reserved_bytes": 120 + rank,
                "cuda_peak_allocated_bytes": 100 + rank,
                "cuda_peak_reserved_bytes": 120 + rank,
                "kv_capacity_bytes": 64,
            }
            for rank in range(4)
        )

    def configure_command_timeline(
        self,
        enabled,
        max_rows,
        timeout_s,
    ):
        assert enabled is True
        assert max_rows > 0
        assert timeout_s == 60.0
        self.command_timeline_configurations += 1
        self.events.append("configure-command-timeline")
        return {
            "enabled": True,
            "max_rows": max_rows,
            "rank_inventory": [0, 1, 2, 3],
        }

    def configure_decode_internal_profile(
        self,
        enabled,
        profile_label,
        *,
        timeout_s,
    ):
        assert enabled is True
        assert profile_label
        assert timeout_s == 60.0
        self.decode_profile_configurations += 1
        self.events.append("configure-decode-profile")
        return {
            "enabled": True,
            "rank_inventory": [0, 1, 2, 3],
        }

    def reset_command_timeline(self, timeout_s):
        assert timeout_s == 60.0
        self.events.append("reset-command-timeline")
        return tuple(
            {"rank": rank, "enabled": True, "max_rows": 8192}
            for rank in range(4)
        )

    def reset_decode_internal_profile(self, *, timeout_s):
        assert timeout_s == 60.0
        self.events.append("reset-decode-profile")
        return tuple(
            {
                "rank": rank,
                "enabled": True,
                "profile_label": "command-timeline",
            }
            for rank in range(4)
        )

    def begin_command_timeline_repeat(
        self,
        repeat_index,
        *,
        request_set_sha256,
    ):
        assert repeat_index >= 0
        assert self.command_timeline_repeat is None
        self.command_timeline_repeat = repeat_index
        self.command_timeline_request_set_sha256 = request_set_sha256
        self.events.append(
            ("begin-repeat", repeat_index, request_set_sha256)
        )
        return {
            "repeat_index": repeat_index,
            "request_set_sha256": request_set_sha256,
        }

    def end_command_timeline_repeat(self):
        assert self.command_timeline_repeat is not None
        repeat_index = self.command_timeline_repeat
        request_set_sha256 = self.command_timeline_request_set_sha256
        self.last_command_timeline_repeat = repeat_index
        self.last_command_timeline_request_set_sha256 = request_set_sha256
        self.command_timeline_repeat = None
        self.command_timeline_request_set_sha256 = None
        self.events.append(("end-repeat", repeat_index))
        return {
            "repeat_index": repeat_index,
            "request_set_sha256": request_set_sha256,
        }

    def command_timeline_snapshots(self, timeout_s):
        assert timeout_s == 60.0
        self.events.append("command-snapshot")
        selected_sha256 = "b" * 64
        return tuple(
            {
                "schema_version": 1,
                "rank": rank,
                "enabled": True,
                "clock": {"boot_id": "boot"},
                "rows": [
                    {
                        "rank": rank,
                        "command_id": 10 + index,
                        "method_name": "step",
                        "requires_ack": True,
                        "engine_step_id": index,
                        "repeat_index": (
                            self.last_command_timeline_repeat
                        ),
                        "request_set_sha256": (
                            self.last_command_timeline_request_set_sha256
                        ),
                        "batch_kind": "speculative",
                        "speculative_selected_sequence_ids_sha256": (
                            selected_sha256
                        ),
                        "dispatch_started_monotonic_ns": 100 + index,
                        "dispatch_published_monotonic_ns": 110 + index,
                    }
                    for index in range(self.step_index)
                ],
                "dropped_rows": 0,
            }
            for rank in range(4)
        )

    def finalize_decode_internal_profile(
        self,
        *,
        already_synchronized_rank,
        timeout_s,
    ):
        assert already_synchronized_rank == 0
        assert timeout_s == 60.0
        for rank in range(1, 4):
            self.events.append(("profiler-synchronize", rank))
        self.events.append("cuda-snapshot")
        repeat_index = self.last_command_timeline_repeat
        request_set_sha256 = (
            self.last_command_timeline_request_set_sha256
        )
        selected_sha256 = "b" * 64
        return {
            "enabled": True,
            "rank_inventory": [0, 1, 2, 3],
            "ranks": [
                {
                    "rank": rank,
                    "enabled": True,
                    "finalization_status": "complete",
                    "steps": [
                        {
                            "rank": rank,
                            "command_id": 10 + index,
                            "engine_step_id": index,
                            "repeat_index": repeat_index,
                            "request_set_sha256": request_set_sha256,
                            "speculative_selected_sequence_ids_sha256": (
                                selected_sha256
                            ),
                        }
                        for index in range(self.step_index)
                    ],
                    "collectives": [
                        {
                            "rank": rank,
                            "command_id": 10 + index,
                            "engine_step_id": index,
                            "repeat_index": repeat_index,
                            "request_set_sha256": request_set_sha256,
                            "speculative_selected_sequence_ids_sha256": (
                                selected_sha256
                            ),
                        }
                        for index in range(self.step_index)
                    ],
                    "dropped_steps": 0,
                    "dropped_collectives": 0,
                }
                for rank in range(4)
            ],
        }

    def engine_step_timeline_snapshot(self):
        self.events.append("step-snapshot")
        return {
            "schema_version": 1,
            "enabled": True,
            "steps": [
                {
                    "engine_step_id": index,
                    "repeat_index": self.last_command_timeline_repeat,
                    "request_set_sha256": (
                        self.last_command_timeline_request_set_sha256
                    ),
                    "batch_kind": "speculative",
                    "speculative_selected_sequence_ids_sha256": (
                        "b" * 64
                    ),
                }
                for index in range(self.step_index)
            ],
            "dropped_steps": 0,
        }

    def memory_snapshots(self, *, timeout_s):
        assert timeout_s == 60.0
        self.events.append("memory")
        return tuple(
            {
                "rank": rank,
                "cuda_allocated_bytes": 110 + rank,
                "cuda_reserved_bytes": 130 + rank,
                "cuda_peak_allocated_bytes": 180 + rank,
                "cuda_peak_reserved_bytes": 220 + rank,
                "kv_capacity_bytes": 64,
            }
            for rank in range(4)
        )

    def autoregressive_draft_authority_snapshots(
        self,
        *,
        timeout_s,
    ):
        assert timeout_s == 60.0
        self.events.append("authority")
        rows = tuple(
            _authority_rank(
                rank,
                offset=self.authority_offset,
            )
            for rank in range(4)
        )
        self.authority_offset += 3
        return rows

    def flush_pending_hybrid_state_releases(self, *, timeout_s):
        assert timeout_s == 60.0
        self.events.append("flush")

    def add_request(self, token_ids, sampling_params):
        assert len(token_ids) == 256
        sequence_id = self.next_sequence_id
        self.next_sequence_id += 1
        self.pending[sequence_id] = 0
        self.events.append("add")

    def step(self):
        self.events.append(f"step-{self.step_index}")
        output_rows = []
        deltas = {}
        finished = []
        proposal_ids = {}
        accepted_counts = {}
        for sequence_id in list(self.pending):
            emitted = self.pending[sequence_id]
            token_count = 1 if emitted == 0 else 15
            token_ids = list(range(emitted, emitted + token_count))
            deltas[sequence_id] = token_ids
            proposal_ids[sequence_id] = [31, 32, 33, 34]
            accepted_counts[sequence_id] = 2
            self.pending[sequence_id] += token_count
            if self.pending[sequence_id] == 16:
                output_rows.append((sequence_id, list(range(16))))
                finished.append(sequence_id)
                del self.pending[sequence_id]
        self.last_step_observation = {
            "new_completion_tokens_by_seq": deltas,
            "finished_seq_ids": finished,
            "speculative_proposal_token_ids_by_seq": proposal_ids,
            "speculative_accepted_draft_token_counts": accepted_counts,
            "speculative_runtime_timing_ms": {
                "first_target_batch_ms": 1.0 + self.step_index,
                "draft_proposal_ms": 0.0,
                "reserve_blocks_ms": 0.1,
                "tail_batch_ms": 2.0 + self.step_index,
                "kv_materialize_ms": 0.2,
                "accept_sample_ms": 0.3,
                "commit_metadata_ms": 0.0,
            },
        }
        self.step_index += 1
        return output_rows, None


class _FakeWorkerAdapter:
    def __init__(self):
        self.engine = _FakeWorkerEngine()
        self.close_calls = 0

    def close(self):
        self.close_calls += 1


class _FakeSamplingParams:
    def __init__(self, **kwargs):
        self.kwargs = dict(kwargs)


def _worker_prompt_rows(batch_size=1):
    return [
        {
            "prompt_index": prompt_index,
            "token_ids": [prompt_index + 1] * 256,
            "token_count": 256,
            "sha256": "a" * 64,
        }
        for prompt_index in range(batch_size)
    ]


def test_worker_records_step_end_timing_memory_and_proposal_kv_delta():
    engine = _FakeWorkerEngine()
    clock = iter(
        (1_000_000_000, 1_200_000_000, 1_500_000_000)
    )
    synchronize_calls = []

    run = _worker_module().run_request_batch(
        engine=engine,
        policy="learned",
        prompt_rows=_worker_prompt_rows(),
        sampling_params=_FakeSamplingParams(),
        expected_output_tokens=16,
        synchronize=lambda: synchronize_calls.append(True),
        clock_ns=lambda: next(clock),
        repeat=0,
    )

    assert run["repeat"] == 0
    assert run["outputs"] == [list(range(16))]
    assert run["timing"]["per_request"][0]["ttft_s"] == pytest.approx(
        0.2
    )
    assert run["timing"]["per_request"][0]["tpot_s"] == pytest.approx(
        0.02
    )
    assert run["runtime"]["proposed_tokens"] == 8
    assert run["runtime"]["accepted_draft_tokens"] == 4
    assert run["runtime"]["acceptance_rate"] == 0.5
    assert run["runtime"]["stage_timing"] == {
        "step_count": 2,
        "steps": [
            {
                "step_index": 0,
                "timing_ms": {
                    "first_target_batch_ms": 1.0,
                    "draft_proposal_ms": 0.0,
                    "reserve_blocks_ms": 0.1,
                    "tail_batch_ms": 2.0,
                    "kv_materialize_ms": 0.2,
                    "accept_sample_ms": 0.3,
                    "commit_metadata_ms": 0.0,
                },
            },
            {
                "step_index": 1,
                "timing_ms": {
                    "first_target_batch_ms": 2.0,
                    "draft_proposal_ms": 0.0,
                    "reserve_blocks_ms": 0.1,
                    "tail_batch_ms": 3.0,
                    "kv_materialize_ms": 0.2,
                    "accept_sample_ms": 0.3,
                    "commit_metadata_ms": 0.0,
                },
            },
        ],
        "totals_ms": {
            "first_target_batch_ms": 3.0,
            "draft_proposal_ms": 0.0,
            "reserve_blocks_ms": 0.2,
            "tail_batch_ms": 5.0,
            "kv_materialize_ms": 0.4,
            "accept_sample_ms": 0.6,
            "commit_metadata_ms": 0.0,
        },
    }
    assert run["runtime"]["draft_executor_timing"] == {
        "ranks": [
            {
                "rank": rank,
                "prompt_bootstrap": 3.0,
                "proposal_forward": 6.0,
                "proposal_finalize": 9.0,
            }
            for rank in range(4)
        ],
        "max_rank_ms": {
            "prompt_bootstrap": 3.0,
            "proposal_forward": 6.0,
            "proposal_finalize": 9.0,
        },
    }
    proposal_detail = run["runtime"][
        "draft_executor_proposal_detail"
    ]
    expected_detail = {
        "setup": 0.6,
        "backend_submit": 0.9,
        "selection_collective": 1.2,
        "decode_authority": 0.3,
        "token_readback": 0.9,
        "materialize_register": 0.6,
    }
    assert [row["rank"] for row in proposal_detail["ranks"]] == list(
        range(4)
    )
    for row in proposal_detail["ranks"]:
        for key, expected in expected_detail.items():
            assert row[key] == pytest.approx(expected)
    for key, expected in expected_detail.items():
        assert proposal_detail["max_rank_ms"][key] == pytest.approx(
            expected
        )
    assert proposal_detail["critical_rank"] == 0
    for key, expected in expected_detail.items():
        assert proposal_detail["critical_rank_ms"][key] == pytest.approx(
            expected
        )
    assert proposal_detail["detail_sum_ms"] == pytest.approx(4.5)
    assert proposal_detail["residual_ms"] == pytest.approx(1.5)
    assert len(run["memory"]["ranks"]) == 4
    assert run["memory"]["peak_allocated_bytes"] == 183
    assert run["proposal_kv"]["totals"] == {
        "h2d_entries": 12,
        "h2d_bytes": 120,
        "d2h_entries": 12,
        "d2h_bytes": 60,
    }
    assert run["correctness"] == {
        "proposal_token_rows": [
            [[31, 32, 33, 34]],
            [[31, 32, 33, 34]],
        ],
        "accepted_prefix_counts": [2, 2],
        "transaction_digest": "a" * 64,
        "active_transaction_count": 0,
        "rank_graph_counters": [
            {
                "rank": rank,
                "capture_attempts": 1,
                "captures": 1,
                "replays": 4,
                "quarantines": 0,
                "fallback_pre_replay": 0,
            }
            for rank in range(4)
        ],
        "rank_graph_quarantine_details": [
            {
                "rank": rank,
                "quarantine_details": {},
            }
            for rank in range(4)
        ],
        "rank_graph_quarantined": [
            {
                "rank": rank,
                "quarantined": {},
            }
            for rank in range(4)
        ],
        "rank_graph_resources": [
            {
                "rank": rank,
                "ready_entry_count": 1,
                "static_bytes": 55_000,
                "reserved_bytes": 700_000_000,
                "total_capture_ns": 1_500_000_000,
            }
            for rank in range(4)
        ],
    }
    assert engine.events.index("reset") < engine.events.index("step-0")
    assert engine.events.index("memory") > engine.events.index("step-1")
    assert len(synchronize_calls) == 3
    assert "command_timeline" not in run["runtime"]
    assert not any(
        (
            isinstance(event, str)
            and (
                "timeline" in event
                or "decode-profile" in event
                or event.endswith("-snapshot")
            )
        )
        for event in engine.events
    )


def test_worker_command_timeline_reset_snapshot_order_and_cardinality():
    engine = _FakeWorkerEngine()
    clock_values = iter(
        (1_000_000_000, 1_200_000_000, 1_500_000_000)
    )

    def synchronize():
        engine.events.append("synchronize")

    def clock_ns():
        engine.events.append("request-clock")
        return next(clock_values)

    run = _worker_module().run_request_batch(
        engine=engine,
        policy="learned",
        prompt_rows=_worker_prompt_rows(batch_size=4),
        sampling_params=_FakeSamplingParams(),
        expected_output_tokens=16,
        synchronize=synchronize,
        clock_ns=clock_ns,
        repeat=3,
        command_timeline=True,
    )

    timeline = run["runtime"]["command_timeline"]
    assert timeline["schema_version"] == 1
    assert len(timeline["rank_snapshots"]) == 4
    assert len(timeline["cuda_rank_snapshots"]) == 4
    assert len(timeline["engine_steps"]) == 2
    assert all(
        row["steps"]
        for row in timeline["cuda_rank_snapshots"]
    )
    first_request_clock = engine.events.index("request-clock")
    assert engine.events.index("authority") < engine.events.index(
        "reset-command-timeline"
    )
    assert engine.events.index("reset") < engine.events.index(
        "reset-command-timeline"
    )
    assert engine.events.index("reset-decode-profile") + 1 == (
        first_request_clock
    )
    final_step = engine.events.index("step-1")
    final_synchronize = engine.events.index(
        "synchronize",
        final_step,
    )
    profiler_synchronizes = [
        event
        for event in engine.events
        if (
            isinstance(event, tuple)
            and event[0] == "profiler-synchronize"
        )
    ]
    assert profiler_synchronizes == [
        ("profiler-synchronize", 1),
        ("profiler-synchronize", 2),
        ("profiler-synchronize", 3),
    ]
    assert all(
        engine.events.index(event) > final_synchronize
        for event in profiler_synchronizes
    )
    assert all(event[1] != 0 for event in profiler_synchronizes)
    for event in (
        "command-snapshot",
        "cuda-snapshot",
        "step-snapshot",
    ):
        assert engine.events.index(event) > final_synchronize
    begin_event = next(
        event
        for event in engine.events
        if isinstance(event, tuple)
        and event[0] == "begin-repeat"
    )
    assert begin_event == (
        "begin-repeat",
        3,
        hashlib.sha256(
            json.dumps(
                [
                    row["token_ids"]
                    for row in _worker_prompt_rows(batch_size=4)
                ],
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest(),
    )


def test_worker_proposal_detail_uses_parent_critical_rank():
    before = tuple(
        _authority_rank(rank, offset=0)
        for rank in range(4)
    )
    after = copy.deepcopy(before)
    parent_deltas = (10.0, 9.0, 8.0, 7.0)
    detail_deltas = (
        {
            "setup": 1.0,
            "backend_submit": 1.0,
            "selection_collective": 1.0,
            "decode_authority": 1.0,
            "token_readback": 1.0,
            "materialize_register": 1.0,
        },
        {
            "setup": 7.0,
            "backend_submit": 0.0,
            "selection_collective": 0.0,
            "decode_authority": 0.0,
            "token_readback": 0.0,
            "materialize_register": 0.0,
        },
        {
            "setup": 0.0,
            "backend_submit": 7.0,
            "selection_collective": 0.0,
            "decode_authority": 0.0,
            "token_readback": 0.0,
            "materialize_register": 0.0,
        },
        {
            "setup": 0.0,
            "backend_submit": 0.0,
            "selection_collective": 7.0,
            "decode_authority": 0.0,
            "token_readback": 0.0,
            "materialize_register": 0.0,
        },
    )
    for rank, row in enumerate(after):
        row["executor"]["timing_ms"]["proposal_forward"] += (
            parent_deltas[rank]
        )
        for key, delta in detail_deltas[rank].items():
            row["executor"]["proposal_forward_detail_ms"][key] += delta

    timing = _worker_module()._draft_executor_timing_delta(
        before,
        after,
    )
    detail = (
        _worker_module()._draft_executor_proposal_detail_delta(
            before,
            after,
            draft_executor_timing=timing,
        )
    )

    assert detail["critical_rank"] == 0
    assert detail["critical_rank_ms"] == detail_deltas[0]
    assert detail["detail_sum_ms"] == pytest.approx(6.0)
    assert detail["residual_ms"] == pytest.approx(4.0)
    assert sum(detail["max_rank_ms"].values()) > (
        timing["max_rank_ms"]["proposal_forward"]
    )


def test_policy_campaign_runs_one_warmup_three_measured_and_closes():
    adapter = _FakeWorkerAdapter()
    repeats = []
    engine_factory_calls = []

    def run_batch_fn(
        *,
        engine,
        policy,
        prompt_rows,
        sampling_params,
        expected_output_tokens,
        synchronize,
        clock_ns,
        repeat,
    ):
        repeats.append(repeat)
        return _run(
            policy=policy,
            batch_size=len(prompt_rows),
            repeat=repeat,
        )

    def engine_factory(*args, **kwargs):
        engine_factory_calls.append((args, kwargs))
        return adapter

    result = _worker_module().run_policy_campaign(
        target_model="/models/target",
        draft_model="/models/draft",
        policy="learned",
        batch_size=4,
        engine_factory=engine_factory,
        sampling_params_type=_FakeSamplingParams,
        synchronize=lambda: None,
        clock_ns=lambda: 0,
        run_batch_fn=run_batch_fn,
    )

    assert repeats == [-1, 0, 1, 2]
    assert len(result["warmup_runs"]) == 1
    assert len(result["measured_runs"]) == 3
    assert adapter.close_calls == 1
    assert result["tensor_parallel_size"] == 4
    assert result["proposal_kv_allocator"] == "direct"
    assert result["proposal_slot_capacity"] == 4 * (256 + 16 + 4)
    assert engine_factory_calls[0][1]["proposal_slot_capacity"] == (
        4 * (256 + 16 + 4)
    )
    assert set(result) == {
        "policy",
        "batch_size",
        "prompt_rows",
        "warmup_runs",
        "measured_runs",
        "target_checkpoint_identifier",
        "draft_checkpoint_identifier",
        "tokenizer_identifier",
        "dtype",
        "tensor_parallel_size",
        "proposal_kv_allocator",
        "proposal_slot_capacity",
        "cuda_graph_mode",
        "cuda_graph_budget_overrides",
    }


def _command_timeline_graph_run(
    *,
    repeat,
    replays,
    timeline_repeat_index,
    request_set_sha256,
    mode="graph",
):
    run = _run(policy="learned", batch_size=4, repeat=repeat)
    graph = mode == "graph"
    command_id = 101
    engine_step_id = 201
    selected_sequence_ids_sha256 = "a" * 64
    command_identity = {
        "command_id": command_id,
        "method_name": "step",
        "requires_ack": True,
        "engine_step_id": engine_step_id,
        "repeat_index": timeline_repeat_index,
        "request_set_sha256": request_set_sha256,
        "batch_kind": "speculative",
        "speculative_selected_sequence_ids_sha256": (
            selected_sequence_ids_sha256
        ),
        "dispatch_started_monotonic_ns": 1_000,
        "dispatch_published_monotonic_ns": 2_000,
    }
    cuda_identity = {
        "command_id": command_id,
        "engine_step_id": engine_step_id,
        "repeat_index": timeline_repeat_index,
        "request_set_sha256": request_set_sha256,
        "speculative_selected_sequence_ids_sha256": (
            selected_sequence_ids_sha256
        ),
    }
    engine_step_identity = {
        "engine_step_id": engine_step_id,
        "repeat_index": timeline_repeat_index,
        "request_set_sha256": request_set_sha256,
        "batch_kind": "speculative",
        "speculative_selected_sequence_ids_sha256": (
            selected_sequence_ids_sha256
        ),
    }
    run["correctness"] = {
        "rank_graph_counters": [
            {
                "rank": rank,
                "capture_attempts": 1 if graph else 0,
                "captures": 1 if graph else 0,
                "replays": replays if graph else 0,
                "quarantines": 0,
                "fallback_pre_replay": 0,
            }
            for rank in range(4)
        ],
        "rank_graph_resources": [
            {
                "rank": rank,
                "ready_entry_count": 1 if graph else 0,
                "static_bytes": 55_000 if graph else 0,
                "reserved_bytes": 700_000_000 if graph else 0,
                "total_capture_ns": 1_500_000_000 if graph else 0,
            }
            for rank in range(4)
        ],
    }
    run["runtime"]["command_timeline"] = {
        "schema_version": 1,
        "rank_snapshots": [
            {
                "schema_version": 1,
                "rank": rank,
                "enabled": True,
                "clock": {"boot_id": "boot"},
                "rows": [{
                    "rank": rank,
                    **copy.deepcopy(command_identity),
                }],
                "dropped_rows": 0,
            }
            for rank in range(4)
        ],
        "cuda_rank_snapshots": [
            {
                "rank": rank,
                "steps": [{
                    "rank": rank,
                    **copy.deepcopy(cuda_identity),
                }],
                "collectives": [{
                    "rank": rank,
                    **copy.deepcopy(cuda_identity),
                }],
                "dropped_steps": 0,
                "dropped_collectives": 0,
            }
            for rank in range(4)
        ],
        "engine_steps": [engine_step_identity],
        "engine_dropped_steps": 0,
    }
    return run


def _run_command_timeline_campaign(*, mutate_run=None):
    adapter = _FakeWorkerAdapter()

    def run_batch_fn(**kwargs):
        repeat = kwargs["repeat"]
        timeline_repeat_index = kwargs[
            "command_timeline_repeat_index"
        ]
        request_set_sha256 = _worker_module()._request_set_sha256(
            kwargs["prompt_rows"]
        )
        run = _command_timeline_graph_run(
            repeat=repeat,
            replays=repeat + 2,
            timeline_repeat_index=timeline_repeat_index,
            request_set_sha256=request_set_sha256,
        )
        if mutate_run is not None:
            mutate_run(run, timeline_repeat_index)
        return run

    return _worker_module().run_policy_campaign(
        target_model="/models/target",
        draft_model="/models/draft",
        policy="learned",
        batch_size=4,
        engine_factory=lambda *args, **kwargs: adapter,
        sampling_params_type=_FakeSamplingParams,
        synchronize=lambda: None,
        clock_ns=lambda: 0,
        run_batch_fn=run_batch_fn,
        warmup_runs=1,
        measured_runs=5,
        cuda_graph_mode="graph",
        command_timeline=True,
    )


def test_policy_campaign_command_timeline_is_exact_tp4_b4_q4_and_once():
    adapter = _FakeWorkerAdapter()
    timeline_repeat_indices = []
    monotonic_times = iter(range(100, 112))
    wall_times = iter(range(1_000, 1_012))

    def run_batch_fn(**kwargs):
        repeat = kwargs["repeat"]
        timeline_repeat_index = kwargs[
            "command_timeline_repeat_index"
        ]
        timeline_repeat_indices.append(timeline_repeat_index)
        run = _command_timeline_graph_run(
            repeat=repeat,
            replays=repeat + 2,
            timeline_repeat_index=timeline_repeat_index,
            request_set_sha256=_worker_module()._request_set_sha256(
                kwargs["prompt_rows"]
            ),
        )
        return run

    result = _worker_module().run_policy_campaign(
        target_model="/models/target",
        draft_model="/models/draft",
        policy="learned",
        batch_size=4,
        engine_factory=lambda *args, **kwargs: adapter,
        sampling_params_type=_FakeSamplingParams,
        synchronize=lambda: None,
        clock_ns=lambda: 0,
        monotonic_clock_ns=lambda: next(monotonic_times),
        wall_clock_ns=lambda: next(wall_times),
        run_batch_fn=run_batch_fn,
        warmup_runs=1,
        measured_runs=5,
        cuda_graph_mode="graph",
        command_timeline=True,
    )

    assert adapter.engine.command_timeline_configurations == 1
    assert adapter.engine.decode_profile_configurations == 1
    assert len(result["warmup_runs"]) == 1
    assert len(result["measured_runs"]) == 5
    assert timeline_repeat_indices == [0, 1, 2, 3, 4, 5]
    assert [run["repeat"] for run in result["warmup_runs"]] == [-1]
    assert [run["repeat"] for run in result["measured_runs"]] == [
        0,
        1,
        2,
        3,
        4,
    ]
    assert [
        run["command_timeline_repeat_index"]
        for run in result["warmup_runs"] + result["measured_runs"]
    ] == [0, 1, 2, 3, 4, 5]
    assert [
        run["correctness"]["rank_graph_counters"][0]["replays"]
        for run in result["warmup_runs"] + result["measured_runs"]
    ] == [1, 2, 3, 4, 5, 6]
    for run in result["warmup_runs"] + result["measured_runs"]:
        interval = run["campaign_interval"]
        assert interval["finished_at_unix_ns"] > (
            interval["started_at_unix_ns"]
        )
        assert interval["finished_at_monotonic_ns"] > (
            interval["started_at_monotonic_ns"]
        )
    assert all(
        len(run["runtime"]["command_timeline"]["rank_snapshots"])
        == 4
        and len(
            run["runtime"]["command_timeline"][
                "cuda_rank_snapshots"
            ]
        )
        == 4
        and run["runtime"]["command_timeline"]["engine_steps"]
        for run in result["measured_runs"]
    )


def test_policy_campaign_rejects_diagnostic_counts_without_timeline():
    adapter = _FakeWorkerAdapter()

    with pytest.raises(
        ValueError,
        match="one warmup and five measured runs require command timeline",
    ):
        _worker_module().run_policy_campaign(
            target_model="/models/target",
            draft_model="/models/draft",
            policy="learned",
            batch_size=4,
            engine_factory=lambda *args, **kwargs: adapter,
            sampling_params_type=_FakeSamplingParams,
            synchronize=lambda: None,
            clock_ns=lambda: 0,
            run_batch_fn=lambda **kwargs: _run(
                policy=kwargs["policy"],
                batch_size=len(kwargs["prompt_rows"]),
                repeat=kwargs["repeat"],
            ),
            warmup_runs=1,
            measured_runs=5,
        )


def test_policy_campaign_command_timeline_rejects_graph_counter_drift():
    adapter = _FakeWorkerAdapter()

    def run_batch_fn(**kwargs):
        repeat = kwargs["repeat"]
        run = _command_timeline_graph_run(
            repeat=repeat,
            replays=repeat + 2,
            timeline_repeat_index=kwargs[
                "command_timeline_repeat_index"
            ],
            request_set_sha256=_worker_module()._request_set_sha256(
                kwargs["prompt_rows"]
            ),
        )
        if repeat == 2:
            run["correctness"]["rank_graph_counters"][3][
                "captures"
            ] = 2
        return run

    with pytest.raises(
        ValueError,
        match="graph capture counters changed",
    ):
        _worker_module().run_policy_campaign(
            target_model="/models/target",
            draft_model="/models/draft",
            policy="learned",
            batch_size=4,
            engine_factory=lambda *args, **kwargs: adapter,
            sampling_params_type=_FakeSamplingParams,
            synchronize=lambda: None,
            clock_ns=lambda: 0,
            run_batch_fn=run_batch_fn,
            warmup_runs=1,
            measured_runs=5,
            cuda_graph_mode="graph",
            command_timeline=True,
        )


def test_policy_campaign_graph_warmup_failure_reports_rank_and_evidence():
    def mutate_run(run, timeline_repeat_index):
        if timeline_repeat_index == 0:
            run["correctness"]["rank_graph_counters"][2][
                "replays"
            ] = 0

    with pytest.raises(ValueError) as error:
        _run_command_timeline_campaign(mutate_run=mutate_run)

    message = str(error.value)
    assert "graph warmup counters are invalid" in message
    assert "rank=2" in message
    assert (
        'counters={"capture_attempts":1,"captures":1,'
        '"fallback_pre_replay":0,"quarantines":0,"rank":2,'
        '"replays":0}'
    ) in message
    assert (
        'resources={"rank":2,"ready_entry_count":1,'
        '"reserved_bytes":700000000,"static_bytes":55000,'
        '"total_capture_ns":1500000000}'
    ) in message


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda timeline: timeline["rank_snapshots"][0].update(
                rows=[]
            ),
            "command row evidence is missing",
        ),
        (
            lambda timeline: timeline["rank_snapshots"][0].update(
                dropped_rows=1
            ),
            "command rows were dropped",
        ),
        (
            lambda timeline: timeline["cuda_rank_snapshots"][0].update(
                dropped_steps=1
            ),
            "CUDA evidence was dropped",
        ),
        (
            lambda timeline: timeline["cuda_rank_snapshots"][0].update(
                dropped_collectives=1
            ),
            "CUDA evidence was dropped",
        ),
        (
            lambda timeline: timeline.update(engine_dropped_steps=1),
            "engine step evidence was dropped",
        ),
        (
            lambda timeline: timeline["rank_snapshots"][0]["rows"][
                0
            ].update(repeat_index=99),
            "repeat identity mismatch",
        ),
        (
            lambda timeline: timeline["cuda_rank_snapshots"][0][
                "steps"
            ][0].update(request_set_sha256="b" * 64),
            "request digest mismatch",
        ),
        (
            lambda timeline: timeline["cuda_rank_snapshots"][0][
                "steps"
            ][0].update(command_id=999),
            "unknown command identity",
        ),
        (
            lambda timeline: timeline["cuda_rank_snapshots"][0][
                "steps"
            ][0].update(engine_step_id=999),
            "unknown engine step identity",
        ),
        (
            lambda timeline: timeline["engine_steps"][0].update(
                speculative_selected_sequence_ids_sha256="b" * 64
            ),
            "selected-sequence digest mismatch",
        ),
        (
            lambda timeline: timeline["rank_snapshots"][0]["rows"][
                0
            ].pop("command_id"),
            "command identity is malformed",
        ),
        (
            lambda timeline: timeline["cuda_rank_snapshots"][0][
                "steps"
            ][0].pop("request_set_sha256"),
            "CUDA step identity is malformed",
        ),
        (
            lambda timeline: timeline["engine_steps"][0].pop(
                "repeat_index"
            ),
            "engine step identity is malformed",
        ),
        (
            lambda timeline: timeline["rank_snapshots"][3]["rows"][
                0
            ].update(command_id=999),
            "command inventories differ across ranks",
        ),
        (
            lambda timeline: timeline["rank_snapshots"][1].update(
                rank=True
            ),
            "rank inventory is invalid",
        ),
        (
            lambda timeline: timeline["cuda_rank_snapshots"][1].update(
                rank=True
            ),
            "rank inventory is invalid",
        ),
        (
            lambda timeline: timeline["rank_snapshots"][1]["rows"][
                0
            ].update(rank=True),
            "command identity is malformed",
        ),
        (
            lambda timeline: timeline["cuda_rank_snapshots"][1][
                "steps"
            ][0].update(rank=True),
            "CUDA step identity is malformed",
        ),
        (
            lambda timeline: timeline["cuda_rank_snapshots"][1][
                "collectives"
            ][0].update(rank=True),
            "CUDA collective identity is malformed",
        ),
    ],
)
def test_policy_campaign_command_timeline_rejects_invalid_evidence(
    mutation,
    message,
):
    def mutate_run(run, timeline_repeat_index):
        if timeline_repeat_index == 2:
            mutation(run["runtime"]["command_timeline"])

    with pytest.raises(ValueError, match=message):
        _run_command_timeline_campaign(mutate_run=mutate_run)


def test_policy_campaign_command_timeline_rejects_warmup_reuse():
    def mutate_run(run, timeline_repeat_index):
        if timeline_repeat_index != 1:
            return
        timeline = run["runtime"]["command_timeline"]
        for rank_snapshot in timeline["rank_snapshots"]:
            rank_snapshot["rows"][0]["repeat_index"] = 0
        for cuda_snapshot in timeline["cuda_rank_snapshots"]:
            for row in (
                cuda_snapshot["steps"]
                + cuda_snapshot["collectives"]
            ):
                row["repeat_index"] = 0
        timeline["engine_steps"][0]["repeat_index"] = 0

    with pytest.raises(ValueError, match="repeat identity mismatch"):
        _run_command_timeline_campaign(mutate_run=mutate_run)


@pytest.mark.parametrize(
    ("timeline_repeat_index", "public_repeat"),
    (
        (0, 0),
        (2, 99),
    ),
)
def test_policy_campaign_command_timeline_rejects_public_repeat_drift(
    timeline_repeat_index,
    public_repeat,
):
    def mutate_run(run, actual_timeline_repeat_index):
        if actual_timeline_repeat_index == timeline_repeat_index:
            run["repeat"] = public_repeat

    with pytest.raises(ValueError, match="public repeat mismatch"):
        _run_command_timeline_campaign(mutate_run=mutate_run)


def test_worker_cli_command_timeline_is_opt_in():
    module = _worker_module()
    common = [
        "--target-model",
        "/models/target",
        "--draft-model",
        "/models/draft",
        "--policy",
        "learned",
        "--batch-size",
        "4",
        "--out",
        "/tmp/result.json",
    ]

    assert module.parse_args(common).command_timeline is False
    assert module.parse_args(
        [*common, "--command-timeline"]
    ).command_timeline is True


def test_policy_campaign_forwards_explicit_graph_budget_overrides():
    adapter = _FakeWorkerAdapter()
    engine_factory_calls = []

    def engine_factory(*args, **kwargs):
        engine_factory_calls.append((args, kwargs))
        return adapter

    _worker_module().run_policy_campaign(
        target_model="/models/target",
        draft_model="/models/draft",
        policy="learned",
        batch_size=4,
        engine_factory=engine_factory,
        sampling_params_type=_FakeSamplingParams,
        synchronize=lambda: None,
        clock_ns=lambda: 0,
        run_batch_fn=lambda **kwargs: _run(
            policy=kwargs["policy"],
            batch_size=len(kwargs["prompt_rows"]),
            repeat=kwargs["repeat"],
        ),
        warmup_runs=0,
        measured_runs=1,
        cuda_graph_mode="graph",
        cuda_graph_max_reserved_bytes=4 * 1024 * 1024 * 1024,
        cuda_graph_max_total_capture_ns=120_000_000_000,
        cuda_graph_max_single_capture_ns=60_000_000_000,
    )

    kwargs = engine_factory_calls[0][1]
    assert kwargs[
        "cuda_graph_max_reserved_bytes"
    ] == 4 * 1024 * 1024 * 1024
    assert kwargs[
        "cuda_graph_max_total_capture_ns"
    ] == 120_000_000_000
    assert kwargs[
        "cuda_graph_max_single_capture_ns"
    ] == 60_000_000_000


def test_policy_campaign_supports_two_warmups_and_eight_measured():
    adapter = _FakeWorkerAdapter()
    repeats = []

    def run_batch_fn(**kwargs):
        repeats.append(kwargs["repeat"])
        return _run(
            policy=kwargs["policy"],
            batch_size=len(kwargs["prompt_rows"]),
            repeat=kwargs["repeat"],
        )

    result = _worker_module().run_policy_campaign(
        target_model="/models/target",
        draft_model="/models/draft",
        policy="learned",
        batch_size=4,
        engine_factory=lambda *args, **kwargs: adapter,
        sampling_params_type=_FakeSamplingParams,
        synchronize=lambda: None,
        clock_ns=lambda: 0,
        run_batch_fn=run_batch_fn,
        warmup_runs=2,
        measured_runs=8,
    )

    assert repeats == [-2, -1, *range(8)]
    assert len(result["warmup_runs"]) == 2
    assert len(result["measured_runs"]) == 8
    assert adapter.close_calls == 1


def test_policy_campaign_records_deterministic_campaign_intervals():
    adapter = _FakeWorkerAdapter()
    repeats = []
    wall_times = iter((1_000, 2_000, 3_000, 4_000))

    def run_batch_fn(**kwargs):
        repeats.append(kwargs["repeat"])
        return _run(
            policy=kwargs["policy"],
            batch_size=len(kwargs["prompt_rows"]),
            repeat=kwargs["repeat"],
        )

    result = _worker_module().run_policy_campaign(
        target_model="/models/target",
        draft_model="/models/draft",
        policy="learned",
        batch_size=4,
        engine_factory=lambda *args, **kwargs: adapter,
        sampling_params_type=_FakeSamplingParams,
        synchronize=lambda: None,
        clock_ns=lambda: 0,
        wall_clock_ns=lambda: next(wall_times),
        run_batch_fn=run_batch_fn,
        warmup_runs=1,
        measured_runs=1,
    )

    assert repeats == [-1, 0]
    assert result["warmup_runs"][0]["campaign_interval"] == {
        "started_at_unix_ns": 1_000,
        "finished_at_unix_ns": 2_000,
    }
    assert result["measured_runs"][0]["campaign_interval"] == {
        "started_at_unix_ns": 3_000,
        "finished_at_unix_ns": 4_000,
    }
    assert result["measured_runs"][0]["repeat"] == 0
    assert adapter.close_calls == 1


def test_policy_campaign_rejects_invalid_campaign_interval():
    adapter = _FakeWorkerAdapter()
    wall_times = iter((2_000, 1_000))

    with pytest.raises(ValueError, match="campaign interval is invalid"):
        _worker_module().run_policy_campaign(
            target_model="/models/target",
            draft_model="/models/draft",
            policy="target",
            batch_size=1,
            engine_factory=lambda *args, **kwargs: adapter,
            sampling_params_type=_FakeSamplingParams,
            synchronize=lambda: None,
            clock_ns=lambda: 0,
            wall_clock_ns=lambda: next(wall_times),
            run_batch_fn=lambda **kwargs: _run(
                policy=kwargs["policy"],
                batch_size=len(kwargs["prompt_rows"]),
                repeat=kwargs["repeat"],
            ),
            warmup_runs=1,
            measured_runs=1,
        )

    assert adapter.close_calls == 1


def test_policy_campaign_closes_adapter_on_failure():
    adapter = _FakeWorkerAdapter()

    with pytest.raises(RuntimeError, match="injected"):
        _worker_module().run_policy_campaign(
            target_model="/models/target",
            draft_model="/models/draft",
            policy="target",
            batch_size=1,
            engine_factory=lambda *args, **kwargs: adapter,
            sampling_params_type=_FakeSamplingParams,
            synchronize=lambda: None,
            clock_ns=lambda: 0,
            run_batch_fn=lambda **kwargs: (
                (_ for _ in ()).throw(RuntimeError("injected"))
            ),
        )

    assert adapter.close_calls == 1


def test_parent_gate_launches_four_isolated_worker_cells(tmp_path):
    commands = []

    def worker_runner(command, *, log_path, cwd):
        commands.append(list(command))
        policy = command[command.index("--policy") + 1]
        batch_size = int(
            command[command.index("--batch-size") + 1]
        )
        output_path = Path(
            command[command.index("--out") + 1]
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(_worker(policy, batch_size)),
            encoding="utf-8",
        )
        Path(log_path).write_text(
            f"{policy}:b{batch_size}\n",
            encoding="utf-8",
        )
        return 0

    source = tmp_path / "source.py"
    source.write_text("source\n", encoding="utf-8")
    output_path = tmp_path / "result.json"

    artifact = _gate().run_performance_gate(
        target_model="/models/target",
        draft_model="/models/draft",
        output_path=output_path,
        repo_root=tmp_path,
        worker_script=WORKER_PATH,
        worker_runner=worker_runner,
        python_executable=sys.executable,
        source_files=("source.py",),
        environment=_environment(),
    )

    assert len(commands) == 4
    assert {
        (
            command[command.index("--policy") + 1],
            int(command[command.index("--batch-size") + 1]),
        )
        for command in commands
    } == {
        ("target", 1),
        ("learned", 1),
        ("target", 4),
        ("learned", 4),
    }
    assert json.loads(output_path.read_text(encoding="utf-8")) == artifact
    assert artifact["direction"] == "POSITIVE"


def test_parent_gate_propagates_worker_failure(tmp_path):
    source = tmp_path / "source.py"
    source.write_text("source\n", encoding="utf-8")

    def worker_runner(command, *, log_path, cwd):
        policy = command[command.index("--policy") + 1]
        batch_size = int(
            command[command.index("--batch-size") + 1]
        )
        if policy == "learned" and batch_size == 4:
            return 17
        output_path = Path(
            command[command.index("--out") + 1]
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(_worker(policy, batch_size)),
            encoding="utf-8",
        )
        return 0

    with pytest.raises(RuntimeError, match="learned:b4"):
        _gate().run_performance_gate(
            target_model="/models/target",
            draft_model="/models/draft",
            output_path=tmp_path / "result.json",
            repo_root=tmp_path,
            worker_script=WORKER_PATH,
            worker_runner=worker_runner,
            python_executable=sys.executable,
            source_files=("source.py",),
            environment=_environment(),
        )


def test_independent_verifier_recomputes_and_checks_source_hashes(
    tmp_path,
):
    source = tmp_path / "source.py"
    source.write_text("before\n", encoding="utf-8")
    artifact = _gate().build_performance_artifact(
        worker_results=_workers(),
        environment=_environment(),
        source_files={
            "source.py": hashlib.sha256(
                source.read_bytes()
            ).hexdigest(),
        },
    )
    artifact_path = tmp_path / "result.json"
    artifact_path.write_text(
        json.dumps(artifact),
        encoding="utf-8",
    )

    receipt = _verifier().verify_performance_artifact(
        artifact_path,
        tmp_path,
    )

    assert receipt == {
        "status": "PASS",
        "classification": "PILOT_ONLY",
        "direction": "POSITIVE",
        "batch_directions": {
            "1": "IMPROVED",
            "4": "IMPROVED",
        },
        "source_files_verified": 1,
    }

    source.write_text("after\n", encoding="utf-8")
    with pytest.raises(ValueError, match="source hash"):
        _verifier().verify_performance_artifact(
            artifact_path,
            tmp_path,
        )


def test_b4_diagnostic_retains_raw_detail_and_classifies_stationarity():
    artifact = _diagnostic().build_b4_timing_diagnostic(
        target_worker=_diagnostic_worker("target"),
        learned_worker=_diagnostic_worker("learned"),
        environment=_environment(),
        source_files={"tools/source.py": "a" * 64},
    )

    receipt = _diagnostic().validate_b4_timing_diagnostic(artifact)

    assert artifact["schema_version"] == 1
    assert artifact["status"] == "PASS"
    assert artifact["classification"] == "STABLE"
    assert artifact["exact_parity"] is True
    assert len(artifact["workers"]["target"]["warmup_runs"]) == 2
    assert len(artifact["workers"]["learned"]["measured_runs"]) == 8
    assert artifact["workers"]["learned"]["measured_runs"][0][
        "runtime"
    ]["draft_executor_proposal_detail"]["ranks"][3][
        "backend_submit"
    ] > 0.0
    stationarity = {
        (row["policy"], row["metric"]): row
        for row in artifact["stationarity"]["rows"]
    }
    assert stationarity[("learned", "e2e_s")]["stable"] is True
    assert stationarity[
        ("learned", "executor_proposal_forward_ms")
    ]["stable"] is True
    assert stationarity[
        ("learned", "executor_detail_backend_submit_ms")
    ]["count"] == 8
    assert receipt == {
        "status": "PASS",
        "classification": "STABLE",
        "exact_parity": True,
        "measured_runs": 8,
    }


def test_b4_diagnostic_classifies_strong_late_drift_as_unstable():
    learned = _diagnostic_worker("learned")
    for run in learned["measured_runs"][4:]:
        for row in run["timing"]["per_request"]:
            row["completion_latency_s"] *= 2.0

    artifact = _diagnostic().build_b4_timing_diagnostic(
        target_worker=_diagnostic_worker("target"),
        learned_worker=learned,
        environment=_environment(),
        source_files={"tools/source.py": "a" * 64},
    )

    assert artifact["classification"] == "UNSTABLE"
    learned_e2e = next(
        row
        for row in artifact["stationarity"]["rows"]
        if row["policy"] == "learned" and row["metric"] == "e2e_s"
    )
    assert learned_e2e["stable"] is False


def test_b4_diagnostic_rejects_parity_and_source_drift(tmp_path):
    target = _diagnostic_worker("target")
    learned = _diagnostic_worker("learned")
    learned["measured_runs"][3]["outputs"][0][0] = 999
    with pytest.raises(ValueError, match="parity"):
        _diagnostic().build_b4_timing_diagnostic(
            target_worker=target,
            learned_worker=learned,
            environment=_environment(),
            source_files={"tools/source.py": "a" * 64},
        )

    source = tmp_path / "source.py"
    source.write_text("before\n", encoding="utf-8")
    artifact = _diagnostic().build_b4_timing_diagnostic(
        target_worker=_diagnostic_worker("target"),
        learned_worker=_diagnostic_worker("learned"),
        environment=_environment(),
        source_files={
            "source.py": hashlib.sha256(source.read_bytes()).hexdigest(),
        },
    )
    artifact_path = tmp_path / "diagnostic.json"
    artifact_path.write_text(json.dumps(artifact), encoding="utf-8")
    receipt = _diagnostic_verifier().verify_b4_timing_diagnostic(
        artifact_path,
        tmp_path,
    )
    assert receipt["source_files_verified"] == 1

    source.write_text("after\n", encoding="utf-8")
    with pytest.raises(ValueError, match="source hash"):
        _diagnostic_verifier().verify_b4_timing_diagnostic(
            artifact_path,
            tmp_path,
        )


def test_b4_diagnostic_remote_runner_is_bounded_and_dual_verified():
    assert DIAGNOSTIC_REMOTE_SCRIPT_PATH.exists()
    source = DIAGNOSTIC_REMOTE_SCRIPT_PATH.read_text(encoding="utf-8")

    for required in (
        "sitian@10.232.195.203",
        "ControlMaster=no",
        "ControlPath=none",
        "/data00/home/sitian/miniconda3/envs/py311/bin/python",
        'GPU_INDICES="${GPU_INDICES:-3,4,6,7}"',
        'HARD_TIMEOUT_SECONDS="${HARD_TIMEOUT_SECONDS:-5400}"',
        "timeout --signal=TERM --kill-after=30s",
        "--warmup-runs 2",
        "--measured-runs 8",
        "-m pytest -q",
        "executor-tests.log",
        "autoregressive_draft_b4_timing_diagnostic.py",
        "verify_autoregressive_draft_b4_timing_diagnostic.py",
        "verify.remote.json",
        "verify.local.json",
        "manifest.sha256",
        "shasum -a 256 -c manifest.sha256",
    ):
        assert required in source

    assert source.count(
        "verify_autoregressive_draft_b4_timing_diagnostic.py"
    ) >= 2
    assert "pkill" not in source
    assert "killall" not in source
    assert "kill -9" not in source


def test_remote_runner_is_source_bound_bounded_and_dual_verified():
    assert REMOTE_SCRIPT_PATH.exists()
    source = REMOTE_SCRIPT_PATH.read_text(encoding="utf-8")

    for required in (
        "sitian@10.232.195.203",
        "ControlMaster=no",
        "ControlPath=none",
        "/data00/home/sitian/miniconda3/envs/py311/bin/python",
        "target-qwen3-1.7b",
        "/draft",
        'REMOTE_PACKAGE_ROOT="${REMOTE_PACKAGE_ROOT:-${REMOTE_BASE}/run_packages}"',
        'PYTHONPATH="${remote_package_root}:${remote_source}"',
        'GPU_INDICES="${GPU_INDICES:-3,4,6,7}"',
        'DIST_PORT="${DIST_PORT:-29631}"',
        'MASTER_PORT="${MASTER_PORT:-29731}"',
        'HARD_TIMEOUT_SECONDS="${HARD_TIMEOUT_SECONDS:-3600}"',
        "source.tar",
        "gpu-before.txt",
        "gpu-after.txt",
        "timeout --signal=TERM",
        "autoregressive_draft_performance_gate.py",
        "autoregressive_draft_performance_worker.py",
        "verify_autoregressive_draft_performance_gate.py",
        "verify.remote.json",
        "verify.local.json",
        "manifest.sha256",
        "shasum -a 256 -c manifest.sha256",
    ):
        assert required in source

    assert "pkill" not in source
    assert "killall" not in source
    assert "kill -9" not in source
