"""Dependency-light contract tests for the focused H2D diagnostic."""

from __future__ import annotations

import ast
import copy
import importlib.util
import os
import types

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCHEMA = "qwen35.tp4-32k-h2d-slot-reuse-causal-diagnostic.v1"
GATE_PATH = os.path.join(
    ROOT,
    "tools",
    "qwen35_tp4_32k_h2d_slot_reuse_causal_diagnostic_gate.py",
)
WORKER_PATH = os.path.join(
    ROOT,
    "tools",
    "qwen35_tp4_32k_h2d_slot_reuse_causal_diagnostic_worker.py",
)
VERIFIER_PATH = os.path.join(
    ROOT,
    "tools",
    "verify_qwen35_tp4_32k_h2d_slot_reuse_causal_diagnostic.py",
)


def _load_class_method(relative_path, class_name, method_name):
    path = os.path.join(ROOT, relative_path)
    tree = ast.parse(open(path).read(), filename=path)
    class_node = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef)
        and node.name == class_name
    )
    method_node = next(
        node
        for node in class_node.body
        if isinstance(node, ast.FunctionDef)
        and node.name == method_name
    )
    method_node.decorator_list = []
    module = ast.Module(body=[method_node], type_ignores=[])
    namespace = {"H2D_SLOT_REUSE_SCHEMA": SCHEMA}
    exec(
        compile(
            ast.fix_missing_locations(module),
            path,
            "exec",
        ),
        namespace,
    )
    return namespace[method_name]


def _engine_method(name):
    return _load_class_method(
        "tinyvllm/engine/llm_engine.py",
        "LLMEngine",
        name,
    )


def _runner_method(name):
    return _load_class_method(
        "tinyvllm/engine/model_runner.py",
        "ModelRunner",
        name,
    )


def _acks(rows):
    return tuple(
        types.SimpleNamespace(
            rank=row["rank"],
            result=row,
        )
        for row in rows
    )


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_engine_configure_collects_exact_rank_inventory():
    configure = _engine_method(
        "configure_h2d_slot_reuse_diagnostic"
    )
    engine = types.SimpleNamespace(
        model_runner=types.SimpleNamespace(world_size=4),
        call_model_runner_acknowledged=lambda *args, **kwargs: (
            {"rank": 0, "mode": "observe"},
            _acks([
                {"rank": 1, "mode": "observe"},
                {"rank": 2, "mode": "observe"},
                {"rank": 3, "mode": "observe"},
            ]),
        ),
    )
    assert configure(engine, "observe", timeout_s=60.0) == {
        "mode": "observe",
        "rank_inventory": [0, 1, 2, 3],
    }


@pytest.mark.parametrize(
    "rows",
    (
        [
            {"rank": 0, "mode": "observe"},
            {"rank": 1, "mode": "observe"},
            {"rank": 3, "mode": "observe"},
        ],
        [
            {"rank": 0, "mode": "observe"},
            {"rank": 1, "mode": "observe"},
            {"rank": 1, "mode": "observe"},
            {"rank": 3, "mode": "observe"},
        ],
        [
            {"rank": 0, "mode": "observe"},
            {"rank": 1, "mode": "observe"},
            {"rank": 2, "mode": "off"},
            {"rank": 3, "mode": "observe"},
        ],
    ),
)
def test_engine_configure_rejects_rank_or_mode_mismatch(rows):
    configure = _engine_method(
        "configure_h2d_slot_reuse_diagnostic"
    )
    engine = types.SimpleNamespace(
        model_runner=types.SimpleNamespace(world_size=4),
        call_model_runner_acknowledged=lambda *args, **kwargs: (
            rows[0],
            _acks(rows[1:]),
        ),
    )
    with pytest.raises(ValueError):
        configure(engine, "observe", timeout_s=60.0)


def test_engine_context_is_all_rank_and_exact():
    configure_context = _engine_method(
        "set_h2d_slot_reuse_diagnostic_context"
    )
    rows = [
        {
            "rank": rank,
            "engine_step": 7,
            "attention_stage": "decode",
        }
        for rank in range(4)
    ]
    engine = types.SimpleNamespace(
        model_runner=types.SimpleNamespace(world_size=4),
        call_model_runner_acknowledged=lambda *args, **kwargs: (
            rows[0],
            _acks(rows[1:]),
        ),
    )
    assert configure_context(
        engine,
        7,
        attention_stage="decode",
        timeout_s=60.0,
    ) == {
        "engine_step": 7,
        "attention_stage": "decode",
        "rank_inventory": [0, 1, 2, 3],
    }


def _drain_row(rank, mode="observe", schema=SCHEMA):
    return {
        "rank": rank,
        "schema": schema,
        "mode": mode,
        "stream_inventory": [],
        "read_rows": [],
        "overwrite_rows": [],
    }


def test_engine_drain_returns_rank_sorted_rows():
    drain = _engine_method(
        "drain_h2d_slot_reuse_diagnostic"
    )
    rows = [_drain_row(rank) for rank in range(4)]
    engine = types.SimpleNamespace(
        model_runner=types.SimpleNamespace(world_size=4),
        call_model_runner_acknowledged=lambda *args, **kwargs: (
            rows[0],
            _acks([rows[3], rows[1], rows[2]]),
        ),
    )
    result = drain(
        engine,
        timing_epsilon_ms=0.2,
        expected_mode="observe",
        timeout_s=60.0,
    )
    assert tuple(row["rank"] for row in result) == (0, 1, 2, 3)


@pytest.mark.parametrize(
    "mutate",
    (
        lambda rows: rows.__setitem__(
            2,
            _drain_row(2, schema="wrong"),
        ),
        lambda rows: rows.__setitem__(
            2,
            _drain_row(2, mode="off"),
        ),
        lambda rows: rows.__setitem__(
            2,
            _drain_row(1),
        ),
    ),
)
def test_engine_drain_rejects_schema_mode_and_rank_mismatch(mutate):
    drain = _engine_method(
        "drain_h2d_slot_reuse_diagnostic"
    )
    rows = [_drain_row(rank) for rank in range(4)]
    mutate(rows)
    engine = types.SimpleNamespace(
        model_runner=types.SimpleNamespace(world_size=4),
        call_model_runner_acknowledged=lambda *args, **kwargs: (
            rows[0],
            _acks(rows[1:]),
        ),
    )
    with pytest.raises(ValueError):
        drain(
            engine,
            timing_epsilon_ms=0.2,
            expected_mode="observe",
            timeout_s=60.0,
        )


def test_model_runner_receipts_are_exact_and_require_kv_offload():
    configure = _runner_method(
        "configure_h2d_slot_reuse_diagnostic"
    )
    context = _runner_method(
        "set_h2d_slot_reuse_diagnostic_context"
    )
    drain = _runner_method(
        "drain_h2d_slot_reuse_diagnostic"
    )
    manager = types.SimpleNamespace(
        configure_h2d_slot_reuse_diagnostic=(
            lambda mode: {"rank": 3, "mode": mode}
        ),
        set_h2d_slot_reuse_context=lambda **kwargs: kwargs,
        drain_h2d_slot_reuse_diagnostic=lambda **kwargs: (
            _drain_row(3)
        ),
    )
    runner = types.SimpleNamespace(rank=3, kv_offload=manager)
    assert configure(runner, "observe") == {
        "rank": 3,
        "mode": "observe",
    }
    assert context(runner, 5, "decode") == {
        "rank": 3,
        "engine_step": 5,
        "attention_stage": "decode",
    }
    assert context(runner, 0, "prefill") == {
        "rank": 3,
        "engine_step": 0,
        "attention_stage": "prefill",
    }
    with pytest.raises(ValueError, match="prefill or decode"):
        context(runner, 1, "spec_verify")
    assert drain(runner, 0.2) == _drain_row(3)
    with pytest.raises(RuntimeError, match="KV offload"):
        configure(
            types.SimpleNamespace(rank=0, kv_offload=None),
            "observe",
        )


def test_gate_frozen_constants_and_baseline_only_keys():
    gate = _load_module("focused_h2d_gate", GATE_PATH)
    assert gate.SCHEMA == SCHEMA
    assert gate.MODES == ("observe", "control")
    assert gate.BATCH_SIZES == (1, 4)
    assert gate.REQUIRED_CELL_KEYS == (
        "observe:b1",
        "observe:b4",
        "control:b1",
        "control:b4",
    )
    assert gate.POLICY == "baseline"
    assert gate.PROMPT_TOKENS == 32768
    assert gate.MAX_OUTPUT_TOKENS == 8
    assert gate.MAX_PROPOSAL_TOKENS == 4
    assert gate.WORLD_SIZE == 4
    assert gate.BLOCK_SIZE == 256
    assert gate.GPU_BLOCKS == 68
    assert gate.LOGICAL_BLOCKS == 640
    assert gate.BLOCKWISE_BLOCKS == 8
    assert gate.TIMING_EPSILON_MS == 0.2
    assert gate.TOP_K == 5
    with pytest.raises(ValueError):
        gate.cell_key("native_mtp", 1)
    with pytest.raises(ValueError):
        gate.cell_key("baseline", 1)


def test_compact_prediction_logits_uses_stable_top_k_order():
    gate = _load_module("focused_h2d_gate_logits", GATE_PATH)
    rows = gate.compact_prediction_logits(
        [[0.1, 0.9, 0.9, -1.0, 0.2, 0.3]],
        observation={
            "prediction_rows": [
                {
                    "sequence_id": 7,
                    "prompt_index": 0,
                    "input_token_id": 11,
                    "position": 32768,
                    "context_length": 32769,
                }
            ],
        },
        prediction_index=1,
        top_k=5,
    )
    assert rows == [{
        "sequence_id": 7,
        "prompt_index": 0,
        "prediction_index": 1,
        "input_token_id": 11,
        "position": 32768,
        "context_length": 32769,
        "top_tokens": [1, 2, 5, 4, 0],
        "top_logits": [0.9, 0.9, 0.3, 0.2, 0.1],
        "top1_margin": 0.0,
        "argmax_token": 1,
    }]


class _FakeGenerationEngine:
    def __init__(self, target_forward_capture):
        self.calls = []
        self.step_index = 0
        self.last_step_observation = None
        self.target_forward_capture = target_forward_capture
        self.prompt_token_ids = []
        self.scheduler = types.SimpleNamespace(
            waiting=[17],
            prefilling=[],
            running=[],
        )
        self.model_runner = types.SimpleNamespace(
            call=self._model_runner_call,
        )

    def _model_runner_call(self, method_name, *args, **kwargs):
        if method_name == "run":
            self.target_forward_capture[
                "ordinary_decode_target_forward_calls"
            ] += 1
        return None

    def configure_h2d_slot_reuse_diagnostic(
        self,
        mode,
        *,
        timeout_s,
    ):
        self.calls.append(("configure", mode, timeout_s))
        return {"mode": mode, "rank_inventory": [0, 1, 2, 3]}

    def enable_step_logits_authority_recording(
        self,
        enabled,
        *,
        timeout_s,
    ):
        self.calls.append(("logits", enabled, timeout_s))

    def add_request(self, token_ids, sampling_params):
        self.calls.append(("add", tuple(token_ids)))
        self.prompt_token_ids.append(list(token_ids))

    def is_finished(self):
        return self.step_index >= 8

    def set_h2d_slot_reuse_diagnostic_context(
        self,
        engine_step,
        attention_stage,
        *,
        timeout_s,
    ):
        self.calls.append(
            ("context", engine_step, attention_stage)
        )

    def step(self):
        index = self.step_index
        self.step_index += 1
        self.model_runner.call("run", (), index == 0)
        self.scheduler.waiting = []
        self.scheduler.running = [17]
        self.last_step_observation = {
            "policy_branch": (
                "legacy_prefill" if index == 0 else "legacy_decode"
            ),
            "batch_kind": None,
            "is_prefill": index == 0,
            "do_sample": True,
            "scheduled": [{
                "seq_id": 17,
                "is_decode": index > 0,
                "do_sample": True,
                "prefill_chunk_start": 0,
                "prefill_chunk_end": (
                    len(self.prompt_token_ids[0])
                    if index == 0
                    else 0
                ),
                "prefill_chunk_final": index == 0,
            }],
            "new_completion_tokens_by_seq": {
                17: [200 + index],
            },
            "finished_seq_ids": [17] if index == 7 else [],
            "speculative_selected_seq_ids": [],
            "speculative_suppressed_seq_ids": [],
            "speculative_output_token_counts": {},
            "speculative_accepted_draft_token_counts": {},
            "speculative_proposal_token_counts": {},
            "speculative_proposal_token_ids_by_seq": {},
            "speculative_accepted_draft_token_ids_by_seq": {},
            "speculative_proposal_row_count": 0,
            "speculative_first_target_callback_count": 0,
            "speculative_fixed_q_group_count": 0,
        }
        return [
            (17, list(range(200, 200 + self.step_index)))
        ], None

    def read_step_logits_authority(self):
        return [[0.0, 2.0, 1.0, -1.0, 0.5, 0.25]]

    def flush_pending_hybrid_state_releases(self, *, timeout_s):
        self.calls.append(("flush", timeout_s))

    def drain_h2d_slot_reuse_diagnostic(
        self,
        *,
        timing_epsilon_ms,
        expected_mode,
        timeout_s,
    ):
        self.calls.append(("drain", expected_mode))
        return tuple(_drain_row(rank) for rank in range(4))

    def kv_offload_summaries(self, *, timeout_s):
        return tuple(
            {
                "h2d_pair_inventory": [[4, 0], [5, 1]],
                "h2d_span_inventory": [[4, 0, 2]],
                "d2h_pair_inventory": [[0, 0]],
                "d2h_span_inventory": [[0, 0, 1]],
            }
            for _ in range(4)
        )


def test_worker_generation_captures_only_indices_zero_and_one_and_cleans():
    worker = _load_module("focused_h2d_worker", WORKER_PATH)
    target_forward_capture = {
        "ordinary_decode_target_forward_calls": 0,
    }
    engine = _FakeGenerationEngine(target_forward_capture)
    result = worker.run_generation_with_h2d_slot_reuse_diagnostic(
        engine=engine,
        prompt_rows=[{"token_ids": [1, 2, 3]}],
        sampling_params=object(),
        synchronize=lambda: None,
        mode="observe",
        batch_size=1,
        repetition=0,
        timing_epsilon_ms=0.2,
        target_forward_capture=target_forward_capture,
    )
    assert [
        row["prediction_index"]
        for row in result["compact_logit_rows"]
    ] == [0, 1]
    assert result["output_rows"][0]["token_ids"] == list(
        range(200, 208)
    )
    assert [
        {
            key: row[key]
            for key in (
                "sequence_id",
                "prompt_index",
                "input_token_id",
                "position",
                "context_length",
            )
        }
        for row in result["compact_logit_rows"]
    ] == [
        {
            "sequence_id": 17,
            "prompt_index": 0,
            "input_token_id": 3,
            "position": 2,
            "context_length": 3,
        },
        {
            "sequence_id": 17,
            "prompt_index": 0,
            "input_token_id": 200,
            "position": 3,
            "context_length": 4,
        },
    ]
    assert result["target_forward_count"] == 8
    assert ("logits", False, 60.0) in engine.calls
    assert ("configure", "off", 60.0) in engine.calls
    assert [
        call[2]
        for call in engine.calls
        if call[0] == "context"
    ] == ["prefill"] + ["decode"] * 7
    assert not any(
        call[0] == "spec_verify_trace"
        for call in engine.calls
    )


def test_worker_cleanup_runs_when_drain_fails_and_preserves_error():
    worker = _load_module("focused_h2d_worker_failure", WORKER_PATH)
    target_forward_capture = {
        "ordinary_decode_target_forward_calls": 0,
    }
    engine = _FakeGenerationEngine(target_forward_capture)

    def fail_drain(**kwargs):
        raise RuntimeError("rank 2 drain failed")

    engine.drain_h2d_slot_reuse_diagnostic = fail_drain
    with pytest.raises(RuntimeError, match="rank 2 drain failed"):
        worker.run_generation_with_h2d_slot_reuse_diagnostic(
            engine=engine,
            prompt_rows=[{"token_ids": [1]}],
            sampling_params=object(),
            synchronize=lambda: None,
            mode="observe",
            batch_size=1,
            repetition=0,
            timing_epsilon_ms=0.2,
            target_forward_capture=target_forward_capture,
        )
    assert ("logits", False, 60.0) in engine.calls
    assert ("configure", "off", 60.0) in engine.calls


def test_ordinary_forward_capture_counts_prefill_and_decode_run_calls():
    worker = _load_module(
        "focused_h2d_worker_forward_capture",
        WORKER_PATH,
    )
    calls = []

    def original_call(method_name, *args, **kwargs):
        calls.append((method_name, args, kwargs))
        return method_name

    runner = types.SimpleNamespace(call=original_call)
    engine = types.SimpleNamespace(model_runner=runner)
    with worker.capture_ordinary_target_forwards(
        engine
    ) as capture:
        assert runner.call("run", (), True) == "run"
        assert runner.call("run", (), False) == "run"
        assert runner.call("memory_snapshot") == "memory_snapshot"
    assert capture["ordinary_decode_target_forward_calls"] == 2
    assert runner.call is original_call


def test_attention_stage_uses_scheduler_prefill_state():
    worker = _load_module(
        "focused_h2d_worker_attention_stage",
        WORKER_PATH,
    )
    scheduler = types.SimpleNamespace(
        waiting=[1],
        prefilling=[],
        running=[],
    )
    engine = types.SimpleNamespace(scheduler=scheduler)
    assert worker._ordinary_attention_stage(engine) == "prefill"
    scheduler.waiting = []
    scheduler.prefilling = [1]
    assert worker._ordinary_attention_stage(engine) == "prefill"
    scheduler.prefilling = []
    scheduler.running = [1]
    assert worker._ordinary_attention_stage(engine) == "decode"


def test_focused_repetition_rejects_nonbaseline_before_worker_load():
    worker = _load_module(
        "focused_h2d_worker_reject_policy",
        WORKER_PATH,
    )
    loaded = []
    with pytest.raises(ValueError, match="baseline"):
        worker.run_focused_repetition(
            model_path="/checkpoint",
            gpu_indices=(0, 1, 2, 3),
            policy="native_mtp",
            mode="observe",
            batch_size=1,
            repetition=0,
            dist_port=1,
            master_port=2,
            frozen_dependencies={},
            torch_module=object(),
            driver_version="driver",
            repo_root=ROOT,
            frozen_worker_loader=lambda: loaded.append(True),
            source_digest_fn=lambda root: "source",
        )
    assert loaded == []


def test_focused_repetition_assembles_frozen_baseline_cell():
    worker = _load_module(
        "focused_h2d_worker_repetition",
        WORKER_PATH,
    )

    class FrozenWorker:
        @staticmethod
        def run_policy_cell(**kwargs):
            assert kwargs["policy"] == "baseline"
            capture = {
                "ordinary_decode_target_forward_calls": 0,
            }
            engine = _FakeGenerationEngine(capture)
            prompt_rows = [{
                "prompt_index": 0,
                "token_count": 3,
                "token_ids": [1, 2, 3],
                "sha256": "prompt",
            }]
            output_rows, observations = kwargs[
                "run_generation_fn"
            ](
                engine=engine,
                prompt_rows=prompt_rows,
                sampling_params=object(),
                synchronize=lambda: None,
            )
            assert len(observations) == 8
            return {
                "policy": "baseline",
                "batch_size": 1,
                "model_identity": {
                    "target_model_manifest_sha256": "checkpoint",
                },
                "engine_config": {},
                "prompt_rows": prompt_rows,
                "output_rows": output_rows,
                "kv_rank_deltas": [
                    {"rank": rank, "h2d_copies": 1}
                    for rank in range(4)
                ],
                "kv_capacity_rows": [
                    {
                        "rank": rank,
                        "peak_resident_blocks": 68,
                    }
                    for rank in range(4)
                ],
                "cleanup": {
                    "rank_exit_codes": [0, 0, 0, 0],
                    "process_group_destroyed": True,
                    "shared_memory_released": True,
                    "owned_children_remaining": [],
                    "engine_exit_called": True,
                },
            }

    runtime_metadata = {
        "torch_version": "2.test",
        "torch_cuda_runtime_version": "12.test",
        "nvidia_driver_version": "999.test",
        "cuda_device_names": ["GPU"] * 4,
    }
    row = worker.run_focused_repetition(
        model_path="/checkpoint",
        gpu_indices=(0, 1, 2, 3),
        policy="baseline",
        mode="observe",
        batch_size=1,
        repetition=3,
        dist_port=1,
        master_port=2,
        frozen_dependencies={
            "engine_factory": object(),
            "sampling_params_type": object(),
            "runtime_type": object(),
            "synchronize": lambda: None,
        },
        torch_module=object(),
        driver_version="driver",
        repo_root=ROOT,
        frozen_worker_loader=lambda: FrozenWorker,
        source_digest_fn=lambda root: "source",
        runtime_metadata_collector=(
            lambda **kwargs: runtime_metadata
        ),
    )
    assert row["schema"] == SCHEMA
    assert row["policy"] == "baseline"
    assert row["mode"] == "observe"
    assert row["repetition"] == 3
    assert row["source_tree_sha256"] == "source"
    assert row["checkpoint_sha256"] == "checkpoint"
    assert row["target_forward_count"] == 8
    assert len(row["compact_logit_rows"]) == 2
    assert row["kv_rank_deltas"][0][
        "h2d_pair_inventory"
    ] == [[4, 0], [5, 1]]
    assert row["kv_rank_deltas"][0][
        "d2h_span_inventory"
    ] == [[0, 0, 1]]
    assert row["cell_digest_sha256"]


def test_focused_campaign_builds_exact_four_cells():
    worker = _load_module(
        "focused_h2d_worker_campaign",
        WORKER_PATH,
    )
    gate, synthetic = _synthetic_campaign()
    calls = []

    def run_repetition(**kwargs):
        key = gate.cell_key(
            kwargs["mode"],
            kwargs["batch_size"],
        )
        repetition = kwargs["repetition"]
        calls.append((key, repetition))
        return copy.deepcopy(
            synthetic["cells"][key][repetition]
        )

    artifact = worker.run_focused_campaign(
        repetitions=2,
        repetition_runner=run_repetition,
        policy="baseline",
    )
    assert tuple(artifact["cells"]) == gate.REQUIRED_CELL_KEYS
    assert calls == [
        (key, repetition)
        for key in gate.REQUIRED_CELL_KEYS
        for repetition in range(2)
    ]


def _synthetic_repetition(
    gate,
    *,
    mode,
    batch_size,
    repetition,
    index1_logit,
    unsafe,
    waited,
    output_token=9,
):
    prompt_rows = [
        {
            "prompt_index": index,
            "sequence_id": index,
            "token_ids": [index, 1, 2],
        }
        for index in range(batch_size)
    ]
    output_rows = [
        {
            "prompt_index": index,
            "sequence_id": index,
            "token_count": 8,
            "token_ids": [output_token] * 8,
            "sha256": gate._json_sha256(
                [output_token] * 8
            ),
        }
        for index in range(batch_size)
    ]
    compact = []
    for prediction_index, logit in ((0, 1.0), (1, index1_logit)):
        compact.append({
            "sequence_id": 0,
            "prompt_index": 0,
            "prediction_index": prediction_index,
            "input_token_id": 10 + prediction_index,
            "position": 32768 + prediction_index,
            "context_length": 32769 + prediction_index,
            "top_tokens": [1, 2, 3, 4, 5],
            "top_logits": [logit, 0.5, 0.4, 0.3, 0.2],
            "top1_margin": logit - 0.5,
            "argmax_token": 1,
        })
    rank_slot_rows = []
    for rank in range(4):
        overwrite_rows = []
        if rank == 0:
            overwrite_rows.append({
                "rank": 0,
                "attention_stage": "decode",
                "physical_slot": 61,
                "old_occupancy_generation": 4,
                "read_event_ordinals": [1],
                "control_wait_event_ordinals": (
                    [1] if waited else []
                ),
                "control_wait_count": 1 if waited else 0,
                "timing_status": (
                    "UNSAFE_OVERLAP_OBSERVED"
                    if unsafe
                    else "READ_COMPLETED_BEFORE_H2D"
                ),
                "read_done_after_h2d_start_ms": (
                    0.5 if unsafe else -0.5
                ),
            })
        rank_slot_rows.append({
            "rank": rank,
            "schema": gate.SCHEMA,
            "mode": mode,
            "stream_inventory": [100 + rank],
            "read_rows": [{
                "rank": rank,
                "attention_stage": "decode",
            }],
            "overwrite_rows": overwrite_rows,
        })
    movement = [
        {
            "rank": rank,
            "h2d_copies": 10,
            "h2d_bytes": 1000,
            "d2h_copies": 2,
            "d2h_bytes": 200,
            "h2d_batches": 5,
            "d2h_batches": 1,
            "h2d_batch_spans": 5,
            "d2h_batch_spans": 1,
            "evictions": 8,
            "peak_resident_blocks": 68,
            "h2d_pair_inventory": [[4, 0], [5, 1]],
            "d2h_pair_inventory": [[0, 0]],
            "h2d_span_inventory": [[4, 0, 2]],
            "d2h_span_inventory": [[0, 0, 1]],
        }
        for rank in range(4)
    ]
    row = {
        "schema": gate.SCHEMA,
        "mode": mode,
        "policy": gate.POLICY,
        "batch_size": batch_size,
        "repetition": repetition,
        "world_size": 4,
        "prompt_tokens": 32768,
        "max_output_tokens": 8,
        "max_proposal_tokens": 4,
        "block_size": 256,
        "gpu_blocks": 68,
        "logical_blocks": 640,
        "blockwise_blocks": 8,
        "async_copy": True,
        "batch_copy": True,
        "writeback_on_evict": False,
        "enforce_eager": True,
        "torch_version": "2.test",
        "torch_cuda_runtime_version": "12.test",
        "nvidia_driver_version": "999.test",
        "cuda_device_names": ["GPU"] * 4,
        "source_tree_sha256": "source",
        "checkpoint_sha256": "checkpoint",
        "timing_epsilon_ms": 0.2,
        "prompt_rows": prompt_rows,
        "output_rows": output_rows,
        "compact_logit_rows": compact,
        "rank_slot_rows": rank_slot_rows,
        "step_observations": [
            {
                "execution_mode": "baseline",
                "proposal_callback_count": 0,
                "shadow_target_forward_count": 0,
                "authority_normal_decode_target_forward_calls": 1,
            }
            for _ in range(8)
        ],
        "target_forward_count": 8,
        "kv_rank_deltas": movement,
        "kv_capacity_rows": [
            {"rank": rank, "peak_resident_blocks": 68}
            for rank in range(4)
        ],
        "cleanup": {
            "rank_exit_codes": [0, 0, 0, 0],
            "process_group_destroyed": True,
            "shared_memory_released": True,
            "owned_children_remaining": [],
            "engine_exit_called": True,
        },
    }
    row["cell_digest_sha256"] = gate._json_sha256(row)
    return row


def _synthetic_campaign(*, control_removes_drift=True, repetitions=2):
    gate = _load_module("focused_h2d_gate_campaign", GATE_PATH)
    cells = {key: [] for key in gate.REQUIRED_CELL_KEYS}
    for repetition in range(repetitions):
        cells["observe:b1"].append(_synthetic_repetition(
            gate,
            mode="observe",
            batch_size=1,
            repetition=repetition,
            index1_logit=1.0,
            unsafe=True,
            waited=False,
        ))
        cells["observe:b4"].append(_synthetic_repetition(
            gate,
            mode="observe",
            batch_size=4,
            repetition=repetition,
            index1_logit=0.8,
            unsafe=True,
            waited=False,
        ))
        cells["control:b1"].append(_synthetic_repetition(
            gate,
            mode="control",
            batch_size=1,
            repetition=repetition,
            index1_logit=1.0,
            unsafe=False,
            waited=True,
        ))
        cells["control:b4"].append(_synthetic_repetition(
            gate,
            mode="control",
            batch_size=4,
            repetition=repetition,
            index1_logit=(
                1.0 if control_removes_drift else 0.8
            ),
            unsafe=False,
            waited=True,
            output_token=(
                9 if control_removes_drift else 8
            ),
        ))
    return gate, {"schema": gate.SCHEMA, "cells": cells}


def _refresh_repetition_digest(gate, row):
    digest_input = dict(row)
    digest_input.pop("cell_digest_sha256", None)
    row["cell_digest_sha256"] = gate._json_sha256(
        digest_input
    )


@pytest.mark.parametrize(
    "mutate",
    (
        lambda row: row["output_rows"][0].update({
            "token_count": 7,
            "token_ids": [9] * 7,
            "sha256": "wrong",
        }),
        lambda row: row["step_observations"][0].update({
            "target_forward_trace_rows": [{"stage": "verify_tail"}],
        }),
        lambda row: row["step_observations"][0].update({
            "side_state_lineage_rows": [{"event": "apply"}],
        }),
        lambda row: row["step_observations"][0].update({
            "authority_normal_decode_target_forward_calls": 2,
        }),
    ),
)
def test_tampered_repetition_is_inconclusive(mutate):
    gate, artifact = _synthetic_campaign()
    tampered = copy.deepcopy(artifact)
    row = tampered["cells"]["control:b4"][0]
    mutate(row)
    _refresh_repetition_digest(gate, row)
    result = gate.evaluate_campaign(tampered)
    assert result["classification"].endswith("=INCONCLUSIVE")
    assert result["inconclusive"] is True


@pytest.mark.parametrize(
    "mutate",
    (
        lambda row: row["prompt_rows"].reverse(),
        lambda row: row.update({
            "source_tree_sha256": "different-source",
        }),
        lambda row: row.update({
            "checkpoint_sha256": "different-checkpoint",
        }),
        lambda row: row.update({"async_copy": False}),
        lambda row: row.update({"world_size": 3}),
        lambda row: row["kv_rank_deltas"][0].update({
            "h2d_copies": 11,
        }),
        lambda row: row["kv_rank_deltas"][0].update({
            "h2d_bytes": 1001,
        }),
        lambda row: row["kv_rank_deltas"][0].update({
            "h2d_batches": 6,
        }),
        lambda row: row["kv_rank_deltas"][0].update({
            "h2d_batch_spans": 6,
        }),
        lambda row: row["kv_rank_deltas"][0].update({
            "d2h_copies": 3,
        }),
        lambda row: row["kv_rank_deltas"][0].update({
            "d2h_bytes": 201,
        }),
        lambda row: row["kv_rank_deltas"][0].update({
            "d2h_batches": 2,
        }),
        lambda row: row["kv_rank_deltas"][0].update({
            "d2h_batch_spans": 2,
        }),
        lambda row: row["kv_rank_deltas"][0].update({
            "evictions": 9,
        }),
        lambda row: row["kv_rank_deltas"][0][
            "h2d_pair_inventory"
        ].append([6, 2]),
        lambda row: row["kv_rank_deltas"][0][
            "d2h_span_inventory"
        ].append([1, 1, 1]),
        lambda row: row["kv_capacity_rows"][0].update({
            "peak_resident_blocks": 67,
        }),
        lambda row: row["cleanup"].update({
            "process_group_destroyed": False,
        }),
        lambda row: row.update({"torch_version": ""}),
        lambda row: row.update({"cuda_device_names": ["GPU"] * 3}),
        lambda row: row["rank_slot_rows"][0]["read_rows"][0].update({
            "attention_stage": "spec_verify",
        }),
        lambda row: row["step_observations"][0].update({
            "proposal_callback_count": 1,
        }),
        lambda row: row["step_observations"][0].update({
            "shadow_target_forward_count": 1,
        }),
    ),
)
def test_invariant_or_authority_mutation_is_inconclusive(mutate):
    gate, artifact = _synthetic_campaign()
    row = artifact["cells"]["control:b4"][0]
    mutate(row)
    _refresh_repetition_digest(gate, row)
    result = gate.evaluate_campaign(artifact)
    assert result["classification"].endswith("=INCONCLUSIVE")


def test_decision_matrix_supported_is_exclusive():
    gate, artifact = _synthetic_campaign()
    result = gate.evaluate_campaign(artifact)
    assert result["classification"].endswith("=SUPPORTED")
    assert (
        result["supported"],
        result["rejected"],
        result["inconclusive"],
    ) == (True, False, False)


def test_cross_batch_prompt_zero_identity_is_required():
    gate, artifact = _synthetic_campaign()
    for key in ("observe:b4", "control:b4"):
        for row in artifact["cells"][key]:
            row["prompt_rows"][0]["token_ids"] = [99, 1, 2]
            _refresh_repetition_digest(gate, row)
    result = gate.evaluate_campaign(artifact)
    assert result["classification"].endswith("=INCONCLUSIVE")
    assert "cross-batch prompt-0 identity differs" in result["reasons"]


def test_cross_batch_prediction_identity_is_required():
    gate, artifact = _synthetic_campaign()
    for key in ("observe:b4", "control:b4"):
        for row in artifact["cells"][key]:
            prediction = next(
                logit_row
                for logit_row in row["compact_logit_rows"]
                if logit_row["prediction_index"] == 1
            )
            prediction["position"] += 1
            prediction["context_length"] += 1
            _refresh_repetition_digest(gate, row)
    result = gate.evaluate_campaign(artifact)
    assert result["classification"].endswith("=INCONCLUSIVE")
    assert "cross-batch prediction identity differs" in result["reasons"]


def test_decision_matrix_rejected_is_exclusive():
    gate, artifact = _synthetic_campaign(
        control_removes_drift=False
    )
    result = gate.evaluate_campaign(artifact)
    assert result["classification"].endswith("=REJECTED")
    assert (
        result["supported"],
        result["rejected"],
        result["inconclusive"],
    ) == (False, True, False)


def test_single_repetition_is_inconclusive():
    gate, artifact = _synthetic_campaign(repetitions=1)
    result = gate.evaluate_campaign(artifact)
    assert result["classification"].endswith("=INCONCLUSIVE")
    assert (
        result["supported"],
        result["rejected"],
        result["inconclusive"],
    ) == (False, False, True)


def test_movement_invariant_difference_is_inconclusive():
    gate, artifact = _synthetic_campaign()
    row = artifact["cells"]["control:b4"][0]
    row["kv_rank_deltas"][0]["h2d_bytes"] += 1
    digest_input = dict(row)
    digest_input.pop("cell_digest_sha256")
    row["cell_digest_sha256"] = gate._json_sha256(
        digest_input
    )
    result = gate.evaluate_campaign(artifact)
    assert result["classification"].endswith("=INCONCLUSIVE")


def test_missing_prediction_index_one_is_inconclusive():
    gate, artifact = _synthetic_campaign()
    row = artifact["cells"]["control:b4"][0]
    row["compact_logit_rows"] = [
        logit_row
        for logit_row in row["compact_logit_rows"]
        if logit_row["prediction_index"] != 1
    ]
    _refresh_repetition_digest(gate, row)
    result = gate.evaluate_campaign(artifact)
    assert result["classification"].endswith("=INCONCLUSIVE")


def test_complete_observe_without_unsafe_overlap_is_rejected():
    gate, artifact = _synthetic_campaign()
    for key in ("observe:b1", "observe:b4"):
        for row in artifact["cells"][key]:
            overwrite = row["rank_slot_rows"][0][
                "overwrite_rows"
            ][0]
            overwrite["timing_status"] = (
                "READ_COMPLETED_BEFORE_H2D"
            )
            overwrite["read_done_after_h2d_start_ms"] = -0.5
            _refresh_repetition_digest(gate, row)
    result = gate.evaluate_campaign(artifact)
    assert result["classification"].endswith("=REJECTED")


def test_ambiguous_timing_is_inconclusive():
    gate, artifact = _synthetic_campaign()
    row = artifact["cells"]["observe:b4"][0]
    overwrite = row["rank_slot_rows"][0]["overwrite_rows"][0]
    overwrite["timing_status"] = "ORDERING_AMBIGUOUS"
    overwrite["read_done_after_h2d_start_ms"] = 0.0
    _refresh_repetition_digest(gate, row)
    result = gate.evaluate_campaign(artifact)
    assert result["classification"].endswith("=INCONCLUSIVE")


def test_missing_control_predecessor_wait_is_inconclusive():
    gate, artifact = _synthetic_campaign()
    row = artifact["cells"]["control:b4"][0]
    overwrite = row["rank_slot_rows"][0]["overwrite_rows"][0]
    overwrite["control_wait_event_ordinals"] = []
    overwrite["control_wait_count"] = 0
    _refresh_repetition_digest(gate, row)
    result = gate.evaluate_campaign(artifact)
    assert result["classification"].endswith("=INCONCLUSIVE")


def test_incomplete_rank_stream_lifecycle_is_inconclusive():
    gate, artifact = _synthetic_campaign()
    row = artifact["cells"]["observe:b4"][0]
    row["rank_slot_rows"][2]["stream_inventory"] = []
    _refresh_repetition_digest(gate, row)
    result = gate.evaluate_campaign(artifact)
    assert result["classification"].endswith("=INCONCLUSIVE")


def test_repetition_accepts_ordinary_prefill_slot_rows():
    gate, artifact = _synthetic_campaign()
    row = artifact["cells"]["observe:b1"][0]
    row["rank_slot_rows"][0]["read_rows"][0][
        "attention_stage"
    ] = "prefill"
    _refresh_repetition_digest(gate, row)
    assert gate.validate_repetition(row)["mode"] == "observe"


def _write_verifier_artifact(tmp_path, artifact):
    path = tmp_path / "artifact.json"
    path.write_text(
        __import__("json").dumps(artifact),
        encoding="utf-8",
    )
    return path


def _set_campaign_identities(gate, artifact, source, checkpoint):
    for repetitions in artifact["cells"].values():
        for row in repetitions:
            row["source_tree_sha256"] = source
            row["checkpoint_sha256"] = checkpoint
            _refresh_repetition_digest(gate, row)


def test_verifier_accepts_completed_inconclusive_campaign(tmp_path):
    verifier = _load_module("focused_h2d_verifier", VERIFIER_PATH)
    gate, artifact = _synthetic_campaign(repetitions=1)
    _set_campaign_identities(
        gate,
        artifact,
        "live-source",
        "live-checkpoint",
    )
    _write_verifier_artifact(tmp_path, artifact)
    result = verifier.verify_run(
        run_dir=tmp_path,
        repo_root=ROOT,
        model_path="/checkpoint",
        source_digest_fn=lambda root: "live-source",
        checkpoint_digest_fn=lambda model: "live-checkpoint",
    )
    assert result["classification"] == "PASS"
    assert result["decision"]["inconclusive"] is True


def test_verifier_fails_closed_on_tampered_artifact(tmp_path):
    verifier = _load_module(
        "focused_h2d_verifier_tamper",
        VERIFIER_PATH,
    )
    gate, artifact = _synthetic_campaign()
    _set_campaign_identities(
        gate,
        artifact,
        "live-source",
        "live-checkpoint",
    )
    artifact["cells"]["control:b4"][0]["target_forward_count"] = 9
    _write_verifier_artifact(tmp_path, artifact)
    result = verifier.verify_run(
        run_dir=tmp_path,
        repo_root=ROOT,
        model_path="/checkpoint",
        source_digest_fn=lambda root: "live-source",
        checkpoint_digest_fn=lambda model: "live-checkpoint",
    )
    assert result["classification"] == "FAIL"
    assert result["decision"] is None
