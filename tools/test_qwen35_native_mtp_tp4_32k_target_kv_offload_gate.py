from __future__ import annotations

from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest
import torch


TOOLS = Path(__file__).resolve().parent
GATE_PATH = (
    TOOLS
    / "qwen35_native_mtp_tp4_32k_target_kv_offload_gate.py"
)
FROZEN_GATE_PATH = (
    TOOLS
    / "qwen35_native_mtp_tp4_16k_target_kv_offload_gate.py"
)
VERIFIER_PATH = (
    TOOLS
    / (
        "verify_qwen35_native_mtp_tp4_32k_"
        "target_kv_offload_gate.py"
    )
)
REMOTE_RUNNER_PATH = (
    TOOLS
    / (
        "run_qwen35_native_mtp_tp4_32k_"
        "target_kv_offload_remote.sh"
    )
)
WORKER_PATH = (
    TOOLS
    / (
        "qwen35_native_mtp_tp4_32k_"
        "target_kv_offload_worker.py"
    )
)


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


gate = _load_module(
    "qwen35_native_mtp_tp4_32k_target_kv_offload_gate",
    GATE_PATH,
)
frozen_test = _load_module(
    "qwen35_native_mtp_tp4_32k_fixtures",
    TOOLS
    / "test_qwen35_native_mtp_tp4_16k_target_kv_offload_gate.py",
)
worker = _load_module(
    "qwen35_native_mtp_tp4_32k_trace_worker",
    WORKER_PATH,
)


def _trace_row(
    *,
    policy,
    sequence_id=3,
    prompt_index=0,
):
    return {
        "schema": (
            "qwen35.native-mtp-tp4-32k-"
            "paired-verify-trace.v1"
        ),
        "policy": policy,
        "batch_size": 1,
        "engine_step": 4,
        "target_forward_ordinal": 2,
        "stage": "verify_tail",
        "execution_mode": "spec_verify",
        "sequence_id": sequence_id,
        "prompt_index": prompt_index,
        "query_offset": 0,
        "query_len": 1,
        "row_index": 0,
        "prediction_index": 3,
        "input_token_id": 2658,
        "position": 32770,
        "context_length": 32771,
        "logical_block_identities": [[127, 1]],
        "logical_block_coverage": [[0, 32512, 32768]],
        "top_tokens": [15, 220, 2658, 8381, 7],
        "top_logits": [9.0, 8.0, 7.0, 6.0, 5.0],
        "top1_margin": 1.0,
        "argmax_token": 15,
    }


def test_pair_trace_rows_uses_prompt_index_not_sequence_id():
    baseline = [_trace_row(
        policy="baseline",
        sequence_id=3,
        prompt_index=0,
    )]
    native = [_trace_row(
        policy="native_mtp",
        sequence_id=91,
        prompt_index=0,
    )]

    paired = worker.pair_target_forward_rows(
        baseline,
        native,
    )

    assert len(paired) == 1
    assert paired[0]["prompt_index"] == 0
    assert paired[0]["baseline_argmax_token"] == 15
    assert paired[0]["native_argmax_token"] == 15


def test_pair_trace_rows_rejects_missing_or_duplicate_matches():
    baseline = [_trace_row(policy="baseline")]
    native = [_trace_row(policy="native_mtp")]

    with pytest.raises(ValueError, match="duplicate native"):
        worker.pair_target_forward_rows(
            baseline,
            native + deepcopy(native),
        )

    with pytest.raises(ValueError, match="missing native"):
        worker.pair_target_forward_rows(baseline, [])


def test_pair_trace_rows_requires_exact_fields():
    baseline = _trace_row(policy="baseline")
    native = _trace_row(policy="native_mtp")
    missing = deepcopy(baseline)
    missing.pop("top1_margin")
    unknown = deepcopy(native)
    unknown["unknown"] = True

    with pytest.raises(ValueError, match="fields mismatch"):
        worker.pair_target_forward_rows([missing], [native])
    with pytest.raises(ValueError, match="fields mismatch"):
        worker.pair_target_forward_rows([baseline], [unknown])


def _side_trace_rows():
    return [
        {
            "sequence_id": 7,
            "event": "first_target_checkpoint",
            "checkpoint_index": 1,
            "committed_input_count": None,
            "fingerprint": "a" * 64,
            "engine_step": 0,
        },
        *[
            {
                "sequence_id": 7,
                "event": "tail_checkpoint",
                "checkpoint_index": checkpoint_index,
                "committed_input_count": None,
                "fingerprint": str(checkpoint_index) * 64,
                "engine_step": 1,
            }
            for checkpoint_index in (2, 3, 4)
        ],
        {
            "sequence_id": 7,
            "event": "selected_checkpoint",
            "checkpoint_index": 3,
            "committed_input_count": 3,
            "fingerprint": "f" * 64,
            "engine_step": 1,
        },
    ]


def _side_observations():
    return [
        {
            "speculative_proposal_token_ids_by_seq": {},
            "speculative_accepted_draft_token_ids_by_seq": {},
            "new_completion_tokens_by_seq": {},
        },
        {
            "speculative_proposal_token_ids_by_seq": {
                7: [15, 15, 2658, 8381],
            },
            "speculative_accepted_draft_token_ids_by_seq": {
                7: [15, 15],
            },
            "new_completion_tokens_by_seq": {
                7: [15, 15, 220],
            },
        },
    ]


def test_assemble_side_state_lineage_enriches_selection():
    lineage = worker.assemble_side_state_lineage(
        policy="native_mtp",
        batch_size=1,
        trace_rows=_side_trace_rows(),
        observations=_side_observations(),
        sequence_to_prompt={7: 0},
    )

    selected = lineage[-1]
    assert selected["committed_input_count"] == 3
    assert selected["checkpoint_index"] == 3
    assert selected["proposal_token_ids"] == [
        15,
        15,
        2658,
        8381,
    ]
    assert selected["accepted_token_ids"] == [15, 15]
    assert selected["verify_input_count"] == 3
    assert selected["fallback_target_token"] == 220


def test_assemble_side_state_lineage_rejects_invariant_drift():
    rows = _side_trace_rows()
    observations = _side_observations()
    mismatched_checkpoint = deepcopy(rows)
    mismatched_checkpoint[-1]["checkpoint_index"] = 2
    with pytest.raises(ValueError, match="committed input"):
        worker.assemble_side_state_lineage(
            policy="native_mtp",
            batch_size=1,
            trace_rows=mismatched_checkpoint,
            observations=observations,
            sequence_to_prompt={7: 0},
        )

    non_prefix = deepcopy(observations)
    non_prefix[1][
        "speculative_accepted_draft_token_ids_by_seq"
    ][7] = [15, 16]
    with pytest.raises(ValueError, match="proposal prefix"):
        worker.assemble_side_state_lineage(
            policy="native_mtp",
            batch_size=1,
            trace_rows=rows,
            observations=non_prefix,
            sequence_to_prompt={7: 0},
        )

    missing_fallback = deepcopy(observations)
    missing_fallback[1]["new_completion_tokens_by_seq"][7] = [
        15,
        15,
    ]
    with pytest.raises(ValueError, match="fallback"):
        worker.assemble_side_state_lineage(
            policy="native_mtp",
            batch_size=1,
            trace_rows=rows,
            observations=missing_fallback,
            sequence_to_prompt={7: 0},
        )


class _TraceOwner:
    def __init__(self, events):
        self.events = events
        self.enabled = []

    def enable_trace_recording(self, enabled):
        self.enabled.append(enabled)
        self.events.append(("owner_enable", enabled))
        return {"enabled": enabled}

    def drain_trace_rows(self):
        self.events.append(("owner_drain",))
        return (
            {
                "sequence_id": 9,
                "event": "first_target_checkpoint",
                "checkpoint_index": 1,
                "committed_input_count": None,
                "fingerprint": "a" * 64,
            },
        )


class _TraceRunner:
    def __init__(self, events, *, with_owner=True):
        self.events = events
        self.enabled = []
        self.contexts = []
        self.qwen35_speculative_state_owner = (
            _TraceOwner(events) if with_owner else None
        )

    def enable_spec_verify_trace_recording(self, enabled):
        self.enabled.append(enabled)
        self.events.append(("runner_enable", enabled))
        return {"rank": 0, "enabled": enabled}

    def set_spec_verify_trace_context(
        self,
        policy,
        batch_size,
        engine_step,
    ):
        context = (policy, batch_size, engine_step)
        self.contexts.append(context)
        self.events.append(("context",) + context)
        return {
            "rank": 0,
            "policy": policy,
            "batch_size": batch_size,
            "engine_step": engine_step,
        }

    def drain_spec_verify_trace_rows(self):
        self.events.append(("runner_drain",))
        policy, batch_size, engine_step = self.contexts[-1]
        row = _trace_row(
            policy=policy,
            sequence_id=9,
        )
        row.pop("prompt_index")
        row["batch_size"] = batch_size
        row["engine_step"] = engine_step
        return (row,)


class _TraceEngine:
    def __init__(self, *, fail=False, with_owner=True):
        self.events = []
        self.model_runner = _TraceRunner(
            self.events,
            with_owner=with_owner,
        )
        self.last_step_observation = None
        self.requests = []
        self.finished = False
        self.fail = fail

    def add_request(self, token_ids, sampling_params):
        self.requests.append((list(token_ids), sampling_params))

    def is_finished(self):
        return self.finished

    def step(self):
        self.events.append(("step",))
        if self.fail:
            raise RuntimeError("generation failed")
        self.finished = True
        self.last_step_observation = {
            "speculative_proposal_token_ids_by_seq": {},
            "speculative_accepted_draft_token_ids_by_seq": {},
            "new_completion_tokens_by_seq": {
                9: list(range(gate.MAX_OUTPUT_TOKENS)),
            },
        }
        return (
            [
                (
                    9,
                    list(range(gate.MAX_OUTPUT_TOKENS)),
                ),
            ],
            1,
        )


def test_paired_trace_generation_orders_lifecycle_and_drains_after_sync():
    engine = _TraceEngine()
    trace_capture = {}

    def synchronize():
        engine.events.append(("sync",))

    output_rows, observations = (
        worker.run_generation_with_paired_trace(
            engine=engine,
            prompt_rows=[{"token_ids": [11]}],
            sampling_params=object(),
            synchronize=synchronize,
            policy="native_mtp",
            batch_size=1,
            trace_capture=trace_capture,
        )
    )

    assert engine.model_runner.enabled == [True, False]
    assert (
        engine.model_runner.qwen35_speculative_state_owner.enabled
        == [True, False]
    )
    assert engine.model_runner.contexts == [
        ("native_mtp", 1, 0),
    ]
    assert engine.events.index(("sync",)) < engine.events.index(
        ("runner_drain",)
    )
    assert engine.events.index(("sync",)) < engine.events.index(
        ("owner_drain",)
    )
    assert output_rows[0]["token_ids"] == list(
        range(gate.MAX_OUTPUT_TOKENS)
    )
    assert len(observations) == 1
    assert trace_capture["target_forward_trace_rows"][0][
        "prompt_index"
    ] == 0
    assert trace_capture["raw_side_state_rows"][0][
        "engine_step"
    ] == 0


def test_paired_trace_generation_disables_recorders_on_failure():
    engine = _TraceEngine(fail=True)
    trace_capture = {"stale": True}

    with pytest.raises(RuntimeError, match="generation failed"):
        worker.run_generation_with_paired_trace(
            engine=engine,
            prompt_rows=[{"token_ids": [11]}],
            sampling_params=object(),
            synchronize=lambda: None,
            policy="native_mtp",
            batch_size=1,
            trace_capture=trace_capture,
        )

    assert engine.model_runner.enabled == [True, False]
    assert (
        engine.model_runner.qwen35_speculative_state_owner.enabled
        == [True, False]
    )


def test_paired_trace_generation_preserves_plain_generation_results():
    traced_engine = _TraceEngine(with_owner=False)
    plain_engine = _TraceEngine(with_owner=False)
    traced_engine.kv_movement = {
        "h2d_copies": 3,
        "d2h_copies": 2,
    }
    plain_engine.kv_movement = deepcopy(
        traced_engine.kv_movement
    )
    traced_engine.cleanup_inventory = {
        "owned_children_remaining": [],
    }
    plain_engine.cleanup_inventory = deepcopy(
        traced_engine.cleanup_inventory
    )

    traced_output, traced_observations = (
        worker.run_generation_with_paired_trace(
            engine=traced_engine,
            prompt_rows=[{"token_ids": [11]}],
            sampling_params=object(),
            synchronize=lambda: None,
            policy="baseline",
            batch_size=1,
            trace_capture={},
        )
    )
    for row in [{"token_ids": [11]}]:
        plain_engine.add_request(
            row["token_ids"],
            object(),
        )
    plain_outputs = {}
    plain_observations = []
    while not plain_engine.is_finished():
        step_outputs, _ = plain_engine.step()
        plain_observations.append(
            dict(plain_engine.last_step_observation)
        )
        for sequence_id, token_ids in step_outputs:
            plain_outputs[int(sequence_id)] = list(token_ids)
    plain_output = [
        {
            "prompt_index": prompt_index,
            "token_count": len(plain_outputs[sequence_id]),
            "token_ids": plain_outputs[sequence_id],
            "sha256": gate._json_sha256(
                plain_outputs[sequence_id]
            ),
        }
        for prompt_index, sequence_id in enumerate(
            sorted(plain_outputs)
        )
    ]

    assert traced_output == plain_output
    assert traced_observations == plain_observations
    assert traced_engine.events.count(("step",)) == (
        plain_engine.events.count(("step",))
    )
    assert traced_engine.kv_movement == plain_engine.kv_movement
    assert (
        traced_engine.cleanup_inventory
        == plain_engine.cleanup_inventory
    )


def _diagnostic_cell(
    policy,
    batch_size,
    *,
    divergent=False,
):
    row = _trace_row(
        policy=policy,
        sequence_id=3 if policy == "baseline" else 91,
    )
    row["batch_size"] = batch_size
    if divergent:
        row["top_logits"][0] = 8.5
    cell = {
        "policy": policy,
        "batch_size": batch_size,
        "output_rows": [],
        "target_forward_trace_rows": [row],
        "side_state_lineage_rows": [],
        "step_observations": [],
        "rank_cleanup_summary": {"status": "clean"},
    }
    cell["cell_digest_sha256"] = gate._json_sha256(cell)
    return cell


def _diagnostic_cells():
    return {
        "baseline:b1": _diagnostic_cell("baseline", 1),
        "native_mtp:b1": _diagnostic_cell(
            "native_mtp",
            1,
            divergent=True,
        ),
        "baseline:b4": _diagnostic_cell("baseline", 4),
        "native_mtp:b4": _diagnostic_cell(
            "native_mtp",
            4,
        ),
    }


def test_build_paired_trace_artifact_selects_first_divergence():
    artifact = worker.build_paired_trace_artifact(
        cells=_diagnostic_cells(),
        source_manifest_sha256="a" * 64,
        target_manifest_sha256=(
            gate.TARGET_MODEL_MANIFEST_SHA256
        ),
        mtp_manifest_sha256=(
            gate.MTP_CHECKPOINT_MANIFEST_SHA256
        ),
    )

    assert artifact["schema"] == worker.TRACE_SCHEMA
    assert artifact["first_divergence"]["prompt_index"] == 0
    assert artifact["first_divergence"]["prediction_index"] == 3
    assert artifact["limitations"] == list(
        worker.TRACE_LIMITATIONS
    )


def test_build_paired_trace_artifact_rejects_cells_and_tensors():
    missing = _diagnostic_cells()
    missing.pop("native_mtp:b4")
    with pytest.raises(ValueError, match="cell keys"):
        worker.build_paired_trace_artifact(
            cells=missing,
            source_manifest_sha256="a" * 64,
            target_manifest_sha256=(
                gate.TARGET_MODEL_MANIFEST_SHA256
            ),
            mtp_manifest_sha256=(
                gate.MTP_CHECKPOINT_MANIFEST_SHA256
            ),
        )

    digest_drift = _diagnostic_cells()
    digest_drift["baseline:b1"]["batch_size"] = 2
    with pytest.raises(ValueError, match="cell digest"):
        worker.build_paired_trace_artifact(
            cells=digest_drift,
            source_manifest_sha256="a" * 64,
            target_manifest_sha256=(
                gate.TARGET_MODEL_MANIFEST_SHA256
            ),
            mtp_manifest_sha256=(
                gate.MTP_CHECKPOINT_MANIFEST_SHA256
            ),
        )

    tensor_cells = _diagnostic_cells()
    tensor_cells["baseline:b1"]["step_observations"] = [
        {"tensor": torch.tensor([1.0])},
    ]
    with pytest.raises(
        ValueError,
        match="trace artifact contains a tensor",
    ):
        worker.build_paired_trace_artifact(
            cells=tensor_cells,
            source_manifest_sha256="a" * 64,
            target_manifest_sha256=(
                gate.TARGET_MODEL_MANIFEST_SHA256
            ),
            mtp_manifest_sha256=(
                gate.MTP_CHECKPOINT_MANIFEST_SHA256
            ),
        )


def test_run_paired_trace_cell_wraps_frozen_cell_without_mutation():
    engine = _TraceEngine()
    inherited_cell = {
        "policy": "native_mtp",
        "batch_size": 1,
        "output_rows": [],
        "cleanup": {"status": "clean"},
    }
    inherited_before = deepcopy(inherited_cell)
    captured = {}
    original = worker.run_policy_cell

    def fake_run_policy_cell(**kwargs):
        captured.update(kwargs)
        output_rows, _ = kwargs["run_generation_fn"](
            engine=engine,
            prompt_rows=[{"token_ids": [11]}],
            sampling_params=object(),
            synchronize=lambda: None,
            target_forward_capture={
                "ordinary_decode_target_forward_calls": 0,
            },
        )
        inherited_cell["output_rows"] = output_rows
        return inherited_cell

    worker.run_policy_cell = fake_run_policy_cell
    try:
        cell = worker.run_paired_trace_cell(
            model_path="/model",
            gpu_indices=(0, 1, 2, 3),
            policy="native_mtp",
            batch_size=1,
            dist_port=29500,
            master_port=29600,
            engine_factory=object(),
            sampling_params_type=object(),
            runtime_type=object(),
            synchronize=lambda: None,
        )
    finally:
        worker.run_policy_cell = original

    assert inherited_before == {
        key: value
        for key, value in inherited_cell.items()
        if key != "output_rows"
    } | {"output_rows": []}
    assert captured["model_path"] == "/model"
    assert cell["policy"] == "native_mtp"
    assert cell["batch_size"] == 1
    assert cell["target_forward_trace_rows"]
    assert cell["side_state_lineage_rows"][0]["event"] == (
        "first_target_checkpoint"
    )
    payload = {
        key: value
        for key, value in cell.items()
        if key != "cell_digest_sha256"
    }
    assert cell["cell_digest_sha256"] == gate._json_sha256(
        payload
    )


def test_run_paired_trace_diagnostic_runs_exact_four_cells(
    tmp_path,
):
    calls = []
    cells = _diagnostic_cells()

    def run_cell_fn(**kwargs):
        calls.append(kwargs["cell_key"])
        return deepcopy(cells[kwargs["cell_key"]])

    original_source = worker.gate.source_tree_sha256
    worker.gate.source_tree_sha256 = (
        lambda repo_root, source_files: "a" * 64
    )
    output_path = tmp_path / "diagnostic" / "trace.json"
    try:
        artifact = worker.run_paired_trace_diagnostic(
            output_path=output_path,
            repo_root=tmp_path,
            cell_kwargs_by_key={
                key: {"cell_key": key}
                for key in (
                    "baseline:b1",
                    "native_mtp:b1",
                    "baseline:b4",
                    "native_mtp:b4",
                )
            },
            run_cell_fn=run_cell_fn,
        )
    finally:
        worker.gate.source_tree_sha256 = original_source

    assert calls == [
        "baseline:b1",
        "native_mtp:b1",
        "baseline:b4",
        "native_mtp:b4",
    ]
    assert output_path.exists()
    assert json.loads(
        output_path.read_text(encoding="utf-8")
    ) == artifact


def _valid_result() -> dict:
    result = deepcopy(frozen_test._result())
    result["schema_version"] = gate.SCHEMA_VERSION
    result["classification"] = gate.CLASSIFICATION
    result["limitations"] = list(gate.REQUIRED_LIMITATIONS)
    for cell in result["cells"].values():
        cell["schema_version"] = gate.SCHEMA_VERSION
        cell["prompt_token_count"] = gate.PROMPT_TOKENS
        cell["prompt_rows"] = frozen_test._token_rows(
            cell["batch_size"],
            token_count=gate.PROMPT_TOKENS,
            start=17,
        )
    for batch_size in gate.BATCH_SIZES:
        cell = result["cells"][f"native_mtp:b{batch_size}"]
        cell["kv_rank_deltas"] = frozen_test._movement_rows(
            positive=True
        )
        for rank_snapshot in cell["rank_snapshots"]:
            executor = rank_snapshot["executor"]
            proposal_transactions_by_id = {
                row["transaction_id"]: row
                for row in executor["proposal_transactions"]
            }
            committed_lengths = {
                sequence_id: 0
                for sequence_id in range(batch_size)
            }
            for transaction in executor[
                "proposal_kv_cache"
            ]["transactions"]:
                proposal = proposal_transactions_by_id.get(
                    transaction["transaction_id"]
                )
                sequence_id = transaction["sequence_id"]
                if proposal is None:
                    transaction["staged_entry_count"] = (
                        gate.PROMPT_TOKENS
                    )
                    transaction["materialized_entry_count"] = (
                        gate.PROMPT_TOKENS
                    )
                    committed_lengths[sequence_id] = (
                        gate.PROMPT_TOKENS
                    )
                else:
                    transaction["original_committed_length"] = (
                        committed_lengths[sequence_id]
                    )
                    committed_lengths[sequence_id] += max(
                        proposal["accepted_proposal_tokens"] - 1,
                        0,
                    )
            for ticket in executor[
                "proposal_kv_cache"
            ]["tickets"]:
                if (
                    ticket["transaction_id"]
                    not in proposal_transactions_by_id
                ):
                    ticket["commit_entry_count"] = (
                        gate.PROMPT_TOKENS
                    )
    return result


def test_contract_constants_are_frozen():
    assert gate.SCHEMA_VERSION == (
        "qwen35.native-mtp-tp4-32k-target-kv-offload.v1"
    )
    assert gate.CLASSIFICATION == (
        "QWEN35_NATIVE_MTP_TP4_32K_TARGET_KV_OFFLOAD_ESTABLISHED"
    )
    assert gate.PROMOTION_CLASSIFICATION == "NOT_PROMOTABLE"
    assert gate.PROMPT_TOKENS == 32768
    assert gate.MAX_OUTPUT_TOKENS == 8
    assert gate.MAX_PROPOSAL_TOKENS == 4
    assert gate.WORLD_SIZE == 4
    assert gate.BATCH_SIZES == (1, 4)
    assert gate.POLICIES == ("baseline", "native_mtp")
    assert gate.MAX_MODEL_LEN == 33024
    assert gate.MAX_NUM_BATCHED_TOKENS == 132096
    assert gate.MAX_NUM_PREFILL_TOKENS_PER_STEP == 1024
    assert gate.KV_OFFLOAD_GPU_BLOCKS == 68
    assert gate.KV_OFFLOAD_LOGICAL_BLOCKS == 640
    assert gate.KV_OFFLOAD_BLOCKWISE_BLOCKS == 8
    assert gate.BLOCK_SIZE == 256
    assert gate.REQUIRED_LIMITATIONS == (
        "phase1_not_promotable",
        "proposal_kv_offload_not_established",
        "tp1_32k_not_established",
        "performance_not_established",
        "kv_quantization_not_established",
        "second_learned_structure_not_established",
    )
    for source in (
        (
            "tools/qwen35_native_mtp_tp4_16k_"
            "target_kv_offload_gate.py"
        ),
        (
            "tools/qwen35_native_mtp_tp4_16k_"
            "target_kv_offload_worker.py"
        ),
        (
            "tools/verify_qwen35_native_mtp_tp4_16k_"
            "target_kv_offload_gate.py"
        ),
        (
            "tools/qwen35_native_mtp_tp4_32k_"
            "target_kv_offload_gate.py"
        ),
        (
            "tools/qwen35_native_mtp_tp4_32k_"
            "target_kv_offload_worker.py"
        ),
        (
            "tools/verify_qwen35_native_mtp_tp4_32k_"
            "target_kv_offload_gate.py"
        ),
        "tinyvllm/engine/spec_verify_trace.py",
        "tinyvllm/engine/qwen35_speculative_trace.py",
    ):
        assert source in gate.DEFAULT_SOURCE_FILES


def test_trace_sources_do_not_change_authority_schema():
    assert gate.SCHEMA_VERSION == (
        "qwen35.native-mtp-tp4-32k-target-kv-offload.v1"
    )
    assert "paired-verify-trace" not in gate.SCHEMA_VERSION
    assert gate.validate_result(_valid_result()) == _valid_result()


def test_trace_activation_is_explicit_and_helpers_add_no_forward():
    repo_root = TOOLS.parent
    diagnostic_worker = WORKER_PATH.read_text(encoding="utf-8")
    assert "enable_spec_verify_trace_recording(True)" in (
        diagnostic_worker
    )
    for relative_path in (
        "tinyvllm/engine/llm_engine.py",
        "tinyvllm/engine/scheduler.py",
        (
            "tools/qwen35_native_mtp_tp4_16k_"
            "target_kv_offload_worker.py"
        ),
    ):
        assert "enable_spec_verify_trace_recording(True)" not in (
            (repo_root / relative_path).read_text(
                encoding="utf-8"
            )
        )
    fingerprint_helper = (
        repo_root
        / "tinyvllm/engine/qwen35_speculative_trace.py"
    ).read_text(encoding="utf-8")
    assert "run_model(" not in fingerprint_helper
    assert "engine.step(" not in fingerprint_helper


def test_loading_32k_gate_does_not_modify_frozen_gate_source():
    before = hashlib.sha256(
        FROZEN_GATE_PATH.read_bytes()
    ).hexdigest()
    _load_module(
        "qwen35_native_mtp_tp4_32k_gate_isolation",
        GATE_PATH,
    )
    after = hashlib.sha256(
        FROZEN_GATE_PATH.read_bytes()
    ).hexdigest()
    assert after == before


def test_gate_cli_dispatches_frozen_main():
    completed = subprocess.run(
        [sys.executable, str(GATE_PATH), "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0
    assert "--model" in completed.stdout
    assert "--gpu-indices" in completed.stdout
    assert "--output-dir" in completed.stdout


def test_validate_result_accepts_positive_native_movement():
    assert gate.validate_result(_valid_result()) == _valid_result()


@pytest.mark.parametrize("batch_size", [1, 4])
@pytest.mark.parametrize(
    ("field", "message"),
    [
        ("h2d_copies", "real H2D copies"),
        ("h2d_bytes", "real H2D bytes"),
        ("d2h_copies", "real D2H copies"),
        ("d2h_bytes", "real D2H bytes"),
    ],
)
def test_validate_result_rejects_zero_native_movement(
    batch_size,
    field,
    message,
):
    result = _valid_result()
    for row in result["cells"][
        f"native_mtp:b{batch_size}"
    ]["kv_rank_deltas"]:
        row[field] = 0
    with pytest.raises(ValueError, match=message):
        gate.validate_result(result)


def test_worker_uses_frozen_long_context_configuration():
    worker = _load_module(
        "qwen35_native_mtp_tp4_32k_target_kv_offload_worker",
        TOOLS
        / (
            "qwen35_native_mtp_tp4_32k_"
            "target_kv_offload_worker.py"
        ),
    )

    class FakeTokenizer:
        def encode(self, text, add_special_tokens):
            assert add_special_tokens is False
            seed = sum(ord(character) for character in text)
            return [
                10 + (seed + 37 * offset) % 997
                for offset in range(16)
            ]

    kwargs = worker.engine_kwargs(
        policy="native_mtp",
        batch_size=1,
    )
    prompt_rows = worker.build_prompt_rows(
        FakeTokenizer(),
        1,
    )

    assert kwargs["tensor_parallel_size"] == 4
    assert kwargs["max_model_len"] == 33024
    assert kwargs["max_num_batched_tokens"] == 132096
    assert kwargs["max_num_prefill_tokens_per_step"] == 1024
    assert kwargs["kv_offload_gpu_blocks"] == 68
    assert kwargs["kv_offload_logical_blocks"] == 640
    assert kwargs["kv_offload_blockwise_blocks"] == 8
    assert kwargs["qwen35_mtp_enabled"] is True
    assert len(prompt_rows[0]["token_ids"]) == 32768
    assert worker.gate.MAX_OUTPUT_TOKENS == 8
    assert worker.gate.MAX_PROPOSAL_TOKENS == 4
    assert "native_mtp" in worker.gate.POLICIES


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n",
        encoding="utf-8",
    )


def _published_run(tmp_path: Path):
    result = gate.validate_result(_valid_result())
    source_files = {
        name: "b" * 64
        for name in gate.DEFAULT_SOURCE_FILES
    }
    result["source_tree_sha256"] = gate.source_hashes_sha256(
        source_files
    )
    run_dir = tmp_path / "authority"
    gate.publish_authority(
        run_dir,
        result,
        source_files=source_files,
    )
    verifier = _load_module(
        "verify_qwen35_native_mtp_tp4_32k_fixture",
        VERIFIER_PATH,
    )
    return verifier, run_dir


def test_verifier_dispatches_32k_contract(tmp_path):
    verifier, run_dir = _published_run(tmp_path)

    assert verifier.gate.SCHEMA_VERSION == gate.SCHEMA_VERSION
    assert verifier.verify_run(run_dir) == {
        "classification": "PASS",
        "failures": [],
    }


@pytest.mark.parametrize(
    ("mutate", "match"),
    (
        (
            lambda run_dir: _mutate_result(
                run_dir,
                lambda result: result.update(
                    schema_version=(
                        "qwen35.native-mtp-tp4-16k-"
                        "target-kv-offload.v1"
                    )
                ),
            ),
            "schema",
        ),
        (
            lambda run_dir: _mutate_result(
                run_dir,
                lambda result: result["cells"][
                    "native_mtp:b1"
                ].update(prompt_token_count=16384),
            ),
            "prompt",
        ),
        (
            lambda run_dir: _mutate_manifest(
                run_dir,
                lambda manifest: manifest["source_files"].pop(
                    next(iter(manifest["source_files"]))
                ),
            ),
            "source file inventory",
        ),
        (
            lambda run_dir: _mutate_result(
                run_dir,
                lambda result: [
                    row.update(h2d_copies=0)
                    for row in result["cells"][
                        "native_mtp:b1"
                    ]["kv_rank_deltas"]
                ],
            ),
            "real H2D copies",
        ),
        (
            lambda run_dir: _mutate_result(
                run_dir,
                lambda result: result.update(
                    undeclared_result_field=True
                ),
            ),
            "result fields",
        ),
        (
            lambda run_dir: _mutate_manifest(
                run_dir,
                lambda manifest: manifest.update(
                    undeclared_manifest_field=True
                ),
            ),
            "source manifest",
        ),
    ),
)
def test_verifier_rejects_32k_tamper(
    tmp_path,
    mutate,
    match,
):
    verifier, run_dir = _published_run(tmp_path)
    mutate(run_dir)

    verified = verifier.verify_run(run_dir)

    assert verified["classification"] == "FAIL"
    assert any(
        match.lower() in failure.lower()
        for failure in verified["failures"]
    )


def _mutate_result(run_dir: Path, mutate) -> None:
    path = run_dir / "result.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    mutate(value)
    _write_json(path, value)


def _mutate_manifest(run_dir: Path, mutate) -> None:
    path = run_dir / "source_manifest.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    mutate(value)
    _write_json(path, value)


def test_remote_runner_is_bounded_and_source_bound():
    text = REMOTE_RUNNER_PATH.read_text(encoding="utf-8")

    for required in (
        "sitian@10.232.195.203",
        "FILE:/Users/bytedance/krb5cc_sitian",
        "ControlMaster=no",
        "ControlPath=none",
        (
            "qwen35_native_mtp_tp4_32k_"
            "target_kv_offload_gate.py"
        ),
        (
            "qwen35_native_mtp_tp4_32k_"
            "target_kv_offload_worker.py"
        ),
        (
            "verify_qwen35_native_mtp_tp4_32k_"
            "target_kv_offload_gate.py"
        ),
        (
            "qwen35_native_mtp_tp4_16k_"
            "target_kv_offload_gate.py"
        ),
        (
            "qwen35_native_mtp_tp4_16k_"
            "target_kv_offload_worker.py"
        ),
        (
            "verify_qwen35_native_mtp_tp4_16k_"
            "target_kv_offload_gate.py"
        ),
        "campaign.status",
        "campaign.pid",
        "campaign.exit_code",
        "authority.failed",
        "REMOTE_COMMAND_RETRY_ATTEMPTS",
        "REMOTE_RSYNC_RETRY_ATTEMPTS",
        "POLL_INTERVAL_SECONDS",
        "head -n 4",
        "campaign already terminal",
        "campaign already running",
        "refusing to replay existing local campaign",
        "refusing to replay existing remote campaign",
        "four-GPU preflight failed",
    ):
        assert required in text
    for forbidden in (
        "ControlMaster=yes",
        "pkill",
        "killall",
        "nvidia-smi --gpu-reset",
        "git clean",
        "while true",
    ):
        assert forbidden not in text
