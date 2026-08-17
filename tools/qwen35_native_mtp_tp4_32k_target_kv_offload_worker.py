from __future__ import annotations

from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
import sys

import torch


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_TOOLS = Path(__file__).resolve().parent
gate = _load_module(
    "qwen35_native_mtp_tp4_32k_target_kv_offload_gate",
    _TOOLS
    / "qwen35_native_mtp_tp4_32k_target_kv_offload_gate.py",
)
_frozen_worker = _load_module(
    "_qwen35_native_mtp_tp4_32k_frozen_worker",
    _TOOLS
    / "qwen35_native_mtp_tp4_16k_target_kv_offload_worker.py",
)
_frozen_worker.gate = gate
_frozen_worker._frozen_worker.gate = gate
_frozen_worker.tp1_worker.gate = gate

for _name, _value in vars(_frozen_worker).items():
    if not _name.startswith("__") and _name != "gate":
        globals()[_name] = _value


TRACE_SCHEMA = (
    "qwen35.native-mtp-tp4-32k-paired-verify-trace.v1"
)
TRACE_LIMITATIONS = (
    "diagnostic_only",
    "full_logits_not_captured",
    "target_kv_shadow_not_established",
    "root_cause_not_established",
    "phase1_not_promotable",
    "performance_not_established",
)

_ENGINE_TRACE_ROW_FIELDS = frozenset({
    "schema",
    "policy",
    "batch_size",
    "engine_step",
    "target_forward_ordinal",
    "stage",
    "execution_mode",
    "sequence_id",
    "query_offset",
    "query_len",
    "row_index",
    "prediction_index",
    "input_token_id",
    "position",
    "context_length",
    "logical_block_identities",
    "logical_block_coverage",
    "top_tokens",
    "top_logits",
    "top1_margin",
    "argmax_token",
})
_ENRICHED_TRACE_ROW_FIELDS = (
    _ENGINE_TRACE_ROW_FIELDS | {"prompt_index"}
)
_RAW_SIDE_STATE_ROW_FIELDS = frozenset({
    "sequence_id",
    "event",
    "checkpoint_index",
    "committed_input_count",
    "fingerprint",
    "engine_step",
})
_OWNER_SIDE_STATE_ROW_FIELDS = (
    _RAW_SIDE_STATE_ROW_FIELDS - {"engine_step"}
)
_FINAL_SIDE_STATE_ROW_FIELDS = frozenset({
    "schema",
    "policy",
    "batch_size",
    "engine_step",
    "sequence_id",
    "event",
    "checkpoint_index",
    "committed_input_count",
    "proposal_token_ids",
    "accepted_token_ids",
    "verify_input_count",
    "fallback_target_token",
    "fingerprint",
})
_DIAGNOSTIC_CELL_FIELDS = frozenset({
    "policy",
    "batch_size",
    "output_rows",
    "target_forward_trace_rows",
    "side_state_lineage_rows",
    "step_observations",
    "rank_cleanup_summary",
    "cell_digest_sha256",
})


def _validate_exact_fields(
    row: dict,
    expected: frozenset[str],
    name: str,
) -> None:
    if not isinstance(row, dict) or set(row) != expected:
        raise ValueError(f"{name} fields mismatch")


def _validate_trace_identity_coverage(
    row: dict,
    name: str,
) -> None:
    identities = row["logical_block_identities"]
    coverage = row["logical_block_coverage"]
    if any(
        not isinstance(entry, (list, tuple))
        or len(entry) != 3
        or int(entry[0]) < 0
        for entry in coverage
    ):
        raise ValueError(f"{name} logical coverage is invalid")
    required = (
        max(int(entry[0]) for entry in coverage) + 1
        if coverage
        else 0
    )
    if len(identities) < required:
        raise ValueError(
            f"{name} logical identity coverage is incomplete"
        )


def _semantic_trace_key(row: dict) -> tuple:
    return (
        row["batch_size"],
        row["prompt_index"],
        row["prediction_index"],
        row["input_token_id"],
        row["position"],
        row["context_length"],
        tuple(
            tuple(value)
            for value in row["logical_block_coverage"]
        ),
    )


def pair_target_forward_rows(
    baseline_rows: list[dict],
    native_rows: list[dict],
) -> list[dict]:
    native_by_key = {}
    for row in native_rows:
        _validate_exact_fields(
            row,
            _ENRICHED_TRACE_ROW_FIELDS,
            "native trace row",
        )
        _validate_trace_identity_coverage(
            row,
            "native trace row",
        )
        key = _semantic_trace_key(row)
        if key in native_by_key:
            raise ValueError("duplicate native trace match")
        native_by_key[key] = row
    paired = []
    for baseline in baseline_rows:
        _validate_exact_fields(
            baseline,
            _ENRICHED_TRACE_ROW_FIELDS,
            "baseline trace row",
        )
        _validate_trace_identity_coverage(
            baseline,
            "baseline trace row",
        )
        key = _semantic_trace_key(baseline)
        native = native_by_key.pop(key, None)
        if native is None:
            raise ValueError("missing native trace match")
        shared = sorted(
            set(baseline["top_tokens"]).intersection(
                native["top_tokens"]
            )
        )
        baseline_logits = dict(zip(
            baseline["top_tokens"],
            baseline["top_logits"],
        ))
        native_logits = dict(zip(
            native["top_tokens"],
            native["top_logits"],
        ))
        paired.append({
            "batch_size": baseline["batch_size"],
            "prompt_index": baseline["prompt_index"],
            "prediction_index": baseline["prediction_index"],
            "input_token_id": baseline["input_token_id"],
            "position": baseline["position"],
            "context_length": baseline["context_length"],
            "baseline_stage": baseline["stage"],
            "native_stage": native["stage"],
            "baseline_query_len": baseline["query_len"],
            "native_query_len": native["query_len"],
            "baseline_top_tokens": baseline["top_tokens"],
            "native_top_tokens": native["top_tokens"],
            "baseline_top_logits": baseline["top_logits"],
            "native_top_logits": native["top_logits"],
            "baseline_argmax_token": baseline["argmax_token"],
            "native_argmax_token": native["argmax_token"],
            "argmax_equal": (
                baseline["argmax_token"]
                == native["argmax_token"]
            ),
            "baseline_logical_block_identities": baseline[
                "logical_block_identities"
            ],
            "native_logical_block_identities": native[
                "logical_block_identities"
            ],
            "logical_block_coverage_equal": (
                baseline["logical_block_coverage"]
                == native["logical_block_coverage"]
            ),
            "shared_token_logit_deltas": {
                str(token_id): (
                    float(native_logits[token_id])
                    - float(baseline_logits[token_id])
                )
                for token_id in shared
            },
            "first_topk_disagreement": (
                baseline["top_tokens"]
                != native["top_tokens"]
                or any(
                    native_logits[token_id]
                    != baseline_logits[token_id]
                    for token_id in shared
                )
            ),
            "baseline_target_forward_ordinal": baseline[
                "target_forward_ordinal"
            ],
            "native_target_forward_ordinal": native[
                "target_forward_ordinal"
            ],
        })
    if native_by_key:
        raise ValueError("unpaired native trace rows remain")
    return sorted(
        paired,
        key=lambda row: (
            row["prompt_index"],
            row["prediction_index"],
            row["native_target_forward_ordinal"],
        ),
    )


def assemble_side_state_lineage(
    *,
    policy: str,
    batch_size: int,
    trace_rows: list[dict],
    observations: list[dict],
    sequence_to_prompt: dict[int, int],
) -> list[dict]:
    if policy not in ("baseline", "native_mtp"):
        raise ValueError("side-state lineage policy is invalid")
    if (
        isinstance(batch_size, bool)
        or not isinstance(batch_size, int)
        or batch_size <= 0
    ):
        raise ValueError(
            "side-state lineage batch size is invalid"
        )
    finalized = []
    for raw in trace_rows:
        _validate_exact_fields(
            raw,
            _RAW_SIDE_STATE_ROW_FIELDS,
            "raw side-state trace row",
        )
        sequence_id = int(raw["sequence_id"])
        if sequence_id not in sequence_to_prompt:
            raise ValueError(
                "side-state sequence prompt mapping is missing"
            )
        engine_step = int(raw["engine_step"])
        if engine_step < 0 or engine_step >= len(observations):
            raise ValueError(
                "side-state engine step is out of range"
            )
        event = raw["event"]
        proposal = ()
        accepted = ()
        verify_input_count = 0
        fallback = None
        committed_input_count = None
        if event == "selected_checkpoint":
            committed_input_count = raw[
                "committed_input_count"
            ]
            if raw["checkpoint_index"] != committed_input_count:
                raise ValueError(
                    "selected checkpoint must equal committed input "
                    "count"
                )
            observation = observations[engine_step]
            proposal = tuple(
                observation[
                    "speculative_proposal_token_ids_by_seq"
                ][sequence_id]
            )
            accepted = tuple(
                observation[
                    "speculative_accepted_draft_token_ids_by_seq"
                ][sequence_id]
            )
            if accepted != proposal[:len(accepted)]:
                raise ValueError(
                    "accepted tokens must be an exact proposal prefix"
                )
            verify_input_count = max(0, len(proposal) - 1)
            expected_committed = 1 + min(
                len(accepted),
                verify_input_count,
            )
            if committed_input_count != expected_committed:
                raise ValueError(
                    "selected committed input count is inconsistent"
                )
            new_tokens = tuple(
                observation["new_completion_tokens_by_seq"][
                    sequence_id
                ]
            )
            if new_tokens[:len(accepted)] != accepted:
                raise ValueError(
                    "accepted tokens must match output prefix"
                )
            if len(accepted) < verify_input_count:
                if len(new_tokens) <= len(accepted):
                    raise ValueError(
                        "partial acceptance requires a fallback target"
                    )
                fallback = new_tokens[len(accepted)]
            elif len(new_tokens) > len(accepted):
                fallback = new_tokens[len(accepted)]
        elif event not in (
            "first_target_checkpoint",
            "tail_checkpoint",
        ):
            raise ValueError(
                "side-state trace event is invalid"
            )
        row = {
            "schema": TRACE_SCHEMA,
            "policy": policy,
            "batch_size": batch_size,
            "engine_step": engine_step,
            "sequence_id": sequence_id,
            "event": event,
            "checkpoint_index": int(
                raw["checkpoint_index"]
            ),
            "committed_input_count": committed_input_count,
            "proposal_token_ids": list(proposal),
            "accepted_token_ids": list(accepted),
            "verify_input_count": verify_input_count,
            "fallback_target_token": (
                int(fallback)
                if fallback is not None
                else None
            ),
            "fingerprint": raw["fingerprint"],
        }
        _validate_exact_fields(
            row,
            _FINAL_SIDE_STATE_ROW_FIELDS,
            "side-state lineage row",
        )
        finalized.append(row)
    return finalized


def run_generation_with_paired_trace(
    *,
    engine,
    prompt_rows,
    sampling_params,
    synchronize,
    policy,
    batch_size,
    trace_capture,
    target_forward_capture=None,
) -> tuple[list[dict], list[dict]]:
    for row in prompt_rows:
        engine.add_request(
            row["token_ids"],
            sampling_params,
        )
    runner = engine.model_runner
    owner = getattr(
        runner,
        "qwen35_speculative_state_owner",
        None,
    )
    runner.enable_spec_verify_trace_recording(True)
    if owner is not None:
        owner.enable_trace_recording(True)
    target_rows = []
    side_rows = []
    observations = []
    outputs_by_id = {}
    try:
        engine_step = 0
        while not engine.is_finished():
            runner.set_spec_verify_trace_context(
                policy,
                batch_size,
                engine_step,
            )
            target_forward_calls_before = (
                0
                if target_forward_capture is None
                else gate._integer(
                    target_forward_capture.get(
                        "ordinary_decode_target_forward_calls",
                        0,
                    ),
                    "ordinary decode target forward calls",
                )
            )
            step_outputs, _ = engine.step()
            synchronize()
            for raw_row in (
                runner.drain_spec_verify_trace_rows()
            ):
                row = dict(raw_row)
                _validate_exact_fields(
                    row,
                    _ENGINE_TRACE_ROW_FIELDS,
                    "engine trace row",
                )
                if row["schema"] != TRACE_SCHEMA:
                    raise ValueError(
                        "engine trace schema mismatch"
                    )
                if (
                    row["policy"] != policy
                    or row["batch_size"] != batch_size
                    or row["engine_step"] != engine_step
                ):
                    raise ValueError(
                        "engine trace context mismatch"
                    )
                _validate_trace_identity_coverage(
                    row,
                    "engine trace row",
                )
                target_rows.append(row)
            if owner is not None:
                for raw_row in owner.drain_trace_rows():
                    row = dict(raw_row)
                    _validate_exact_fields(
                        row,
                        _OWNER_SIDE_STATE_ROW_FIELDS,
                        "owner side-state trace row",
                    )
                    side_rows.append({
                        **row,
                        "engine_step": engine_step,
                    })
            observation = getattr(
                engine,
                "last_step_observation",
                None,
            )
            if not isinstance(observation, dict):
                raise RuntimeError(
                    "engine step observation is unavailable"
                )
            observation = dict(observation)
            if target_forward_capture is not None:
                target_forward_calls_after = gate._integer(
                    target_forward_capture.get(
                        "ordinary_decode_target_forward_calls",
                        0,
                    ),
                    "ordinary decode target forward calls",
                )
                if (
                    target_forward_calls_after
                    < target_forward_calls_before
                ):
                    raise RuntimeError(
                        "ordinary decode target forward count "
                        "regressed"
                    )
                observation[
                    "authority_normal_decode_target_forward_calls"
                ] = (
                    target_forward_calls_after
                    - target_forward_calls_before
                )
            observations.append(observation)
            for sequence_id, token_ids in step_outputs:
                outputs_by_id[int(sequence_id)] = [
                    int(token_id) for token_id in token_ids
                ]
            engine_step += 1
    finally:
        runner.enable_spec_verify_trace_recording(False)
        if owner is not None:
            owner.enable_trace_recording(False)
    output_rows = []
    sequence_to_prompt = {}
    for prompt_index, sequence_id in enumerate(
        sorted(outputs_by_id)
    ):
        token_ids = outputs_by_id[sequence_id]
        if len(token_ids) != gate.MAX_OUTPUT_TOKENS:
            raise RuntimeError(
                "engine output token count mismatch"
            )
        sequence_to_prompt[sequence_id] = prompt_index
        output_rows.append({
            "prompt_index": prompt_index,
            "token_count": len(token_ids),
            "token_ids": token_ids,
            "sha256": gate._json_sha256(token_ids),
        })
    if len(output_rows) != len(prompt_rows):
        raise RuntimeError(
            "engine output inventory does not match prompts"
        )
    enriched_target_rows = []
    for raw_row in target_rows:
        sequence_id = int(raw_row["sequence_id"])
        if sequence_id not in sequence_to_prompt:
            raise ValueError(
                "trace sequence prompt mapping is missing"
            )
        row = {
            **raw_row,
            "prompt_index": sequence_to_prompt[sequence_id],
        }
        _validate_exact_fields(
            row,
            _ENRICHED_TRACE_ROW_FIELDS,
            "enriched trace row",
        )
        enriched_target_rows.append(row)
    trace_capture.clear()
    trace_capture.update({
        "target_forward_trace_rows": enriched_target_rows,
        "raw_side_state_rows": side_rows,
        "sequence_to_prompt": sequence_to_prompt,
        "step_observations": observations,
    })
    return output_rows, observations


def _reject_tensors(value) -> None:
    if isinstance(value, torch.Tensor):
        raise ValueError("trace artifact contains a tensor")
    if isinstance(value, dict):
        for child in value.values():
            _reject_tensors(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            _reject_tensors(child)


def build_paired_trace_artifact(
    *,
    cells: dict[str, dict],
    source_manifest_sha256: str,
    target_manifest_sha256: str,
    mtp_manifest_sha256: str,
) -> dict:
    _reject_tensors(cells)
    expected_cell_keys = {
        "baseline:b1",
        "native_mtp:b1",
        "baseline:b4",
        "native_mtp:b4",
    }
    if not isinstance(cells, dict) or set(cells) != expected_cell_keys:
        raise ValueError("paired trace cell keys mismatch")
    for key, cell in cells.items():
        _validate_exact_fields(
            cell,
            _DIAGNOSTIC_CELL_FIELDS,
            "paired trace cell",
        )
        payload = {
            name: value
            for name, value in cell.items()
            if name != "cell_digest_sha256"
        }
        if (
            cell["cell_digest_sha256"]
            != gate._json_sha256(payload)
        ):
            raise ValueError("paired trace cell digest mismatch")
        expected_policy, batch_label = key.split(":b")
        if (
            cell["policy"] != expected_policy
            or cell["batch_size"] != int(batch_label)
        ):
            raise ValueError("paired trace cell identity mismatch")
    paired_rows = []
    for batch_size in gate.BATCH_SIZES:
        paired_rows.extend(pair_target_forward_rows(
            cells[f"baseline:b{batch_size}"][
                "target_forward_trace_rows"
            ],
            cells[f"native_mtp:b{batch_size}"][
                "target_forward_trace_rows"
            ],
        ))
    divergences = [
        row
        for row in paired_rows
        if (
            not row["logical_block_coverage_equal"]
            or not row["argmax_equal"]
            or row["first_topk_disagreement"]
        )
    ]
    first_divergence = (
        min(
            divergences,
            key=lambda row: (
                row["prompt_index"],
                row["prediction_index"],
                row["native_target_forward_ordinal"],
            ),
        )
        if divergences
        else None
    )
    frozen_contract = {
        "prompt_tokens": gate.PROMPT_TOKENS,
        "output_tokens": gate.MAX_OUTPUT_TOKENS,
        "world_size": gate.WORLD_SIZE,
        "batch_sizes": list(gate.BATCH_SIZES),
        "max_proposal_tokens": gate.MAX_PROPOSAL_TOKENS,
        "max_model_len": gate.MAX_MODEL_LEN,
        "max_num_batched_tokens": (
            gate.MAX_NUM_BATCHED_TOKENS
        ),
        "max_num_prefill_tokens_per_step": (
            gate.MAX_NUM_PREFILL_TOKENS_PER_STEP
        ),
        "kv_offload_gpu_blocks": gate.KV_OFFLOAD_GPU_BLOCKS,
        "kv_offload_logical_blocks": (
            gate.KV_OFFLOAD_LOGICAL_BLOCKS
        ),
        "block_size": gate.BLOCK_SIZE,
    }
    artifact = {
        "schema": TRACE_SCHEMA,
        "created_at_utc": (
            datetime.now(timezone.utc).isoformat()
        ),
        "source_manifest_sha256": source_manifest_sha256,
        "target_manifest_sha256": target_manifest_sha256,
        "mtp_manifest_sha256": mtp_manifest_sha256,
        "frozen_contract": frozen_contract,
        "cells": cells,
        "first_divergence": first_divergence,
        "limitations": list(TRACE_LIMITATIONS),
    }
    _reject_tensors(artifact)
    return artifact


def write_paired_trace_artifact(path, artifact) -> None:
    _reject_tensors(artifact)
    path = Path(path)
    if path.exists():
        raise ValueError(
            "paired trace artifact path already exists"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            artifact,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n",
        encoding="utf-8",
    )


def run_paired_trace_cell(
    *,
    model_path: str,
    gpu_indices: tuple[int, ...],
    policy: str,
    batch_size: int,
    dist_port: int,
    master_port: int,
    engine_factory,
    sampling_params_type,
    runtime_type,
    synchronize,
) -> dict:
    trace_capture = {}

    def traced_generation(**kwargs):
        return run_generation_with_paired_trace(
            **kwargs,
            policy=policy,
            batch_size=batch_size,
            trace_capture=trace_capture,
        )

    cell = run_policy_cell(
        model_path=model_path,
        gpu_indices=gpu_indices,
        policy=policy,
        batch_size=batch_size,
        dist_port=dist_port,
        master_port=master_port,
        engine_factory=engine_factory,
        sampling_params_type=sampling_params_type,
        runtime_type=runtime_type,
        synchronize=synchronize,
        run_generation_fn=traced_generation,
    )
    finalized_lineage = assemble_side_state_lineage(
        policy=policy,
        batch_size=batch_size,
        trace_rows=trace_capture["raw_side_state_rows"],
        observations=trace_capture["step_observations"],
        sequence_to_prompt=trace_capture[
            "sequence_to_prompt"
        ],
    )
    diagnostic_cell = {
        "policy": cell["policy"],
        "batch_size": cell["batch_size"],
        "output_rows": cell["output_rows"],
        "target_forward_trace_rows": trace_capture[
            "target_forward_trace_rows"
        ],
        "side_state_lineage_rows": finalized_lineage,
        "step_observations": trace_capture[
            "step_observations"
        ],
        "rank_cleanup_summary": cell["cleanup"],
    }
    diagnostic_cell["cell_digest_sha256"] = (
        gate._json_sha256(diagnostic_cell)
    )
    return diagnostic_cell


def run_paired_trace_diagnostic(
    *,
    output_path,
    repo_root,
    cell_kwargs_by_key,
    run_cell_fn=run_paired_trace_cell,
) -> dict:
    required_keys = (
        "baseline:b1",
        "native_mtp:b1",
        "baseline:b4",
        "native_mtp:b4",
    )
    if (
        not isinstance(cell_kwargs_by_key, dict)
        or set(cell_kwargs_by_key) != set(required_keys)
    ):
        raise ValueError(
            "paired trace diagnostic cell keys mismatch"
        )
    source_manifest_sha256 = gate.source_tree_sha256(
        repo_root,
        gate.DEFAULT_SOURCE_FILES,
    )
    cells = {}
    for key in required_keys:
        kwargs = cell_kwargs_by_key[key]
        if not isinstance(kwargs, dict):
            raise ValueError(
                "paired trace diagnostic cell kwargs mismatch"
            )
        cells[key] = run_cell_fn(**kwargs)
    artifact = build_paired_trace_artifact(
        cells=cells,
        source_manifest_sha256=source_manifest_sha256,
        target_manifest_sha256=(
            gate.TARGET_MODEL_MANIFEST_SHA256
        ),
        mtp_manifest_sha256=(
            gate.MTP_CHECKPOINT_MANIFEST_SHA256
        ),
    )
    write_paired_trace_artifact(output_path, artifact)
    return artifact


if __name__ == "__main__":
    sys.exit(main())
