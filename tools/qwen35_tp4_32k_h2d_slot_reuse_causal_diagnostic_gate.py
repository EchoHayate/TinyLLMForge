from __future__ import annotations

import hashlib
import json
from pathlib import Path


SCHEMA = "qwen35.tp4-32k-h2d-slot-reuse-causal-diagnostic.v1"
MODES = ("observe", "control")
BATCH_SIZES = (1, 4)
REQUIRED_CELL_KEYS = (
    "observe:b1",
    "observe:b4",
    "control:b1",
    "control:b4",
)
POLICY = "baseline"
PROMPT_TOKENS = 32768
MAX_OUTPUT_TOKENS = 8
MAX_PROPOSAL_TOKENS = 4
WORLD_SIZE = 4
RANKS = tuple(range(WORLD_SIZE))
BLOCK_SIZE = 256
GPU_BLOCKS = 68
LOGICAL_BLOCKS = 640
BLOCKWISE_BLOCKS = 8
TIMING_EPSILON_MS = 0.2
TOP_K = 5
LOGIT_ATOL = 1e-5
LOGIT_RTOL = 1e-5

SOURCE_MANIFEST = (
    "tinyvllm/engine/h2d_slot_reuse_diagnostic.py",
    "tinyvllm/engine/model_runner.py",
    "tinyvllm/layers/attention.py",
    "tinyvllm/engine/llm_engine.py",
    (
        "tools/qwen35_tp4_32k_h2d_slot_reuse_"
        "causal_diagnostic_gate.py"
    ),
    (
        "tools/qwen35_tp4_32k_h2d_slot_reuse_"
        "causal_diagnostic_worker.py"
    ),
    (
        "tools/verify_qwen35_tp4_32k_h2d_slot_reuse_"
        "causal_diagnostic.py"
    ),
)

INVARIANT_FIELDS = (
    "prompt_rows",
    "source_tree_sha256",
    "checkpoint_sha256",
    "world_size",
    "prompt_tokens",
    "max_output_tokens",
    "max_proposal_tokens",
    "block_size",
    "gpu_blocks",
    "logical_blocks",
    "blockwise_blocks",
    "async_copy",
    "batch_copy",
    "writeback_on_evict",
    "enforce_eager",
    "target_forward_count",
    "kv_rank_deltas",
    "kv_capacity_rows",
    "cleanup",
)

REPETITION_FIELDS = frozenset({
    "schema",
    "mode",
    "policy",
    "batch_size",
    "repetition",
    "world_size",
    "prompt_tokens",
    "max_output_tokens",
    "max_proposal_tokens",
    "block_size",
    "gpu_blocks",
    "logical_blocks",
    "blockwise_blocks",
    "async_copy",
    "batch_copy",
    "writeback_on_evict",
    "enforce_eager",
    "torch_version",
    "torch_cuda_runtime_version",
    "nvidia_driver_version",
    "cuda_device_names",
    "source_tree_sha256",
    "checkpoint_sha256",
    "timing_epsilon_ms",
    "prompt_rows",
    "output_rows",
    "compact_logit_rows",
    "rank_slot_rows",
    "step_observations",
    "target_forward_count",
    "kv_rank_deltas",
    "kv_capacity_rows",
    "cleanup",
    "cell_digest_sha256",
})


def cell_key(mode: str, batch_size: int) -> str:
    if mode not in MODES:
        raise ValueError("mode must be observe or control")
    if batch_size not in BATCH_SIZES:
        raise ValueError("batch_size must be 1 or 4")
    return f"{mode}:b{batch_size}"


def _json_sha256(value) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def source_tree_sha256(repo_root: str | Path) -> str:
    root = Path(repo_root)
    rows = []
    for relative in SOURCE_MANIFEST:
        path = root / relative
        if not path.is_file():
            raise ValueError(
                f"source manifest file is missing: {relative}"
            )
        rows.append({
            "path": relative,
            "sha256": hashlib.sha256(
                path.read_bytes()
            ).hexdigest(),
        })
    return _json_sha256(rows)


def compact_prediction_logits(
    logits,
    *,
    observation: dict,
    prediction_index: int,
    top_k: int = TOP_K,
) -> list[dict]:
    rows = logits.tolist() if hasattr(logits, "tolist") else logits
    metadata = observation.get("prediction_rows")
    if not isinstance(metadata, list) or len(metadata) != len(rows):
        raise ValueError(
            "prediction metadata does not match logits"
        )
    if (
        isinstance(prediction_index, bool)
        or not isinstance(prediction_index, int)
        or prediction_index < 0
    ):
        raise ValueError("prediction_index must be nonnegative")
    if top_k != TOP_K:
        raise ValueError("focused compact logits require top_k=5")
    compact = []
    for values, identity in zip(rows, metadata):
        ranked = sorted(
            enumerate(values),
            key=lambda item: (-float(item[1]), item[0]),
        )[:top_k]
        top_logits = [
            float(value) for _, value in ranked
        ]
        compact.append({
            "sequence_id": int(identity["sequence_id"]),
            "prompt_index": int(identity["prompt_index"]),
            "prediction_index": prediction_index,
            "input_token_id": int(identity["input_token_id"]),
            "position": int(identity["position"]),
            "context_length": int(identity["context_length"]),
            "top_tokens": [
                int(token_id) for token_id, _ in ranked
            ],
            "top_logits": top_logits,
            "top1_margin": (
                top_logits[0] - top_logits[1]
                if len(top_logits) > 1
                else None
            ),
            "argmax_token": int(ranked[0][0]),
        })
    return compact


def reject_tensors(value, *, name: str = "artifact") -> None:
    module = type(value).__module__
    if module.startswith("torch"):
        raise ValueError(f"{name} contains a tensor")
    if isinstance(value, dict):
        for key, item in value.items():
            reject_tensors(item, name=f"{name}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            reject_tensors(item, name=f"{name}[{index}]")


def _validate_output_rows(row: dict) -> None:
    output_rows = row["output_rows"]
    if (
        not isinstance(output_rows, list)
        or len(output_rows) != row["batch_size"]
    ):
        raise ValueError("diagnostic output inventory mismatch")
    prompt_indexes = []
    for output in output_rows:
        if not isinstance(output, dict):
            raise ValueError("diagnostic output row is invalid")
        token_ids = output.get("token_ids")
        if (
            output.get("token_count") != MAX_OUTPUT_TOKENS
            or not isinstance(token_ids, list)
            or len(token_ids) != MAX_OUTPUT_TOKENS
            or output.get("sha256") != _json_sha256(token_ids)
        ):
            raise ValueError("diagnostic output token row mismatch")
        prompt_indexes.append(output.get("prompt_index"))
    if sorted(prompt_indexes) != list(range(row["batch_size"])):
        raise ValueError("diagnostic output prompt inventory mismatch")


def _validate_step_observations(row: dict) -> None:
    observations = row["step_observations"]
    if not isinstance(observations, list) or not observations:
        raise ValueError(
            "diagnostic step observations are unavailable"
        )
    ordinary_forwards = 0
    forbidden_fields = (
        "target_forward_trace_rows",
        "side_state_lineage_rows",
        "paired_trace_rows",
        "side_state_rows",
    )
    for observation in observations:
        if not isinstance(observation, dict):
            raise ValueError(
                "diagnostic step observation is invalid"
            )
        if any(field in observation for field in forbidden_fields):
            raise ValueError(
                "focused diagnostic contains paired trace or side state"
            )
        if (
            observation.get("execution_mode") != "baseline"
            or observation.get("proposal_callback_count", 0) != 0
            or observation.get("shadow_target_forward_count", 0) != 0
        ):
            raise ValueError(
                "focused diagnostic contains speculative evidence"
            )
        authority = observation.get(
            "authority_normal_decode_target_forward_calls"
        )
        if authority != 1:
            raise ValueError(
                "focused diagnostic target-forward authority mismatch"
            )
        ordinary_forwards += authority
    if row["target_forward_count"] != ordinary_forwards:
        raise ValueError(
            "diagnostic target-forward count mismatch"
        )


def validate_repetition(value: object) -> dict:
    if not isinstance(value, dict) or set(value) != REPETITION_FIELDS:
        raise ValueError("diagnostic repetition fields mismatch")
    row = dict(value)
    if row["schema"] != SCHEMA:
        raise ValueError("diagnostic schema mismatch")
    if row["mode"] not in MODES or row["policy"] != POLICY:
        raise ValueError("diagnostic mode or policy mismatch")
    if cell_key(row["mode"], row["batch_size"]) not in REQUIRED_CELL_KEYS:
        raise ValueError("diagnostic cell key mismatch")
    expected_constants = {
        "world_size": WORLD_SIZE,
        "prompt_tokens": PROMPT_TOKENS,
        "max_output_tokens": MAX_OUTPUT_TOKENS,
        "max_proposal_tokens": MAX_PROPOSAL_TOKENS,
        "block_size": BLOCK_SIZE,
        "gpu_blocks": GPU_BLOCKS,
        "logical_blocks": LOGICAL_BLOCKS,
        "blockwise_blocks": BLOCKWISE_BLOCKS,
        "timing_epsilon_ms": TIMING_EPSILON_MS,
    }
    for field, expected in expected_constants.items():
        if row[field] != expected:
            raise ValueError(
                f"diagnostic {field} mismatch"
            )
    for field in (
        "torch_version",
        "torch_cuda_runtime_version",
        "nvidia_driver_version",
        "source_tree_sha256",
        "checkpoint_sha256",
    ):
        if not isinstance(row[field], str) or not row[field]:
            raise ValueError(
                f"diagnostic {field} is unavailable"
            )
    if (
        not isinstance(row["cuda_device_names"], list)
        or len(row["cuda_device_names"]) != WORLD_SIZE
        or any(not name for name in row["cuda_device_names"])
    ):
        raise ValueError(
            "diagnostic CUDA device inventory is unavailable"
        )
    if row["policy"] != "baseline":
        raise ValueError("focused diagnostic is baseline-only")
    _validate_output_rows(row)
    if any(
        slot_row.get("attention_stage")
        not in {"prefill", "decode"}
        for rank_row in row["rank_slot_rows"]
        for slot_row in (
            list(rank_row.get("read_rows", []))
            + list(rank_row.get("overwrite_rows", []))
        )
    ):
        raise ValueError("native-MTP stage is forbidden")
    _validate_step_observations(row)
    digest_input = dict(row)
    digest = digest_input.pop("cell_digest_sha256")
    if digest != _json_sha256(digest_input):
        raise ValueError("diagnostic cell digest mismatch")
    reject_tensors(row)
    return row


def validate_artifact(value: object) -> dict:
    if (
        not isinstance(value, dict)
        or set(value) != {"schema", "cells"}
        or value["schema"] != SCHEMA
        or not isinstance(value["cells"], dict)
        or set(value["cells"]) != set(REQUIRED_CELL_KEYS)
    ):
        raise ValueError("diagnostic artifact cell inventory mismatch")
    normalized = {}
    for key in REQUIRED_CELL_KEYS:
        repetitions = value["cells"][key]
        if not isinstance(repetitions, list) or not repetitions:
            raise ValueError("diagnostic cell repetitions are missing")
        seen = set()
        normalized_rows = []
        for repetition in repetitions:
            row = validate_repetition(repetition)
            if row["repetition"] in seen:
                raise ValueError(
                    "duplicate diagnostic repetition ID"
                )
            seen.add(row["repetition"])
            if cell_key(row["mode"], row["batch_size"]) != key:
                raise ValueError("diagnostic cell identity mismatch")
            normalized_rows.append(row)
        normalized[key] = normalized_rows
    return {"schema": SCHEMA, "cells": normalized}


def _rank_slot_inventory(row: dict) -> tuple[dict, ...]:
    rank_rows = row["rank_slot_rows"]
    if (
        not isinstance(rank_rows, list)
        or len(rank_rows) != WORLD_SIZE
    ):
        raise ValueError("rank slot row inventory mismatch")
    by_rank = {}
    for rank_row in rank_rows:
        if (
            not isinstance(rank_row, dict)
            or rank_row.get("rank") not in RANKS
            or rank_row.get("rank") in by_rank
            or rank_row.get("schema") != SCHEMA
            or rank_row.get("mode") != row["mode"]
            or not isinstance(
                rank_row.get("stream_inventory"),
                list,
            )
            or not rank_row["stream_inventory"]
        ):
            raise ValueError("rank slot lifecycle is incomplete")
        by_rank[rank_row["rank"]] = rank_row
    if tuple(sorted(by_rank)) != RANKS:
        raise ValueError("rank slot rank inventory mismatch")
    return tuple(by_rank[rank] for rank in RANKS)


def _index_logit_row(
    repetition: dict,
    prediction_index: int,
) -> dict | None:
    rows = [
        row
        for row in repetition["compact_logit_rows"]
        if row.get("prompt_index") == 0
        and row.get("prediction_index") == prediction_index
    ]
    if len(rows) != 1:
        return None
    return rows[0]


def _logits_match(left: dict | None, right: dict | None) -> bool:
    if left is None or right is None:
        return False
    if (
        left.get("top_tokens") != right.get("top_tokens")
        or left.get("argmax_token") != right.get("argmax_token")
    ):
        return False
    left_logits = left.get("top_logits")
    right_logits = right.get("top_logits")
    if (
        not isinstance(left_logits, list)
        or not isinstance(right_logits, list)
        or len(left_logits) != TOP_K
        or len(right_logits) != TOP_K
    ):
        return False
    return all(
        abs(float(candidate) - float(reference))
        <= LOGIT_ATOL + LOGIT_RTOL * abs(float(reference))
        for reference, candidate in zip(left_logits, right_logits)
    )


def _prediction_identity(
    row: dict | None,
) -> tuple[int, int, int, int] | None:
    if row is None:
        return None
    values = (
        row.get("prediction_index"),
        row.get("input_token_id"),
        row.get("position"),
        row.get("context_length"),
    )
    if any(
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < 0
        for value in values
    ):
        return None
    if values[3] != values[2] + 1:
        return None
    return values


def _output_tokens(repetition: dict) -> list[int] | None:
    rows = [
        row
        for row in repetition["output_rows"]
        if row.get("prompt_index") == 0
    ]
    if len(rows) != 1:
        return None
    return rows[0].get("token_ids")


def _prompt_zero_identity(
    repetition: dict,
) -> tuple[int, ...] | None:
    rows = [
        row
        for row in repetition["prompt_rows"]
        if row.get("prompt_index") == 0
    ]
    if len(rows) != 1:
        return None
    row = rows[0]
    token_ids = row.get("token_ids")
    if (
        not isinstance(token_ids, list)
        or not token_ids
        or any(
            isinstance(token_id, bool)
            or not isinstance(token_id, int)
            or token_id < 0
            for token_id in token_ids
        )
    ):
        return None
    if (
        "token_count" in row
        and row["token_count"] != len(token_ids)
    ):
        return None
    if (
        "sha256" in row
        and row["sha256"] != _json_sha256(token_ids)
    ):
        return None
    return tuple(token_ids)


def _invariant_projection(repetition: dict) -> dict:
    return {
        field: repetition[field]
        for field in INVARIANT_FIELDS
    }


def _hazard_rows(
    repetition: dict,
) -> tuple[list[dict], list[dict], bool]:
    unsafe = []
    eligible = []
    complete = True
    for rank_row in _rank_slot_inventory(repetition):
        for overwrite in rank_row.get("overwrite_rows", []):
            status = overwrite.get("timing_status")
            if status in (
                "NO_PRIOR_OCCUPANCY",
                "NO_PRIOR_READ",
            ):
                continue
            if status == "ORDERING_AMBIGUOUS":
                complete = False
                continue
            if status not in (
                "UNSAFE_OVERLAP_OBSERVED",
                "READ_COMPLETED_BEFORE_H2D",
            ):
                complete = False
                continue
            if overwrite.get(
                "read_done_after_h2d_start_ms"
            ) is None:
                complete = False
                continue
            eligible.append(overwrite)
            if status == "UNSAFE_OVERLAP_OBSERVED":
                unsafe.append(overwrite)
    return unsafe, eligible, complete


def _hazard_key(row: dict) -> tuple:
    return (
        int(row["rank"]),
        int(row["physical_slot"]),
        int(row["old_occupancy_generation"]),
        tuple(int(value) for value in row["read_event_ordinals"]),
    )


def _control_waits_cover(
    observe_unsafe: list[dict],
    control_repetition: dict,
) -> bool:
    control_rows = {}
    for rank_row in _rank_slot_inventory(control_repetition):
        for row in rank_row.get("overwrite_rows", []):
            if row.get("old_occupancy_generation") is None:
                continue
            control_rows[_hazard_key(row)] = row
    for observed in observe_unsafe:
        controlled = control_rows.get(_hazard_key(observed))
        if controlled is None:
            return False
        predecessors = set(
            int(value)
            for value in observed["read_event_ordinals"]
        )
        waited = set(
            int(value)
            for value in controlled.get(
                "control_wait_event_ordinals",
                [],
            )
        )
        if not predecessors or not predecessors.issubset(waited):
            return False
    return True


def evaluate_campaign(value: object) -> dict:
    try:
        artifact = validate_artifact(value)
    except ValueError as error:
        return _decision(
            "INCONCLUSIVE",
            [f"artifact validation failed: {error}"],
            {},
        )
    cells = artifact["cells"]
    inventory = {
        key: sorted(
            int(row["repetition"]) for row in rows
        )
        for key, rows in cells.items()
    }
    reasons = []
    if any(len(rows) < 2 for rows in cells.values()):
        reasons.append(
            "at least two repetitions per cell are required"
        )
    repetition_sets = {
        key: set(inventory[key])
        for key in REQUIRED_CELL_KEYS
    }
    if len({tuple(sorted(values)) for values in repetition_sets.values()}) != 1:
        reasons.append("repetition inventories do not match")
    repetitions = sorted(
        set.intersection(*repetition_sets.values())
        if repetition_sets
        else set()
    )
    by_key = {
        key: {
            int(row["repetition"]): row
            for row in rows
        }
        for key, rows in cells.items()
    }
    invariant_failure = False
    timing_incomplete = False
    logit_incomplete = False
    prompt_identity_failure = False
    prediction_identity_failure = False
    observe_drift_all = True
    control_match_all = True
    control_output_parity_all = True
    observe_has_unsafe_all = True
    control_has_no_unsafe_all = True
    control_waits_cover_all = True
    observe_complete_no_unsafe_all = True
    for repetition in repetitions:
        for batch_size in BATCH_SIZES:
            observe = by_key[
                f"observe:b{batch_size}"
            ][repetition]
            control = by_key[
                f"control:b{batch_size}"
            ][repetition]
            if (
                _invariant_projection(observe)
                != _invariant_projection(control)
            ):
                invariant_failure = True
            try:
                observe_unsafe, observe_eligible, observe_complete = (
                    _hazard_rows(observe)
                )
                control_unsafe, _, control_complete = (
                    _hazard_rows(control)
                )
            except ValueError:
                timing_incomplete = True
                continue
            if not observe_complete or not control_complete:
                timing_incomplete = True
            observe_has_unsafe_all &= bool(observe_unsafe)
            control_has_no_unsafe_all &= not control_unsafe
            observe_complete_no_unsafe_all &= (
                observe_complete
                and bool(observe_eligible)
                and not observe_unsafe
            )
            control_waits_cover_all &= _control_waits_cover(
                observe_unsafe,
                control,
            )
        observe_b1 = by_key["observe:b1"][repetition]
        observe_b4 = by_key["observe:b4"][repetition]
        control_b1 = by_key["control:b1"][repetition]
        control_b4 = by_key["control:b4"][repetition]
        prompt_zero_identities = {
            _prompt_zero_identity(
                by_key[key][repetition]
            )
            for key in REQUIRED_CELL_KEYS
        }
        if (
            None in prompt_zero_identities
            or len(prompt_zero_identities) != 1
        ):
            prompt_identity_failure = True
        required_logit_rows = {
            key: {
                prediction_index: _index_logit_row(
                    by_key[key][repetition],
                    prediction_index,
                )
                for prediction_index in (0, 1)
            }
            for key in REQUIRED_CELL_KEYS
        }
        if any(
            logit_row is None
            for rows in required_logit_rows.values()
            for logit_row in rows.values()
        ):
            logit_incomplete = True
            continue
        for prediction_index in (0, 1):
            identities = {
                _prediction_identity(
                    required_logit_rows[key][prediction_index]
                )
                for key in REQUIRED_CELL_KEYS
            }
            if None in identities or len(identities) != 1:
                prediction_identity_failure = True
        observe_drift_all &= not _logits_match(
            required_logit_rows["observe:b1"][1],
            required_logit_rows["observe:b4"][1],
        )
        control_match_all &= _logits_match(
            required_logit_rows["control:b1"][1],
            required_logit_rows["control:b4"][1],
        )
        control_output_parity_all &= (
            _output_tokens(control_b1)
            == _output_tokens(control_b4)
            and _output_tokens(control_b1) is not None
        )
    if invariant_failure:
        reasons.append(
            "observe/control invariant projection differs"
        )
    if timing_incomplete:
        reasons.append("timing coverage is incomplete or ambiguous")
    if logit_incomplete:
        reasons.append(
            "prediction index 0/1 compact-logit coverage is incomplete"
        )
    if prompt_identity_failure:
        reasons.append("cross-batch prompt-0 identity differs")
    if prediction_identity_failure:
        reasons.append("cross-batch prediction identity differs")
    if reasons:
        return _decision("INCONCLUSIVE", reasons, inventory)
    supported = (
        bool(repetitions)
        and observe_drift_all
        and observe_has_unsafe_all
        and control_has_no_unsafe_all
        and control_waits_cover_all
        and control_match_all
        and control_output_parity_all
    )
    if supported:
        return _decision(
            "SUPPORTED",
            [
                "observe drift and unsafe overlap reproduced",
                "control waits removed overlap and index-1 drift",
            ],
            inventory,
        )
    rejected = (
        observe_drift_all
        and (
            observe_complete_no_unsafe_all
            or (
                observe_has_unsafe_all
                and control_has_no_unsafe_all
                and control_waits_cover_all
                and (
                    not control_match_all
                    or not control_output_parity_all
                )
            )
        )
    )
    if rejected:
        return _decision(
            "REJECTED",
            [
                "valid control did not remove retained drift/output mismatch"
            ],
            inventory,
        )
    return _decision(
        "INCONCLUSIVE",
        [
            "campaign does not satisfy supported or rejected matrix"
        ],
        inventory,
    )


def _decision(
    terminal: str,
    reasons: list[str],
    inventory: dict,
) -> dict:
    classification = (
        "TP4_32K_H2D_SLOT_REUSE_GPU_CAUSALITY="
        f"{terminal}"
    )
    result = {
        "classification": classification,
        "supported": terminal == "SUPPORTED",
        "rejected": terminal == "REJECTED",
        "inconclusive": terminal == "INCONCLUSIVE",
        "reasons": list(reasons),
        "repetition_inventory": inventory,
    }
    if sum((
        result["supported"],
        result["rejected"],
        result["inconclusive"],
    )) != 1:
        raise RuntimeError(
            "terminal diagnostic booleans are not exclusive"
        )
    return result
