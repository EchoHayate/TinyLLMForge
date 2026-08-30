from __future__ import annotations

import math
from statistics import median


EXPECTED_PROFILE = {
    "num_hidden_layers": 64,
    "hidden_size": 5120,
    "vocab_size": 248320,
    "dtype": "bfloat16",
}
EXPECTED_TENSOR_PARALLEL_SIZE = 4
EXPECTED_DECODE_COLLECTIVE_COUNT = 66
MEDIAN_OVERHEAD_LIMIT = 0.03
MAXIMUM_OVERHEAD_LIMIT = 0.05
MINIMUM_OPPORTUNITY_RATIO = 0.05
EVENT_BUDGETS = (0, 8, 16, 32)


def _field(value, name):
    if isinstance(value, dict):
        return value.get(name)
    return getattr(value, name, None)


def _catalog_row(
    *,
    site_id,
    module_path,
    layer_index,
    layer_role,
    operation_name,
    collective_kind,
    local_tensor_shape_formula,
    local_tensor_dtype,
    producer,
    first_consumer,
    requires_replicated_result,
    packing_window,
    elimination_precondition,
    classification,
    site_role,
):
    return {
        "site_id": site_id,
        "module_path": module_path,
        "layer_index": layer_index,
        "layer_role": layer_role,
        "operation_name": operation_name,
        "collective_kind": collective_kind,
        "process_group": "tensor_parallel",
        "expected_calls_per_decode_step": 1,
        "local_tensor_shape_formula": local_tensor_shape_formula,
        "local_tensor_dtype": local_tensor_dtype,
        "producer": producer,
        "first_consumer": first_consumer,
        "requires_replicated_result": requires_replicated_result,
        "packing_window": packing_window,
        "elimination_precondition": elimination_precondition,
        "classification": classification,
        "site_role": site_role,
    }


def build_qwen38_static_collective_catalog(
    text_profile,
    *,
    tensor_parallel_size: int,
    layer_roles=None,
):
    for name, expected in EXPECTED_PROFILE.items():
        actual = _field(text_profile, name)
        if actual != expected:
            raise ValueError(
                f"{name} must be {expected!r}, got {actual!r}"
            )
    if tensor_parallel_size != EXPECTED_TENSOR_PARALLEL_SIZE:
        raise ValueError("tensor_parallel_size must be 4")
    if layer_roles is None:
        layer_roles = tuple(
            "full_attention"
            if layer_index % 4 == 3
            else "linear_attention"
            for layer_index in range(
                EXPECTED_PROFILE["num_hidden_layers"]
            )
        )
    if (
        not isinstance(layer_roles, (list, tuple))
        or len(layer_roles) != EXPECTED_PROFILE["num_hidden_layers"]
        or any(
            layer_role
            not in ("linear_attention", "full_attention")
            for layer_role in layer_roles
        )
    ):
        raise ValueError(
            "layer_roles must describe all 64 attention layers"
        )

    rows = [
        _catalog_row(
            site_id="embedding.input",
            module_path="model.embed_tokens",
            layer_index=None,
            layer_role="embedding",
            operation_name="vocab_parallel_embedding_all_reduce",
            collective_kind="all_reduce",
            local_tensor_shape_formula="[active_tokens, hidden_size]",
            local_tensor_dtype="torch.bfloat16",
            producer="rank-local vocabulary embedding lookup",
            first_consumer="decoder layer 0 input normalization",
            requires_replicated_result=True,
            packing_window=None,
            elimination_precondition=(
                "replicate the complete embedding on every TP rank"
            ),
            classification="MATERIALIZATION_ALTERNATIVE",
            site_role="vocab_parallel_embedding",
        )
    ]
    for layer_index, layer_role in enumerate(layer_roles):
        rows.append(
            _catalog_row(
                site_id=(
                    f"layer.{layer_index:03d}.attention.output"
                ),
                module_path=(
                    f"model.layers.{layer_index}.self_attn.o_proj"
                ),
                layer_index=layer_index,
                layer_role=layer_role,
                operation_name="row_parallel_all_reduce",
                collective_kind="all_reduce",
                local_tensor_shape_formula=(
                    "[active_tokens, hidden_size]"
                ),
                local_tensor_dtype="torch.float32",
                producer="row-parallel attention output projection",
                first_consumer="attention residual addition",
                requires_replicated_result=True,
                packing_window=None,
                elimination_precondition=(
                    "qualified peer-reduction residual-fusion group"
                ),
                classification="MATERIALIZATION_ALTERNATIVE",
                site_role="row_parallel_attention_output",
            )
        )
    rows.append(
        _catalog_row(
            site_id="sampling.greedy_token",
            module_path="engine.tensor_parallel_greedy",
            layer_index=None,
            layer_role="output_head",
            operation_name="greedy_token_broadcast",
            collective_kind="broadcast",
            local_tensor_shape_formula="[active_sequences]",
            local_tensor_dtype="torch.int64",
            producer="rank-0 exact-full-vocabulary argmax",
            first_consumer="next-token input on every TP rank",
            requires_replicated_result=True,
            packing_window=None,
            elimination_precondition=None,
            classification="MANDATORY_IMMEDIATE_CONSUMER",
            site_role="greedy_token_broadcast",
        )
    )
    if len(rows) != EXPECTED_DECODE_COLLECTIVE_COUNT:
        raise AssertionError("Qwen3.8 collective catalog size mismatch")
    return tuple(rows)


def _collective_signature(row):
    return (
        row.get("site_id"),
        row.get("collective_kind"),
        tuple(row.get("tensor_shape", ())),
        row.get("tensor_dtype"),
        row.get("tensor_bytes"),
    )


def validate_collective_census(rows, catalog):
    if not isinstance(rows, (list, tuple)) or len(rows) != 4:
        raise ValueError("collective census requires exactly four ranks")
    expected_site_ids = tuple(row["site_id"] for row in catalog)
    if len(expected_site_ids) != EXPECTED_DECODE_COLLECTIVE_COUNT:
        raise ValueError("catalog collective sequence is incomplete")
    if len(set(expected_site_ids)) != len(expected_site_ids):
        raise ValueError("catalog collective sequence contains duplicates")

    snapshots = {}
    reference = None
    reference_bytes = None
    for snapshot in rows:
        if (
            not isinstance(snapshot, dict)
            or snapshot.get("schema")
            != "tinyllmforge.synchronous-collective-census.v1"
            or snapshot.get("enabled") is not True
            or snapshot.get("finalization_status") != "complete"
        ):
            raise ValueError("invalid collective census snapshot")
        rank = snapshot.get("rank")
        if (
            type(rank) is not int
            or rank not in range(4)
            or rank in snapshots
        ):
            raise ValueError("collective census rank inventory mismatch")
        collectives = snapshot.get("collectives")
        steps = snapshot.get("steps")
        if (
            not isinstance(collectives, list)
            or not isinstance(steps, list)
            or not steps
        ):
            raise ValueError("collective sequence must be a list")
        decode_ordinals = tuple(range(len(steps)))
        if tuple(
            step.get("decode_ordinal") for step in steps
        ) != decode_ordinals or any(
            step.get("collective_count")
            != EXPECTED_DECODE_COLLECTIVE_COUNT
            or step.get("status", "completed") != "completed"
            for step in steps
        ):
            raise ValueError("collective step inventory is invalid")
        by_step = {ordinal: [] for ordinal in decode_ordinals}
        for row in collectives:
            decode_ordinal = row.get("decode_ordinal")
            if decode_ordinal not in by_step:
                raise ValueError(
                    "collective sequence decode ordinal is invalid"
                )
            by_step[decode_ordinal].append(row)
        signature = []
        byte_totals = []
        for decode_ordinal in decode_ordinals:
            step_rows = by_step[decode_ordinal]
            site_ids = tuple(row.get("site_id") for row in step_rows)
            if site_ids != expected_site_ids:
                raise ValueError(
                    "collective sequence does not match static catalog"
                )
            if tuple(
                row.get("collective_ordinal") for row in step_rows
            ) != tuple(range(EXPECTED_DECODE_COLLECTIVE_COUNT)):
                raise ValueError(
                    "collective sequence ordinals are invalid"
                )
            step_signature = tuple(
                _collective_signature(row) for row in step_rows
            )
            signature.append(step_signature)
            byte_totals.append(sum(
                row_signature[-1]
                for row_signature in step_signature
            ))
        signature = tuple(signature)
        byte_totals = tuple(byte_totals)
        if reference is None:
            reference = signature
            reference_bytes = byte_totals
        elif signature != reference:
            raise ValueError("rank collective sequence mismatch")
        snapshots[rank] = snapshot
    if tuple(sorted(snapshots)) != (0, 1, 2, 3):
        raise ValueError("collective census rank inventory mismatch")
    return {
        "coverage_complete": True,
        "rank_inventory": [0, 1, 2, 3],
        "decode_step_count_per_rank": len(reference),
        "collective_count_per_decode_step": (
            EXPECTED_DECODE_COLLECTIVE_COUNT
        ),
        "collective_count_per_rank": (
            len(reference) * EXPECTED_DECODE_COLLECTIVE_COUNT
        ),
        "tensor_bytes_per_decode_step": list(reference_bytes),
        "tensor_bytes_per_rank": sum(reference_bytes),
        "ordered_site_ids": list(expected_site_ids),
    }


def select_event_budget(calibration_rows):
    by_budget = {}
    for row in calibration_rows:
        if not isinstance(row, dict):
            raise ValueError("calibration row must be a dict")
        budget = row.get("budget")
        if budget not in EVENT_BUDGETS or budget in by_budget:
            raise ValueError("calibration event budget inventory mismatch")
        median_ratio = float(row.get("median_overhead_ratio"))
        maximum_ratio = float(row.get("maximum_overhead_ratio"))
        if (
            not math.isfinite(median_ratio)
            or not math.isfinite(maximum_ratio)
            or median_ratio < 0
            or maximum_ratio < 0
            or maximum_ratio < median_ratio
        ):
            raise ValueError("invalid calibration overhead ratios")
        by_budget[budget] = {
            "median": median_ratio,
            "maximum": maximum_ratio,
        }
    if set(by_budget) != set(EVENT_BUDGETS):
        raise ValueError("calibration event budget inventory mismatch")

    def passes(budget):
        row = by_budget[budget]
        return (
            row["median"] <= MEDIAN_OVERHEAD_LIMIT
            and row["maximum"] <= MAXIMUM_OVERHEAD_LIMIT
        )

    if not passes(0):
        return None
    passing = [budget for budget in EVENT_BUDGETS[1:] if passes(budget)]
    return max(passing) if passing else None


def build_consumer_dependency_proofs(catalog):
    proofs = []
    embedding_bytes = (
        EXPECTED_PROFILE["vocab_size"]
        * EXPECTED_PROFILE["hidden_size"]
        * 2
    )
    additional_embedding_bytes = (
        embedding_bytes
        * (EXPECTED_TENSOR_PARALLEL_SIZE - 1)
        // EXPECTED_TENSOR_PARALLEL_SIZE
    )
    for row in catalog:
        if row["site_role"] == "vocab_parallel_embedding":
            proofs.append({
                "candidate_id": "replicate_embedding",
                "site_id": row["site_id"],
                "site_role": row["site_role"],
                "status": "PASS",
                "reason": (
                    "every rank can perform the same embedding lookup "
                    "after full-table replication"
                ),
                "calls_removed_per_decode_step": 1,
                "bytes_removed_per_decode_step": None,
                "replacement_cost_ns": 0,
                "additional_persistent_device_bytes_per_rank": (
                    additional_embedding_bytes
                ),
                "additional_peak_device_bytes_per_rank": (
                    additional_embedding_bytes
                ),
                "unsupported_topologies": ["tensor_parallel_size!=4"],
            })
        else:
            proofs.append({
                "candidate_id": f"retain:{row['site_id']}",
                "site_id": row["site_id"],
                "site_role": row["site_role"],
                "status": "FAIL_IMMEDIATE_CONSUMER",
                "reason": row["first_consumer"],
                "calls_removed_per_decode_step": 0,
                "bytes_removed_per_decode_step": 0,
                "replacement_cost_ns": 0,
                "additional_persistent_device_bytes_per_rank": 0,
                "additional_peak_device_bytes_per_rank": 0,
                "unsupported_topologies": [],
            })
    return tuple(proofs)


def estimate_reduction_ceiling(census, timing, proofs, online):
    if census.get("coverage_complete") is not True:
        raise ValueError("collective census coverage is incomplete")
    median_tpot_ns = int(online.get("median_tpot_ns", 0))
    if median_tpot_ns <= 0:
        raise ValueError("median_tpot_ns must be positive")
    workloads = list(online.get("workloads", ()))
    candidates = []
    for proof in proofs:
        if proof.get("status") != "PASS":
            continue
        candidate_id = proof["candidate_id"]
        sample = timing.get(candidate_id)
        if not isinstance(sample, dict):
            continue
        sampled_ns = max(
            0,
            int(sample.get("sampled_collective_cuda_ns", 0)),
        )
        replacement_ns = max(
            0,
            int(proof.get("replacement_cost_ns", 0)),
        )
        uncertainty_ns = max(
            0,
            int(sample.get("profiler_uncertainty_ns", 0)),
        )
        lower_ns = max(
            0,
            sampled_ns - replacement_ns - uncertainty_ns,
        )
        candidates.append({
            "candidate_id": candidate_id,
            "site_role": proof["site_role"],
            "proof_status": proof["status"],
            "calls_removed_per_decode_step": (
                proof["calls_removed_per_decode_step"]
            ),
            "bytes_removed_per_decode_step": (
                proof["bytes_removed_per_decode_step"]
            ),
            "sampled_collective_cuda_ns": sampled_ns,
            "replacement_cost_ns": replacement_ns,
            "profiler_uncertainty_ns": uncertainty_ns,
            "additional_persistent_device_bytes_per_rank": (
                proof[
                    "additional_persistent_device_bytes_per_rank"
                ]
            ),
            "additional_peak_device_bytes_per_rank": (
                proof["additional_peak_device_bytes_per_rank"]
            ),
            "lower_bound_tpot_reduction_ns": lower_ns,
            "upper_bound_tpot_reduction_ns": sampled_ns,
            "lower_bound_tpot_opportunity_ratio": (
                lower_ns / median_tpot_ns
            ),
            "upper_bound_tpot_opportunity_ratio": (
                sampled_ns / median_tpot_ns
            ),
            "affected_workloads": workloads,
            "unsupported_topologies": list(
                proof.get("unsupported_topologies", ())
            ),
        })
    return {
        "median_tpot_ns": median_tpot_ns,
        "candidates": candidates,
    }


def classify_collective_reduction(summary):
    if summary.get("correctness_pass") is not True:
        return "INVALID_CORRECTNESS"
    if summary.get("resource_identity_pass") is not True:
        return "INVALID_RESOURCE_IDENTITY"
    if summary.get("coverage_complete") is not True:
        return "INCONCLUSIVE_INCOMPLETE_COVERAGE"
    if summary.get("profiler_overhead_pass") is not True:
        return "INCONCLUSIVE_PROFILER_OVERHEAD"
    for candidate in summary.get("candidates", ()):
        if (
            candidate.get("proof_status") == "PASS"
            and float(
                candidate.get(
                    "lower_bound_tpot_opportunity_ratio",
                    0.0,
                )
            )
            >= MINIMUM_OPPORTUNITY_RATIO
        ):
            return "GO_SYNC_COLLECTIVE_REDUCTION"
    return "NO_GO_NO_REDUCIBLE_COLLECTIVE"
