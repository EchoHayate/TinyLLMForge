from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics


SCHEMA_VERSION = "qwen35.tp4-decode-internal-case.v1"
SUMMARY_SCHEMA_VERSION = "qwen35.tp4-decode-internal-summary.v1"
POLICIES = ("recompute", "exact_restore")
REPETITIONS = tuple(range(5))
RANKS = tuple(range(4))
GENERATED_TOKENS = 8
RELATIVE_REGRESSION = 1.03
FIRST_STEP_ABSOLUTE_NS = 2_000_000
STEADY_STEP_ABSOLUTE_NS = 1_000_000
COLLECTIVE_ABSOLUTE_NS = 500_000
NON_CUDA_ABSOLUTE_NS = 1_000_000


def _require_non_negative_integer(value, name):
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < 0
    ):
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _read_json(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _read_jsonl(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        return [
            json.loads(line)
            for line in handle
            if line.strip()
        ]


def _percentile(values, percentile):
    ordered = sorted(values)
    if not ordered:
        raise ValueError("percentile requires at least one value")
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * percentile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return round(
        ordered[lower]
        + (ordered[upper] - ordered[lower]) * fraction
    )


def _require_sha256(value, name):
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA256")
    return value


def validate_decode_profile(payload):
    if not isinstance(payload, dict):
        raise ValueError("decode profile must be an object")
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("decode profile schema version mismatch")
    if payload.get("variant") != "decode_internal":
        raise ValueError("decode profile variant mismatch")
    if payload.get("resource_policy") != "shared-low-utilization":
        raise ValueError("decode profile resource policy mismatch")
    if payload.get("exclusive") is not False:
        raise ValueError("decode profile exclusive flag mismatch")
    if payload.get("workload") != "w2_long_reuse":
        raise ValueError("decode profile workload mismatch")
    if payload.get("policy") not in POLICIES:
        raise ValueError("decode profile policy mismatch")
    if payload.get("phase") not in {"warmup", "measured", "nsys_replay"}:
        raise ValueError("decode profile phase mismatch")
    if payload.get("generated_tokens") != GENERATED_TOKENS:
        raise ValueError("decode profile generated tokens mismatch")
    if payload.get("units") != "nanoseconds":
        raise ValueError("decode profile units mismatch")
    if payload.get("finalization_status") != "complete":
        raise ValueError("decode profile finalization status mismatch")
    _require_sha256(
        payload.get("source_tree_sha256"),
        "source_tree_sha256",
    )
    _require_sha256(
        payload.get("workload_manifest_sha256"),
        "workload_manifest_sha256",
    )
    if payload.get("rank_inventory") != list(RANKS):
        raise ValueError("decode profile rank inventory mismatch")
    ranks = payload.get("ranks")
    if (
        not isinstance(ranks, list)
        or len(ranks) != len(RANKS)
    ):
        raise ValueError("decode profile ranks are incomplete")
    by_rank = {}
    reference_signature = None
    decode_ordinals = None
    for rank_payload in ranks:
        if not isinstance(rank_payload, dict):
            raise ValueError("decode profile rank row is invalid")
        rank = rank_payload.get("rank")
        if rank not in RANKS or rank in by_rank:
            raise ValueError("decode profile rank is invalid")
        steps = rank_payload.get("steps")
        collectives = rank_payload.get("collectives")
        if not isinstance(steps, list) or not steps:
            raise ValueError(f"rank {rank} steps are invalid")
        if not isinstance(collectives, list):
            raise ValueError(f"rank {rank} collectives are invalid")
        step_indices = []
        signature = []
        rank_decode_ordinals = []
        known_steps = {}
        for step in steps:
            if not isinstance(step, dict):
                raise ValueError(f"rank {rank} step is invalid")
            step_index = _require_non_negative_integer(
                step.get("step_index"),
                f"rank {rank} step_index",
            )
            if step_index in known_steps:
                raise ValueError(f"rank {rank} step is duplicate")
            step_indices.append(step_index)
            is_decode = step.get("is_decode")
            if not isinstance(is_decode, bool):
                raise ValueError(f"rank {rank} step decode flag is invalid")
            decode_ordinal = step.get("decode_ordinal")
            if is_decode:
                _require_non_negative_integer(
                    decode_ordinal,
                    f"rank {rank} decode ordinal",
                )
                rank_decode_ordinals.append(decode_ordinal)
            elif decode_ordinal is not None:
                raise ValueError(
                    f"rank {rank} prefill decode ordinal is invalid"
                )
            active_count = _require_non_negative_integer(
                step.get("active_sequence_count"),
                f"rank {rank} active sequence count",
            )
            request_digest = _require_sha256(
                step.get("request_set_sha256"),
                f"rank {rank} request set digest",
            )
            wall_ns = _require_non_negative_integer(
                step.get("wall_ns"),
                f"rank {rank} wall_ns",
            )
            cuda_ns = _require_non_negative_integer(
                step.get("cuda_ns"),
                f"rank {rank} cuda_ns",
            )
            upper_ns = _require_non_negative_integer(
                step.get("non_cuda_upper_bound_ns"),
                f"rank {rank} non_cuda_upper_bound_ns",
            )
            if upper_ns != max(0, wall_ns - cuda_ns):
                raise ValueError(
                    f"rank {rank} non-CUDA upper bound mismatch"
                )
            signature.append((
                step_index,
                is_decode,
                decode_ordinal,
                active_count,
                request_digest,
            ))
            known_steps[step_index] = step
        if step_indices != sorted(step_indices):
            raise ValueError(f"rank {rank} step order mismatch")
        request_groups = []
        current_digest = None
        current_ordinals = []
        for step in steps:
            if not step["is_decode"]:
                continue
            digest = step["request_set_sha256"]
            if digest != current_digest:
                if current_ordinals:
                    request_groups.append(current_ordinals)
                current_digest = digest
                current_ordinals = []
            current_ordinals.append(step["decode_ordinal"])
        if current_ordinals:
            request_groups.append(current_ordinals)
        if (
            not request_groups
            or any(
                group != list(range(GENERATED_TOKENS - 1))
                for group in request_groups
            )
        ):
            raise ValueError(
                f"rank {rank} request decode step inventory mismatch"
            )
        if reference_signature is None:
            reference_signature = signature
            decode_ordinals = list(range(GENERATED_TOKENS - 1))
            request_group_count = len(request_groups)
        elif signature != reference_signature:
            raise ValueError(
                f"rank {rank} step or request alignment mismatch"
            )
        for collective in collectives:
            if not isinstance(collective, dict):
                raise ValueError(f"rank {rank} collective is invalid")
            step_index = collective.get("step_index")
            step = known_steps.get(step_index)
            if step is None or not step["is_decode"]:
                raise ValueError(
                    f"rank {rank} collective references unknown step"
                )
            if collective.get("rank") != rank:
                raise ValueError(f"rank {rank} collective rank mismatch")
            if (
                collective.get("decode_ordinal")
                != step["decode_ordinal"]
            ):
                raise ValueError(
                    f"rank {rank} collective decode ordinal mismatch"
                )
            if not isinstance(collective.get("operation"), str):
                raise ValueError(
                    f"rank {rank} collective operation is invalid"
                )
            if not isinstance(collective.get("tensor_shape"), list):
                raise ValueError(
                    f"rank {rank} collective tensor shape is invalid"
                )
            if not isinstance(collective.get("tensor_dtype"), str):
                raise ValueError(
                    f"rank {rank} collective tensor dtype is invalid"
                )
            _require_non_negative_integer(
                collective.get("wall_ns"),
                f"rank {rank} collective wall_ns",
            )
            _require_non_negative_integer(
                collective.get("cuda_ns"),
                f"rank {rank} collective cuda_ns",
            )
        by_rank[rank] = rank_payload
    if tuple(sorted(by_rank)) != RANKS:
        raise ValueError("decode profile rank inventory is incomplete")
    result = dict(payload)
    result["ranks"] = [by_rank[rank] for rank in RANKS]
    result["rank_inventory"] = list(RANKS)
    result["decode_ordinals"] = list(decode_ordinals)
    result["request_group_count"] = request_group_count
    result["first_decode_ordinal"] = 0
    result["steady_decode_ordinals"] = list(decode_ordinals[1:])
    return result


def _case_id(policy, repetition):
    return f"w2_long_reuse__measured__r{repetition}__{policy}"


def _case_metrics(payload):
    ranks = payload["ranks"]
    reference_decode_steps = [
        step
        for step in ranks[0]["steps"]
        if step["is_decode"]
    ]
    by_step = []
    for reference in reference_decode_steps:
        ordinal = reference["decode_ordinal"]
        request_digest = reference["request_set_sha256"]
        rank_steps = [
            next(
                step
                for step in rank_payload["steps"]
                if step.get("decode_ordinal") == ordinal
                and step.get("request_set_sha256")
                == request_digest
            )
            for rank_payload in ranks
        ]
        rank_collective_ns = []
        rank_collective_counts = []
        for rank_payload in ranks:
            events = [
                event
                for event in rank_payload["collectives"]
                if event["decode_ordinal"] == ordinal
                and event["step_index"]
                == rank_steps[rank_payload["rank"]]["step_index"]
            ]
            rank_collective_counts.append(len(events))
            rank_collective_ns.append(sum(
                event["cuda_ns"] for event in events
            ))
        by_step.append({
            "decode_ordinal": ordinal,
            "wall_ns": max(step["wall_ns"] for step in rank_steps),
            "cuda_ns": max(step["cuda_ns"] for step in rank_steps),
            "non_cuda_upper_bound_ns": max(
                step["non_cuda_upper_bound_ns"]
                for step in rank_steps
            ),
            "collective_cuda_ns": max(rank_collective_ns),
            "collective_count": max(rank_collective_counts),
            "wall_imbalance_ns": (
                max(step["wall_ns"] for step in rank_steps)
                - min(step["wall_ns"] for step in rank_steps)
            ),
            "cuda_imbalance_ns": (
                max(step["cuda_ns"] for step in rank_steps)
                - min(step["cuda_ns"] for step in rank_steps)
            ),
        })
    first_rows = [
        row for row in by_step if row["decode_ordinal"] == 0
    ]
    steady = [
        row for row in by_step if row["decode_ordinal"] != 0
    ]
    return {
        "case_id": payload["case_id"],
        "policy": payload["policy"],
        "repetition": payload["repetition"],
        "first_wall_ns": round(statistics.median(
            row["wall_ns"] for row in first_rows
        )),
        "first_cuda_ns": round(statistics.median(
            row["cuda_ns"] for row in first_rows
        )),
        "first_collective_cuda_ns": round(statistics.median(
            row["collective_cuda_ns"] for row in first_rows
        )),
        "first_non_cuda_upper_bound_ns": (
            round(statistics.median(
                row["non_cuda_upper_bound_ns"]
                for row in first_rows
            ))
        ),
        "steady_wall_ns": round(statistics.median(
            row["wall_ns"] for row in steady
        )),
        "steady_wall_p90_ns": _percentile(
            [row["wall_ns"] for row in steady],
            0.9,
        ),
        "steady_cuda_ns": round(statistics.median(
            row["cuda_ns"] for row in steady
        )),
        "steady_cuda_p90_ns": _percentile(
            [row["cuda_ns"] for row in steady],
            0.9,
        ),
        "steady_collective_cuda_ns": round(statistics.median(
            row["collective_cuda_ns"] for row in steady
        )),
        "steady_collective_count": round(statistics.median(
            row["collective_count"] for row in steady
        )),
        "steady_non_cuda_upper_bound_ns": round(statistics.median(
            row["non_cuda_upper_bound_ns"] for row in steady
        )),
        "median_wall_imbalance_ns": round(statistics.median(
            row["wall_imbalance_ns"] for row in by_step
        )),
        "median_cuda_imbalance_ns": round(statistics.median(
            row["cuda_imbalance_ns"] for row in by_step
        )),
    }


def _median(rows, name):
    return round(statistics.median(row[name] for row in rows))


def _policy_summary(rows):
    return {
        "median_first_wall_ns": _median(rows, "first_wall_ns"),
        "median_first_cuda_ns": _median(rows, "first_cuda_ns"),
        "median_first_collective_cuda_ns": _median(
            rows,
            "first_collective_cuda_ns",
        ),
        "median_first_non_cuda_upper_bound_ns": _median(
            rows,
            "first_non_cuda_upper_bound_ns",
        ),
        "median_steady_wall_ns": _median(rows, "steady_wall_ns"),
        "median_steady_wall_p90_ns": _median(
            rows,
            "steady_wall_p90_ns",
        ),
        "median_steady_cuda_ns": _median(rows, "steady_cuda_ns"),
        "median_steady_cuda_p90_ns": _median(
            rows,
            "steady_cuda_p90_ns",
        ),
        "median_collective_cuda_ns": _median(
            rows,
            "steady_collective_cuda_ns",
        ),
        "median_collective_count": _median(
            rows,
            "steady_collective_count",
        ),
        "median_steady_non_cuda_upper_bound_ns": _median(
            rows,
            "steady_non_cuda_upper_bound_ns",
        ),
        "median_wall_imbalance_ns": _median(
            rows,
            "median_wall_imbalance_ns",
        ),
        "median_cuda_imbalance_ns": _median(
            rows,
            "median_cuda_imbalance_ns",
        ),
    }


def _direction_consistent(ratios):
    median_ratio = statistics.median(ratios)
    direction = 1 if median_ratio > 1 else -1 if median_ratio < 1 else 0
    count = sum(
        1
        for ratio in ratios
        if (1 if ratio > 1 else -1 if ratio < 1 else 0) == direction
    )
    return direction, count >= 4


def _material_regression(exact, recompute, absolute_floor):
    return (
        exact - recompute >= absolute_floor
        and exact / recompute >= RELATIVE_REGRESSION
    )


def select_representative_repetition(pairs):
    ratios = [row["steady_state_ratio"] for row in pairs]
    target = statistics.median(ratios)
    return min(
        pairs,
        key=lambda row: (
            abs(row["steady_state_ratio"] - target),
            row["repetition"],
        ),
    )["repetition"]


def _read_case_rows(path):
    rows = _read_jsonl(path)
    if len(rows) != 4:
        raise ValueError(f"case rows are incomplete: {path.parent.name}")
    for row in rows:
        if (
            row.get("generated_tokens") != GENERATED_TOKENS
            or len(row.get("output_token_ids", []))
            != GENERATED_TOKENS
        ):
            raise ValueError(
                f"case generated tokens mismatch: {path.parent.name}"
            )
    return rows


def aggregate_decode_profiles(root):
    root = Path(root)
    cases = {}
    outputs = {}
    for repetition in REPETITIONS:
        for policy in POLICIES:
            case_id = _case_id(policy, repetition)
            case_dir = root / case_id
            profile_path = case_dir / "decode_profile.json"
            rows_path = case_dir / "case_rows.jsonl"
            if not profile_path.is_file() or not rows_path.is_file():
                raise ValueError(
                    f"decode profile requires five paired repetitions: {case_id}"
                )
            payload = validate_decode_profile(
                _read_json(profile_path)
            )
            if (
                payload["case_id"] != case_id
                or payload["policy"] != policy
                or payload["phase"] != "measured"
                or payload["repetition"] != repetition
            ):
                raise ValueError(f"decode profile identity mismatch: {case_id}")
            cases[(policy, repetition)] = _case_metrics(payload)
            outputs[(policy, repetition)] = _read_case_rows(rows_path)
    pairs = []
    for repetition in REPETITIONS:
        recompute = cases[("recompute", repetition)]
        exact = cases[("exact_restore", repetition)]
        recompute_rows = outputs[("recompute", repetition)]
        exact_rows = outputs[("exact_restore", repetition)]
        for request_index, (left, right) in enumerate(
            zip(recompute_rows, exact_rows)
        ):
            if left["output_token_ids"] != right["output_token_ids"]:
                raise ValueError(
                    "output parity mismatch: "
                    f"repetition={repetition}, request={request_index}"
                )
        pairs.append({
            "repetition": repetition,
            "first_step_ratio": (
                exact["first_wall_ns"] / recompute["first_wall_ns"]
            ),
            "steady_state_ratio": (
                exact["steady_wall_ns"] / recompute["steady_wall_ns"]
            ),
            "collective_ratio": (
                exact["steady_collective_cuda_ns"]
                / recompute["steady_collective_cuda_ns"]
            ),
            "non_cuda_ratio": (
                exact["steady_non_cuda_upper_bound_ns"]
                / recompute["steady_non_cuda_upper_bound_ns"]
            ),
        })
    by_policy_rows = {
        policy: [
            cases[(policy, repetition)]
            for repetition in REPETITIONS
        ]
        for policy in POLICIES
    }
    by_policy = {
        policy: _policy_summary(rows)
        for policy, rows in by_policy_rows.items()
    }
    first_ratios = [row["first_step_ratio"] for row in pairs]
    steady_ratios = [row["steady_state_ratio"] for row in pairs]
    collective_ratios = [row["collective_ratio"] for row in pairs]
    non_cuda_ratios = [row["non_cuda_ratio"] for row in pairs]
    recompute = by_policy["recompute"]
    exact = by_policy["exact_restore"]
    directions = [
        _direction_consistent(values)
        for values in (
            first_ratios,
            steady_ratios,
            collective_ratios,
            non_cuda_ratios,
        )
    ]
    specific_consistency = all(
        direction > 0 and consistent
        for direction, consistent in directions
    )
    specific = []
    if _material_regression(
        exact["median_collective_cuda_ns"],
        recompute["median_collective_cuda_ns"],
        COLLECTIVE_ABSOLUTE_NS,
    ):
        specific.append("collective_regression")
    if _material_regression(
        exact["median_first_wall_ns"],
        recompute["median_first_wall_ns"],
        FIRST_STEP_ABSOLUTE_NS,
    ):
        specific.append("first_step_regression")
    if _material_regression(
        exact["median_steady_wall_ns"],
        recompute["median_steady_wall_ns"],
        STEADY_STEP_ABSOLUTE_NS,
    ):
        specific.append("steady_state_regression")
    if _material_regression(
        exact["median_steady_non_cuda_upper_bound_ns"],
        recompute["median_steady_non_cuda_upper_bound_ns"],
        NON_CUDA_ABSOLUTE_NS,
    ):
        specific.append("non_cuda_or_sync_upper_bound_regression")
    if not specific:
        classification = "no_material_decode_regression"
    elif not specific_consistency:
        classification = "mixed_or_inconclusive"
    else:
        classification = specific[0]
    return {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "generated_tokens": GENERATED_TOKENS,
        "measured_pairs": len(pairs),
        "paired_repetitions": pairs,
        "by_policy": by_policy,
        "first_step": {
            "paired_ratios": first_ratios,
            "paired_median_ratio": statistics.median(first_ratios),
            "ratio_of_medians": (
                exact["median_first_wall_ns"]
                / recompute["median_first_wall_ns"]
            ),
        },
        "steady_state": {
            "paired_ratios": steady_ratios,
            "paired_median_ratio": statistics.median(steady_ratios),
            "ratio_of_medians": (
                exact["median_steady_wall_ns"]
                / recompute["median_steady_wall_ns"]
            ),
        },
        "collective": {
            "paired_ratios": collective_ratios,
            "paired_median_ratio": statistics.median(collective_ratios),
            "ratio_of_medians": (
                exact["median_collective_cuda_ns"]
                / recompute["median_collective_cuda_ns"]
            ),
        },
        "non_cuda_upper_bound": {
            "paired_ratios": non_cuda_ratios,
            "paired_median_ratio": statistics.median(non_cuda_ratios),
            "ratio_of_medians": (
                exact["median_steady_non_cuda_upper_bound_ns"]
                / recompute["median_steady_non_cuda_upper_bound_ns"]
            ),
            "evidence_boundary": (
                "step wall minus CUDA interval is an upper bound that "
                "combines host orchestration, launch gaps, and possible "
                "synchronization waiting"
            ),
        },
        "representative_repetition": (
            select_representative_repetition(pairs)
        ),
        "classification": classification,
        "thresholds": {
            "relative_regression": RELATIVE_REGRESSION,
            "first_step_absolute_ns": FIRST_STEP_ABSOLUTE_NS,
            "steady_step_absolute_ns": STEADY_STEP_ABSOLUTE_NS,
            "collective_absolute_ns": COLLECTIVE_ABSOLUTE_NS,
            "non_cuda_absolute_ns": NON_CUDA_ABSOLUTE_NS,
        },
    }


def _atomic_write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    summary = aggregate_decode_profiles(args.input_root)
    _atomic_write_json(args.output, summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
