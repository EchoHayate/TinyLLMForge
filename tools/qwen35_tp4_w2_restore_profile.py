from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics


SCHEMA_VERSION = "qwen35.tp4-w2-restore-profile.v1"
CASE_SCHEMA_VERSION = (
    "qwen35.tp4-w2-restore-profile-case.v1"
)
POLICIES = ("recompute", "exact_restore")
REPETITIONS = tuple(range(5))


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


def _case_id(policy, repetition):
    return (
        f"w2_long_reuse__measured__r{repetition}__{policy}"
    )


def _prepared_request_ids(events):
    return {
        event.get("request_id")
        for event in events
        if event.get("name") == "restore_prepare"
        and event.get("status") == "ok"
        and event.get("request_id") is not None
    }


def _sum_events(events, name, *, request_ids=None, limit=None):
    matched = [
        event
        for event in events
        if event.get("name") == name
        and event.get("status") == "ok"
        and (
            request_ids is None
            or event.get("request_id") in request_ids
        )
    ]
    if limit is not None:
        matched = matched[-limit:]
    return sum(
        event["duration_ns"]
        for event in matched
    )


def _case_metrics(root, policy, repetition, generated_tokens):
    case_id = _case_id(policy, repetition)
    case_dir = root / case_id
    profile = _read_json(case_dir / "profile.json")
    rows = _read_jsonl(case_dir / "case_rows.jsonl")
    if (
        profile.get("schema_version") != CASE_SCHEMA_VERSION
        or profile.get("case_id") != case_id
        or profile.get("policy") != policy
        or profile.get("workload") != "w2_long_reuse"
        or profile.get("phase") != "measured"
        or profile.get("repetition") != repetition
        or not isinstance(profile.get("events"), list)
        or len(rows) != 4
    ):
        raise ValueError(f"invalid profile case: {case_id}")
    profile_generated_tokens = profile.get(
        "generated_tokens",
        64,
    )
    if (
        profile_generated_tokens != generated_tokens
        or (
            "generated_tokens" not in profile
            and generated_tokens != 64
        )
        or (
            "generated_tokens" in profile
            and profile.get("canonical_generated_tokens") != 64
        )
        or (
            "generated_tokens" in profile
            and profile.get("variant")
            != (
                "canonical_output"
                if generated_tokens == 64
                else "short_output"
            )
        )
    ):
        raise ValueError(
            f"profile generated tokens mismatch: {case_id}"
        )
    for row in rows:
        if (
            row.get("generated_tokens") != generated_tokens
            or len(row.get("output_token_ids", []))
            != generated_tokens
            or len(row.get("decode_step_ns", []))
            != max(generated_tokens - 1, 0)
        ):
            raise ValueError(
                f"row generated tokens mismatch: {case_id}"
            )
    prepared_request_ids = _prepared_request_ids(
        profile["events"]
    )
    makespan_ns = max(row["e2e_ns"] for row in rows)
    return {
        "case_id": case_id,
        "policy": policy,
        "repetition": repetition,
        "request_count": len(rows),
        "generated_tokens": generated_tokens,
        "makespan_ns": makespan_ns,
        "median_ttft_ns": int(statistics.median(
            row["ttft_ns"] for row in rows
        )),
        "median_decode_ns": int(statistics.median(
            sum(row["decode_step_ns"]) for row in rows
        )),
        "executed_prefill_tokens": sum(
            row["executed_prefill_tokens"] for row in rows
        ),
        "reused_kv_tokens": sum(
            row["reused_kv_tokens"] for row in rows
        ),
        "release_flush_ns": _sum_events(
            profile["events"],
            "release_flush",
            limit=len(prepared_request_ids),
        ),
        "restore_total_ns": _sum_events(
            profile["events"],
            "restore_total",
            request_ids=prepared_request_ids,
        ),
        "restore_prepare_ns": _sum_events(
            profile["events"],
            "restore_prepare",
        ),
        "restore_validate_ns": _sum_events(
            profile["events"],
            "restore_validate",
        ),
        "restore_commit_ns": _sum_events(
            profile["events"],
            "restore_commit",
        ),
        "restore_rollback_ns": _sum_events(
            profile["events"],
            "restore_rollback",
        ),
    }


def _median_summary(rows):
    fields = (
        "makespan_ns",
        "median_ttft_ns",
        "median_decode_ns",
        "executed_prefill_tokens",
        "reused_kv_tokens",
        "release_flush_ns",
        "restore_total_ns",
        "restore_prepare_ns",
        "restore_validate_ns",
        "restore_commit_ns",
        "restore_rollback_ns",
    )
    return {
        f"median_{field}": int(statistics.median(
            row[field] for row in rows
        ))
        for field in fields
    }


def aggregate_profile(root, *, generated_tokens=64):
    if (
        isinstance(generated_tokens, bool)
        or not isinstance(generated_tokens, int)
        or generated_tokens <= 0
        or generated_tokens > 64
    ):
        raise ValueError("generated tokens must be in [1, 64]")
    root = Path(root)
    cases = {}
    missing = []
    for repetition in REPETITIONS:
        for policy in POLICIES:
            case_id = _case_id(policy, repetition)
            case_dir = root / case_id
            if not (
                (case_dir / "profile.json").is_file()
                and (case_dir / "case_rows.jsonl").is_file()
            ):
                missing.append(case_id)
                continue
            cases[(policy, repetition)] = _case_metrics(
                root,
                policy,
                repetition,
                generated_tokens,
            )
    if missing or len(cases) != 10:
        raise ValueError(
            "profile requires five paired repetitions; "
            f"missing={missing}"
        )
    paired = []
    for repetition in REPETITIONS:
        recompute = cases[("recompute", repetition)]
        exact = cases[("exact_restore", repetition)]
        paired.append({
            "repetition": repetition,
            "recompute": recompute,
            "exact_restore": exact,
            "makespan_speedup": (
                recompute["makespan_ns"]
                / exact["makespan_ns"]
            ),
            "ttft_speedup": (
                recompute["median_ttft_ns"]
                / exact["median_ttft_ns"]
            ),
        })
    by_policy = {
        policy: [
            cases[(policy, repetition)]
            for repetition in REPETITIONS
        ]
        for policy in POLICIES
    }
    summary = {
        policy: _median_summary(by_policy[policy])
        for policy in POLICIES
    }
    recompute_summary = summary["recompute"]
    exact_summary = summary["exact_restore"]
    restore_total_ns = exact_summary[
        "median_restore_total_ns"
    ]
    paired_median_speedup = statistics.median(
        row["makespan_speedup"] for row in paired
    )
    ratio_of_medians_speedup = (
        recompute_summary["median_makespan_ns"]
        / exact_summary["median_makespan_ns"]
    )
    paired_direction = (
        1 if paired_median_speedup > 1.0
        else -1 if paired_median_speedup < 1.0
        else 0
    )
    ratio_direction = (
        1 if ratio_of_medians_speedup > 1.0
        else -1 if ratio_of_medians_speedup < 1.0
        else 0
    )
    direction_agreement = paired_direction == ratio_direction
    if not direction_agreement:
        makespan_classification = "inconclusive"
    elif paired_direction > 0:
        makespan_classification = "stable_speedup"
    elif paired_direction < 0:
        makespan_classification = "stable_slowdown"
    else:
        makespan_classification = "no_change"
    return {
        "schema_version": SCHEMA_VERSION,
        "workload": "w2_long_reuse",
        "generated_tokens": generated_tokens,
        "measured_repetitions": 5,
        "summary": summary,
        "comparison": {
            "prefill_token_reduction_fraction": (
                1.0
                - (
                    exact_summary[
                        "median_executed_prefill_tokens"
                    ]
                    / recompute_summary[
                        "median_executed_prefill_tokens"
                    ]
                )
            ),
            "median_makespan_speedup": paired_median_speedup,
            "ratio_of_median_makespans_speedup": (
                ratio_of_medians_speedup
            ),
            "makespan_speedup_direction_agreement": (
                direction_agreement
            ),
            "makespan_classification": (
                makespan_classification
            ),
            "median_ttft_speedup": statistics.median(
                row["ttft_speedup"] for row in paired
            ),
            "median_restore_total_per_request_ns": (
                restore_total_ns // 4
            ),
            "median_restore_prepare_share": (
                exact_summary["median_restore_prepare_ns"]
                / restore_total_ns
                if restore_total_ns
                else 0.0
            ),
            "median_exact_decode_share_of_makespan": (
                exact_summary["median_median_decode_ns"]
                / exact_summary["median_makespan_ns"]
            ),
        },
        "paired_repetitions": paired,
        "evidence_boundary": (
            "restore_prepare includes rank-local restore work plus "
            "acknowledgement transport and waiting"
        ),
    }


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--generated-tokens",
        type=int,
        default=64,
    )
    args = parser.parse_args(argv)
    result = aggregate_profile(
        args.input_root,
        generated_tokens=args.generated_tokens,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            result,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result["summary"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
