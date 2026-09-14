#!/usr/bin/env python3
"""Classify the source-bound KV8 + Quest amortization policy gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


EXPECTED_CONTEXT = 8192
EXPECTED_BATCHES = (4, 6, 8, 10, 12, 16, 19)
EXPECTED_DEPTHS = (0.0, 0.25, 0.5, 0.75, 1.0)
EXPECTED_SAMPLES = 24
FALLBACK_BATCHES = frozenset((4, 6))
ACTIVE_BATCHES = frozenset((8, 10, 12, 16, 19))
MAX_REGRESSION = 0.03
MIN_WALL_RECOVERY = 0.50
MAX_OVERALL_QUALITY_LOSS_PP = 5.0
MAX_DEPTH_QUALITY_LOSS_PP = 20.0


def _load(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _row_map(payload, arm, failures):
    rows = {}
    for row in payload.get("rows", []):
        key = (
            row.get("context_length"),
            row.get("batch"),
        )
        if key in rows:
            failures.append(
                f"{arm}_duplicate_cell_{key[0]}x{key[1]}"
            )
        rows[key] = row
    for batch in EXPECTED_BATCHES:
        key = (EXPECTED_CONTEXT, batch)
        if key not in rows:
            failures.append(
                f"{arm}_missing_cell_{EXPECTED_CONTEXT}x{batch}"
            )
    return rows


def _validate_common_identity(
    performance,
    provenance,
    quality,
    quality_provenance,
    failures,
):
    expected_arms = {"bf16", "kv8", "fixed", "adaptive"}
    if set(performance) != expected_arms:
        failures.append("performance_arm_set_mismatch")
        return
    if set(provenance) != expected_arms:
        failures.append("performance_provenance_arm_set_mismatch")
        return

    revisions = {
        item.get("source_revision")
        for item in provenance.values()
    }
    revisions.add(quality_provenance.get("source_revision"))
    if len(revisions) != 1 or None in revisions:
        failures.append("source_revision_mismatch")
    for arm, item in provenance.items():
        if int(item.get("tinyvllm_dirty_paths", -1)) != 0:
            failures.append(f"{arm}_source_dirty")
    if int(quality_provenance.get("source_dirty_paths", -1)) != 0:
        failures.append("quality_source_dirty")

    expected_configs = {
        "bf16": (0, -1, 0),
        "kv8": (8, -1, 0),
        "fixed": (8, 16, 0),
        "adaptive": (8, 16, 128),
    }
    common_keys = (
        "model_path",
        "enforce_eager",
        "gpu_memory_utilization",
        "kv_blocks_requested",
        "seed",
        "warmup_steps",
        "measured_steps",
    )
    reference = performance["bf16"].get("configuration", {})
    for arm, payload in performance.items():
        config = payload.get("configuration", {})
        for key in common_keys:
            if config.get(key) != reference.get(key):
                failures.append(f"{arm}_{key}_mismatch")
        if payload.get("grid_spec") != (
            "8192:4,6,8,10,12,16,19"
        ):
            failures.append(f"{arm}_grid_mismatch")
        bits, top_k, threshold = expected_configs[arm]
        if config.get("kv_quant_bits") != bits:
            failures.append(f"{arm}_kv_quant_bits_mismatch")
        if config.get("quest_top_k_blocks") != top_k:
            failures.append(f"{arm}_quest_top_k_mismatch")
        if config.get("quest_min_saved_blocks") != threshold:
            failures.append(
                f"{arm}_quest_min_saved_blocks_mismatch"
            )
        engines = payload.get("engines", [])
        if len(engines) != 1:
            failures.append(f"{arm}_engine_count_mismatch")
            continue
        identity = engines[0].get("identity") or {}
        for key, expected in (
            ("enforce_eager", True),
            ("kv_blocks_requested", 640),
            ("num_kvcache_blocks", 640),
            ("kv_quant_bits", bits),
            ("quest_top_k_blocks", top_k),
            ("quest_min_seq_len", 512),
            ("quest_min_saved_blocks", threshold),
        ):
            if identity.get(key) != expected:
                failures.append(f"{arm}_engine_{key}_mismatch")

    args = quality.get("args", {})
    for key, expected in (
        ("model", quality_provenance.get("model")),
        ("context_lens", [EXPECTED_CONTEXT]),
        ("depths", list(EXPECTED_DEPTHS)),
        ("num_trials", 5),
        ("fixed_prompts", True),
        ("kv_quant_bits", 8),
        ("quest_min_seq_len", 512),
        ("quest_min_saved_blocks", 128),
        ("top_k_blocks_list", [-1, 16]),
    ):
        if args.get(key) != expected:
            failures.append(f"quality_{key}_mismatch")


def _event_is_policy_consistent(event):
    if event.get("status") != "valid":
        return False
    lengths = event.get("sequence_lengths")
    blocks = event.get("sequence_block_counts")
    batch = event.get("batch_size")
    if (
        not isinstance(lengths, (list, tuple))
        or not isinstance(blocks, (list, tuple))
        or len(lengths) != len(blocks)
        or len(lengths) != batch
    ):
        return False
    requested = event.get("requested_top_k")
    threshold = event.get("min_saved_blocks")
    if requested != 16 or threshold != 128:
        return False
    saved = sum(
        max(0, int(block_count) - requested)
        for block_count in blocks
    )
    if event.get("saved_blocks") != saved:
        return False
    expected_active = saved >= threshold
    return (
        event.get("reason")
        == ("active" if expected_active else "below_saved_blocks")
        and event.get("resolved_top_k")
        == (requested if expected_active else -1)
    )


def _validate_performance_row(
    arm,
    row,
    batch,
    failures,
):
    prefix = f"{arm}_{EXPECTED_CONTEXT}x{batch}"
    if not row.get("measured"):
        failures.append(f"{prefix}_not_measured")
    if not row.get("observed_batch_is_stable"):
        failures.append(f"{prefix}_unstable_batch")
    if not row.get("prefill_tokens_match"):
        failures.append(f"{prefix}_prefill_mismatch")
    step = row.get("step") or {}
    if step.get("count") != EXPECTED_SAMPLES:
        failures.append(f"{prefix}_sample_count_mismatch")
    dispatch = row.get("dispatch_measured") or {}
    if (
        dispatch.get("steps") != EXPECTED_SAMPLES
        or dispatch.get("graph_steps") != 0
    ):
        failures.append(f"{prefix}_dispatch_mismatch")

    events = row.get("quest_activation_measured_events")
    if not isinstance(events, list) or len(events) != EXPECTED_SAMPLES:
        failures.append(f"{prefix}_activation_event_count_mismatch")
        return
    observation_ids = [
        event.get("observation_id") for event in events
    ]
    if (
        any(value is None for value in observation_ids)
        or observation_ids != sorted(set(observation_ids))
    ):
        failures.append(f"{prefix}_activation_observation_ids_invalid")

    if arm != "adaptive":
        return
    if not all(_event_is_policy_consistent(event) for event in events):
        failures.append(
            f"adaptive_activation_mismatch_{EXPECTED_CONTEXT}x{batch}"
        )
        return
    should_activate = batch in ACTIVE_BATCHES
    if not all(
        (event["resolved_top_k"] == 16) == should_activate
        for event in events
    ):
        failures.append(
            f"adaptive_activation_mismatch_{EXPECTED_CONTEXT}x{batch}"
        )


def _normalised_answer(value):
    if value is None:
        return None
    return str(value).strip()


def _quality_setting(payload, top_k, failures):
    matches = [
        setting
        for setting in payload.get("results", [])
        if setting.get("top_k") == top_k
    ]
    if len(matches) != 1:
        failures.append(f"quality_top_k_{top_k}_setting_count")
        return None
    return matches[0]


def _quality_rows(setting):
    rows = {}
    for detail in setting.get("details", []):
        key = (
            detail.get("ctx_len"),
            detail.get("depth"),
            detail.get("trial"),
            detail.get("magic"),
        )
        rows[key] = detail
    return rows


def _accuracy(rows):
    hits = [
        _normalised_answer(row.get("answer"))
        == str(row.get("magic"))
        for row in rows.values()
    ]
    return sum(hits) / len(hits) if hits else 0.0


def _depth_accuracy(rows):
    result = {}
    for depth in EXPECTED_DEPTHS:
        selected = [
            row
            for row in rows.values()
            if row.get("depth") == depth
        ]
        result[depth] = (
            sum(
                _normalised_answer(row.get("answer"))
                == str(row.get("magic"))
                for row in selected
            )
            / len(selected)
            if selected
            else 0.0
        )
    return result


def _validate_quality_activation(setting, failures):
    activation = setting.get("quest_activation")
    if not isinstance(activation, dict):
        failures.append("adaptive_quality_activation_missing")
        return
    events = activation.get("events")
    if (
        not isinstance(events, list)
        or activation.get("events_dropped") != 0
        or len(events) != activation.get("steps")
    ):
        failures.append("adaptive_quality_activation_events_incomplete")
        return
    if not events or not any(
        event.get("reason") == "active"
        for event in events
    ):
        failures.append("adaptive_quality_has_no_active_step")
    if not all(_event_is_policy_consistent(event) for event in events):
        failures.append("adaptive_quality_activation_mismatch")


def build_report(
    performance,
    provenance,
    quality,
    quality_provenance,
):
    identity_failures = []
    threshold_failures = []
    _validate_common_identity(
        performance,
        provenance,
        quality,
        quality_provenance,
        identity_failures,
    )

    row_maps = {}
    for arm, payload in performance.items():
        row_maps[arm] = _row_map(
            payload,
            arm,
            identity_failures,
        )
        for batch in EXPECTED_BATCHES:
            row = row_maps[arm].get((EXPECTED_CONTEXT, batch))
            if row is not None:
                _validate_performance_row(
                    arm,
                    row,
                    batch,
                    identity_failures,
                )

    performance_cells = []
    if not identity_failures:
        for batch in EXPECTED_BATCHES:
            bf16_ms = row_maps["bf16"][
                (EXPECTED_CONTEXT, batch)
            ]["step"]["median_ms"]
            kv8_ms = row_maps["kv8"][
                (EXPECTED_CONTEXT, batch)
            ]["step"]["median_ms"]
            fixed_ms = row_maps["fixed"][
                (EXPECTED_CONTEXT, batch)
            ]["step"]["median_ms"]
            adaptive_ms = row_maps["adaptive"][
                (EXPECTED_CONTEXT, batch)
            ]["step"]["median_ms"]
            row = {
                "context_length": EXPECTED_CONTEXT,
                "batch": batch,
                "bf16_step_ms": bf16_ms,
                "kv8_step_ms": kv8_ms,
                "fixed_quest_step_ms": fixed_ms,
                "adaptive_step_ms": adaptive_ms,
                "adaptive_vs_kv8_ratio": adaptive_ms / kv8_ms,
                "adaptive_vs_fixed_ratio": adaptive_ms / fixed_ms,
            }
            performance_cells.append(row)
            if (
                batch in FALLBACK_BATCHES
                and adaptive_ms / kv8_ms - 1.0 > MAX_REGRESSION
            ):
                threshold_failures.append(
                    "fallback_regression_over_3pct_"
                    f"{EXPECTED_CONTEXT}x{batch}"
                )
            if (
                batch in ACTIVE_BATCHES
                and adaptive_ms / fixed_ms - 1.0 > MAX_REGRESSION
            ):
                threshold_failures.append(
                    "active_regression_over_3pct_"
                    f"{EXPECTED_CONTEXT}x{batch}"
                )
        wall = performance_cells[-1]
        denominator = wall["kv8_step_ms"] - wall["bf16_step_ms"]
        wall_recovery = (
            (wall["kv8_step_ms"] - wall["adaptive_step_ms"])
            / denominator
            if denominator > 0
            else None
        )
        if (
            wall_recovery is None
            or wall_recovery < MIN_WALL_RECOVERY
        ):
            threshold_failures.append(
                "wall_excess_latency_recovery_below_50pct"
            )
    else:
        wall_recovery = None

    full_setting = _quality_setting(
        quality,
        -1,
        identity_failures,
    )
    adaptive_setting = _quality_setting(
        quality,
        16,
        identity_failures,
    )
    quality_report = {}
    if full_setting is not None and adaptive_setting is not None:
        full_rows = _quality_rows(full_setting)
        adaptive_rows = _quality_rows(adaptive_setting)
        expected_count = len(EXPECTED_DEPTHS) * 5
        if len(full_rows) != expected_count:
            identity_failures.append("quality_full_case_count_mismatch")
        if len(adaptive_rows) != expected_count:
            identity_failures.append(
                "quality_adaptive_case_count_mismatch"
            )
        if set(full_rows) != set(adaptive_rows):
            identity_failures.append("quality_pairing_mismatch")
        _validate_quality_activation(
            adaptive_setting,
            identity_failures,
        )
        if set(full_rows) == set(adaptive_rows) and full_rows:
            full_accuracy = _accuracy(full_rows)
            adaptive_accuracy = _accuracy(adaptive_rows)
            full_by_depth = _depth_accuracy(full_rows)
            adaptive_by_depth = _depth_accuracy(adaptive_rows)
            overall_delta_pp = (
                adaptive_accuracy - full_accuracy
            ) * 100.0
            depth_rows = []
            for depth in EXPECTED_DEPTHS:
                delta_pp = (
                    adaptive_by_depth[depth]
                    - full_by_depth[depth]
                ) * 100.0
                depth_rows.append(
                    {
                        "depth": depth,
                        "kv8_full_accuracy": full_by_depth[depth],
                        "adaptive_accuracy": adaptive_by_depth[depth],
                        "delta_pp": delta_pp,
                    }
                )
                if delta_pp < -MAX_DEPTH_QUALITY_LOSS_PP:
                    threshold_failures.append(
                        "depth_quality_loss_over_20pp_"
                        f"{depth}"
                    )
            if overall_delta_pp < -MAX_OVERALL_QUALITY_LOSS_PP:
                threshold_failures.append(
                    "overall_quality_loss_over_5pp"
                )
            quality_report = {
                "kv8_full_overall_accuracy": full_accuracy,
                "adaptive_overall_accuracy": adaptive_accuracy,
                "adaptive_vs_full_delta_pp": overall_delta_pp,
                "per_depth": depth_rows,
            }

    identity_failures = sorted(set(identity_failures))
    threshold_failures = sorted(set(threshold_failures))
    if identity_failures:
        classification = (
            "INCONCLUSIVE_KV8_QUEST_AMORTIZATION_POLICY"
        )
    elif threshold_failures:
        classification = "NO_GO_KV8_QUEST_AMORTIZATION_POLICY"
    else:
        classification = "GO_KV8_QUEST_AMORTIZATION_POLICY"

    return {
        "kind": "kvcapacity_kv8_quest_adaptive_gate",
        "classification": classification,
        "identity_failures": identity_failures,
        "threshold_failures": threshold_failures,
        "performance_cells": performance_cells,
        "wall_excess_latency_recovery": wall_recovery,
        "quality": quality_report,
        "claim_boundary": (
            "Qwen3-8B, A100 80GB PCIe, TP1, eager decode, "
            "8192-token synthetic contexts, 640 KV blocks, "
            "fixed needle prompts; default remains disabled"
        ),
    }


def render_markdown(report):
    lines = [
        "# KV8 + Quest Amortization-Aware Activation Gate",
        "",
        f"Classification: `{report['classification']}`",
        "",
        "## Performance",
        "",
        "| Batch | bf16 ms | KV8 ms | Fixed Quest ms | Adaptive ms | Adaptive/KV8 | Adaptive/Fixed |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in report["performance_cells"]:
        lines.append(
            "| {batch} | {bf16_step_ms:.3f} | {kv8_step_ms:.3f} | "
            "{fixed_quest_step_ms:.3f} | {adaptive_step_ms:.3f} | "
            "{adaptive_vs_kv8_ratio:.3f}x | "
            "{adaptive_vs_fixed_ratio:.3f}x |".format(**row)
        )
    lines.extend(
        [
            "",
            "## Failures",
            "",
            "- Identity: "
            + (
                ", ".join(report["identity_failures"])
                if report["identity_failures"]
                else "none"
            ),
            "- Threshold: "
            + (
                ", ".join(report["threshold_failures"])
                if report["threshold_failures"]
                else "none"
            ),
            "",
            "## Claim boundary",
            "",
            report["claim_boundary"],
            "",
        ]
    )
    return "\n".join(lines)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for arm in ("bf16", "kv8", "fixed", "adaptive"):
        parser.add_argument(
            f"--{arm}-performance",
            required=True,
        )
        parser.add_argument(
            f"--{arm}-provenance",
            required=True,
        )
    parser.add_argument("--quality", required=True)
    parser.add_argument("--quality-provenance", required=True)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--out-markdown", required=True)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    performance = {
        arm: _load(getattr(args, f"{arm}_performance"))
        for arm in ("bf16", "kv8", "fixed", "adaptive")
    }
    provenance = {
        arm: _load(getattr(args, f"{arm}_provenance"))
        for arm in ("bf16", "kv8", "fixed", "adaptive")
    }
    report = build_report(
        performance,
        provenance,
        _load(args.quality),
        _load(args.quality_provenance),
    )
    out_json = Path(args.out_json)
    out_markdown = Path(args.out_markdown)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_markdown.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    out_markdown.write_text(
        render_markdown(report),
        encoding="utf-8",
    )
    print(report["classification"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
