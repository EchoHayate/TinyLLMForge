#!/usr/bin/env python3
"""Independent classifier for the Qwen3-8B KV8 + Quest diagnostic gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


EXPECTED_BATCHES = (4, 8, 12, 16, 19)
EXPECTED_CELLS = {(8192, batch) for batch in EXPECTED_BATCHES}
EXPECTED_DEPTHS = (0.0, 0.25, 0.5, 0.75, 1.0)
MAX_OVERALL_QUALITY_LOSS_PP = 5.0
MAX_DEPTH_QUALITY_LOSS_PP = 20.0
MIN_WALL_EXCESS_RECOVERY = 0.50


def _load(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _perf_arm(name, payload, bits, top_k):
    failures = []
    config = payload.get("configuration", {})
    expected_config = {
        "enforce_eager": True,
        "gpu_memory_utilization": 0.85,
        "kv_blocks_requested": 640,
        "kv_quant_bits": bits,
        "measured_steps": 24,
        "quest_top_k_blocks": top_k,
        "seed": 20260913,
        "warmup_steps": 24,
    }
    for key, expected in expected_config.items():
        if config.get(key) != expected:
            failures.append(
                f"{name}_configuration_{key}_expected_{expected!r}_got_{config.get(key)!r}"
            )
    if config.get("model_path", "").rstrip("/").split("/")[-1] != "Qwen3-8B":
        failures.append(f"{name}_model_is_not_Qwen3-8B")
    if payload.get("grid_spec") != "8192:4,8,12,16,19":
        failures.append(f"{name}_grid_mismatch")

    engines = payload.get("engines", [])
    if len(engines) != 1:
        failures.append(f"{name}_expected_one_engine")
    else:
        identity = engines[0].get("identity", {})
        engine_expected = {
            "enforce_eager": True,
            "kv_blocks_requested": 640,
            "num_kvcache_blocks": 640,
            "kv_quant_bits": bits,
            "quest_top_k_blocks": top_k,
        }
        if top_k > 0:
            engine_expected["quest_min_seq_len"] = 512
        for key, expected in engine_expected.items():
            if identity.get(key) != expected:
                failures.append(
                    f"{name}_engine_{key}_expected_{expected!r}_got_{identity.get(key)!r}"
                )

    rows = {}
    for row in payload.get("rows", []):
        key = (row.get("context_length"), row.get("batch"))
        if key in rows:
            failures.append(f"{name}_duplicate_cell_{key}")
            continue
        rows[key] = row
    missing = sorted(EXPECTED_CELLS - set(rows))
    extra = sorted(set(rows) - EXPECTED_CELLS)
    if missing:
        failures.append(f"{name}_missing_cells_{missing}")
    if extra:
        failures.append(f"{name}_extra_cells_{extra}")

    measured = {}
    for key in sorted(EXPECTED_CELLS):
        row = rows.get(key)
        if row is None:
            continue
        if not row.get("measured") or not row.get("step"):
            failures.append(f"{name}_cell_{key}_not_measured")
            continue
        if not row.get("observed_batch_is_stable"):
            failures.append(f"{name}_cell_{key}_batch_not_stable")
        if not row.get("prefill_tokens_match"):
            failures.append(f"{name}_cell_{key}_prefill_mismatch")
        step = row["step"]
        if step.get("count") != 24:
            failures.append(f"{name}_cell_{key}_sample_count_not_24")
        dispatch = row.get("dispatch_measured", {})
        if (
            dispatch.get("all_graph") is not False
            or dispatch.get("graph_steps") != 0
            or dispatch.get("steps") != 24
        ):
            failures.append(f"{name}_cell_{key}_not_fully_eager")
        try:
            measured[key] = {
                "step_ms": float(step["median_ms"]),
                "prompt_digests": tuple(row["prompt_digests"]),
            }
        except (KeyError, TypeError, ValueError):
            failures.append(f"{name}_cell_{key}_invalid_measurement")
    return failures, measured


def _quality_setting(payload, expected_bits, expected_top_ks, name):
    failures = []
    args = payload.get("args", {})
    expected_args = {
        "context_lens": [8192],
        "depths": list(EXPECTED_DEPTHS),
        "enforce_eager": True,
        "fixed_prompts": True,
        "gpu_memory_utilization": 0.85,
        "kv_quant_bits": expected_bits,
        "max_model_len": 16384,
        "max_num_seqs": 32,
        "max_output_len": 16,
        "needle_style": "newline",
        "num_trials": 5,
        "quest_min_seq_len": 512,
        "seed": 0,
        "tp_size": 1,
        "top_k_blocks_list": list(expected_top_ks),
    }
    for key, expected in expected_args.items():
        if args.get(key) != expected:
            failures.append(
                f"{name}_quality_{key}_expected_{expected!r}_got_{args.get(key)!r}"
            )
    if args.get("model", "").rstrip("/").split("/")[-1] != "Qwen3-8B":
        failures.append(f"{name}_quality_model_is_not_Qwen3-8B")

    by_top_k = {}
    for result in payload.get("results", []):
        top_k = result.get("top_k")
        if top_k in by_top_k:
            failures.append(f"{name}_quality_duplicate_top_k_{top_k}")
        by_top_k[top_k] = result
    if set(by_top_k) != set(expected_top_ks):
        failures.append(
            f"{name}_quality_top_k_set_expected_{list(expected_top_ks)}_got_"
            f"{sorted(by_top_k, key=lambda value: repr(value))}"
        )

    settings = {}
    expected_case_coords = {
        (8192, depth, trial)
        for depth in EXPECTED_DEPTHS
        for trial in range(5)
    }
    for top_k in expected_top_ks:
        result = by_top_k.get(top_k)
        if result is None:
            continue
        details = result.get("details", [])
        cases = {}
        for row in details:
            coord = (row.get("ctx_len"), row.get("depth"), row.get("trial"))
            if coord in cases:
                failures.append(f"{name}_quality_top_k_{top_k}_duplicate_case_{coord}")
                continue
            if not str(row.get("raw") or "").strip():
                failures.append(f"{name}_quality_top_k_{top_k}_empty_output_{coord}")
            answer = row.get("answer")
            magic = row.get("magic")
            recomputed_hit = answer is not None and str(answer) == str(magic)
            if bool(row.get("hit")) != recomputed_hit:
                failures.append(
                    f"{name}_quality_top_k_{top_k}_producer_hit_mismatch_{coord}"
                )
            cases[coord] = {
                "magic": magic,
                "hit": recomputed_hit,
            }
        missing = sorted(expected_case_coords - set(cases))
        extra = sorted(set(cases) - expected_case_coords)
        if missing:
            failures.append(f"{name}_quality_top_k_{top_k}_missing_cases_{missing}")
        if extra:
            failures.append(f"{name}_quality_top_k_{top_k}_extra_cases_{extra}")
        overall = (
            sum(case["hit"] for case in cases.values()) / len(cases)
            if cases
            else None
        )
        per_depth = {}
        for depth in EXPECTED_DEPTHS:
            depth_cases = [
                case
                for (context, case_depth, _), case in cases.items()
                if context == 8192 and case_depth == depth
            ]
            per_depth[depth] = (
                sum(case["hit"] for case in depth_cases) / len(depth_cases)
                if depth_cases
                else None
            )
        settings[top_k] = {
            "cases": cases,
            "overall_accuracy": overall,
            "per_depth_accuracy": per_depth,
            "throughput_tok_s": result.get("throughput_tok_s"),
        }
    return failures, settings


def _pairing_failure(left, right):
    if set(left["cases"]) != set(right["cases"]):
        return True
    return any(
        left["cases"][coord]["magic"] != right["cases"][coord]["magic"]
        for coord in left["cases"]
    )


def _quality_provenance_failures(provenance):
    failures = []
    expected = {
        "source_dirty_paths": 0,
        "fixed_prompts": True,
        "needle_style": "newline",
        "context_lens": [8192],
        "depths": list(EXPECTED_DEPTHS),
        "num_trials": 5,
        "quest_top_k_blocks": 16,
        "quest_min_seq_len": 512,
    }
    for key, value in expected.items():
        if provenance.get(key) != value:
            failures.append(
                f"quality_provenance_{key}_expected_{value!r}_got_{provenance.get(key)!r}"
            )
    revision = provenance.get("source_revision")
    if (
        not isinstance(revision, str)
        or len(revision) != 40
        or any(character not in "0123456789abcdef" for character in revision)
    ):
        failures.append("quality_source_revision_invalid")
    if provenance.get("source_dirty_paths") != 0:
        failures.append("quality_source_dirty")
    if provenance.get("model", "").rstrip("/").split("/")[-1] != "Qwen3-8B":
        failures.append("quality_provenance_model_is_not_Qwen3-8B")
    remote_dir = provenance.get("remote_dir", "")
    if not remote_dir.startswith("/data00/home/sitian/tllm/kvcapacity-runs/"):
        failures.append("quality_remote_dir_outside_approved_root")
    return failures


def build_report(
    bf16_perf,
    kv8_perf,
    quest_perf,
    bf16_provenance,
    kv8_provenance,
    quest_provenance,
    bf16_quality,
    kv8_quality,
    quality_provenance,
):
    identity_failures = []
    threshold_failures = []

    perf_specs = (
        ("bf16", bf16_perf, bf16_provenance, 0, -1),
        ("kv8", kv8_perf, kv8_provenance, 8, -1),
        ("quest", quest_perf, quest_provenance, 8, 16),
    )
    perf_rows = {}
    for name, payload, provenance, bits, top_k in perf_specs:
        failures, rows = _perf_arm(name, payload, bits, top_k)
        identity_failures.extend(failures)
        perf_rows[name] = rows
        if provenance.get("tinyvllm_dirty_paths") != 0:
            identity_failures.append(f"{name}_performance_source_dirty")
        if provenance.get("mode") != "sweep":
            identity_failures.append(f"{name}_performance_mode_not_sweep")

    revisions = {item[2].get("source_revision") for item in perf_specs}
    if len(revisions) != 1 or None in revisions:
        identity_failures.append("performance_source_revision_mismatch")
    worker_hashes = {item[2].get("worker_sha256") for item in perf_specs}
    if len(worker_hashes) != 1 or None in worker_hashes:
        identity_failures.append("performance_worker_sha256_mismatch")

    shared_config_keys = (
        "model_path",
        "gpu_memory_utilization",
        "kv_blocks_requested",
        "measured_steps",
        "seed",
        "warmup_steps",
    )
    for key in shared_config_keys:
        values = {
            payload.get("configuration", {}).get(key)
            for _, payload, _, _, _ in perf_specs
        }
        if len(values) != 1:
            identity_failures.append(f"performance_{key}_mismatch")
    if len({payload.get("grid_spec") for _, payload, _, _, _ in perf_specs}) != 1:
        identity_failures.append("performance_grid_mismatch")

    common_cells = set.intersection(*(set(rows) for rows in perf_rows.values()))
    for cell in sorted(common_cells):
        digests = {perf_rows[name][cell]["prompt_digests"] for name in perf_rows}
        if len(digests) != 1:
            identity_failures.append(f"performance_prompt_mismatch_{cell}")

    quality_failures, bf16_settings = _quality_setting(
        bf16_quality, 0, (-1,), "bf16"
    )
    identity_failures.extend(quality_failures)
    quality_failures, kv8_settings = _quality_setting(
        kv8_quality, 8, (-1, 16), "kv8"
    )
    identity_failures.extend(quality_failures)
    identity_failures.extend(_quality_provenance_failures(quality_provenance))

    bf16_setting = bf16_settings.get(-1)
    kv8_setting = kv8_settings.get(-1)
    quest_setting = kv8_settings.get(16)
    if kv8_setting and quest_setting and _pairing_failure(kv8_setting, quest_setting):
        identity_failures.append("kv8_quality_cases_not_paired")
    if bf16_setting and kv8_setting and _pairing_failure(bf16_setting, kv8_setting):
        identity_failures.append("bf16_kv8_quality_cases_not_paired")

    performance_cells = []
    if not identity_failures:
        for context, batch in sorted(EXPECTED_CELLS):
            bf16_ms = perf_rows["bf16"][(context, batch)]["step_ms"]
            kv8_ms = perf_rows["kv8"][(context, batch)]["step_ms"]
            quest_ms = perf_rows["quest"][(context, batch)]["step_ms"]
            performance_cells.append(
                {
                    "context_length": context,
                    "batch": batch,
                    "bf16_step_ms": bf16_ms,
                    "kv8_step_ms": kv8_ms,
                    "quest_step_ms": quest_ms,
                    "quest_vs_kv8_ratio": quest_ms / kv8_ms,
                    "quest_faster_than_kv8": quest_ms < kv8_ms,
                }
            )
        if not all(cell["quest_faster_than_kv8"] for cell in performance_cells):
            threshold_failures.append("quest_not_faster_in_every_cell")

    wall_cell = None
    if performance_cells:
        wall = next(cell for cell in performance_cells if cell["batch"] == 19)
        denominator = wall["kv8_step_ms"] - wall["bf16_step_ms"]
        recovery = (
            (wall["kv8_step_ms"] - wall["quest_step_ms"]) / denominator
            if denominator > 0
            else None
        )
        wall_cell = {**wall, "excess_latency_recovery": recovery}
        if recovery is None or recovery < MIN_WALL_EXCESS_RECOVERY:
            threshold_failures.append("wall_excess_latency_recovery_below_50pct")

    quality = None
    if not identity_failures and kv8_setting and quest_setting:
        overall_delta_pp = (
            quest_setting["overall_accuracy"] - kv8_setting["overall_accuracy"]
        ) * 100.0
        per_depth = []
        for depth in EXPECTED_DEPTHS:
            full = kv8_setting["per_depth_accuracy"][depth]
            sparse = quest_setting["per_depth_accuracy"][depth]
            per_depth.append(
                {
                    "depth": depth,
                    "kv8_full_accuracy": full,
                    "kv8_quest_accuracy": sparse,
                    "delta_pp": (sparse - full) * 100.0,
                }
            )
        quality = {
            "bf16_overall_accuracy": bf16_setting["overall_accuracy"],
            "kv8_full_overall_accuracy": kv8_setting["overall_accuracy"],
            "kv8_quest_overall_accuracy": quest_setting["overall_accuracy"],
            "kv8_quest_vs_full_delta_pp": overall_delta_pp,
            "per_depth": per_depth,
            "throughput_tok_s": {
                "bf16": bf16_setting["throughput_tok_s"],
                "kv8_full": kv8_setting["throughput_tok_s"],
                "kv8_quest": quest_setting["throughput_tok_s"],
            },
        }
        if overall_delta_pp < -MAX_OVERALL_QUALITY_LOSS_PP - 1e-9:
            threshold_failures.append("overall_accuracy_loss_over_5pp")
        if any(
            row["delta_pp"] < -MAX_DEPTH_QUALITY_LOSS_PP - 1e-9
            for row in per_depth
        ):
            threshold_failures.append("per_depth_accuracy_loss_over_20pp")

    if identity_failures:
        classification = "INCONCLUSIVE"
    elif threshold_failures:
        classification = "NO_GO_KV8_QUEST"
    else:
        classification = "GO_TO_FUSED_KERNEL_SCOPE"

    return {
        "kind": "kvcapacity_kv8_quest_gate",
        "classification": classification,
        "identity_failures": identity_failures,
        "threshold_failures": threshold_failures,
        "quality_source_revision": quality_provenance.get("source_revision"),
        "performance_cells": performance_cells,
        "wall_cell": wall_cell,
        "quality": quality,
        "thresholds": {
            "quest_faster_in_every_cell": True,
            "minimum_wall_excess_latency_recovery": MIN_WALL_EXCESS_RECOVERY,
            "maximum_overall_quality_loss_pp": MAX_OVERALL_QUALITY_LOSS_PP,
            "maximum_per_depth_quality_loss_pp": MAX_DEPTH_QUALITY_LOSS_PP,
        },
        "claim_boundary": (
            "Qwen3-8B, A100 80GB PCIe, TP1, eager decode, context 8192, "
            "batches 4/8/12/16/19, pinned 640-block KV pool, synthetic fixed "
            "needle workload; no graph-path, production, TP2/TP4, or cross-model claim"
        ),
    }


def render_markdown(report):
    lines = [
        "# KV8 + Quest Selective-Dequant Gate",
        "",
        f"Classification: **{report['classification']}**",
        "",
    ]
    if report["identity_failures"]:
        lines.append("## Identity/completeness failures")
        lines.append("")
        lines.extend(f"- `{failure}`" for failure in report["identity_failures"])
        lines.append("")
    if report["threshold_failures"]:
        lines.append("## Threshold failures")
        lines.append("")
        lines.extend(f"- `{failure}`" for failure in report["threshold_failures"])
        lines.append("")
    if report["performance_cells"]:
        lines.extend(
            [
                "## Performance",
                "",
                "| B | bf16 ms | KV8 ms | KV8+Quest ms | Quest/KV8 |",
                "|---:|---:|---:|---:|---:|",
            ]
        )
        for row in report["performance_cells"]:
            lines.append(
                "| {batch} | {bf16_step_ms:.3f} | {kv8_step_ms:.3f} | "
                "{quest_step_ms:.3f} | {quest_vs_kv8_ratio:.3f}x |".format(**row)
            )
        lines.extend(
            [
                "",
                "B=19 excess-latency recovery: "
                f"**{report['wall_cell']['excess_latency_recovery'] * 100.0:.1f}%**",
                "",
            ]
        )
    if report["quality"]:
        quality = report["quality"]
        lines.extend(
            [
                "## Quality",
                "",
                f"- bf16 overall: {quality['bf16_overall_accuracy'] * 100.0:.1f}%",
                f"- KV8 full overall: {quality['kv8_full_overall_accuracy'] * 100.0:.1f}%",
                f"- KV8+Quest overall: {quality['kv8_quest_overall_accuracy'] * 100.0:.1f}%",
                f"- Quest vs KV8 full: {quality['kv8_quest_vs_full_delta_pp']:+.1f} pp",
                "",
                "| Depth | KV8 full | KV8+Quest | Delta |",
                "|---:|---:|---:|---:|",
            ]
        )
        for row in quality["per_depth"]:
            lines.append(
                f"| {row['depth']:.2f} | {row['kv8_full_accuracy'] * 100.0:.1f}% | "
                f"{row['kv8_quest_accuracy'] * 100.0:.1f}% | {row['delta_pp']:+.1f} pp |"
            )
        lines.append("")
    lines.extend(["## Claim boundary", "", report["claim_boundary"], ""])
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bf16-performance", required=True)
    parser.add_argument("--kv8-performance", required=True)
    parser.add_argument("--quest-performance", required=True)
    parser.add_argument("--bf16-provenance", required=True)
    parser.add_argument("--kv8-provenance", required=True)
    parser.add_argument("--quest-provenance", required=True)
    parser.add_argument("--bf16-quality", required=True)
    parser.add_argument("--kv8-quality", required=True)
    parser.add_argument("--quality-provenance", required=True)
    parser.add_argument("--json-out", required=True)
    parser.add_argument("--markdown-out", required=True)
    args = parser.parse_args()

    report = build_report(
        _load(args.bf16_performance),
        _load(args.kv8_performance),
        _load(args.quest_performance),
        _load(args.bf16_provenance),
        _load(args.kv8_provenance),
        _load(args.quest_provenance),
        _load(args.bf16_quality),
        _load(args.kv8_quality),
        _load(args.quality_provenance),
    )
    Path(args.json_out).write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    markdown = render_markdown(report)
    Path(args.markdown_out).write_text(markdown, encoding="utf-8")
    print(markdown)


if __name__ == "__main__":
    main()
