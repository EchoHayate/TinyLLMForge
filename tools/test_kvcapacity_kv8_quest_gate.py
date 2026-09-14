from __future__ import annotations

import copy
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import kvcapacity_kv8_quest_gate as gate


BATCHES = (4, 8, 12, 16, 19)


def make_perf(bits, top_k, times):
    rows = []
    for batch, step_ms in zip(BATCHES, times):
        rows.append(
            {
                "context_length": 8192,
                "batch": batch,
                "measured": True,
                "observed_batch_is_stable": True,
                "prefill_tokens_match": True,
                "prompt_digests": [f"prompt-{batch}-{index}" for index in range(batch)],
                "step": {"median_ms": step_ms, "count": 24},
                "dispatch_measured": {
                    "all_graph": False,
                    "graph_steps": 0,
                    "steps": 24,
                    "counts": {"eager:feature_disabled": 24},
                },
            }
        )
    return {
        "grid_spec": "8192:4,8,12,16,19",
        "configuration": {
            "enforce_eager": True,
            "gpu_memory_utilization": 0.85,
            "kv_blocks_requested": 640,
            "kv_quant_bits": bits,
            "measured_steps": 24,
            "model_path": "/models/Qwen3-8B",
            "quest_min_seq_len": 512,
            "quest_top_k_blocks": top_k,
            "seed": 20260913,
            "warmup_steps": 24,
        },
        "engines": [
            {
                "context_length": 8192,
                "identity": {
                    "enforce_eager": True,
                    "kv_blocks_requested": 640,
                    "num_kvcache_blocks": 640,
                    "kv_quant_bits": bits,
                    "quest_min_seq_len": 512,
                    "quest_top_k_blocks": top_k,
                },
            }
        ],
        "rows": rows,
    }


def make_provenance(revision="a" * 40):
    return {
        "source_revision": revision,
        "tinyvllm_dirty_paths": 0,
        "worker_sha256": "b" * 64,
        "cuda_device": "2",
        "mode": "sweep",
    }


def make_quality_provenance():
    return {
        "source_revision": "c" * 40,
        "source_dirty_paths": 0,
        "remote_dir": (
            "/data00/home/sitian/tllm/kvcapacity-runs/"
            "20260915-kv8quest-quality-test"
        ),
        "cuda_device": "2",
        "model": "/models/Qwen3-8B",
        "fixed_prompts": True,
        "needle_style": "newline",
        "context_lens": [8192],
        "depths": list(gate.EXPECTED_DEPTHS),
        "num_trials": 5,
        "quest_top_k_blocks": 16,
        "quest_min_seq_len": 512,
    }


def make_quality(bits, settings):
    results = []
    for top_k, hits_by_depth in settings:
        details = []
        for depth, hit_count in zip(gate.EXPECTED_DEPTHS, hits_by_depth):
            for trial in range(5):
                magic = 10000 + int(depth * 100) * 10 + trial
                hit = trial < hit_count
                details.append(
                    {
                        "ctx_len": 8192,
                        "depth": depth,
                        "trial": trial,
                        "magic": magic,
                        "answer": str(magic) if hit else "99999",
                        "hit": hit,
                        "raw": str(magic) if hit else "99999",
                    }
                )
        results.append(
            {
                "top_k": top_k,
                "overall_accuracy": sum(row["hit"] for row in details) / 25,
                "details": details,
            }
        )
    return {
        "args": {
            "context_lens": [8192],
            "depths": list(gate.EXPECTED_DEPTHS),
            "enforce_eager": True,
            "fixed_prompts": True,
            "gpu_memory_utilization": 0.85,
            "kv_quant_bits": bits,
            "max_model_len": 16384,
            "max_num_seqs": 32,
            "max_output_len": 16,
            "model": "/models/Qwen3-8B",
            "needle_style": "newline",
            "num_trials": 5,
            "quest_min_seq_len": 512,
            "seed": 0,
            "tp_size": 1,
            "top_k_blocks_list": [top_k for top_k, _ in settings],
        },
        "results": results,
    }


def good_inputs():
    perf = {
        "bf16": make_perf(0, -1, [40, 44, 48, 52, 56]),
        "kv8": make_perf(8, -1, [70, 110, 150, 210, 250]),
        "quest": make_perf(8, 16, [65, 90, 110, 130, 145]),
    }
    provenance = {name: make_provenance() for name in perf}
    quality = {
        "bf16": make_quality(0, [(-1, [5, 5, 5, 5, 5])]),
        "kv8": make_quality(
            8,
            [
                (-1, [5, 5, 5, 5, 5]),
                (16, [5, 5, 5, 5, 5]),
            ],
        ),
        "provenance": make_quality_provenance(),
    }
    return perf, provenance, quality


def classify(perf, provenance, quality):
    return gate.build_report(
        perf["bf16"],
        perf["kv8"],
        perf["quest"],
        provenance["bf16"],
        provenance["kv8"],
        provenance["quest"],
        quality["bf16"],
        quality["kv8"],
        quality["provenance"],
    )


def test_all_thresholds_pass_goes_to_fused_kernel_scope():
    perf, provenance, quality = good_inputs()
    report = classify(perf, provenance, quality)
    assert report["classification"] == "GO_TO_FUSED_KERNEL_SCOPE"
    assert report["wall_cell"]["excess_latency_recovery"] > 0.5


def test_mismatched_performance_source_is_inconclusive():
    perf, provenance, quality = good_inputs()
    provenance["quest"]["source_revision"] = "c" * 40
    report = classify(perf, provenance, quality)
    assert report["classification"] == "INCONCLUSIVE"
    assert "performance_source_revision_mismatch" in report["identity_failures"]


def test_incomplete_cell_is_inconclusive():
    perf, provenance, quality = good_inputs()
    perf["quest"]["rows"].pop()
    report = classify(perf, provenance, quality)
    assert report["classification"] == "INCONCLUSIVE"
    assert any("missing_cells" in failure for failure in report["identity_failures"])


def test_quest_must_improve_every_performance_cell():
    perf, provenance, quality = good_inputs()
    perf["quest"]["rows"][0]["step"]["median_ms"] = 71
    report = classify(perf, provenance, quality)
    assert report["classification"] == "NO_GO_KV8_QUEST"
    assert "quest_not_faster_in_every_cell" in report["threshold_failures"]


def test_wall_cell_must_recover_half_of_excess_latency():
    perf, provenance, quality = good_inputs()
    perf["quest"]["rows"][-1]["step"]["median_ms"] = 180
    report = classify(perf, provenance, quality)
    assert report["classification"] == "NO_GO_KV8_QUEST"
    assert "wall_excess_latency_recovery_below_50pct" in report["threshold_failures"]


def test_overall_quality_loss_over_five_points_is_no_go():
    perf, provenance, quality = good_inputs()
    quality["kv8"] = make_quality(
        8,
        [
            (-1, [5, 5, 5, 5, 5]),
            (16, [4, 4, 4, 4, 4]),
        ],
    )
    report = classify(perf, provenance, quality)
    assert report["classification"] == "NO_GO_KV8_QUEST"
    assert "overall_accuracy_loss_over_5pp" in report["threshold_failures"]


def test_any_depth_quality_loss_over_twenty_points_is_no_go():
    perf, provenance, quality = good_inputs()
    quality["kv8"] = make_quality(
        8,
        [
            (-1, [5, 5, 5, 5, 5]),
            (16, [3, 5, 5, 5, 5]),
        ],
    )
    report = classify(perf, provenance, quality)
    assert report["classification"] == "NO_GO_KV8_QUEST"
    assert "per_depth_accuracy_loss_over_20pp" in report["threshold_failures"]


def test_unpaired_quality_cases_are_inconclusive():
    perf, provenance, quality = good_inputs()
    mutated = copy.deepcopy(quality["kv8"])
    mutated["results"][1]["details"][0]["magic"] += 1
    quality["kv8"] = mutated
    report = classify(perf, provenance, quality)
    assert report["classification"] == "INCONCLUSIVE"
    assert "kv8_quality_cases_not_paired" in report["identity_failures"]


def test_quality_model_identity_mismatch_is_inconclusive():
    perf, provenance, quality = good_inputs()
    quality["kv8"]["args"]["model"] = "/models/not-qwen"
    report = classify(perf, provenance, quality)
    assert report["classification"] == "INCONCLUSIVE"
    assert "kv8_quality_model_is_not_Qwen3-8B" in report["identity_failures"]


def test_quality_hit_is_recomputed_from_answer_and_magic():
    perf, provenance, quality = good_inputs()
    row = quality["kv8"]["results"][1]["details"][0]
    row["answer"] = "99999"
    row["hit"] = True
    report = classify(perf, provenance, quality)
    assert report["classification"] == "INCONCLUSIVE"
    assert any(
        failure.startswith("kv8_quality_top_k_16_producer_hit_mismatch")
        for failure in report["identity_failures"]
    )


def test_dirty_quality_source_is_inconclusive():
    perf, provenance, quality = good_inputs()
    quality["provenance"]["source_dirty_paths"] = 1
    report = classify(perf, provenance, quality)
    assert report["classification"] == "INCONCLUSIVE"
    assert "quality_source_dirty" in report["identity_failures"]
