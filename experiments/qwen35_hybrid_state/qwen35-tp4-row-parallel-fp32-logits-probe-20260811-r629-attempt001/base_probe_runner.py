from __future__ import annotations

import json
from pathlib import Path
import sys


def _top_rows(torch_module, logits, count=10):
    values, indices = torch_module.topk(logits.float().reshape(-1), count)
    return [
        {
            "token_id": int(token_id),
            "logit": float(value),
        }
        for value, token_id in zip(values.tolist(), indices.tolist())
    ]


def _selected_logits(logits, token_ids):
    flattened = logits.float().reshape(-1)
    return {
        str(token_id): float(flattened[token_id].item())
        for token_id in token_ids
    }


def _comparator(torch_module):
    def compare(engine_logits, reference_logits, *, atol):
        if len(engine_logits) != len(reference_logits):
            raise ValueError("logits count mismatch")
        rows = []
        maximum = 0.0
        allclose = True
        first_mismatch_step = None
        first_engine_argmax = None
        first_reference_argmax = None
        for step, (engine_row, reference_row) in enumerate(
            zip(engine_logits, reference_logits)
        ):
            engine_float = engine_row.float()
            reference_float = reference_row.float()
            if engine_float.shape != reference_float.shape:
                raise ValueError("logits shape mismatch")
            difference = (engine_float - reference_float).abs()
            max_abs_diff = float(difference.max().item())
            maximum = max(maximum, max_abs_diff)
            step_allclose = bool(
                torch_module.allclose(
                    engine_float,
                    reference_float,
                    atol=atol,
                    rtol=0.0,
                )
            )
            allclose = allclose and step_allclose
            engine_top = _top_rows(torch_module, engine_float)
            reference_top = _top_rows(torch_module, reference_float)
            engine_argmax = engine_top[0]["token_id"]
            reference_argmax = reference_top[0]["token_id"]
            selected = sorted({
                13,
                68,
                197,
                engine_argmax,
                reference_argmax,
                engine_top[1]["token_id"],
                reference_top[1]["token_id"],
            })
            rows.append({
                "step": step,
                "allclose": step_allclose,
                "max_abs_diff": max_abs_diff,
                "mean_abs_diff": float(difference.mean().item()),
                "nonzero_count": int(
                    torch_module.count_nonzero(difference).item()
                ),
                "engine_argmax": engine_argmax,
                "reference_argmax": reference_argmax,
                "engine_top2_margin": (
                    engine_top[0]["logit"] - engine_top[1]["logit"]
                ),
                "reference_top2_margin": (
                    reference_top[0]["logit"]
                    - reference_top[1]["logit"]
                ),
                "engine_top10": engine_top,
                "reference_top10": reference_top,
                "engine_selected_logits": _selected_logits(
                    engine_float,
                    selected,
                ),
                "reference_selected_logits": _selected_logits(
                    reference_float,
                    selected,
                ),
                "selected_engine_minus_reference": {
                    str(token_id): float(
                        (
                            engine_float.reshape(-1)[token_id]
                            - reference_float.reshape(-1)[token_id]
                        ).item()
                    )
                    for token_id in selected
                },
            })
            if not step_allclose and first_mismatch_step is None:
                first_mismatch_step = step
                first_engine_argmax = engine_argmax
                first_reference_argmax = reference_argmax
        return {
            "max_abs_diff": maximum,
            "allclose": allclose,
            "first_mismatch_step": first_mismatch_step,
            "per_step_max_abs_diff": [
                row["max_abs_diff"] for row in rows
            ],
            "first_mismatch_engine_argmax": first_engine_argmax,
            "first_mismatch_reference_argmax": first_reference_argmax,
            "step_diagnostics": rows,
        }

    return compare


def main():
    source_root = Path(sys.argv[1]).resolve()
    output_path = Path(sys.argv[2]).resolve()
    sys.path.insert(0, str(source_root / "tools"))
    sys.path.insert(0, str(source_root))

    import torch
    import qwen35_tp4_cached_first_divergence_probe as probe
    import qwen35_tp4_engine_correctness_executor as executor
    import qwen35_tp4_engine_official_reference_executor as official

    configuration = executor.ExecutorConfiguration(
        model_dir=(
            "/data00/home/sitian/sitian-workspace01/tllm/"
            "qwen35-hybrid-state-runs/"
            "qwen35-2b-hybrid-acquire-20260723-222004/model"
        ),
        model_manifest_path=(
            "/data00/home/sitian/sitian-workspace01/tllm/"
            "qwen35-hybrid-state-runs/"
            "qwen35-2b-hybrid-acquire-20260723-222004/"
            "model_manifest.json"
        ),
        model_manifest_sha256=(
            "3e650a908234771c3cf1ac4e20c4d38f"
            "e69982efedaf4a3e631ad0b14aad7dd0"
        ),
        source_tree_sha256=(
            "71d36e6ecb237603e9142d07e2c9b9944"
            "0fabb3a5d4a5111bba28bf3eac2f843"
        ),
        workload_manifest_path=(
            "/data00/home/sitian/sitian-workspace01/tllm/"
            "qwen35-tp4-hybrid-prefix-benchmark-runs/"
            "qwen35-tp4-decode-row-parallel-20260811-r621-attempt001/"
            "workload_manifest.json"
        ),
        workload_manifest_sha256=(
            "71909b825d1a8d162604f6cc3d34ad41"
            "3b2af6c191425ec007859715a4d084e3"
        ),
        model_fingerprint=(
            "3e650a908234771c3cf1ac4e20c4d38f"
            "e69982efedaf4a3e631ad0b14aad7dd0"
        ),
        gpu_indices=(2, 4, 5, 6),
        dist_port=16321,
        master_port=16322,
        max_cache_entries=8,
        max_cache_bytes=1 << 30,
        timeout_s=600.0,
    )
    result = probe.run_probe(
        configuration=configuration,
        engine_factory=probe.backend._default_engine_factory,
        reference_executor_factory=(
            official.build_official_reference_executor_factory(
                configuration
            )
        ),
        logits_comparator=_comparator(torch),
        workload="w2_long_reuse",
        request_index=1,
        generated_tokens=1,
    )
    result["diagnostic_identity"] = {
        "tag": (
            "qwen35-tp4-row-parallel-logits-probe-"
            "20260811-r622-attempt001"
        ),
        "source_tree_sha256": configuration.source_tree_sha256,
        "gpu_indices": list(configuration.gpu_indices),
        "resource_policy": "shared-low-utilization",
        "exclusive": False,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
