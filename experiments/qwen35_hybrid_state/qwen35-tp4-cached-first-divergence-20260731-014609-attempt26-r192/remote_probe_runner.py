from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import torch


def main():
    remote = Path(sys.argv[1])
    source = Path(sys.argv[2])
    inputs = remote / "inputs"
    output = remote / "output"
    sys.path.insert(0, str(source / "tools"))
    sys.path.insert(0, str(source))

    import run_qwen35_tp4_engine_correctness_authority as driver
    import qwen35_tp4_engine_official_reference_executor as official
    import qwen35_tp4_cached_first_divergence_probe as probe
    import qwen35_tp4_cached_continuation_correctness_contract as contract
    import qwen35_tp4_cached_continuation_backend_session as backend

    configuration = driver.load_configuration(
        inputs / "executor_configuration.json",
        source_inventory_path=inputs / "source_inventory.json",
    )
    payload = contract.workload_payload("w1_medium_reuse")
    prompt = backend._request_prompt(payload, 0)
    source_prompt = (
        list(payload["shared_prefix_token_ids"])
        + list(payload["source_suffix_token_ids"])
    )

    def tensor(value):
        return value[0] if isinstance(value, tuple) else value

    def last_row(value):
        value = tensor(value)
        if value.ndim == 3:
            value = value[:, -1, :]
        elif value.ndim == 2:
            value = value[-1:, :]
        else:
            raise RuntimeError(
                f"unexpected boundary tensor rank: {value.ndim}"
            )
        return value.detach().float().cpu().clone()

    official_rows = {}
    official_backend = official.TransformersGreedyReferenceBackend(
        configuration,
        gpu_index=configuration.gpu_indices[0],
    )
    hooks = []
    try:
        model = official_backend._model()
        text = model.model

        def save_official(name):
            def hook(_module, _args, value):
                official_rows[name] = last_row(value)
                return value
            return hook

        hooks.append(
            text.embed_tokens.register_forward_hook(
                save_official("hidden_0")
            )
        )
        for index, layer in enumerate(text.layers):
            hooks.append(
                layer.register_forward_hook(
                    save_official(f"hidden_{index + 1}")
                )
            )
        hooks.append(
            text.norm.register_forward_hook(
                save_official("final_norm")
            )
        )
        hooks.append(
            model.lm_head.register_forward_hook(
                save_official("lm_head")
            )
        )
        input_ids = torch.tensor(
            [prompt],
            dtype=torch.int64,
            device=torch.device("cuda:0"),
        )
        with torch.inference_mode():
            model(
                input_ids=input_ids,
                use_cache=False,
                return_dict=True,
            )
    finally:
        for hook in hooks:
            hook.remove()
        official_backend.close()

    engine = probe.backend._default_engine_factory(configuration)
    phase = {"name": None}
    engine_rows = {"recompute": {}, "restore": {}}
    hooks = []
    try:
        engine.configure_qwen35_hybrid_prefix_publication_runtime(
            model_fingerprint=configuration.model_fingerprint,
            max_entries=configuration.max_cache_entries,
            max_bytes=configuration.max_cache_bytes,
            timeout_s=configuration.timeout_s,
        )
        model = engine.model_runner.model

        def save_engine(name):
            def hook(_module, _args, value):
                name_phase = phase["name"]
                if name_phase in engine_rows:
                    engine_rows[name_phase][name] = last_row(value)
                return value
            return hook

        def save_engine_input(name):
            def hook(_module, args):
                name_phase = phase["name"]
                if name_phase in engine_rows:
                    engine_rows[name_phase][name] = last_row(args)
            return hook

        for index, layer in enumerate(model.layer_stack.layers):
            hooks.append(
                layer.input_layernorm.register_forward_pre_hook(
                    save_engine_input(f"hidden_{index}")
                )
            )
        hooks.append(
            model.final_norm.register_forward_pre_hook(
                save_engine_input("hidden_24")
            )
        )
        hooks.append(
            model.final_norm.register_forward_hook(
                save_engine("final_norm")
            )
        )
        hooks.append(
            model.lm_head.register_forward_hook(
                save_engine("lm_head")
            )
        )

        engine.clear_qwen35_hybrid_prefix_caches(
            timeout_s=configuration.timeout_s,
        )
        phase["name"] = "recompute"
        probe._run_request(
            engine,
            prompt,
            1,
            timeout_s=configuration.timeout_s,
        )
        phase["name"] = None
        probe._run_request(
            engine,
            source_prompt,
            1,
            timeout_s=configuration.timeout_s,
            record_logits=False,
        )
        phase["name"] = "restore"
        probe._run_request(
            engine,
            prompt,
            1,
            timeout_s=configuration.timeout_s,
        )
        phase["name"] = None
    finally:
        for hook in hooks:
            hook.remove()
        cleanup = engine.exit()

    names = [
        *(f"hidden_{index}" for index in range(25)),
        "final_norm",
        "lm_head",
    ]

    def compare(left, right):
        if left.shape != right.shape:
            return {
                "shape_equal": False,
                "left_shape": list(left.shape),
                "right_shape": list(right.shape),
            }
        difference = (left - right).abs()
        return {
            "shape_equal": True,
            "shape": list(left.shape),
            "max_abs_diff": float(difference.max().item()),
            "mean_abs_diff": float(difference.mean().item()),
            "nonzero_count": int(torch.count_nonzero(difference).item()),
            "allclose": bool(torch.allclose(
                left,
                right,
                atol=contract.REGISTERED_LOGITS_ATOL,
                rtol=0.0,
            )),
        }

    official_vs_recompute = []
    recompute_vs_restore = []
    for name in names:
        if (
            name not in official_rows
            or name not in engine_rows["recompute"]
            or name not in engine_rows["restore"]
        ):
            raise RuntimeError(f"missing boundary capture: {name}")
        official_vs_recompute.append({
            "name": name,
            **compare(
                official_rows[name],
                engine_rows["recompute"][name],
            ),
        })
        recompute_vs_restore.append({
            "name": name,
            **compare(
                engine_rows["recompute"][name],
                engine_rows["restore"][name],
            ),
        })

    def first_mismatch(rows):
        for row in rows:
            if row.get("allclose") is not True:
                return row["name"]
        return None

    result = {
        "schema_version": "qwen35.tp4-cached-boundary-probe.v1",
        "workload": "w1_medium_reuse",
        "request_index": 0,
        "prompt_tokens": len(prompt),
        "official_vs_recompute": official_vs_recompute,
        "recompute_vs_restore": recompute_vs_restore,
        "official_vs_recompute_first_mismatch": first_mismatch(
            official_vs_recompute
        ),
        "recompute_vs_restore_first_mismatch": first_mismatch(
            recompute_vs_restore
        ),
        "cleanup": cleanup,
        "claim_boundary": (
            "diagnostic boundary comparison only; no cached-continuation "
            "correctness, Engine correctness, performance, cache, memory, "
            "compression, quality, or accuracy claim"
        ),
    }
    temporary = output / ".result.json.tmp"
    final = output / "result.json"
    temporary.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, final)
    print(json.dumps({
        "official_vs_recompute_first_mismatch": (
            result["official_vs_recompute_first_mismatch"]
        ),
        "recompute_vs_restore_first_mismatch": (
            result["recompute_vs_restore_first_mismatch"]
        ),
        "result_path": str(final),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
