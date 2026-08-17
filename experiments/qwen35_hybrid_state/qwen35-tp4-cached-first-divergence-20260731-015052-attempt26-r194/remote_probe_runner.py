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

    def tail(value):
        value = tensor(value)
        if value.ndim == 4:
            value = value[:, -64:].squeeze(0)
        elif value.ndim == 3:
            value = value[:, -64:, :].squeeze(0)
        elif value.ndim == 2:
            value = value[-64:]
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
        layer = text.layers[3]
        attention = layer.self_attn
        local_query_heads = attention.config.num_attention_heads // 4
        local_kv_heads = max(1, attention.config.num_key_value_heads // 4)

        def save(name):
            def hook(_module, _args, value):
                official_rows[name] = tail(value)
                return value
            return hook

        def save_input(name):
            def hook(_module, args):
                official_rows[name] = tail(args)
            return hook

        def save_q_projection(_module, _args, value):
            captured = value[:, -64:].reshape(
                64,
                attention.config.num_attention_heads,
                2 * attention.head_dim,
            )
            official_rows["layer3_q_projection_local"] = (
                captured[:, :local_query_heads]
                .reshape(64, -1)
                .detach().float().cpu().clone()
            )
            return value

        def save_kv_projection(name):
            def hook(_module, _args, value):
                captured = value[:, -64:].reshape(
                    64,
                    attention.config.num_key_value_heads,
                    attention.head_dim,
                )
                official_rows[name] = (
                    captured[:, :local_kv_heads]
                    .reshape(64, -1)
                    .detach().float().cpu().clone()
                )
                return value
            return hook

        def save_q_norm(_module, _args, value):
            official_rows["layer3_q_norm_local"] = (
                value[:, -64:, :local_query_heads]
                .reshape(64, -1)
                .detach().float().cpu().clone()
            )
            return value

        def save_k_norm(_module, _args, value):
            official_rows["layer3_k_norm_local"] = (
                value[:, -64:, :local_kv_heads]
                .reshape(64, -1)
                .detach().float().cpu().clone()
            )
            return value

        def save_output_input(_module, args):
            value = tensor(args)[:, -64:].reshape(
                64,
                attention.config.num_attention_heads,
                attention.head_dim,
            )
            official_rows["layer3_output_input_local"] = (
                value[:, :local_query_heads]
                .reshape(64, -1)
                .detach().float().cpu().clone()
            )

        hooks.append(
            layer.input_layernorm.register_forward_hook(
                save("layer3_input_norm")
            )
        )
        hooks.append(attention.q_proj.register_forward_hook(save_q_projection))
        hooks.append(
            attention.k_proj.register_forward_hook(
                save_kv_projection("layer3_k_projection_local")
            )
        )
        hooks.append(
            attention.v_proj.register_forward_hook(
                save_kv_projection("layer3_v_projection_local")
            )
        )
        hooks.append(attention.q_norm.register_forward_hook(save_q_norm))
        hooks.append(attention.k_norm.register_forward_hook(save_k_norm))
        hooks.append(attention.o_proj.register_forward_pre_hook(save_output_input))
        hooks.append(
            attention.o_proj.register_forward_hook(
                save("layer3_attention_output")
            )
        )
        hooks.append(
            layer.post_attention_layernorm.register_forward_pre_hook(
                save_input("layer3_attention_residual")
            )
        )
        hooks.append(
            layer.post_attention_layernorm.register_forward_hook(
                save("layer3_post_norm")
            )
        )
        hooks.append(
            layer.mlp.gate_proj.register_forward_hook(
                save("layer3_gate_proj")
            )
        )
        hooks.append(
            layer.mlp.up_proj.register_forward_hook(
                save("layer3_up_proj")
            )
        )
        hooks.append(
            layer.mlp.register_forward_hook(
                save("layer3_mlp_output")
            )
        )
        hooks.append(
            text.layers[4].input_layernorm.register_forward_pre_hook(
                save_input("layer3_output")
            )
        )
        input_ids = torch.tensor(
            [prompt],
            dtype=torch.int64,
            device=torch.device("cuda:0"),
        )
        with torch.inference_mode():
            model(input_ids=input_ids, use_cache=False, return_dict=True)
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
        layer = model.layer_stack.layers[3]
        attention = layer.full_attention

        def save(name):
            def hook(_module, _args, value):
                name_phase = phase["name"]
                if name_phase in engine_rows:
                    engine_rows[name_phase][name] = tail(value)
                return value
            return hook

        def save_input(name):
            def hook(_module, args):
                name_phase = phase["name"]
                if name_phase in engine_rows:
                    engine_rows[name_phase][name] = tail(args)
            return hook

        def save_gate_up(_module, _args, value):
            name_phase = phase["name"]
            if name_phase in engine_rows:
                gate, up = value.chunk(2, dim=-1)
                engine_rows[name_phase]["layer3_gate_proj"] = tail(gate)
                engine_rows[name_phase]["layer3_up_proj"] = tail(up)
            return value

        hooks.append(
            layer.input_layernorm.register_forward_hook(
                save("layer3_input_norm")
            )
        )
        hooks.append(
            attention.q_projection.register_forward_hook(
                save("layer3_q_projection_local")
            )
        )
        hooks.append(
            attention.k_projection.register_forward_hook(
                save("layer3_k_projection_local")
            )
        )
        hooks.append(
            attention.v_projection.register_forward_hook(
                save("layer3_v_projection_local")
            )
        )
        hooks.append(
            attention.q_norm.register_forward_hook(
                save("layer3_q_norm_local")
            )
        )
        hooks.append(
            attention.k_norm.register_forward_hook(
                save("layer3_k_norm_local")
            )
        )
        hooks.append(
            attention.output_projection.register_forward_pre_hook(
                save_input("layer3_output_input_local")
            )
        )
        hooks.append(
            attention.output_projection.register_forward_hook(
                save("layer3_attention_output")
            )
        )
        hooks.append(
            layer.post_attention_layernorm.register_forward_pre_hook(
                save_input("layer3_attention_residual")
            )
        )
        hooks.append(
            layer.post_attention_layernorm.register_forward_hook(
                save("layer3_post_norm")
            )
        )
        hooks.append(
            layer.mlp.gate_up_proj.register_forward_hook(save_gate_up)
        )
        hooks.append(
            layer.mlp.register_forward_hook(save("layer3_mlp_output"))
        )
        hooks.append(
            model.layer_stack.layers[4].input_layernorm.register_forward_pre_hook(
                save_input("layer3_output")
            )
        )

        engine.clear_qwen35_hybrid_prefix_caches(
            timeout_s=configuration.timeout_s,
        )
        phase["name"] = "recompute"
        probe._run_request(
            engine, prompt, 1, timeout_s=configuration.timeout_s
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
            engine, prompt, 1, timeout_s=configuration.timeout_s
        )
        phase["name"] = None
    finally:
        for hook in hooks:
            hook.remove()
        cleanup = engine.exit()

    names = [
        "layer3_input_norm",
        "layer3_q_projection_local",
        "layer3_k_projection_local",
        "layer3_v_projection_local",
        "layer3_q_norm_local",
        "layer3_k_norm_local",
        "layer3_output_input_local",
        "layer3_attention_output",
        "layer3_attention_residual",
        "layer3_post_norm",
        "layer3_gate_proj",
        "layer3_up_proj",
        "layer3_mlp_output",
        "layer3_output",
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
            **compare(official_rows[name], engine_rows["recompute"][name]),
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
        "schema_version": "qwen35.tp4-cached-layer3-full-attention.v1",
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
