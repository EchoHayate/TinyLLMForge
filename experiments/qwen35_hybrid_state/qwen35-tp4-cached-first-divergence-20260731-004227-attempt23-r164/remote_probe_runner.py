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
    import tinyvllm.layers.qwen35_linear_attention as engine_linear_module

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
        layer = text.layers[1]
        linear = layer.linear_attn
        official_original_chunk = linear.chunk_gated_delta_rule
        official_original_conv = linear.causal_conv1d_fn

        def save_official(name):
            def hook(_module, _args, value):
                official_rows[name] = last_row(value)
                return value
            return hook

        def save_official_input(name):
            def hook(_module, args):
                official_rows[name] = last_row(args)
            return hook

        def save_official_local_projection(name, kind):
            def hook(_module, _args, value):
                row = last_row(value)
                if kind == "qkv":
                    query, key, projected_value = row.split(
                        (
                            linear.key_dim,
                            linear.key_dim,
                            linear.value_dim,
                        ),
                        dim=-1,
                    )
                    row = torch.cat(
                        (
                            query[:, : linear.key_dim // 4],
                            key[:, : linear.key_dim // 4],
                            projected_value[
                                :, : linear.value_dim // 4
                            ],
                        ),
                        dim=-1,
                    )
                elif kind == "value":
                    row = row[:, : linear.value_dim // 4]
                elif kind == "heads":
                    row = row[:, : linear.num_v_heads // 4]
                else:
                    raise RuntimeError(
                        f"unsupported official projection kind: {kind}"
                    )
                official_rows[name] = row
                return value
            return hook

        def official_local_conv(value):
            row = value[:, :, -1].float().cpu()
            query, key, projected_value = row.split(
                (
                    linear.key_dim,
                    linear.key_dim,
                    linear.value_dim,
                ),
                dim=-1,
            )
            return torch.cat(
                (
                    query[:, : linear.key_dim // 4],
                    key[:, : linear.key_dim // 4],
                    projected_value[:, : linear.value_dim // 4],
                ),
                dim=-1,
            ).clone()

        if official_original_conv is None:
            def save_official_fallback_conv(
                _module,
                _args,
                value,
            ):
                convolved = torch.nn.functional.silu(
                    value[:, :, : len(prompt)]
                )
                official_rows["layer1_conv_local"] = (
                    official_local_conv(convolved)
                )
                return value

            hooks.append(
                linear.conv1d.register_forward_hook(
                    save_official_fallback_conv
                )
            )
        else:
            def wrapped_official_conv(*args, **kwargs):
                value = official_original_conv(*args, **kwargs)
                official_rows["layer1_conv_local"] = (
                    official_local_conv(value)
                )
                return value

            linear.causal_conv1d_fn = wrapped_official_conv

        def wrapped_official_chunk(*args, **kwargs):
            value = official_original_chunk(*args, **kwargs)
            core = value[0]
            official_rows["layer1_core_local"] = (
                core[:, -1, : linear.num_v_heads // 4]
                .reshape(1, -1)
                .detach()
                .float()
                .cpu()
                .clone()
            )
            return value

        linear.chunk_gated_delta_rule = wrapped_official_chunk

        hooks.append(
            layer.input_layernorm.register_forward_hook(
                save_official("layer1_input_norm")
            )
        )
        hooks.append(
            linear.in_proj_qkv.register_forward_hook(
                save_official_local_projection(
                    "layer1_in_proj_qkv_local",
                    "qkv",
                )
            )
        )
        hooks.append(
            linear.in_proj_z.register_forward_hook(
                save_official_local_projection(
                    "layer1_in_proj_z_local",
                    "value",
                )
            )
        )
        hooks.append(
            linear.in_proj_b.register_forward_hook(
                save_official_local_projection(
                    "layer1_in_proj_b_local",
                    "heads",
                )
            )
        )
        hooks.append(
            linear.in_proj_a.register_forward_hook(
                save_official_local_projection(
                    "layer1_in_proj_a_local",
                    "heads",
                )
            )
        )
        hooks.append(
            linear.register_forward_hook(
                save_official("layer1_linear_output")
            )
        )
        def gated_formula_rows(
            destination,
            prefix,
            core,
            gate,
            weight,
            token_count,
            head_count,
            head_dim,
            selected_heads,
            eps,
        ):
            def save(name, value):
                if value.ndim == 1:
                    row = value.reshape(1, -1)
                elif value.shape == (selected_heads, head_dim):
                    row = value.reshape(1, -1)
                else:
                    row = value.reshape(
                        token_count,
                        head_count,
                        head_dim,
                    )[-1, :selected_heads].reshape(1, -1)
                destination[prefix + name] = (
                    row.detach().float().cpu().clone()
                )

            core = core.reshape(
                token_count,
                head_count,
                head_dim,
            )[-1, :selected_heads].reshape(-1, head_dim)
            gate = gate.reshape(
                token_count,
                head_count,
                head_dim,
            )[-1, :selected_heads].reshape(-1, head_dim)
            core_fp32 = core.float()
            variance = core_fp32.pow(2).mean(
                dim=-1,
                keepdim=True,
            )
            inv_rms = torch.rsqrt(
                variance + float(eps)
            )
            normalized_fp32 = core_fp32 * inv_rms
            normalized_cast = normalized_fp32.to(core.dtype)
            weighted = weight * normalized_cast
            gate_silu = torch.nn.functional.silu(gate.float())
            product = weighted * gate_silu
            formula_output = product.to(core.dtype)
            save("core_input", core)
            save("gate_input", gate)
            save("weight", weight)
            save("variance_fp32", variance.expand_as(core_fp32))
            save("inv_rms_fp32", inv_rms.expand_as(core_fp32))
            save("normalized_fp32", normalized_fp32)
            save("normalized_cast", normalized_cast)
            save("weighted", weighted)
            save("gate_silu_fp32", gate_silu)
            save("product", product)
            save("formula_output", formula_output)

        def capture_official_gated_inputs(_module, args):
            core, gate = args[:2]
            gated_formula_rows(
                official_rows,
                "layer1_gated_",
                core,
                gate,
                linear.norm.weight,
                len(prompt),
                linear.num_v_heads,
                linear.head_v_dim,
                linear.num_v_heads // 4,
                linear.layer_norm_epsilon,
            )

        hooks.append(
            linear.norm.register_forward_pre_hook(
                capture_official_gated_inputs
            )
        )
        hooks.append(
            linear.norm.register_forward_hook(
                lambda _module, _args, value: (
                    official_rows.__setitem__(
                        "layer1_gated_local",
                        value.reshape(
                            len(prompt),
                            linear.num_v_heads,
                            linear.head_v_dim,
                        )[
                            -1,
                            : linear.num_v_heads // 4,
                        ].reshape(1, -1).detach().float().cpu().clone(),
                    )
                    or value
                )
            )
        )
        official_rows["layer1_gated_module_type"] = (
            type(linear.norm).__module__
            + "."
            + type(linear.norm).__qualname__
        )
        hooks.append(
            linear.out_proj.register_forward_hook(
                save_official("layer1_out_proj")
            )
        )
        hooks.append(
            layer.post_attention_layernorm.register_forward_pre_hook(
                save_official_input("layer1_attention_residual")
            )
        )
        hooks.append(
            layer.post_attention_layernorm.register_forward_hook(
                save_official("layer1_post_norm")
            )
        )
        hooks.append(
            layer.mlp.gate_proj.register_forward_hook(
                save_official("layer1_gate_proj")
            )
        )
        hooks.append(
            layer.mlp.up_proj.register_forward_hook(
                save_official("layer1_up_proj")
            )
        )
        hooks.append(
            layer.mlp.register_forward_hook(
                save_official("layer1_mlp_output")
            )
        )
        hooks.append(
            text.layers[2].input_layernorm.register_forward_pre_hook(
                save_official_input("layer1_output")
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
        if "linear" in locals():
            linear.chunk_gated_delta_rule = official_original_chunk
            linear.causal_conv1d_fn = official_original_conv
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
        layer = model.layer_stack.layers[1]
        linear = layer.linear_attention
        engine_original_conv = (
            engine_linear_module.qwen35_causal_depthwise_conv
        )
        engine_original_chunk = (
            engine_linear_module.qwen35_gated_delta_chunk
        )
        engine_original_recurrent = (
            engine_linear_module.qwen35_gated_delta_recurrent
        )
        engine_original_gated = (
            engine_linear_module.qwen35_gated_rmsnorm
        )

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

        def save_engine_gate_up(_module, _args, value):
            name_phase = phase["name"]
            if name_phase in engine_rows:
                gate, up = value.chunk(2, dim=-1)
                engine_rows[name_phase]["layer1_gate_proj"] = last_row(
                    gate
                )
                engine_rows[name_phase]["layer1_up_proj"] = last_row(up)
            return value

        def wrapped_engine_conv(
            projected_qkv,
            conv_state,
            weight,
            **kwargs,
        ):
            value = engine_original_conv(
                projected_qkv,
                conv_state,
                weight,
                **kwargs,
            )
            name_phase = phase["name"]
            if (
                name_phase in engine_rows
                and weight.data_ptr() == linear.conv_weight.data_ptr()
            ):
                engine_rows[name_phase]["layer1_conv_local"] = last_row(
                    value[0]
                )
            return value

        def capture_engine_core(function):
            def wrapped(
                query,
                key,
                value,
                a,
                b,
                A_log,
                dt_bias,
                recurrent_state,
                **kwargs,
            ):
                result = function(
                    query,
                    key,
                    value,
                    a,
                    b,
                    A_log,
                    dt_bias,
                    recurrent_state,
                    **kwargs,
                )
                name_phase = phase["name"]
                if (
                    name_phase in engine_rows
                    and A_log.data_ptr() == linear.A_log.data_ptr()
                ):
                    engine_rows[name_phase]["layer1_core_local"] = (
                        result[0][-1:].reshape(1, -1)
                        .detach()
                        .float()
                        .cpu()
                        .clone()
                    )
                return result
            return wrapped

        def wrapped_engine_gated(
            core,
            gate,
            weight,
            **kwargs,
        ):
            value = engine_original_gated(
                core,
                gate,
                weight,
                **kwargs,
            )
            name_phase = phase["name"]
            if (
                name_phase in engine_rows
                and weight.data_ptr() == linear.norm_weight.data_ptr()
            ):
                gated_formula_rows(
                    engine_rows[name_phase],
                    "layer1_gated_",
                    core,
                    gate,
                    weight,
                    core.shape[0] // linear.local_value_heads,
                    linear.local_value_heads,
                    linear.value_head_dim,
                    linear.local_value_heads,
                    kwargs.get("eps", linear.norm_eps),
                )
            name_phase = phase["name"]
            if (
                name_phase in engine_rows
                and weight.data_ptr() == linear.norm_weight.data_ptr()
            ):
                engine_rows[name_phase]["layer1_gated_local"] = (
                    value.reshape(
                        -1,
                        linear.local_value_heads,
                        linear.value_head_dim,
                    )[-1:].reshape(1, -1)
                    .detach()
                    .float()
                    .cpu()
                    .clone()
                )
            return value

        engine_linear_module.qwen35_causal_depthwise_conv = (
            wrapped_engine_conv
        )
        engine_linear_module.qwen35_gated_delta_chunk = (
            capture_engine_core(engine_original_chunk)
        )
        engine_linear_module.qwen35_gated_delta_recurrent = (
            capture_engine_core(engine_original_recurrent)
        )
        engine_linear_module.qwen35_gated_rmsnorm = (
            wrapped_engine_gated
        )

        hooks.append(
            layer.input_layernorm.register_forward_hook(
                save_engine("layer1_input_norm")
            )
        )
        hooks.append(
            linear.in_proj_qkv.register_forward_hook(
                save_engine("layer1_in_proj_qkv_local")
            )
        )
        hooks.append(
            linear.in_proj_z.register_forward_hook(
                save_engine("layer1_in_proj_z_local")
            )
        )
        hooks.append(
            linear.in_proj_b.register_forward_hook(
                save_engine("layer1_in_proj_b_local")
            )
        )
        hooks.append(
            linear.in_proj_a.register_forward_hook(
                save_engine("layer1_in_proj_a_local")
            )
        )
        hooks.append(
            linear.register_forward_hook(
                save_engine("layer1_linear_output")
            )
        )
        hooks.append(
            linear.out_proj.register_forward_hook(
                save_engine("layer1_out_proj")
            )
        )
        hooks.append(
            layer.post_attention_layernorm.register_forward_pre_hook(
                save_engine_input("layer1_attention_residual")
            )
        )
        hooks.append(
            layer.post_attention_layernorm.register_forward_hook(
                save_engine("layer1_post_norm")
            )
        )
        hooks.append(
            layer.mlp.gate_up_proj.register_forward_hook(
                save_engine_gate_up
            )
        )
        hooks.append(
            layer.mlp.register_forward_hook(
                save_engine("layer1_mlp_output")
            )
        )
        hooks.append(
            model.layer_stack.layers[
                2
            ].input_layernorm.register_forward_pre_hook(
                save_engine_input("layer1_output")
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
        engine_linear_module.qwen35_causal_depthwise_conv = (
            engine_original_conv
        )
        engine_linear_module.qwen35_gated_delta_chunk = (
            engine_original_chunk
        )
        engine_linear_module.qwen35_gated_delta_recurrent = (
            engine_original_recurrent
        )
        engine_linear_module.qwen35_gated_rmsnorm = (
            engine_original_gated
        )
        cleanup = engine.exit()

    names = [
        "layer1_conv_local",
        "layer1_core_local",
        "layer1_gated_core_input",
        "layer1_gated_gate_input",
        "layer1_gated_weight",
        "layer1_gated_variance_fp32",
        "layer1_gated_inv_rms_fp32",
        "layer1_gated_normalized_fp32",
        "layer1_gated_normalized_cast",
        "layer1_gated_weighted",
        "layer1_gated_gate_silu_fp32",
        "layer1_gated_product",
        "layer1_gated_formula_output",
        "layer1_gated_local",
        "layer1_out_proj",
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
        "schema_version": "qwen35.tp4-cached-layer0-core-probe.v1",
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
        "official_gated_module_type": official_rows[
            "layer1_gated_module_type"
        ],
        "official_actual_vs_formula": compare(
            official_rows["layer1_gated_local"],
            official_rows["layer1_gated_formula_output"],
        ),
        "engine_recompute_actual_vs_formula": compare(
            engine_rows["recompute"]["layer1_gated_local"],
            engine_rows["recompute"]["layer1_gated_formula_output"],
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
