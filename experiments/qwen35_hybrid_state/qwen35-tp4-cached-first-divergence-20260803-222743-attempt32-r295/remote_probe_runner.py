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
    import tinyvllm.layers.qwen35_full_attention as engine_full_module
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

    def tail(value):
        value = tensor(value)
        if value.ndim == 4:
            value = value[:, -64:].squeeze(0)
        elif value.ndim == 3:
            if value.shape[0] == 1:
                value = value[:, -64:, :].squeeze(0)
            else:
                value = value[-64:]
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
        official_layer_outputs = {}
        layer_output_hooks = []

        def save_layer_output(layer_index):
            def hook(_module, _args, value):
                official_layer_outputs[str(layer_index)] = tail(value)
                return value
            return hook

        for layer_index, decoder_layer in enumerate(text.layers):
            layer_output_hooks.append(
                decoder_layer.register_forward_hook(
                    save_layer_output(layer_index)
                )
            )
        official_layer0 = text.layers[0]
        official_layer0_components = {}
        official_linear_components = {}
        official_linear = official_layer0.linear_attn

        def shard_official_linear_component(name, value):
            tp_rank = 0
            tp_size = len(configuration.gpu_indices)
            if name in ("in_proj_z", "in_proj_b", "in_proj_a"):
                local_width = value.shape[-1] // tp_size
                return value.narrow(
                    -1,
                    tp_rank * local_width,
                    local_width,
                )
            if name in (
                "causal_conv",
                "in_proj_qkv",
            ):
                key_width = official_linear.key_dim
                value_width = official_linear.value_dim
                pieces = []
                global_offset = 0
                for global_width in (
                    key_width,
                    key_width,
                    value_width,
                ):
                    local_width = global_width // tp_size
                    pieces.append(value.narrow(
                        -1,
                        global_offset + tp_rank * local_width,
                        local_width,
                    ))
                    global_offset += global_width
                return torch.cat(pieces, dim=-1)
            if name == "recurrent_core":
                local_heads = value.shape[-2] // tp_size
                return value.narrow(
                    -2,
                    tp_rank * local_heads,
                    local_heads,
                )
            if name == "gated_rmsnorm":
                local_heads = value.shape[0] // tp_size
                return value.narrow(
                    0,
                    tp_rank * local_heads,
                    local_heads,
                )
            if name == "out_proj":
                return value
            raise RuntimeError(
                f"unknown linear component shard: {name}"
            )

        def canonical_linear_component(name, value, *, official_side):
            value = tensor(value)
            if name == "causal_conv":
                if official_side:
                    value = value.transpose(1, 2).squeeze(0)
            elif name in (
                "in_proj_qkv",
                "in_proj_z",
                "in_proj_b",
                "in_proj_a",
                "out_proj",
            ):
                if official_side:
                    value = value.squeeze(0)
            elif name == "recurrent_core":
                if official_side:
                    value = value.squeeze(0)
            elif name == "gated_rmsnorm":
                pass
            else:
                raise RuntimeError(
                    f"unknown linear component capture: {name}"
                )
            if official_side:
                value = shard_official_linear_component(name, value)
            return value.detach().float().cpu().clone()

        def save_official_linear_module(name):
            def hook(_module, args, value):
                hidden = tensor(args)
                is_decode = (
                    hidden.shape[0] == official_linear.num_v_heads
                    if name == "gated_rmsnorm"
                    else hidden.shape[0] == 1
                )
                if is_decode:
                    official_linear_components[name] = (
                        canonical_linear_component(
                            name,
                            value,
                            official_side=True,
                        )
                    )
                return value
            return hook

        for name in (
            "in_proj_qkv",
            "in_proj_z",
            "in_proj_b",
            "in_proj_a",
            "norm",
            "out_proj",
        ):
            capture_name = (
                "gated_rmsnorm" if name == "norm" else name
            )
            hooks.append(
                getattr(official_linear, name).register_forward_hook(
                    save_official_linear_module(capture_name)
                )
            )

        original_official_conv_update = (
            official_linear.causal_conv1d_update
        )
        original_official_recurrent = (
            official_linear.recurrent_gated_delta_rule
        )

        def wrapped_official_conv_update(*args, **kwargs):
            value = original_official_conv_update(*args, **kwargs)
            mixed = args[0] if args else kwargs["x"]
            if mixed.shape[-1] == 1:
                official_linear_components["causal_conv"] = (
                    canonical_linear_component(
                        "causal_conv",
                        value,
                        official_side=True,
                    )
                )
            return value

        def wrapped_official_recurrent(*args, **kwargs):
            value = original_official_recurrent(*args, **kwargs)
            query = args[0] if args else kwargs["query"]
            if query.shape[1] == 1:
                official_linear_components["recurrent_core"] = (
                    canonical_linear_component(
                        "recurrent_core",
                        value[0],
                        official_side=True,
                    )
                )
            return value

        official_linear.causal_conv1d_update = (
            wrapped_official_conv_update
        )
        official_linear.recurrent_gated_delta_rule = (
            wrapped_official_recurrent
        )

        def save_official_component(name):
            def hook(_module, _args, value):
                official_layer0_components[name] = tail(value)
                return value
            return hook

        def save_official_input(name):
            def hook(_module, args):
                official_layer0_components[name] = tail(args)
            return hook

        hooks.append(
            official_layer0.input_layernorm.register_forward_pre_hook(
                save_official_input("input")
            )
        )
        hooks.append(
            official_layer0.input_layernorm.register_forward_hook(
                save_official_component("input_norm")
            )
        )
        hooks.append(
            official_layer0.post_attention_layernorm
            .register_forward_pre_hook(
                save_official_input("attention_residual")
            )
        )
        hooks.append(
            official_layer0.post_attention_layernorm.register_forward_hook(
                save_official_component("post_norm")
            )
        )
        hooks.append(
            official_layer0.mlp.register_forward_hook(
                save_official_component("mlp")
            )
        )

        input_ids = torch.tensor(
            [prompt],
            dtype=torch.int64,
            device=torch.device("cuda:0"),
        )
        with torch.inference_mode():
            prefill = model(
                input_ids=input_ids,
                use_cache=True,
                return_dict=True,
            )
            first_token = prefill.logits[:, -1].argmax(
                dim=-1,
                keepdim=True,
            )
            model(
                input_ids=first_token,
                past_key_values=prefill.past_key_values,
                use_cache=True,
                return_dict=True,
            )
    finally:
        for hook in layer_output_hooks:
            hook.remove()
        for hook in hooks:
            hook.remove()
        if "official_linear" in locals():
            official_linear.causal_conv1d_update = (
                original_official_conv_update
            )
            official_linear.recurrent_gated_delta_rule = (
                original_official_recurrent
            )
        if "attention_globals" in locals():
            attention_globals["apply_rotary_pos_emb"] = original_apply_rotary
            attention_globals["eager_attention_forward"] = (
                original_official_eager_attention
            )
        official_backend.close()

    engine = probe.backend._default_engine_factory(configuration)
    phase = {"name": None}
    engine_rows = {"recompute": {}, "restore": {}}
    eager_calls = {"recompute": 0, "restore": 0}
    hooks = []
    try:
        engine.configure_qwen35_hybrid_prefix_publication_runtime(
            model_fingerprint=configuration.model_fingerprint,
            max_entries=configuration.max_cache_entries,
            max_bytes=configuration.max_cache_bytes,
            timeout_s=configuration.timeout_s,
        )
        model = engine.model_runner.model
        engine_layer_outputs = {"recompute": {}, "restore": {}}
        layer_index_by_identity = {
            id(layer): layer_index
            for layer_index, layer in enumerate(model.layer_stack.layers)
        }
        original_run_full_layer = model.layer_stack._run_full_layer
        original_run_linear_layer = model.layer_stack._run_linear_layer

        def capture_engine_layer_output(layer, value):
            name_phase = phase["name"]
            if name_phase in engine_layer_outputs:
                layer_index = layer_index_by_identity[id(layer)]
                engine_layer_outputs[name_phase][str(layer_index)] = tail(
                    value
                )

        def wrapped_run_full_layer(
            layer,
            token_counts,
            position_ids,
            hidden_states,
        ):
            value = original_run_full_layer(
                layer,
                token_counts,
                position_ids,
                hidden_states,
            )
            capture_engine_layer_output(layer, value)
            return value

        def wrapped_run_linear_layer(
            layer,
            adapter,
            token_counts,
            hidden_states,
            convolution_states,
            recurrent_states,
        ):
            value, candidates = original_run_linear_layer(
                layer,
                adapter,
                token_counts,
                hidden_states,
                convolution_states,
                recurrent_states,
            )
            capture_engine_layer_output(layer, value)
            return value, candidates

        model.layer_stack._run_full_layer = wrapped_run_full_layer
        model.layer_stack._run_linear_layer = wrapped_run_linear_layer
        engine_layer0 = model.layer_stack.layers[0]
        engine_layer0_components = {
            "recompute": {},
            "restore": {},
        }
        engine_linear_components = {
            "recompute": {},
            "restore": {},
        }
        engine_linear = engine_layer0.linear_attention

        def save_engine_linear_module(name):
            def hook(_module, args, value):
                name_phase = phase["name"]
                hidden = tensor(args)
                if (
                    name_phase in engine_linear_components
                    and hidden.shape[0] == 1
                ):
                    engine_linear_components[name_phase][name] = (
                        canonical_linear_component(
                            name,
                            value,
                            official_side=False,
                        )
                    )
                return value
            return hook

        for name in (
            "in_proj_qkv",
            "in_proj_z",
            "in_proj_b",
            "in_proj_a",
            "out_proj",
        ):
            hooks.append(
                getattr(engine_linear, name).register_forward_hook(
                    save_engine_linear_module(name)
                )
            )

        engine_linear_globals = (
            engine_linear_module.Qwen35LinearAttentionShell
            .forward.__globals__
        )
        original_engine_conv = engine_linear_globals[
            "qwen35_causal_depthwise_conv"
        ]
        original_engine_recurrent = engine_linear_globals[
            "qwen35_gated_delta_recurrent"
        ]
        original_engine_norm = engine_linear_globals[
            "qwen35_gated_rmsnorm"
        ]

        def wrapped_engine_conv(*args, **kwargs):
            value = original_engine_conv(*args, **kwargs)
            projected = args[0]
            name_phase = phase["name"]
            if (
                name_phase in engine_linear_components
                and projected.shape[0] == 1
                and args[2] is engine_linear.conv_weight
            ):
                engine_linear_components[name_phase]["causal_conv"] = (
                    canonical_linear_component(
                        "causal_conv",
                        value[0],
                        official_side=False,
                    )
                )
            return value

        def official_style_engine_recurrent(
            query,
            key,
            value,
            a,
            b,
            A_log,
            dt_bias,
            recurrent_state,
        ):
            initial_dtype = query.dtype

            def l2norm(value):
                inverse = torch.rsqrt(
                    (value * value).sum(dim=-1, keepdim=True)
                    + 1e-6
                )
                return value * inverse

            query = l2norm(query).unsqueeze(0)
            key = l2norm(key).unsqueeze(0)
            value = value.unsqueeze(0)
            beta = b.sigmoid().unsqueeze(0)
            decay = (
                -A_log.float().exp()
                * torch.nn.functional.softplus(
                    a.float() + dt_bias
                )
            ).unsqueeze(0)
            query, key, value, beta, decay = [
                item.transpose(1, 2).contiguous().float()
                for item in (query, key, value, beta, decay)
            ]
            query = query * (1 / (query.shape[-1] ** 0.5))
            state = recurrent_state.float().transpose(
                -1, -2
            ).unsqueeze(0)
            outputs = torch.zeros(
                query.shape[0],
                query.shape[1],
                query.shape[2],
                value.shape[-1],
                dtype=value.dtype,
                device=value.device,
            )
            for token_index in range(query.shape[2]):
                query_token = query[:, :, token_index]
                key_token = key[:, :, token_index]
                value_token = value[:, :, token_index]
                decay_token = (
                    decay[:, :, token_index]
                    .exp()
                    .unsqueeze(-1)
                    .unsqueeze(-1)
                )
                beta_token = beta[
                    :, :, token_index
                ].unsqueeze(-1)
                state = state * decay_token
                memory = (
                    state * key_token.unsqueeze(-1)
                ).sum(dim=-2)
                delta = (
                    value_token - memory
                ) * beta_token
                state = (
                    state
                    + key_token.unsqueeze(-1)
                    * delta.unsqueeze(-2)
                )
                outputs[:, :, token_index] = (
                    state * query_token.unsqueeze(-1)
                ).sum(dim=-2)
            return (
                outputs.transpose(1, 2)
                .contiguous()
                .squeeze(0)
                .to(initial_dtype),
                state.squeeze(0)
                .transpose(-1, -2)
                .to(recurrent_state.dtype),
            )

        def wrapped_engine_recurrent(*args, **kwargs):
            query = args[0]
            name_phase = phase["name"]
            is_layer0_decode = (
                name_phase in engine_linear_components
                and query.shape[0] == 1
                and args[5] is engine_linear.A_log
                and args[6] is engine_linear.dt_bias
            )
            value = (
                official_style_engine_recurrent(*args, **kwargs)
                if is_layer0_decode
                else original_engine_recurrent(*args, **kwargs)
            )
            if is_layer0_decode:
                engine_linear_components[name_phase][
                    "recurrent_core"
                ] = canonical_linear_component(
                    "recurrent_core",
                    value[0],
                    official_side=False,
                )
            return value

        def wrapped_engine_norm(*args, **kwargs):
            value = original_engine_norm(*args, **kwargs)
            core = args[0]
            name_phase = phase["name"]
            if (
                name_phase in engine_linear_components
                and core.shape[0] == engine_linear.local_value_heads
                and args[2] is engine_linear.norm_weight
            ):
                engine_linear_components[name_phase][
                    "gated_rmsnorm"
                ] = canonical_linear_component(
                    "gated_rmsnorm",
                    value,
                    official_side=False,
                )
            return value

        engine_linear_globals["qwen35_causal_depthwise_conv"] = (
            wrapped_engine_conv
        )
        engine_linear_globals["qwen35_gated_delta_recurrent"] = (
            wrapped_engine_recurrent
        )
        engine_linear_globals["qwen35_gated_rmsnorm"] = (
            wrapped_engine_norm
        )

        def save_engine_component(name):
            def hook(_module, _args, value):
                name_phase = phase["name"]
                if name_phase in engine_layer0_components:
                    engine_layer0_components[name_phase][name] = tail(
                        value
                    )
                return value
            return hook

        def save_engine_input(name):
            def hook(_module, args):
                name_phase = phase["name"]
                if name_phase in engine_layer0_components:
                    engine_layer0_components[name_phase][name] = tail(
                        args
                    )
            return hook

        hooks.append(
            engine_layer0.input_layernorm.register_forward_pre_hook(
                save_engine_input("input")
            )
        )
        hooks.append(
            engine_layer0.input_layernorm.register_forward_hook(
                save_engine_component("input_norm")
            )
        )
        hooks.append(
            engine_layer0.post_attention_layernorm
            .register_forward_pre_hook(
                save_engine_input("attention_residual")
            )
        )
        hooks.append(
            engine_layer0.post_attention_layernorm.register_forward_hook(
                save_engine_component("post_norm")
            )
        )
        hooks.append(
            engine_layer0.mlp.register_forward_hook(
                save_engine_component("mlp")
            )
        )

        engine.clear_qwen35_hybrid_prefix_caches(
            timeout_s=configuration.timeout_s,
        )
        phase["name"] = "recompute"
        probe._run_request(
            engine, prompt, 2, timeout_s=configuration.timeout_s
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
            engine, prompt, 2, timeout_s=configuration.timeout_s
        )
        phase["name"] = None
    finally:
        model.layer_stack._run_full_layer = original_run_full_layer
        model.layer_stack._run_linear_layer = original_run_linear_layer
        engine_linear_globals["qwen35_causal_depthwise_conv"] = (
            original_engine_conv
        )
        engine_linear_globals["qwen35_gated_delta_recurrent"] = (
            original_engine_recurrent
        )
        engine_linear_globals["qwen35_gated_rmsnorm"] = (
            original_engine_norm
        )
        for hook in hooks:
            hook.remove()
        cleanup = engine.exit()

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

    linear_component_names = (
        "in_proj_qkv",
        "in_proj_z",
        "in_proj_b",
        "in_proj_a",
        "causal_conv",
        "recurrent_core",
        "gated_rmsnorm",
        "out_proj",
    )
    official_vs_recompute_linear_components = []
    recompute_vs_restore_linear_components = []
    for name in linear_component_names:
        if (
            name not in official_linear_components
            or name not in engine_linear_components["recompute"]
            or name not in engine_linear_components["restore"]
        ):
            raise RuntimeError(
                f"missing layer0 linear component capture: {name}"
            )
        official_vs_recompute_linear_components.append({
            "name": name,
            **compare(
                official_linear_components[name],
                engine_linear_components["recompute"][name],
            ),
        })
        recompute_vs_restore_linear_components.append({
            "name": name,
            **compare(
                engine_linear_components["recompute"][name],
                engine_linear_components["restore"][name],
            ),
        })

    component_names = (
        "input",
        "input_norm",
        "attention_residual",
        "post_norm",
        "mlp",
    )
    official_vs_recompute_layer0_components = []
    recompute_vs_restore_layer0_components = []
    for name in component_names:
        if (
            name not in official_layer0_components
            or name not in engine_layer0_components["recompute"]
            or name not in engine_layer0_components["restore"]
        ):
            raise RuntimeError(
                f"missing layer0 component capture: {name}"
            )
        official_vs_recompute_layer0_components.append({
            "name": name,
            **compare(
                official_layer0_components[name],
                engine_layer0_components["recompute"][name],
            ),
        })
        recompute_vs_restore_layer0_components.append({
            "name": name,
            **compare(
                engine_layer0_components["recompute"][name],
                engine_layer0_components["restore"][name],
            ),
        })

    def first_component_mismatch(rows):
        for row in rows:
            if row.get("allclose") is not True:
                return row["name"]
        return None

    official_vs_recompute_layer_outputs = []
    recompute_vs_restore_layer_outputs = []
    for layer_index in range(24):
        key = str(layer_index)
        if (
            key not in official_layer_outputs
            or key not in engine_layer_outputs["recompute"]
            or key not in engine_layer_outputs["restore"]
        ):
            raise RuntimeError(
                f"missing decoder layer output capture: {layer_index}"
            )
        official_vs_recompute_layer_outputs.append({
            "layer_index": layer_index,
            **compare(
                official_layer_outputs[key],
                engine_layer_outputs["recompute"][key],
            ),
        })
        recompute_vs_restore_layer_outputs.append({
            "layer_index": layer_index,
            **compare(
                engine_layer_outputs["recompute"][key],
                engine_layer_outputs["restore"][key],
            ),
        })

    def first_layer_mismatch(rows):
        for row in rows:
            if row.get("allclose") is not True:
                return row["layer_index"]
        return None

    result = {
        "official_vs_recompute_linear_components": (
            official_vs_recompute_linear_components
        ),
        "recompute_vs_restore_linear_components": (
            recompute_vs_restore_linear_components
        ),
        "official_vs_recompute_first_linear_component_mismatch": (
            first_component_mismatch(
                official_vs_recompute_linear_components
            )
        ),
        "recompute_vs_restore_first_linear_component_mismatch": (
            first_component_mismatch(
                recompute_vs_restore_linear_components
            )
        ),
        "official_vs_recompute_layer0_components": (
            official_vs_recompute_layer0_components
        ),
        "recompute_vs_restore_layer0_components": (
            recompute_vs_restore_layer0_components
        ),
        "official_vs_recompute_first_component_mismatch": (
            first_component_mismatch(
                official_vs_recompute_layer0_components
            )
        ),
        "recompute_vs_restore_first_component_mismatch": (
            first_component_mismatch(
                recompute_vs_restore_layer0_components
            )
        ),
        "official_vs_recompute_layer_outputs": (
            official_vs_recompute_layer_outputs
        ),
        "recompute_vs_restore_layer_outputs": (
            recompute_vs_restore_layer_outputs
        ),
        "official_vs_recompute_first_layer_mismatch": (
            first_layer_mismatch(official_vs_recompute_layer_outputs)
        ),
        "recompute_vs_restore_first_layer_mismatch": (
            first_layer_mismatch(recompute_vs_restore_layer_outputs)
        ),
        "tensor_parallel_rank": int(
            engine_linear.in_proj_qkv.tp_rank
        ),
        "tensor_parallel_size": int(
            engine_linear.in_proj_qkv.tp_size
        ),
        "schema_version": "qwen35.tp4-cached-decode-step1-layer0-official-recurrent.v1",
        "workload": "w1_medium_reuse",
        "request_index": 0,
        "prompt_tokens": len(prompt),
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
        "official_vs_recompute_first_linear_component_mismatch": (
            result[
                "official_vs_recompute_first_linear_component_mismatch"
            ]
        ),
        "recompute_vs_restore_first_linear_component_mismatch": (
            result[
                "recompute_vs_restore_first_linear_component_mismatch"
            ]
        ),
        "official_vs_recompute_first_component_mismatch": (
            result[
                "official_vs_recompute_first_component_mismatch"
            ]
        ),
        "recompute_vs_restore_first_component_mismatch": (
            result[
                "recompute_vs_restore_first_component_mismatch"
            ]
        ),
        "official_vs_recompute_first_layer_mismatch": (
            result["official_vs_recompute_first_layer_mismatch"]
        ),
        "recompute_vs_restore_first_layer_mismatch": (
            result["recompute_vs_restore_first_layer_mismatch"]
        ),
        "result_path": str(final),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
