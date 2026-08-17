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
    official_native = {}
    official_delta = {}
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
            local = captured[:, :local_query_heads]
            official_rows["layer3_q_projection_local"] = (
                local.reshape(64, -1)
                .detach().float().cpu().clone()
            )
            official_rows["layer3_query_gate_local"] = (
                local[..., attention.head_dim:]
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
                .squeeze(0)
                .detach().float().cpu().clone()
            )
            return value

        def save_k_norm(_module, _args, value):
            official_rows["layer3_k_norm_local"] = (
                value[:, -64:, :local_kv_heads]
                .squeeze(0)
                .detach().float().cpu().clone()
            )
            return value

        def save_output_input(_module, args):
            value = tensor(args)[:, -64:].reshape(
                64,
                attention.config.num_attention_heads,
                attention.head_dim,
            )
            gated = (
                value[:, :local_query_heads]
                .reshape(64, -1)
                .detach().float().cpu().clone()
            )
            official_rows["layer3_output_input_local"] = gated
            gate = official_rows["layer3_query_gate_local"]
            official_rows["layer3_attention_raw_inferred_local"] = (
                gated / torch.sigmoid(gate)
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
        attention_globals = attention.forward.__func__.__globals__
        original_official_eager_attention = attention_globals[
            "eager_attention_forward"
        ]

        def wrapped_official_eager_attention(
            module,
            query,
            key,
            value,
            attention_mask,
            *args,
            **kwargs,
        ):
            output, weights = original_official_eager_attention(
                module,
                query,
                key,
                value,
                attention_mask,
                *args,
                **kwargs,
            )
            if module is attention:
                official_rows["layer3_attention_raw_local"] = (
                    output[:, -64:, :local_query_heads]
                    .reshape(64, -1)
                    .detach().float().cpu().clone()
                )
                official_rows["layer3_attention_raw_full_shape"] = list(
                    output.shape
                )
                official_rows["layer3_full_key_local"] = (
                    key[:, :local_kv_heads]
                    .transpose(1, 2)
                    .squeeze(0)
                    .detach().float().cpu().clone()
                )
                official_rows["layer3_full_value_local"] = (
                    value[:, :local_kv_heads]
                    .transpose(1, 2)
                    .squeeze(0)
                    .detach().float().cpu().clone()
                )
            return output, weights

        attention_globals["eager_attention_forward"] = (
            wrapped_official_eager_attention
        )
        original_apply_rotary = attention_globals["apply_rotary_pos_emb"]

        rotary_capture_pending = [True]

        def wrapped_apply_rotary(query, key, *args, **kwargs):
            rotated_query, rotated_key = original_apply_rotary(
                query,
                key,
                *args,
                **kwargs,
            )
            if rotary_capture_pending[0]:
                official_rows["layer3_rotary_query_local"] = (
                    rotated_query[:, :local_query_heads, -64:, :]
                    .transpose(1, 2)
                    .squeeze(0)
                    .detach().float().cpu().clone()
                )
                official_rows["layer3_rotary_key_local"] = (
                    rotated_key[:, :local_kv_heads, -64:, :]
                    .transpose(1, 2)
                    .squeeze(0)
                    .detach().float().cpu().clone()
                )
                rotary_capture_pending[0] = False
            return rotated_query, rotated_key

        attention_globals["apply_rotary_pos_emb"] = wrapped_apply_rotary
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
        layer4 = text.layers[4]
        linear4 = layer4.linear_attn
        official_linear_globals = linear4.forward.__func__.__globals__
        original_official_chunk = linear4.chunk_gated_delta_rule

        def wrapped_official_chunk(
            query,
            key,
            value,
            *,
            g,
            beta,
            initial_state=None,
            output_final_state=False,
            use_qk_l2norm_in_kernel=False,
        ):
            core, final_state = original_official_chunk(
                query,
                key,
                value,
                g=g,
                beta=beta,
                initial_state=initial_state,
                output_final_state=output_final_state,
                use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
            )
            if query.shape[1] == len(prompt):
                full_replay_core, full_replay_state = (
                    original_official_chunk(
                        query,
                        key,
                        value,
                        g=g,
                        beta=beta,
                        initial_state=initial_state,
                        output_final_state=True,
                        use_qk_l2norm_in_kernel=(
                            use_qk_l2norm_in_kernel
                        ),
                    )
                )
                prefix_core, prefix_state = original_official_chunk(
                    query[:, :-64],
                    key[:, :-64],
                    value[:, :-64],
                    g=g[:, :-64],
                    beta=beta[:, :-64],
                    initial_state=None,
                    output_final_state=True,
                    use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
                )
                suffix_core, suffix_state = original_official_chunk(
                    query[:, -64:],
                    key[:, -64:],
                    value[:, -64:],
                    g=g[:, -64:],
                    beta=beta[:, -64:],
                    initial_state=prefix_state,
                    output_final_state=True,
                    use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
                )
                official_delta.update({
                    "query": query[:, -64:].detach().clone(),
                    "key": key[:, -64:].detach().clone(),
                    "value": value[:, -64:].detach().clone(),
                    "g": g[:, -64:].detach().clone(),
                    "beta": beta[:, -64:].detach().clone(),
                    "full_suffix_core": core[:, -64:].detach().clone(),
                    "full_replay_suffix_core": (
                        full_replay_core[:, -64:].detach().clone()
                    ),
                    "full_final_state": (
                        full_replay_state.detach().clone()
                    ),
                    "prefix_core": prefix_core.detach().clone(),
                    "prefix_state": prefix_state.detach().clone(),
                    "split_suffix_core": suffix_core.detach().clone(),
                    "split_final_state": suffix_state.detach().clone(),
                })
            return core, final_state

        linear4.chunk_gated_delta_rule = wrapped_official_chunk
        local_key_width = linear4.key_dim // 4
        local_value_width = linear4.value_dim // 4
        local_value_heads = linear4.num_v_heads // 4

        def save_official_layer4(name):
            def hook(_module, _args, value):
                official_rows[name] = tail(value)
                return value
            return hook

        def save_official_layer4_projection(name, kind):
            def hook(_module, _args, value):
                rows = tensor(value)[:, -64:]
                if kind == "qkv":
                    query, key, projected_value = rows.split(
                        (
                            linear4.key_dim,
                            linear4.key_dim,
                            linear4.value_dim,
                        ),
                        dim=-1,
                    )
                    rows = torch.cat((
                        query[..., :local_key_width],
                        key[..., :local_key_width],
                        projected_value[..., :local_value_width],
                    ), dim=-1)
                elif kind == "value":
                    rows = rows[..., :local_value_width]
                elif kind == "heads":
                    rows = rows[..., :local_value_heads]
                else:
                    raise RuntimeError(
                        f"unsupported layer4 projection kind: {kind}"
                    )
                if name == "layer4_projected_qkv":
                    full_rows = tensor(value)
                    full_query, full_key, full_value = full_rows.split(
                        (
                            linear4.key_dim,
                            linear4.key_dim,
                            linear4.value_dim,
                        ),
                        dim=-1,
                    )
                    official_native[
                        "official_layer4_projected_qkv_full"
                    ] = torch.cat((
                        full_query[..., :local_key_width],
                        full_key[..., :local_key_width],
                        full_value[..., :local_value_width],
                    ), dim=-1).squeeze(0).detach().cpu().clone()
                official_rows[name] = (
                    rows.squeeze(0).detach().float().cpu().clone()
                )
                return value
            return hook

        def save_official_layer4_conv(_module, _args, value):
            convolved = torch.nn.functional.silu(
                value[:, :, :len(prompt)]
            ).transpose(1, 2)[:, -64:]
            query, key, projected_value = convolved.split(
                (
                    linear4.key_dim,
                    linear4.key_dim,
                    linear4.value_dim,
                ),
                dim=-1,
            )
            official_rows["layer4_convolved_qkv"] = torch.cat((
                query[..., :local_key_width],
                key[..., :local_key_width],
                projected_value[..., :local_value_width],
            ), dim=-1).squeeze(0).detach().float().cpu().clone()
            return value

        def save_official_layer4_norm_input(_module, args):
            core = args[0].reshape(
                1,
                len(prompt),
                linear4.num_v_heads,
                linear4.head_v_dim,
            )[:, -64:, :local_value_heads]
            gate = args[1].reshape(
                1,
                len(prompt),
                linear4.num_v_heads,
                linear4.head_v_dim,
            )[:, -64:, :local_value_heads]
            official_rows["layer4_delta_core"] = (
                core.squeeze(0).detach().float().cpu().clone()
            )
            official_rows["layer4_norm_gate"] = (
                gate.squeeze(0).detach().float().cpu().clone()
            )

        def save_official_layer4_gated(_module, _args, value):
            gated = value.reshape(
                1,
                len(prompt),
                linear4.num_v_heads,
                linear4.head_v_dim,
            )[:, -64:, :local_value_heads]
            official_rows["layer4_gated_local"] = (
                gated.squeeze(0).detach().float().cpu().clone()
            )
            return value

        hooks.append(
            layer4.input_layernorm.register_forward_hook(
                save_official_layer4("layer4_input_norm")
            )
        )
        hooks.append(
            linear4.in_proj_qkv.register_forward_hook(
                save_official_layer4_projection(
                    "layer4_projected_qkv", "qkv"
                )
            )
        )
        hooks.append(
            linear4.in_proj_z.register_forward_hook(
                save_official_layer4_projection(
                    "layer4_projected_z", "value"
                )
            )
        )
        hooks.append(
            linear4.in_proj_a.register_forward_hook(
                save_official_layer4_projection(
                    "layer4_projected_a", "heads"
                )
            )
        )
        hooks.append(
            linear4.in_proj_b.register_forward_hook(
                save_official_layer4_projection(
                    "layer4_projected_b", "heads"
                )
            )
        )
        hooks.append(
            linear4.conv1d.register_forward_hook(
                save_official_layer4_conv
            )
        )
        hooks.append(
            linear4.norm.register_forward_pre_hook(
                save_official_layer4_norm_input
            )
        )
        hooks.append(
            linear4.norm.register_forward_hook(
                save_official_layer4_gated
            )
        )
        hooks.append(
            linear4.register_forward_hook(
                save_official_layer4("layer4_mixer_output")
            )
        )
        hooks.append(
            layer4.post_attention_layernorm.register_forward_hook(
                save_official_layer4("layer4_post_norm")
            )
        )
        hooks.append(
            layer4.mlp.register_forward_hook(
                save_official_layer4("layer4_mlp_output")
            )
        )
        hooks.append(
            text.layers[5].input_layernorm.register_forward_pre_hook(
                lambda _module, args: official_rows.__setitem__(
                    "layer4_output", tail(args)
                )
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
        if "linear4" in locals():
            linear4.chunk_gated_delta_rule = original_official_chunk
        for hook in layer_output_hooks:
            hook.remove()
        for hook in hooks:
            hook.remove()
        if "attention_globals" in locals():
            attention_globals["apply_rotary_pos_emb"] = original_apply_rotary
            attention_globals["eager_attention_forward"] = (
                original_official_eager_attention
            )
        official_backend.close()

    engine = probe.backend._default_engine_factory(configuration)
    phase = {"name": None}
    engine_rows = {"recompute": {}, "restore": {}}
    engine_delta = {"recompute": {}, "restore": {}}
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
        layer = model.layer_stack.layers[3]
        attention = layer.full_attention
        original_eager_attention = (
            engine_full_module.qwen35_cached_prefill_eager_attention
        )

        def wrapped_eager_attention(*args, **kwargs):
            value = original_eager_attention(*args, **kwargs)
            name_phase = phase["name"]
            layer3_cache = (
                args[3].data_ptr()
                == attention.attention_backend.k_cache.data_ptr()
            )
            if name_phase in engine_rows and layer3_cache:
                eager_calls[name_phase] += 1
                context = args[5]
                query = args[0]
                key_cache = args[3]
                value_cache = args[4]
                block_size = key_cache.shape[1]
                k_length = int(context.cu_seqlens_k[-1].item())
                block_count = (k_length + block_size - 1) // block_size
                block_ids = context.block_tables[
                    0, :block_count
                ].to(torch.long)
                dense_key = key_cache[block_ids].reshape(
                    -1,
                    key_cache.shape[2],
                    key_cache.shape[3],
                )[:k_length]
                dense_value = value_cache[block_ids].reshape(
                    -1,
                    value_cache.shape[2],
                    value_cache.shape[3],
                )[:k_length]
                engine_rows[name_phase]["layer3_full_key_local"] = (
                    dense_key.detach().float().cpu().clone()
                )
                engine_rows[name_phase]["layer3_full_value_local"] = (
                    dense_value.detach().float().cpu().clone()
                )
                repeats = kwargs["num_heads"] // dense_key.shape[1]
                repeated_key = dense_key.repeat_interleave(
                    repeats, dim=1
                )
                repeated_value = dense_value.repeat_interleave(
                    repeats, dim=1
                )
                padded_query = torch.cat(
                    (
                        query.new_zeros(k_length - query.shape[0], *query.shape[1:]),
                        query,
                    ),
                    dim=0,
                )
                row_query = padded_query.transpose(0, 1).unsqueeze(0)
                row_key = repeated_key.transpose(0, 1).unsqueeze(0)
                row_value = repeated_value.transpose(0, 1).unsqueeze(0)
                scores = torch.matmul(
                    row_query,
                    row_key.transpose(2, 3),
                ) * kwargs["scale"]
                positions = torch.arange(k_length, device=query.device)
                causal_mask = (
                    positions.unsqueeze(0) > positions.unsqueeze(1)
                )
                scores = scores.masked_fill(
                    causal_mask.view(1, 1, k_length, k_length),
                    float("-inf"),
                )
                probabilities = torch.softmax(
                    scores,
                    dim=-1,
                    dtype=torch.float32,
                ).to(query.dtype)
                padded_output = torch.matmul(
                    probabilities,
                    row_value,
                ).transpose(1, 2)
                engine_rows[name_phase][
                    "layer3_attention_padded_query_local"
                ] = (
                    padded_output[:, -64:]
                    .reshape(64, -1)
                    .detach().float().cpu().clone()
                )
                engine_rows[name_phase]["layer3_eager_q_offsets"] = (
                    context.cu_seqlens_q.detach().cpu().clone()
                )
                engine_rows[name_phase]["layer3_eager_k_offsets"] = (
                    context.cu_seqlens_k.detach().cpu().clone()
                )
                engine_rows[name_phase]["layer3_attention_raw_local"] = (
                    value[-64:].reshape(
                        64,
                        attention.local_query_heads,
                        attention.head_dim,
                    )
                    .reshape(64, -1)
                    .detach().float().cpu().clone()
                )
            return value

        engine_full_module.qwen35_cached_prefill_eager_attention = (
            wrapped_eager_attention
        )
        import tinyvllm.layers.qwen35_linear_attention as engine_linear_module

        layer4 = model.layer_stack.layers[4]
        linear4 = layer4.linear_attention
        layer4_active = [False]
        original_linear_conv = engine_linear_module.qwen35_causal_depthwise_conv
        original_linear_chunk = engine_linear_module.qwen35_gated_delta_chunk
        original_linear_norm = engine_linear_module.qwen35_gated_rmsnorm

        def save_engine_layer4(name):
            def hook(_module, _args, value):
                name_phase = phase["name"]
                if name_phase in engine_rows:
                    engine_rows[name_phase][name] = tail(value)
                return value
            return hook

        def save_engine_layer4_input(name):
            def hook(_module, args):
                name_phase = phase["name"]
                if name_phase in engine_rows:
                    engine_rows[name_phase][name] = tail(args)
            return hook

        def layer4_pre(_module, _args):
            layer4_active[0] = True

        def layer4_post(_module, _args, value):
            layer4_active[0] = False
            return value

        def wrapped_linear_conv(*args, **kwargs):
            projected_qkv = args[0]
            conv_state = args[1]
            weight = args[2]
            value, state = original_linear_conv(*args, **kwargs)
            name_phase = phase["name"]
            if layer4_active[0] and name_phase in engine_rows:
                engine_rows[name_phase]["layer4_convolved_qkv"] = tail(
                    value
                )
                engine_rows[name_phase]["layer4_conv_state"] = (
                    conv_state.detach().float().cpu().clone()
                )
                engine_rows[name_phase]["layer4_conv_combined"] = torch.cat(
                    (
                        conv_state,
                        projected_qkv.transpose(0, 1),
                    ),
                    dim=-1,
                ).detach().float().cpu().clone()
                official_full = official_native[
                    "official_layer4_projected_qkv_full"
                ].to(
                    device=projected_qkv.device,
                    dtype=projected_qkv.dtype,
                )
                full_replay, _ = original_linear_conv(
                    official_full,
                    torch.zeros_like(conv_state),
                    weight,
                )
                reconstructed_full = torch.cat(
                    (
                        official_full[:-64],
                        projected_qkv,
                    ),
                    dim=0,
                )
                reconstructed_full_replay, _ = original_linear_conv(
                    reconstructed_full,
                    torch.zeros_like(conv_state),
                    weight,
                )
                official_history3 = official_full[
                    -67:-64
                ].transpose(0, 1).contiguous()
                official_history_state = torch.cat(
                    (
                        torch.zeros_like(conv_state[:, :1]),
                        official_history3,
                    ),
                    dim=-1,
                )
                official_history_short_replay, _ = original_linear_conv(
                    projected_qkv,
                    official_history_state,
                    weight,
                )
                engine_rows[name_phase][
                    "official_history_short_replay"
                ] = tail(official_history_short_replay)
                engine_rows[name_phase][
                    "layer4_conv_effective_state"
                ] = conv_state[:, -3:].detach().float().cpu().clone()
                engine_rows[name_phase]["layer4_conv_state_first3"] = (
                    conv_state[:, :3].detach().float().cpu().clone()
                )
                engine_rows[name_phase]["layer4_conv_full_replay"] = tail(
                    full_replay
                )
                engine_rows[name_phase][
                    "layer4_conv_reconstructed_full_replay"
                ] = tail(reconstructed_full_replay)
            return value, state

        def wrapped_linear_chunk(
            query,
            key,
            value,
            a,
            b,
            A_log,
            dt_bias,
            recurrent_state_v_k,
            *,
            chunk_size=64,
        ):
            core, final_state = original_linear_chunk(
                query,
                key,
                value,
                a,
                b,
                A_log,
                dt_bias,
                recurrent_state_v_k,
                chunk_size=chunk_size,
            )
            name_phase = phase["name"]
            if layer4_active[0] and name_phase in engine_delta:
                engine_delta[name_phase].update({
                    "query": query[-64:].detach().clone(),
                    "key": key[-64:].detach().clone(),
                    "value": value[-64:].detach().clone(),
                    "a": a[-64:].detach().clone(),
                    "b": b[-64:].detach().clone(),
                    "initial_state": recurrent_state_v_k.detach().clone(),
                    "core": core[-64:].detach().clone(),
                    "final_state": final_state.detach().clone(),
                    "A_log": A_log.detach().clone(),
                    "dt_bias": dt_bias.detach().clone(),
                })
            return core, final_state

        def wrapped_linear_norm(core, gate, weight, **kwargs):
            name_phase = phase["name"]
            if layer4_active[0] and name_phase in engine_rows:
                engine_rows[name_phase]["layer4_delta_core"] = (
                    core.reshape(
                        -1,
                        linear4.local_value_heads,
                        linear4.value_head_dim,
                    )[-64:].detach().float().cpu().clone()
                )
                engine_rows[name_phase]["layer4_norm_gate"] = (
                    gate.reshape(
                        -1,
                        linear4.local_value_heads,
                        linear4.value_head_dim,
                    )[-64:].detach().float().cpu().clone()
                )
            return original_linear_norm(core, gate, weight, **kwargs)

        engine_linear_module.qwen35_causal_depthwise_conv = wrapped_linear_conv
        engine_linear_module.qwen35_gated_delta_chunk = wrapped_linear_chunk
        engine_linear_module.qwen35_gated_rmsnorm = wrapped_linear_norm
        hooks.append(linear4.register_forward_pre_hook(layer4_pre))
        hooks.append(linear4.register_forward_hook(layer4_post))
        hooks.append(
            layer4.input_layernorm.register_forward_hook(
                save_engine_layer4("layer4_input_norm")
            )
        )
        hooks.append(
            linear4.in_proj_qkv.register_forward_hook(
                save_engine_layer4("layer4_projected_qkv")
            )
        )
        hooks.append(
            linear4.in_proj_z.register_forward_hook(
                save_engine_layer4("layer4_projected_z")
            )
        )
        hooks.append(
            linear4.in_proj_a.register_forward_hook(
                save_engine_layer4("layer4_projected_a")
            )
        )
        hooks.append(
            linear4.in_proj_b.register_forward_hook(
                save_engine_layer4("layer4_projected_b")
            )
        )
        hooks.append(
            linear4.out_proj.register_forward_pre_hook(
                save_engine_layer4_input("layer4_gated_local")
            )
        )
        hooks.append(
            linear4.register_forward_hook(
                save_engine_layer4("layer4_mixer_output")
            )
        )
        hooks.append(
            layer4.post_attention_layernorm.register_forward_hook(
                save_engine_layer4("layer4_post_norm")
            )
        )
        hooks.append(
            layer4.mlp.register_forward_hook(
                save_engine_layer4("layer4_mlp_output")
            )
        )

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

        def save_engine_query_projection(_module, _args, value):
            name_phase = phase["name"]
            if name_phase in engine_rows:
                paired = value[-64:].reshape(
                    64,
                    attention.local_query_heads,
                    2 * attention.head_dim,
                )
                engine_rows[name_phase]["layer3_q_projection_local"] = (
                    paired.reshape(64, -1)
                    .detach().float().cpu().clone()
                )
                engine_rows[name_phase]["layer3_query_gate_local"] = (
                    paired[..., attention.head_dim:]
                    .reshape(64, -1)
                    .detach().float().cpu().clone()
                )
            return value

        def save_engine_rotary(_module, args, value):
            name_phase = phase["name"]
            if name_phase in engine_rows:
                position_ids = args[0]
                engine_rows[name_phase]["layer3_position_ids"] = (
                    position_ids.detach().cpu().clone()
                )
            name_phase = phase["name"]
            if name_phase in engine_rows:
                query, key = value
                engine_rows[name_phase]["layer3_rotary_query_local"] = (
                    query[-64:].reshape(
                        64,
                        attention.local_query_heads,
                        attention.head_dim,
                    )
                    .detach().float().cpu().clone()
                )
                engine_rows[name_phase]["layer3_rotary_key_local"] = (
                    key[-64:].reshape(
                        64,
                        attention.local_kv_heads,
                        attention.head_dim,
                    )
                    .detach().float().cpu().clone()
                )
            return value

        def save_engine_attention_raw(_module, _args, value):
            name_phase = phase["name"]
            if name_phase in engine_rows:
                engine_rows[name_phase]["layer3_attention_raw_local"] = (
                    value[-64:].reshape(
                        64,
                        attention.local_query_heads,
                        attention.head_dim,
                    )
                    .reshape(64, -1)
                    .detach().float().cpu().clone()
                )
            return value

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
                save_engine_query_projection
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
            attention.rotary.register_forward_hook(save_engine_rotary)
        )
        hooks.append(
            attention.attention_backend.register_forward_hook(
                save_engine_attention_raw
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
        model.layer_stack._run_full_layer = original_run_full_layer
        model.layer_stack._run_linear_layer = original_run_linear_layer
        engine_linear_module.qwen35_causal_depthwise_conv = original_linear_conv
        engine_linear_module.qwen35_gated_delta_chunk = original_linear_chunk
        engine_linear_module.qwen35_gated_rmsnorm = original_linear_norm
        for hook in hooks:
            hook.remove()
        engine_full_module.qwen35_cached_prefill_eager_attention = (
            original_eager_attention
        )
        cleanup = engine.exit()

    names = [
        "layer3_input_norm",
        "layer3_q_projection_local",
        "layer3_k_projection_local",
        "layer3_v_projection_local",
        "layer3_q_norm_local",
        "layer3_k_norm_local",
        "layer3_query_gate_local",
        "layer3_rotary_query_local",
        "layer3_rotary_key_local",
        "layer3_attention_raw_local",
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

    layer4_names = [
        "layer4_input_norm",
        "layer4_projected_qkv",
        "layer4_projected_z",
        "layer4_projected_a",
        "layer4_projected_b",
        "layer4_convolved_qkv",
        "layer4_delta_core",
        "layer4_norm_gate",
        "layer4_gated_local",
        "layer4_mixer_output",
        "layer4_post_norm",
        "layer4_mlp_output",
        "layer4_output",
    ]
    for name_phase in ("recompute", "restore"):
        engine_rows[name_phase]["layer4_output"] = (
            engine_layer_outputs[name_phase]["4"]
        )
    official_vs_recompute_layer4_components = []
    recompute_vs_restore_layer4_components = []
    for name in layer4_names:
        if (
            name not in official_rows
            or name not in engine_rows["recompute"]
            or name not in engine_rows["restore"]
        ):
            raise RuntimeError(f"missing layer4 capture: {name}")
        official_vs_recompute_layer4_components.append({
            "name": name,
            **compare(
                official_rows[name],
                engine_rows["recompute"][name],
            ),
        })
        recompute_vs_restore_layer4_components.append({
            "name": name,
            **compare(
                engine_rows["recompute"][name],
                engine_rows["restore"][name],
            ),
        })

    official_full = official_native[
        "official_layer4_projected_qkv_full"
    ].float()
    official_history = official_full[-68:-64].transpose(0, 1).contiguous()
    official_history3 = (
        official_full[-67:-64].transpose(0, 1).contiguous()
    )
    conv_shape_comparisons = {
        "engine_effective_state_vs_official_history3": compare(
            engine_rows["recompute"]["layer4_conv_effective_state"],
            official_history3,
        ),
        "engine_state_first3_vs_official_history3": compare(
            engine_rows["recompute"]["layer4_conv_state_first3"],
            official_history3,
        ),
        "engine_state_last3_vs_official_history4_first3": compare(
            engine_rows["recompute"]["layer4_conv_effective_state"],
            official_history[:, :3],
        ),
        "official_vs_official_history_short_replay": compare(
            official_rows["layer4_convolved_qkv"],
            engine_rows["recompute"]["official_history_short_replay"],
        ),
        "engine_normal_vs_official_history_short_replay": compare(
            engine_rows["recompute"]["layer4_convolved_qkv"],
            engine_rows["recompute"]["official_history_short_replay"],
        ),
        "engine_state_vs_official_history": compare(
            engine_rows["recompute"]["layer4_conv_state"],
            official_history,
        ),
        "recompute_vs_restore_conv_state": compare(
            engine_rows["recompute"]["layer4_conv_state"],
            engine_rows["restore"]["layer4_conv_state"],
        ),
        "engine_normal_vs_full_replay": compare(
            engine_rows["recompute"]["layer4_convolved_qkv"],
            engine_rows["recompute"]["layer4_conv_full_replay"],
        ),
        "official_vs_full_replay": compare(
            official_rows["layer4_convolved_qkv"],
            engine_rows["recompute"]["layer4_conv_full_replay"],
        ),
        "official_vs_reconstructed_full_replay": compare(
            official_rows["layer4_convolved_qkv"],
            engine_rows["recompute"][
                "layer4_conv_reconstructed_full_replay"
            ],
        ),
        "full_vs_reconstructed_full_replay": compare(
            engine_rows["recompute"]["layer4_conv_full_replay"],
            engine_rows["recompute"][
                "layer4_conv_reconstructed_full_replay"
            ],
        ),
    }

    engine_recompute = engine_delta["recompute"]
    official_prefix_state_v_k = (
        official_delta["prefix_state"][0, :linear4.local_value_heads]
        .transpose(-1, -2)
        .contiguous()
        .to(
            device=engine_recompute["initial_state"].device,
            dtype=engine_recompute["initial_state"].dtype,
        )
    )
    official_query = (
        official_delta["query"][:, :, :linear4.local_value_heads]
        .squeeze(0)
        .to(
            device=engine_recompute["query"].device,
            dtype=engine_recompute["query"].dtype,
        )
    )
    official_key = (
        official_delta["key"][:, :, :linear4.local_value_heads]
        .squeeze(0)
        .to(
            device=engine_recompute["key"].device,
            dtype=engine_recompute["key"].dtype,
        )
    )
    official_value = (
        official_delta["value"][:, :, :linear4.local_value_heads]
        .squeeze(0)
        .to(
            device=engine_recompute["value"].device,
            dtype=engine_recompute["value"].dtype,
        )
    )
    official_g = (
        official_delta["g"][:, :, :linear4.local_value_heads]
        .squeeze(0)
        .to(device=engine_recompute["a"].device)
    )
    official_beta = (
        official_delta["beta"][:, :, :linear4.local_value_heads]
        .squeeze(0)
        .to(device=engine_recompute["b"].device)
    )
    official_a = (
        torch.log(torch.expm1(
            (-official_g / torch.exp(
                engine_recompute["A_log"].float()
            )[None, :]).clamp_min(torch.finfo(torch.float32).tiny)
        ))
        - engine_recompute["dt_bias"].float()[None, :]
    ).to(engine_recompute["a"].dtype)
    official_b = torch.logit(
        official_beta.float().clamp(
            torch.finfo(torch.float32).eps,
            1.0 - torch.finfo(torch.float32).eps,
        )
    ).to(engine_recompute["b"].dtype)
    engine_on_official_state_core, engine_on_official_state_final = (
        original_linear_chunk(
            engine_recompute["query"],
            engine_recompute["key"],
            engine_recompute["value"],
            engine_recompute["a"],
            engine_recompute["b"],
            engine_recompute["A_log"],
            engine_recompute["dt_bias"],
            official_prefix_state_v_k,
        )
    )
    engine_on_official_inputs_core, engine_on_official_inputs_final = (
        original_linear_chunk(
            official_query,
            official_key,
            official_value,
            official_a,
            official_b,
            engine_recompute["A_log"],
            engine_recompute["dt_bias"],
            official_prefix_state_v_k,
        )
    )
    engine_state_k_v = official_delta["prefix_state"].clone()
    engine_state_k_v[:, :linear4.local_value_heads] = (
        engine_recompute["initial_state"].transpose(-1, -2)
        .unsqueeze(0)
        .to(
            device=engine_state_k_v.device,
            dtype=engine_state_k_v.dtype,
        )
    )
    official_on_engine_state_core, official_on_engine_state_final = (
        original_official_chunk(
            official_delta["query"],
            official_delta["key"],
            official_delta["value"],
            g=official_delta["g"],
            beta=official_delta["beta"],
            initial_state=engine_state_k_v,
            output_final_state=True,
            use_qk_l2norm_in_kernel=True,
        )
    )
    official_suffix_core_local = (
        official_delta["full_suffix_core"][
            0, :, :linear4.local_value_heads
        ].detach().float().cpu()
    )
    official_split_core_local = (
        official_delta["split_suffix_core"][
            0, :, :linear4.local_value_heads
        ].detach().float().cpu()
    )
    official_final_state_local_v_k = (
        official_delta["full_final_state"][
            0, :linear4.local_value_heads
        ].transpose(-1, -2).detach().float().cpu()
    )
    official_split_final_state_local_v_k = (
        official_delta["split_final_state"][
            0, :linear4.local_value_heads
        ].transpose(-1, -2).detach().float().cpu()
    )
    cross_replay_comparisons = {
        "official_full_vs_official_full_replay_core": compare(
            official_suffix_core_local,
            official_delta["full_replay_suffix_core"][
                0, :, :linear4.local_value_heads
            ].detach().float().cpu(),
        ),
        "official_full_vs_official_split_core": compare(
            official_suffix_core_local,
            official_split_core_local,
        ),
        "official_full_vs_engine_normal_core": compare(
            official_suffix_core_local,
            engine_recompute["core"].detach().float().cpu(),
        ),
        "official_prefix_state_vs_engine_prefix_state": compare(
            official_prefix_state_v_k.detach().float().cpu(),
            engine_recompute["initial_state"].detach().float().cpu(),
        ),
        "official_full_final_vs_official_split_final": compare(
            official_final_state_local_v_k,
            official_split_final_state_local_v_k,
        ),
        "official_full_final_vs_engine_normal_final": compare(
            official_final_state_local_v_k,
            engine_recompute["final_state"].detach().float().cpu(),
        ),
        "official_full_vs_engine_on_official_state_core": compare(
            official_suffix_core_local,
            engine_on_official_state_core.detach().float().cpu(),
        ),
        "official_full_vs_official_on_engine_state_core": compare(
            official_suffix_core_local,
            official_on_engine_state_core[
                0, :, :linear4.local_value_heads
            ].detach().float().cpu(),
        ),
        "official_full_vs_engine_on_official_inputs_core": compare(
            official_suffix_core_local,
            engine_on_official_inputs_core.detach().float().cpu(),
        ),
        "engine_normal_vs_engine_on_official_state_core": compare(
            engine_recompute["core"].detach().float().cpu(),
            engine_on_official_state_core.detach().float().cpu(),
        ),
        "official_full_final_vs_engine_on_official_state_final": compare(
            official_final_state_local_v_k,
            engine_on_official_state_final.detach().float().cpu(),
        ),
        "official_full_final_vs_official_on_engine_state_final": compare(
            official_final_state_local_v_k,
            official_on_engine_state_final[
                0, :linear4.local_value_heads
            ].transpose(-1, -2).detach().float().cpu(),
        ),
        "official_full_final_vs_engine_on_official_inputs_final": compare(
            official_final_state_local_v_k,
            engine_on_official_inputs_final.detach().float().cpu(),
        ),
        "engine_recompute_vs_restore_initial_state": compare(
            engine_delta["recompute"]["initial_state"].detach().float().cpu(),
            engine_delta["restore"]["initial_state"].detach().float().cpu(),
        ),
        "engine_recompute_vs_restore_core": compare(
            engine_delta["recompute"]["core"].detach().float().cpu(),
            engine_delta["restore"]["core"].detach().float().cpu(),
        ),
    }

    result = {
        "cross_replay_comparisons": cross_replay_comparisons,
        "cross_replay_shapes": {
            "official_prefix_state": list(
                official_delta["prefix_state"].shape
            ),
            "engine_prefix_state": list(
                engine_recompute["initial_state"].shape
            ),
            "official_suffix_query": list(
                official_delta["query"].shape
            ),
            "engine_suffix_query": list(
                engine_recompute["query"].shape
            ),
        },
        "conv_shape_comparisons": conv_shape_comparisons,
        "conv_shapes": {
            "official_full_projected_qkv": list(official_full.shape),
            "engine_conv_state": list(
                engine_rows["recompute"]["layer4_conv_state"].shape
            ),
            "engine_conv_combined": list(
                engine_rows["recompute"]["layer4_conv_combined"].shape
            ),
        },
        "official_vs_recompute_layer4_components": (
            official_vs_recompute_layer4_components
        ),
        "recompute_vs_restore_layer4_components": (
            recompute_vs_restore_layer4_components
        ),
        "official_vs_recompute_first_layer4_mismatch": first_mismatch(
            official_vs_recompute_layer4_components
        ),
        "recompute_vs_restore_first_layer4_mismatch": first_mismatch(
            recompute_vs_restore_layer4_components
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
        "eager_calls": eager_calls,
        "official_vs_engine_recompute_full_key": compare(
            official_rows["layer3_full_key_local"],
            engine_rows["recompute"]["layer3_full_key_local"],
        ),
        "official_vs_engine_recompute_full_value": compare(
            official_rows["layer3_full_value_local"],
            engine_rows["recompute"]["layer3_full_value_local"],
        ),
        "official_vs_engine_recompute_padded_query_attention": compare(
            official_rows["layer3_attention_raw_local"],
            engine_rows["recompute"][
                "layer3_attention_padded_query_local"
            ],
        ),
        "engine_normal_vs_padded_query_attention": compare(
            engine_rows["recompute"]["layer3_attention_raw_local"],
            engine_rows["recompute"][
                "layer3_attention_padded_query_local"
            ],
        ),
        "official_attention_raw_full_shape": official_rows[
            "layer3_attention_raw_full_shape"
        ],
        "official_direct_vs_inferred_attention_raw": compare(
            official_rows["layer3_attention_raw_local"],
            official_rows["layer3_attention_raw_inferred_local"],
        ),
        "engine_recompute_eager_q_offsets": (
            engine_rows["recompute"]["layer3_eager_q_offsets"].tolist()
        ),
        "engine_recompute_eager_k_offsets": (
            engine_rows["recompute"]["layer3_eager_k_offsets"].tolist()
        ),
        "engine_restore_eager_q_offsets": (
            engine_rows["restore"]["layer3_eager_q_offsets"].tolist()
        ),
        "engine_restore_eager_k_offsets": (
            engine_rows["restore"]["layer3_eager_k_offsets"].tolist()
        ),
        "engine_recompute_position_ids": (
            engine_rows["recompute"]["layer3_position_ids"].tolist()
        ),
        "engine_restore_position_ids": (
            engine_rows["restore"]["layer3_position_ids"].tolist()
        ),
        "schema_version": "qwen35.tp4-cached-layer4-gated-delta-cross-replay.v1",
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
        "cross_replay_comparisons": (
            result["cross_replay_comparisons"]
        ),
        "conv_shape_comparisons": result["conv_shape_comparisons"],
        "conv_shapes": result["conv_shapes"],
        "official_vs_recompute_first_layer4_mismatch": (
            result["official_vs_recompute_first_layer4_mismatch"]
        ),
        "recompute_vs_restore_first_layer4_mismatch": (
            result["recompute_vs_restore_first_layer4_mismatch"]
        ),
        "official_vs_recompute_first_layer_mismatch": (
            result["official_vs_recompute_first_layer_mismatch"]
        ),
        "recompute_vs_restore_first_layer_mismatch": (
            result["recompute_vs_restore_first_layer_mismatch"]
        ),
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
