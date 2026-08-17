from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import types
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
        official_layer3 = text.layers[3]
        official_attention = official_layer3.self_attn
        official_layer3_components = {}
        official_layer8 = text.layers[8]
        official_layer8_components = {}

        def save_official_layer8(name):
            def hook(_module, _args, value):
                tensor_value = tensor(value)
                official_layer8_components[name] = (
                    tensor_value.reshape(
                        -1,
                        tensor_value.shape[-1],
                    )[-1].detach().float().cpu().clone()
                )
                return value
            return hook

        def save_official_layer8_input(name):
            def hook(_module, args):
                tensor_value = tensor(args)
                official_layer8_components[name] = (
                    tensor_value.reshape(
                        -1,
                        tensor_value.shape[-1],
                    )[-1].detach().float().cpu().clone()
                )
            return hook

        hooks.append(
            official_layer8.input_layernorm.register_forward_pre_hook(
                save_official_layer8_input("input")
            )
        )
        hooks.append(
            official_layer8.linear_attn.register_forward_hook(
                save_official_layer8("linear_output")
            )
        )
        hooks.append(
            official_layer8.post_attention_layernorm
            .register_forward_pre_hook(
                save_official_layer8_input("attention_residual")
            )
        )
        hooks.append(
            official_layer8.post_attention_layernorm.register_forward_hook(
                save_official_layer8("post_norm")
            )
        )
        hooks.append(
            official_layer8.mlp.register_forward_hook(
                save_official_layer8("mlp")
            )
        )
        official_linear8 = official_layer8.linear_attn
        official_layer8_deep = {}
        official_original_conv_update8 = (
            official_linear8.causal_conv1d_update
        )
        official_original_recurrent8 = (
            official_linear8.recurrent_gated_delta_rule
        )

        def official_store(name, value):
            official_layer8_deep[name] = (
                value.detach().float().cpu().clone()
            )

        def official_local_projection(name, kind):
            def hook(_module, _args, value):
                row = value.reshape(-1, value.shape[-1])[-1:]
                if kind == "qkv":
                    query, key, projected_value = row.split(
                        (
                            official_linear8.key_dim,
                            official_linear8.key_dim,
                            official_linear8.value_dim,
                        ),
                        dim=-1,
                    )
                    row = torch.cat(
                        (
                            query[
                                :,
                                : official_linear8.key_dim // 4,
                            ],
                            key[
                                :,
                                : official_linear8.key_dim // 4,
                            ],
                            projected_value[
                                :,
                                : official_linear8.value_dim // 4,
                            ],
                        ),
                        dim=-1,
                    )
                elif kind == "value":
                    row = row[
                        :,
                        : official_linear8.value_dim // 4,
                    ]
                elif kind == "heads":
                    row = row[
                        :,
                        : official_linear8.num_v_heads // 4,
                    ]
                else:
                    raise RuntimeError(
                        f"unsupported official projection kind: {kind}"
                    )
                official_store(name, row)
                return value
            return hook

        def official_local_conv(value):
            row = value[:, :, -1]
            query, key, projected_value = row.split(
                (
                    official_linear8.key_dim,
                    official_linear8.key_dim,
                    official_linear8.value_dim,
                ),
                dim=-1,
            )
            return torch.cat(
                (
                    query[
                        :,
                        : official_linear8.key_dim // 4,
                    ],
                    key[
                        :,
                        : official_linear8.key_dim // 4,
                    ],
                    projected_value[
                        :,
                        : official_linear8.value_dim // 4,
                    ],
                ),
                dim=-1,
            )

        def wrapped_official_conv_update8(*args, **kwargs):
            value = official_original_conv_update8(*args, **kwargs)
            official_store(
                "conv_local",
                official_local_conv(value),
            )
            return value

        def wrapped_official_recurrent8(*args, **kwargs):
            query, key, value = args[:3]
            g = kwargs["g"]
            beta = kwargs["beta"]
            initial_state = kwargs["initial_state"]
            local_heads = official_linear8.num_v_heads // 4
            query_normalized = query * torch.rsqrt(
                (query * query).sum(dim=-1, keepdim=True) + 1e-6
            )
            key_normalized = key * torch.rsqrt(
                (key * key).sum(dim=-1, keepdim=True) + 1e-6
            )
            official_store(
                "query_local",
                query_normalized[
                    :, -1, :local_heads
                ].reshape(1, -1),
            )
            official_store(
                "key_local",
                key_normalized[
                    :, -1, :local_heads
                ].reshape(1, -1),
            )
            official_store(
                "value_local",
                value[:, -1, :local_heads].reshape(1, -1),
            )
            official_store(
                "decay_local",
                g[:, -1, :local_heads].reshape(1, -1),
            )
            official_store(
                "beta_local",
                beta[:, -1, :local_heads].reshape(1, -1),
            )
            official_store(
                "initial_state_local",
                initial_state[:, :local_heads],
            )
            result = official_original_recurrent8(*args, **kwargs)
            core, final_state = result
            official_store(
                "core_local",
                core[:, -1, :local_heads].reshape(1, -1),
            )
            official_store(
                "final_state_local",
                final_state[:, :local_heads],
            )
            return result

        official_linear8.causal_conv1d_update = (
            wrapped_official_conv_update8
        )
        official_linear8.recurrent_gated_delta_rule = (
            wrapped_official_recurrent8
        )
        hooks.append(
            official_layer8.input_layernorm.register_forward_hook(
                lambda _module, _args, value: (
                    official_store(
                        "input_norm",
                        value.reshape(-1, value.shape[-1])[-1:],
                    )
                    or value
                )
            )
        )
        hooks.append(
            official_linear8.in_proj_qkv.register_forward_hook(
                official_local_projection("qkv_projection_local", "qkv")
            )
        )
        hooks.append(
            official_linear8.in_proj_z.register_forward_hook(
                official_local_projection("z_projection_local", "value")
            )
        )
        hooks.append(
            official_linear8.in_proj_a.register_forward_hook(
                official_local_projection("a_projection_local", "heads")
            )
        )
        hooks.append(
            official_linear8.in_proj_b.register_forward_hook(
                official_local_projection("b_projection_local", "heads")
            )
        )
        hooks.append(
            official_linear8.norm.register_forward_hook(
                lambda _module, _args, value: (
                    official_store(
                        "gated_local",
                        value.reshape(
                            -1,
                            official_linear8.num_v_heads,
                            official_linear8.head_v_dim,
                        )[
                            -1:,
                            : official_linear8.num_v_heads // 4,
                        ].reshape(1, -1),
                    )
                    or value
                )
            )
        )
        hooks.append(
            official_linear8.out_proj.register_forward_pre_hook(
                lambda _module, args: official_store(
                    "out_projection_input_full",
                    tensor(args).reshape(
                        -1,
                        tensor(args).shape[-1],
                    )[-1:],
                )
            )
        )
        hooks.append(
            official_linear8.out_proj.register_forward_hook(
                lambda _module, _args, value: (
                    official_store(
                        "out_projection_output",
                        value.reshape(-1, value.shape[-1])[-1:],
                    )
                    or value
                )
            )
        )

        official_postprocess = {}

        def save_official_postprocess(name):
            def hook(_module, _args, value):
                tensor_value = tensor(value)
                official_postprocess[name] = (
                    tensor_value.reshape(
                        -1,
                        tensor_value.shape[-1],
                    )[-1].detach().float().cpu().clone()
                )
                return value
            return hook

        def save_official_postprocess_input(name):
            def hook(_module, args):
                tensor_value = tensor(args)
                official_postprocess[name] = (
                    tensor_value.reshape(
                        -1,
                        tensor_value.shape[-1],
                    )[-1].detach().float().cpu().clone()
                )
            return hook

        hooks.append(
            text.norm.register_forward_pre_hook(
                save_official_postprocess_input("final_norm_input")
            )
        )
        hooks.append(
            text.norm.register_forward_hook(
                save_official_postprocess("final_norm_output")
            )
        )
        hooks.append(
            model.lm_head.register_forward_hook(
                save_official_postprocess("logits")
            )
        )
        local_query_heads = official_attention.config.num_attention_heads // 4
        local_kv_heads = max(
            1,
            official_attention.config.num_key_value_heads // 4,
        )

        def store_official(name, value):
            official_layer3_components[name] = (
                value.detach().float().cpu().clone()
            )

        def save_official_vector(name):
            def hook(_module, _args, value):
                value = tensor(value)
                store_official(name, value.reshape(-1, value.shape[-1])[-1])
                return value
            return hook

        def save_official_vector_input(name):
            def hook(_module, args):
                value = tensor(args)
                store_official(name, value.reshape(-1, value.shape[-1])[-1])
            return hook

        def save_official_q_projection(_module, _args, value):
            paired = value[:, -1].reshape(
                official_attention.config.num_attention_heads,
                2 * official_attention.head_dim,
            )[:local_query_heads]
            store_official("q_projection", paired)
            store_official(
                "query_gate",
                paired[..., official_attention.head_dim:],
            )
            return value

        def save_official_kv_projection(name):
            def hook(_module, _args, value):
                projected = value[:, -1].reshape(
                    official_attention.config.num_key_value_heads,
                    official_attention.head_dim,
                )[:local_kv_heads]
                store_official(name, projected)
                return value
            return hook

        def save_official_q_norm(_module, _args, value):
            store_official(
                "q_norm",
                value[:, -1, :local_query_heads].squeeze(0),
            )
            return value

        def save_official_k_norm(_module, _args, value):
            store_official(
                "k_norm",
                value[:, -1, :local_kv_heads].squeeze(0),
            )
            return value

        def save_official_output_input(_module, args):
            value = tensor(args)[:, -1].reshape(
                official_attention.config.num_attention_heads,
                official_attention.head_dim,
            )[:local_query_heads]
            store_official("output_input", value)

        hooks.append(
            official_layer3.input_layernorm.register_forward_pre_hook(
                save_official_vector_input("input")
            )
        )
        hooks.append(
            official_layer3.input_layernorm.register_forward_hook(
                save_official_vector("input_norm")
            )
        )
        hooks.append(
            official_attention.q_proj.register_forward_hook(
                save_official_q_projection
            )
        )
        hooks.append(
            official_attention.k_proj.register_forward_hook(
                save_official_kv_projection("k_projection")
            )
        )
        hooks.append(
            official_attention.v_proj.register_forward_hook(
                save_official_kv_projection("v_projection")
            )
        )
        hooks.append(
            official_attention.q_norm.register_forward_hook(
                save_official_q_norm
            )
        )
        hooks.append(
            official_attention.k_norm.register_forward_hook(
                save_official_k_norm
            )
        )
        hooks.append(
            official_attention.o_proj.register_forward_pre_hook(
                save_official_output_input
            )
        )
        hooks.append(
            official_attention.o_proj.register_forward_hook(
                save_official_vector("attention_output")
            )
        )
        hooks.append(
            official_layer3.post_attention_layernorm
            .register_forward_pre_hook(
                save_official_vector_input("attention_residual")
            )
        )
        hooks.append(
            official_layer3.post_attention_layernorm.register_forward_hook(
                save_official_vector("post_norm")
            )
        )
        hooks.append(
            official_layer3.mlp.register_forward_hook(
                save_official_vector("mlp")
            )
        )

        attention_globals = (
            official_attention.forward.__func__.__globals__
        )
        original_official_eager_attention = attention_globals[
            "eager_attention_forward"
        ]
        original_apply_rotary = attention_globals[
            "apply_rotary_pos_emb"
        ]

        official_rotary_scope = {"active": False}
        official_decode_rotary_input = {
            "position_ids": None,
            "cos": None,
            "sin": None,
        }

        def enter_official_layer3_attention(_module, _args):
            official_rotary_scope["active"] = True

        def exit_official_layer3_attention(_module, _args, value):
            official_rotary_scope["active"] = False
            return value

        def save_official_rotary_embedding(_module, args, value):
            position_ids = args[1]
            cos, sin = value
            if position_ids.shape[-1] == 1:
                official_decode_rotary_input["position_ids"] = (
                    position_ids[:, 0, -1].detach().cpu().tolist()
                )
                official_decode_rotary_input["cos"] = (
                    cos[0, -1].detach().float().cpu().clone()
                )
                official_decode_rotary_input["sin"] = (
                    sin[0, -1].detach().float().cpu().clone()
                )
            return value

        hooks.append(
            official_attention.register_forward_pre_hook(
                enter_official_layer3_attention
            )
        )
        hooks.append(
            official_attention.register_forward_hook(
                exit_official_layer3_attention
            )
        )
        hooks.append(
            text.rotary_emb.register_forward_hook(
                save_official_rotary_embedding
            )
        )

        def wrapped_apply_rotary(query, key, *args, **kwargs):
            rotated_query, rotated_key = original_apply_rotary(
                query,
                key,
                *args,
                **kwargs,
            )
            if official_rotary_scope["active"]:
                store_official(
                    "rotary_query",
                    rotated_query[
                        0,
                        :local_query_heads,
                        -1,
                    ],
                )
                store_official(
                    "rotary_key",
                    rotated_key[
                        0,
                        :local_kv_heads,
                        -1,
                    ],
                )
            return rotated_query, rotated_key

        attention_globals["apply_rotary_pos_emb"] = wrapped_apply_rotary

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
            next_token = prefill.logits[:, -1].argmax(
                dim=-1,
                keepdim=True,
            )
            continuation = prefill
            for _decode_step in range(4):
                continuation = model(
                    input_ids=next_token,
                    past_key_values=continuation.past_key_values,
                    use_cache=True,
                    return_dict=True,
                )
                next_token = continuation.logits[:, -1].argmax(
                    dim=-1,
                    keepdim=True,
                )
    finally:
        for hook in layer_output_hooks:
            hook.remove()
        for hook in hooks:
            hook.remove()
        if "official_linear8" in locals():
            official_linear8.causal_conv1d_update = (
                official_original_conv_update8
            )
            official_linear8.recurrent_gated_delta_rule = (
                official_original_recurrent8
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
        engine_layer3 = model.layer_stack.layers[3]
        engine_attention = engine_layer3.full_attention
        engine_layer8 = model.layer_stack.layers[8]
        engine_layer8_components = {
            "recompute": {},
            "restore": {},
        }

        def save_engine_layer8(name):
            def hook(_module, _args, value):
                name_phase = phase["name"]
                if name_phase in engine_layer8_components:
                    tensor_value = tensor(value)
                    engine_layer8_components[name_phase][name] = (
                        tensor_value.reshape(
                            -1,
                            tensor_value.shape[-1],
                        )[-1].detach().float().cpu().clone()
                    )
                return value
            return hook

        def save_engine_layer8_input(name):
            def hook(_module, args):
                name_phase = phase["name"]
                if name_phase in engine_layer8_components:
                    tensor_value = tensor(args)
                    engine_layer8_components[name_phase][name] = (
                        tensor_value.reshape(
                            -1,
                            tensor_value.shape[-1],
                        )[-1].detach().float().cpu().clone()
                    )
            return hook

        hooks.append(
            engine_layer8.input_layernorm.register_forward_pre_hook(
                save_engine_layer8_input("input")
            )
        )
        hooks.append(
            engine_layer8.linear_attention.register_forward_hook(
                save_engine_layer8("linear_output")
            )
        )
        hooks.append(
            engine_layer8.post_attention_layernorm
            .register_forward_pre_hook(
                save_engine_layer8_input("attention_residual")
            )
        )
        hooks.append(
            engine_layer8.post_attention_layernorm.register_forward_hook(
                save_engine_layer8("post_norm")
            )
        )
        hooks.append(
            engine_layer8.mlp.register_forward_hook(
                save_engine_layer8("mlp")
            )
        )
        engine_linear8 = engine_layer8.linear_attention
        engine_layer8_deep = {
            "recompute": {},
            "restore": {},
        }
        engine_layer8_replay = {
            "recompute": {},
            "restore": {},
        }
        engine_original_conv8 = (
            engine_linear_module.qwen35_causal_depthwise_conv
        )
        engine_original_chunk8 = (
            engine_linear_module.qwen35_gated_delta_chunk
        )
        engine_original_recurrent8 = (
            engine_linear_module.qwen35_gated_delta_recurrent
        )
        engine_original_gated8 = (
            engine_linear_module.qwen35_gated_rmsnorm
        )
        engine_original_out_linear8 = (
            engine_linear8.out_proj._linear_forward_unpartitioned
        )

        def engine_store(name, value):
            name_phase = phase["name"]
            if name_phase in engine_layer8_deep:
                engine_layer8_deep[name_phase][name] = (
                    value.detach().float().cpu().clone()
                )

        def engine_projection(name):
            def hook(_module, _args, value):
                if value.shape[0] == 1:
                    engine_store(
                        name,
                        value.reshape(-1, value.shape[-1])[-1:],
                    )
                return value
            return hook

        def batch_shaped_recurrent_replay(
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
            query = query * torch.rsqrt(
                (query * query).sum(dim=-1, keepdim=True) + 1e-6
            )
            key = key * torch.rsqrt(
                (key * key).sum(dim=-1, keepdim=True) + 1e-6
            )
            query = query.float() * (query.shape[-1] ** -0.5)
            key = key.float()
            value = value.float()
            beta = torch.sigmoid(b).float()
            decay = (
                -torch.exp(A_log.float())
                * torch.nn.functional.softplus(
                    a.float() + dt_bias.float()
                )
            )
            query, key, value, beta, decay = [
                tensor.unsqueeze(0).transpose(1, 2).contiguous()
                for tensor in (query, key, value, beta, decay)
            ]
            state = (
                recurrent_state.float()
                .transpose(-1, -2)
                .unsqueeze(0)
            )
            outputs = []
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
                beta_token = beta[:, :, token_index].unsqueeze(-1)
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
                outputs.append(
                    (
                        state
                        * query_token.unsqueeze(-1)
                    ).sum(dim=-2)
                )
            return (
                torch.stack(outputs, dim=2)
                .transpose(1, 2)
                .squeeze(0)
                .to(initial_dtype)
            )

        def wrapped_engine_conv8(
            projected_qkv,
            conv_state,
            weight,
            **kwargs,
        ):
            result = engine_original_conv8(
                projected_qkv,
                conv_state,
                weight,
                **kwargs,
            )
            if (
                weight.data_ptr()
                == engine_linear8.conv_weight.data_ptr()
                and projected_qkv.shape[0] == 1
            ):
                engine_store("conv_local", result[0].reshape(1, -1))
            return result

        def capture_engine_core8(function):
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
                is_target = (
                    A_log.data_ptr()
                    == engine_linear8.A_log.data_ptr()
                    and query.shape[0] == 1
                )
                if is_target:
                    query_fp32 = (
                        query
                        * torch.rsqrt(
                            (query * query).sum(
                                dim=-1,
                                keepdim=True,
                            )
                            + 1e-6
                        )
                    ).float()
                    key_fp32 = (
                        key
                        * torch.rsqrt(
                            (key * key).sum(
                                dim=-1,
                                keepdim=True,
                            )
                            + 1e-6
                        )
                    ).float()
                    decay = (
                        -torch.exp(A_log.float())
                        * torch.nn.functional.softplus(
                            a.float() + dt_bias.float()
                        )
                    )
                    beta = torch.sigmoid(b)
                    engine_store(
                        "query_local",
                        query_fp32[-1:].reshape(1, -1),
                    )
                    engine_store(
                        "key_local",
                        key_fp32[-1:].reshape(1, -1),
                    )
                    engine_store(
                        "value_local",
                        value[-1:].reshape(1, -1),
                    )
                    engine_store(
                        "decay_local",
                        decay[-1:].reshape(1, -1),
                    )
                    engine_store(
                        "beta_local",
                        beta[-1:].reshape(1, -1),
                    )
                    engine_store(
                        "initial_state_local",
                        recurrent_state.transpose(
                            -1, -2
                        ).unsqueeze(0),
                    )
                    name_phase = phase["name"]
                    engine_layer8_replay[name_phase]["core"] = (
                        batch_shaped_recurrent_replay(
                            query,
                            key,
                            value,
                            a,
                            b,
                            A_log,
                            dt_bias,
                            recurrent_state,
                        )
                    )
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
                if is_target:
                    engine_store(
                        "core_local",
                        result[0][-1:].reshape(1, -1),
                    )
                    engine_store(
                        "final_state_local",
                        result[1].transpose(
                            -1, -2
                        ).unsqueeze(0),
                    )
                return result
            return wrapped

        def wrapped_engine_gated8(core, gate, weight, **kwargs):
            result = engine_original_gated8(
                core,
                gate,
                weight,
                **kwargs,
            )
            if (
                weight.data_ptr()
                == engine_linear8.norm_weight.data_ptr()
            ):
                engine_store("gated_local", result.reshape(1, -1))
                name_phase = phase["name"]
                replay_core = engine_layer8_replay[
                    name_phase
                ]["core"].reshape_as(core)
                engine_layer8_replay[name_phase]["gated"] = (
                    engine_original_gated8(
                        replay_core,
                        gate,
                        weight,
                        **kwargs,
                    ).reshape(1, -1)
                )
            return result

        def wrapped_engine_out_linear8(self, gathered, bias):
            if gathered.shape[0] == 1:
                engine_store(
                    "out_projection_input_full",
                    gathered.reshape(1, -1),
                )
                name_phase = phase["name"]
                replay_local = engine_layer8_replay[
                    name_phase
                ]["gated"]
                replay_shards = [
                    torch.empty_like(replay_local)
                    for _ in range(
                        torch.distributed.get_world_size()
                    )
                ]
                torch.distributed.all_gather(
                    replay_shards,
                    replay_local,
                )
                engine_store(
                    "batch_replay_gated_full",
                    torch.cat(replay_shards, dim=-1),
                )
            return engine_original_out_linear8(gathered, bias)

        engine_linear_module.qwen35_causal_depthwise_conv = (
            wrapped_engine_conv8
        )
        engine_linear_module.qwen35_gated_delta_chunk = (
            capture_engine_core8(engine_original_chunk8)
        )
        engine_linear_module.qwen35_gated_delta_recurrent = (
            capture_engine_core8(engine_original_recurrent8)
        )
        engine_linear_module.qwen35_gated_rmsnorm = (
            wrapped_engine_gated8
        )
        engine_linear8.out_proj._linear_forward_unpartitioned = (
            types.MethodType(
                wrapped_engine_out_linear8,
                engine_linear8.out_proj,
            )
        )
        hooks.append(
            engine_layer8.input_layernorm.register_forward_hook(
                engine_projection("input_norm")
            )
        )
        hooks.append(
            engine_linear8.in_proj_qkv.register_forward_hook(
                engine_projection("qkv_projection_local")
            )
        )
        hooks.append(
            engine_linear8.in_proj_z.register_forward_hook(
                engine_projection("z_projection_local")
            )
        )
        hooks.append(
            engine_linear8.in_proj_a.register_forward_hook(
                engine_projection("a_projection_local")
            )
        )
        hooks.append(
            engine_linear8.in_proj_b.register_forward_hook(
                engine_projection("b_projection_local")
            )
        )
        hooks.append(
            engine_linear8.out_proj.register_forward_hook(
                engine_projection("out_projection_output")
            )
        )

        engine_postprocess = {
            "recompute": {},
            "restore": {},
        }

        def save_engine_postprocess(name):
            def hook(_module, _args, value):
                name_phase = phase["name"]
                if name_phase not in engine_postprocess:
                    return value
                if value is None:
                    return value
                tensor_value = tensor(value)
                engine_postprocess[name_phase][name] = (
                    tensor_value.reshape(
                        -1,
                        tensor_value.shape[-1],
                    )[-1].detach().float().cpu().clone()
                )
                return value
            return hook

        def save_engine_postprocess_input(name):
            def hook(_module, args):
                name_phase = phase["name"]
                if name_phase not in engine_postprocess:
                    return
                tensor_value = tensor(args)
                engine_postprocess[name_phase][name] = (
                    tensor_value.reshape(
                        -1,
                        tensor_value.shape[-1],
                    )[-1].detach().float().cpu().clone()
                )
            return hook

        hooks.append(
            model.final_norm.register_forward_pre_hook(
                save_engine_postprocess_input("final_norm_input")
            )
        )
        hooks.append(
            model.final_norm.register_forward_hook(
                save_engine_postprocess("final_norm_output")
            )
        )
        hooks.append(
            model.lm_head.register_forward_hook(
                save_engine_postprocess("logits")
            )
        )
        engine_layer3_components = {
            "recompute": {},
            "restore": {},
        }

        def store_engine(name, value):
            name_phase = phase["name"]
            if name_phase in engine_layer3_components:
                engine_layer3_components[name_phase][name] = (
                    value.detach().float().cpu().clone()
                )

        def save_engine_vector(name):
            def hook(_module, _args, value):
                value = tensor(value)
                if value.shape[0] == 1:
                    store_engine(name, value.reshape(-1, value.shape[-1])[-1])
                return value
            return hook

        def save_engine_vector_input(name):
            def hook(_module, args):
                value = tensor(args)
                if value.shape[0] == 1:
                    store_engine(name, value.reshape(-1, value.shape[-1])[-1])
            return hook

        def save_engine_q_projection(_module, _args, value):
            if value.shape[0] == 1:
                paired = value[-1].reshape(
                    engine_attention.local_query_heads,
                    2 * engine_attention.head_dim,
                )
                store_engine("q_projection", paired)
                store_engine(
                    "query_gate",
                    paired[..., engine_attention.head_dim:],
                )
            return value

        def save_engine_kv_projection(name):
            def hook(_module, _args, value):
                if value.shape[0] == 1:
                    store_engine(
                        name,
                        value[-1].reshape(
                            engine_attention.local_kv_heads,
                            engine_attention.head_dim,
                        ),
                    )
                return value
            return hook

        def save_engine_head_tensor(name):
            def hook(_module, _args, value):
                if value.shape[0] == 1:
                    store_engine(name, value[-1])
                return value
            return hook

        engine_decode_position_ids = {
            "recompute": None,
            "restore": None,
        }
        engine_decode_rotary_inputs = {
            "recompute": {"cos": None, "sin": None},
            "restore": {"cos": None, "sin": None},
        }

        def save_engine_rotary(module, args, value):
            query, key = value
            if query.shape[0] == 1:
                store_engine(
                    "rotary_query",
                    query[-1].reshape(
                        engine_attention.local_query_heads,
                        engine_attention.head_dim,
                    ),
                )
                store_engine(
                    "rotary_key",
                    key[-1].reshape(
                        engine_attention.local_kv_heads,
                        engine_attention.head_dim,
                    ),
                )
                name_phase = phase["name"]
                if name_phase in engine_decode_position_ids:
                    engine_decode_position_ids[name_phase] = (
                        args[0].detach().cpu().tolist()
                    )
                    frequencies = module._selected_frequencies(args[0])
                    embedding = torch.cat(
                        (frequencies, frequencies),
                        dim=-1,
                    )
                    engine_decode_rotary_inputs[name_phase]["cos"] = (
                        embedding.cos()[-1].to(query.dtype)
                        .detach().float().cpu().clone()
                    )
                    engine_decode_rotary_inputs[name_phase]["sin"] = (
                        embedding.sin()[-1].to(query.dtype)
                        .detach().float().cpu().clone()
                    )
            return value

        def save_engine_output_input(_module, args):
            value = tensor(args)
            if value.shape[0] == 1:
                store_engine(
                    "output_input",
                    value[-1].reshape(
                        engine_attention.local_query_heads,
                        engine_attention.head_dim,
                    ),
                )

        hooks.append(
            engine_layer3.input_layernorm.register_forward_pre_hook(
                save_engine_vector_input("input")
            )
        )
        hooks.append(
            engine_layer3.input_layernorm.register_forward_hook(
                save_engine_vector("input_norm")
            )
        )
        hooks.append(
            engine_attention.q_projection.register_forward_hook(
                save_engine_q_projection
            )
        )
        hooks.append(
            engine_attention.k_projection.register_forward_hook(
                save_engine_kv_projection("k_projection")
            )
        )
        hooks.append(
            engine_attention.v_projection.register_forward_hook(
                save_engine_kv_projection("v_projection")
            )
        )
        hooks.append(
            engine_attention.q_norm.register_forward_hook(
                save_engine_head_tensor("q_norm")
            )
        )
        hooks.append(
            engine_attention.k_norm.register_forward_hook(
                save_engine_head_tensor("k_norm")
            )
        )
        hooks.append(
            engine_attention.rotary.register_forward_hook(
                save_engine_rotary
            )
        )
        hooks.append(
            engine_attention.output_projection.register_forward_pre_hook(
                save_engine_output_input
            )
        )
        hooks.append(
            engine_attention.output_projection.register_forward_hook(
                save_engine_vector("attention_output")
            )
        )
        hooks.append(
            engine_layer3.post_attention_layernorm
            .register_forward_pre_hook(
                save_engine_vector_input("attention_residual")
            )
        )
        hooks.append(
            engine_layer3.post_attention_layernorm.register_forward_hook(
                save_engine_vector("post_norm")
            )
        )
        hooks.append(
            engine_layer3.mlp.register_forward_hook(
                save_engine_vector("mlp")
            )
        )

        engine.clear_qwen35_hybrid_prefix_caches(
            timeout_s=configuration.timeout_s,
        )
        phase["name"] = "recompute"
        probe._run_request(
            engine, prompt, 5, timeout_s=configuration.timeout_s
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
            engine, prompt, 5, timeout_s=configuration.timeout_s
        )
        phase["name"] = None
    finally:
        model.layer_stack._run_full_layer = original_run_full_layer
        model.layer_stack._run_linear_layer = original_run_linear_layer
        if "engine_linear8" in locals():
            engine_linear_module.qwen35_causal_depthwise_conv = (
                engine_original_conv8
            )
            engine_linear_module.qwen35_gated_delta_chunk = (
                engine_original_chunk8
            )
            engine_linear_module.qwen35_gated_delta_recurrent = (
                engine_original_recurrent8
            )
            engine_linear_module.qwen35_gated_rmsnorm = (
                engine_original_gated8
            )
            engine_linear8.out_proj._linear_forward_unpartitioned = (
                engine_original_out_linear8
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
        nonzero = torch.nonzero(
            difference.reshape(-1),
            as_tuple=False,
        )
        first_index = (
            None
            if nonzero.numel() == 0
            else int(nonzero[0].item())
        )
        comparison = {
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
        if first_index is not None:
            comparison.update({
                "first_nonzero_flat_index": first_index,
                "first_nonzero_left": float(
                    left.reshape(-1)[first_index].item()
                ),
                "first_nonzero_right": float(
                    right.reshape(-1)[first_index].item()
                ),
            })
        if left.ndim == 2 and left.shape[1] % 4 == 0:
            shard_width = left.shape[1] // 4
            comparison["quarter_shards"] = []
            for shard_index in range(4):
                start = shard_index * shard_width
                end = start + shard_width
                shard_difference = difference[:, start:end]
                comparison["quarter_shards"].append({
                    "shard_index": shard_index,
                    "start": start,
                    "end": end,
                    "max_abs_diff": float(
                        shard_difference.max().item()
                    ),
                    "nonzero_count": int(
                        torch.count_nonzero(
                            shard_difference
                        ).item()
                    ),
                })
        return comparison

    layer8_names = (
        "input",
        "linear_output",
        "attention_residual",
        "post_norm",
        "mlp",
    )
    for name in layer8_names:
        if name not in official_layer8_components:
            raise RuntimeError(f"missing official layer8 capture: {name}")
        for name_phase in ("recompute", "restore"):
            if name not in engine_layer8_components[name_phase]:
                raise RuntimeError(
                    f"missing Engine {name_phase} layer8 capture: {name}"
                )

    deep_names = (
        "input_norm",
        "qkv_projection_local",
        "z_projection_local",
        "a_projection_local",
        "b_projection_local",
        "conv_local",
        "query_local",
        "key_local",
        "value_local",
        "decay_local",
        "beta_local",
        "initial_state_local",
        "core_local",
        "final_state_local",
        "gated_local",
        "out_projection_input_full",
        "batch_replay_gated_full",
        "out_projection_output",
    )
    for name in deep_names:
        official_name = (
            "out_projection_input_full"
            if name == "batch_replay_gated_full"
            else name
        )
        if official_name not in official_layer8_deep:
            raise RuntimeError(
                f"missing official layer8 deep capture: {name}"
            )
        for name_phase in ("recompute", "restore"):
            if name not in engine_layer8_deep[name_phase]:
                raise RuntimeError(
                    f"missing Engine {name_phase} layer8 deep "
                    f"capture: {name}"
                )

    postprocess_names = (
        "final_norm_input",
        "final_norm_output",
        "logits",
    )
    for name in postprocess_names:
        if name not in official_postprocess:
            raise RuntimeError(
                f"missing official postprocess capture: {name}"
            )
        for name_phase in ("recompute", "restore"):
            if name not in engine_postprocess[name_phase]:
                raise RuntimeError(
                    f"missing Engine {name_phase} postprocess capture: "
                    f"{name}"
                )

    component_names = (
        "input",
        "input_norm",
        "q_projection",
        "k_projection",
        "v_projection",
        "q_norm",
        "k_norm",
        "query_gate",
        "rotary_query",
        "rotary_key",
        "output_input",
        "attention_output",
        "attention_residual",
        "post_norm",
        "mlp",
    )
    official_vs_recompute_layer3_components = []
    recompute_vs_restore_layer3_components = []
    for name in component_names:
        if (
            name not in official_layer3_components
            or name not in engine_layer3_components["recompute"]
            or name not in engine_layer3_components["restore"]
        ):
            raise RuntimeError(
                f"missing layer3 decode component capture: {name}"
            )
        official_vs_recompute_layer3_components.append({
            "name": name,
            **compare(
                official_layer3_components[name],
                engine_layer3_components["recompute"][name],
            ),
        })
        recompute_vs_restore_layer3_components.append({
            "name": name,
            **compare(
                engine_layer3_components["recompute"][name],
                engine_layer3_components["restore"][name],
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

    if any(
        official_decode_rotary_input[name] is None
        for name in ("position_ids", "cos", "sin")
    ):
        raise RuntimeError("official decode rotary input was not captured")
    if any(
        engine_decode_rotary_inputs["recompute"][name] is None
        for name in ("cos", "sin")
    ):
        raise RuntimeError("Engine recompute rotary input was not captured")

    result = {
        "official_decode_position_ids": (
            official_decode_rotary_input["position_ids"]
        ),
        "engine_decode_position_ids": engine_decode_position_ids,
        "official_vs_recompute_rotary_cos": compare(
            official_decode_rotary_input["cos"],
            engine_decode_rotary_inputs["recompute"]["cos"],
        ),
        "official_vs_recompute_rotary_sin": compare(
            official_decode_rotary_input["sin"],
            engine_decode_rotary_inputs["recompute"]["sin"],
        ),
        "official_vs_recompute_layer8_deep": [
            {
                "name": name,
                **compare(
                    official_layer8_deep[
                        "out_projection_input_full"
                        if name == "batch_replay_gated_full"
                        else name
                    ],
                    engine_layer8_deep["recompute"][name],
                ),
            }
            for name in deep_names
        ],
        "recompute_vs_restore_layer8_deep": [
            {
                "name": name,
                **compare(
                    engine_layer8_deep["recompute"][name],
                    engine_layer8_deep["restore"][name],
                ),
            }
            for name in deep_names
        ],
        "official_vs_recompute_layer8_deep_first_mismatch": (
            first_component_mismatch([
                {
                    "name": name,
                    **compare(
                        official_layer8_deep[
                            "out_projection_input_full"
                            if name == "batch_replay_gated_full"
                            else name
                        ],
                        engine_layer8_deep["recompute"][name],
                    ),
                }
                for name in deep_names
            ])
        ),
        "recompute_vs_restore_layer8_deep_first_mismatch": (
            first_component_mismatch([
                {
                    "name": name,
                    **compare(
                        engine_layer8_deep["recompute"][name],
                        engine_layer8_deep["restore"][name],
                    ),
                }
                for name in deep_names
            ])
        ),
        "official_vs_recompute_layer8": [
            {
                "name": name,
                **compare(
                    official_layer8_components[name],
                    engine_layer8_components["recompute"][name],
                ),
            }
            for name in layer8_names
        ],
        "recompute_vs_restore_layer8": [
            {
                "name": name,
                **compare(
                    engine_layer8_components["recompute"][name],
                    engine_layer8_components["restore"][name],
                ),
            }
            for name in layer8_names
        ],
        "official_vs_recompute_postprocess": [
            {
                "name": name,
                **compare(
                    official_postprocess[name],
                    engine_postprocess["recompute"][name],
                ),
            }
            for name in postprocess_names
        ],
        "recompute_vs_restore_postprocess": [
            {
                "name": name,
                **compare(
                    engine_postprocess["recompute"][name],
                    engine_postprocess["restore"][name],
                ),
            }
            for name in postprocess_names
        ],
        "official_vs_recompute_layer3_components": (
            official_vs_recompute_layer3_components
        ),
        "recompute_vs_restore_layer3_components": (
            recompute_vs_restore_layer3_components
        ),
        "official_vs_recompute_first_component_mismatch": (
            first_component_mismatch(
                official_vs_recompute_layer3_components
            )
        ),
        "recompute_vs_restore_first_component_mismatch": (
            first_component_mismatch(
                recompute_vs_restore_layer3_components
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
        "schema_version": "qwen35.tp4-cached-decode-step4-layer8-deep.v6",
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
