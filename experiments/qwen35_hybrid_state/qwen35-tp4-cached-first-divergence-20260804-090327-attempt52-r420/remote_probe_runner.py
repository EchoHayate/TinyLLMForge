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
        official_layer7 = text.layers[7]
        official_attention = official_layer7.self_attn
        official_layer7_components = {}
        local_query_heads = official_attention.config.num_attention_heads // 4
        local_kv_heads = max(
            1,
            official_attention.config.num_key_value_heads // 4,
        )

        def store_official(name, value):
            official_layer7_components[name] = (
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

        official_rank1_head2 = {}

        def save_official_q_projection(_module, _args, value):
            paired_all = value[:, -1].reshape(
                official_attention.config.num_attention_heads,
                2 * official_attention.head_dim,
            )
            official_rank1_head2["gate"] = (
                paired_all[2, official_attention.head_dim:]
                .detach().float().cpu().clone()
            )
            paired = paired_all[:local_query_heads]
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
            value_all = tensor(args)[:, -1].reshape(
                official_attention.config.num_attention_heads,
                official_attention.head_dim,
            )
            official_rank1_head2["post_gate"] = (
                value_all[2].detach().float().cpu().clone()
            )
            store_official(
                "output_input",
                value_all[:local_query_heads],
            )

        hooks.append(
            official_layer7.input_layernorm.register_forward_pre_hook(
                save_official_vector_input("input")
            )
        )
        hooks.append(
            official_layer7.input_layernorm.register_forward_hook(
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
            official_layer7.post_attention_layernorm
            .register_forward_pre_hook(
                save_official_vector_input("attention_residual")
            )
        )
        hooks.append(
            official_layer7.post_attention_layernorm.register_forward_hook(
                save_official_vector("post_norm")
            )
        )
        hooks.append(
            official_layer7.mlp.register_forward_hook(
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

        def enter_official_layer7_attention(_module, _args):
            official_rotary_scope["active"] = True

        def exit_official_layer7_attention(_module, _args, value):
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
                enter_official_layer7_attention
            )
        )
        hooks.append(
            official_attention.register_forward_hook(
                exit_official_layer7_attention
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

        official_attention_core = {}

        def tensor_layout(value):
            return {
                "shape": list(value.shape),
                "stride": list(value.stride()),
                "contiguous": bool(value.is_contiguous()),
                "dtype": str(value.dtype),
            }

        def wrapped_official_eager_attention(
            module,
            query,
            key,
            value,
            attention_mask,
            scaling,
            *args,
            **kwargs,
        ):
            output_value = original_official_eager_attention(
                module,
                query,
                key,
                value,
                attention_mask,
                scaling,
                *args,
                **kwargs,
            )
            if official_rotary_scope["active"] and query.shape[2] == 1:
                attention_output = (
                    output_value[0]
                    if isinstance(output_value, tuple)
                    else output_value
                )
                local_query = query[:, :local_query_heads]
                local_key = key[:, :local_kv_heads]
                local_value = value[:, :local_kv_heads]
                repeated_key = attention_globals["repeat_kv"](
                    local_key,
                    local_query_heads // local_kv_heads,
                )
                repeated_value = attention_globals["repeat_kv"](
                    local_value,
                    local_query_heads // local_kv_heads,
                )
                scores = torch.matmul(
                    local_query,
                    repeated_key.transpose(2, 3),
                ) * scaling
                if attention_mask is not None:
                    scores = scores + attention_mask
                probabilities = torch.softmax(
                    scores,
                    dim=-1,
                    dtype=torch.float32,
                ).to(query.dtype)
                replay_output = torch.matmul(
                    probabilities,
                    repeated_value,
                ).transpose(1, 2).contiguous()
                full_repeated_key = attention_globals["repeat_kv"](
                    key,
                    (
                        official_attention.config.num_attention_heads
                        // official_attention.config.num_key_value_heads
                    ),
                )
                full_repeated_value = attention_globals["repeat_kv"](
                    value,
                    (
                        official_attention.config.num_attention_heads
                        // official_attention.config.num_key_value_heads
                    ),
                )
                rank1_query = query[:, 2:3]
                rank1_key = full_repeated_key[:, 2:3]
                rank1_value = full_repeated_value[:, 2:3]
                rank1_scores = torch.matmul(
                    rank1_query,
                    rank1_key.transpose(2, 3),
                ) * scaling
                if attention_mask is not None:
                    rank1_scores = rank1_scores + attention_mask
                rank1_probabilities = torch.softmax(
                    rank1_scores,
                    dim=-1,
                    dtype=torch.float32,
                ).to(query.dtype)
                rank1_replay_output = torch.matmul(
                    rank1_probabilities,
                    rank1_value,
                )
                official_rank1_head2.update({
                    "query": (
                        rank1_query.detach().float().cpu().clone()
                    ),
                    "key": rank1_key.detach().float().cpu().clone(),
                    "value": (
                        rank1_value.detach().float().cpu().clone()
                    ),
                    "scores": (
                        rank1_scores.detach().float().cpu().clone()
                    ),
                    "probabilities": (
                        rank1_probabilities
                        .detach().float().cpu().clone()
                    ),
                    "replay_output": (
                        rank1_replay_output[0, 0, 0]
                        .detach().float().cpu().clone()
                    ),
                    "pre_gate": (
                        attention_output[0, 0, 2]
                        .detach().float().cpu().clone()
                    ),
                })
                official_attention_core.update({
                    "query": (
                        local_query.detach().float().cpu().clone()
                    ),
                    "key": local_key.detach().float().cpu().clone(),
                    "value": local_value.detach().float().cpu().clone(),
                    "repeated_key": (
                        repeated_key.detach().float().cpu().clone()
                    ),
                    "repeated_value": (
                        repeated_value.detach().float().cpu().clone()
                    ),
                    "scores": scores.detach().float().cpu().clone(),
                    "probabilities": (
                        probabilities.detach().float().cpu().clone()
                    ),
                    "output": (
                        attention_output[
                            :, :, :local_query_heads
                        ].detach().float().cpu().clone()
                    ),
                    "replay_output": (
                        replay_output.detach().float().cpu().clone()
                    ),
                    "layouts": {
                        "query": tensor_layout(local_query),
                        "key": tensor_layout(local_key),
                        "value": tensor_layout(local_value),
                        "repeated_key": tensor_layout(repeated_key),
                        "repeated_value": tensor_layout(repeated_value),
                        "scores": tensor_layout(scores),
                        "probabilities": tensor_layout(probabilities),
                        "output": tensor_layout(attention_output),
                    },
                    "mask_shape": (
                        None
                        if attention_mask is None
                        else list(attention_mask.shape)
                    ),
                    "scaling": float(scaling),
                })
            return output_value

        attention_globals["apply_rotary_pos_emb"] = wrapped_apply_rotary
        attention_globals["eager_attention_forward"] = (
            wrapped_official_eager_attention
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
            next_token = prefill.logits[:, -1].argmax(
                dim=-1,
                keepdim=True,
            )
            continuation = prefill
            for _decode_step in range(19):
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
        engine_layer7 = model.layer_stack.layers[7]
        engine_attention = engine_layer7.full_attention
        engine_layer7_components = {
            "recompute": {},
            "restore": {},
        }

        def store_engine(name, value):
            name_phase = phase["name"]
            if name_phase in engine_layer7_components:
                engine_layer7_components[name_phase][name] = (
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
            engine_layer7.input_layernorm.register_forward_pre_hook(
                save_engine_vector_input("input")
            )
        )
        hooks.append(
            engine_layer7.input_layernorm.register_forward_hook(
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
        engine_attention_scope = {"active": False}
        engine_attention_core = {
            "recompute": {},
            "restore": {},
        }
        original_engine_cached_decode = (
            engine_full_module.qwen35_cached_decode_eager_attention
        )

        def enter_engine_layer7_attention(_module, _args):
            engine_attention_scope["active"] = True

        def exit_engine_layer7_attention(_module, _args, value):
            engine_attention_scope["active"] = False
            return value

        def wrapped_engine_cached_decode(
            query,
            current_key,
            current_value,
            key_cache,
            value_cache,
            context,
            *,
            num_heads,
            head_dim,
            scale,
        ):
            output_value = original_engine_cached_decode(
                query,
                current_key,
                current_value,
                key_cache,
                value_cache,
                context,
                num_heads=num_heads,
                head_dim=head_dim,
                scale=scale,
            )
            name_phase = phase["name"]
            if (
                engine_attention_scope["active"]
                and query.shape[0] == 1
                and name_phase in engine_attention_core
            ):
                block_size = key_cache.shape[1]
                context_length = int(context.context_lens[0].item())
                block_count = (
                    context_length + block_size - 1
                ) // block_size
                block_ids = context.block_tables[
                    0, :block_count
                ].to(torch.long)
                key = key_cache[block_ids].reshape(
                    -1,
                    key_cache.shape[2],
                    key_cache.shape[3],
                )[:context_length]
                cached_value = value_cache[block_ids].reshape(
                    -1,
                    value_cache.shape[2],
                    value_cache.shape[3],
                )[:context_length]
                repeats = num_heads // key.shape[1]
                repeated_key_token_major = key.repeat_interleave(
                    repeats,
                    dim=1,
                )
                repeated_value_token_major = cached_value.repeat_interleave(
                    repeats,
                    dim=1,
                )
                row_query = query.transpose(0, 1).unsqueeze(0)
                repeated_key = (
                    repeated_key_token_major.transpose(0, 1).unsqueeze(0)
                )
                repeated_value = (
                    repeated_value_token_major.transpose(0, 1).unsqueeze(0)
                )
                scores = torch.matmul(
                    row_query,
                    repeated_key.transpose(2, 3),
                ) * scale
                probabilities = torch.softmax(
                    scores,
                    dim=-1,
                    dtype=torch.float32,
                ).to(query.dtype)
                replay_output = torch.matmul(
                    probabilities,
                    repeated_value,
                ).transpose(1, 2).reshape(1, num_heads * head_dim)

                contiguous_key = repeated_key.contiguous()
                contiguous_value = repeated_value.contiguous()
                contiguous_scores = torch.matmul(
                    row_query.contiguous(),
                    contiguous_key.transpose(2, 3),
                ) * scale
                contiguous_probabilities = torch.softmax(
                    contiguous_scores,
                    dim=-1,
                    dtype=torch.float32,
                ).to(query.dtype)
                contiguous_output = torch.matmul(
                    contiguous_probabilities,
                    contiguous_value,
                ).transpose(1, 2).reshape(1, num_heads * head_dim)
                engine_attention_core[name_phase] = {
                    "query": (
                        query.transpose(0, 1).unsqueeze(0)
                        .detach().float().cpu().clone()
                    ),
                    "key": (
                        key.transpose(0, 1).unsqueeze(0)
                        .detach().float().cpu().clone()
                    ),
                    "value": (
                        cached_value.transpose(0, 1).unsqueeze(0)
                        .detach().float().cpu().clone()
                    ),
                    "repeated_key": (
                        repeated_key.detach().float().cpu().clone()
                    ),
                    "repeated_value": (
                        repeated_value.detach().float().cpu().clone()
                    ),
                    "scores": scores.detach().float().cpu().clone(),
                    "probabilities": (
                        probabilities.detach().float().cpu().clone()
                    ),
                    "output": (
                        output_value.detach().float().cpu().clone()
                    ),
                    "replay_output": (
                        replay_output.detach().float().cpu().clone()
                    ),
                    "contiguous_scores": (
                        contiguous_scores.detach().float().cpu().clone()
                    ),
                    "contiguous_probabilities": (
                        contiguous_probabilities
                        .detach().float().cpu().clone()
                    ),
                    "contiguous_output": (
                        contiguous_output.detach().float().cpu().clone()
                    ),
                    "layouts": {
                        "query": tensor_layout(query),
                        "key": tensor_layout(
                            key.transpose(0, 1).unsqueeze(0)
                        ),
                        "value": tensor_layout(
                            cached_value.transpose(0, 1).unsqueeze(0)
                        ),
                        "repeated_key": tensor_layout(repeated_key),
                        "repeated_value": tensor_layout(repeated_value),
                        "contiguous_key": tensor_layout(contiguous_key),
                        "contiguous_value": tensor_layout(
                            contiguous_value
                        ),
                        "scores": tensor_layout(scores),
                        "probabilities": tensor_layout(probabilities),
                        "output": tensor_layout(output_value),
                    },
                    "context_length": context_length,
                    "block_ids": block_ids.detach().cpu().tolist(),
                    "scaling": float(scale),
                }
            return output_value

        hooks.append(
            engine_attention.register_forward_pre_hook(
                enter_engine_layer7_attention
            )
        )
        hooks.append(
            engine_attention.register_forward_hook(
                exit_engine_layer7_attention
            )
        )
        engine_full_module.qwen35_cached_decode_eager_attention = (
            wrapped_engine_cached_decode
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
            engine_layer7.post_attention_layernorm
            .register_forward_pre_hook(
                save_engine_vector_input("attention_residual")
            )
        )
        hooks.append(
            engine_layer7.post_attention_layernorm.register_forward_hook(
                save_engine_vector("post_norm")
            )
        )
        hooks.append(
            engine_layer7.mlp.register_forward_hook(
                save_engine_vector("mlp")
            )
        )

        engine.clear_qwen35_hybrid_prefix_caches(
            timeout_s=configuration.timeout_s,
        )
        phase["name"] = "recompute"
        probe._run_request(
            engine, prompt, 20, timeout_s=configuration.timeout_s
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
            engine, prompt, 20, timeout_s=configuration.timeout_s
        )
        phase["name"] = None
    finally:
        model.layer_stack._run_full_layer = original_run_full_layer
        model.layer_stack._run_linear_layer = original_run_linear_layer
        for hook in hooks:
            hook.remove()
        if "original_engine_cached_decode" in locals():
            engine_full_module.qwen35_cached_decode_eager_attention = (
                original_engine_cached_decode
            )
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
    official_vs_recompute_layer7_components = []
    recompute_vs_restore_layer7_components = []
    for name in component_names:
        if (
            name not in official_layer7_components
            or name not in engine_layer7_components["recompute"]
            or name not in engine_layer7_components["restore"]
        ):
            raise RuntimeError(
                f"missing layer7 decode component capture: {name}"
            )
        official_vs_recompute_layer7_components.append({
            "name": name,
            **compare(
                official_layer7_components[name],
                engine_layer7_components["recompute"][name],
            ),
        })
        recompute_vs_restore_layer7_components.append({
            "name": name,
            **compare(
                engine_layer7_components["recompute"][name],
                engine_layer7_components["restore"][name],
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

    rank1_fixture_path = Path('/tmp/qwen35_attempt52_r420_rank1_layer7_step19_core.pt')
    if not rank1_fixture_path.is_file():
        raise RuntimeError("rank1 layer7 step19 fixture missing")
    rank1_fixture = torch.load(
        rank1_fixture_path,
        map_location="cpu",
        weights_only=True,
    )
    rank1_core_names = (
        "query",
        "key",
        "value",
        "scores",
        "probabilities",
        "replay_output",
        "pre_gate",
        "gate",
        "post_gate",
    )
    for name in rank1_core_names:
        if name not in official_rank1_head2:
            raise RuntimeError(
                f"official rank1 head2 {name} missing"
            )
        if name not in rank1_fixture:
            raise RuntimeError(
                f"Engine rank1 head2 {name} missing"
            )

    attention_core_names = (
        "query",
        "key",
        "value",
        "repeated_key",
        "repeated_value",
        "scores",
        "probabilities",
        "output",
        "replay_output",
        "layouts",
        "scaling",
    )
    if any(
        name not in official_attention_core
        for name in attention_core_names
    ):
        raise RuntimeError("official layer7 attention core was not captured")
    for name_phase in ("recompute", "restore"):
        if any(
            name not in engine_attention_core[name_phase]
            for name in attention_core_names + (
                "contiguous_scores",
                "contiguous_probabilities",
                "contiguous_output",
                "context_length",
                "block_ids",
            )
        ):
            raise RuntimeError(
                f"Engine {name_phase} layer7 attention core "
                "was not captured"
            )

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
        "rank1_head2_fixture_metadata": {
            "rank": rank1_fixture["rank"],
            "layer_index": rank1_fixture["layer_index"],
            "context_length": rank1_fixture["context_length"],
        },
        "official_vs_rank1_head2_pre_gate": compare(
            official_rank1_head2["pre_gate"],
            rank1_fixture["pre_gate"],
        ),
        "official_vs_rank1_head2_gate": compare(
            official_rank1_head2["gate"],
            rank1_fixture["gate"],
        ),
        "official_vs_rank1_head2_post_gate": compare(
            official_rank1_head2["post_gate"],
            rank1_fixture["post_gate"],
        ),
        "official_vs_rank1_head2_query": compare(
            official_rank1_head2["query"],
            rank1_fixture["query"],
        ),
        "official_vs_rank1_head2_key": compare(
            official_rank1_head2["key"],
            rank1_fixture["key"],
        ),
        "official_vs_rank1_head2_value": compare(
            official_rank1_head2["value"],
            rank1_fixture["value"],
        ),
        "official_vs_rank1_head2_scores": compare(
            official_rank1_head2["scores"],
            rank1_fixture["scores"],
        ),
        "official_vs_rank1_head2_probabilities": compare(
            official_rank1_head2["probabilities"],
            rank1_fixture["probabilities"],
        ),
        "official_vs_rank1_head2_replay_output": compare(
            official_rank1_head2["replay_output"],
            rank1_fixture["local_replay_output"],
        ),
        "official_vs_rank1_head2_global_replay_output": compare(
            official_rank1_head2["replay_output"],
            rank1_fixture["global_replay_output"],
        ),
        "engine_rank1_head2_actual_vs_local_replay": compare(
            rank1_fixture["pre_gate"],
            rank1_fixture["local_replay_output"],
        ),
        "engine_rank1_head2_actual_vs_global_replay": compare(
            rank1_fixture["pre_gate"],
            rank1_fixture["global_replay_output"],
        ),
        "official_attention_layouts": (
            official_attention_core["layouts"]
        ),
        "engine_attention_layouts": {
            name_phase: engine_attention_core[name_phase]["layouts"]
            for name_phase in ("recompute", "restore")
        },
        "official_attention_scaling": (
            official_attention_core["scaling"]
        ),
        "engine_attention_scaling": {
            name_phase: engine_attention_core[name_phase]["scaling"]
            for name_phase in ("recompute", "restore")
        },
        "engine_attention_context_length": {
            name_phase: engine_attention_core[name_phase][
                "context_length"
            ]
            for name_phase in ("recompute", "restore")
        },
        "engine_attention_block_ids": {
            name_phase: engine_attention_core[name_phase]["block_ids"]
            for name_phase in ("recompute", "restore")
        },
        "official_vs_recompute_attention_query": compare(
            official_attention_core["query"],
            engine_attention_core["recompute"]["query"],
        ),
        "official_vs_recompute_attention_key": compare(
            official_attention_core["key"],
            engine_attention_core["recompute"]["key"],
        ),
        "official_vs_recompute_attention_value": compare(
            official_attention_core["value"],
            engine_attention_core["recompute"]["value"],
        ),
        "official_vs_recompute_attention_repeated_key": compare(
            official_attention_core["repeated_key"],
            engine_attention_core["recompute"]["repeated_key"],
        ),
        "official_vs_recompute_attention_repeated_value": compare(
            official_attention_core["repeated_value"],
            engine_attention_core["recompute"]["repeated_value"],
        ),
        "official_vs_recompute_attention_scores": compare(
            official_attention_core["scores"],
            engine_attention_core["recompute"]["scores"],
        ),
        "official_vs_recompute_attention_probabilities": compare(
            official_attention_core["probabilities"],
            engine_attention_core["recompute"]["probabilities"],
        ),
        "official_vs_recompute_attention_output": compare(
            official_attention_core["output"],
            engine_attention_core["recompute"]["output"].reshape(
                1,
                1,
                engine_attention.local_query_heads,
                engine_attention.head_dim,
            ),
        ),
        "official_vs_recompute_attention_replay_output": compare(
            official_attention_core["replay_output"],
            engine_attention_core["recompute"]["replay_output"].reshape(
                1,
                1,
                engine_attention.local_query_heads,
                engine_attention.head_dim,
            ),
        ),
        "official_vs_recompute_attention_contiguous_scores": compare(
            official_attention_core["scores"],
            engine_attention_core["recompute"]["contiguous_scores"],
        ),
        "official_vs_recompute_attention_contiguous_probabilities": compare(
            official_attention_core["probabilities"],
            engine_attention_core["recompute"][
                "contiguous_probabilities"
            ],
        ),
        "official_vs_recompute_attention_contiguous_output": compare(
            official_attention_core["output"],
            engine_attention_core["recompute"][
                "contiguous_output"
            ].reshape(
                1,
                1,
                engine_attention.local_query_heads,
                engine_attention.head_dim,
            ),
        ),
        "engine_attention_output_vs_replay": compare(
            engine_attention_core["recompute"]["output"],
            engine_attention_core["recompute"]["replay_output"],
        ),
        "recompute_vs_restore_attention_key": compare(
            engine_attention_core["recompute"]["key"],
            engine_attention_core["restore"]["key"],
        ),
        "recompute_vs_restore_attention_value": compare(
            engine_attention_core["recompute"]["value"],
            engine_attention_core["restore"]["value"],
        ),
        "official_vs_recompute_layer7_components": (
            official_vs_recompute_layer7_components
        ),
        "recompute_vs_restore_layer7_components": (
            recompute_vs_restore_layer7_components
        ),
        "official_vs_recompute_first_component_mismatch": (
            first_component_mismatch(
                official_vs_recompute_layer7_components
            )
        ),
        "recompute_vs_restore_first_component_mismatch": (
            first_component_mismatch(
                recompute_vs_restore_layer7_components
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
        "schema_version": "qwen35.tp4-cached-decode-step19-rank1-head2-core.v7",
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
