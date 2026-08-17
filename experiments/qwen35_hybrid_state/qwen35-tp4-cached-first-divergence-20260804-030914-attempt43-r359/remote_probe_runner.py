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
        official_layer3 = text.layers[3]
        official_attention = official_layer3.self_attn
        official_layer3_components = {}
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

        def summarize_token_major(tensor_value, block_size):
            token_count = tensor_value.shape[0]
            sample_indices = sorted(set(
                index
                for index in (
                    0,
                    1,
                    block_size - 1,
                    block_size,
                    token_count - block_size - 1,
                    token_count - 2,
                    token_count - 1,
                )
                if 0 <= index < token_count
            ))
            sample_index_tensor = torch.tensor(
                sample_indices,
                dtype=torch.long,
                device=tensor_value.device,
            )
            block_count = (
                token_count + block_size - 1
            ) // block_size
            padded_count = block_count * block_size
            if padded_count == token_count:
                padded = tensor_value
            else:
                padded = torch.cat(
                    (
                        tensor_value,
                        tensor_value.new_zeros(
                            padded_count - token_count,
                            *tensor_value.shape[1:],
                        ),
                    ),
                    dim=0,
                )
            blocks = padded.reshape(
                block_count,
                block_size,
                tensor_value.shape[1],
                tensor_value.shape[2],
            ).float()
            block_stats = torch.stack(
                (
                    blocks.sum(dim=(1, 3)),
                    blocks.abs().sum(dim=(1, 3)),
                    blocks.square().sum(dim=(1, 3)),
                    blocks.amax(dim=(1, 3)),
                    blocks.amin(dim=(1, 3)),
                ),
                dim=-1,
            )
            return {
                "token_count": token_count,
                "sample_indices": sample_indices,
                "samples": (
                    tensor_value.index_select(0, sample_index_tensor)
                    .detach().float().cpu().clone()
                ),
                "block_stats": (
                    block_stats.detach().cpu().clone()
                ),
            }

        official_attention_core = {}

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
                token_major_key = (
                    key[0, :local_kv_heads].transpose(0, 1)
                )
                token_major_value = (
                    value[0, :local_kv_heads].transpose(0, 1)
                )
                official_attention_core.update({
                    "query": (
                        query[0, :local_query_heads, -1]
                        .detach().float().cpu().clone()
                    ),
                    "key_summary": summarize_token_major(
                        token_major_key,
                        256,
                    ),
                    "value_summary": summarize_token_major(
                        token_major_value,
                        256,
                    ),
                    "output": (
                        attention_output[0, -1, :local_query_heads]
                        .detach().float().cpu().clone()
                    ),
                    "mask_shape": (
                        None
                        if attention_mask is None
                        else list(attention_mask.shape)
                    ),
                    "mask_tail": (
                        None
                        if attention_mask is None
                        else attention_mask[
                            0, 0, -1, -8:
                        ].detach().float().cpu().clone()
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
        engine_attention_core = {
            "recompute": {},
            "restore": {},
        }

        def save_engine_attention_core(_module, args, value):
            query, _current_key, _current_value = args
            name_phase = phase["name"]
            if query.shape[0] != 1 or name_phase not in engine_attention_core:
                return value
            context = engine_full_module.get_context()
            block_size = engine_attention.k_cache.shape[1]
            context_length = int(context.context_lens[0].item())
            block_count = (
                context_length + block_size - 1
            ) // block_size
            block_ids = context.block_tables[
                0, :block_count
            ].to(torch.long)
            key = engine_attention.k_cache[block_ids].reshape(
                -1,
                engine_attention.local_kv_heads,
                engine_attention.head_dim,
            )[:context_length]
            cached_value = engine_attention.v_cache[
                block_ids
            ].reshape(
                -1,
                engine_attention.local_kv_heads,
                engine_attention.head_dim,
            )[:context_length]
            query_heads = query.reshape(
                engine_attention.local_query_heads,
                engine_attention.head_dim,
            )
            flash_output = value.reshape(
                engine_attention.local_query_heads,
                engine_attention.head_dim,
            )
            repeats = (
                engine_attention.local_query_heads
                // engine_attention.local_kv_heads
            )
            eager_rows = []
            score_maxima = []
            score_argmax = []
            probability_sums = []
            scale = engine_attention.head_dim ** -0.5
            for query_head in range(
                engine_attention.local_query_heads
            ):
                kv_head = query_head // repeats
                scores = torch.matmul(
                    key[:, kv_head].float(),
                    query_heads[query_head].float(),
                ) * scale
                probabilities = torch.softmax(
                    scores,
                    dim=-1,
                    dtype=torch.float32,
                ).to(query.dtype)
                eager_rows.append(
                    torch.matmul(
                        probabilities.unsqueeze(0),
                        cached_value[:, kv_head],
                    ).squeeze(0)
                )
                score_maxima.append(scores.max())
                score_argmax.append(scores.argmax())
                probability_sums.append(
                    probabilities.float().sum()
                )
            eager_output = torch.stack(eager_rows, dim=0)
            engine_attention_core[name_phase] = {
                "query": (
                    query_heads.detach().float().cpu().clone()
                ),
                "key_summary": summarize_token_major(
                    key,
                    block_size,
                ),
                "value_summary": summarize_token_major(
                    cached_value,
                    block_size,
                ),
                "flash_output": (
                    flash_output.detach().float().cpu().clone()
                ),
                "eager_output": (
                    eager_output.detach().float().cpu().clone()
                ),
                "score_maxima": (
                    torch.stack(score_maxima)
                    .detach().cpu().tolist()
                ),
                "score_argmax": (
                    torch.stack(score_argmax)
                    .detach().cpu().tolist()
                ),
                "probability_sums": (
                    torch.stack(probability_sums)
                    .detach().cpu().tolist()
                ),
                "context_length": context_length,
                "block_ids": block_ids.detach().cpu().tolist(),
                "scaling": float(scale),
            }
            return value

        hooks.append(
            engine_attention.attention_backend.register_forward_hook(
                save_engine_attention_core
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

    required_official_attention_core = (
        "query",
        "key_summary",
        "value_summary",
        "output",
        "mask_shape",
        "mask_tail",
        "scaling",
    )
    if any(
        name not in official_attention_core
        for name in required_official_attention_core
    ):
        raise RuntimeError("official attention summary was not captured")
    for name_phase in ("recompute", "restore"):
        if any(
            name not in engine_attention_core[name_phase]
            for name in (
                "query",
                "key_summary",
                "value_summary",
                "flash_output",
                "eager_output",
                "score_maxima",
                "score_argmax",
                "probability_sums",
                "context_length",
                "block_ids",
                "scaling",
            )
        ):
            raise RuntimeError(
                f"Engine {name_phase} attention summary was not captured"
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
        "official_attention_mask_shape": (
            official_attention_core["mask_shape"]
        ),
        "official_attention_mask_tail": (
            None
            if official_attention_core["mask_tail"] is None
            else official_attention_core["mask_tail"].tolist()
        ),
        "official_attention_scaling": (
            official_attention_core["scaling"]
        ),
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
        "engine_attention_scaling": {
            name_phase: engine_attention_core[name_phase]["scaling"]
            for name_phase in ("recompute", "restore")
        },
        "engine_attention_score_maxima": {
            name_phase: engine_attention_core[name_phase][
                "score_maxima"
            ]
            for name_phase in ("recompute", "restore")
        },
        "engine_attention_score_argmax": {
            name_phase: engine_attention_core[name_phase][
                "score_argmax"
            ]
            for name_phase in ("recompute", "restore")
        },
        "engine_attention_probability_sums": {
            name_phase: engine_attention_core[name_phase][
                "probability_sums"
            ]
            for name_phase in ("recompute", "restore")
        },
        "official_vs_recompute_attention_query": compare(
            official_attention_core["query"],
            engine_attention_core["recompute"]["query"],
        ),
        "official_vs_recompute_key_samples": compare(
            official_attention_core["key_summary"]["samples"],
            engine_attention_core["recompute"][
                "key_summary"
            ]["samples"],
        ),
        "official_vs_recompute_key_block_stats": compare(
            official_attention_core["key_summary"]["block_stats"],
            engine_attention_core["recompute"][
                "key_summary"
            ]["block_stats"],
        ),
        "official_vs_recompute_value_samples": compare(
            official_attention_core["value_summary"]["samples"],
            engine_attention_core["recompute"][
                "value_summary"
            ]["samples"],
        ),
        "official_vs_recompute_value_block_stats": compare(
            official_attention_core["value_summary"]["block_stats"],
            engine_attention_core["recompute"][
                "value_summary"
            ]["block_stats"],
        ),
        "attention_sample_indices": {
            "official_key": official_attention_core[
                "key_summary"
            ]["sample_indices"],
            "official_value": official_attention_core[
                "value_summary"
            ]["sample_indices"],
            "engine_recompute_key": engine_attention_core[
                "recompute"
            ]["key_summary"]["sample_indices"],
            "engine_recompute_value": engine_attention_core[
                "recompute"
            ]["value_summary"]["sample_indices"],
        },
        "official_vs_recompute_attention_flash_output": compare(
            official_attention_core["output"],
            engine_attention_core["recompute"]["flash_output"],
        ),
        "official_vs_recompute_attention_eager_replay": compare(
            official_attention_core["output"],
            engine_attention_core["recompute"]["eager_output"],
        ),
        "engine_flash_vs_eager_attention_output": compare(
            engine_attention_core["recompute"]["flash_output"],
            engine_attention_core["recompute"]["eager_output"],
        ),
        "recompute_vs_restore_key_samples": compare(
            engine_attention_core["recompute"][
                "key_summary"
            ]["samples"],
            engine_attention_core["restore"][
                "key_summary"
            ]["samples"],
        ),
        "recompute_vs_restore_key_block_stats": compare(
            engine_attention_core["recompute"][
                "key_summary"
            ]["block_stats"],
            engine_attention_core["restore"][
                "key_summary"
            ]["block_stats"],
        ),
        "recompute_vs_restore_value_samples": compare(
            engine_attention_core["recompute"][
                "value_summary"
            ]["samples"],
            engine_attention_core["restore"][
                "value_summary"
            ]["samples"],
        ),
        "recompute_vs_restore_value_block_stats": compare(
            engine_attention_core["recompute"][
                "value_summary"
            ]["block_stats"],
            engine_attention_core["restore"][
                "value_summary"
            ]["block_stats"],
        ),
        "recompute_vs_restore_attention_eager_output": compare(
            engine_attention_core["recompute"]["eager_output"],
            engine_attention_core["restore"]["eager_output"],
        ),
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
        "schema_version": "qwen35.tp4-cached-decode-step1-layer3-attention-summary.v4",
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
