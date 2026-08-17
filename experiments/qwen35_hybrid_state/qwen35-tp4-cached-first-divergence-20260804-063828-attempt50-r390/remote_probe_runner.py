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
        official_rank3 = {}
        official_original_recurrent8 = (
            official_linear8.recurrent_gated_delta_rule
        )

        def official_rank3_store(name, value):
            official_rank3[name] = (
                value.detach().cpu().clone()
            )

        def wrapped_official_recurrent8(*args, **kwargs):
            query, key, value = args[:3]
            initial_state = kwargs["initial_state"]
            official_rank3_store("query", query[:, -1, 12:16])
            official_rank3_store("key", key[:, -1, 12:16])
            official_rank3_store("value", value[:, -1, 12:16])
            official_rank3_store("a", kwargs["g"][:, -1, 12:16])
            official_rank3_store(
                "b_sigmoid",
                kwargs["beta"][:, -1, 12:16],
            )
            official_rank3_store(
                "initial_state_k_v",
                initial_state[:, 12:16],
            )
            result = official_original_recurrent8(*args, **kwargs)
            official_rank3_store(
                "core",
                result[0][:, -1, 12:16],
            )
            official_rank3_store(
                "final_state_k_v",
                result[1][:, 12:16],
            )
            return result

        official_linear8.recurrent_gated_delta_rule = (
            wrapped_official_recurrent8
        )
        hooks.append(
            official_linear8.norm.register_forward_pre_hook(
                lambda _module, args: (
                    official_rank3_store(
                        "gated_core",
                        args[0].reshape(
                            -1,
                            official_linear8.num_v_heads,
                            official_linear8.head_v_dim,
                        )[-1:, 12:16],
                    )
                    or official_rank3_store(
                        "gate",
                        args[1].reshape(
                            -1,
                            official_linear8.num_v_heads,
                            official_linear8.head_v_dim,
                        )[-1:, 12:16],
                    )
                    or None
                )
            )
        )
        hooks.append(
            official_linear8.norm.register_forward_hook(
                lambda _module, _args, value: (
                    official_rank3_store(
                        "gated",
                        value.reshape(
                            -1,
                            official_linear8.num_v_heads,
                            official_linear8.head_v_dim,
                        )[-1:, 12:16],
                    )
                    or value
                )
            )
        )
        official_rank3_store(
            "norm_weight",
            official_linear8.norm.weight,
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
        for hook in hooks:
            hook.remove()
        cleanup = engine.exit()

    capture_root = (
        source / "diag-rank3-layer8-r390"
    )
    rank3_recompute = torch.load(
        capture_root / "rank3-decode-03.pt",
        map_location="cpu",
        weights_only=True,
    )
    rank3_restore = torch.load(
        capture_root / "rank3-decode-07.pt",
        map_location="cpu",
        weights_only=True,
    )

    def gated_parts(core, gate, weight):
        core_fp32 = core.float()
        variance_fp32 = core_fp32.pow(2).mean(
            dim=-1,
            keepdim=True,
        )
        normalized_fp32 = core_fp32 * torch.rsqrt(
            variance_fp32 + 1e-6
        )
        normalized_cast = normalized_fp32.to(core.dtype)
        weighted = weight * normalized_cast
        silu_fp32 = torch.nn.functional.silu(gate.float())
        product_fp32 = weighted * silu_fp32
        return {
            "variance_fp32": variance_fp32,
            "normalized_fp32": normalized_fp32,
            "normalized_cast": normalized_cast,
            "weighted": weighted,
            "silu_fp32": silu_fp32,
            "product_fp32": product_fp32,
            "final_cast": product_fp32.to(core.dtype),
        }

    official_gated_parts = gated_parts(
        official_rank3["gated_core"],
        official_rank3["gate"],
        official_rank3["norm_weight"],
    )
    engine_gated_parts = gated_parts(
        rank3_recompute["gated_core"].reshape(1, 4, 128),
        rank3_recompute["gate"].reshape(1, 4, 128),
        rank3_recompute["norm_weight"],
    )

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
        "official_vs_engine_rank3": [
            {
                "name": "query",
                **compare(
                    official_rank3["query"],
                    rank3_recompute["query"].float(),
                ),
            },
            {
                "name": "key",
                **compare(
                    official_rank3["key"],
                    rank3_recompute["key"].float(),
                ),
            },
            {
                "name": "value",
                **compare(
                    official_rank3["value"],
                    rank3_recompute["value"].float(),
                ),
            },
            {
                "name": "initial_state",
                **compare(
                    official_rank3["initial_state_k_v"],
                    rank3_recompute[
                        "initial_state_physical"
                    ].transpose(-1, -2).unsqueeze(0).float(),
                ),
            },
            {
                "name": "core",
                **compare(
                    official_rank3["core"].float(),
                    rank3_recompute["core"].float(),
                ),
            },
            {
                "name": "final_state",
                **compare(
                    official_rank3["final_state_k_v"],
                    rank3_recompute[
                        "final_state_physical"
                    ].transpose(-1, -2).unsqueeze(0).float(),
                ),
            },
            {
                "name": "gated_core",
                **compare(
                    official_rank3["gated_core"].float(),
                    rank3_recompute["gated_core"].reshape(
                        1, 4, 128
                    ).float(),
                ),
            },
            {
                "name": "gate",
                **compare(
                    official_rank3["gate"].float(),
                    rank3_recompute["gate"].reshape(
                        1, 4, 128
                    ).float(),
                ),
            },
            {
                "name": "norm_weight",
                **compare(
                    official_rank3["norm_weight"].float(),
                    rank3_recompute["norm_weight"].float(),
                ),
            },
            {
                "name": "gated",
                **compare(
                    official_rank3["gated"].float(),
                    rank3_recompute["gated"].reshape(
                        1, 4, 128
                    ).float(),
                ),
            },
        ],
        "engine_recompute_vs_restore_rank3": [
            {
                "name": name,
                **compare(
                    rank3_recompute[name].float(),
                    rank3_restore[name].float(),
                ),
            }
            for name in (
                "query",
                "key",
                "value",
                "initial_state_physical",
                "core",
                "final_state_physical",
                "gated_core",
                "gate",
                "norm_weight",
                "gated",
            )
        ],
        "official_vs_engine_rank3_gated_parts": [
            {
                "name": name,
                **compare(
                    official_gated_parts[name].float(),
                    engine_gated_parts[name].float(),
                ),
            }
            for name in official_gated_parts
        ],
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
        "schema_version": "qwen35.tp4-cached-decode-step4-layer8-rank3-fixture.v9",
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
