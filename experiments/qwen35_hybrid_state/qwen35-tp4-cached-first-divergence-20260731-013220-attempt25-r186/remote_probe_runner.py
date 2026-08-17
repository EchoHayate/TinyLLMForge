from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import torch
import torch.nn.functional as F
import tinyvllm.layers.gated_delta as engine_gated_module


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

    def trace_chunk_core(
        destination,
        prefix,
        query,
        key,
        value,
        g,
        beta,
        initial_state,
        *,
        selected_heads,
        chunk_size=64,
    ):
        initial_dtype = query.dtype
        raw_query = query[:, -chunk_size:, :selected_heads]
        raw_key = key[:, -chunk_size:, :selected_heads]
        raw_value = value[:, -chunk_size:, :selected_heads]
        raw_g = g[:, -chunk_size:, :selected_heads]
        raw_beta = beta[:, -chunk_size:, :selected_heads]

        def save_raw(name, tensor):
            destination[prefix + name] = (
                tensor[0].transpose(0, 1)
                .detach().float().cpu().clone()
            )

        save_raw("raw_query", raw_query)
        save_raw("raw_key", raw_key)
        save_raw("raw_value", raw_value)
        save_raw("raw_g", raw_g.unsqueeze(-1))
        save_raw("raw_beta", raw_beta.unsqueeze(-1))
        query = engine_gated_module.qwen35_l2norm(query)
        key = engine_gated_module.qwen35_l2norm(key)
        query, key, value, beta, g = [
            tensor.transpose(1, 2).contiguous().to(torch.float32)
            for tensor in (query, key, value, beta, g)
        ]
        batch_size, num_heads, sequence_length, key_dim = key.shape
        value_dim = value.shape[-1]
        pad_size = (
            chunk_size - sequence_length % chunk_size
        ) % chunk_size
        query = F.pad(query, (0, 0, 0, pad_size))
        key = F.pad(key, (0, 0, 0, pad_size))
        value = F.pad(value, (0, 0, 0, pad_size))
        beta = F.pad(beta, (0, pad_size))
        g = F.pad(g, (0, pad_size))
        total_sequence_length = sequence_length + pad_size
        query = query * (1 / (query.shape[-1] ** 0.5))
        value_beta = value * beta.unsqueeze(-1)
        key_beta = key * beta.unsqueeze(-1)
        query, key, value, key_beta, value_beta = [
            tensor.reshape(
                tensor.shape[0],
                tensor.shape[1],
                -1,
                chunk_size,
                tensor.shape[-1],
            )
            for tensor in (
                query,
                key,
                value,
                key_beta,
                value_beta,
            )
        ]
        g = g.reshape(g.shape[0], g.shape[1], -1, chunk_size)
        diagonal_mask = torch.triu(
            torch.ones(
                chunk_size,
                chunk_size,
                dtype=torch.bool,
                device=query.device,
            ),
            diagonal=0,
        )
        g = g.cumsum(dim=-1)
        decay_mask = (
            (
                g.unsqueeze(-1) - g.unsqueeze(-2)
            ).tril().exp().float()
        ).tril()
        attention = -(
            (key_beta @ key.transpose(-1, -2)) * decay_mask
        ).masked_fill(diagonal_mask, 0)
        for index in range(1, chunk_size):
            row = attention[..., index, :index].clone()
            sub = attention[..., :index, :index].clone()
            attention[..., index, :index] = (
                row + (row.unsqueeze(-1) * sub).sum(-2)
            )
        attention = attention + torch.eye(
            chunk_size,
            dtype=attention.dtype,
            device=attention.device,
        )
        transformed_value = attention @ value_beta
        key_cumulative_decay = attention @ (
            key_beta * g.exp().unsqueeze(-1)
        )
        state = (
            torch.zeros(
                batch_size,
                num_heads,
                key_dim,
                value_dim,
                dtype=transformed_value.dtype,
                device=transformed_value.device,
            )
            if initial_state is None
            else initial_state.to(transformed_value)
        )
        output = torch.zeros_like(transformed_value)
        final_chunk = total_sequence_length // chunk_size - 1

        def save(name, tensor):
            selected = tensor[0, :selected_heads]
            destination[prefix + name] = (
                selected.detach().float().cpu().clone()
            )

        for index in range(total_sequence_length // chunk_size):
            query_chunk = query[:, :, index]
            key_chunk = key[:, :, index]
            value_chunk = transformed_value[:, :, index]
            intra = (
                query_chunk @ key_chunk.transpose(-1, -2)
            ) * decay_mask[:, :, index]
            previous = key_cumulative_decay[:, :, index] @ state
            value_new = value_chunk - previous
            inter = (
                query_chunk
                * g[:, :, index, :, None].exp()
            ) @ state
            intra_value = intra @ value_new
            if index == final_chunk:
                save("state_before", state)
                save("query_chunk", query_chunk)
                save("key_chunk", key_chunk)
                save("value_chunk", value_chunk)
                save(
                    "key_cumulative_decay",
                    key_cumulative_decay[:, :, index],
                )
                save("decay_mask", decay_mask[:, :, index])
                save("intra", intra)
                save("previous", previous)
                save("value_new", value_new)
                save("inter", inter)
                save("intra_value", intra_value)
            output[:, :, index] = inter + intra_value
            state = (
                state
                * g[:, :, index, -1, None, None].exp()
                + (
                    key_chunk
                    * (
                        g[:, :, index, -1, None]
                        - g[:, :, index]
                    ).exp()[..., None]
                ).transpose(-1, -2)
                @ value_new
            )
            if index == final_chunk:
                save("output_chunk", output[:, :, index])
                save("state_after", state)
        return (
            output.reshape(
                output.shape[0],
                output.shape[1],
                -1,
                value_dim,
            )[:, :, :sequence_length]
            .transpose(1, 2)
            .contiguous()
            .to(initial_dtype),
            state,
        )

    official_rows = {}
    official_backend = official.TransformersGreedyReferenceBackend(
        configuration,
        gpu_index=configuration.gpu_indices[0],
    )
    hooks = []
    try:
        model = official_backend._model()
        text = model.model
        layer = text.layers[0]
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
                official_rows["layer0_conv_local"] = (
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
                official_rows["layer0_conv_local"] = (
                    official_local_conv(value)
                )
                return value

            linear.causal_conv1d_fn = wrapped_official_conv

        def wrapped_official_chunk(*args, **kwargs):
            value = official_original_chunk(*args, **kwargs)
            query, key, projected_value = args[:3]
            local_heads = linear.num_v_heads // 4
            initial_state = kwargs.get("initial_state")
            traced, _ = trace_chunk_core(
                official_rows,
                "layer0_trace_",
                query[:, :, :local_heads],
                key[:, :, :local_heads],
                projected_value[:, :, :local_heads],
                kwargs["g"][:, :, :local_heads],
                kwargs["beta"][:, :, :local_heads],
                (
                    None
                    if initial_state is None
                    else initial_state[:, :local_heads]
                ),
                selected_heads=local_heads,
            )
            official_rows["layer0_trace_replay_output"] = (
                traced[:, -1, : linear.num_v_heads // 4]
                .reshape(1, -1)
                .detach().float().cpu().clone()
            )
            core = value[0]
            official_rows["layer0_core_local"] = (
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
                save_official("layer0_input_norm")
            )
        )
        hooks.append(
            linear.in_proj_qkv.register_forward_hook(
                save_official_local_projection(
                    "layer0_in_proj_qkv_local",
                    "qkv",
                )
            )
        )
        hooks.append(
            linear.in_proj_z.register_forward_hook(
                save_official_local_projection(
                    "layer0_in_proj_z_local",
                    "value",
                )
            )
        )
        hooks.append(
            linear.in_proj_b.register_forward_hook(
                save_official_local_projection(
                    "layer0_in_proj_b_local",
                    "heads",
                )
            )
        )
        hooks.append(
            linear.in_proj_a.register_forward_hook(
                save_official_local_projection(
                    "layer0_in_proj_a_local",
                    "heads",
                )
            )
        )
        hooks.append(
            linear.register_forward_hook(
                save_official("layer0_linear_output")
            )
        )
        hooks.append(
            linear.norm.register_forward_hook(
                lambda _module, _args, value: (
                    official_rows.__setitem__(
                        "layer0_gated_local",
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
        hooks.append(
            linear.out_proj.register_forward_hook(
                save_official("layer0_out_proj")
            )
        )
        hooks.append(
            layer.post_attention_layernorm.register_forward_pre_hook(
                save_official_input("layer0_attention_residual")
            )
        )
        hooks.append(
            layer.post_attention_layernorm.register_forward_hook(
                save_official("layer0_post_norm")
            )
        )
        hooks.append(
            layer.mlp.gate_proj.register_forward_hook(
                save_official("layer0_gate_proj")
            )
        )
        hooks.append(
            layer.mlp.up_proj.register_forward_hook(
                save_official("layer0_up_proj")
            )
        )
        hooks.append(
            layer.mlp.register_forward_hook(
                save_official("layer0_mlp_output")
            )
        )
        hooks.append(
            text.layers[0].input_layernorm.register_forward_pre_hook(
                save_official_input("layer0_output")
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
        layer = model.layer_stack.layers[0]
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
                engine_rows[name_phase]["layer0_gate_proj"] = last_row(
                    gate
                )
                engine_rows[name_phase]["layer0_up_proj"] = last_row(up)
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
                engine_rows[name_phase]["layer0_conv_local"] = last_row(
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
                    traced, _ = trace_chunk_core(
                        engine_rows[name_phase],
                        "layer0_trace_",
                        query.unsqueeze(0),
                        key.unsqueeze(0),
                        value.unsqueeze(0),
                        (
                            -A_log.float().exp()
                            * F.softplus(a.float() + dt_bias)
                        ).unsqueeze(0),
                        b.sigmoid().unsqueeze(0),
                        recurrent_state.transpose(
                            -1,
                            -2,
                        ).unsqueeze(0),
                        selected_heads=linear.local_value_heads,
                    )
                    engine_rows[name_phase][
                        "layer0_trace_replay_output"
                    ] = (
                        traced[:, -1].reshape(1, -1)
                        .detach().float().cpu().clone()
                    )
                name_phase = phase["name"]
                if (
                    name_phase in engine_rows
                    and A_log.data_ptr() == linear.A_log.data_ptr()
                ):
                    engine_rows[name_phase]["layer0_core_local"] = (
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
                engine_rows[name_phase]["layer0_gated_local"] = (
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
                save_engine("layer0_input_norm")
            )
        )
        hooks.append(
            linear.in_proj_qkv.register_forward_hook(
                save_engine("layer0_in_proj_qkv_local")
            )
        )
        hooks.append(
            linear.in_proj_z.register_forward_hook(
                save_engine("layer0_in_proj_z_local")
            )
        )
        hooks.append(
            linear.in_proj_b.register_forward_hook(
                save_engine("layer0_in_proj_b_local")
            )
        )
        hooks.append(
            linear.in_proj_a.register_forward_hook(
                save_engine("layer0_in_proj_a_local")
            )
        )
        hooks.append(
            linear.register_forward_hook(
                save_engine("layer0_linear_output")
            )
        )
        hooks.append(
            linear.out_proj.register_forward_hook(
                save_engine("layer0_out_proj")
            )
        )
        hooks.append(
            layer.post_attention_layernorm.register_forward_pre_hook(
                save_engine_input("layer0_attention_residual")
            )
        )
        hooks.append(
            layer.post_attention_layernorm.register_forward_hook(
                save_engine("layer0_post_norm")
            )
        )
        hooks.append(
            layer.mlp.gate_up_proj.register_forward_hook(
                save_engine_gate_up
            )
        )
        hooks.append(
            layer.mlp.register_forward_hook(
                save_engine("layer0_mlp_output")
            )
        )
        hooks.append(
            model.layer_stack.layers[
                1
            ].input_layernorm.register_forward_pre_hook(
                save_engine_input("layer0_output")
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
        "layer0_conv_local",
        "layer0_core_local",
        "layer0_trace_raw_query",
        "layer0_trace_raw_key",
        "layer0_trace_raw_value",
        "layer0_trace_raw_g",
        "layer0_trace_raw_beta",
        "layer0_trace_state_before",
        "layer0_trace_query_chunk",
        "layer0_trace_key_chunk",
        "layer0_trace_value_chunk",
        "layer0_trace_key_cumulative_decay",
        "layer0_trace_decay_mask",
        "layer0_trace_intra",
        "layer0_trace_previous",
        "layer0_trace_value_new",
        "layer0_trace_inter",
        "layer0_trace_intra_value",
        "layer0_trace_output_chunk",
        "layer0_trace_state_after",
        "layer0_trace_replay_output",
        "layer0_gated_local",
        "layer0_out_proj",
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
