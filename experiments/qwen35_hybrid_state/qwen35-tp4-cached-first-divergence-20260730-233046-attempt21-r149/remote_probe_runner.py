from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import torch


DIAGNOSTIC_ENV = "TINYVLLM_QWEN35_RANK_RECEIPT_DIR"
_PATCH_STATE = {}


def _atomic_torch_save(path, payload):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _install_model_capture(model, rank, receipt_dir):
    if getattr(model, "_r149_capture_installed", False):
        return
    import tinyvllm.layers.qwen35_linear_attention as linear_module

    layer = model.layer_stack.layers[0]
    linear = layer.linear_attention
    original_chunk = linear_module.qwen35_gated_delta_chunk
    original_recurrent = linear_module.qwen35_gated_delta_recurrent
    original_gated = linear_module.qwen35_gated_rmsnorm
    counters = {"core": 0, "gate": 0, "weight": 0, "gated": 0}
    pending = []

    def save(kind, tensor, *, token_count, mode):
        index = counters[kind]
        counters[kind] += 1
        path = (
            Path(receipt_dir)
            / (
                f"rank-{rank:02d}-{kind}-{index:04d}-"
                f"{mode}-tokens-{token_count}.pt"
            )
        )
        _atomic_torch_save(path, {
            "rank": int(rank),
            "kind": kind,
            "index": int(index),
            "mode": mode,
            "token_count": int(token_count),
            "tensor": tensor.detach().cpu().clone(),
        })

    def wrap_core(function, mode):
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
            if A_log.data_ptr() == linear.A_log.data_ptr():
                token_count = int(query.shape[0])
                core = result[0][-1:].reshape(1, -1)
                save(
                    "core",
                    core,
                    token_count=token_count,
                    mode=mode,
                )
                pending.append((token_count, mode))
            return result
        return wrapped

    def wrapped_gated(core, gate, weight, **kwargs):
        value = original_gated(core, gate, weight, **kwargs)
        if weight.data_ptr() == linear.norm_weight.data_ptr():
            if not pending:
                raise RuntimeError("r149 gated capture lacks matching core")
            token_count, mode = pending.pop(0)
            save(
                "gate",
                gate[-linear.local_value_heads:].reshape(1, -1),
                token_count=token_count,
                mode=mode,
            )
            save(
                "weight",
                weight.reshape(1, -1),
                token_count=token_count,
                mode=mode,
            )
            save(
                "gated",
                value[-linear.local_value_heads:].reshape(1, -1),
                token_count=token_count,
                mode=mode,
            )
        return value

    linear_module.qwen35_gated_delta_chunk = wrap_core(
        original_chunk,
        "chunk",
    )
    linear_module.qwen35_gated_delta_recurrent = wrap_core(
        original_recurrent,
        "recurrent",
    )
    linear_module.qwen35_gated_rmsnorm = wrapped_gated
    model._r149_capture_installed = True


def _install_rank_patch():
    receipt_dir = os.environ.get(DIAGNOSTIC_ENV)
    if not receipt_dir or _PATCH_STATE:
        return
    import tinyvllm.engine.model_runner as model_runner

    original = model_runner._initialize_model_runner_model

    def wrapped_initialize(
        config,
        *,
        rank,
        load_legacy_model,
        load_qwen35_model,
    ):
        model, owner = original(
            config,
            rank=rank,
            load_legacy_model=load_legacy_model,
            load_qwen35_model=load_qwen35_model,
        )
        if getattr(config.hf_config, "model_type", None) == "qwen3_5":
            _install_model_capture(model, rank, receipt_dir)
        return model, owner

    model_runner._initialize_model_runner_model = wrapped_initialize
    _PATCH_STATE["module"] = model_runner
    _PATCH_STATE["original"] = original


_install_rank_patch()


def _last_row(value):
    if isinstance(value, tuple):
        value = value[0]
    if value.ndim == 3:
        value = value[:, -1, :]
    elif value.ndim == 2:
        value = value[-1:, :]
    else:
        raise RuntimeError(f"unexpected tensor rank: {value.ndim}")
    return value.detach().cpu().clone()


def _compare(left, right, *, atol):
    left = left.float()
    right = right.float()
    difference = (left - right).abs()
    nonzero = torch.nonzero(difference.reshape(-1), as_tuple=False)
    first = None if nonzero.numel() == 0 else int(nonzero[0].item())
    row = {
        "shape": list(left.shape),
        "max_abs_diff": float(difference.max().item()),
        "mean_abs_diff": float(difference.mean().item()),
        "nonzero_count": int(torch.count_nonzero(difference).item()),
        "allclose": bool(torch.allclose(
            left,
            right,
            atol=atol,
            rtol=0.0,
        )),
    }
    if first is not None:
        row.update({
            "first_nonzero_flat_index": first,
            "first_nonzero_left": float(left.reshape(-1)[first].item()),
            "first_nonzero_right": float(right.reshape(-1)[first].item()),
        })
    return row


def main():
    remote = Path(sys.argv[1])
    source = Path(sys.argv[2])
    inputs = remote / "inputs"
    output = remote / "output"
    receipts = output / "rank_receipts"
    receipts.mkdir(parents=True, exist_ok=False)
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

    official_rows = {}
    reference = official.TransformersGreedyReferenceBackend(
        configuration,
        gpu_index=configuration.gpu_indices[0],
    )
    hooks = []
    try:
        model = reference._model()
        linear = model.model.layers[0].linear_attn
        original_chunk = linear.chunk_gated_delta_rule

        def wrapped_chunk(*args, **kwargs):
            value = original_chunk(*args, **kwargs)
            official_rows["core"] = (
                value[0][:, -1, :]
                .reshape(1, -1)
                .detach()
                .cpu()
                .clone()
            )
            return value

        linear.chunk_gated_delta_rule = wrapped_chunk
        def official_norm_hook(module, args, value):
            official_rows["gate"] = (
                args[1].reshape(
                    len(prompt),
                    linear.num_v_heads,
                    linear.head_v_dim,
                )[-1:].reshape(1, -1).detach().cpu().clone()
            )
            official_rows["weight"] = (
                module.weight.reshape(1, -1).detach().cpu().clone()
            )
            official_rows["gated"] = (
                value.reshape(
                    len(prompt),
                    linear.num_v_heads,
                    linear.head_v_dim,
                )[-1:].reshape(1, -1).detach().cpu().clone()
            )
            return value

        hooks.append(linear.norm.register_forward_hook(official_norm_hook))
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
            linear.chunk_gated_delta_rule = original_chunk
        reference.close()

    previous_pythonpath = os.environ.get("PYTHONPATH")
    previous_runner = os.environ.get(
        "TINYVLLM_QWEN35_RANK_RECEIPT_RUNNER"
    )
    sitecustomize = inputs / "sitecustomize.py"
    sitecustomize.write_text(
        "\n".join([
            "import importlib.util",
            "import os",
            "import sys",
            "path = os.environ.get(",
            "    'TINYVLLM_QWEN35_RANK_RECEIPT_RUNNER'",
            ")",
            "if path:",
            "    name = '_tinyvllm_r149_rank_receipt_runner'",
            "    module = sys.modules.get(name)",
            "    if module is None:",
            "        spec = importlib.util.spec_from_file_location(",
            "            name, path",
            "        )",
            "        module = importlib.util.module_from_spec(spec)",
            "        sys.modules[name] = module",
            "        spec.loader.exec_module(module)",
            "    module._install_rank_patch()",
            "",
        ]),
        encoding="utf-8",
    )
    os.environ[DIAGNOSTIC_ENV] = str(receipts)
    os.environ["TINYVLLM_QWEN35_RANK_RECEIPT_RUNNER"] = (
        str(inputs / "remote_probe_runner.py")
    )
    os.environ["PYTHONPATH"] = os.pathsep.join(
        value
        for value in (
            str(inputs),
            previous_pythonpath,
        )
        if value
    )
    _install_rank_patch()
    engine = probe.backend._default_engine_factory(configuration)
    cleanup = None
    try:
        engine.configure_qwen35_hybrid_prefix_publication_runtime(
            model_fingerprint=configuration.model_fingerprint,
            max_entries=configuration.max_cache_entries,
            max_bytes=configuration.max_cache_bytes,
            timeout_s=configuration.timeout_s,
        )
        engine.clear_qwen35_hybrid_prefix_caches(
            timeout_s=configuration.timeout_s,
        )
        probe._run_request(
            engine,
            prompt,
            1,
            timeout_s=configuration.timeout_s,
        )
        probe._run_request(
            engine,
            source_prompt,
            1,
            timeout_s=configuration.timeout_s,
            record_logits=False,
        )
        probe._run_request(
            engine,
            prompt,
            1,
            timeout_s=configuration.timeout_s,
        )
    finally:
        cleanup = engine.exit()
        os.environ.pop(DIAGNOSTIC_ENV, None)
        if previous_runner is None:
            os.environ.pop(
                "TINYVLLM_QWEN35_RANK_RECEIPT_RUNNER",
                None,
            )
        else:
            os.environ[
                "TINYVLLM_QWEN35_RANK_RECEIPT_RUNNER"
            ] = previous_runner
        if previous_pythonpath is None:
            os.environ.pop("PYTHONPATH", None)
        else:
            os.environ["PYTHONPATH"] = previous_pythonpath

    rows = []
    for path in sorted(receipts.glob("rank-*.pt")):
        payload = torch.load(path, map_location="cpu", weights_only=False)
        payload["path"] = path.name
        rows.append(payload)
    inventory = {}
    for rank in range(4):
        rank_rows = [row for row in rows if row["rank"] == rank]
        inventory[str(rank)] = [
            {
                key: value
                for key, value in row.items()
                if key != "tensor"
            }
            for row in rank_rows
        ]
        if not rank_rows:
            raise RuntimeError(f"missing rank receipt: {rank}")

    recompute = []
    restore = []
    for kind in ("core", "gate", "gated"):
        recompute_shards = []
        restore_shards = []
        for rank in range(4):
            candidates = [
                row for row in rows
                if row["rank"] == rank
                and row["kind"] == kind
                and row["mode"] == "chunk"
            ]
            if len(candidates) < 3:
                raise RuntimeError(
                    f"insufficient rank {rank} {kind} chunk receipts"
                )
            recompute_shards.append(candidates[1]["tensor"])
            restore_shards.append(candidates[-1]["tensor"])
        recompute_full = torch.cat(recompute_shards, dim=-1)
        restore_full = torch.cat(restore_shards, dim=-1)
        recompute.append({
            "name": f"layer0_{kind}_full",
            **_compare(
                official_rows[kind],
                recompute_full,
                atol=contract.REGISTERED_LOGITS_ATOL,
            ),
        })
        restore.append({
            "name": f"layer0_{kind}_full",
            **_compare(
                recompute_full,
                restore_full,
                atol=contract.REGISTERED_LOGITS_ATOL,
            ),
        })

    weight_comparisons = []
    for rank in range(4):
        candidates = [
            row for row in rows
            if row["rank"] == rank
            and row["kind"] == "weight"
            and row["mode"] == "chunk"
        ]
        if len(candidates) < 3:
            raise RuntimeError(
                f"insufficient rank {rank} weight chunk receipts"
            )
        weight_comparisons.append({
            "name": f"layer0_norm_weight_rank{rank}",
            **_compare(
                official_rows["weight"],
                candidates[1]["tensor"],
                atol=0.0,
            ),
        })

    def first_mismatch(rows):
        for row in rows:
            if row["allclose"] is not True:
                return row["name"]
        return None

    result = {
        "schema_version": "qwen35.tp4-cached-rank-input-production-fix.v1",
        "official_vs_recompute": recompute,
        "recompute_vs_restore": restore,
        "official_vs_recompute_first_mismatch": first_mismatch(
            recompute
        ),
        "recompute_vs_restore_first_mismatch": first_mismatch(
            restore
        ),
        "receipt_inventory": inventory,
        "weight_comparisons": weight_comparisons,
        "cleanup": cleanup,
        "claim_boundary": (
            "diagnostic rank receipts only; no cached-continuation "
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
        "result_path": str(final),
        "official_vs_recompute_first_mismatch": (
            result["official_vs_recompute_first_mismatch"]
        ),
        "recompute_vs_restore_first_mismatch": (
            result["recompute_vs_restore_first_mismatch"]
        ),
        "official_vs_recompute": recompute,
        "recompute_vs_restore": restore,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
