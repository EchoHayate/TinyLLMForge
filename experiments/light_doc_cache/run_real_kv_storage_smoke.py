"""Run Light Doc Cache storage/recovery smoke on real HF past_key_values."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys

_REPO_ROOT = Path(__file__).resolve().parents[2]
_RUNTIME_PATH = _REPO_ROOT / "tinyvllm" / "engine" / "light_doc_cache_runtime.py"
_SPEC = importlib.util.spec_from_file_location("light_doc_cache_runtime", _RUNTIME_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_RUNTIME = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _RUNTIME
_SPEC.loader.exec_module(_RUNTIME)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--policy-file", required=True)
    parser.add_argument("--repo-root", default=str(_REPO_ROOT))
    parser.add_argument("--text-file", default=None)
    parser.add_argument("--prompt", default="Light Doc Cache storage smoke.")
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--task-id", default="smoothquant_status")
    parser.add_argument("--doc-id", default="second")
    parser.add_argument("--recover-mode", choices=("none", "fill", "repeat_last", "linear_tail", "oracle"), default="linear_tail")
    parser.add_argument("--recover-ridge", type=float, default=1e-6)
    parser.add_argument("--recover-fill-value", type=float, default=0.0)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--dtype", default="auto", choices=("auto", "float32", "float16", "bfloat16"))
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = _pick_device(args.device, torch)
    dtype = _pick_dtype(args.dtype, device, torch)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True, local_files_only=True)
    model_kwargs = dict(trust_remote_code=True, local_files_only=True)
    try:
        model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype, **model_kwargs)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype, **model_kwargs)
    model.eval().to(device)

    if args.text_file:
        text = Path(args.text_file).read_text(encoding="utf-8")
    else:
        text = args.prompt
    encoded = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=args.max_tokens,
        add_special_tokens=True,
    )
    encoded = {key: value.to(device) for key, value in encoded.items()}
    with torch.inference_mode():
        output = model(**encoded, use_cache=True, return_dict=True)

    kv_cache = _hf_past_to_runtime_kv(output.past_key_values, expected_heads=_expected_kv_heads(model), block_size=args.block_size)
    seq_len = int(encoded["input_ids"].shape[1])
    policy = _RUNTIME.load_light_doc_cache_policy(args.policy_file)
    config = _RUNTIME.build_config_from_policy_dirs(
        policy,
        repo_root=args.repo_root,
        num_layers=int(kv_cache.shape[1]),
        num_kv_heads=int(kv_cache.shape[4]),
        enabled=True,
    )
    plan = _RUNTIME.build_light_doc_cache_runtime_plan(
        config,
        task_id=args.task_id,
        doc_id=args.doc_id,
        seq_len=seq_len,
    )

    storage = _RUNTIME.LightDocCacheCompressedKVStorage.from_full_kv(kv_cache, plan, fill_value=-1.0)
    recover_missing_fn = _make_recovery_callback(args.recover_mode, kv_cache, plan, args)
    restored = storage.restore_to_full_shape(recover_missing_fn=recover_missing_fn)
    storage_summary = storage.summary()
    error_metrics = _RUNTIME.evaluate_restored_kv_error(kv_cache, restored, plan)
    result = {
        "claim_boundary": "real_hf_kv_storage_smoke_not_runtime_hot_path",
        "error_metrics": error_metrics,
        "input_tokens": seq_len,
        "kv_cache_shape": list(kv_cache.shape),
        "model": args.model,
        "plan": plan.as_summary(),
        "recovery_mode": args.recover_mode,
        "storage": storage_summary,
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "real_kv_storage_summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    lines = [
        "# Real KV Storage Smoke",
        "",
        "Boundary: real HF past_key_values storage/recovery smoke; not wired into TinyLLM runtime hot path.",
        "",
        f"- Model: `{args.model}`",
        f"- Input tokens: `{seq_len}`",
        f"- KV cache shape: `{list(kv_cache.shape)}`",
        f"- Recovery mode: `{args.recover_mode}`",
        f"- Missing-token MSE: `{error_metrics['mse_missing_compact_tokens']:.6g}`",
        f"- Missing-token max abs error: `{error_metrics['max_abs_missing_compact_tokens']:.6g}`",
        f"- Full tensor bytes: `{storage_summary['full_tensor_bytes']:,}`",
        f"- Stored tensor bytes: `{storage_summary['stored_tensor_bytes']:,}`",
        f"- Saved tensor bytes: `{storage_summary['saved_tensor_bytes']:,}`",
        f"- Byte saving fraction: `{storage_summary['byte_saving_fraction']:.2%}`",
    ]
    (output_dir / "real_kv_storage_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "byte_saving_fraction": storage_summary["byte_saving_fraction"],
                "mse_missing_compact_tokens": error_metrics["mse_missing_compact_tokens"],
            },
            sort_keys=True,
        )
    )
    return 0


def _pick_device(name, torch):
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def _pick_dtype(name, device, torch):
    if name == "float32":
        return torch.float32
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    return torch.bfloat16 if device.type == "cuda" and torch.cuda.is_bf16_supported() else torch.float32


def _expected_kv_heads(model) -> int | None:
    return getattr(model.config, "num_key_value_heads", None) or getattr(model.config, "num_attention_heads", None)


def _legacy_past_key_values(past_key_values):
    if hasattr(past_key_values, "to_legacy_cache"):
        past_key_values = past_key_values.to_legacy_cache()
    if hasattr(past_key_values, "key_cache") and hasattr(past_key_values, "value_cache"):
        return list(zip(past_key_values.key_cache, past_key_values.value_cache))
    return [(entry[0], entry[1]) for entry in past_key_values]


def _normalize_layer_cache(tensor, expected_heads: int | None):
    if expected_heads is not None:
        if int(tensor.shape[1]) == int(expected_heads):
            return tensor
        if int(tensor.shape[2]) == int(expected_heads):
            return tensor.permute(0, 2, 1, 3).contiguous()
    if int(tensor.shape[1]) <= 128 and int(tensor.shape[2]) >= int(tensor.shape[1]):
        return tensor
    return tensor.permute(0, 2, 1, 3).contiguous()


def _hf_past_to_runtime_kv(past_key_values, *, expected_heads: int | None, block_size: int):
    import torch

    layer_pairs = _legacy_past_key_values(past_key_values)
    keys = []
    values = []
    for key, value in layer_pairs:
        keys.append(_normalize_layer_cache(key.detach(), expected_heads)[0])
        values.append(_normalize_layer_cache(value.detach(), expected_heads)[0])
    key_tensor = torch.stack(keys, dim=0)
    value_tensor = torch.stack(values, dim=0)
    layers, kv_heads, seq_len, head_dim = key_tensor.shape
    blocks = (seq_len + int(block_size) - 1) // int(block_size)
    padded_tokens = blocks * int(block_size)
    dtype = key_tensor.dtype
    device = key_tensor.device
    kv_cache = torch.zeros((2, layers, blocks, int(block_size), kv_heads, head_dim), dtype=dtype, device=device)
    kv_cache[0].reshape(layers, padded_tokens, kv_heads, head_dim)[:, :seq_len, :, :] = key_tensor.permute(0, 2, 1, 3)
    kv_cache[1].reshape(layers, padded_tokens, kv_heads, head_dim)[:, :seq_len, :, :] = value_tensor.permute(0, 2, 1, 3)
    return kv_cache


def _make_recovery_callback(mode: str, kv_cache, plan, args):
    if mode == "oracle":
        return _RUNTIME.make_oracle_recovery_callback(kv_cache, plan)
    if mode == "linear_tail":
        return _RUNTIME.make_linear_tail_recovery_callback(ridge=args.recover_ridge)
    if mode == "repeat_last":
        return _RUNTIME.make_repeat_last_recovery_callback()
    if mode == "fill":
        def recover_missing(**kwargs):
            import torch

            return torch.full(
                (2, kwargs["missing_tokens"], kwargs["head_dim"]),
                float(args.recover_fill_value),
                dtype=kwargs["dtype"],
                device=kwargs["stored_tokens"].device,
            )

        return recover_missing
    return None


if __name__ == "__main__":
    raise SystemExit(main())
