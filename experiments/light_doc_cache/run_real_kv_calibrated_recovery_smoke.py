"""Fit and apply a calibrated recovery bank on real HF past_key_values."""

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
    parser.add_argument("--calibration-prompt", default="Light Doc Cache calibration prompt.")
    parser.add_argument("--target-prompt", default="Light Doc Cache target prompt.")
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--task-id", default="smoothquant_status")
    parser.add_argument("--doc-id", default="second")
    parser.add_argument("--source-count", type=int, default=2)
    parser.add_argument("--recover-ridge", type=float, default=1e-6)
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

    calibration_kv, calibration_tokens = _prompt_to_runtime_kv(
        model,
        tokenizer,
        args.calibration_prompt,
        max_tokens=args.max_tokens,
        block_size=args.block_size,
    )
    target_kv, target_tokens = _prompt_to_runtime_kv(
        model,
        tokenizer,
        args.target_prompt,
        max_tokens=args.max_tokens,
        block_size=args.block_size,
    )

    policy = _RUNTIME.load_light_doc_cache_policy(args.policy_file)
    plan = _RUNTIME.build_light_doc_cache_runtime_plan(
        _RUNTIME.build_config_from_policy_dirs(
            policy,
            repo_root=args.repo_root,
            num_layers=int(target_kv.shape[1]),
            num_kv_heads=int(target_kv.shape[4]),
            enabled=True,
        ),
        task_id=args.task_id,
        doc_id=args.doc_id,
        seq_len=min(int(calibration_tokens), int(target_tokens)),
    )
    calibration_kv = calibration_kv[:, :, :, :, :, :]
    target_kv = target_kv[:, :, :, :, :, :]
    source_heads = _build_multi_source_heads(plan, int(args.source_count))
    bank = _RUNTIME.fit_multi_source_recovery_bank(
        calibration_kv,
        plan,
        source_heads=source_heads,
        ridge=args.recover_ridge,
    )
    storage = _RUNTIME.LightDocCacheCompressedKVStorage.from_full_kv(target_kv, plan, fill_value=-1.0)
    callback = _RUNTIME.make_calibrated_multi_source_recovery_callback(storage, bank)
    restored = storage.restore_to_full_shape(recover_missing_fn=callback)
    error_metrics = _RUNTIME.evaluate_restored_kv_error(target_kv, restored, plan)
    storage_summary = storage.summary()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    bank_path = output_dir / "multi_source_recovery_bank.json"
    _RUNTIME.save_multi_source_recovery_bank(bank, bank_path)
    summary = {
        "claim_boundary": "real_hf_kv_calibrated_recovery_not_runtime_hot_path",
        "calibration_tokens": int(calibration_tokens),
        "target_tokens": int(target_tokens),
        "effective_plan_tokens": int(plan.seq_len),
        "error_metrics": error_metrics,
        "kv_cache_shape": list(target_kv.shape),
        "model": args.model,
        "plan": plan.as_summary(),
        "recovery_bank_file": str(bank_path),
        "source_count": int(args.source_count),
        "storage": storage_summary,
    }
    (output_dir / "real_kv_calibrated_recovery_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_report(output_dir / "real_kv_calibrated_recovery_report.md", summary)
    print(
        json.dumps(
            {
                "mse_missing_compact_tokens": error_metrics["mse_missing_compact_tokens"],
                "output_dir": str(output_dir),
                "recovery_bank_file": str(bank_path),
            },
            sort_keys=True,
        )
    )
    return 0


def _prompt_to_runtime_kv(model, tokenizer, prompt: str, *, max_tokens: int, block_size: int):
    import torch

    encoded = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=max_tokens,
        add_special_tokens=True,
    )
    encoded = {key: value.to(model.device) for key, value in encoded.items()}
    with torch.inference_mode():
        output = model(**encoded, use_cache=True, return_dict=True)
    kv_cache = _hf_past_to_runtime_kv(
        output.past_key_values,
        expected_heads=_expected_kv_heads(model),
        block_size=block_size,
    )
    return kv_cache, int(encoded["input_ids"].shape[1])


def _build_multi_source_heads(plan, source_count: int) -> dict[tuple[int, int], list[tuple[int, int]]]:
    source_count = max(1, int(source_count))
    compact_heads = set(plan.recovered_heads)
    full_heads = [
        (layer, head)
        for layer in range(int(plan.num_layers))
        for head in range(int(plan.num_kv_heads))
        if (layer, head) not in compact_heads
    ]
    if not full_heads:
        raise ValueError("calibrated recovery requires at least one retained full source head")
    source_heads = {}
    for target_layer, target_head in plan.recovered_heads:
        same_layer = [
            (layer, head)
            for layer, head in full_heads
            if layer == int(target_layer) and head != int(target_head)
        ]
        candidates = [*same_layer, *[head for head in full_heads if head not in set(same_layer)]]
        source_heads[(int(target_layer), int(target_head))] = candidates[:source_count]
    return source_heads


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


def _write_report(path: Path, summary: dict) -> None:
    error_metrics = summary["error_metrics"]
    lines = [
        "# Real KV Calibrated Recovery Smoke",
        "",
        "Boundary: real HF past_key_values calibrated recovery smoke; not wired into TinyLLM runtime hot path.",
        "",
        f"- Model: `{summary['model']}`",
        f"- Calibration tokens: `{summary['calibration_tokens']}`",
        f"- Target tokens: `{summary['target_tokens']}`",
        f"- Effective plan tokens: `{summary['effective_plan_tokens']}`",
        f"- KV cache shape: `{summary['kv_cache_shape']}`",
        f"- Source count: `{summary['source_count']}`",
        f"- Recovery bank file: `{summary['recovery_bank_file']}`",
        f"- Missing-token MSE: `{error_metrics['mse_missing_compact_tokens']:.6g}`",
        f"- Missing-token max abs error: `{error_metrics['max_abs_missing_compact_tokens']:.6g}`",
        f"- Stored tensor bytes: `{summary['storage']['stored_tensor_bytes']:,}`",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
