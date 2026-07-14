"""Fit/apply a calibrated recovery bank from TinyLLM ModelRunner.kv_cache."""

from __future__ import annotations

import argparse
import atexit
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
    parser.add_argument("--calibration-prompt", default="Light Doc Cache TinyLLM calibration prompt.")
    parser.add_argument(
        "--calibration-prompt-extra",
        action="append",
        default=[],
        help="Additional calibration prompt. Can be passed multiple times.",
    )
    parser.add_argument(
        "--calibration-prompts-file",
        default="",
        help="Optional UTF-8 text file with one calibration prompt per non-empty line.",
    )
    parser.add_argument("--target-prompt", default="Light Doc Cache TinyLLM target prompt.")
    parser.add_argument("--task-id", default="smoothquant_status")
    parser.add_argument("--doc-id", default="second")
    parser.add_argument("--max-model-len", type=int, default=512)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.30)
    parser.add_argument("--max-output-len", type=int, default=1)
    parser.add_argument("--source-count", type=int, default=2)
    parser.add_argument(
        "--source-map",
        choices=["same_layer", "calibration_fit", "calibration_holdout"],
        default="same_layer",
        help="How to choose retained source heads for calibrated recovery.",
    )
    parser.add_argument("--recover-ridge", type=float, default=1e-6)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)

    from tinyvllm import LLM, SamplingParams

    llm = LLM(
        args.model,
        tensor_parallel_size=1,
        max_model_len=args.max_model_len,
        max_num_batched_tokens=args.max_model_len,
        max_num_seqs=1,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=True,
    )
    try:
        sampling_params = SamplingParams(max_tokens=args.max_output_len, ignore_eos=True)
        calibration_prompts = _collect_calibration_prompts(
            args.calibration_prompt,
            args.calibration_prompt_extra,
            args.calibration_prompts_file,
        )
        calibration_samples = [
            _run_prompt_and_copy_kv(llm, prompt, sampling_params)
            for prompt in calibration_prompts
        ]
        calibration_kv = stack_calibration_kv_samples(
            calibration_samples,
            block_size=int(llm.model_runner.kv_cache.shape[3]),
        )
        calibration_tokens = sum(int(tokens) for _, tokens in calibration_samples)
        target_kv, target_tokens = _run_prompt_and_copy_kv(llm, args.target_prompt, sampling_params)
        summary = run_calibrated_smoke(
            calibration_kv=calibration_kv,
            target_kv=target_kv,
            calibration_tokens=calibration_tokens,
            target_tokens=target_tokens,
            policy_file=args.policy_file,
            repo_root=args.repo_root,
            task_id=args.task_id,
            doc_id=args.doc_id,
            model=args.model,
            source_count=args.source_count,
            source_map=args.source_map,
            recover_ridge=args.recover_ridge,
            output_dir=Path(args.output_dir),
        )
    finally:
        try:
            atexit.unregister(llm.exit)
        except Exception:
            pass
        llm.exit()
    print(
        json.dumps(
            {
                "mse_missing_compact_tokens": summary["error_metrics"]["mse_missing_compact_tokens"],
                "output_dir": args.output_dir,
                "recovery_bank_file": summary["recovery_bank_file"],
            },
            sort_keys=True,
        )
    )
    return 0


def _collect_calibration_prompts(
    calibration_prompt: str,
    extra_prompts: list[str],
    prompts_file: str,
) -> list[str]:
    prompts = [calibration_prompt]
    prompts.extend(extra_prompts or [])
    if prompts_file:
        prompts.extend(
            line.strip()
            for line in Path(prompts_file).read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    return prompts


def stack_calibration_kv_samples(samples: list[tuple[object, int]], *, block_size: int):
    """Stack prompt KV prefixes into one calibration KV tensor.

    TinyLLM allocates a large fixed KV cache per prompt. For calibration, only
    the prompt-token prefix is meaningful, so this helper packs multiple prompt
    prefixes into a fresh `[2, layers, blocks, block_size, kv_heads, head_dim]`
    tensor and zero-fills padding tokens in the last block.
    """

    if not samples:
        raise ValueError("at least one calibration KV sample is required")
    block_size = int(block_size)
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    first_kv, _ = samples[0]
    first_shape = tuple(int(dim) for dim in first_kv.shape)
    if len(first_shape) != 6:
        raise ValueError("calibration KV must be shaped [2, layers, blocks, block_size, kv_heads, head_dim]")
    if first_shape[0] != 2:
        raise ValueError("calibration KV dim0 must be 2 for K/V")
    if first_shape[3] != block_size:
        raise ValueError("block_size does not match calibration KV shape")
    total_tokens = sum(int(tokens) for _, tokens in samples)
    if total_tokens <= 0:
        raise ValueError("calibration token total must be positive")
    output_blocks = (total_tokens + block_size - 1) // block_size
    output_shape = (first_shape[0], first_shape[1], output_blocks, block_size, first_shape[4], first_shape[5])
    stacked = _RUNTIME._full_array(  # pylint: disable=protected-access
        output_shape,
        0.0,
        getattr(first_kv, "dtype", None),
        device=getattr(first_kv, "device", None),
    )
    write_start = 0
    for kv_cache, tokens in samples:
        tokens = int(tokens)
        if tokens <= 0:
            continue
        shape = tuple(int(dim) for dim in kv_cache.shape)
        if shape[0] != first_shape[0] or shape[1] != first_shape[1] or shape[3:] != first_shape[3:]:
            raise ValueError("all calibration KV samples must share K/V, layer, block-size, head, and head-dim shape")
        if shape[2] * shape[3] < tokens:
            raise ValueError("calibration KV sample cannot hold its prompt tokens")
        flattened = _flatten_kv_tokens(kv_cache)[:, :, :tokens, :, :]
        _assign_flat_kv_tokens(stacked, flattened, write_start)
        write_start += tokens
    return stacked


def copy_kv_prompt_prefix(kv_cache, prompt_tokens: int):
    """Copy only the blocks needed for a prompt prefix and zero-fill padding."""

    prompt_tokens = int(prompt_tokens)
    if prompt_tokens <= 0:
        raise ValueError("prompt_tokens must be positive")
    shape = tuple(int(dim) for dim in kv_cache.shape)
    if len(shape) != 6:
        raise ValueError("kv_cache must be shaped [2, layers, blocks, block_size, kv_heads, head_dim]")
    if shape[2] * shape[3] < prompt_tokens:
        raise ValueError("kv_cache cannot hold prompt_tokens")
    prefix_blocks = (prompt_tokens + shape[3] - 1) // shape[3]
    prefix_shape = (shape[0], shape[1], prefix_blocks, shape[3], shape[4], shape[5])
    prefix = _RUNTIME._full_array(  # pylint: disable=protected-access
        prefix_shape,
        0.0,
        getattr(kv_cache, "dtype", None),
        device=getattr(kv_cache, "device", None),
    )
    flattened = _flatten_kv_tokens(kv_cache)[:, :, :prompt_tokens, :, :]
    _assign_flat_kv_tokens(prefix, flattened, 0)
    return prefix


def _flatten_kv_tokens(kv_cache):
    shape = tuple(int(dim) for dim in kv_cache.shape)
    return kv_cache.reshape(shape[0], shape[1], shape[2] * shape[3], shape[4], shape[5])


def _assign_flat_kv_tokens(kv_cache, flat_tokens, start_token: int) -> None:
    for token_offset in range(int(flat_tokens.shape[2])):
        token_index = int(start_token) + token_offset
        block = token_index // int(kv_cache.shape[3])
        offset = token_index % int(kv_cache.shape[3])
        kv_cache[:, :, block, offset, :, :] = flat_tokens[:, :, token_offset, :, :]


def run_calibrated_smoke(
    *,
    calibration_kv,
    target_kv,
    calibration_tokens: int,
    target_tokens: int,
    policy_file: str,
    repo_root: str | Path,
    task_id: str,
    doc_id: str | None,
    model: str,
    source_count: int,
    source_map: str = "same_layer",
    recover_ridge: float,
    output_dir: Path,
) -> dict:
    policy = _RUNTIME.load_light_doc_cache_policy(policy_file)
    config = _RUNTIME.build_config_from_policy_dirs(
        policy,
        repo_root=repo_root,
        num_layers=int(target_kv.shape[1]),
        num_kv_heads=int(target_kv.shape[4]),
        enabled=True,
    )
    calibration_plan = _RUNTIME.build_light_doc_cache_runtime_plan(
        config,
        task_id=task_id,
        doc_id=doc_id,
        seq_len=int(calibration_tokens),
    )
    target_plan = _RUNTIME.build_light_doc_cache_runtime_plan(
        config,
        task_id=task_id,
        doc_id=doc_id,
        seq_len=int(target_tokens),
    )
    bank, calibration_plan, source_heads = fit_calibration_recovery_bank(
        calibration_kv=calibration_kv,
        calibration_tokens=calibration_tokens,
        policy_file=policy_file,
        repo_root=repo_root,
        task_id=task_id,
        doc_id=doc_id,
        source_count=int(source_count),
        source_map=source_map,
        recover_ridge=recover_ridge,
    )
    storage = _RUNTIME.LightDocCacheCompressedKVStorage.from_full_kv(target_kv, target_plan, fill_value=-1.0)
    callback = _RUNTIME.make_calibrated_multi_source_recovery_callback(storage, bank)
    restored = storage.restore_to_full_shape(recover_missing_fn=callback)
    error_metrics = _RUNTIME.evaluate_restored_kv_error(target_kv, restored, target_plan)
    storage_summary = storage.summary()

    output_dir.mkdir(parents=True, exist_ok=True)
    bank_path = output_dir / "multi_source_recovery_bank.json"
    _RUNTIME.save_multi_source_recovery_bank(bank, bank_path)
    summary = {
        "calibration_tokens": int(calibration_tokens),
        "calibration_plan_tokens": int(calibration_plan.seq_len),
        "claim_boundary": "tinyllm_calibrated_kv_smoke_not_attention_hot_path",
        "effective_plan_tokens": int(target_plan.seq_len),
        "error_metrics": error_metrics,
        "kv_cache_shape": list(target_kv.shape),
        "model": model,
        "calibration_plan": calibration_plan.as_summary(),
        "plan": target_plan.as_summary(),
        "recovery_bank_file": str(bank_path),
        "source_count": int(source_count),
        "source_map": source_map,
        "source_heads": {
            f"{target[0]}:{target[1]}": [[int(layer), int(head)] for layer, head in sources]
            for target, sources in sorted(source_heads.items())
        },
        "storage": storage_summary,
        "target_tokens": int(target_tokens),
    }
    (output_dir / "tinyllm_calibrated_kv_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_report(output_dir / "tinyllm_calibrated_kv_report.md", summary)
    return summary


def fit_calibration_recovery_bank(
    *,
    calibration_kv,
    calibration_tokens: int,
    policy_file: str,
    repo_root: str | Path,
    task_id: str,
    doc_id: str | None,
    source_count: int,
    source_map: str,
    recover_ridge: float,
):
    policy = _RUNTIME.load_light_doc_cache_policy(policy_file)
    config = _RUNTIME.build_config_from_policy_dirs(
        policy,
        repo_root=repo_root,
        num_layers=int(calibration_kv.shape[1]),
        num_kv_heads=int(calibration_kv.shape[4]),
        enabled=True,
    )
    calibration_plan = _RUNTIME.build_light_doc_cache_runtime_plan(
        config,
        task_id=task_id,
        doc_id=doc_id,
        seq_len=int(calibration_tokens),
    )
    if source_map == "calibration_fit":
        source_heads = build_calibration_fit_source_heads(
            calibration_kv,
            calibration_plan,
            source_count=int(source_count),
            ridge=recover_ridge,
        )
    elif source_map == "calibration_holdout":
        source_heads = build_calibration_holdout_source_heads(
            calibration_kv,
            calibration_plan,
            source_count=int(source_count),
            ridge=recover_ridge,
        )
    elif source_map == "same_layer":
        source_heads = _build_multi_source_heads(
            calibration_plan,
            int(source_count),
        )
    else:
        raise ValueError(f"unsupported source_map: {source_map}")
    bank = _RUNTIME.fit_multi_source_recovery_bank(
        calibration_kv,
        calibration_plan,
        source_heads=source_heads,
        ridge=recover_ridge,
    )
    return bank, calibration_plan, source_heads


def _run_prompt_and_copy_kv(llm, prompt: str, sampling_params):
    llm.generate([prompt], sampling_params, use_tqdm=False)
    prompt_tokens = len(llm.tokenizer.encode(prompt))
    kv_cache = copy_kv_prompt_prefix(llm.model_runner.kv_cache.detach(), prompt_tokens)
    return kv_cache, prompt_tokens


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


def build_calibration_fit_source_heads(
    calibration_kv,
    plan,
    *,
    source_count: int,
    ridge: float,
) -> dict[tuple[int, int], list[tuple[int, int]]]:
    """Select retained source heads by calibration-prefix reconstruction fit."""

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
    for target in plan.recovered_heads:
        target = (int(target[0]), int(target[1]))
        scores = [
            (_calibration_source_fit_mse(calibration_kv, plan, target, source, float(ridge)), source)
            for source in full_heads
        ]
        scores.sort(key=lambda item: (item[0], item[1][0] != target[0], item[1]))
        source_heads[target] = [source for _, source in scores[:source_count]]
    return source_heads


def build_calibration_holdout_source_heads(
    calibration_kv,
    plan,
    *,
    source_count: int,
    ridge: float,
) -> dict[tuple[int, int], list[tuple[int, int]]]:
    """Select retained source heads by predicting calibration holdout tokens."""

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
    for target in plan.recovered_heads:
        target = (int(target[0]), int(target[1]))
        scores = [
            (_calibration_source_holdout_mse(calibration_kv, plan, target, source, float(ridge)), source)
            for source in full_heads
        ]
        scores.sort(key=lambda item: (item[0], item[1][0] != target[0], item[1]))
        source_heads[target] = [source for _, source in scores[:source_count]]
    return source_heads


def _calibration_source_fit_mse(calibration_kv, plan, target, source, ridge: float) -> float:
    selected_tokens = max(1, int(round(int(plan.seq_len) * 0.5)))
    target_tokens = _flatten_script_head_tokens(calibration_kv[:, target[0], :, :, target[1], :])[
        :, :selected_tokens, :
    ]
    source_tokens = _flatten_script_head_tokens(calibration_kv[:, source[0], :, :, source[1], :])[
        :, :selected_tokens, :
    ]
    try:
        import numpy as np

        if "numpy" in str(type(target_tokens)):
            source_float = source_tokens.astype("float64", copy=False)
            target_float = target_tokens.astype("float64", copy=False)
            source_mean = source_float.mean(axis=1, keepdims=True)
            target_mean = target_float.mean(axis=1, keepdims=True)
            centered_source = source_float - source_mean
            centered_target = target_float - target_mean
            denominator = (centered_source * centered_source).sum(axis=1, keepdims=True) + float(ridge)
            slope = (centered_source * centered_target).sum(axis=1, keepdims=True) / denominator
            bias = target_mean - slope * source_mean
            diff = (slope * source_float + bias) - target_float
            return float(np.mean(diff * diff))
    except Exception:
        pass
    try:
        source_float = source_tokens.to("cpu").to(dtype=__import__("torch").float64)
        target_float = target_tokens.to("cpu").to(dtype=__import__("torch").float64)
        source_mean = source_float.mean(dim=1, keepdim=True)
        target_mean = target_float.mean(dim=1, keepdim=True)
        centered_source = source_float - source_mean
        centered_target = target_float - target_mean
        denominator = (centered_source * centered_source).sum(dim=1, keepdim=True) + float(ridge)
        slope = (centered_source * centered_target).sum(dim=1, keepdim=True) / denominator
        bias = target_mean - slope * source_mean
        diff = (slope * source_float + bias) - target_float
        return float((diff * diff).mean().item())
    except Exception as exc:
        raise RuntimeError("calibration source fit requires numpy or torch") from exc


def _calibration_source_holdout_mse(calibration_kv, plan, target, source, ridge: float) -> float:
    selected_tokens = max(1, int(round(int(plan.seq_len) * 0.5)))
    if selected_tokens >= int(plan.seq_len):
        return _calibration_source_fit_mse(calibration_kv, plan, target, source, ridge)
    target_tokens = _flatten_script_head_tokens(calibration_kv[:, target[0], :, :, target[1], :])[
        :, : int(plan.seq_len), :
    ]
    source_tokens = _flatten_script_head_tokens(calibration_kv[:, source[0], :, :, source[1], :])[
        :, : int(plan.seq_len), :
    ]
    holdout_tokens = int(plan.seq_len) - selected_tokens
    pred = _predict_single_source_missing(
        source_tokens=source_tokens,
        target_tokens=target_tokens,
        selected_tokens=selected_tokens,
        missing_tokens=holdout_tokens,
        ridge=float(ridge),
    )
    target_holdout = target_tokens[:, selected_tokens:int(plan.seq_len), :]
    if hasattr(pred, "device") and hasattr(target_holdout, "to"):
        target_holdout = target_holdout.to(pred.device).to(dtype=pred.dtype)
    diff = pred - target_holdout
    return _mean_square_script(diff)


def _predict_single_source_missing(*, source_tokens, target_tokens, selected_tokens: int, missing_tokens: int, ridge: float):
    try:
        import numpy as np

        if "numpy" in str(type(target_tokens)):
            source_float = source_tokens.astype("float64", copy=False)
            target_float = target_tokens.astype("float64", copy=False)
            source_selected = source_float[:, :selected_tokens, :]
            target_selected = target_float[:, :selected_tokens, :]
            source_missing = source_float[:, selected_tokens:selected_tokens + missing_tokens, :]
            source_mean = source_selected.mean(axis=1, keepdims=True)
            target_mean = target_selected.mean(axis=1, keepdims=True)
            centered_source = source_selected - source_mean
            centered_target = target_selected - target_mean
            denominator = (centered_source * centered_source).sum(axis=1, keepdims=True) + float(ridge)
            slope = (centered_source * centered_target).sum(axis=1, keepdims=True) / denominator
            bias = target_mean - slope * source_mean
            return slope * source_missing + bias
    except Exception:
        pass
    try:
        import torch

        source_float = source_tokens.to("cpu").to(dtype=torch.float64)
        target_float = target_tokens.to("cpu").to(dtype=torch.float64)
        source_selected = source_float[:, :selected_tokens, :]
        target_selected = target_float[:, :selected_tokens, :]
        source_missing = source_float[:, selected_tokens:selected_tokens + missing_tokens, :]
        source_mean = source_selected.mean(dim=1, keepdim=True)
        target_mean = target_selected.mean(dim=1, keepdim=True)
        centered_source = source_selected - source_mean
        centered_target = target_selected - target_mean
        denominator = (centered_source * centered_source).sum(dim=1, keepdim=True) + float(ridge)
        slope = (centered_source * centered_target).sum(dim=1, keepdim=True) / denominator
        bias = target_mean - slope * source_mean
        return slope * source_missing + bias
    except Exception as exc:
        raise RuntimeError("calibration source holdout prediction requires numpy or torch") from exc


def _mean_square_script(value) -> float:
    try:
        import numpy as np

        if "numpy" in str(type(value)):
            return float(np.mean(value * value))
    except Exception:
        pass
    try:
        return float((value * value).mean().item())
    except Exception as exc:
        raise RuntimeError("mean square requires numpy or torch") from exc


def _flatten_script_head_tokens(head_view):
    shape = tuple(int(dim) for dim in head_view.shape)
    return head_view.reshape(shape[0], shape[1] * shape[2], shape[3])


def _write_report(path: Path, summary: dict) -> None:
    error_metrics = summary["error_metrics"]
    lines = [
        "# TinyLLM Calibrated KV Smoke",
        "",
        "Boundary: fits/applies a calibrated bank from TinyLLM ModelRunner.kv_cache; no attention hot-path or KV allocation lifetime change.",
        "",
        f"- Model: `{summary['model']}`",
        f"- Calibration tokens: `{summary['calibration_tokens']}`",
        f"- Calibration plan tokens: `{summary['calibration_plan_tokens']}`",
        f"- Target tokens: `{summary['target_tokens']}`",
        f"- Effective plan tokens: `{summary['effective_plan_tokens']}`",
        f"- KV cache shape: `{summary['kv_cache_shape']}`",
        f"- Source map: `{summary['source_map']}`",
        f"- Source count: `{summary['source_count']}`",
        f"- Recovery bank file: `{summary['recovery_bank_file']}`",
        f"- Missing-token MSE: `{error_metrics['mse_missing_compact_tokens']:.6g}`",
        f"- Missing-token max abs error: `{error_metrics['max_abs_missing_compact_tokens']:.6g}`",
        f"- Stored tensor bytes: `{summary['storage']['stored_tensor_bytes']:,}`",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
