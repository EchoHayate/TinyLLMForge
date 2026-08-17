"""Summarize Light Doc Cache accounting from TinyLLM ModelRunner.kv_cache."""

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
    parser.add_argument("--prompt", default="Light Doc Cache TinyLLM KV summary smoke.")
    parser.add_argument("--task-id", default="smoothquant_status")
    parser.add_argument("--doc-id", default="second")
    parser.add_argument("--max-model-len", type=int, default=512)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.30)
    parser.add_argument("--max-output-len", type=int, default=1)
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--write-sidecar-storage", action="store_true")
    parser.add_argument("--recover-mode", choices=("none", "fill", "repeat_last", "linear_tail", "oracle"), default="linear_tail")
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
        enforce_eager=args.enforce_eager,
    )
    prompt_tokens = len(llm.tokenizer.encode(args.prompt))
    try:
        llm.generate([args.prompt], SamplingParams(max_tokens=args.max_output_len, ignore_eos=True), use_tqdm=False)
        summary = write_tinyllm_kv_summary(
            model_runner=llm.model_runner,
            policy_file=args.policy_file,
            repo_root=args.repo_root,
            task_id=args.task_id,
            doc_id=args.doc_id,
            seq_len=prompt_tokens,
            output_dir=Path(args.output_dir),
            model=args.model,
            prompt_tokens=prompt_tokens,
        )
        sidecar_summary = None
        if args.write_sidecar_storage:
            sidecar_summary = write_tinyllm_sidecar_storage_summary(
                model_runner=llm.model_runner,
                policy_file=args.policy_file,
                repo_root=args.repo_root,
                task_id=args.task_id,
                doc_id=args.doc_id,
                seq_len=prompt_tokens,
                output_dir=Path(args.output_dir),
                model=args.model,
                prompt_tokens=prompt_tokens,
                recover_mode=args.recover_mode,
                recover_ridge=args.recover_ridge,
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
                "output_dir": args.output_dir,
                "full_kv_bytes": summary["storage"]["full_kv_bytes"],
                "planned_byte_saving_fraction": summary["storage"]["planned_byte_saving_fraction"],
                "sidecar_byte_saving_fraction": (
                    None if sidecar_summary is None else sidecar_summary["sidecar_storage"]["byte_saving_fraction"]
                ),
            },
            sort_keys=True,
        )
    )
    return 0


def write_tinyllm_kv_summary(
    *,
    model_runner,
    policy_file: str,
    repo_root: str | Path,
    task_id: str,
    doc_id: str | None,
    seq_len: int,
    output_dir: str | Path,
    model: str,
    prompt_tokens: int,
) -> dict:
    policy = _RUNTIME.load_light_doc_cache_policy(policy_file)
    kv_cache = getattr(model_runner, "kv_cache", None)
    if kv_cache is None:
        raise RuntimeError("model_runner does not expose kv_cache")
    shape = tuple(int(dim) for dim in kv_cache.shape)
    if len(shape) != 6:
        raise ValueError("model_runner.kv_cache must be shaped [2, layers, blocks, block_size, kv_heads, head_dim]")
    config = _RUNTIME.build_config_from_policy_dirs(
        policy,
        repo_root=repo_root,
        num_layers=shape[1],
        num_kv_heads=shape[4],
        enabled=True,
    )
    plan = _RUNTIME.build_light_doc_cache_runtime_plan(
        config,
        task_id=task_id,
        doc_id=doc_id,
        seq_len=seq_len,
    )
    runner_summary = _RUNTIME.build_model_runner_light_doc_cache_summary(model_runner, plan)
    if runner_summary is None:
        raise RuntimeError("model_runner light-doc-cache summary returned None")
    summary = {
        "claim_boundary": "tinyllm_allocated_kv_summary_only",
        "allocated_kv_cache_bytes": _allocated_kv_cache_bytes(kv_cache),
        "model": model,
        "prompt_tokens": int(prompt_tokens),
        "kv_cache_shape": list(shape),
        "runner_summary": runner_summary,
        "plan": runner_summary["plan"],
        "storage": runner_summary["storage"],
    }
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "tinyllm_kv_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    lines = [
        "# TinyLLM KV Summary Smoke",
        "",
        "Boundary: reads TinyLLM ModelRunner.kv_cache allocation and planned Light Doc Cache accounting only; no runtime compression is applied.",
        "",
        f"- Model: `{model}`",
        f"- Prompt tokens: `{prompt_tokens}`",
        f"- KV cache shape: `{list(shape)}`",
        f"- Allocated KV cache bytes: `{summary['allocated_kv_cache_bytes']:,}`",
        f"- Logical full KV bytes for plan seq_len: `{summary['storage']['full_kv_bytes']:,}`",
        f"- Planned recovered KV bytes: `{summary['storage']['planned_recovered_kv_bytes']:,}`",
        f"- Planned stored KV bytes: `{summary['storage']['planned_stored_kv_bytes']:,}`",
        f"- Planned byte saving fraction: `{summary['storage']['planned_byte_saving_fraction']:.2%}`",
        f"- Planned compression ratio: `{summary['plan']['compression_ratio']:.4f}x`",
    ]
    (output_dir / "tinyllm_kv_summary_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


def _allocated_kv_cache_bytes(kv_cache) -> int:
    numel = 1
    for dim in kv_cache.shape:
        numel *= int(dim)
    return int(numel * kv_cache.element_size())


def write_tinyllm_sidecar_storage_summary(
    *,
    model_runner,
    policy_file: str,
    repo_root: str | Path,
    task_id: str,
    doc_id: str | None,
    seq_len: int,
    output_dir: str | Path,
    model: str,
    prompt_tokens: int,
    recover_mode: str,
    recover_ridge: float,
) -> dict:
    policy = _RUNTIME.load_light_doc_cache_policy(policy_file)
    kv_cache = getattr(model_runner, "kv_cache", None)
    if kv_cache is None:
        raise RuntimeError("model_runner does not expose kv_cache")
    shape = tuple(int(dim) for dim in kv_cache.shape)
    if len(shape) != 6:
        raise ValueError("model_runner.kv_cache must be shaped [2, layers, blocks, block_size, kv_heads, head_dim]")
    config = _RUNTIME.build_config_from_policy_dirs(
        policy,
        repo_root=repo_root,
        num_layers=shape[1],
        num_kv_heads=shape[4],
        enabled=True,
    )
    plan = _RUNTIME.build_light_doc_cache_runtime_plan(
        config,
        task_id=task_id,
        doc_id=doc_id,
        seq_len=seq_len,
    )
    recover_missing_fn = _make_recovery_callback(recover_mode, kv_cache, plan, recover_ridge)
    if hasattr(model_runner, "light_doc_cache_materialize_sidecar"):
        sidecar_summary = model_runner.light_doc_cache_materialize_sidecar(
            plan,
            fill_value=-1.0,
            recover_missing_fn=recover_missing_fn,
            evaluate_readback=True,
        )
    else:
        _, sidecar_summary = _RUNTIME.materialize_light_doc_cache_sidecar(
            kv_cache,
            plan,
            fill_value=-1.0,
            recover_missing_fn=recover_missing_fn,
            evaluate_readback=True,
        )
    if sidecar_summary is None:
        raise RuntimeError("model_runner light-doc-cache sidecar materialization returned None")
    error_metrics = sidecar_summary["error_metrics"]
    sidecar_storage = sidecar_summary["sidecar_storage"]
    logical_full_bytes = sidecar_summary["logical_full_kv_bytes"]
    logical_stored_bytes = sidecar_storage["stored_tensor_bytes"]
    summary = {
        "claim_boundary": "tinyllm_sidecar_storage_readback_not_hot_path",
        "error_metrics": error_metrics,
        "kv_cache_shape": list(shape),
        "logical_full_kv_bytes": logical_full_bytes,
        "logical_saved_kv_bytes": sidecar_summary["logical_saved_kv_bytes"],
        "logical_stored_kv_bytes": logical_stored_bytes,
        "logical_byte_saving_fraction": sidecar_summary["logical_byte_saving_fraction"],
        "model": model,
        "plan": plan.as_summary(),
        "prompt_tokens": int(prompt_tokens),
        "recovery_mode": recover_mode,
        "sidecar_storage": sidecar_storage,
    }
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "tinyllm_sidecar_storage_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    lines = [
        "# TinyLLM Sidecar Storage Smoke",
        "",
        "Boundary: materializes a compressed sidecar from ModelRunner.kv_cache and restores into a temporary tensor; no attention hot-path read is changed.",
        "",
        f"- Model: `{model}`",
        f"- Prompt tokens: `{prompt_tokens}`",
        f"- KV cache shape: `{list(shape)}`",
        f"- Recovery mode: `{recover_mode}`",
        f"- Sidecar full tensor bytes: `{sidecar_storage['full_tensor_bytes']:,}`",
        f"- Sidecar stored tensor bytes: `{sidecar_storage['stored_tensor_bytes']:,}`",
        f"- Sidecar saved tensor bytes: `{sidecar_storage['saved_tensor_bytes']:,}`",
        f"- Sidecar allocated-capacity byte saving fraction: `{sidecar_storage['byte_saving_fraction']:.2%}`",
        f"- Logical full KV bytes for plan seq_len: `{logical_full_bytes:,}`",
        f"- Logical stored KV bytes: `{logical_stored_bytes:,}`",
        f"- Logical byte saving fraction: `{summary['logical_byte_saving_fraction']:.2%}`",
        f"- Missing-token MSE: `{error_metrics['mse_missing_compact_tokens']:.6g}`",
        f"- Missing-token max abs error: `{error_metrics['max_abs_missing_compact_tokens']:.6g}`",
    ]
    (output_dir / "tinyllm_sidecar_storage_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


def _make_recovery_callback(mode: str, kv_cache, plan, recover_ridge: float):
    if mode == "oracle":
        return _RUNTIME.make_oracle_recovery_callback(kv_cache, plan)
    if mode == "linear_tail":
        return _RUNTIME.make_linear_tail_recovery_callback(ridge=recover_ridge)
    if mode == "repeat_last":
        return _RUNTIME.make_repeat_last_recovery_callback()
    if mode == "fill":
        def recover_missing(**kwargs):
            stored_tokens = kwargs["stored_tokens"]
            if hasattr(stored_tokens, "new_full"):
                return stored_tokens.new_full((2, kwargs["missing_tokens"], kwargs["head_dim"]), 0.0)
            import numpy as np

            return np.zeros((2, kwargs["missing_tokens"], kwargs["head_dim"]), dtype=getattr(stored_tokens, "dtype", None))

        return recover_missing
    return None


if __name__ == "__main__":
    raise SystemExit(main())
