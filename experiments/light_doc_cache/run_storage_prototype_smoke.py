"""Run a small CPU Light Doc Cache storage prototype smoke."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
_RUNTIME_PATH = _REPO_ROOT / "tinyvllm" / "engine" / "light_doc_cache_runtime.py"
_SPEC = importlib.util.spec_from_file_location("light_doc_cache_runtime", _RUNTIME_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_RUNTIME = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _RUNTIME
_SPEC.loader.exec_module(_RUNTIME)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy-file", required=True)
    parser.add_argument("--repo-root", default=str(_REPO_ROOT))
    parser.add_argument("--task-id", default="smoothquant_status")
    parser.add_argument("--doc-id", default="second")
    parser.add_argument("--seq-len", type=int, default=12)
    parser.add_argument("--num-layers", type=int, default=28)
    parser.add_argument("--num-kv-heads", type=int, default=8)
    parser.add_argument("--num-blocks", type=int, default=2)
    parser.add_argument("--block-size", type=int, default=8)
    parser.add_argument("--head-dim", type=int, default=2)
    parser.add_argument("--kv-pattern", choices=("arange", "nonlinear"), default="arange")
    parser.add_argument("--recover-mode", choices=("none", "fill", "repeat_last", "linear_tail", "oracle"), default=None)
    parser.add_argument("--recover-ridge", type=float, default=1e-6)
    parser.add_argument("--recover-fill-value", type=float, default=None)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)

    policy = _RUNTIME.load_light_doc_cache_policy(args.policy_file)
    config = _RUNTIME.build_config_from_policy_dirs(
        policy,
        repo_root=args.repo_root,
        num_layers=args.num_layers,
        num_kv_heads=args.num_kv_heads,
        enabled=True,
    )
    plan = _RUNTIME.build_light_doc_cache_runtime_plan(
        config,
        task_id=args.task_id,
        doc_id=args.doc_id,
        seq_len=args.seq_len,
    )
    shape = (2, args.num_layers, args.num_blocks, args.block_size, args.num_kv_heads, args.head_dim)
    kv = _make_toy_kv(shape, args.kv_pattern)
    storage = _RUNTIME.LightDocCacheCompressedKVStorage.from_full_kv(kv, plan, fill_value=-1.0)
    recovery_mode = args.recover_mode
    if recovery_mode is None:
        recovery_mode = "fill" if args.recover_fill_value is not None else "none"

    def recover_missing(**kwargs):
        return np.full(
            (2, kwargs["missing_tokens"], kwargs["head_dim"]),
            0.0 if args.recover_fill_value is None else args.recover_fill_value,
            dtype=kwargs["dtype"],
        )

    if recovery_mode == "oracle":
        recover_missing_fn = _RUNTIME.make_oracle_recovery_callback(kv, plan)
    elif recovery_mode == "linear_tail":
        recover_missing_fn = _RUNTIME.make_linear_tail_recovery_callback(ridge=args.recover_ridge)
    elif recovery_mode == "repeat_last":
        recover_missing_fn = _RUNTIME.make_repeat_last_recovery_callback()
    elif recovery_mode == "fill":
        recover_missing_fn = recover_missing
    else:
        recover_missing_fn = None
    restored = storage.restore_to_full_shape(
        recover_missing_fn=recover_missing_fn
    )
    storage_summary = storage.summary()
    error_metrics = _RUNTIME.evaluate_restored_kv_error(kv, restored, plan)
    output = {
        "claim_boundary": storage_summary["claim_boundary"],
        "error_metrics": error_metrics,
        "plan": plan.as_summary(),
        "kv_pattern": args.kv_pattern,
        "recovery_mode": recovery_mode,
        "storage": storage_summary,
        "restored_shape": list(restored.shape),
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "storage_prototype_summary.json").write_text(
        json.dumps(output, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    lines = [
        "# Light Doc Cache Storage Prototype Smoke",
        "",
        "Boundary: CPU/toy storage prototype; missing compact-head tokens are fill/repeat-last/linear-tail/oracle baselines, not trained model-quality tensors.",
        "",
        f"- Recovery mode: `{recovery_mode}`",
        f"- KV pattern: `{args.kv_pattern}`",
        f"- Missing-token MSE: `{error_metrics['mse_missing_compact_tokens']:.6g}`",
        f"- Missing-token max abs error: `{error_metrics['max_abs_missing_compact_tokens']:.6g}`",
        f"- Full tensor bytes: `{storage_summary['full_tensor_bytes']:,}`",
        f"- Stored tensor bytes: `{storage_summary['stored_tensor_bytes']:,}`",
        f"- Saved tensor bytes: `{storage_summary['saved_tensor_bytes']:,}`",
        f"- Byte saving fraction: `{storage_summary['byte_saving_fraction']:.2%}`",
        f"- Compact heads: `{storage_summary['compact_heads']}`",
        f"- Full heads: `{storage_summary['full_heads']}`",
    ]
    (output_dir / "storage_prototype_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"output_dir": str(output_dir), "byte_saving_fraction": storage_summary["byte_saving_fraction"]}))
    return 0


def _make_toy_kv(shape: tuple[int, ...], pattern: str):
    if pattern == "arange":
        return np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
    if pattern == "nonlinear":
        _, layers, blocks, block_size, kv_heads, head_dim = shape
        tokens = blocks * block_size
        token_pos = np.arange(tokens, dtype=np.float32).reshape(1, 1, blocks, block_size, 1, 1)
        layer_pos = np.arange(layers, dtype=np.float32).reshape(1, layers, 1, 1, 1, 1)
        head_pos = np.arange(kv_heads, dtype=np.float32).reshape(1, 1, 1, 1, kv_heads, 1)
        dim_pos = np.arange(head_dim, dtype=np.float32).reshape(1, 1, 1, 1, 1, head_dim)
        kv_pos = np.arange(2, dtype=np.float32).reshape(2, 1, 1, 1, 1, 1)
        return (
            np.sin(token_pos * 0.73 + layer_pos * 0.11 + head_pos * 0.17 + dim_pos * 0.03)
            + 0.05 * token_pos * token_pos
            + 0.01 * layer_pos
            - 0.02 * head_pos
            + 0.5 * kv_pos
        ).astype(np.float32)
    raise ValueError(f"unknown kv pattern: {pattern}")


if __name__ == "__main__":
    raise SystemExit(main())
