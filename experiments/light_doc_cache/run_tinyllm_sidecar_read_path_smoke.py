"""Compare TinyLLM decode logits after reading from a restored sidecar buffer.

This smoke is default-off and does not modify attention kernels. It runs a
normal prefill, materializes the Light Doc Cache sidecar, restores it into a
temporary full KV tensor, temporarily points each attention layer at that
restored tensor, and compares one decode-step logits against the original
`ModelRunner.kv_cache` read path.
"""

from __future__ import annotations

import argparse
import atexit
import importlib.util
import json
from contextlib import contextmanager
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
    parser.add_argument("--prompt", default="Light Doc Cache TinyLLM read path smoke.")
    parser.add_argument("--task-id", default="smoothquant_status")
    parser.add_argument("--doc-id", default="second")
    parser.add_argument("--max-model-len", type=int, default=512)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.30)
    parser.add_argument(
        "--recover-mode",
        choices=(
            "none",
            "fill",
            "repeat_last",
            "linear_tail",
            "correlated",
            "multi_correlated",
            "calibrated_multi_correlated",
            "oracle",
        ),
        default="linear_tail",
    )
    parser.add_argument(
        "--recovery-bank-file",
        default=None,
        help="JSON recovery bank for --recover-mode calibrated_multi_correlated.",
    )
    parser.add_argument(
        "--correlated-source-map",
        choices=("same_layer", "prefix_fit"),
        default="same_layer",
        help="Source-head selection strategy for --recover-mode correlated.",
    )
    parser.add_argument(
        "--multi-correlated-source-count",
        type=int,
        default=2,
        help="Number of retained full heads per compact head for --recover-mode multi_correlated.",
    )
    parser.add_argument("--recover-ridge", type=float, default=1e-6)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)

    from tinyvllm import LLM

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
        summary = run_read_path_smoke(
            llm=llm,
            policy_file=args.policy_file,
            repo_root=args.repo_root,
            prompt=args.prompt,
            task_id=args.task_id,
            doc_id=args.doc_id,
            output_dir=Path(args.output_dir),
            model=args.model,
            recover_mode=args.recover_mode,
            recover_ridge=args.recover_ridge,
            correlated_source_map=args.correlated_source_map,
            multi_correlated_source_count=args.multi_correlated_source_count,
            recovery_bank_file=args.recovery_bank_file,
        )
    finally:
        try:
            atexit.unregister(llm.exit)
        except Exception:
            pass
        llm.exit()
    print(json.dumps(summary["logit_compare"], sort_keys=True))
    return 0


def run_read_path_smoke(
    *,
    llm,
    policy_file: str,
    repo_root: str | Path,
    prompt: str,
    task_id: str,
    doc_id: str | None,
    output_dir: str | Path,
    model: str,
    recover_mode: str,
    recover_ridge: float,
    correlated_source_map: str = "same_layer",
    multi_correlated_source_count: int = 2,
    recovery_bank_file: str | None = None,
) -> dict:
    import torch
    from tinyvllm.engine.sequence import Sequence, SequenceStatus
    from tinyvllm.sampling_params import SamplingParams
    from tinyvllm.utils.context import reset_context

    runner = llm.model_runner
    block_manager = llm.scheduler.block_manager
    block_manager.hash_to_block_id.clear()
    prompt_ids = llm.tokenizer.encode(prompt, add_special_tokens=False)
    if len(prompt_ids) < 1:
        raise ValueError("prompt must encode to at least one token")
    seq = Sequence(prompt_ids, SamplingParams(temperature=0.0, max_tokens=2, ignore_eos=True))
    block_manager.allocate(seq)
    seq.status = SequenceStatus.RUNNING
    seq.num_computed_tokens = len(seq)
    seq.prefill_chunk_start = seq.num_cached_tokens
    seq.prefill_chunk_end = len(seq)
    seq.prefill_chunk_final = True

    try:
        input_ids, positions = runner.prepare_prefill([seq])
        prefill_logits = runner.run_model(input_ids, positions, True)
        reset_context()
        first_decode_token = int(torch.argmax(prefill_logits[-1], dim=-1).item())
        seq.append_token(first_decode_token)
        block_manager.may_append(seq)

        input_ids, positions = runner.prepare_decode([seq])
        original_logits = runner.run_model(input_ids, positions, False)
        reset_context()

        policy = _RUNTIME.load_light_doc_cache_policy(policy_file)
        shape = tuple(int(dim) for dim in runner.kv_cache.shape)
        plan = _RUNTIME.build_light_doc_cache_runtime_plan(
            _RUNTIME.build_config_from_policy_dirs(
                policy,
                repo_root=repo_root,
                num_layers=shape[1],
                num_kv_heads=shape[4],
                enabled=True,
            ),
            task_id=task_id,
            doc_id=doc_id,
            seq_len=len(prompt_ids),
        )
        sequence_kv = pack_sequence_kv_blocks(runner.kv_cache, seq.block_table)
        storage = _RUNTIME.LightDocCacheCompressedKVStorage.from_full_kv(
            sequence_kv,
            plan,
            fill_value=-1.0,
        )
        recover_missing_fn = _make_recovery_callback(
            recover_mode,
            sequence_kv,
            plan,
            recover_ridge,
            storage,
            correlated_source_map=correlated_source_map,
            multi_correlated_source_count=multi_correlated_source_count,
            recovery_bank_file=recovery_bank_file,
        )
        restored = storage.restore_to_full_shape(recover_missing_fn=recover_missing_fn)
        sidecar_summary = _RUNTIME._build_sidecar_materialization_summary(
            plan,
            storage,
            sequence_kv,
            restored,
            evaluate_readback=True,
        )
        restored_kv_cache = scatter_sequence_kv_blocks(
            runner.kv_cache,
            restored,
            seq.block_table,
        )

        input_ids, positions = runner.prepare_decode([seq])
        with _temporary_model_runner_kv_cache(runner, restored_kv_cache):
            restored_logits = runner.run_model(input_ids, positions, False)
        reset_context()

        diff = (restored_logits - original_logits).detach().to(torch.float32)
        compare = {
            "max_abs_logit_diff": float(diff.abs().max().item()),
            "mean_abs_logit_diff": float(diff.abs().mean().item()),
            "original_argmax": int(torch.argmax(original_logits[-1], dim=-1).item()),
            "restored_argmax": int(torch.argmax(restored_logits[-1], dim=-1).item()),
            "argmax_match": bool(
                int(torch.argmax(original_logits[-1], dim=-1).item())
                == int(torch.argmax(restored_logits[-1], dim=-1).item())
            ),
        }
        summary = {
            "claim_boundary": "default_off_restored_sidecar_read_path_logits_compare",
            "model": model,
            "prompt_tokens": len(prompt_ids),
            "decode_token": first_decode_token,
            "kv_cache_shape": list(shape),
            "plan": plan.as_summary(),
            "recovery_mode": recover_mode,
            "correlated_source_map": correlated_source_map if recover_mode == "correlated" else None,
            "multi_correlated_source_count": multi_correlated_source_count if recover_mode == "multi_correlated" else None,
            "recovery_bank_file": recovery_bank_file if recover_mode == "calibrated_multi_correlated" else None,
            "sidecar": sidecar_summary,
            "logit_compare": compare,
        }
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "tinyllm_sidecar_read_path_summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        _write_report(output_dir / "tinyllm_sidecar_read_path_report.md", summary)
        return summary
    finally:
        try:
            reset_context()
        except Exception:
            pass
        block_manager.deallocate(seq)


def pack_sequence_kv_blocks(kv_cache, block_table: list[int]):
    block_ids = [int(block_id) for block_id in block_table]
    if not block_ids:
        raise ValueError("sequence block table must not be empty")
    packed = kv_cache[:, :, block_ids, :, :, :]
    if hasattr(packed, "clone"):
        return packed.clone()
    if hasattr(packed, "copy"):
        return packed.copy()
    raise TypeError("kv_cache block selection must support clone() or copy()")


def scatter_sequence_kv_blocks(kv_cache, packed_kv, block_table: list[int]):
    block_ids = [int(block_id) for block_id in block_table]
    if int(packed_kv.shape[2]) != len(block_ids):
        raise ValueError("packed KV block count must match sequence block table")
    if hasattr(kv_cache, "clone"):
        restored = kv_cache.clone()
    elif hasattr(kv_cache, "copy"):
        restored = kv_cache.copy()
    else:
        raise TypeError("kv_cache must support clone() or copy()")
    for packed_block, physical_block in enumerate(block_ids):
        restored[:, :, physical_block, :, :, :] = packed_kv[
            :, :, packed_block, :, :, :
        ]
    return restored


@contextmanager
def _temporary_model_runner_kv_cache(runner, restored_kv):
    original_kv_cache = runner.kv_cache
    original_pairs = []
    layer_id = 0
    for module in runner.model.modules():
        if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
            original_pairs.append((module, module.k_cache, module.v_cache))
            module.k_cache = restored_kv[0, layer_id]
            module.v_cache = restored_kv[1, layer_id]
            layer_id += 1
    runner.kv_cache = restored_kv
    try:
        yield
    finally:
        runner.kv_cache = original_kv_cache
        for module, k_cache, v_cache in original_pairs:
            module.k_cache = k_cache
            module.v_cache = v_cache


def _make_recovery_callback(
    mode: str,
    kv_cache,
    plan,
    recover_ridge: float,
    storage=None,
    *,
    correlated_source_map: str = "same_layer",
    multi_correlated_source_count: int = 2,
    recovery_bank_file: str | None = None,
):
    if mode == "oracle":
        return _RUNTIME.make_oracle_recovery_callback(kv_cache, plan)
    if mode == "correlated":
        if storage is None:
            storage = _RUNTIME.LightDocCacheCompressedKVStorage.from_full_kv(kv_cache, plan, fill_value=-1.0)
        if correlated_source_map == "prefix_fit":
            source_heads = _RUNTIME.build_correlated_source_head_map(storage, ridge=recover_ridge)
        elif correlated_source_map == "same_layer":
            source_heads = _build_correlated_source_heads(plan)
        else:
            raise ValueError(f"unknown correlated_source_map: {correlated_source_map}")
        return _RUNTIME.make_correlated_head_recovery_callback(
            storage,
            source_heads=source_heads,
            ridge=recover_ridge,
        )
    if mode == "multi_correlated":
        if storage is None:
            storage = _RUNTIME.LightDocCacheCompressedKVStorage.from_full_kv(kv_cache, plan, fill_value=-1.0)
        return _RUNTIME.make_multi_source_correlated_head_recovery_callback(
            storage,
            source_heads=_build_multi_correlated_source_heads(plan, int(multi_correlated_source_count)),
            ridge=recover_ridge,
        )
    if mode == "calibrated_multi_correlated":
        if recovery_bank_file is None:
            raise ValueError("--recovery-bank-file is required for calibrated_multi_correlated")
        if storage is None:
            storage = _RUNTIME.LightDocCacheCompressedKVStorage.from_full_kv(kv_cache, plan, fill_value=-1.0)
        bank = _RUNTIME.load_multi_source_recovery_bank(recovery_bank_file)
        return _RUNTIME.make_calibrated_multi_source_recovery_callback(storage, bank)
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


def _build_multi_correlated_source_heads(plan, source_count: int) -> dict[tuple[int, int], list[tuple[int, int]]]:
    source_count = max(1, int(source_count))
    compact_heads = set(plan.recovered_heads)
    full_heads = [
        (layer, head)
        for layer in range(int(plan.num_layers))
        for head in range(int(plan.num_kv_heads))
        if (layer, head) not in compact_heads
    ]
    if not full_heads:
        raise ValueError("multi-correlated recovery requires at least one retained full source head")
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


def _build_correlated_source_heads(plan) -> dict[tuple[int, int], tuple[int, int]]:
    compact_heads = set(plan.recovered_heads)
    full_heads = [
        (layer, head)
        for layer in range(int(plan.num_layers))
        for head in range(int(plan.num_kv_heads))
        if (layer, head) not in compact_heads
    ]
    if not full_heads:
        raise ValueError("correlated recovery requires at least one retained full source head")
    source_heads = {}
    for target_layer, target_head in plan.recovered_heads:
        same_layer = [
            (layer, head)
            for layer, head in full_heads
            if layer == int(target_layer) and head != int(target_head)
        ]
        source_heads[(int(target_layer), int(target_head))] = same_layer[0] if same_layer else full_heads[0]
    return source_heads


def _write_report(path: Path, summary: dict) -> None:
    compare = summary["logit_compare"]
    sidecar = summary["sidecar"]
    lines = [
        "# TinyLLM Sidecar Read-Path Smoke",
        "",
        "Boundary: temporarily points attention layer cache pointers at a restored sidecar buffer for one decode-step logits comparison; no hot-path code or KV allocation lifetime is changed.",
        "",
        f"- Model: `{summary['model']}`",
        f"- Prompt tokens: `{summary['prompt_tokens']}`",
        f"- KV cache shape: `{summary['kv_cache_shape']}`",
        f"- Recovery mode: `{summary['recovery_mode']}`",
        f"- Logical stored KV bytes: `{sidecar['logical_stored_kv_bytes']:,}`",
        f"- Logical byte saving fraction: `{sidecar['logical_byte_saving_fraction']:.2%}`",
        f"- Missing-token MSE: `{sidecar['error_metrics']['mse_missing_compact_tokens']:.6g}`",
        f"- Max abs logit diff: `{compare['max_abs_logit_diff']:.6g}`",
        f"- Mean abs logit diff: `{compare['mean_abs_logit_diff']:.6g}`",
        f"- Argmax match: `{compare['argmax_match']}`",
        f"- Original argmax: `{compare['original_argmax']}`",
        f"- Restored argmax: `{compare['restored_argmax']}`",
    ]
    if summary["correlated_source_map"] is not None:
        lines.insert(8, f"- Correlated source map: `{summary['correlated_source_map']}`")
    if summary["multi_correlated_source_count"] is not None:
        lines.insert(8, f"- Multi-correlated source count: `{summary['multi_correlated_source_count']}`")
    if summary["recovery_bank_file"] is not None:
        lines.insert(8, f"- Recovery bank file: `{summary['recovery_bank_file']}`")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
