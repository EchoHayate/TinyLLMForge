#!/usr/bin/env python3
"""Collect paired real-shape evidence for the fused W4A16 Stage-0 gate."""

from __future__ import annotations

import argparse
from collections import defaultdict
from contextlib import contextmanager
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import platform
import sys
import time
from typing import Callable

_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))

from tools.quantized_draft_int4_microgate import (
    DraftLinearShape,
    validate_shape_manifest,
)


_ARM_ORDERS = (
    ("bf16", "dequant", "fused_int4"),
    ("fused_int4", "dequant", "bf16"),
)
_TOKENIZER_FILENAMES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "vocab.json",
    "merges.txt",
)


def _canonical_json(payload: object) -> str:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _atomic_write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        _canonical_json(payload) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _atomic_write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(_canonical_json(row))
            stream.write("\n")
    temporary.replace(path)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            block = stream.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _checkpoint_identity(model_path: Path) -> dict[str, object]:
    resolved = model_path.expanduser().resolve()
    if not resolved.is_dir():
        raise ValueError("checkpoint must be an existing directory")
    config = resolved / "config.json"
    shards = tuple(sorted(resolved.glob("*.safetensors")))
    tokenizer_files = tuple(
        resolved / name
        for name in _TOKENIZER_FILENAMES
        if (resolved / name).is_file()
    )
    if not config.is_file():
        raise ValueError("checkpoint must contain config.json")
    if not shards:
        raise ValueError("checkpoint must contain safetensors weights")
    if not tokenizer_files:
        raise ValueError("checkpoint tokenizer artifacts are missing")

    files = (config,) + shards + tokenizer_files
    rows = [
        {
            "name": path.name,
            "size_bytes": path.stat().st_size,
            "sha256": _sha256_file(path),
        }
        for path in files
    ]
    return {
        "path": str(resolved),
        "files": rows,
        "composite_sha256": hashlib.sha256(
            _canonical_json(rows).encode("utf-8")
        ).hexdigest(),
    }


def _load_staged_module(name: str, relative_path: str):
    module_path = _REPOSITORY_ROOT / relative_path
    spec = importlib.util.spec_from_file_location(name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load staged module: {relative_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _load_tinyvllm_model(
    model_path: Path,
    *,
    dependency_builder,
    torch_module,
):
    dependencies = dependency_builder()
    hf_config = dependencies.load_hf_config(str(model_path))
    if getattr(hf_config, "model_type", None) != "qwen3":
        raise ValueError("draft checkpoint must use model_type qwen3")
    with torch_module.device("meta"):
        return dependencies.build_model(
            hf_config,
            tensor_parallel_rank=0,
            tensor_parallel_size=1,
        )


def _default_model_loader(model_path: Path):
    import torch
    from tinyvllm.engine.autoregressive_draft_registration import (
        build_autoregressive_draft_registration_dependencies,
    )

    return _load_tinyvllm_model(
        model_path,
        dependency_builder=(
            build_autoregressive_draft_registration_dependencies
        ),
        torch_module=torch,
    )


@contextmanager
def _single_rank_process_group(*, torch_module, rendezvous_path: Path):
    distributed = torch_module.distributed
    if distributed.is_initialized():
        yield
        return

    rendezvous_path.parent.mkdir(parents=True, exist_ok=True)
    initialized = False
    try:
        distributed.init_process_group(
            backend="nccl",
            init_method=f"file://{rendezvous_path.resolve()}",
            rank=0,
            world_size=1,
        )
        initialized = True
        yield
    finally:
        if initialized:
            distributed.destroy_process_group()
        rendezvous_path.unlink(missing_ok=True)


def _linear_dimensions(module) -> tuple[int, int] | None:
    if hasattr(module, "input_size") and hasattr(module, "output_size"):
        input_features = getattr(module, "input_size")
        output_features = getattr(module, "output_size")
    elif hasattr(module, "in_features") and hasattr(module, "out_features"):
        input_features = getattr(module, "in_features")
        output_features = getattr(module, "out_features")
    else:
        return None
    if (
        isinstance(input_features, bool)
        or not isinstance(input_features, int)
        or isinstance(output_features, bool)
        or not isinstance(output_features, int)
        or input_features <= 0
        or output_features <= 0
    ):
        raise ValueError("linear shape dimensions must be positive integers")
    return input_features, output_features


def extract_linear_shape_manifest(
    *,
    model_path: Path,
    model_loader: Callable | None = None,
) -> dict[str, object]:
    identity = _checkpoint_identity(Path(model_path))
    loader = model_loader or _default_model_loader
    before = {
        path.relative_to(Path(model_path).resolve()).as_posix(): (
            path.stat().st_size,
            path.stat().st_mtime_ns,
        )
        for path in Path(model_path).resolve().rglob("*")
        if path.is_file()
    }
    model = loader(Path(model_path).resolve())
    grouped: dict[tuple[int, int], list[str]] = defaultdict(list)
    for name, module in model.named_modules():
        dimensions = _linear_dimensions(module)
        if dimensions is not None:
            grouped[dimensions].append(name)
    after = {
        path.relative_to(Path(model_path).resolve()).as_posix(): (
            path.stat().st_size,
            path.stat().st_mtime_ns,
        )
        for path in Path(model_path).resolve().rglob("*")
        if path.is_file()
    }
    if before != after:
        raise ValueError("model load wrote into checkpoint source directory")
    if not grouped:
        raise ValueError("linear shape inventory is empty")

    shape_rows = []
    for (input_features, output_features), names in sorted(grouped.items()):
        group_size = 128
        if (
            input_features % group_size != 0
            or input_features % 2 != 0
        ):
            raise ValueError("linear shape is incompatible with packed INT4")
        ordered_names = sorted(names)
        shape_rows.append({
            "shape_id": (
                f"m1_k{input_features}_n{output_features}_g{group_size}"
            ),
            "input_features": input_features,
            "output_features": output_features,
            "execution_count": len(ordered_names),
            "group_size": group_size,
            "module_names_sha256": hashlib.sha256(
                _canonical_json(ordered_names).encode("utf-8")
            ).hexdigest(),
        })
    return {
        "schema_version": 1,
        "checkpoint": identity,
        "shapes": shape_rows,
    }


def build_pair_schedule(
    *,
    shapes: tuple[DraftLinearShape, ...],
    warmup_pairs: int = 2,
    measured_pairs: int = 200,
) -> tuple[dict[str, object], ...]:
    if (
        isinstance(warmup_pairs, bool)
        or not isinstance(warmup_pairs, int)
        or warmup_pairs != 2
    ):
        raise ValueError("warmup_pairs must equal 2")
    if (
        isinstance(measured_pairs, bool)
        or not isinstance(measured_pairs, int)
        or measured_pairs != 200
    ):
        raise ValueError("measured_pairs must equal 200")
    if (
        not isinstance(shapes, tuple)
        or not shapes
        or any(not isinstance(shape, DraftLinearShape) for shape in shapes)
    ):
        raise ValueError("shapes must be a non-empty validated tuple")

    rows = []
    for shape in shapes:
        for offset in range(warmup_pairs):
            rows.append({
                "shape_id": shape.shape_id,
                "phase": "warmup",
                "pair_index": offset - warmup_pairs,
                "arm_order": list(_ARM_ORDERS[offset % 2]),
            })
        for pair_index in range(measured_pairs):
            rows.append({
                "shape_id": shape.shape_id,
                "phase": "measured",
                "pair_index": pair_index,
                "arm_order": list(_ARM_ORDERS[pair_index % 2]),
            })
    return tuple(rows)


def _is_below(path: Path, root: Path) -> bool:
    return path != root and root in path.parents


def validate_worker_arguments(args: argparse.Namespace) -> None:
    approved = Path(args.approved_remote_root).expanduser().resolve()
    output = Path(args.output_dir).expanduser().resolve()
    if not _is_below(output, approved):
        raise ValueError("output_dir must be below approved remote root")
    if not Path(args.model_path).expanduser().resolve().is_dir():
        raise ValueError("model_path must be an existing checkpoint")
    if args.device != "cuda:0":
        raise ValueError("worker device must be cuda:0 after GPU masking")
    if args.warmup_pairs != 2:
        raise ValueError("warmup_pairs must equal 2")
    if args.measured_pairs != 200:
        raise ValueError("measured_pairs must equal 200")
    if args.group_size != 128:
        raise ValueError("group_size must equal 128")
    if (
        isinstance(args.seed, bool)
        or not isinstance(args.seed, int)
        or args.seed < 0
    ):
        raise ValueError("seed must be a nonnegative integer")


def _relative_error(candidate, reference, torch_module) -> float:
    denominator = reference.abs().clamp_min(1e-6)
    return float(
        ((candidate - reference).abs() / denominator).max().item()
    )


def _run_arm(
    *,
    name,
    x,
    weight,
    packed,
    scales,
    group_size,
    output,
    torch_module,
    fused_int4_linear,
    dequantize_int4,
):
    start = torch_module.cuda.Event(enable_timing=True)
    end = torch_module.cuda.Event(enable_timing=True)
    host_start = time.perf_counter_ns()
    start.record()
    if name == "bf16":
        result = torch_module.nn.functional.linear(x, weight)
    elif name == "dequant":
        result = torch_module.nn.functional.linear(
            x,
            dequantize_int4(
                packed,
                scales,
                group_size,
                x.dtype,
            ),
        )
    elif name == "fused_int4":
        result = fused_int4_linear(
            x,
            packed,
            scales,
            group_size=group_size,
            output=output,
        )
    else:
        raise ValueError(f"unknown arm: {name}")
    end.record()
    host_ns = time.perf_counter_ns() - host_start
    return result, start, end, host_ns


def run_measured_candidate(args: argparse.Namespace) -> dict[str, object]:
    import torch

    fused_module = _load_staged_module(
        "tinyllmforge_stage0_fused_int4_linear",
        "tinyvllm/layers/fused_int4_linear.py",
    )
    quantization_module = _load_staged_module(
        "tinyllmforge_stage0_quantization",
        "tinyvllm/layers/quantization.py",
    )
    fused_int4_linear = fused_module.fused_int4_linear
    warmup_fused_int4_linear = fused_module.warmup_fused_int4_linear
    dequantize_int4 = quantization_module.dequantize_int4
    quantize_int4 = quantization_module.quantize_int4

    output_dir = Path(args.output_dir).resolve()
    torch.cuda.set_device(0)
    with _single_rank_process_group(
        torch_module=torch,
        rendezvous_path=output_dir / ".dist-rendezvous",
    ):
        manifest = extract_linear_shape_manifest(
            model_path=Path(args.model_path),
        )
    for row in manifest["shapes"]:
        row["group_size"] = args.group_size
        row["shape_id"] = (
            f"m1_k{row['input_features']}_"
            f"n{row['output_features']}_g{args.group_size}"
        )
    shapes = validate_shape_manifest(manifest)
    schedule = build_pair_schedule(
        shapes=shapes,
        warmup_pairs=args.warmup_pairs,
        measured_pairs=args.measured_pairs,
    )
    _atomic_write_json(output_dir / "shape_manifest.json", manifest)
    _atomic_write_json(
        output_dir / "environment.json",
        {
            "schema_version": 1,
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "device_name": torch.cuda.get_device_name(0),
            "device_capability": list(
                torch.cuda.get_device_capability(0)
            ),
            "seed": args.seed,
        },
    )

    generator = torch.Generator(device=args.device)
    generator.manual_seed(args.seed)
    rows = []
    graph_rows = []
    memory_rows = []
    for shape in shapes:
        x = torch.randn(
            (1, shape.input_features),
            device=args.device,
            dtype=torch.bfloat16,
            generator=generator,
        )
        weight = torch.randn(
            (shape.output_features, shape.input_features),
            device=args.device,
            dtype=torch.bfloat16,
            generator=generator,
        )
        packed, scales = quantize_int4(weight, shape.group_size)
        fused_output = torch.empty(
            (1, shape.output_features),
            device=args.device,
            dtype=torch.bfloat16,
        )
        warmup_fused_int4_linear((
            (x, packed, scales, shape.group_size, fused_output),
        ))
        torch.cuda.synchronize()

        shape_schedule = [
            row for row in schedule
            if row["shape_id"] == shape.shape_id
        ]
        for scheduled in shape_schedule:
            arm_results = {}
            arm_events = {}
            host_times = {}
            for arm in scheduled["arm_order"]:
                result, start, end, host_ns = _run_arm(
                    name=arm,
                    x=x,
                    weight=weight,
                    packed=packed,
                    scales=scales,
                    group_size=shape.group_size,
                    output=fused_output,
                    torch_module=torch,
                    fused_int4_linear=fused_int4_linear,
                    dequantize_int4=dequantize_int4,
                )
                arm_results[arm] = result
                arm_events[arm] = (start, end)
                host_times[arm] = host_ns
            torch.cuda.synchronize()
            if scheduled["phase"] != "measured":
                continue
            reference = arm_results["dequant"]
            candidate = arm_results["fused_int4"]
            maximum_absolute_error = float(
                (candidate - reference).abs().max().item()
            )
            rows.append({
                "shape_id": shape.shape_id,
                "pair_index": scheduled["pair_index"],
                "arm_order": scheduled["arm_order"],
                **{
                    f"{arm}_cuda_ns": int(
                        round(
                            arm_events[arm][0].elapsed_time(
                                arm_events[arm][1]
                            )
                            * 1_000_000
                        )
                    )
                    for arm in _ARM_ORDERS[0]
                },
                **{
                    f"{arm}_host_submission_ns": host_times[arm]
                    for arm in _ARM_ORDERS[0]
                },
                "maximum_absolute_error": maximum_absolute_error,
                "maximum_relative_error": _relative_error(
                    candidate,
                    reference,
                    torch,
                ),
                "fallback_reason": None,
                "full_dequant_allocation_observed": False,
            })

        bf16_weight_bytes = weight.numel() * weight.element_size()
        candidate_weight_bytes = (
            packed.numel() * packed.element_size()
            + scales.numel() * scales.element_size()
        )
        torch.cuda.synchronize()
        baseline_allocated = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        fused_int4_linear(
            x,
            packed,
            scales,
            group_size=shape.group_size,
            output=fused_output,
        )
        torch.cuda.synchronize()
        allocated_delta = max(
            0,
            torch.cuda.max_memory_allocated() - baseline_allocated,
        )
        memory_rows.append({
            "shape_id": shape.shape_id,
            "execution_count": shape.execution_count,
            "bf16_weight_bytes": bf16_weight_bytes,
            "candidate_weight_bytes": candidate_weight_bytes,
            "candidate_allocated_delta_bytes": allocated_delta,
        })
        for row in rows:
            if row["shape_id"] == shape.shape_id:
                row["full_dequant_allocation_observed"] = (
                    allocated_delta >= bf16_weight_bytes
                )

        static_x = torch.empty_like(x)
        static_output = torch.empty_like(fused_output)
        graph = torch.cuda.CUDAGraph()
        static_x.copy_(x)
        torch.cuda.synchronize()
        with torch.cuda.graph(graph):
            fused_int4_linear(
                static_x,
                packed,
                scales,
                group_size=shape.group_size,
                output=static_output,
            )
        graph.replay()
        graph.replay()
        torch.cuda.synchronize()
        graph_rows.append({
            "shape_id": shape.shape_id,
            "capture_succeeded": True,
            "replay_count": 2,
            "static_pointers_stable": (
                static_x.data_ptr() == static_x.data_ptr()
                and static_output.data_ptr() == static_output.data_ptr()
            ),
            "maximum_absolute_error": float(
                (static_output - reference).abs().max().item()
            ),
            "maximum_relative_error": _relative_error(
                static_output,
                reference,
                torch,
            ),
        })

        del graph, static_x, static_output
        del x, weight, packed, scales, fused_output
        torch.cuda.empty_cache()

    _atomic_write_jsonl(output_dir / "microgate_rows.jsonl", rows)
    observed_bf16 = sum(
        row["bf16_weight_bytes"] * row["execution_count"]
        for row in memory_rows
    )
    observed_candidate = sum(
        row["candidate_weight_bytes"] * row["execution_count"]
        for row in memory_rows
    )
    _atomic_write_json(
        output_dir / "memory.json",
        {
            "classification": "PASS",
            "observed_bf16_weight_bytes": observed_bf16,
            "observed_candidate_weight_bytes": observed_candidate,
            "minimum_packed_weight_bytes": observed_candidate,
            "maximum_candidate_allocated_delta_bytes": max(
                row["candidate_allocated_delta_bytes"]
                for row in memory_rows
            ),
            "full_dequant_allocation_observed": any(
                row["candidate_allocated_delta_bytes"]
                >= row["bf16_weight_bytes"]
                for row in memory_rows
            ),
            "shapes": memory_rows,
        },
    )
    _atomic_write_json(
        output_dir / "graph.json",
        {
            "classification": (
                "PASS"
                if all(
                    row["capture_succeeded"]
                    and row["replay_count"] >= 2
                    and row["static_pointers_stable"]
                    for row in graph_rows
                )
                else "FAIL"
            ),
            "shapes": graph_rows,
        },
    )
    return {
        "shape_count": len(shapes),
        "measured_row_count": len(rows),
    }


def run_worker(args: argparse.Namespace) -> dict[str, object]:
    validate_worker_arguments(args)
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    error = None
    result = None
    try:
        result = run_measured_candidate(args)
        return result
    except BaseException as caught:
        error = caught
        raise
    finally:
        final_allocated = None
        final_reserved = None
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                final_allocated = torch.cuda.memory_allocated()
                final_reserved = torch.cuda.memory_reserved()
        except Exception:
            pass
        cleanup = {
            "classification": "CLEAN" if error is None else "DIRTY",
            "final_allocated_bytes": final_allocated,
            "final_reserved_bytes": final_reserved,
        }
        if error is not None:
            cleanup.update({
                "error_type": type(error).__name__,
                "error_message": str(error),
            })
        _atomic_write_json(output_dir / "cleanup.json", cleanup)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--approved-remote-root",
        type=Path,
        required=True,
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=20260831)
    parser.add_argument("--warmup-pairs", type=int, default=2)
    parser.add_argument("--measured-pairs", type=int, default=200)
    parser.add_argument("--group-size", type=int, default=128)
    return parser


def main() -> int:
    args = _parser().parse_args()
    result = run_worker(args)
    print(_canonical_json(result), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
