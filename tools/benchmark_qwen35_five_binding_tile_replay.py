from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import tempfile
import time
import types

import safetensors
from safetensors import safe_open
from safetensors.torch import save_file
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
for package_name in (
    "tinyvllm",
    "tinyvllm.layers",
    "tinyvllm.models",
):
    if package_name not in sys.modules:
        package = types.ModuleType(package_name)
        package.__path__ = [
            str(ROOT / package_name.replace(".", "/"))
        ]
        sys.modules[package_name] = package

from tinyvllm.layers.linear import SegmentedColumnParallelLinear
from tinyvllm.models.qwen35_checkpoint import (
    Qwen35CheckpointLoadTarget,
    Qwen35CheckpointSource,
    Qwen35CheckpointTensorLoad,
    Qwen35CheckpointTensorMetadata,
)
from tinyvllm.models.qwen35_checkpoint_binding import (
    Qwen35CheckpointBindingPlan,
    Qwen35CheckpointTensorBinding,
)
from tinyvllm.models.qwen35_checkpoint_tiles import (
    build_qwen35_checkpoint_tile_plan,
)


SCHEMA_VERSION = "qwen35.five-binding-synthetic-tile-replay.v1"
MIB = 1 << 20
TP_SIZE = 2
TP_RANK = 1
BF16_BYTES = 2
MATRIX_WIDTH = 2048
CONV_KERNEL = 4


@dataclass
class _SyntheticCase:
    kind: str
    source_name: str
    source: torch.Tensor
    destination: torch.Tensor
    expected: torch.Tensor
    binding_plan: Qwen35CheckpointBindingPlan
    owner: object | None


def _positive_integer(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _validate_inputs(tile_mib, repeats, local_payload_mib):
    repeats = _positive_integer(repeats, "repeats")
    local_payload_mib = _positive_integer(
        local_payload_mib,
        "local_payload_mib",
    )
    if not isinstance(tile_mib, tuple) or not tile_mib:
        raise ValueError("tile_mib must be a non-empty tuple")
    tile_mib = tuple(
        _positive_integer(value, "tile_mib entry")
        for value in tile_mib
    )
    return tile_mib, repeats, local_payload_mib


def _source_tensor(shape, seed):
    element_count = 1
    for dimension in shape:
        element_count *= dimension
    values = torch.arange(element_count, dtype=torch.int32)
    values.remainder_(127)
    values.add_(seed)
    return values.to(dtype=torch.bfloat16).reshape(shape)


def _load(
    *,
    source_name,
    target,
    shape,
    transform="identity",
):
    payload_bytes = BF16_BYTES
    for dimension in shape:
        payload_bytes *= dimension
    return Qwen35CheckpointTensorLoad(
        weight=Qwen35CheckpointLoadTarget(
            source=Qwen35CheckpointSource(
                name=source_name,
                shard=f"{source_name}.safetensors",
            ),
            target=target,
            packed_slot=None,
        ),
        metadata=Qwen35CheckpointTensorMetadata(
            dtype="BF16",
            shape=shape,
            data_offsets=(0, payload_bytes),
        ),
        transform=transform,
    )


def _binding(
    *,
    load,
    destination,
    loader_kind,
    local_shape,
):
    return Qwen35CheckpointTensorBinding(
        load=load,
        destination_name=load.weight.target,
        destination=destination,
        destination_kind=(
            "parameter"
            if isinstance(destination, torch.nn.Parameter)
            else "buffer"
        ),
        loader_kind=loader_kind,
        local_shape=local_shape,
        destination_slice=None,
    )


def _plan(binding):
    return Qwen35CheckpointBindingPlan(
        bindings=(binding,),
        tensor_parallel_size=TP_SIZE,
        tensor_parallel_rank=TP_RANK,
    )


def _matrix_rows(local_payload_bytes):
    row_bytes = MATRIX_WIDTH * BF16_BYTES
    if local_payload_bytes % row_bytes:
        raise ValueError(
            "local_payload_mib must produce complete matrix rows"
        )
    return local_payload_bytes // row_bytes


def _build_axis0_case(local_payload_bytes):
    local_rows = _matrix_rows(local_payload_bytes)
    source_shape = (local_rows * TP_SIZE, MATRIX_WIDTH)
    source = _source_tensor(source_shape, 1)
    expected = source.narrow(
        0,
        TP_RANK * local_rows,
        local_rows,
    ).clone()
    destination = torch.empty_like(expected)
    load = _load(
        source_name="axis0",
        target="embed_tokens.weight",
        shape=source_shape,
    )
    binding = _binding(
        load=load,
        destination=destination,
        loader_kind="custom_parameter_loader",
        local_shape=tuple(expected.shape),
    )
    return _SyntheticCase(
        kind="axis0",
        source_name=load.weight.source.name,
        source=source,
        destination=destination,
        expected=expected,
        binding_plan=_plan(binding),
        owner=None,
    )


def _build_axis1_case(local_payload_bytes):
    rows = _matrix_rows(local_payload_bytes)
    local_columns = MATRIX_WIDTH
    source_shape = (rows, local_columns * TP_SIZE)
    source = _source_tensor(source_shape, 17)
    expected = source.narrow(
        1,
        TP_RANK * local_columns,
        local_columns,
    ).clone()
    destination = torch.empty_like(expected)
    load = _load(
        source_name="axis1",
        target="layers.0.mlp.down_proj.weight",
        shape=source_shape,
    )
    binding = _binding(
        load=load,
        destination=destination,
        loader_kind="custom_parameter_loader",
        local_shape=tuple(expected.shape),
    )
    return _SyntheticCase(
        kind="axis1",
        source_name=load.weight.source.name,
        source=source,
        destination=destination,
        expected=expected,
        binding_plan=_plan(binding),
        owner=None,
    )


def _segmented_owner(input_size, output_sizes):
    original_world_size = torch.distributed.get_world_size
    original_rank = torch.distributed.get_rank
    torch.distributed.get_world_size = lambda: TP_SIZE
    torch.distributed.get_rank = lambda: TP_RANK
    try:
        owner = SegmentedColumnParallelLinear(
            input_size,
            output_sizes,
            bias=False,
        )
        return owner.to(dtype=torch.bfloat16)
    finally:
        torch.distributed.get_world_size = original_world_size
        torch.distributed.get_rank = original_rank


def _build_segmented_case(local_payload_bytes):
    local_rows = _matrix_rows(local_payload_bytes)
    if local_rows % 4:
        raise ValueError(
            "local_payload_mib must support segmented row quarters"
        )
    local_sizes = (
        local_rows // 4,
        local_rows // 4,
        local_rows // 2,
    )
    global_sizes = tuple(value * TP_SIZE for value in local_sizes)
    owner = _segmented_owner(MATRIX_WIDTH, global_sizes)
    destination = owner.weight
    source_shape = (sum(global_sizes), MATRIX_WIDTH)
    source = _source_tensor(source_shape, 33)
    expected_parts = []
    global_offset = 0
    for global_size, local_size in zip(
        global_sizes,
        local_sizes,
        strict=True,
    ):
        expected_parts.append(
            source.narrow(
                0,
                global_offset + TP_RANK * local_size,
                local_size,
            )
        )
        global_offset += global_size
    expected = torch.cat(expected_parts, dim=0)
    load = _load(
        source_name="segmented_axis0",
        target="layers.0.linear_attention.in_proj_qkv.weight",
        shape=source_shape,
    )
    binding = _binding(
        load=load,
        destination=destination,
        loader_kind="custom_parameter_loader",
        local_shape=tuple(expected.shape),
    )
    return _SyntheticCase(
        kind="segmented_axis0",
        source_name=load.weight.source.name,
        source=source,
        destination=destination,
        expected=expected,
        binding_plan=_plan(binding),
        owner=owner,
    )


def _build_squeeze_case(local_payload_bytes):
    row_bytes = CONV_KERNEL * BF16_BYTES
    if local_payload_bytes % row_bytes:
        raise ValueError(
            "local_payload_mib must produce complete convolution rows"
        )
    local_rows = local_payload_bytes // row_bytes
    source_shape = (local_rows * TP_SIZE, 1, CONV_KERNEL)
    source = _source_tensor(source_shape, 49)
    expected = source.narrow(
        0,
        TP_RANK * local_rows,
        local_rows,
    ).squeeze(1).clone()
    destination = torch.empty_like(expected)
    load = _load(
        source_name="squeeze_axis0",
        target="layers.0.linear_attention.conv_weight",
        shape=source_shape,
        transform="squeeze_conv_channel",
    )
    binding = _binding(
        load=load,
        destination=destination,
        loader_kind="direct_buffer_copy",
        local_shape=tuple(expected.shape),
    )
    return _SyntheticCase(
        kind="squeeze_axis0",
        source_name=load.weight.source.name,
        source=source,
        destination=destination,
        expected=expected,
        binding_plan=_plan(binding),
        owner=None,
    )


def _build_replicated_case(local_payload_bytes):
    length = local_payload_bytes // BF16_BYTES
    source_shape = (length,)
    source = _source_tensor(source_shape, 65)
    expected = source.clone()
    destination = torch.empty_like(expected)
    load = _load(
        source_name="replicated",
        target="final_norm.weight",
        shape=source_shape,
    )
    binding = _binding(
        load=load,
        destination=destination,
        loader_kind="default_parameter_copy",
        local_shape=source_shape,
    )
    return _SyntheticCase(
        kind="replicated",
        source_name=load.weight.source.name,
        source=source,
        destination=destination,
        expected=expected,
        binding_plan=_plan(binding),
        owner=None,
    )


def _case_builders():
    return (
        _build_axis0_case,
        _build_axis1_case,
        _build_segmented_case,
        _build_squeeze_case,
        _build_replicated_case,
    )


def _checksum(tensor):
    return float(tensor.to(dtype=torch.float32).sum().item())


def _run_once(case, tile_plan, shard_path):
    with torch.no_grad():
        case.destination.fill_(-1)
    started = time.perf_counter()
    with safe_open(shard_path, framework="pt", device="cpu") as handle:
        slice_view = handle.get_slice(case.source_name)
        for tile in tile_plan.tiles:
            materialized = slice_view[tile.source_slices]
            with torch.no_grad():
                tile.destination[tile.destination_slices].copy_(
                    materialized
                )
    elapsed = time.perf_counter() - started
    if not torch.equal(case.destination, case.expected):
        raise RuntimeError(
            f"{case.kind} synthetic replay destination mismatch"
        )
    return elapsed, _checksum(case.destination)


def _timing_record(
    raw_seconds,
    *,
    tile_count,
    requested_tile_bytes,
    peak_tile_bytes,
    destination_checksum,
):
    return {
        "raw_seconds": [float(value) for value in raw_seconds],
        "median_seconds": float(statistics.median(raw_seconds)),
        "min_seconds": float(min(raw_seconds)),
        "max_seconds": float(max(raw_seconds)),
        "tile_count": tile_count,
        "requested_tile_bytes": requested_tile_bytes,
        "peak_tile_bytes": peak_tile_bytes,
        "destination_checksum": destination_checksum,
        "exact_destination_verified": True,
    }


def _cpu_description():
    if sys.platform == "darwin":
        for key in ("machdep.cpu.brand_string", "hw.model"):
            try:
                value = subprocess.check_output(
                    ("/usr/sbin/sysctl", "-n", key),
                    text=True,
                    stderr=subprocess.DEVNULL,
                ).strip()
            except (OSError, subprocess.CalledProcessError):
                value = ""
            if value:
                return value
    processor = platform.processor().strip()
    return processor or platform.machine()


def _environment_record():
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "pytorch": torch.__version__,
        "safetensors": safetensors.__version__,
        "platform": platform.platform(),
        "cpu": _cpu_description(),
    }


def run_qwen35_five_binding_tile_replay_calibration(
    *,
    tile_mib: tuple[int, ...] = (4, 8, 16, 32),
    repeats: int = 3,
    local_payload_mib: int = 32,
) -> dict:
    tile_mib, repeats, local_payload_mib = _validate_inputs(
        tile_mib,
        repeats,
        local_payload_mib,
    )
    local_payload_bytes = local_payload_mib * MIB
    case_results = []

    with tempfile.TemporaryDirectory(
        prefix="qwen35-five-binding-replay-",
    ) as temporary_directory:
        directory = Path(temporary_directory)
        for build_case in _case_builders():
            case = build_case(local_payload_bytes)
            source_payload_bytes = (
                case.source.numel() * case.source.element_size()
            )
            expected_checksum = _checksum(case.expected)
            shard_path = directory / f"{case.kind}.safetensors"
            save_file({case.source_name: case.source}, shard_path)
            del case.source

            budget_results = []
            for requested_mib in tile_mib:
                requested_tile_bytes = requested_mib * MIB
                tile_plan = build_qwen35_checkpoint_tile_plan(
                    case.binding_plan,
                    max_tile_bytes=requested_tile_bytes,
                )
                if (
                    tile_plan.binding_count != 1
                    or tile_plan.source_count != 1
                    or any(
                        tile.kind != case.kind
                        for tile in tile_plan.tiles
                    )
                    or tile_plan.destination_bytes
                    != local_payload_bytes
                    or tile_plan.peak_tile_bytes
                    > requested_tile_bytes
                ):
                    raise RuntimeError(
                        f"{case.kind} synthetic tile plan mismatch"
                    )

                _run_once(case, tile_plan, shard_path)
                raw_seconds = []
                destination_checksum = None
                for _ in range(repeats):
                    elapsed, destination_checksum = _run_once(
                        case,
                        tile_plan,
                        shard_path,
                    )
                    raw_seconds.append(elapsed)
                budget_results.append(
                    _timing_record(
                        raw_seconds,
                        tile_count=len(tile_plan.tiles),
                        requested_tile_bytes=requested_tile_bytes,
                        peak_tile_bytes=tile_plan.peak_tile_bytes,
                        destination_checksum=destination_checksum,
                    )
                )

            destination_checksum = _checksum(case.destination)
            if destination_checksum != expected_checksum:
                raise RuntimeError(
                    f"{case.kind} synthetic checksum mismatch"
                )
            case_results.append({
                "kind": case.kind,
                "source_shape": list(
                    case.binding_plan.bindings[0].load.metadata.shape
                ),
                "local_shape": list(case.destination.shape),
                "source_payload_bytes": source_payload_bytes,
                "local_payload_bytes": local_payload_bytes,
                "expected_checksum": expected_checksum,
                "destination_checksum": destination_checksum,
                "budget_results": budget_results,
            })

    return {
        "schema_version": SCHEMA_VERSION,
        "environment": _environment_record(),
        "configuration": {
            "tensor_parallel_size": TP_SIZE,
            "tensor_parallel_rank": TP_RANK,
            "tile_mib": list(tile_mib),
            "repeats": repeats,
            "warmups": 1,
            "local_payload_mib": local_payload_mib,
        },
        "cases": case_results,
        "interpretation_limits": [
            (
                "This is a warm-cache CPU replay over temporary synthetic "
                "safetensors payloads and exact planner-generated slices."
            ),
            (
                "It does not measure real checkpoint loading, cold-cache "
                "or disk throughput, RSS, allocator, or page-cache peak."
            ),
            (
                "It does not establish GPU, inference, KV-cache, "
                "compression, memory, accuracy, or quality benefits."
            ),
        ],
    }


def _atomic_write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(file_descriptor, "w", encoding="utf-8") as handle:
            json.dump(
                payload,
                handle,
                indent=2,
                ensure_ascii=False,
                allow_nan=False,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


def _parse_tile_mib(value):
    try:
        parsed = tuple(int(item) for item in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "tile MiB values must be comma-separated integers"
        ) from error
    if not parsed or any(item <= 0 for item in parsed):
        raise argparse.ArgumentTypeError(
            "tile MiB values must be positive"
        )
    return parsed


def _parse_arguments():
    parser = argparse.ArgumentParser(
        description=(
            "Replay all five Qwen3.5 checkpoint tile grammars over "
            "temporary synthetic CPU safetensors payloads."
        )
    )
    parser.add_argument(
        "--tile-mib",
        type=_parse_tile_mib,
        default=(4, 8, 16, 32),
    )
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--local-payload-mib", type=int, default=32)
    parser.add_argument("--output-json", type=Path)
    return parser.parse_args()


def main():
    arguments = _parse_arguments()
    result = run_qwen35_five_binding_tile_replay_calibration(
        tile_mib=arguments.tile_mib,
        repeats=arguments.repeats,
        local_payload_mib=arguments.local_payload_mib,
    )
    if arguments.output_json is not None:
        _atomic_write_json(arguments.output_json, result)
    print(
        json.dumps(
            result,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
