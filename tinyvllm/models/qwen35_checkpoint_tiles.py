from __future__ import annotations

from dataclasses import dataclass

import torch

from tinyvllm.layers.linear import SegmentedColumnParallelLinear
from tinyvllm.models.qwen35_checkpoint_binding import (
    Qwen35CheckpointBindingPlan,
    Qwen35CheckpointTensorBinding,
)


@dataclass(frozen=True)
class Qwen35CheckpointTile:
    binding_index: int
    source_name: str
    shard: str
    source_tensor_shape: tuple[int, ...]
    source_slices: tuple[slice | int, ...]
    tile_shape: tuple[int, ...]
    destination: torch.Tensor
    destination_slices: tuple[slice, ...]
    destination_shape: tuple[int, ...]
    dtype: torch.dtype
    byte_count: int
    target: str
    kind: str


@dataclass(frozen=True)
class Qwen35CheckpointTilePlan:
    tiles: tuple[Qwen35CheckpointTile, ...]
    tensor_parallel_size: int
    tensor_parallel_rank: int
    binding_count: int
    source_count: int
    destination_bytes: int
    peak_tile_bytes: int


_DTYPES = {
    "BF16": (torch.bfloat16, 2),
    "F32": (torch.float32, 4),
}
_AXIS_ONE_SUFFIXES = (
    "linear_attention.out_proj.weight",
    "full_attention.output_projection.weight",
)
_REPLICATED_SUFFIXES = (
    "input_layernorm.weight",
    "post_attention_layernorm.weight",
    "full_attention.q_norm.weight",
    "full_attention.k_norm.weight",
    "linear_attention.norm_weight",
    "linear_attention.in_proj_b.weight",
    "linear_attention.in_proj_a.weight",
    "linear_attention.in_proj_qkv.weight",
    "linear_attention.in_proj_z.weight",
    "full_attention.q_projection.weight",
    "full_attention.k_projection.weight",
    "full_attention.v_projection.weight",
    "mlp.gate_up_proj.weight",
    "mlp.down_proj.weight",
)
_AXIS_ZERO_SUFFIXES = (
    "linear_attention.A_log",
    "linear_attention.dt_bias",
)


def _positive_budget(value) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value <= 0
    ):
        raise ValueError("max_tile_bytes must be a positive integer")
    return value


def _validate_tp_context(
    tensor_parallel_size,
    tensor_parallel_rank,
) -> tuple[int, int]:
    if (
        isinstance(tensor_parallel_size, bool)
        or not isinstance(tensor_parallel_size, int)
        or tensor_parallel_size <= 0
    ):
        raise ValueError(
            "tensor_parallel_size must be a positive integer"
        )
    if (
        isinstance(tensor_parallel_rank, bool)
        or not isinstance(tensor_parallel_rank, int)
        or tensor_parallel_rank < 0
        or tensor_parallel_rank >= tensor_parallel_size
    ):
        raise ValueError(
            "tensor_parallel_rank must be in "
            "[0, tensor_parallel_size)"
        )
    return tensor_parallel_size, tensor_parallel_rank


def _product(values: tuple[int, ...]) -> int:
    result = 1
    for value in values:
        result *= value
    return result


def _rows_per_tile(
    unit_bytes: int,
    max_tile_bytes: int,
    target: str,
) -> int:
    if unit_bytes <= 0:
        raise ValueError(f"{target} tile unit must be positive")
    if unit_bytes > max_tile_bytes:
        raise ValueError(
            f"{target} indivisible tile unit {unit_bytes} "
            "exceeds max_tile_bytes"
        )
    return max_tile_bytes // unit_bytes


def _range_tiles(length: int, rows_per_tile: int):
    start = 0
    while start < length:
        end = min(length, start + rows_per_tile)
        yield start, end
        start = end


def _destination_slices(
    shape: tuple[int, ...],
    row_start: int,
    row_end: int,
    *,
    row_offset: int = 0,
    column_start: int = 0,
    column_end: int | None = None,
) -> tuple[slice, ...]:
    if len(shape) == 1:
        return (slice(row_offset + row_start, row_offset + row_end),)
    if len(shape) != 2:
        raise ValueError("tiled checkpoint destination rank is unsupported")
    if column_end is None:
        column_end = shape[1]
    return (
        slice(row_offset + row_start, row_offset + row_end),
        slice(column_start, column_end),
    )


def _make_tile(
    *,
    binding_index: int,
    binding: Qwen35CheckpointTensorBinding,
    source_slices: tuple[slice | int, ...],
    tile_shape: tuple[int, ...],
    destination_slices: tuple[slice, ...],
    dtype: torch.dtype,
    byte_width: int,
    kind: str,
) -> Qwen35CheckpointTile:
    byte_count = _product(tile_shape) * byte_width
    return Qwen35CheckpointTile(
        binding_index=binding_index,
        source_name=binding.load.weight.source.name,
        shard=binding.load.weight.source.shard,
        source_tensor_shape=binding.load.metadata.shape,
        source_slices=source_slices,
        tile_shape=tile_shape,
        destination=binding.destination,
        destination_slices=destination_slices,
        destination_shape=tile_shape,
        dtype=dtype,
        byte_count=byte_count,
        target=binding.load.weight.target,
        kind=kind,
    )


def _classify_binding(
    binding: Qwen35CheckpointTensorBinding,
) -> str:
    target = binding.load.weight.target
    transform = binding.load.transform
    if transform not in ("identity", "squeeze_conv_channel"):
        raise ValueError(
            f"unsupported checkpoint transform: {transform}"
        )
    if (
        target.endswith("linear_attention.conv_weight")
        and transform == "squeeze_conv_channel"
        and binding.loader_kind == "direct_buffer_copy"
    ):
        return "squeeze_axis0"
    if transform != "identity":
        raise ValueError(
            f"unsupported tiled checkpoint binding: {target}"
        )
    if (
        target == "final_norm.weight"
        or target.endswith(_REPLICATED_SUFFIXES)
    ):
        if binding.loader_kind not in (
            "default_parameter_copy",
            "direct_buffer_copy",
            "custom_parameter_loader",
        ):
            raise ValueError(
                f"unsupported tiled checkpoint binding: {target}"
            )
        return "replicated"
    if target.endswith(_AXIS_ONE_SUFFIXES):
        if binding.loader_kind != "custom_parameter_loader":
            raise ValueError(
                f"unsupported tiled checkpoint binding: {target}"
            )
        return "axis1"
    if (
        (
            target in ("embed_tokens.weight", "lm_head.weight")
            or target.endswith(_AXIS_ZERO_SUFFIXES)
        )
        and binding.loader_kind in (
            "custom_parameter_loader",
            "direct_buffer_copy",
        )
    ):
        return "axis0"
    raise ValueError(
        f"unsupported tiled checkpoint binding: {target}"
    )


def _validate_destination_slice(
    binding: Qwen35CheckpointTensorBinding,
) -> tuple[int, int]:
    destination_rows = binding.destination.shape[0]
    if binding.destination_slice is None:
        if binding.local_shape[0] != destination_rows:
            raise ValueError(
                f"{binding.load.weight.target} destination shape is invalid"
            )
        return 0, destination_rows
    offset, length = binding.destination_slice
    if (
        isinstance(offset, bool)
        or isinstance(length, bool)
        or not isinstance(offset, int)
        or not isinstance(length, int)
        or offset < 0
        or length <= 0
        or offset + length > destination_rows
        or length != binding.local_shape[0]
    ):
        raise ValueError(
            f"{binding.load.weight.target} destination slice is invalid"
        )
    return offset, length


def _axis_zero_tiles(
    binding_index: int,
    binding: Qwen35CheckpointTensorBinding,
    *,
    tensor_parallel_size: int,
    tensor_parallel_rank: int,
    dtype: torch.dtype,
    byte_width: int,
    max_tile_bytes: int,
) -> tuple[Qwen35CheckpointTile, ...]:
    source_shape = binding.load.metadata.shape
    local_shape = binding.local_shape
    if len(source_shape) not in (1, 2) or len(local_shape) != len(
        source_shape
    ):
        raise ValueError(
            f"{binding.load.weight.target} axis-0 shape is invalid"
        )
    row_offset, local_rows = _validate_destination_slice(binding)
    if local_rows != local_shape[0]:
        raise ValueError(
            f"{binding.load.weight.target} local rows are invalid"
        )
    if source_shape[0] != local_rows * tensor_parallel_size:
        raise ValueError(
            f"{binding.load.weight.target} global rows must match "
            "TP-local rows"
        )
    trailing_shape = source_shape[1:]
    if local_shape[1:] != trailing_shape:
        raise ValueError(
            f"{binding.load.weight.target} axis-0 trailing shape is invalid"
        )
    unit_bytes = _product(trailing_shape) * byte_width
    rows_per_tile = _rows_per_tile(
        unit_bytes,
        max_tile_bytes,
        binding.load.weight.target,
    )
    global_start = tensor_parallel_rank * local_rows
    tiles = []
    for local_start, local_end in _range_tiles(
        local_rows,
        rows_per_tile,
    ):
        source_rows = slice(
            global_start + local_start,
            global_start + local_end,
        )
        if len(source_shape) == 1:
            source_slices = (source_rows,)
            tile_shape = (local_end - local_start,)
        else:
            source_slices = (
                source_rows,
                slice(0, source_shape[1]),
            )
            tile_shape = (
                local_end - local_start,
                source_shape[1],
            )
        tiles.append(_make_tile(
            binding_index=binding_index,
            binding=binding,
            source_slices=source_slices,
            tile_shape=tile_shape,
            destination_slices=_destination_slices(
                tuple(binding.destination.shape),
                local_start,
                local_end,
                row_offset=row_offset,
            ),
            dtype=dtype,
            byte_width=byte_width,
            kind="axis0",
        ))
    return tuple(tiles)


def _axis_one_tiles(
    binding_index: int,
    binding: Qwen35CheckpointTensorBinding,
    *,
    tensor_parallel_size: int,
    tensor_parallel_rank: int,
    dtype: torch.dtype,
    byte_width: int,
    max_tile_bytes: int,
) -> tuple[Qwen35CheckpointTile, ...]:
    source_shape = binding.load.metadata.shape
    local_shape = binding.local_shape
    if (
        len(source_shape) != 2
        or len(local_shape) != 2
        or source_shape[0] != local_shape[0]
    ):
        raise ValueError(
            f"{binding.load.weight.target} axis-1 shape is invalid"
        )
    if binding.destination_slice is not None:
        raise ValueError(
            f"{binding.load.weight.target} destination slice is invalid"
        )
    local_columns = local_shape[1]
    if source_shape[1] != local_columns * tensor_parallel_size:
        raise ValueError(
            f"{binding.load.weight.target} global columns must match "
            "TP-local columns"
        )
    source_column_start = tensor_parallel_rank * local_columns
    source_column_end = source_column_start + local_columns
    if source_column_end > source_shape[1]:
        raise ValueError(
            f"{binding.load.weight.target} axis-1 columns are invalid"
        )
    unit_bytes = local_columns * byte_width
    rows_per_tile = _rows_per_tile(
        unit_bytes,
        max_tile_bytes,
        binding.load.weight.target,
    )
    tiles = []
    for row_start, row_end in _range_tiles(
        source_shape[0],
        rows_per_tile,
    ):
        tile_shape = (row_end - row_start, local_columns)
        tiles.append(_make_tile(
            binding_index=binding_index,
            binding=binding,
            source_slices=(
                slice(row_start, row_end),
                slice(source_column_start, source_column_end),
            ),
            tile_shape=tile_shape,
            destination_slices=(
                slice(row_start, row_end),
                slice(0, local_columns),
            ),
            dtype=dtype,
            byte_width=byte_width,
            kind="axis1",
        ))
    return tuple(tiles)


def _segmented_tiles(
    binding_index: int,
    binding: Qwen35CheckpointTensorBinding,
    *,
    tensor_parallel_size: int,
    tensor_parallel_rank: int,
    dtype: torch.dtype,
    byte_width: int,
    max_tile_bytes: int,
) -> tuple[Qwen35CheckpointTile, ...]:
    loader = getattr(binding.destination, "weight_loader", None)
    owner = getattr(loader, "__self__", None)
    if type(owner) is not SegmentedColumnParallelLinear:
        raise ValueError(
            f"{binding.load.weight.target} segmented loader owner is invalid"
        )
    source_shape = binding.load.metadata.shape
    if (
        len(source_shape) != 2
        or source_shape[0] != sum(owner.output_sizes)
        or source_shape[1] != owner.input_size
        or tuple(binding.destination.shape)
        != (sum(owner.local_output_sizes), owner.input_size)
        or binding.destination_slice is not None
    ):
        raise ValueError(
            f"{binding.load.weight.target} segmented sizes are invalid"
        )
    if any(
        global_rows != local_rows * tensor_parallel_size
        for global_rows, local_rows in zip(
            owner.output_sizes,
            owner.local_output_sizes,
        )
    ):
        raise ValueError(
            f"{binding.load.weight.target} segmented global rows "
            "must match TP-local rows"
        )
    unit_bytes = source_shape[1] * byte_width
    rows_per_tile = _rows_per_tile(
        unit_bytes,
        max_tile_bytes,
        binding.load.weight.target,
    )
    tiles = []
    global_offset = 0
    local_offset = 0
    for global_rows, local_rows in zip(
        owner.output_sizes,
        owner.local_output_sizes,
    ):
        rank_start = global_offset + tensor_parallel_rank * local_rows
        for segment_start, segment_end in _range_tiles(
            local_rows,
            rows_per_tile,
        ):
            tile_shape = (
                segment_end - segment_start,
                source_shape[1],
            )
            tiles.append(_make_tile(
                binding_index=binding_index,
                binding=binding,
                source_slices=(
                    slice(
                        rank_start + segment_start,
                        rank_start + segment_end,
                    ),
                    slice(0, source_shape[1]),
                ),
                tile_shape=tile_shape,
                destination_slices=(
                    slice(
                        local_offset + segment_start,
                        local_offset + segment_end,
                    ),
                    slice(0, source_shape[1]),
                ),
                dtype=dtype,
                byte_width=byte_width,
                kind="segmented_axis0",
            ))
        global_offset += global_rows
        local_offset += local_rows
    return tuple(tiles)


def _convolution_tiles(
    binding_index: int,
    binding: Qwen35CheckpointTensorBinding,
    *,
    tensor_parallel_size: int,
    tensor_parallel_rank: int,
    dtype: torch.dtype,
    byte_width: int,
    max_tile_bytes: int,
) -> tuple[Qwen35CheckpointTile, ...]:
    source_shape = binding.load.metadata.shape
    local_shape = binding.local_shape
    if (
        len(source_shape) != 3
        or source_shape[1] != 1
        or len(local_shape) != 2
        or local_shape[1] != source_shape[2]
        or tuple(binding.destination.shape) != local_shape
        or binding.destination_slice is not None
    ):
        raise ValueError(
            f"{binding.load.weight.target} convolution shape is invalid"
        )
    source_segments = binding.source_segments
    if (
        not source_segments
        or any(
            isinstance(segment, bool)
            or not isinstance(segment, int)
            or segment <= 0
            or segment % tensor_parallel_size != 0
            for segment in source_segments
        )
        or sum(source_segments) != source_shape[0]
        or sum(
            segment // tensor_parallel_size
            for segment in source_segments
        ) != local_shape[0]
    ):
        raise ValueError(
            f"{binding.load.weight.target} channel segments are invalid"
        )
    unit_bytes = source_shape[2] * byte_width
    rows_per_tile = _rows_per_tile(
        unit_bytes,
        max_tile_bytes,
        binding.load.weight.target,
    )
    tiles = []
    global_offset = 0
    local_offset = 0
    for global_rows in source_segments:
        local_rows = global_rows // tensor_parallel_size
        rank_start = global_offset + tensor_parallel_rank * local_rows
        for segment_start, segment_end in _range_tiles(
            local_rows,
            rows_per_tile,
        ):
            tile_shape = (
                segment_end - segment_start,
                source_shape[2],
            )
            tiles.append(_make_tile(
                binding_index=binding_index,
                binding=binding,
                source_slices=(
                    slice(
                        rank_start + segment_start,
                        rank_start + segment_end,
                    ),
                    0,
                    slice(0, source_shape[2]),
                ),
                tile_shape=tile_shape,
                destination_slices=(
                    slice(
                        local_offset + segment_start,
                        local_offset + segment_end,
                    ),
                    slice(0, source_shape[2]),
                ),
                dtype=dtype,
                byte_width=byte_width,
                kind="squeeze_axis0",
            ))
        global_offset += global_rows
        local_offset += local_rows
    return tuple(tiles)


def _replicated_tiles(
    binding_index: int,
    binding: Qwen35CheckpointTensorBinding,
    *,
    dtype: torch.dtype,
    byte_width: int,
    max_tile_bytes: int,
) -> tuple[Qwen35CheckpointTile, ...]:
    source_shape = binding.load.metadata.shape
    row_offset, destination_rows = _validate_destination_slice(binding)
    if (
        len(source_shape) not in (1, 2)
        or tuple(binding.local_shape) != source_shape
        or destination_rows != source_shape[0]
        or tuple(binding.destination.shape[1:]) != source_shape[1:]
    ):
        raise ValueError(
            f"{binding.load.weight.target} replicated shape is invalid"
        )
    trailing_shape = source_shape[1:]
    unit_bytes = _product(trailing_shape) * byte_width
    rows_per_tile = _rows_per_tile(
        unit_bytes,
        max_tile_bytes,
        binding.load.weight.target,
    )
    tiles = []
    for start, end in _range_tiles(source_shape[0], rows_per_tile):
        tile_shape = (end - start, *trailing_shape)
        source_slices = (slice(start, end),) + tuple(
            slice(0, dimension)
            for dimension in trailing_shape
        )
        destination_slices = _destination_slices(
            tuple(binding.destination.shape),
            start,
            end,
            row_offset=row_offset,
        )
        tiles.append(_make_tile(
            binding_index=binding_index,
            binding=binding,
            source_slices=source_slices,
            tile_shape=tile_shape,
            destination_slices=destination_slices,
            dtype=dtype,
            byte_width=byte_width,
            kind="replicated",
        ))
    return tuple(tiles)


def _validate_source_contracts(
    bindings: tuple[Qwen35CheckpointTensorBinding, ...],
) -> int:
    contracts = {}
    for binding in bindings:
        source = binding.load.weight.source
        current = (source.shard, binding.load.metadata)
        existing = contracts.get(source.name)
        if existing is not None and existing != current:
            raise ValueError(
                f"conflicting checkpoint source contract: {source.name}"
            )
        contracts[source.name] = current
    return len(contracts)


def build_qwen35_checkpoint_tile_plan(
    binding_plan: Qwen35CheckpointBindingPlan,
    *,
    max_tile_bytes: int,
) -> Qwen35CheckpointTilePlan:
    if type(binding_plan) is not Qwen35CheckpointBindingPlan:
        raise ValueError(
            "binding_plan must be an exact Qwen35CheckpointBindingPlan"
        )
    budget = _positive_budget(max_tile_bytes)
    tensor_parallel_size, tensor_parallel_rank = (
        _validate_tp_context(
            binding_plan.tensor_parallel_size,
            binding_plan.tensor_parallel_rank,
        )
    )
    for binding in binding_plan.bindings:
        if type(binding) is not Qwen35CheckpointTensorBinding:
            raise ValueError(
                "binding plan entries must be exact "
                "Qwen35CheckpointTensorBinding values"
            )
        if (
            not isinstance(binding.destination, torch.Tensor)
            or binding.destination.device.type not in ("cpu", "meta")
        ):
            raise ValueError(
                f"{binding.load.weight.target} destination device is invalid"
            )
    source_count = _validate_source_contracts(binding_plan.bindings)

    tiles = []
    for binding_index, binding in enumerate(binding_plan.bindings):
        dtype_contract = _DTYPES.get(binding.load.metadata.dtype)
        if dtype_contract is None:
            raise ValueError(
                "unsupported tiled checkpoint dtype: "
                f"{binding.load.metadata.dtype}"
            )
        dtype, byte_width = dtype_contract
        allows_runtime_cast = (
            binding.load.weight.target.endswith(
                "linear_attention.norm_weight"
            )
            and dtype == torch.float32
            and binding.destination.dtype == torch.bfloat16
        )
        if binding.destination.dtype != dtype and not allows_runtime_cast:
            raise ValueError(
                f"{binding.load.weight.target} destination dtype is invalid"
            )
        kind = _classify_binding(binding)
        shared = {
            "binding_index": binding_index,
            "binding": binding,
            "dtype": dtype,
            "byte_width": byte_width,
            "max_tile_bytes": budget,
        }
        if kind == "axis0":
            binding_tiles = _axis_zero_tiles(
                **shared,
                tensor_parallel_size=tensor_parallel_size,
                tensor_parallel_rank=tensor_parallel_rank,
            )
        elif kind == "axis1":
            binding_tiles = _axis_one_tiles(
                **shared,
                tensor_parallel_size=tensor_parallel_size,
                tensor_parallel_rank=tensor_parallel_rank,
            )
        elif kind == "segmented_axis0":
            binding_tiles = _segmented_tiles(
                **shared,
                tensor_parallel_size=tensor_parallel_size,
                tensor_parallel_rank=tensor_parallel_rank,
            )
        elif kind == "squeeze_axis0":
            binding_tiles = _convolution_tiles(
                **shared,
                tensor_parallel_size=tensor_parallel_size,
                tensor_parallel_rank=tensor_parallel_rank,
            )
        else:
            binding_tiles = _replicated_tiles(**shared)
        if not binding_tiles:
            raise ValueError(
                f"{binding.load.weight.target} tile coverage is empty"
            )
        metadata_bytes = (
            binding.load.metadata.data_offsets[1]
            - binding.load.metadata.data_offsets[0]
        )
        expected_metadata_bytes = (
            _product(binding.load.metadata.shape) * byte_width
        )
        if metadata_bytes != expected_metadata_bytes:
            raise ValueError(
                f"{binding.load.weight.target} metadata byte count "
                "is invalid"
            )
        if any(tile.byte_count > budget for tile in binding_tiles):
            raise ValueError(
                f"{binding.load.weight.target} tile exceeds max_tile_bytes"
            )
        tiles.extend(binding_tiles)

    destination_bytes = sum(tile.byte_count for tile in tiles)
    peak_tile_bytes = max(
        (tile.byte_count for tile in tiles),
        default=0,
    )
    return Qwen35CheckpointTilePlan(
        tiles=tuple(tiles),
        tensor_parallel_size=tensor_parallel_size,
        tensor_parallel_rank=tensor_parallel_rank,
        binding_count=len(binding_plan.bindings),
        source_count=source_count,
        destination_bytes=destination_bytes,
        peak_tile_bytes=peak_tile_bytes,
    )
