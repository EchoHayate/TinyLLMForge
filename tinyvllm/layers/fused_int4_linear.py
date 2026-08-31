"""Model-neutral fused W4A16 linear primitive.

The candidate consumes the packed layout produced by ``quantize_int4``
directly.  It is intentionally not wired into ``LinearBase``: Stage 0 must
qualify the primitive on real draft-model shapes before runtime integration.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Iterable

import torch


_SUPPORTED_GROUP_SIZES = frozenset((32, 64, 128))
_SUPPORTED_M = frozenset((1, 2, 4, 8))
_K_ALIGNMENT = 64
_N_ALIGNMENT = 16


@dataclass(frozen=True)
class FusedInt4Support:
    supported: bool
    reason: str | None


def _device_identity(tensor: torch.Tensor) -> tuple[object, object]:
    device = tensor.device
    return device.type, device.index


def fused_int4_support(
    x: torch.Tensor,
    packed_weight: torch.Tensor,
    scales: torch.Tensor,
    *,
    group_size: int,
) -> FusedInt4Support:
    tensors = (x, packed_weight, scales)
    if any(getattr(tensor, "ndim", None) != 2 for tensor in tensors):
        return FusedInt4Support(False, "invalid_rank")

    devices = tuple(_device_identity(tensor) for tensor in tensors)
    if any(device_type != "cuda" for device_type, _ in devices):
        return FusedInt4Support(False, "not_cuda")
    if len(set(devices)) != 1:
        return FusedInt4Support(False, "not_cuda")

    if x.dtype not in (torch.float16, torch.bfloat16):
        return FusedInt4Support(
            False,
            "unsupported_activation_dtype",
        )
    if packed_weight.dtype != torch.uint8 or scales.dtype != torch.float32:
        return FusedInt4Support(
            False,
            "unsupported_activation_dtype",
        )
    if any(not tensor.is_contiguous() for tensor in tensors):
        return FusedInt4Support(False, "noncontiguous")

    if group_size not in _SUPPORTED_GROUP_SIZES:
        return FusedInt4Support(False, "unsupported_group_size")

    m, k = x.shape
    n, packed_k = packed_weight.shape
    if packed_k * 2 != k:
        return FusedInt4Support(False, "packed_shape_mismatch")
    if scales.shape != (n, k // group_size):
        return FusedInt4Support(False, "scale_shape_mismatch")
    if (
        m not in _SUPPORTED_M
        or k % _K_ALIGNMENT != 0
        or n % _N_ALIGNMENT != 0
    ):
        return FusedInt4Support(False, "unsupported_alignment")

    return FusedInt4Support(True, None)


def _validate_output(
    output: torch.Tensor,
    *,
    x: torch.Tensor,
    output_features: int,
) -> None:
    if (
        getattr(output, "ndim", None) != 2
        or tuple(output.shape) != (x.shape[0], output_features)
        or output.dtype != x.dtype
        or _device_identity(output) != _device_identity(x)
        or not output.is_contiguous()
    ):
        raise ValueError("output does not satisfy fused INT4 contract")


def _triton_modules():
    import triton
    import triton.language as tl

    return triton, tl


@lru_cache(maxsize=1)
def _compiled_kernel():
    triton_module, tl_module = _triton_modules()
    tl = tl_module

    @triton_module.jit
    def kernel(
        x_ptr,
        packed_weight_ptr,
        scales_ptr,
        output_ptr,
        M: tl.constexpr,
        N: tl.constexpr,
        K: tl.constexpr,
        GROUP_SIZE: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        program_m = tl.program_id(0)
        program_n = tl.program_id(1)
        offsets_m = program_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offsets_n = program_n * BLOCK_N + tl.arange(0, BLOCK_N)
        accumulator = tl.zeros(
            (BLOCK_M, BLOCK_N),
            dtype=tl.float32,
        )

        for block_index in range(0, tl.cdiv(K, BLOCK_K)):
            offsets_k = (
                block_index * BLOCK_K + tl.arange(0, BLOCK_K)
            )
            x_pointers = (
                x_ptr
                + offsets_m[:, None] * K
                + offsets_k[None, :]
            )
            x_tile = tl.load(
                x_pointers,
                mask=(
                    (offsets_m[:, None] < M)
                    & (offsets_k[None, :] < K)
                ),
                other=0.0,
            )

            packed_offsets_k = offsets_k // 2
            packed_pointers = (
                packed_weight_ptr
                + offsets_n[:, None] * (K // 2)
                + packed_offsets_k[None, :]
            )
            packed = tl.load(
                packed_pointers,
                mask=(
                    (offsets_n[:, None] < N)
                    & (offsets_k[None, :] < K)
                ),
                other=0,
            )
            low = packed & 0x0F
            high = (packed >> 4) & 0x0F
            quantized = tl.where(
                (offsets_k[None, :] & 1) == 0,
                low,
                high,
            ).to(tl.int8) - 8

            scale_pointers = (
                scales_ptr
                + offsets_n[:, None] * (K // GROUP_SIZE)
                + (offsets_k[None, :] // GROUP_SIZE)
            )
            scale = tl.load(
                scale_pointers,
                mask=(
                    (offsets_n[:, None] < N)
                    & (offsets_k[None, :] < K)
                ),
                other=0.0,
            )
            weight_tile = (
                quantized.to(tl.float32) * scale
            ).to(x_tile.dtype)
            accumulator += tl.dot(
                x_tile,
                tl.trans(weight_tile),
                out_dtype=tl.float32,
            )

        output_pointers = (
            output_ptr
            + offsets_m[:, None] * N
            + offsets_n[None, :]
        )
        tl.store(
            output_pointers,
            accumulator,
            mask=(
                (offsets_m[:, None] < M)
                & (offsets_n[None, :] < N)
            ),
        )

    return triton_module, kernel


def _launch_configuration(
    *,
    m: int,
    n: int,
    k: int,
) -> dict[str, int]:
    del k
    if m <= 2:
        return {
            "BLOCK_M": 2,
            "BLOCK_N": 64 if n % 64 == 0 else 32,
            "BLOCK_K": 64,
            "num_warps": 4,
            "num_stages": 3,
        }
    return {
        "BLOCK_M": 8,
        "BLOCK_N": 64 if n % 64 == 0 else 32,
        "BLOCK_K": 64,
        "num_warps": 4,
        "num_stages": 3,
    }


def _launch_triton(
    *,
    x: torch.Tensor,
    packed_weight: torch.Tensor,
    scales: torch.Tensor,
    group_size: int,
    output: torch.Tensor,
) -> None:
    triton_module, kernel = _compiled_kernel()
    m, k = x.shape
    n = packed_weight.shape[0]
    config = _launch_configuration(m=m, n=n, k=k)
    grid = (
        triton_module.cdiv(m, config["BLOCK_M"]),
        triton_module.cdiv(n, config["BLOCK_N"]),
    )
    kernel[grid](
        x,
        packed_weight,
        scales,
        output,
        M=m,
        N=n,
        K=k,
        GROUP_SIZE=group_size,
        BLOCK_M=config["BLOCK_M"],
        BLOCK_N=config["BLOCK_N"],
        BLOCK_K=config["BLOCK_K"],
        num_warps=config["num_warps"],
        num_stages=config["num_stages"],
    )


def fused_int4_linear(
    x: torch.Tensor,
    packed_weight: torch.Tensor,
    scales: torch.Tensor,
    *,
    group_size: int,
    output: torch.Tensor | None = None,
) -> torch.Tensor:
    support = fused_int4_support(
        x,
        packed_weight,
        scales,
        group_size=group_size,
    )
    if not support.supported:
        raise ValueError(
            f"fused INT4 input is unsupported: {support.reason}"
        )

    output_features = packed_weight.shape[0]
    if output is None:
        output = torch.empty(
            (x.shape[0], output_features),
            dtype=x.dtype,
            device=x.device,
        )
    else:
        _validate_output(
            output,
            x=x,
            output_features=output_features,
        )

    _launch_triton(
        x=x,
        packed_weight=packed_weight,
        scales=scales,
        group_size=group_size,
        output=output,
    )
    return output


def warmup_fused_int4_linear(
    cases: Iterable[
        tuple[
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            int,
            torch.Tensor | None,
        ]
    ],
) -> None:
    for x, packed_weight, scales, group_size, output in cases:
        fused_int4_linear(
            x,
            packed_weight,
            scales,
            group_size=group_size,
            output=output,
        )
