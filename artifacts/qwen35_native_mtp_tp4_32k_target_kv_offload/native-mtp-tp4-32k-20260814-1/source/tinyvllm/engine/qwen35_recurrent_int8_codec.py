from __future__ import annotations

from dataclasses import dataclass
import math

import torch


QWEN35_RECURRENT_INT8_CODEC = (
    "qwen35_recurrent_symmetric_int8_per_row_v1"
)


@dataclass(frozen=True)
class Qwen35EncodedRecurrentInt8:
    codec: str
    values: torch.Tensor
    scales: torch.Tensor
    source_shape: tuple[int, int, int]
    source_dtype: torch.dtype
    logical_bytes: int
    payload_bytes: int
    scale_bytes: int
    encoded_bytes: int


def _validate_source(recurrent):
    if not isinstance(recurrent, torch.Tensor):
        raise ValueError("recurrent must be a tensor")
    if recurrent.ndim != 3:
        raise ValueError("recurrent must be rank three")
    if any(dimension <= 0 for dimension in recurrent.shape):
        raise ValueError("recurrent dimensions must be positive")
    if recurrent.dtype != torch.float32:
        raise ValueError("recurrent must use torch.float32")
    if not torch.isfinite(recurrent).all().item():
        raise ValueError("recurrent must contain only finite values")


def encode_qwen35_recurrent_int8_per_row(
    recurrent: torch.Tensor,
) -> Qwen35EncodedRecurrentInt8:
    _validate_source(recurrent)
    source = recurrent.detach().clone().contiguous()
    amax = source.abs().amax(dim=-1)
    scales = torch.where(
        amax == 0,
        torch.ones_like(amax, dtype=torch.float32),
        amax / 127.0,
    ).contiguous()
    values = torch.round(source / scales.unsqueeze(-1))
    values = values.clamp(-127, 127).to(torch.int8).contiguous()
    if torch.any(values == -128).item():
        raise RuntimeError("encoded recurrent contains forbidden -128")
    logical_bytes = source.numel() * source.element_size()
    payload_bytes = values.untyped_storage().nbytes()
    scale_bytes = scales.untyped_storage().nbytes()
    return Qwen35EncodedRecurrentInt8(
        codec=QWEN35_RECURRENT_INT8_CODEC,
        values=values,
        scales=scales,
        source_shape=tuple(source.shape),
        source_dtype=source.dtype,
        logical_bytes=logical_bytes,
        payload_bytes=payload_bytes,
        scale_bytes=scale_bytes,
        encoded_bytes=payload_bytes + scale_bytes,
    )


def _validate_encoded(encoded):
    if type(encoded) is not Qwen35EncodedRecurrentInt8:
        raise ValueError(
            "encoded must be an exact Qwen35EncodedRecurrentInt8"
        )
    if encoded.codec != QWEN35_RECURRENT_INT8_CODEC:
        raise ValueError("encoded codec identity mismatch")
    if (
        not isinstance(encoded.source_shape, tuple)
        or len(encoded.source_shape) != 3
    ):
        raise ValueError("encoded source shape must be rank three")
    if any(
        isinstance(dimension, bool)
        or not isinstance(dimension, int)
        or dimension <= 0
        for dimension in encoded.source_shape
    ):
        raise ValueError(
            "encoded source shape dimensions must be positive integers"
        )
    if not isinstance(encoded.values, torch.Tensor):
        raise ValueError("encoded values must be a tensor")
    if not isinstance(encoded.scales, torch.Tensor):
        raise ValueError("encoded scales must be a tensor")
    if encoded.values.dtype != torch.int8:
        raise ValueError("encoded values must use torch.int8")
    if tuple(encoded.values.shape) != encoded.source_shape:
        raise ValueError("encoded values shape mismatch")
    if not encoded.values.is_contiguous():
        raise ValueError("encoded values must be contiguous")
    if encoded.scales.dtype != torch.float32:
        raise ValueError("encoded scales must use torch.float32")
    if tuple(encoded.scales.shape) != encoded.source_shape[:-1]:
        raise ValueError("encoded scales shape mismatch")
    if not encoded.scales.is_contiguous():
        raise ValueError("encoded scales must be contiguous")
    if encoded.values.device != encoded.scales.device:
        raise ValueError(
            "encoded values and scales must use the same device"
        )
    if encoded.source_dtype != torch.float32:
        raise ValueError("encoded source dtype must be torch.float32")
    if not torch.isfinite(encoded.scales).all().item():
        raise ValueError("encoded scales must be finite")
    if not torch.all(encoded.scales > 0).item():
        raise ValueError("encoded scales must be positive")
    if torch.any(encoded.values == -128).item():
        raise ValueError("encoded values contain forbidden -128")
    payload_bytes = encoded.values.untyped_storage().nbytes()
    scale_bytes = encoded.scales.untyped_storage().nbytes()
    if encoded.payload_bytes != payload_bytes:
        raise ValueError("encoded payload byte accounting mismatch")
    if encoded.scale_bytes != scale_bytes:
        raise ValueError("encoded scale byte accounting mismatch")
    if encoded.encoded_bytes != payload_bytes + scale_bytes:
        raise ValueError("encoded total byte accounting mismatch")
    logical_bytes = math.prod(encoded.source_shape) * 4
    if encoded.logical_bytes != logical_bytes:
        raise ValueError("encoded logical byte accounting mismatch")


def decode_qwen35_recurrent_int8_per_row(
    encoded: Qwen35EncodedRecurrentInt8,
    *,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    _validate_encoded(encoded)
    target = encoded.values.device if device is None else torch.device(device)
    values = encoded.values.to(device=target, dtype=torch.float32)
    scales = encoded.scales.to(device=target, dtype=torch.float32)
    decoded = (values * scales.unsqueeze(-1)).contiguous()
    if tuple(decoded.shape) != encoded.source_shape:
        raise RuntimeError("decoded recurrent shape mismatch")
    if not torch.isfinite(decoded).all().item():
        raise RuntimeError("decoded recurrent must be finite")
    return decoded


def qwen35_recurrent_int8_error_metrics(
    source: torch.Tensor,
    decoded: torch.Tensor,
) -> dict[str, int | float | bool]:
    _validate_source(source)
    _validate_source(decoded)
    if source.shape != decoded.shape:
        raise ValueError("source and decoded shapes must match")
    source64 = source.detach().to(device="cpu", dtype=torch.float64)
    decoded64 = decoded.detach().to(device="cpu", dtype=torch.float64)
    difference = decoded64 - source64
    absolute = difference.abs()
    source_norm = torch.linalg.vector_norm(source64)
    decoded_norm = torch.linalg.vector_norm(decoded64)
    difference_norm = torch.linalg.vector_norm(difference)
    if source_norm.item() == 0.0:
        if decoded_norm.item() != 0.0:
            raise ValueError(
                "decoded must be zero when source has zero source norm"
            )
        relative_l2_error = 0.0
        cosine_similarity = 1.0
    else:
        relative_l2_error = (
            difference_norm / source_norm
        ).item()
        if decoded_norm.item() == 0.0:
            cosine_similarity = 0.0
        else:
            cosine_similarity = torch.dot(
                source64.reshape(-1),
                decoded64.reshape(-1),
            ).item() / (
                source_norm.item() * decoded_norm.item()
            )
    return {
        "element_count": source.numel(),
        "finite_source": True,
        "finite_decoded": True,
        "max_abs_error": absolute.max().item(),
        "mean_abs_error": absolute.mean().item(),
        "rmse": torch.sqrt(
            torch.mean(difference.square())
        ).item(),
        "relative_l2_error": relative_l2_error,
        "cosine_similarity": max(
            -1.0,
            min(1.0, cosine_similarity),
        ),
    }
