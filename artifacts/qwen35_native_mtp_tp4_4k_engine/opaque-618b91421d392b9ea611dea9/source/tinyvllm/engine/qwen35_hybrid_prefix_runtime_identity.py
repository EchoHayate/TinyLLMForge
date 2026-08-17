from __future__ import annotations

from dataclasses import dataclass

import torch

from tinyvllm.engine.qwen35_hybrid_model_owner import (
    Qwen35HybridModelOwner,
)


_SUPPORTED_DTYPES = {
    torch.float16: "float16",
    torch.bfloat16: "bfloat16",
    torch.float32: "float32",
}


def validate_qwen35_model_fingerprint(value):
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(
            character not in "0123456789abcdef"
            for character in value
        )
    ):
        raise ValueError(
            "model_fingerprint must be a lowercase SHA256"
        )
    return value


@dataclass(frozen=True)
class Qwen35HybridPrefixRuntimeIdentity:
    model_fingerprint: str
    layout_fingerprint: str
    dtype: torch.dtype

    def __post_init__(self):
        validate_qwen35_model_fingerprint(
            self.model_fingerprint
        )
        if (
            not isinstance(self.layout_fingerprint, str)
            or not self.layout_fingerprint
        ):
            raise ValueError(
                "layout_fingerprint must be a non-empty string"
            )
        if self.dtype not in _SUPPORTED_DTYPES:
            raise ValueError("runtime identity dtype is unsupported")

    def rank_row(self, participant_id):
        if (
            isinstance(participant_id, bool)
            or not isinstance(participant_id, int)
            or participant_id < 0
        ):
            raise ValueError(
                "participant_id must be a non-negative integer"
            )
        return {
            "participant_id": participant_id,
            "model_fingerprint": self.model_fingerprint,
            "layout_fingerprint": self.layout_fingerprint,
            "dtype": _SUPPORTED_DTYPES[self.dtype],
        }


def bind_qwen35_hybrid_prefix_runtime_identity(
    owner,
    model_fingerprint,
):
    if type(owner) is not Qwen35HybridModelOwner:
        raise ValueError(
            "owner must be a Qwen35HybridModelOwner"
        )
    model_fingerprint = validate_qwen35_model_fingerprint(
        model_fingerprint
    )
    dtypes = {
        component.dtype
        for component in owner.pool.layout.components
        if component.role == "linear_convolution"
    }
    if len(dtypes) != 1:
        raise ValueError(
            "hybrid prefix runtime identity requires one "
            "convolution dtype"
        )
    dtype = next(iter(dtypes))
    if dtype not in _SUPPORTED_DTYPES:
        raise ValueError("hybrid prefix runtime dtype is unsupported")
    return Qwen35HybridPrefixRuntimeIdentity(
        model_fingerprint=model_fingerprint,
        layout_fingerprint=owner.pool.layout.fingerprint,
        dtype=dtype,
    )
