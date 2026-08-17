from __future__ import annotations

from dataclasses import dataclass

from tinyvllm.engine.qwen35_recurrent_int8_codec import (
    QWEN35_RECURRENT_INT8_CODEC,
)


QWEN35_HYBRID_PREFIX_EXACT = "exact_restore"
QWEN35_HYBRID_PREFIX_RECURRENT_INT8 = "recurrent_int8_per_row"
QWEN35_HYBRID_PREFIX_DEFAULT = QWEN35_HYBRID_PREFIX_EXACT
QWEN35_HYBRID_PREFIX_REPRESENTATIONS = (
    QWEN35_HYBRID_PREFIX_EXACT,
    QWEN35_HYBRID_PREFIX_RECURRENT_INT8,
)
QWEN35_HYBRID_PREFIX_EXACT_VERSION = "qwen35_hybrid_prefix_exact_v1"
QWEN35_HYBRID_PREFIX_INT8_VERSION = (
    "qwen35_hybrid_prefix_recurrent_int8_v1"
)


@dataclass(frozen=True)
class Qwen35HybridPrefixRepresentation:
    name: str
    version: str
    codec: str | None


def resolve_qwen35_hybrid_prefix_representation(value):
    if value == QWEN35_HYBRID_PREFIX_EXACT:
        return Qwen35HybridPrefixRepresentation(
            value,
            QWEN35_HYBRID_PREFIX_EXACT_VERSION,
            None,
        )
    if value == QWEN35_HYBRID_PREFIX_RECURRENT_INT8:
        return Qwen35HybridPrefixRepresentation(
            value,
            QWEN35_HYBRID_PREFIX_INT8_VERSION,
            QWEN35_RECURRENT_INT8_CODEC,
        )
    raise ValueError(
        f"unsupported Qwen3.5 hybrid prefix representation: {value}"
    )
