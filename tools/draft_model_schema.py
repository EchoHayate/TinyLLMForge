"""Profiler-only draft model input/output schema helpers."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class DraftModelStubConfig:
    seed: int = 23
    stub_version: int = 1


@dataclass
class DraftModelInput:
    hidden_rows: list[list[float]]
    candidate_token_ids: list[int]
    top_k: int
    source_shape: list[int] | None = None
    source_dtype: str | None = None
    source_device: str | None = None

    @classmethod
    def from_rows(cls, hidden_rows, candidate_token_ids, top_k: int = 3,
                  source_shape=None, source_dtype=None, source_device=None) -> "DraftModelInput":
        return cls(
            hidden_rows=[[float(value) for value in row] for row in hidden_rows or []],
            candidate_token_ids=[int(token_id) for token_id in candidate_token_ids],
            top_k=max(1, int(top_k)),
            source_shape=[int(dim) for dim in source_shape] if source_shape is not None else None,
            source_dtype=str(source_dtype) if source_dtype is not None else None,
            source_device=str(source_device) if source_device is not None else None,
        )

    def to_dict(self) -> dict:
        return {
            "hidden_rows": self.hidden_rows,
            "candidate_token_ids": self.candidate_token_ids,
            "top_k": self.top_k,
            "source_shape": self.source_shape,
            "source_dtype": self.source_dtype,
            "source_device": self.source_device,
        }

    def schema(self) -> dict:
        hidden_dim = len(self.hidden_rows[0]) if self.hidden_rows else 0
        return {
            "hidden_rows": len(self.hidden_rows),
            "hidden_dim": hidden_dim,
            "candidate_count": len(self.candidate_token_ids),
            "top_k": self.top_k,
            "source_shape": self.source_shape,
            "source_dtype": self.source_dtype,
            "source_device": self.source_device,
        }


@dataclass
class DraftModelResult:
    candidate_token_ids: list[list[int]]
    candidate_logits: list[list[float]]
    draft_token_ids: list[int]
    draft_scores: list[float]
    preview: list[dict]
    metadata: dict
    timing_ms: dict

    def to_dict(self) -> dict:
        return {
            "candidate_token_ids": self.candidate_token_ids,
            "candidate_logits": self.candidate_logits,
            "draft_token_ids": self.draft_token_ids,
            "draft_scores": self.draft_scores,
            "preview": self.preview,
            "metadata": self.metadata,
            "timing_ms": self.timing_ms,
        }
