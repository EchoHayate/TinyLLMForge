"""Profiler-only draft model input/output schema helpers."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class DraftModelStubConfig:
    seed: int = 23
    stub_version: int = 1


@dataclass
class DraftModelContract:
    expected_hidden_dim: int | None = None
    target_vocab_size: int | None = None
    draft_vocab_size: int | None = None
    tokenizer_family: str | None = None
    draft_tokenizer_family: str | None = None


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


def validate_draft_model_contract(draft_input: DraftModelInput, contract: DraftModelContract | None = None) -> dict:
    contract = contract or DraftModelContract()
    actual_hidden_dim = len(draft_input.hidden_rows[0]) if draft_input.hidden_rows else 0
    candidate_id_min = min(draft_input.candidate_token_ids) if draft_input.candidate_token_ids else None
    candidate_id_max = max(draft_input.candidate_token_ids) if draft_input.candidate_token_ids else None

    metadata = {
        "expected_hidden_dim": contract.expected_hidden_dim,
        "actual_hidden_dim": actual_hidden_dim,
        "target_vocab_size": contract.target_vocab_size,
        "draft_vocab_size": contract.draft_vocab_size,
        "tokenizer_family": contract.tokenizer_family,
        "draft_tokenizer_family": contract.draft_tokenizer_family,
        "candidate_id_min": candidate_id_min,
        "candidate_id_max": candidate_id_max,
        "compatible": True,
    }

    if contract.expected_hidden_dim is not None and actual_hidden_dim != int(contract.expected_hidden_dim):
        raise ValueError(
            f"hidden_dim mismatch: expected {int(contract.expected_hidden_dim)}, got {actual_hidden_dim}"
        )
    if contract.target_vocab_size is not None and candidate_id_max is not None:
        if candidate_id_min < 0 or candidate_id_max >= int(contract.target_vocab_size):
            raise ValueError(
                "candidate token id out of target vocab: "
                f"min={candidate_id_min}, max={candidate_id_max}, target_vocab_size={int(contract.target_vocab_size)}"
            )
    if contract.draft_vocab_size is not None and candidate_id_max is not None:
        if candidate_id_min < 0 or candidate_id_max >= int(contract.draft_vocab_size):
            raise ValueError(
                "candidate token id out of draft vocab: "
                f"min={candidate_id_min}, max={candidate_id_max}, draft_vocab_size={int(contract.draft_vocab_size)}"
            )
    if contract.tokenizer_family and contract.draft_tokenizer_family:
        if str(contract.tokenizer_family) != str(contract.draft_tokenizer_family):
            raise ValueError(
                "tokenizer family mismatch: "
                f"target={contract.tokenizer_family}, draft={contract.draft_tokenizer_family}"
            )
    return metadata


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
