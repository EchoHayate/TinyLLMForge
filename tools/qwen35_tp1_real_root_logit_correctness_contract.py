"""Frozen contract for the Qwen3.5 TP1 real root-logit correctness gate."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Iterable, Mapping

import torch


TOKENIZER_VOCAB_SIZE = 248044
MODEL_VOCAB_SIZE = 248320
DECISION_TOPK = 20
ABS_DIFF_PERCENTILE_FIELDS = ("p50", "p95", "p99", "p99_9")
FINAL_CLASSIFICATIONS = ("PASS", "NO_GO_LOGIT")

_P17 = (
    237734, 105227, 220508, 88001, 203282, 70775, 186056, 53549,
    168830, 36323, 151604, 19097, 134378, 1871, 117152, 232433,
    99926,
)
_P65 = (
    72098, 187379, 54872, 170153, 37646, 152927, 20420, 135701,
    3194, 118475, 233756, 101249, 216530, 84023, 199304, 66797,
    182078, 49571, 164852, 32345, 147626, 15119, 130400, 245681,
    113174, 228455, 95948, 211229, 78722, 194003, 61496, 176777,
    44270, 159551, 27044, 142325, 9818, 125099, 240380, 107873,
    223154, 90647, 205928, 73421, 188702, 56195, 171476, 38969,
    154250, 21743, 137024, 4517, 119798, 235079, 102572, 217853,
    85346, 200627, 68120, 183401, 50894, 166175, 33668, 148949,
    16442,
)
_SYNTHETIC = (
    128, 129, 255, 256, 1024, 32768, 65536, 124022, 186033,
    247787, 248043,
)


def _token_sha256(token_ids: tuple[int, ...]) -> str:
    payload = json.dumps(
        list(token_ids),
        ensure_ascii=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class PromptCase:
    case_id: str
    token_ids: tuple[int, ...]
    token_sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.case_id, str) or not self.case_id:
            raise ValueError("case_id must be a non-empty string")
        if not self.token_ids:
            raise ValueError("token_ids must not be empty")
        if any(
            type(token_id) is not int
            or token_id <= 0
            or token_id >= TOKENIZER_VOCAB_SIZE
            for token_id in self.token_ids
        ):
            raise ValueError("token IDs must be positive tokenizer IDs")
        if self.token_sha256 != _token_sha256(self.token_ids):
            raise ValueError("token_sha256 does not match token_ids")


@dataclass(frozen=True)
class ComparisonTolerance:
    atol: float
    rtol: float

    def __post_init__(self) -> None:
        for name, value in (("atol", self.atol), ("rtol", self.rtol)):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or float(value) < 0.0
            ):
                raise ValueError(f"{name} must be finite and non-negative")


BF16_DECISION_TOLERANCE = ComparisonTolerance(atol=2e-5, rtol=0.0)


_PROMPT_CASES = (
    PromptCase(
        "p17",
        _P17,
        "be8a139b93467e0b0ed92999e8feec6de8fbaac4a2c4faf4786f798bb00cceb9",
    ),
    PromptCase(
        "p65",
        _P65,
        "2391c5bbc31e842e8c362e591458d05541b1566409f03672d192fe6a9702a264",
    ),
    PromptCase(
        "synthetic",
        _SYNTHETIC,
        "a36985347858070c7c917b110c793414192e691ffe160be66276b6022c940819",
    ),
)


def prompt_cases() -> tuple[PromptCase, ...]:
    return _PROMPT_CASES


def _canonical_row(value: torch.Tensor, *, label: str) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise ValueError(f"{label} logits must be a torch.Tensor")
    if value.ndim != 1 or value.numel() < 2:
        raise ValueError(f"{label} logits must be a one-dimensional row")
    row = value.detach().to(device="cpu", dtype=torch.float32).contiguous()
    if not bool(torch.isfinite(row).all()):
        raise ValueError(f"{label} logits must be finite")
    return row


def _float32_sha256(value: torch.Tensor) -> str:
    row = value.detach().to(device="cpu", dtype=torch.float32).contiguous()
    payload = row.numpy().tobytes(order="C")
    return hashlib.sha256(payload).hexdigest()


def _ranked_topk(
    value: torch.Tensor,
    *,
    topk: int,
) -> tuple[list[int], list[float]]:
    if (
        isinstance(topk, bool)
        or not isinstance(topk, int)
        or topk < 2
        or topk > value.numel()
    ):
        raise ValueError("topk must be between two and vocabulary width")
    ranked_tensor = torch.argsort(
        value,
        descending=True,
        stable=True,
    )[:topk]
    ranked = [int(token_id) for token_id in ranked_tensor.tolist()]
    return ranked, [float(item) for item in value[ranked_tensor].tolist()]


def compare_logits(
    native_logits: torch.Tensor,
    official_logits: torch.Tensor,
    *,
    tolerance: ComparisonTolerance,
    topk: int = DECISION_TOPK,
) -> dict[str, object]:
    native_source_dtype = str(native_logits.dtype).removeprefix("torch.")
    native = _canonical_row(native_logits, label="native")
    official = _canonical_row(official_logits, label="official")
    if native.shape != official.shape:
        raise ValueError(
            "native and official logit shape mismatch: "
            f"{tuple(native.shape)} != {tuple(official.shape)}"
        )

    native_ids, native_values = _ranked_topk(native, topk=topk)
    official_ids, official_values = _ranked_topk(official, topk=topk)
    absolute = (native - official).abs()
    threshold = (
        float(tolerance.atol)
        + float(tolerance.rtol) * official.abs()
    )
    scaled = absolute / threshold.clamp_min(
        torch.finfo(torch.float32).tiny
    )
    quantiles = torch.quantile(
        absolute,
        torch.tensor(
            [0.5, 0.95, 0.99, 0.999],
            dtype=torch.float32,
        ),
    )
    cosine = torch.nn.functional.cosine_similarity(
        native.reshape(1, -1),
        official.reshape(1, -1),
    ).clamp(min=-1.0, max=1.0)
    native_margin = native_values[0] - native_values[1]
    official_margin = official_values[0] - official_values[1]

    return {
        "shape": list(native.shape),
        "source_dtype": native_source_dtype,
        "comparison_dtype": "float32",
        "native_full_logit_sha256": _float32_sha256(native),
        "official_full_logit_sha256": _float32_sha256(official),
        "native_topk_token_ids": native_ids,
        "native_topk_logits": native_values,
        "official_topk_token_ids": official_ids,
        "official_topk_logits": official_values,
        "native_winner_token_id": native_ids[0],
        "native_runner_up_token_id": native_ids[1],
        "native_winner_logit": native_values[0],
        "native_runner_up_logit": native_values[1],
        "native_winner_margin": native_margin,
        "official_winner_token_id": official_ids[0],
        "official_runner_up_token_id": official_ids[1],
        "official_winner_logit": official_values[0],
        "official_runner_up_logit": official_values[1],
        "official_winner_margin": official_margin,
        "max_abs_diff": float(absolute.max().item()),
        "mean_abs_diff": float(absolute.mean().item()),
        "abs_diff_percentiles": {
            name: float(value)
            for name, value in zip(
                ABS_DIFF_PERCENTILE_FIELDS,
                quantiles.tolist(),
            )
        },
        "cosine_similarity": float(cosine.item()),
        "allclose_violation_count": int(
            (absolute > threshold).sum().item()
        ),
        "max_allclose_scaled_error": float(scaled.max().item()),
        "tolerance": {
            "atol": float(tolerance.atol),
            "rtol": float(tolerance.rtol),
        },
    }


def _row_decision_preserved(row: Mapping[str, object]) -> bool:
    native_winner = int(row["native_winner_token_id"])
    official_winner = int(row["official_winner_token_id"])
    native_runner_up = int(row["native_runner_up_token_id"])
    official_runner_up = int(row["official_runner_up_token_id"])
    native_topk = tuple(int(value) for value in row["native_topk_token_ids"])
    official_topk = tuple(
        int(value) for value in row["official_topk_token_ids"]
    )
    native_margin = float(row["native_winner_margin"])
    official_margin = float(row["official_winner_margin"])
    if native_winner != official_winner:
        return False
    if official_winner not in native_topk or native_winner not in official_topk:
        return False
    if official_margin > 0.0:
        return native_margin > 0.0
    if official_margin == 0.0:
        return (
            native_margin == 0.0
            and native_winner == official_winner
            and native_runner_up == official_runner_up
        )
    return False


def classify_rows(rows: Iterable[Mapping[str, object]]) -> str:
    values = tuple(rows)
    if not values:
        raise ValueError("at least one comparison row is required")
    return (
        "PASS"
        if all(
            _row_decision_preserved(row)
            and int(row["allclose_violation_count"]) == 0
            for row in values
        )
        else "NO_GO_LOGIT"
    )
