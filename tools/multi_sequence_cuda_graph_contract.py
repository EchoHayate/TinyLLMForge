"""Frozen contracts for multi-sequence CUDA Graph diagnostics and gating."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


DIAGNOSTIC_BATCH_SIZES = (2, 3, 4, 5, 8, 9, 16)
DIAGNOSTIC_TRAJECTORIES = (
    "uniform-short",
    "ragged-context",
    "duplicate-and-distinct",
)
DIAGNOSTIC_MODES = ("eager", "exact_graph", "rounded_graph")
ROUNDED_GRAPH_SIZE = {2: 4, 3: 4, 4: 8, 5: 8, 8: 16, 9: 16, 16: 32}
DIAGNOSTIC_REPETITIONS = 3
WARMUP_STEPS = 2
MEASURED_STEPS = 16
LOGIT_RTOL = 1e-3
LOGIT_ATOL = 1e-2

PRODUCTION_THRESHOLDS = {
    "aggregate_decode_ratio": 1.15,
    "stable_decode_ratio": 1.25,
    "minimum_request_ratio": 0.95,
    "maximum_p95_itl_ratio": 1.05,
    "maximum_p99_itl_ratio": 1.10,
    "peak_reserved_ratio": 1.02,
    "initialization_ratio": 1.05,
    "stable_graph_hit_rate": 0.60,
}

_LOWER_BOUND_THRESHOLDS = frozenset(
    {
        "aggregate_decode_ratio",
        "stable_decode_ratio",
        "minimum_request_ratio",
        "stable_graph_hit_rate",
    }
)
_UPPER_BOUND_THRESHOLDS = frozenset(PRODUCTION_THRESHOLDS) - (
    _LOWER_BOUND_THRESHOLDS
)


@dataclass(frozen=True)
class DiagnosticCase:
    batch_size: int
    trajectory: str
    mode: str
    repetition: int
    graph_size: int

    @property
    def case_id(self) -> str:
        return (
            f"b{self.batch_size}__{self.trajectory}__"
            f"{self.mode}__r{self.repetition}"
        )


def diagnostic_graph_size(batch_size: int, mode: str) -> int:
    if batch_size not in DIAGNOSTIC_BATCH_SIZES:
        raise ValueError(f"unsupported batch size: {batch_size}")
    if mode not in DIAGNOSTIC_MODES:
        raise ValueError(f"unsupported mode: {mode}")
    if mode == "rounded_graph":
        return ROUNDED_GRAPH_SIZE[batch_size]
    return batch_size


def build_diagnostic_matrix() -> tuple[DiagnosticCase, ...]:
    return tuple(
        DiagnosticCase(
            batch_size=batch_size,
            trajectory=trajectory,
            mode=mode,
            repetition=repetition,
            graph_size=diagnostic_graph_size(batch_size, mode),
        )
        for repetition in range(DIAGNOSTIC_REPETITIONS)
        for trajectory in DIAGNOSTIC_TRAJECTORIES
        for batch_size in DIAGNOSTIC_BATCH_SIZES
        for mode in DIAGNOSTIC_MODES
    )


def canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def canonical_json_sha256(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as artifact:
        for chunk in iter(lambda: artifact.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_metadata(tensor) -> dict:
    import torch

    detached = tensor.detach().cpu().contiguous()
    finite = bool(torch.isfinite(detached).all().item())
    byte_view = detached.view(torch.uint8)
    return {
        "dtype": str(detached.dtype),
        "shape": list(detached.shape),
        "finite": finite,
        "sha256": hashlib.sha256(byte_view.numpy().tobytes()).hexdigest(),
    }


def compare_tensor_pair(
    eager,
    candidate,
    *,
    rtol: float = LOGIT_RTOL,
    atol: float = LOGIT_ATOL,
) -> dict:
    import torch

    eager_cpu = eager.detach().cpu().contiguous()
    candidate_cpu = candidate.detach().cpu().contiguous()
    shape_equal = eager_cpu.shape == candidate_cpu.shape
    dtype_equal = eager_cpu.dtype == candidate_cpu.dtype
    finite = bool(
        torch.isfinite(eager_cpu).all().item()
        and torch.isfinite(candidate_cpu).all().item()
    )
    argmax_equal = False
    close = False
    max_abs_error = None
    max_rel_error = None

    if shape_equal and eager_cpu.ndim > 0:
        argmax_equal = bool(
            torch.equal(
                torch.argmax(eager_cpu, dim=-1),
                torch.argmax(candidate_cpu, dim=-1),
            )
        )

    if shape_equal and dtype_equal and finite:
        eager_float = eager_cpu.float()
        candidate_float = candidate_cpu.float()
        absolute_error = (eager_float - candidate_float).abs()
        denominator = eager_float.abs().clamp_min(torch.finfo(torch.float32).tiny)
        max_abs_error = float(absolute_error.max().item())
        max_rel_error = float((absolute_error / denominator).max().item())
        try:
            torch.testing.assert_close(
                eager_cpu,
                candidate_cpu,
                rtol=rtol,
                atol=atol,
            )
        except AssertionError:
            close = False
        else:
            close = True

    return {
        "shape_equal": shape_equal,
        "dtype_equal": dtype_equal,
        "finite": finite,
        "argmax_equal": argmax_equal,
        "close": close,
        "max_abs_error": max_abs_error,
        "max_rel_error": max_rel_error,
        "rtol": rtol,
        "atol": atol,
    }


def _index_unique_rows(
    rows: Iterable[dict],
    *,
    evidence_name: str,
) -> tuple[dict[str, dict], list[str]]:
    indexed = {}
    failures = []
    for row in rows:
        case_id = row.get("case_id")
        if not isinstance(case_id, str):
            failures.append(f"{evidence_name}: missing case_id")
            continue
        if case_id in indexed:
            failures.append(f"{evidence_name}: duplicate case_id {case_id}")
            continue
        indexed[case_id] = row
    return indexed, failures


def _validate_case_row(
    row: dict,
    case: DiagnosticCase,
    *,
    evidence_name: str,
) -> list[str]:
    failures = []
    expected = {
        "case_id": case.case_id,
        "mode": case.mode,
        "batch_size": case.batch_size,
        "graph_size": case.graph_size,
    }
    if evidence_name == "matrix":
        expected.update(
            {
                "trajectory": case.trajectory,
                "repetition": case.repetition,
                "status": "PASS",
            }
        )
    for field, expected_value in expected.items():
        if row.get(field) != expected_value:
            failures.append(
                f"{evidence_name}: {case.case_id} {field}="
                f"{row.get(field)!r}, expected {expected_value!r}"
            )
    return failures


def _candidate_row_correct(
    logit_row: dict,
    layer_row: dict,
    kv_row: dict,
) -> bool:
    return all(
        (
            logit_row.get("finite") is True,
            logit_row.get("argmax_equal") is True,
            logit_row.get("close") is True,
            layer_row.get("finite") is True,
            layer_row.get("close") is True,
            isinstance(layer_row.get("required_layer_count"), int),
            layer_row.get("required_layer_count") > 0,
            layer_row.get("observed_layer_count")
            == layer_row.get("required_layer_count"),
            kv_row.get("active_slots_equal") is True,
            kv_row.get("unexpected_slot_mutations") == [],
        )
    )


def classify_diagnostic(
    *,
    matrix_rows: list[dict],
    logit_results: list[dict],
    layer_results: list[dict],
    kv_results: list[dict],
) -> dict:
    matrix = build_diagnostic_matrix()
    expected_matrix_ids = {case.case_id for case in matrix}
    candidate_cases = tuple(case for case in matrix if case.mode != "eager")
    expected_candidate_ids = {case.case_id for case in candidate_cases}

    matrix_by_id, failures = _index_unique_rows(
        matrix_rows,
        evidence_name="matrix",
    )
    logit_by_id, logit_failures = _index_unique_rows(
        logit_results,
        evidence_name="logits",
    )
    layer_by_id, layer_failures = _index_unique_rows(
        layer_results,
        evidence_name="layers",
    )
    kv_by_id, kv_failures = _index_unique_rows(
        kv_results,
        evidence_name="kv",
    )
    failures.extend(logit_failures)
    failures.extend(layer_failures)
    failures.extend(kv_failures)

    evidence_sets = (
        ("matrix", set(matrix_by_id), expected_matrix_ids),
        ("logits", set(logit_by_id), expected_candidate_ids),
        ("layers", set(layer_by_id), expected_candidate_ids),
        ("kv", set(kv_by_id), expected_candidate_ids),
    )
    for evidence_name, actual_ids, expected_ids in evidence_sets:
        missing = sorted(expected_ids - actual_ids)
        unexpected = sorted(actual_ids - expected_ids)
        if missing:
            failures.append(f"{evidence_name}: missing {missing}")
        if unexpected:
            failures.append(f"{evidence_name}: unexpected {unexpected}")

    for case in matrix:
        row = matrix_by_id.get(case.case_id)
        if row is not None:
            failures.extend(
                _validate_case_row(row, case, evidence_name="matrix")
            )
    for case in candidate_cases:
        for evidence_name, indexed in (
            ("logits", logit_by_id),
            ("layers", layer_by_id),
            ("kv", kv_by_id),
        ):
            row = indexed.get(case.case_id)
            if row is not None:
                failures.extend(
                    _validate_case_row(
                        row,
                        case,
                        evidence_name=evidence_name,
                    )
                )

    if failures:
        return {
            "classification": "INCOMPLETE",
            "rounded_classification": "INCOMPLETE",
            "failures": failures,
        }

    mode_correctness = {"exact_graph": True, "rounded_graph": True}
    corrupt_case_ids = {"exact_graph": [], "rounded_graph": []}
    for case in candidate_cases:
        correct = _candidate_row_correct(
            logit_by_id[case.case_id],
            layer_by_id[case.case_id],
            kv_by_id[case.case_id],
        )
        if not correct:
            mode_correctness[case.mode] = False
            corrupt_case_ids[case.mode].append(case.case_id)

    return {
        "classification": (
            "EXACT_REPLAY_CORRECT"
            if mode_correctness["exact_graph"]
            else "EXACT_REPLAY_CORRUPT"
        ),
        "rounded_classification": (
            "ROUNDED_REPLAY_CORRECT"
            if mode_correctness["rounded_graph"]
            else "ROUNDED_REPLAY_CORRUPT"
        ),
        "failures": [],
        "corrupt_exact_case_ids": corrupt_case_ids["exact_graph"],
        "corrupt_rounded_case_ids": corrupt_case_ids["rounded_graph"],
    }


def classify_production_gate(case_rows: list[dict]) -> dict:
    failures = []
    if not case_rows:
        failures.append("no production case rows")

    for row_index, row in enumerate(case_rows):
        prefix = f"row {row_index}"
        structural_failures = row.get("structural_failures")
        correctness_failures = row.get("correctness_failures")
        if structural_failures != []:
            failures.append(f"{prefix}: structural failures")
        if correctness_failures != []:
            failures.append(f"{prefix}: correctness failures")
        if row.get("measured_repetitions_complete") is not True:
            failures.append(f"{prefix}: measured repetitions incomplete")

        for field, threshold in PRODUCTION_THRESHOLDS.items():
            value = row.get(field)
            if not isinstance(value, (int, float)):
                failures.append(f"{prefix}: missing numeric {field}")
                continue
            if field in _LOWER_BOUND_THRESHOLDS and value < threshold:
                failures.append(
                    f"{prefix}: {field}={value} below {threshold}"
                )
            if field in _UPPER_BOUND_THRESHOLDS and value > threshold:
                failures.append(
                    f"{prefix}: {field}={value} above {threshold}"
                )

    return {
        "classification": "GO" if not failures else "NO_GO",
        "failures": failures,
        "thresholds": dict(PRODUCTION_THRESHOLDS),
    }
