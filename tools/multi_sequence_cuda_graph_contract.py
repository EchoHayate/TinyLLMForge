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
AUTO_FLASH_ATTN_NUM_SPLITS = 0
HEURISTIC_POLICY_NAME = "fa2_263_heuristic_exact_width"
HEURISTIC_POLICY_CASE_ID = "fa2-263-exact-width"
SAME_POLICY_MODES = (
    "candidate_eager_heuristic",
    "exact_graph_heuristic",
    "rounded_graph_heuristic",
)
LEGACY_COMPATIBILITY_POLICIES = (
    "legacy_eager_auto",
    "candidate_eager_heuristic",
)
SPLIT_POLICIES = {
    "legacy_eager_auto": {
        "split_policy_name": "auto",
        "flash_attn_num_splits": AUTO_FLASH_ATTN_NUM_SPLITS,
    },
    "candidate_eager_heuristic": {
        "split_policy_name": HEURISTIC_POLICY_NAME,
        "flash_attn_num_splits": None,
    },
    "exact_graph_heuristic": {
        "split_policy_name": HEURISTIC_POLICY_NAME,
        "flash_attn_num_splits": None,
    },
    "rounded_graph_heuristic": {
        "split_policy_name": HEURISTIC_POLICY_NAME,
        "flash_attn_num_splits": None,
    },
}
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
    split_policy_name: str
    flash_attn_num_splits: int | None

    @property
    def case_id(self) -> str:
        return (
            f"b{self.batch_size}__{self.trajectory}__"
            f"{self.mode}__{_case_id_policy_name(self.split_policy_name)}"
            f"__r{self.repetition}"
        )


@dataclass(frozen=True)
class LegacyCompatibilityCase:
    batch_size: int
    trajectory: str
    policy: str
    repetition: int
    split_policy_name: str
    flash_attn_num_splits: int | None

    @property
    def pair_id(self) -> str:
        return (
            f"b{self.batch_size}__{self.trajectory}"
            f"__compat__r{self.repetition}"
        )

    @property
    def case_id(self) -> str:
        split_suffix = (
            ""
            if self.flash_attn_num_splits is None
            else f"-s{self.flash_attn_num_splits}"
        )
        return (
            f"{self.pair_id}__{self.policy}"
            f"__{_case_id_policy_name(self.split_policy_name)}"
            f"{split_suffix}"
        )


def _case_id_policy_name(split_policy_name: str) -> str:
    if split_policy_name == HEURISTIC_POLICY_NAME:
        return HEURISTIC_POLICY_CASE_ID
    return split_policy_name


def split_policy_for(execution_name: str) -> tuple[str, int | None]:
    try:
        policy = SPLIT_POLICIES[execution_name]
    except KeyError as exc:
        raise ValueError(
            f"unsupported split execution policy: {execution_name}"
        ) from exc
    return (
        str(policy["split_policy_name"]),
        (
            None
            if policy["flash_attn_num_splits"] is None
            else int(policy["flash_attn_num_splits"])
        ),
    )


def diagnostic_graph_size(batch_size: int, mode: str) -> int:
    if batch_size not in DIAGNOSTIC_BATCH_SIZES:
        raise ValueError(f"unsupported batch size: {batch_size}")
    if mode not in SAME_POLICY_MODES:
        raise ValueError(f"unsupported mode: {mode}")
    if mode == "rounded_graph_heuristic":
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
            split_policy_name=split_policy_for(mode)[0],
            flash_attn_num_splits=split_policy_for(mode)[1],
        )
        for repetition in range(DIAGNOSTIC_REPETITIONS)
        for trajectory in DIAGNOSTIC_TRAJECTORIES
        for batch_size in DIAGNOSTIC_BATCH_SIZES
        for mode in SAME_POLICY_MODES
    )


def build_legacy_compatibility_matrix(
) -> tuple[LegacyCompatibilityCase, ...]:
    return tuple(
        LegacyCompatibilityCase(
            batch_size=batch_size,
            trajectory=trajectory,
            policy=policy,
            repetition=repetition,
            split_policy_name=split_policy_for(policy)[0],
            flash_attn_num_splits=split_policy_for(policy)[1],
        )
        for repetition in range(DIAGNOSTIC_REPETITIONS)
        for trajectory in DIAGNOSTIC_TRAJECTORIES
        for batch_size in DIAGNOSTIC_BATCH_SIZES
        for policy in LEGACY_COMPATIBILITY_POLICIES
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
    candidate_cases = tuple(
        case
        for case in matrix
        if case.mode != "candidate_eager_heuristic"
    )
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

    mode_correctness = {
        "exact_graph_heuristic": True,
        "rounded_graph_heuristic": True,
    }
    corrupt_case_ids = {
        "exact_graph_heuristic": [],
        "rounded_graph_heuristic": [],
    }
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
            if mode_correctness["exact_graph_heuristic"]
            else "EXACT_REPLAY_CORRUPT"
        ),
        "rounded_classification": (
            "ROUNDED_REPLAY_CORRECT"
            if mode_correctness["rounded_graph_heuristic"]
            else "ROUNDED_REPLAY_CORRUPT"
        ),
        "failures": [],
        "corrupt_exact_case_ids": corrupt_case_ids[
            "exact_graph_heuristic"
        ],
        "corrupt_rounded_case_ids": corrupt_case_ids[
            "rounded_graph_heuristic"
        ],
    }


def _validate_legacy_process_row(
    row: dict,
    case: LegacyCompatibilityCase,
) -> list[str]:
    expected = {
        "case_id": case.case_id,
        "pair_id": case.pair_id,
        "batch_size": case.batch_size,
        "trajectory": case.trajectory,
        "policy": case.policy,
        "repetition": case.repetition,
        "split_policy_name": case.split_policy_name,
        "flash_attn_num_splits": case.flash_attn_num_splits,
        "comparison_policy_name": "legacy_auto_vs_heuristic",
        "status": "PASS",
    }
    return [
        (
            f"process: {case.case_id} {field}={row.get(field)!r}, "
            f"expected {expected_value!r}"
        )
        for field, expected_value in expected.items()
        if row.get(field) != expected_value
    ]


def _validate_legacy_pair_row(
    row: dict,
    case: LegacyCompatibilityCase,
    *,
    evidence_name: str,
) -> list[str]:
    expected = {
        "pair_id": case.pair_id,
        "batch_size": case.batch_size,
        "trajectory": case.trajectory,
        "repetition": case.repetition,
        "comparison_policy_name": "legacy_auto_vs_heuristic",
    }
    return [
        (
            f"{evidence_name}: {case.pair_id} {field}="
            f"{row.get(field)!r}, expected {expected_value!r}"
        )
        for field, expected_value in expected.items()
        if row.get(field) != expected_value
    ]


def _legacy_pair_correct(
    logit_row: dict,
    kv_row: dict,
    token_row: dict,
) -> bool:
    return all(
        (
            logit_row.get("finite") is True,
            logit_row.get("argmax_equal") is True,
            logit_row.get("close") is True,
            token_row.get("tokens_equal") is True,
            kv_row.get("touched_slot_sets_equal") is True,
            kv_row.get("unexpected_slot_mutations") == [],
        )
    )


def classify_legacy_compatibility(
    *,
    process_rows: list[dict],
    logit_results: list[dict],
    kv_results: list[dict],
    token_results: list[dict],
) -> dict:
    matrix = build_legacy_compatibility_matrix()
    expected_process_ids = {case.case_id for case in matrix}
    candidate_cases = tuple(
        case
        for case in matrix
        if case.policy == "candidate_eager_heuristic"
    )
    expected_pair_ids = {case.pair_id for case in candidate_cases}

    process_by_id, failures = _index_unique_rows(
        process_rows,
        evidence_name="process",
    )

    pair_indexes = {}
    for evidence_name, rows in (
        ("logits", logit_results),
        ("kv", kv_results),
        ("tokens", token_results),
    ):
        indexed = {}
        for row in rows:
            pair_id = row.get("pair_id")
            if not isinstance(pair_id, str):
                failures.append(f"{evidence_name}: missing pair_id")
                continue
            if pair_id in indexed:
                failures.append(
                    f"{evidence_name}: duplicate pair_id {pair_id}"
                )
                continue
            indexed[pair_id] = row
        pair_indexes[evidence_name] = indexed

    process_ids = set(process_by_id)
    missing_processes = sorted(expected_process_ids - process_ids)
    unexpected_processes = sorted(process_ids - expected_process_ids)
    if missing_processes:
        failures.append(f"process: missing {missing_processes}")
    if unexpected_processes:
        failures.append(f"process: unexpected {unexpected_processes}")

    for evidence_name, indexed in pair_indexes.items():
        actual_ids = set(indexed)
        missing = sorted(expected_pair_ids - actual_ids)
        unexpected = sorted(actual_ids - expected_pair_ids)
        if missing:
            failures.append(f"{evidence_name}: missing {missing}")
        if unexpected:
            failures.append(f"{evidence_name}: unexpected {unexpected}")

    for case in matrix:
        row = process_by_id.get(case.case_id)
        if row is not None:
            failures.extend(_validate_legacy_process_row(row, case))
    for case in candidate_cases:
        for evidence_name, indexed in pair_indexes.items():
            row = indexed.get(case.pair_id)
            if row is not None:
                failures.extend(
                    _validate_legacy_pair_row(
                        row,
                        case,
                        evidence_name=evidence_name,
                    )
                )

    if failures:
        return {
            "classification": "INCOMPLETE",
            "failures": failures,
            "incompatible_pair_ids": [],
        }

    incompatible_pair_ids = [
        case.pair_id
        for case in candidate_cases
        if not _legacy_pair_correct(
            pair_indexes["logits"][case.pair_id],
            pair_indexes["kv"][case.pair_id],
            pair_indexes["tokens"][case.pair_id],
        )
    ]
    return {
        "classification": (
            "LEGACY_COMPATIBLE"
            if not incompatible_pair_ids
            else "LEGACY_INCOMPATIBLE"
        ),
        "failures": [],
        "incompatible_pair_ids": incompatible_pair_ids,
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
