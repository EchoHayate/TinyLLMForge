from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REPORT_PATH = ROOT / "experiments" / "light_doc_cache" / "make_multi_target_read_path_report.py"
SPEC = importlib.util.spec_from_file_location("make_multi_target_read_path_report", REPORT_PATH)
assert SPEC is not None and SPEC.loader is not None
REPORT = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = REPORT
SPEC.loader.exec_module(REPORT)


def _target(target_id: str, category: str, bucket: str) -> dict[str, str]:
    return {
        "id": target_id,
        "category": category,
        "length_bucket": bucket,
        "prompt": f"Prompt for {target_id} with enough non-empty text.",
    }


def _valid_payload() -> dict[str, object]:
    return {
        "version": 1,
        "targets": [
            _target("short_fact", "short_factual", "short"),
            _target("structured", "structured_text", "short"),
            _target("code", "source_code", "medium"),
            _target("math", "mathematical_reasoning", "medium"),
            _target("ood", "out_of_distribution", "medium"),
            _target("document_qa", "long_document_qa", "long"),
            _target("repetitive", "repetitive_text", "long"),
            _target("cross_paragraph", "cross_paragraph_dependency", "long"),
        ],
    }


def test_validate_target_dataset_accepts_required_matrix() -> None:
    targets = REPORT.validate_target_dataset(_valid_payload())
    assert [target["id"] for target in targets] == [
        "short_fact",
        "structured",
        "code",
        "math",
        "ood",
        "document_qa",
        "repetitive",
        "cross_paragraph",
    ]


def test_validate_target_dataset_rejects_duplicate_ids() -> None:
    payload = _valid_payload()
    payload["targets"][1]["id"] = "short_fact"
    try:
        REPORT.validate_target_dataset(payload)
    except ValueError as exc:
        assert "duplicate target id" in str(exc)
    else:
        raise AssertionError("duplicate target IDs must fail")


def test_validate_target_dataset_requires_bucket_coverage() -> None:
    payload = _valid_payload()
    for target in payload["targets"]:
        if target["length_bucket"] == "long":
            target["length_bucket"] = "medium"
    try:
        REPORT.validate_target_dataset(payload)
    except ValueError as exc:
        assert "at least two targets in each length bucket" in str(exc)
    else:
        raise AssertionError("missing long targets must fail")


def test_repository_target_dataset_is_valid() -> None:
    payload = REPORT.load_target_dataset(
        ROOT / "experiments" / "light_doc_cache" / "read_path_multi_target_prompts_v1.json"
    )
    assert payload["version"] == 1
    assert len(payload["targets"]) == 8


def _row(
    target_index: int,
    mode: str,
    mean_diff: float,
    *,
    argmax_match: bool = True,
    status: str = "success",
) -> dict[str, object]:
    return {
        "target_id": f"target_{target_index}",
        "category": "short_factual",
        "length_bucket": "short",
        "mode": mode,
        "role": "trained" if mode == "calibration_holdout" else "baseline",
        "status": status,
        "error": "",
        "prompt_tokens": 20 + target_index,
        "calibration_bank_sha256": "a" * 64 if mode == "calibration_holdout" else "",
        "logical_byte_saving_fraction": 0.1763,
        "missing_tokens": 100 + target_index,
        "missing_mse": 10.0 + target_index,
        "missing_mae": 2.0,
        "missing_max_abs": 20.0,
        "max_abs_logit_diff": mean_diff * 4,
        "mean_abs_logit_diff": mean_diff,
        "argmax_match": argmax_match,
        "original_argmax": 100,
        "restored_argmax": 100 if argmax_match else 101,
        "artifact": f"targets/target_{target_index}/{mode}",
    }


def test_nearest_rank_percentile_is_deterministic() -> None:
    assert REPORT.nearest_rank_percentile([1.0, 2.0, 3.0, 4.0, 5.0], 0.90) == 5.0
    assert REPORT.nearest_rank_percentile([5.0, 1.0, 3.0, 2.0], 0.50) == 2.0


def test_gate_passes_only_when_every_condition_holds() -> None:
    rows = []
    correlated = [1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00]
    holdout = [0.80, 0.85, 0.90, 0.90, 0.95, 1.00, 1.05, 1.10]
    for index in range(8):
        rows.append(_row(index, "correlated_same_layer_target", correlated[index]))
        rows.append(_row(index, "calibration_holdout", holdout[index]))
        rows.append(_row(index, "repeat_last_target", 1.20))
    gate = REPORT.evaluate_gate(rows)
    assert gate["decision"] == "GO"
    assert gate["paired_targets"] == 8
    assert gate["holdout_win_count"] == 5
    assert gate["holdout_win_rate"] == 0.625


def test_gate_fails_on_missing_pair_or_argmax_regression() -> None:
    rows = []
    for index in range(8):
        rows.append(_row(index, "correlated_same_layer_target", 1.0))
        if index != 7:
            rows.append(
                _row(
                    index,
                    "calibration_holdout",
                    0.8,
                    argmax_match=index != 6,
                )
            )
    gate = REPORT.evaluate_gate(rows)
    assert gate["decision"] == "NO_GO"
    assert "all eight paired targets completed" in gate["failed_conditions"]
    assert "no correlated argmax match regressed" in gate["failed_conditions"]


def test_write_outputs_keeps_per_target_setup_fields(tmp_path: Path) -> None:
    rows = [
        _row(0, "correlated_same_layer_target", 1.0),
        _row(1, "correlated_same_layer_target", 0.8),
    ]
    summary = REPORT.aggregate_rows(rows)
    REPORT.write_outputs(tmp_path, rows, summary)
    csv_path = tmp_path / "multi_target_rows.csv"
    csv_text = csv_path.read_text(encoding="utf-8")
    assert b"\r\n" not in csv_path.read_bytes()
    assert "target_0" in csv_text
    assert "target_1" in csv_text
    assert ",20," in csv_text
    assert ",21," in csv_text
    assert (tmp_path / "multi_target_summary.json").exists()
    assert "# Light Doc Cache Multi-Target Gate" in (
        tmp_path / "multi_target_report.md"
    ).read_text(encoding="utf-8")


def test_run_target_matrix_attempts_all_modes_and_records_failures(
    tmp_path: Path,
) -> None:
    driver_path = (
        ROOT
        / "experiments"
        / "light_doc_cache"
        / "run_tinyllm_read_path_multi_target.py"
    )
    spec = importlib.util.spec_from_file_location(
        "run_tinyllm_read_path_multi_target",
        driver_path,
    )
    assert spec is not None and spec.loader is not None
    driver = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = driver
    spec.loader.exec_module(driver)
    targets = _valid_payload()["targets"][:2]
    calls = []

    def fake_run_mode(*, target, mode, output_dir):
        calls.append((target["id"], mode))
        if target["id"] == "structured" and mode == "calibration_holdout":
            raise RuntimeError("synthetic target failure")
        return _row(
            0 if target["id"] == "short_fact" else 1,
            mode,
            0.5,
        )

    rows, manifest = driver.run_target_matrix(
        targets=targets,
        output_dir=tmp_path,
        calibration_bank_sha256="b" * 64,
        run_mode=fake_run_mode,
    )
    assert len(calls) == 6
    assert len(rows) == 6
    failed = [
        row
        for row in rows
        if row["target_id"] == "structured"
        and row["mode"] == "calibration_holdout"
    ][0]
    assert failed["status"] == "failed"
    assert failed["error"] == "RuntimeError: synthetic target failure"
    assert manifest["calibration_bank_sha256"] == "b" * 64


def test_sha256_file_is_stable(tmp_path: Path) -> None:
    path = tmp_path / "bank.json"
    path.write_text('{"kind":"test"}\n', encoding="utf-8")
    assert len(REPORT.hashlib_sha256_file(path)) == 64


def test_multi_target_remote_runner_has_required_safety_contract() -> None:
    script = (
        ROOT
        / "experiments"
        / "light_doc_cache"
        / "run_tinyllm_read_path_multi_target_remote.sh"
    )
    text = script.read_text(encoding="utf-8")
    for needle in [
        "set -euo pipefail",
        "sitian@10.232.195.203",
        "CONTROL_PATH",
        "rsync",
        "--relative",
        "nvidia-smi --query-gpu=index,memory.used",
        "TINYVLLM_DIST_PORT",
        "MASTER_PORT",
        "TARGET_LIMIT",
        "multi_target_summary.json",
    ]:
        assert needle in text


def test_read_path_pack_and_scatter_follow_sequence_block_table() -> None:
    read_path_script = (
        ROOT
        / "experiments"
        / "light_doc_cache"
        / "run_tinyllm_sidecar_read_path_smoke.py"
    )
    spec = importlib.util.spec_from_file_location(
        "run_tinyllm_sidecar_read_path_smoke_for_multi_target_test",
        read_path_script,
    )
    assert spec is not None and spec.loader is not None
    read_path = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = read_path
    spec.loader.exec_module(read_path)

    full = np.arange(2 * 1 * 5 * 2 * 1 * 1).reshape(2, 1, 5, 2, 1, 1)
    packed = read_path.pack_sequence_kv_blocks(full, [3, 1])
    assert np.array_equal(packed[:, :, 0], full[:, :, 3])
    assert np.array_equal(packed[:, :, 1], full[:, :, 1])

    restored_packed = packed + 1000
    scattered = read_path.scatter_sequence_kv_blocks(full, restored_packed, [3, 1])
    assert np.array_equal(scattered[:, :, 3], restored_packed[:, :, 0])
    assert np.array_equal(scattered[:, :, 1], restored_packed[:, :, 1])
    assert np.array_equal(scattered[:, :, 0], full[:, :, 0])
