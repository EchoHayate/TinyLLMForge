from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

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
