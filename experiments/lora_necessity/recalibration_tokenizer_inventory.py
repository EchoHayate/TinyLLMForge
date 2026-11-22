import copy
import hashlib

from pilot_protocol import FLAGS, fields, require
from recalibration_data_contract import FILE_PINS, _plain, load_dataset
from recalibration_rendering import EXAMPLES_SHA256, first_round, second_round
from recalibration_validation import build_report
from validation_contract import digest, encoded


RENDERING_PINS = {
    "recalibration_rendering.py": "ac8286fe3ceb567e23d3a04b32e69d5447d443000c707b3af334cdec94b4baf0",
    "prompt_rendering.py": "2016c3eb0255d1d4a40f6072c7407b099fa3e4e569345c8d1ac778d5bd7bb18f",
}
PLANNED_COUNTS = {"first": 24, "references": 48, "sampled_second": 48}


def _source_identity(rendering_raw):
    fields(rendering_raw, RENDERING_PINS)
    require(all(type(name) is str for name in rendering_raw), "rendering key type")
    for name, expected in RENDERING_PINS.items():
        raw = rendering_raw[name]
        require(type(raw) is bytes and len(raw) <= 1048576
                and hashlib.sha256(raw).hexdigest() == expected, "renderer identity")
    return dict(RENDERING_PINS)


def _text_sha(value):
    return None if value is None else hashlib.sha256(value.encode("utf-8")).hexdigest()


def _visible_feedback(report, task_id):
    rows = [row for row in report["rows"] if row["task_id"] == task_id
            and row["role"] == "initial" and row["case_id"] in ("P01", "P02")]
    require([row["case_id"] for row in rows] == ["P01", "P02"], "public observation identity")
    require(all(row["status"] == "completed" and type(row["expected_match"]) is bool
                for row in rows), "completed public observations")
    return {"kind": "visible_tests", "cases": [
        {"id": row["case_id"], "status": "pass" if row["expected_match"] else "mismatch"}
        for row in rows]}


def _append(rows, category, task, arm, messages, answer=None,
            reference_id=None, branch=None, feedback=None):
    row = {
        "ordinal": len(rows), "category": category, "task_id": task["id"], "arm": arm,
        "reference_id": reference_id, "branch": branch,
        "messages": copy.deepcopy(messages), "messages_sha256": digest(messages),
        "answer": answer, "answer_sha256": _text_sha(answer),
        "initial_source_sha256": _text_sha(task["initial_source"]),
        "examples_sha256": EXAMPLES_SHA256 if arm == "text" else None,
        "public_feedback_sha256": None if feedback is None else digest(feedback),
    }
    rows.append({**row, "row_sha256": digest(row)})


def build_synthetic_inventory(raw_files, worker_raw, run_ids, records, rendering_raw, *,
                              expected_runtime):
    rendering_sha256 = _source_identity(rendering_raw)
    dataset = load_dataset(raw_files)
    report = build_report(raw_files, worker_raw, run_ids, records, expected_runtime=expected_runtime)
    satisfied = report["matrix_complete"] is True and report["data_checks_satisfied"] is True
    rows = []
    if satisfied:
        for category in PLANNED_COUNTS:
            for task_index, (task, oracle) in enumerate(zip(dataset["prompts"], dataset["oracles"])):
                arms = ("base", "text") if task_index % 2 == 0 else ("text", "base")
                for arm in arms:
                    first = first_round(task, arm, raw_files["supervision_v1.json"])
                    if category == "first":
                        _append(rows, category, task, arm, first)
                    elif category == "references":
                        for reference in oracle["references"]:
                            _append(rows, category, task, arm, first,
                                    answer=reference["source"], reference_id=reference["id"])
                    else:
                        feedback_first = (task_index % 4 < 2) == (arm == "base")
                        branches = ("feedback", "retry") if feedback_first else ("retry", "feedback")
                        for branch in branches:
                            feedback = (_visible_feedback(report, task["id"])
                                        if branch == "feedback" else None)
                            messages = second_round(task, arm, raw_files["supervision_v1.json"],
                                                    branch, feedback)
                            _append(rows, category, task, arm, messages,
                                    branch=branch, feedback=feedback)
    counts = {category: sum(row["category"] == category for row in rows)
              for category in PLANNED_COUNTS}
    require(counts == (PLANNED_COUNTS if satisfied else {
        category: 0 for category in PLANNED_COUNTS}), "inventory row counts")
    inventory = {
        "kind": "LORA_RECALIBRATION_TOKENIZER_INVENTORY", "version": 1,
        "data_version": "recalibration_v2_r1", "file_sha256": dict(FILE_PINS),
        "status": "SYNTHETIC_ONLY" if satisfied else "DEPENDENCY_UNSATISFIED",
        "synthetic_only": True, "evidence_kind": "synthetic",
        "planned": 120, "planned_counts": dict(PLANNED_COUNTS),
        "constructed_counts": counts, "rows": rows,
        "examples_sha256": EXAMPLES_SHA256, "rendering_sha256": rendering_sha256,
        "validation_report_sha256": digest(report),
        "validation_plan_sha256": report["plan_sha256"],
        "dynamic_evidence_sha256": report["evidence_sha256"],
        "expected_runtime_sha256": report["expected_runtime_sha256"],
        "worker_sha256": report["worker_sha256"], "run_ids": list(run_ids),
        "synthetic_data_checks_satisfied": satisfied,
        "dependency_first_stop": copy.deepcopy(report["first_stop"]),
        "limits": {"input_tokens": 4096, "answer_tokens_including_eos": 256,
                   "enable_thinking": False},
        "run_authorized": False, "tokenizer_dispatch_authorized": False,
        "execution_provenance_verified": False, "real_dynamic_validation_complete": False,
        "tokenizer_length_validated": False, "all_possible_second_rounds_validated": False,
        **{flag: False for flag in FLAGS},
    }
    return {**inventory, "inventory_sha256": digest(inventory)}


def validate_synthetic_inventory(inventory, raw_files, worker_raw, run_ids,
                                 records, rendering_raw, *, expected_runtime):
    _plain(inventory)
    expected = build_synthetic_inventory(raw_files, worker_raw, run_ids, records,
                                         rendering_raw, expected_runtime=expected_runtime)
    require(encoded(inventory) == encoded(expected), "synthetic inventory reconstruction differs")
