import ast
from collections import Counter
import copy
import hashlib
import re

from pilot_protocol import ARMS, FLAGS, fields, require
from prompt_rendering import FEEDBACK_STATUSES, INSTRUCTION, PUBLIC_FIELDS, REVISION
from recalibration_data_contract import CASE_IDS, SOURCE_IDS, _outcome, _plain, _source, _text
from recalibration_validation import WORKER_SHA256
from validation_contract import digest, encoded, parse_json
from validation_real_report import _events, _process_result
from validation_report import parse_outcome


CONFIRMATORY_FILE_PINS = {
    "manifest.json": "8d9beab7820e4f1b731cabcfc16ba43502d7faf83d6256807bd5974209a59597",
    "oracles.json": "8d251cdc6d2f7163d6e928965bd46a04d0b418904cd89eb6cfa83dcb3f1d3a04",
    "prompts.json": "f78ea75b5a3010f5d5d5038f29d7f22037aad3c020d47ba5705b1048244ce37b",
    "structure.json": "5e8083da8cb8323ade8ee9b0360204c795dd4f8124a061304fc6e1e5ecf21b68",
    "supervision.json": "1b2173eb0cd45dec646485e2e4b6eedb03d3e44aad43caff8aab8111f05213cc",
}
SUPERVISION_IDS = tuple("V%02d" % idx for idx in range(1, 9))
TRANSFER_IDS = tuple("WT%02d" % idx for idx in range(1, 13))
RETENTION_IDS = tuple("WR%02d" % idx for idx in range(1, 13))
EVAL_IDS = TRANSFER_IDS + RETENTION_IDS
ALL_TASK_IDS = SUPERVISION_IDS + EVAL_IDS
TARGET_FAMILIES = ("bounds", "multiplicity", "order", "reset")
RETENTION_FAMILIES = ("balance", "dedup", "dispatch", "window")


def _sha256(raw):
    return hashlib.sha256(raw).hexdigest()


def _public_projection(task):
    return {name: copy.deepcopy(task[name]) for name in PUBLIC_FIELDS}


def _witness_projection(task):
    wit = task["initial_witness"]
    return {
        "public_case_id": wit["public_case_id"],
        "predicted_initial": copy.deepcopy(wit["predicted_initial"]),
    }


def _supervision_examples(supervision_list):
    return [
        {
            **_public_projection(item),
            "initial_witness": _witness_projection(item),
            "reference_source": item["reference_source"],
        }
        for item in supervision_list
    ]


def load_confirmatory_dataset(raw_files):
    fields(raw_files, CONFIRMATORY_FILE_PINS)
    for fname, expected_sha in CONFIRMATORY_FILE_PINS.items():
        raw = raw_files[fname]
        require(type(raw) is bytes and _sha256(raw) == expected_sha, "file pin: " + fname)

    manifest = parse_json(raw_files["manifest.json"])
    supervision = parse_json(raw_files["supervision.json"])
    prompts = parse_json(raw_files["prompts.json"])
    oracles = parse_json(raw_files["oracles.json"])
    structure = parse_json(raw_files["structure.json"])

    for obj in (manifest, supervision, prompts, oracles, structure):
        _plain(obj)

    require(
        manifest.get("kind") == "LORA_CONFIRMATORY_V2_MANIFEST"
        and manifest.get("data_version") == "confirmatory_v2_r1"
        and all(manifest.get(flag_name) is False for flag_name in FLAGS)
        and manifest.get("file_sha256") == {
            key_n: val_s for key_n, val_s in CONFIRMATORY_FILE_PINS.items() if key_n != "manifest.json"
        },
        "manifest header",
    )
    require(tuple(item["id"] for item in supervision) == SUPERVISION_IDS, "supervision IDs")
    require(tuple(item["id"] for item in prompts) == EVAL_IDS, "eval prompt IDs")
    require(tuple(item["id"] for item in oracles) == ALL_TASK_IDS, "oracle IDs")
    require(tuple(item["id"] for item in structure["tasks"]) == ALL_TASK_IDS, "structure IDs")

    oracles_map = {item["id"]: item for item in oracles}
    struct_map = {item["id"]: item for item in structure["tasks"]}
    seen_entries = set()
    fam_counts = {"supervision": Counter(), "transfer": Counter(), "retention": Counter()}

    for sup_item in supervision:
        tid = sup_item["id"]
        oracle = oracles_map[tid]
        meta = struct_map[tid]
        entry_point = sup_item["entry_point"]
        require(entry_point not in seen_entries, "duplicate entry_point")
        seen_entries.add(entry_point)
        require(
            entry_point == oracle["entry_point"] == meta["entry_point"]
            and meta["split"] == "supervision"
            and sup_item["family"] == oracle["family"] == meta["family"],
            "supervision metadata alignment",
        )
        fam_counts["supervision"][meta["family"]] += 1
        _validate_oracle_and_public(
            sup_item["public_cases"], sup_item["initial_source"], oracle, meta
        )
        require(
            sup_item["reference_source"] == oracle["references"][0]["source"]
            and sup_item["initial_witness"]["public_case_id"] == oracle["incorrect_programs"][0]["witness"]["case_id"]
            and encoded(sup_item["initial_witness"]["predicted_initial"])
            == encoded(oracle["incorrect_programs"][0]["witness"]["predicted"]),
            "supervision reference/witness mismatch",
        )

    for pr_item in prompts:
        tid = pr_item["id"]
        oracle = oracles_map[tid]
        meta = struct_map[tid]
        entry_point = pr_item["entry_point"]
        require(entry_point not in seen_entries, "duplicate entry_point")
        seen_entries.add(entry_point)
        expected_split = "transfer" if tid in TRANSFER_IDS else "retention"
        require(
            pr_item["split"] == expected_split
            and oracle["split"] == expected_split
            and meta["split"] == expected_split
            and pr_item["family"] == oracle["family"] == meta["family"],
            "eval split alignment",
        )
        fam_counts[expected_split][meta["family"]] += 1
        _validate_oracle_and_public(
            pr_item["public_cases"], pr_item["initial_source"], oracle, meta
        )

    require(
        dict(fam_counts["supervision"]) == {fam: 2 for fam in TARGET_FAMILIES},
        "supervision family balance",
    )
    require(
        dict(fam_counts["transfer"]) == {fam: 3 for fam in TARGET_FAMILIES},
        "transfer family balance",
    )
    require(
        dict(fam_counts["retention"]) == {fam: 3 for fam in RETENTION_FAMILIES},
        "retention family balance",
    )
    return {
        "manifest": manifest,
        "supervision": supervision,
        "prompts": prompts,
        "oracles": oracles,
        "structure": structure,
    }


def _validate_oracle_and_public(public_cases, initial_source, oracle, meta):
    entry_point = meta["entry_point"]
    parameters = meta["parameters"]
    cases = oracle["cases"]
    require(type(cases) is list and len(cases) == 8, "8 cases required")
    for case_obj, expected_cid in zip(cases, CASE_IDS):
        require(
            case_obj["id"] == expected_cid and len(case_obj["args"]) == len(parameters),
            "case id/arity",
        )
        _outcome(case_obj["expected"])
    require(len({encoded(case_obj["args"]) for case_obj in cases}) == 8, "distinct case args")
    require(len(public_cases) == 2, "2 public cases")
    for pub_c, full_c in zip(public_cases, cases[:2]):
        require(
            encoded(pub_c) == encoded({key_n: full_c[key_n] for key_n in ("id", "args", "expected")}),
            "public case projection mismatch",
        )
    refs = oracle["references"]
    incs = oracle["incorrect_programs"]
    require(len(refs) == 2 and len(incs) == 3, "2 references and 3 variants required")
    all_sources = [ref_item["source"] for ref_item in refs] + [inc_item["source"] for inc_item in incs]
    require(len(set(all_sources)) == 5, "5 distinct sources required")
    require(
        ast.dump(ast.parse(refs[0]["source"])) != ast.dump(ast.parse(refs[1]["source"])),
        "distinct reference ASTs required",
    )
    for ref_item, ref_id in zip(refs, SOURCE_IDS[:2]):
        require(ref_item["id"] == ref_id, "ref id")
        _source(ref_item["source"], entry_point, parameters)
    cases_by_id = {case_obj["id"]: case_obj for case_obj in cases}
    for inc_item, inc_id in zip(incs, SOURCE_IDS[2:]):
        require(inc_item["id"] == inc_id, "variant id")
        _source(inc_item["source"], entry_point, parameters)
        wit = inc_item["witness"]
        _outcome(wit["predicted"])
        require(
            encoded(wit["predicted"]) != encoded(cases_by_id[wit["case_id"]]["expected"]),
            "witness must disagree with expected",
        )
    require(incs[0]["witness"]["case_id"] in CASE_IDS[:2], "initial witness must be public")
    require(incs[1]["witness"]["case_id"] in CASE_IDS[2:], "variant_1 witness must be private")
    require(incs[0]["source"] == initial_source, "initial_source mismatch")


def _load_supervision_only(supervision_raw):
    require(
        type(supervision_raw) is bytes
        and _sha256(supervision_raw) == CONFIRMATORY_FILE_PINS["supervision.json"],
        "supervision.json digest mismatch",
    )
    sup_list = parse_json(supervision_raw)
    require(tuple(item["id"] for item in sup_list) == SUPERVISION_IDS, "supervision order")
    return sup_list


def _compact_outcome(outcome_obj):
    if outcome_obj["kind"] == "return":
        return {"ret": outcome_obj["value"]}
    return {"err": outcome_obj["type"]}


def _format_task_block(task, witness_obj=None, reference_source=None):
    compact_cases = [
        [c_it["id"], c_it["args"], _compact_outcome(c_it["expected"])]
        for c_it in task["public_cases"]
    ]
    header_json = encoded({
        "id": task["id"],
        "fn": task["entry_point"],
        "spec": task["spec"],
        "cases": compact_cases,
    }).decode("ascii")
    lines = [
        "Task: " + header_json,
        "Initial:\n" + task["initial_source"].rstrip("\n"),
    ]
    if witness_obj is not None:
        wit_json = encoded([
            witness_obj["public_case_id"],
            _compact_outcome(witness_obj["predicted_initial"]),
        ]).decode("ascii")
        lines.append("Failure: " + wit_json)
    if reference_source is not None:
        lines.append("Repair:\n" + reference_source.rstrip("\n"))
    return "\n".join(lines)


def _render_user_prompt(task, examples_list=None, witness_obj=None):
    content = INSTRUCTION
    if examples_list is not None:
        ex_blocks = [
            _format_task_block(
                item,
                witness_obj=_witness_projection(item),
                reference_source=item["reference_source"],
            )
            for item in examples_list
        ]
        content += "\n\nSupervised examples:\n" + "\n\n".join(ex_blocks)
    content += "\n\nCurrent task:\n" + _format_task_block(task, witness_obj=witness_obj)
    return [{"role": "user", "content": content}]


def confirmatory_training_jobs(supervision_raw):
    sup_list = _load_supervision_only(supervision_raw)
    return [
        {
            "task_id": item["id"],
            "messages": _render_user_prompt(
                item,
                examples_list=None,
                witness_obj=_witness_projection(item),
            ),
            "answer": item["reference_source"],
        }
        for item in sup_list
    ]


def confirmatory_first_round(task, arm, supervision_raw):
    require(arm in ARMS, "unknown arm")
    sup_list = _load_supervision_only(supervision_raw)
    examples_list = sup_list if arm in ("text", "lora_text") else None
    return _render_user_prompt(task, examples_list=examples_list, witness_obj=None)


def confirmatory_second_round(task, arm, supervision_raw, initial_candidate, feedback):
    require(type(initial_candidate) is str and len(initial_candidate.encode("utf-8")) <= 65536, "candidate cap")
    _plain(feedback)
    require(type(feedback) is dict, "feedback dict")
    if feedback.get("kind") == "invalid_source":
        fields(feedback, ("kind",))
    else:
        fields(feedback, ("kind", "cases"))
        require(feedback["kind"] == "visible_tests" and len(feedback["cases"]) == 2, "visible_tests")
        for c_item, expected_cid in zip(feedback["cases"], CASE_IDS[:2]):
            fields(c_item, ("id", "status"))
            require(c_item["id"] == expected_cid and c_item["status"] in FEEDBACK_STATUSES, "feedback case")
    base_msgs = confirmatory_first_round(task, arm, supervision_raw)
    return base_msgs + [
        {"role": "assistant", "content": initial_candidate},
        {"role": "user", "content": REVISION + "\n" + encoded(feedback).decode("ascii")},
    ]


def build_confirmatory_validation_batches(raw_files, worker_raw, run_ids):
    dataset = load_confirmatory_dataset(raw_files)
    require(type(worker_raw) is bytes and _sha256(worker_raw) == WORKER_SHA256, "worker pin")
    require(type(run_ids) is list and len(run_ids) == 40 and len(set(run_ids)) == 40, "40 unique run_ids")
    for rid in run_ids:
        require(type(rid) is str and re.fullmatch("[0-9a-f]{32}", rid) is not None, "run_id hex")

    worker_text = worker_raw.decode("utf-8")
    all_positions = []
    for oracle in dataset["oracles"]:
        sources_list = [
            ("reference", ref["id"], ref["source"], None)
            for ref in oracle["references"]
        ] + [
            ("incorrect", inc["id"], inc["source"], inc["witness"])
            for inc in oracle["incorrect_programs"]
        ]
        for s_role, s_id, s_code, wit_obj in sources_list:
            for case_obj in oracle["cases"]:
                pos_idx = len(all_positions)
                req_obj = {
                    "request_id": "conf-val-%04d" % pos_idx,
                    "source": s_code,
                    "entry_point": oracle["entry_point"],
                    "args": copy.deepcopy(case_obj["args"]),
                }
                is_wit = wit_obj is not None and wit_obj["case_id"] == case_obj["id"]
                all_positions.append({
                    "position_index": pos_idx,
                    "task_id": oracle["id"],
                    "source_role": s_role,
                    "source_id": s_id,
                    "case_id": case_obj["id"],
                    "expected": copy.deepcopy(case_obj["expected"]),
                    "witness_predicted": copy.deepcopy(wit_obj["predicted"]) if is_wit else None,
                    "request": req_obj,
                    "request_sha256": digest(req_obj),
                })

    require(len(all_positions) == 1280, "1280 validation positions required")
    batches = []
    for b_idx, rid in enumerate(run_ids):
        slice_pos = all_positions[b_idx * 32:(b_idx + 1) * 32]
        bundle_obj = {
            "kind": "LORA_REMOTE_VALIDATION_BUNDLE",
            "mode": "matrix",
            "run_id": rid,
            "worker_source": worker_text,
            "requests": [pos["request"] for pos in slice_pos],
        }
        bundle_raw = encoded(bundle_obj)
        batches.append({
            "batch_index": b_idx,
            "run_id": rid,
            "requests": bundle_obj["requests"],
            "positions": slice_pos,
            "bundle_raw": bundle_raw,
            "bundle_sha256": _sha256(bundle_raw),
        })
    return batches


def score_confirmatory_validation_batches(raw_files, worker_raw, run_ids, batch_records, expected_runtime):
    batches = build_confirmatory_validation_batches(raw_files, worker_raw, run_ids)
    require(len(batch_records) == 40, "40 batch records required")
    ref_passed = 0
    wit_matched = 0
    var1_public_passed = 0
    completed_count = 0

    for b_meta, b_rec in zip(batches, batch_records):
        require(b_rec["batch_index"] == b_meta["batch_index"], "batch index")
        stdout_raw = _process_result(b_rec["transport"], outer=True)
        events, truncated = _events(stdout_raw, timed_out=False)
        require(not truncated and b_rec["transport"]["status"] == "completed", "batch transport")
        ready_ev = events[0]
        done_ev = events[-1]
        row_evs = events[1:-1]
        require(
            ready_ev.get("kind") == "ready"
            and ready_ev.get("run_id") == b_meta["run_id"]
            and ready_ev.get("bundle_sha256") == b_meta["bundle_sha256"]
            and ready_ev.get("worker_sha256") == WORKER_SHA256
            and encoded(ready_ev.get("runtime")) == encoded(expected_runtime),
            "batch ready mismatch",
        )
        require(
            done_ev.get("kind") == "done"
            and done_ev.get("received") == 32
            and done_ev.get("stopped") is False
            and len(row_evs) == 32,
            "batch done mismatch",
        )
        for ord_idx, (pos, row_ev) in enumerate(zip(b_meta["positions"], row_evs)):
            require(
                row_ev.get("ordinal") == ord_idx and row_ev.get("request_sha256") == pos["request_sha256"],
                "row ordinal/sha",
            )
            res = row_ev["result"]
            require(res["status"] == "completed" and res["process_reaped"] is True, "worker completed")
            outcome = parse_outcome(_process_result(res))
            completed_count += 1
            if pos["source_role"] == "reference":
                require(encoded(outcome) == encoded(pos["expected"]), "reference mismatch: " + pos["task_id"])
                ref_passed += 1
            elif pos["source_id"] == "variant_1" and pos["case_id"] in CASE_IDS[:2]:
                require(encoded(outcome) == encoded(pos["expected"]), "variant_1 public case must pass: " + pos["task_id"])
                var1_public_passed += 1
            elif pos["witness_predicted"] is not None:
                require(
                    encoded(outcome) == encoded(pos["witness_predicted"])
                    and encoded(outcome) != encoded(pos["expected"]),
                    "witness mismatch: " + pos["task_id"],
                )
                wit_matched += 1

    require(
        completed_count == 1280 and ref_passed == 512 and var1_public_passed == 64 and wit_matched == 96,
        "validation totals",
    )
    return {
        "kind": "LORA_CONFIRMATORY_V2_DYNAMIC_REPORT",
        "version": 1,
        "data_version": "confirmatory_v2_r1",
        "completed": completed_count,
        "references_passed": ref_passed,
        "witnesses_matched": wit_matched,
        "all_checks_passed": True,
        **{flag: False for flag in FLAGS},
    }


def build_confirmatory_tokenizer_projection(raw_files):
    dataset = load_confirmatory_dataset(raw_files)
    sup_raw = raw_files["supervision.json"]
    oracles_map = {item["id"]: item for item in dataset["oracles"]}
    rows = []

    for tjob in confirmatory_training_jobs(sup_raw):
        entry = {
            "ordinal": len(rows),
            "category": "training",
            "task_id": tjob["task_id"],
            "messages": tjob["messages"],
            "answer": tjob["answer"],
        }
        rows.append({
            "ordinal": entry["ordinal"],
            "category": entry["category"],
            "task_id": entry["task_id"],
            "messages": entry["messages"],
            "answer": entry["answer"],
            "row_sha256": digest(entry),
        })

    for task in dataset["prompts"]:
        for arm in ("base", "text"):
            f_msgs = confirmatory_first_round(task, arm, sup_raw)
            entry_first = {
                "ordinal": len(rows),
                "category": "first",
                "task_id": task["id"],
                "messages": f_msgs,
                "answer": None,
            }
            rows.append({**entry_first, "row_sha256": digest(entry_first)})

    for task in dataset["prompts"]:
        oracle = oracles_map[task["id"]]
        for ref_item in oracle["references"]:
            for arm in ("base", "text"):
                f_msgs = confirmatory_first_round(task, arm, sup_raw)
                entry_ref = {
                    "ordinal": len(rows),
                    "category": "references",
                    "task_id": task["id"],
                    "messages": f_msgs,
                    "answer": ref_item["source"],
                }
                rows.append({**entry_ref, "row_sha256": digest(entry_ref)})

    worst_fb = {
        "kind": "visible_tests",
        "cases": [{"id": "P01", "status": "program_error"}, {"id": "P02", "status": "program_error"}],
    }
    for task in dataset["prompts"]:
        oracle = oracles_map[task["id"]]
        longest_cand = max(
            [oracle["references"][0]["source"], oracle["references"][1]["source"], task["initial_source"]],
            key=lambda txt: len(txt.encode("utf-8")),
        )
        for arm in ("base", "text"):
            s_msgs = confirmatory_second_round(task, arm, sup_raw, longest_cand, worst_fb)
            entry_sec = {
                "ordinal": len(rows),
                "category": "sampled_second",
                "task_id": task["id"],
                "messages": s_msgs,
                "answer": None,
            }
            rows.append({**entry_sec, "row_sha256": digest(entry_sec)})

    require(len(rows) == 200, "200 tokenizer audit rows required")
    return {
        "kind": "LORA_CONFIRMATORY_V2_TOKENIZER_PROJECTION",
        "version": 1,
        "data_version": "confirmatory_v2_r1",
        "rows": rows,
    }
