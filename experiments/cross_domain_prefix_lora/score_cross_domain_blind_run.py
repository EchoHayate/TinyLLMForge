from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import sys

from experiments.cross_domain_prefix_lora.domain_sop_contracts import (
    detect_cross_domain_contamination,
    score_general_retention_prediction,
    score_structured_sop_prediction,
)


def _sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def score_python_candidates_locally_isolated(
    records: list[dict],
    prompts_by_id: dict[str, dict],
    oracles_by_id: dict[str, dict],
) -> dict:
    repo_lora = Path(__file__).resolve().parents[1] / "lora_necessity"
    if str(repo_lora) not in sys.path:
        sys.path.insert(0, str(repo_lora))
    from recalibration_diagnostic_scoring import candidate_valid

    passed_ids: list[str] = []
    contaminated_cnt = 0
    total_cases_passed = 0
    total_cases = 0
    per_task = {}

    for rec in records:
        tid = rec["id"]
        cand = rec["text"]
        t_prompt = prompts_by_id[tid]
        orc = oracles_by_id[tid]
        contam = detect_cross_domain_contamination("code_bugfix", cand)
        if contam:
            contaminated_cnt += 1
        is_val = candidate_valid(t_prompt, cand)
        cases_ok = 0
        if is_val:
            ns: dict = {}
            try:
                exec(compile(cand, f"<{tid}>", "exec"), ns)
                fn = ns.get(t_prompt["entry_point"])
            except Exception:
                fn = None
            if callable(fn):
                for c_obj in orc["cases"]:
                    exp = c_obj["expected"]
                    try:
                        got = fn(*copy.deepcopy(c_obj["args"]))
                        if exp["kind"] == "return" and got == exp["value"] and type(got) is type(exp["value"]):
                            cases_ok += 1
                    except ValueError:
                        if exp["kind"] == "raises" and exp.get("type") == "ValueError":
                            cases_ok += 1
                    except Exception:
                        pass
        total_cases += len(orc["cases"])
        total_cases_passed += cases_ok
        all_ok = (cases_ok == len(orc["cases"])) and is_val and (not contam)
        if all_ok:
            passed_ids.append(tid)
        per_task[tid] = {
            "valid_syntax": is_val,
            "cases_passed": cases_ok,
            "total_cases": len(orc["cases"]),
            "strict_pass": all_ok,
            "contaminated": contam,
        }

    return {
        "tasks_total": len(records),
        "strict_pass_count": len(passed_ids),
        "strict_pass_rate": round(len(passed_ids) / float(len(records)), 4),
        "case_level_pass_count": total_cases_passed,
        "case_level_total": total_cases,
        "case_level_rate": round(total_cases_passed / float(max(total_cases, 1)), 4),
        "contamination_count": contaminated_cnt,
        "contamination_rate": round(contaminated_cnt / float(len(records)), 4),
        "passed_ids": passed_ids,
        "per_task": per_task,
    }


def score_sop_cell(domain: str, cell_obj: dict, domain_oracles: dict[str, dict]) -> dict:
    recs = cell_obj["records"]
    strict_ids: list[str] = []
    valid_json_cnt = 0
    matched_fields_sum = 0
    total_fields_sum = 0
    contam_cnt = 0
    per_case = {}

    for rec in recs:
        cid = rec["id"]
        exp_d = domain_oracles[cid]
        sc = score_structured_sop_prediction(domain, rec["text"], exp_d)
        if sc["valid_json"]:
            valid_json_cnt += 1
        if sc["strict_pass"]:
            strict_ids.append(cid)
        if sc["contaminated"]:
            contam_cnt += 1
        matched_fields_sum += sc["matched_fields"]
        total_fields_sum += sc["total_fields"]
        per_case[cid] = {
            "valid_json": sc["valid_json"],
            "strict_pass": sc["strict_pass"],
            "matched_fields": sc["matched_fields"],
            "contaminated": sc["contaminated"],
        }

    return {
        "tasks_total": len(recs),
        "valid_json_count": valid_json_cnt,
        "strict_pass_count": len(strict_ids),
        "strict_pass_rate": round(len(strict_ids) / float(len(recs)), 4),
        "field_matched_count": matched_fields_sum,
        "field_total_count": total_fields_sum,
        "field_accuracy": round(matched_fields_sum / float(max(total_fields_sum, 1)), 4),
        "contamination_count": contam_cnt,
        "contamination_rate": round(contam_cnt / float(len(recs)), 4),
        "mean_input_tokens": cell_obj["mean_input_tokens"],
        "total_wall_ms": cell_obj["total_wall_ms"],
        "median_wall_ms": cell_obj["median_wall_ms"],
        "passed_ids": strict_ids,
        "per_case": per_case,
    }


def score_general_cell(cell_obj: dict, pub_general_inputs: list[dict], gen_oracles: dict[str, str]) -> dict:
    by_id = {x["id"]: x for x in pub_general_inputs}
    recs = cell_obj["records"]
    passed_ids: list[str] = []
    contam_cnt = 0
    per_case = {}

    for rec in recs:
        gid = rec["id"]
        sc = score_general_retention_prediction(by_id[gid], rec["text"], gen_oracles[gid])
        if sc["passed"]:
            passed_ids.append(gid)
        if sc["contaminated"]:
            contam_cnt += 1
        per_case[gid] = sc

    return {
        "tasks_total": len(recs),
        "strict_pass_count": len(passed_ids),
        "strict_pass_rate": round(len(passed_ids) / float(len(recs)), 4),
        "contamination_count": contam_cnt,
        "contamination_rate": round(contam_cnt / float(len(recs)), 4),
        "mean_input_tokens": cell_obj["mean_input_tokens"],
        "total_wall_ms": cell_obj["total_wall_ms"],
        "median_wall_ms": cell_obj["median_wall_ms"],
        "passed_ids": passed_ids,
        "per_case": per_case,
    }


def score_full_cross_domain_stage1(repo_root: Path, stage1_raw: dict) -> dict:
    data_dir = repo_root / "experiments/cross_domain_prefix_lora/data/cross_domain_v1"
    manifest = json.loads((data_dir / "manifest.json").read_text("utf-8"))
    orc_raw = (data_dir / "blind_oracles.json").read_bytes()
    if _sha256_bytes(orc_raw) != manifest["blind_oracles_sha256"]:
        raise ValueError("Offline blind_oracles.json SHA256 mismatch")
    blind_oracles = json.loads(orc_raw.decode("utf-8"))
    pub_bundle = json.loads((data_dir / "public_bundle.json").read_text("utf-8"))

    conf_dir = repo_root / "experiments/lora_necessity/data/confirmatory_v2_r1"
    prompts_by_id = {x["id"]: x for x in json.loads((conf_dir / "prompts.json").read_text("utf-8"))}
    oracles_by_id = {x["id"]: x for x in json.loads((conf_dir / "oracles.json").read_text("utf-8"))}

    scored_long_text = {
        "code_bugfix": {
            **score_python_candidates_locally_isolated(
                stage1_raw["long_text_baselines"]["code_bugfix"]["records"],
                prompts_by_id,
                oracles_by_id,
            ),
            "mean_input_tokens": stage1_raw["long_text_baselines"]["code_bugfix"]["mean_input_tokens"],
            "total_wall_ms": stage1_raw["long_text_baselines"]["code_bugfix"]["total_wall_ms"],
            "median_wall_ms": stage1_raw["long_text_baselines"]["code_bugfix"]["median_wall_ms"],
        },
        "med_triage": score_sop_cell(
            "med_triage",
            stage1_raw["long_text_baselines"]["med_triage"],
            blind_oracles["med_triage"],
        ),
        "legal_compliance": score_sop_cell(
            "legal_compliance",
            stage1_raw["long_text_baselines"]["legal_compliance"],
            blind_oracles["legal_compliance"],
        ),
    }

    scored_matrix: dict[str, dict[str, dict]] = {}
    for slot_name, dom_map in stage1_raw["matrix_generations"].items():
        scored_matrix[slot_name] = {}
        for eval_dom, cell_obj in dom_map.items():
            if eval_dom == "code_bugfix":
                sc_code = score_python_candidates_locally_isolated(
                    cell_obj["records"], prompts_by_id, oracles_by_id
                )
                scored_matrix[slot_name][eval_dom] = {
                    **sc_code,
                    "mean_input_tokens": cell_obj["mean_input_tokens"],
                    "total_wall_ms": cell_obj["total_wall_ms"],
                    "median_wall_ms": cell_obj["median_wall_ms"],
                }
            elif eval_dom in ("med_triage", "legal_compliance"):
                scored_matrix[slot_name][eval_dom] = score_sop_cell(
                    eval_dom, cell_obj, blind_oracles[eval_dom]
                )
            else:
                scored_matrix[slot_name][eval_dom] = score_general_cell(
                    cell_obj,
                    pub_bundle["general_retention"]["blind_test_inputs"],
                    blind_oracles["general_retention"],
                )

    in_domain_summary = {}
    for dom_name, slot_key in (
        ("code_bugfix", "slot_1_code_lora"),
        ("med_triage", "slot_2_med_lora"),
        ("legal_compliance", "slot_3_legal_lora"),
    ):
        b_cell = scored_matrix["slot_0_base"][dom_name]
        t_cell = scored_long_text[dom_name]
        l_cell = scored_matrix[slot_key][dom_name]
        in_domain_summary[dom_name] = {
            "base_pass": f"{b_cell['strict_pass_count']}/{b_cell['tasks_total']}",
            "long_text_pass": f"{t_cell['strict_pass_count']}/{t_cell['tasks_total']}",
            "lora_pass": f"{l_cell['strict_pass_count']}/{l_cell['tasks_total']}",
            "base_submetric_rate": b_cell.get("field_accuracy", b_cell.get("case_level_rate")),
            "long_text_submetric_rate": t_cell.get("field_accuracy", t_cell.get("case_level_rate")),
            "lora_submetric_rate": l_cell.get("field_accuracy", l_cell.get("case_level_rate")),
            "speedup_lora_vs_long_text": round(
                t_cell["total_wall_ms"] / max(l_cell["total_wall_ms"], 1e-6), 3
            ),
            "prompt_token_reduction": f"{t_cell['mean_input_tokens']} -> {l_cell['mean_input_tokens']}",
        }

    off_diag_cells = [
        ("slot_1_code_lora", "med_triage"),
        ("slot_1_code_lora", "legal_compliance"),
        ("slot_1_code_lora", "general_retention"),
        ("slot_2_med_lora", "code_bugfix"),
        ("slot_2_med_lora", "legal_compliance"),
        ("slot_2_med_lora", "general_retention"),
        ("slot_3_legal_lora", "code_bugfix"),
        ("slot_3_legal_lora", "med_triage"),
        ("slot_3_legal_lora", "general_retention"),
    ]
    total_off_diag_contam = sum(
        scored_matrix[s_k][d_k]["contamination_count"] for s_k, d_k in off_diag_cells
    )
    total_off_diag_evals = sum(
        scored_matrix[s_k][d_k]["tasks_total"] for s_k, d_k in off_diag_cells
    )

    gpu_info = dict(stage1_raw["gpu_summary"])
    gpu_info.pop("index", None)

    return {
        "kind": "CROSS_DOMAIN_BLIND_EVALUATION_AND_MATRIX_REPORT",
        "version": "cross_domain_v1",
        "single_shot_blind_protocol": True,
        "zero_blind_oracle_in_gpu_payload": bool(stage1_raw["zero_blind_oracle_in_gpu_payload"]),
        "public_bundle_sha256": manifest["public_bundle_sha256"],
        "blind_oracles_sha256": manifest["blind_oracles_sha256"],
        "base_model_sha256_verified_unchanged": bool(stage1_raw["base_model_sha256_verified_unchanged"]),
        "gpu_summary": gpu_info,
        "domain_training_summaries": stage1_raw["domain_training_summaries"],
        "in_domain_blind_summary": in_domain_summary,
        "off_diagonal_contamination": {
            "contaminated_outputs": total_off_diag_contam,
            "total_off_diagonal_outputs": total_off_diag_evals,
            "contamination_rate": round(total_off_diag_contam / float(max(total_off_diag_evals, 1)), 6),
        },
        "scored_long_text_baselines": scored_long_text,
        "scored_4x4_matrix": scored_matrix,
    }
