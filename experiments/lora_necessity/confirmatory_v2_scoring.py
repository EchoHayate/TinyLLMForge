from collections import Counter
import copy
import hashlib
import re

from confirmatory_v2_contract import (
    CONFIRMATORY_FILE_PINS,
    EVAL_IDS,
    RETENTION_IDS,
    TRANSFER_IDS,
    confirmatory_first_round,
    confirmatory_second_round,
    load_confirmatory_dataset,
)
from pilot_protocol import ARMS, FLAGS, fields, require
from recalibration_data_contract import CASE_IDS, _plain
from recalibration_diagnostic_scoring import (
    CANDIDATE_STATES,
    _classify_candidate,
    _decode_scoring_batch,
    candidate_valid,
    validate_public_feedback,
)
from recalibration_validation import WORKER_SHA256
from validation_contract import digest, encoded, parse_json


CONFIRMATORY_ROLES = ("initial", "feedback")
VERDICT_STATUSES = (
    "INVALID_PROTOCOL",
    "INCONCLUSIVE_INCOMPLETE",
    "OBSERVED_RETENTION_REGRESSION",
    "POSITIVE_MATCHED_LORA_GAIN",
    "COMPRESSION_SPEEDUP_WITHOUT_LORA_TEXT_GAIN",
    "NO_LORA_GAIN_OBSERVED",
)


def _check_hex(text, length=32):
    require(
        type(text) is str and re.fullmatch("[0-9a-f]{%d}" % length, text) is not None,
        "invalid hex identifier",
    )


def _load_prompts_only(prompts_raw, supervision_raw):
    require(
        type(prompts_raw) is bytes
        and hashlib.sha256(prompts_raw).hexdigest() == CONFIRMATORY_FILE_PINS["prompts.json"],
        "prompts.json pin mismatch",
    )
    require(
        type(supervision_raw) is bytes
        and hashlib.sha256(supervision_raw).hexdigest() == CONFIRMATORY_FILE_PINS["supervision.json"],
        "supervision.json pin mismatch",
    )
    prompts = parse_json(prompts_raw)
    require(tuple(item["id"] for item in prompts) == EVAL_IDS, "eval prompt order")
    return prompts


def build_confirmatory_schedule(prompts_raw, supervision_raw, *, run_id):
    _check_hex(run_id, 32)
    prompts = _load_prompts_only(prompts_raw, supervision_raw)
    episodes = []
    for task_idx, task in enumerate(prompts):
        rot = task_idx % 4
        arm_order = ARMS[rot:] + ARMS[:rot]
        for arm in arm_order:
            ep_idx = len(episodes)
            f_msgs = confirmatory_first_round(task, arm, supervision_raw)
            episodes.append({
                "episode_index": ep_idx,
                "task_id": task["id"],
                "split": task["split"],
                "family": task["family"],
                "entry_point": task["entry_point"],
                "arm": arm,
                "first_messages": f_msgs,
                "first_messages_sha256": digest(f_msgs),
                "call_ids": {
                    role_name: hashlib.sha256(
                        encoded(["conf_v2_call", run_id, ep_idx, task["id"], arm, role_name])
                    ).hexdigest()[:32]
                    for role_name in CONFIRMATORY_ROLES
                },
            })
    require(len(episodes) == 96, "96 episodes required")
    return {
        "kind": "LORA_CONFIRMATORY_V2_SCHEDULE",
        "version": 1,
        "data_version": "confirmatory_v2_r1",
        "run_id": run_id,
        "prompts_sha256": CONFIRMATORY_FILE_PINS["prompts.json"],
        "supervision_sha256": CONFIRMATORY_FILE_PINS["supervision.json"],
        "scheduled_episodes": 96,
        "scheduled_generation_calls": 192,
        "episodes": episodes,
    }


def plan_confirmatory_positions(raw_files, phase_one_records):
    dataset = load_confirmatory_dataset(raw_files)
    prompts_map = {item["id"]: item for item in dataset["prompts"]}
    oracles_map = {item["id"]: item for item in dataset["oracles"]}
    require(type(phase_one_records) is list and len(phase_one_records) <= 96, "phase_one_records bounds")
    positions = []
    valid_positions = []

    for ep_idx, ep_rec in enumerate(phase_one_records):
        require(ep_rec["episode_index"] == ep_idx, "episode_index order")
        tid = ep_rec["task_id"]
        arm = ep_rec["arm"]
        task_prompt = prompts_map[tid]
        oracle = oracles_map[tid]
        for role_name in CONFIRMATORY_ROLES:
            cand_src = ep_rec["candidates"][role_name]
            is_valid = candidate_valid(task_prompt, cand_src)
            for case_obj in oracle["cases"]:
                pos_index = len(positions)
                req_obj = None
                req_sha = None
                if is_valid:
                    req_obj = {
                        "request_id": "conf-p2-%04d" % len(valid_positions),
                        "source": cand_src,
                        "entry_point": task_prompt["entry_point"],
                        "args": copy.deepcopy(case_obj["args"]),
                    }
                    req_sha = digest(req_obj)
                entry = {
                    "position_index": pos_index,
                    "episode_index": ep_idx,
                    "task_id": tid,
                    "split": task_prompt["split"],
                    "family": task_prompt["family"],
                    "arm": arm,
                    "role": role_name,
                    "case_id": case_obj["id"],
                    "valid_source": is_valid,
                    "source": cand_src,
                    "request": req_obj,
                    "request_sha256": req_sha,
                    "expected": copy.deepcopy(case_obj["expected"]),
                }
                positions.append(entry)
                if is_valid:
                    valid_positions.append(entry)

    batches = [
        valid_positions[offset:offset + 32]
        for offset in range(0, len(valid_positions), 32)
    ]
    return {"all_positions": positions, "valid_positions": valid_positions, "batches": batches}


def build_confirmatory_continuation_bundle(worker_raw, run_id, batch_index, sub_index, positions_slice):
    _check_hex(run_id, 32)
    require(
        type(worker_raw) is bytes and hashlib.sha256(worker_raw).hexdigest() == WORKER_SHA256,
        "worker pin",
    )
    require(type(batch_index) is int and 0 <= batch_index < 48, "batch_index bounds")
    require(type(sub_index) is int and 1 <= sub_index <= 32, "sub_index bounds")
    require(type(positions_slice) is list and 1 <= len(positions_slice) <= 32, "positions_slice")
    cont_run_id = hashlib.sha256(
        encoded(["conf_v2_p2_cont", run_id, batch_index, sub_index])
    ).hexdigest()[:32]
    bundle_obj = {
        "kind": "LORA_REMOTE_VALIDATION_BUNDLE",
        "mode": "matrix",
        "run_id": cont_run_id,
        "worker_source": worker_raw.decode("utf-8"),
        "requests": [pos["request"] for pos in positions_slice],
    }
    bundle_raw = encoded(bundle_obj)
    return {
        "batch_index": batch_index,
        "sub_index": sub_index,
        "run_id": cont_run_id,
        "bundle_raw": bundle_raw,
        "bundle_sha256": hashlib.sha256(bundle_raw).hexdigest(),
        "worker_sha256": WORKER_SHA256,
        "positions": list(positions_slice),
    }


def build_confirmatory_scoring_bundles(raw_files, worker_raw, run_id, phase_one_records, scoring_run_ids):
    _check_hex(run_id, 32)
    require(
        type(worker_raw) is bytes and hashlib.sha256(worker_raw).hexdigest() == WORKER_SHA256,
        "worker pin",
    )
    worker_text = worker_raw.decode("utf-8")
    planned = plan_confirmatory_positions(raw_files, phase_one_records)
    batch_slices = planned["batches"]
    require(
        type(scoring_run_ids) is list and len(scoring_run_ids) == len(batch_slices),
        "scoring_run_ids length mismatch",
    )
    seen_ids = {run_id}
    built = []
    for b_idx, (b_slice, b_rid) in enumerate(zip(batch_slices, scoring_run_ids)):
        _check_hex(b_rid, 32)
        require(b_rid not in seen_ids, "duplicate scoring run_id")
        seen_ids.add(b_rid)
        bundle_obj = {
            "kind": "LORA_REMOTE_VALIDATION_BUNDLE",
            "mode": "matrix",
            "run_id": b_rid,
            "worker_source": worker_text,
            "requests": [pos["request"] for pos in b_slice],
        }
        bundle_raw = encoded(bundle_obj)
        built.append({
            "batch_index": b_idx,
            "run_id": b_rid,
            "bundle_raw": bundle_raw,
            "bundle_sha256": hashlib.sha256(bundle_raw).hexdigest(),
            "worker_sha256": WORKER_SHA256,
            "positions": b_slice,
        })
    return {
        "run_ids": list(scoring_run_ids),
        "batches": built,
        "all_positions": planned["all_positions"],
    }


def _paired_arm_comparison(task_ids, indexed_episodes, role_name, baseline_arm, target_arm, run_complete):
    both_right = 0
    both_wrong = 0
    wrong_to_right = 0
    right_to_wrong = 0
    missing_or_unknown = 0
    for tid in task_ids:
        ep_base = indexed_episodes.get((tid, baseline_arm))
        ep_targ = indexed_episodes.get((tid, target_arm))
        if ep_base is None or ep_targ is None:
            missing_or_unknown += 1
            continue
        r_base = ep_base["roles"][role_name]
        r_targ = ep_targ["roles"][role_name]
        known_states = ("PASS", "SEMANTIC_FAIL", "EXECUTION_FAIL")
        if (
            r_base["state"] not in known_states
            or r_targ["state"] not in known_states
            or r_base["has_unknown_case"]
            or r_targ["has_unknown_case"]
        ):
            missing_or_unknown += 1
            continue
        b_pass = r_base["state"] == "PASS"
        t_pass = r_targ["state"] == "PASS"
        if b_pass and t_pass:
            both_right += 1
        elif (not b_pass) and (not t_pass):
            both_wrong += 1
        elif (not b_pass) and t_pass:
            wrong_to_right += 1
        else:
            right_to_wrong += 1

    subset_net = wrong_to_right - right_to_wrong
    full_net = subset_net if (run_complete and missing_or_unknown == 0) else None
    return {
        "baseline_arm": baseline_arm,
        "target_arm": target_arm,
        "both_right": both_right,
        "both_wrong": both_wrong,
        "wrong_to_right": wrong_to_right,
        "right_to_wrong": right_to_wrong,
        "missing_or_unknown": missing_or_unknown,
        "observed_subset_net": subset_net,
        "net_gain": full_net,
    }


def score_confirmatory_run(
    raw_files,
    worker_raw,
    run_id,
    phase_one_records,
    scoring_run_ids,
    final_bundle_records,
    *,
    continuation_records=(),
    expected_runtime,
    training_summary=None,
    protocol_contradictions=(),
    incomplete_reasons=(),
    shared_costs=None,
):
    load_confirmatory_dataset(raw_files)
    expected_sched = build_confirmatory_schedule(
        raw_files["prompts.json"], raw_files["supervision.json"], run_id=run_id
    )["episodes"]
    bundles_plan = build_confirmatory_scoring_bundles(
        raw_files, worker_raw, run_id, phase_one_records, scoring_run_ids
    )
    require(len(final_bundle_records) <= len(bundles_plan["batches"]), "excess bundle records")

    cont_by_batch = {}
    for crec in continuation_records:
        cont_by_batch.setdefault(crec["batch_index"], []).append(crec)

    pos_observations = {}
    protocol_bad = list(protocol_contradictions)
    inc_reasons = list(incomplete_reasons)
    for ep_idx, ep_rec in enumerate(phase_one_records):
        sched_ep = expected_sched[ep_idx]
        if (
            ep_rec.get("episode_index") != ep_idx
            or ep_rec.get("task_id") != sched_ep["task_id"]
            or ep_rec.get("arm") != sched_ep["arm"]
        ):
            protocol_bad.append("schedule_identity_mismatch")

    if len(phase_one_records) != 96:
        inc_reasons.append("incomplete_phase_one_episodes")
    if len(final_bundle_records) != len(bundles_plan["batches"]):
        inc_reasons.append("incomplete_final_scoring_batches")
    if training_summary is None:
        inc_reasons.append("missing_lora_training_summary")

    final_exec_ns = 0
    for b_meta, b_rec in zip(bundles_plan["batches"], final_bundle_records):
        batch_cont = cont_by_batch.pop(b_meta["batch_index"], [])
        obs_map, full_ok, partial_ok, proto_err = _decode_scoring_batch(
            b_meta, b_rec, expected_runtime
        )
        if proto_err:
            protocol_bad.append("scoring_batch_protocol_mismatch")
            inc_reasons.append("scoring_batch_incomplete")
            if batch_cont:
                protocol_bad.append("unexpected_continuation_record")
            continue
        for pos_idx, val in obs_map.items():
            pos_observations[pos_idx] = val
            final_exec_ns += val["elapsed_ns"]
        if full_ok:
            if batch_cont:
                protocol_bad.append("unexpected_continuation_record")
            continue
        if not partial_ok:
            inc_reasons.append("scoring_batch_incomplete")
            if batch_cont:
                protocol_bad.append("unexpected_continuation_record")
            continue

        rem_slice = b_meta["positions"][len(obs_map):]
        settled = False
        for exp_sub, crec in enumerate(batch_cont, start=1):
            if not rem_slice:
                protocol_bad.append("excess_continuation_record")
                break
            exp_cont = build_confirmatory_continuation_bundle(
                worker_raw, run_id, b_meta["batch_index"], exp_sub, rem_slice
            )
            if (
                crec.get("sub_index") != exp_sub
                or crec.get("run_id") != exp_cont["run_id"]
                or crec.get("position_indices") != [pos["position_index"] for pos in rem_slice]
            ):
                protocol_bad.append("continuation_binding_mismatch")
                break
            c_obs, c_full, c_part, c_proto = _decode_scoring_batch(
                exp_cont,
                crec,
                expected_runtime,
                check_fields=(
                    "batch_index", "sub_index", "run_id", "position_indices", "bundle_raw", "transport"
                ),
            )
            if c_proto:
                protocol_bad.append("continuation_protocol_mismatch")
                break
            for pos_idx, val in c_obs.items():
                pos_observations[pos_idx] = val
                final_exec_ns += val["elapsed_ns"]
            if c_full:
                rem_slice = []
                settled = True
            elif c_part:
                rem_slice = rem_slice[len(c_obs):]
            else:
                break
        if not settled:
            inc_reasons.append("scoring_batch_incomplete")

    if cont_by_batch:
        protocol_bad.append("orphan_continuation_record")

    all_pos = bundles_plan["all_positions"]
    arm_costs = {
        arm_name: {
            "generation_calls": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "forward_calls": 0,
            "generation_wall_ns": 0,
            "initial_wall_ns": 0,
            "feedback_wall_ns": 0,
            "public_execution_ns": 0,
            "max_peak_allocated_bytes": 0,
            "max_peak_reserved_bytes": 0,
        }
        for arm_name in ARMS
    }

    raw_eps = []
    for ep_idx, ep_rec in enumerate(phase_one_records):
        tid = ep_rec["task_id"]
        arm = ep_rec["arm"]
        try:
            validated_pub_fb = validate_public_feedback(ep_rec.get("public_feedback"))
        except Exception:
            validated_pub_fb = copy.deepcopy(ep_rec.get("public_feedback"))
            protocol_bad.append("invalid_public_feedback_schema")

        roles_out = {}
        for r_idx, role_name in enumerate(CONFIRMATORY_ROLES):
            base_pos = (ep_idx * 16) + (r_idx * 8)
            slice_pos = all_pos[base_pos:base_pos + 8]
            valid_src = slice_pos[0]["valid_source"]
            case_recs = []
            for p_item in slice_pos:
                obs = pos_observations.get(p_item["position_index"])
                if obs is None:
                    case_recs.append(None)
                else:
                    matched = (
                        obs["status"] == "completed"
                        and obs["outcome"] is not None
                        and encoded(obs["outcome"]) == encoded(p_item["expected"])
                    )
                    case_recs.append({
                        "status": obs["status"],
                        "process_reaped": obs["process_reaped"],
                        "matched": matched,
                    })
            classified = _classify_candidate(valid_src, case_recs)
            if classified["state"] == "UNKNOWN" or classified["has_unknown_case"]:
                inc_reasons.append("unknown_candidate_outcome")
            if role_name == "initial" and isinstance(validated_pub_fb, dict):
                if (validated_pub_fb.get("kind") == "invalid_source") != (not valid_src):
                    protocol_bad.append("public_feedback_validity_mismatch")
                elif valid_src and validated_pub_fb.get("kind") == "visible_tests":
                    for p1_st, p2_st in zip(
                        [c_it["status"] for c_it in validated_pub_fb["cases"]],
                        classified["case_statuses"][:2],
                    ):
                        if p2_st != "unknown" and {p1_st, p2_st} == {"pass", "mismatch"}:
                            protocol_bad.append("phase1_phase2_public_semantic_contradiction")
            roles_out[role_name] = classified

            gcost = ep_rec["generation_costs"][role_name]
            arm_costs[arm]["generation_calls"] += 1
            arm_costs[arm]["input_tokens"] += gcost["input_tokens"]
            arm_costs[arm]["output_tokens"] += gcost["output_tokens"]
            arm_costs[arm]["forward_calls"] += gcost["forward_calls"]
            arm_costs[arm]["generation_wall_ns"] += gcost["wall_ns"]
            arm_costs[arm][role_name + "_wall_ns"] += gcost["wall_ns"]
            if type(gcost.get("peak_allocated_bytes")) is int:
                arm_costs[arm]["max_peak_allocated_bytes"] = max(
                    arm_costs[arm]["max_peak_allocated_bytes"], gcost["peak_allocated_bytes"]
                )
            if type(gcost.get("peak_reserved_bytes")) is int:
                arm_costs[arm]["max_peak_reserved_bytes"] = max(
                    arm_costs[arm]["max_peak_reserved_bytes"], gcost["peak_reserved_bytes"]
                )
        arm_costs[arm]["public_execution_ns"] += ep_rec["public_execution_ns"]
        raw_eps.append((ep_idx, tid, ep_rec["split"], arm, validated_pub_fb, roles_out))

    episodes_out = []
    indexed_eps = {}
    for ep_idx, tid, sp_name, arm, validated_pub_fb, roles_out in raw_eps:
        if protocol_bad:
            for role_name in CONFIRMATORY_ROLES:
                roles_out[role_name]["state"] = "INVALID_EVIDENCE"
        ep_entry = {
            "episode_index": ep_idx,
            "task_id": tid,
            "split": sp_name,
            "arm": arm,
            "public_feedback": validated_pub_fb,
            "roles": roles_out,
        }
        episodes_out.append(ep_entry)
        indexed_eps[(tid, arm)] = ep_entry

    run_complete = bool(not protocol_bad and not inc_reasons)
    by_split = {}
    for sp_name, sp_ids in (("transfer", TRANSFER_IDS), ("retention", RETENTION_IDS)):
        by_split[sp_name] = {}
        for arm in ARMS:
            role_counts = {
                role_name: {st_name: 0 for st_name in CANDIDATE_STATES}
                for role_name in CONFIRMATORY_ROLES
            }
            for tid in sp_ids:
                ep_obj = indexed_eps.get((tid, arm))
                if ep_obj is not None:
                    for role_name in CONFIRMATORY_ROLES:
                        role_counts[role_name][ep_obj["roles"][role_name]["state"]] += 1
            by_split[sp_name][arm] = {
                "planned_tasks": 12,
                "initial": role_counts["initial"],
                "feedback": role_counts["feedback"],
            }

    comparisons = {}
    for role_name in CONFIRMATORY_ROLES:
        comparisons[role_name] = {}
        for sp_name, sp_ids in (("transfer", TRANSFER_IDS), ("retention", RETENTION_IDS)):
            comparisons[role_name][sp_name] = {
                "lora_text_vs_text": _paired_arm_comparison(
                    sp_ids, indexed_eps, role_name, "text", "lora_text", run_complete
                ),
                "lora_vs_text": _paired_arm_comparison(
                    sp_ids, indexed_eps, role_name, "text", "lora", run_complete
                ),
                "lora_vs_base": _paired_arm_comparison(
                    sp_ids, indexed_eps, role_name, "base", "lora", run_complete
                ),
                "text_vs_base": _paired_arm_comparison(
                    sp_ids, indexed_eps, role_name, "base", "text", run_complete
                ),
            }

    amortization = None
    if training_summary is not None and run_complete:
        fixed_ns = (
            training_summary["core_wall_ns"]
            + training_summary["adapter_save_ns"]
            + training_summary["adapter_load_ns"]
        )
        text_total_ep_ns = arm_costs["text"]["generation_wall_ns"] + arm_costs["text"]["public_execution_ns"]
        lora_total_ep_ns = arm_costs["lora"]["generation_wall_ns"] + arm_costs["lora"]["public_execution_ns"]
        per_ep_saved_ns = (text_total_ep_ns - lora_total_ep_ns) / 24.0
        text_r1_ep_ns = arm_costs["text"]["initial_wall_ns"] / 24.0
        lora_r1_ep_ns = arm_costs["lora"]["initial_wall_ns"] / 24.0
        per_r1_saved_ns = text_r1_ep_ns - lora_r1_ep_ns
        amortization = {
            "lora_fixed_overhead_ns": fixed_ns,
            "per_episode_saved_ns_lora_vs_text": per_ep_saved_ns,
            "break_even_episodes_lora_vs_text": (fixed_ns / per_ep_saved_ns) if per_ep_saved_ns > 0 else None,
            "per_single_turn_saved_ns_lora_vs_text": per_r1_saved_ns,
            "break_even_single_turns_lora_vs_text": (fixed_ns / per_r1_saved_ns) if per_r1_saved_ns > 0 else None,
            "text_to_lora_generation_wall_ratio": (
                arm_costs["text"]["generation_wall_ns"] / arm_costs["lora"]["generation_wall_ns"]
                if arm_costs["lora"]["generation_wall_ns"] > 0
                else None
            ),
        }

    if protocol_bad:
        verdict_status = "INVALID_PROTOCOL"
    elif inc_reasons:
        verdict_status = "INCONCLUSIVE_INCOMPLETE"
    else:
        fb_tr_matched = comparisons["feedback"]["transfer"]["lora_text_vs_text"]
        fb_ret_matched = comparisons["feedback"]["retention"]["lora_text_vs_text"]
        fb_tr_comp = comparisons["feedback"]["transfer"]["lora_vs_text"]
        fb_ret_base = comparisons["feedback"]["retention"]["lora_vs_base"]
        if fb_ret_matched["right_to_wrong"] > 0:
            verdict_status = "OBSERVED_RETENTION_REGRESSION"
        elif fb_tr_matched["net_gain"] is not None and fb_tr_matched["net_gain"] >= 2:
            verdict_status = "POSITIVE_MATCHED_LORA_GAIN"
        elif (
            fb_tr_comp["net_gain"] is not None
            and fb_tr_comp["net_gain"] >= 0
            and fb_ret_base["right_to_wrong"] == 0
            and arm_costs["lora"]["generation_wall_ns"] * 2 < arm_costs["text"]["generation_wall_ns"]
        ):
            verdict_status = "COMPRESSION_SPEEDUP_WITHOUT_LORA_TEXT_GAIN"
        else:
            verdict_status = "NO_LORA_GAIN_OBSERVED"

    return {
        "kind": "LORA_CONFIRMATORY_V2_REPORT",
        "version": 1,
        "data_version": "confirmatory_v2_r1",
        "run_id": run_id,
        "file_sha256": dict(CONFIRMATORY_FILE_PINS),
        "verdict": {
            "status": verdict_status,
            "protocol_contradictions": sorted(set(protocol_bad)),
            "incomplete_reasons": sorted(set(inc_reasons)),
        },
        "training": copy.deepcopy(training_summary),
        "by_split": by_split,
        "comparisons": comparisons,
        "episodes": episodes_out,
        "costs": {
            "by_arm": arm_costs,
            "amortization": amortization,
            "final_scoring_execution_ns": final_exec_ns,
            "shared": copy.deepcopy(shared_costs),
        },
        "run_authorized": False,
        "trusted_collection_accepted": False,
        **{flag: False for flag in FLAGS},
    }
