import ast
from collections import Counter
import copy
import hashlib
import re

from pilot_protocol import FLAGS, fields, require
from recalibration_data_contract import CASE_IDS, FILE_PINS, TASK_IDS, _outcome, _plain, load_dataset
from recalibration_rendering import EXAMPLES_SHA256, REVISION, first_round
from recalibration_validation import WORKER_SHA256
from validation_contract import digest, encoded, parse_json
from validation_real_report import _events, _process_result
from validation_report import parse_outcome


CANDIDATE_ROLES = ("initial", "feedback", "retry")
CANDIDATE_STATES = ("PASS", "SEMANTIC_FAIL", "EXECUTION_FAIL", "UNKNOWN", "INVALID_EVIDENCE")
PUBLIC_FEEDBACK_STATUSES = ("pass", "mismatch", "program_error", "timeout", "output_limit")
CALIBRATION_STATUSES = (
    "INVALID_PROTOCOL",
    "INCONCLUSIVE_INCOMPLETE",
    "INCONCLUSIVE_CALIBRATION",
    "DEVELOPMENT_DIAGNOSTIC_USABLE",
)


def _check_hex(text, length=64):
    require(
        type(text) is str and re.fullmatch("[0-9a-f]{%d}" % length, text) is not None,
        "invalid hex identifier",
    )


def _load_prompts_only(prompts_raw, supervision_raw):
    require(
        type(prompts_raw) is bytes
        and hashlib.sha256(prompts_raw).hexdigest() == FILE_PINS["prompts.json"],
        "prompts.json digest mismatch",
    )
    require(
        type(supervision_raw) is bytes
        and hashlib.sha256(supervision_raw).hexdigest() == FILE_PINS["supervision_v1.json"],
        "supervision_v1.json digest mismatch",
    )
    prompts = parse_json(prompts_raw)
    require(
        type(prompts) is list
        and len(prompts) == 12
        and tuple(item["id"] for item in prompts) == TASK_IDS,
        "prompts list order",
    )
    return prompts


def candidate_valid(task, candidate):
    if type(candidate) is not str:
        return False
    try:
        raw_utf8 = candidate.encode("utf-8")
    except UnicodeError:
        return False
    if len(raw_utf8) == 0 or len(raw_utf8) > 4096 or len(encoded(candidate)) > 3584:
        return False
    try:
        tree = ast.parse(candidate, mode="exec")
        if len(tree.body) != 1 or type(tree.body[0]) is not ast.FunctionDef:
            return False
        func_node = tree.body[0]
        initial_fn = ast.parse(task["initial_source"]).body[0]
        if (
            func_node.name != task["entry_point"]
            or func_node.decorator_list
            or func_node.returns is not None
            or getattr(func_node, "type_params", [])
            or ast.dump(func_node.args) != ast.dump(initial_fn.args)
        ):
            return False
        for subnode in ast.walk(tree):
            if isinstance(subnode, (ast.Import, ast.ImportFrom, ast.AsyncFunctionDef, ast.ClassDef)):
                return False
            if isinstance(subnode, ast.FunctionDef) and subnode is not func_node:
                return False
            if isinstance(subnode, ast.arg) and subnode.annotation is not None:
                return False
        compile(tree, "<candidate-syntax-only>", "exec", dont_inherit=True)
    except (SyntaxError, ValueError, RecursionError):
        return False
    return True


def build_generation_schedule(prompts_raw, supervision_raw, *, run_id):
    _check_hex(run_id, 32)
    prompts = _load_prompts_only(prompts_raw, supervision_raw)
    episodes = []
    for task_index, task in enumerate(prompts):
        arms = ("base", "text") if task_index % 2 == 0 else ("text", "base")
        for arm in arms:
            feedback_first = (task_index % 4 < 2) == (arm == "base")
            branch_order = ["feedback", "retry"] if feedback_first else ["retry", "feedback"]
            first_msgs = first_round(task, arm, supervision_raw)
            ep_idx = len(episodes)
            episodes.append({
                "episode_index": ep_idx,
                "task_id": task["id"],
                "entry_point": task["entry_point"],
                "arm": arm,
                "branch_order": branch_order,
                "first_messages": first_msgs,
                "first_messages_sha256": digest(first_msgs),
                "examples_sha256": EXAMPLES_SHA256 if arm == "text" else None,
                "call_ids": {
                    role_name: hashlib.sha256(
                        encoded(["diag_call_id", run_id, ep_idx, task["id"], arm, role_name])
                    ).hexdigest()[:32]
                    for role_name in ("initial", branch_order[0], branch_order[1])
                },
            })
    require(len(episodes) == 24, "24 episodes required")
    return {
        "kind": "LORA_RECALIBRATION_DIAGNOSTIC_SCHEDULE",
        "version": 1,
        "data_version": "recalibration_v2_r1",
        "run_id": run_id,
        "prompts_sha256": FILE_PINS["prompts.json"],
        "supervision_sha256": FILE_PINS["supervision_v1.json"],
        "scheduled_episodes": 24,
        "scheduled_generation_calls": 72,
        "episodes": episodes,
    }


def validate_public_feedback(feedback):
    _plain(feedback)
    require(type(feedback) is dict, "public feedback dict")
    if feedback.get("kind") == "invalid_source":
        fields(feedback, ("kind",))
        return copy.deepcopy(feedback)
    fields(feedback, ("kind", "cases"))
    require(feedback["kind"] == "visible_tests", "public feedback kind")
    cases = feedback["cases"]
    require(type(cases) is list and len(cases) == 2, "2 public feedback cases")
    for case_item, expected_id in zip(cases, CASE_IDS[:2]):
        fields(case_item, ("id", "status"))
        require(
            case_item["id"] == expected_id
            and type(case_item["status"]) is str
            and case_item["status"] in PUBLIC_FEEDBACK_STATUSES,
            "public feedback case status",
        )
    return copy.deepcopy(feedback)


def continuation_messages(task, arm, supervision_raw, initial_candidate, branch, *, feedback=None):
    require(type(initial_candidate) is str, "initial candidate text")
    raw_cand = initial_candidate.encode("utf-8")
    require(len(raw_cand) <= 65536, "initial candidate byte cap")
    require(branch in ("feedback", "retry"), "branch name")
    if branch == "retry":
        require(feedback is None, "retry must not receive feedback")
        payload = {"kind": "results_withheld"}
    else:
        payload = validate_public_feedback(feedback)
    base_msgs = first_round(task, arm, supervision_raw)
    return base_msgs + [
        {"role": "assistant", "content": initial_candidate},
        {"role": "user", "content": REVISION + "\n" + encoded(payload).decode("ascii")},
    ]


def plan_final_positions(raw_files, phase_one_records):
    dataset = load_dataset(raw_files)
    prompts_map = {item["id"]: item for item in dataset["prompts"]}
    oracles_map = {item["id"]: item for item in dataset["oracles"]}
    require(type(phase_one_records) is list and len(phase_one_records) <= 24, "phase one records")
    positions = []
    valid_positions = []
    for ep_idx, ep_rec in enumerate(phase_one_records):
        require(ep_rec["episode_index"] == ep_idx, "episode order")
        tid = ep_rec["task_id"]
        arm = ep_rec["arm"]
        task_prompt = prompts_map[tid]
        oracle = oracles_map[tid]
        for role_name in CANDIDATE_ROLES:
            cand_src = ep_rec["candidates"][role_name]
            is_valid = candidate_valid(task_prompt, cand_src)
            for case_obj in oracle["cases"]:
                pos_index = len(positions)
                req_obj = None
                req_sha = None
                if is_valid:
                    req_obj = {
                        "request_id": "diag-%04d" % len(valid_positions),
                        "source": cand_src,
                        "entry_point": task_prompt["entry_point"],
                        "args": copy.deepcopy(case_obj["args"]),
                    }
                    req_sha = digest(req_obj)
                entry = {
                    "position_index": pos_index,
                    "episode_index": ep_idx,
                    "task_id": tid,
                    "arm": arm,
                    "role": role_name,
                    "case_id": case_obj["id"],
                    "is_public": case_obj["id"] in CASE_IDS[:2],
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


def build_final_scoring_bundles(raw_files, worker_raw, run_id, phase_one_records, scoring_run_ids):
    _check_hex(run_id, 32)
    require(
        type(worker_raw) is bytes and hashlib.sha256(worker_raw).hexdigest() == WORKER_SHA256,
        "worker source pin",
    )
    worker_text = worker_raw.decode("utf-8")
    planned = plan_final_positions(raw_files, phase_one_records)
    batch_slices = planned["batches"]
    require(
        type(scoring_run_ids) is list and len(scoring_run_ids) == len(batch_slices),
        "scoring_run_ids length mismatch",
    )
    seen_ids = {run_id}
    built_batches = []
    for batch_idx, (batch_pos_list, b_run_id) in enumerate(zip(batch_slices, scoring_run_ids)):
        _check_hex(b_run_id, 32)
        require(b_run_id not in seen_ids, "duplicate scoring run_id")
        seen_ids.add(b_run_id)
        bundle_obj = {
            "kind": "LORA_REMOTE_VALIDATION_BUNDLE",
            "mode": "matrix",
            "run_id": b_run_id,
            "worker_source": worker_text,
            "requests": [pos["request"] for pos in batch_pos_list],
        }
        bundle_raw = encoded(bundle_obj)
        built_batches.append({
            "batch_index": batch_idx,
            "run_id": b_run_id,
            "bundle_raw": bundle_raw,
            "bundle_sha256": hashlib.sha256(bundle_raw).hexdigest(),
            "worker_sha256": WORKER_SHA256,
            "positions": batch_pos_list,
        })
    return {
        "run_ids": list(scoring_run_ids),
        "batches": built_batches,
        "all_positions": planned["all_positions"],
    }


def build_continuation_bundle(worker_raw, run_id, batch_index, sub_index, positions_slice):
    _check_hex(run_id, 32)
    require(
        type(worker_raw) is bytes and hashlib.sha256(worker_raw).hexdigest() == WORKER_SHA256,
        "worker source pin",
    )
    require(type(batch_index) is int and 0 <= batch_index < 18, "batch_index bounds")
    require(type(sub_index) is int and 1 <= sub_index <= 32, "sub_index bounds")
    require(type(positions_slice) is list and 1 <= len(positions_slice) <= 32, "positions_slice bounds")
    cont_run_id = hashlib.sha256(
        encoded(["diag_p2_cont", run_id, batch_index, sub_index])
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


def _decode_scoring_batch(batch_meta, record, expected_runtime, *, check_fields=("batch_index", "bundle_raw", "transport")):
    fields(record, check_fields)
    require(record["batch_index"] == batch_meta["batch_index"], "scoring batch index")
    require(record["bundle_raw"] == batch_meta["bundle_raw"], "scoring bundle bytes")
    if record["transport"] is None:
        return {}, False, False, False
    stdout_raw = _process_result(record["transport"], outer=True)
    timed_out = record["transport"]["status"] in ("timeout", "output_limit")
    events, truncated = _events(stdout_raw, timed_out=timed_out)
    observed_map = {}
    observed_list = []
    ready_seen = False
    done_seen = False
    done_stopped = False
    expected_positions = batch_meta["positions"]
    for ev in events:
        if ev.get("run_id") != batch_meta["run_id"]:
            return {}, False, False, True
        kind = ev.get("kind")
        if kind == "ready":
            if ready_seen or observed_map:
                return {}, False, False, True
            if (
                ev.get("bundle_sha256") != batch_meta["bundle_sha256"]
                or ev.get("worker_sha256") != WORKER_SHA256
                or encoded(ev.get("runtime")) != encoded(expected_runtime)
            ):
                return {}, False, False, True
            ready_seen = True
        elif kind == "row":
            if not ready_seen or done_seen:
                return {}, False, False, True
            ord_idx = ev.get("ordinal")
            if type(ord_idx) is not int or ord_idx != len(observed_map) or ord_idx >= len(expected_positions):
                return {}, False, False, True
            pos_meta = expected_positions[ord_idx]
            if ev.get("request_sha256") != pos_meta["request_sha256"]:
                return {}, False, False, True
            row_res = ev["result"]
            raw_out = _process_result(row_res)
            parsed_out = parse_outcome(raw_out) if row_res["status"] == "completed" else None
            obs_item = {
                "status": row_res["status"],
                "process_reaped": row_res["process_reaped"],
                "outcome": parsed_out,
                "elapsed_ns": row_res["elapsed_ns"],
            }
            observed_map[pos_meta["position_index"]] = obs_item
            observed_list.append(obs_item)
        elif kind == "done":
            if not ready_seen or done_seen:
                return {}, False, False, True
            if ev.get("received") != len(observed_list) or type(ev.get("stopped")) is not bool:
                return {}, False, False, True
            done_stopped = ev["stopped"]
            done_seen = True
        else:
            return {}, False, False, True
    transport_ok = (
        ready_seen
        and done_seen
        and not truncated
        and record["transport"]["status"] == "completed"
    )
    full_complete = (
        transport_ok
        and len(observed_list) == len(expected_positions)
        and all(item["process_reaped"] is True for item in observed_list)
    )
    partial_stopped = (
        transport_ok
        and done_stopped
        and 1 <= len(observed_list) < len(expected_positions)
        and all(item["process_reaped"] is True for item in observed_list)
        and observed_list[-1]["status"] in ("program_error", "timeout", "output_limit")
    )
    return observed_map, full_complete, partial_stopped, False


def _classify_candidate(valid_source, case_records):
    if not valid_source:
        return {
            "state": "EXECUTION_FAIL",
            "case_statuses": ["not_executed_invalid_source"] * 8,
            "public_passed": 0,
            "private_passed": 0,
            "all_8_normal_completed": False,
            "has_concurrent_exec_error": True,
            "has_unknown_case": False,
            "public_both_normal_pass": False,
            "public_revealed_mismatch": False,
        }
    case_statuses = []
    has_mismatch = False
    has_exec_err = False
    has_unknown = False
    pub_pass = 0
    priv_pass = 0
    for idx, crec in enumerate(case_records):
        if crec is None or not crec["process_reaped"]:
            case_statuses.append("unknown")
            has_unknown = True
        elif crec["status"] != "completed":
            case_statuses.append(crec["status"])
            has_exec_err = True
        elif crec["matched"]:
            case_statuses.append("pass")
            if idx < 2:
                pub_pass += 1
            else:
                priv_pass += 1
        else:
            case_statuses.append("mismatch")
            has_mismatch = True

    if has_mismatch:
        state_val = "SEMANTIC_FAIL"
    elif has_exec_err:
        state_val = "EXECUTION_FAIL"
    elif has_unknown or (pub_pass + priv_pass < 8):
        state_val = "UNKNOWN"
    else:
        state_val = "PASS"

    all_normal = all(st in ("pass", "mismatch") for st in case_statuses)
    pub_both_pass = case_statuses[0] == "pass" and case_statuses[1] == "pass"
    pub_mismatch = ("mismatch" in case_statuses[:2])
    return {
        "state": state_val,
        "case_statuses": case_statuses,
        "public_passed": pub_pass,
        "private_passed": priv_pass,
        "all_8_normal_completed": all_normal,
        "has_concurrent_exec_error": has_exec_err,
        "has_unknown_case": has_unknown,
        "public_both_normal_pass": pub_both_pass,
        "public_revealed_mismatch": pub_mismatch,
    }


def _arm_summary(episodes_for_arm, family_map, *, run_complete):
    role_counts = {
        role: {st: 0 for st in CANDIDATE_STATES}
        for role in CANDIDATE_ROLES
    }
    for ep in episodes_for_arm:
        for role in CANDIDATE_ROLES:
            role_counts[role][ep["roles"][role]["state"]] += 1

    transitions = {}
    for branch in ("feedback", "retry"):
        sem_pairs = []
        excluded = Counter()
        cross_5x5 = {}
        for ep in episodes_for_arm:
            init_info = ep["roles"]["initial"]
            br_info = ep["roles"][branch]
            pair_key = init_info["state"] + "->" + br_info["state"]
            cross_5x5[pair_key] = cross_5x5.get(pair_key, 0) + 1
            if (
                init_info["state"] in ("PASS", "SEMANTIC_FAIL")
                and br_info["state"] in ("PASS", "SEMANTIC_FAIL")
                and init_info["all_8_normal_completed"]
                and br_info["all_8_normal_completed"]
                and not init_info["has_concurrent_exec_error"]
                and not br_info["has_concurrent_exec_error"]
            ):
                sem_pairs.append((init_info["state"], br_info["state"]))
            else:
                excluded[pair_key] += 1
        w2r = sum(first_st == "SEMANTIC_FAIL" and second_st == "PASS" for first_st, second_st in sem_pairs)
        r2w = sum(first_st == "PASS" and second_st == "SEMANTIC_FAIL" for first_st, second_st in sem_pairs)
        r2r = sum(first_st == "PASS" and second_st == "PASS" for first_st, second_st in sem_pairs)
        w2w = sum(first_st == "SEMANTIC_FAIL" and second_st == "SEMANTIC_FAIL" for first_st, second_st in sem_pairs)
        init_wrong = w2r + w2w
        init_right = r2w + r2r
        transitions["initial_to_" + branch] = {
            "joint_semantic_tasks": len(sem_pairs),
            "wrong_to_right": w2r,
            "right_to_wrong": r2w,
            "right_to_right": r2r,
            "wrong_to_wrong": w2w,
            "correction_rate": (w2r / init_wrong) if init_wrong > 0 else None,
            "regression_rate": (r2w / init_right) if init_right > 0 else None,
            "excluded_by_state": dict(excluded),
            "full_5_state_cross": cross_5x5,
        }

    fb_wins_sem = 0
    fb_wins_exec = 0
    rt_wins_sem = 0
    rt_wins_exec = 0
    known_joint = 0
    for ep in episodes_for_arm:
        fb_st = ep["roles"]["feedback"]["state"]
        rt_st = ep["roles"]["retry"]["state"]
        if (
            fb_st in ("PASS", "SEMANTIC_FAIL", "EXECUTION_FAIL")
            and rt_st in ("PASS", "SEMANTIC_FAIL", "EXECUTION_FAIL")
            and not ep["roles"]["feedback"]["has_unknown_case"]
            and not ep["roles"]["retry"]["has_unknown_case"]
        ):
            known_joint += 1
            if fb_st == "PASS" and rt_st == "SEMANTIC_FAIL":
                fb_wins_sem += 1
            elif fb_st == "PASS" and rt_st == "EXECUTION_FAIL":
                fb_wins_exec += 1
            elif rt_st == "PASS" and fb_st == "SEMANTIC_FAIL":
                rt_wins_sem += 1
            elif rt_st == "PASS" and fb_st == "EXECUTION_FAIL":
                rt_wins_exec += 1

    fb_wins_total = fb_wins_sem + fb_wins_exec
    rt_wins_total = rt_wins_sem + rt_wins_exec
    subset_net = fb_wins_total - rt_wins_total
    full_net = (
        subset_net
        if (run_complete and known_joint == 12 and len(episodes_for_arm) == 12)
        else None
    )

    qualifying_sem_fail = 0
    pub_revealed = 0
    priv_only = 0
    pub_exec_discrepant = 0
    revealed_mechanisms = set()
    by_mechanism = {fam: {"PASS": 0, "SEMANTIC_FAIL": 0, "OTHER": 0} for fam in ("bounds", "reset", "multiplicity", "order")}
    for ep in episodes_for_arm:
        init_info = ep["roles"]["initial"]
        pub_fb = ep["public_feedback"]
        p1_cases = pub_fb.get("cases", []) if isinstance(pub_fb, dict) and pub_fb.get("kind") == "visible_tests" else []
        p1_statuses = [c_entry.get("status") for c_entry in p1_cases]
        p1_both_pass = len(p1_statuses) == 2 and p1_statuses[0] == "pass" and p1_statuses[1] == "pass"
        p1_has_mismatch = "mismatch" in p1_statuses
        p1_all_normal = len(p1_statuses) == 2 and all(st in ("pass", "mismatch") for st in p1_statuses)
        fam = family_map[ep["task_id"]]
        if init_info["state"] == "PASS" and p1_both_pass:
            by_mechanism[fam]["PASS"] += 1
        elif init_info["state"] == "SEMANTIC_FAIL" and init_info["all_8_normal_completed"] and p1_all_normal:
            qualifying_sem_fail += 1
            by_mechanism[fam]["SEMANTIC_FAIL"] += 1
            if init_info["public_revealed_mismatch"] and p1_has_mismatch:
                pub_revealed += 1
                revealed_mechanisms.add(fam)
            elif init_info["public_both_normal_pass"] and p1_both_pass:
                priv_only += 1
        elif init_info["state"] == "SEMANTIC_FAIL" and init_info["all_8_normal_completed"]:
            pub_exec_discrepant += 1
            by_mechanism[fam]["OTHER"] += 1
        else:
            by_mechanism[fam]["OTHER"] += 1

    return {
        "planned_tasks": 12,
        "observed_episodes": len(episodes_for_arm),
        "candidates": role_counts,
        "transitions": transitions,
        "feedback_vs_retry": {
            "joint_known_tasks": known_joint,
            "feedback_pass_retry_semantic_fail": fb_wins_sem,
            "feedback_pass_retry_execution_fail": fb_wins_exec,
            "retry_pass_feedback_semantic_fail": rt_wins_sem,
            "retry_pass_feedback_execution_fail": rt_wins_exec,
            "observed_subset_net_count": subset_net,
            "net_gain": full_net,
        },
        "initial_exposure": {
            "qualifying_semantic_fail_tasks": qualifying_sem_fail,
            "public_revealed_semantic_fail": pub_revealed,
            "private_only_semantic_fail": priv_only,
            "public_exec_discrepant_semantic_fail": pub_exec_discrepant,
            "public_revealed_mechanisms": sorted(revealed_mechanisms),
        },
        "initial_by_mechanism": by_mechanism,
    }


def score_diagnostic_run(
    raw_files,
    worker_raw,
    run_id,
    phase_one_records,
    scoring_run_ids,
    final_bundle_records,
    *,
    continuation_records=(),
    expected_runtime,
    protocol_contradictions=(),
    incomplete_reasons=(),
    shared_costs=None,
):
    dataset = load_dataset(raw_files)
    family_map = {item["id"]: item["family"] for item in dataset["structure"]["tasks"]}
    expected_schedule = build_generation_schedule(
        raw_files["prompts.json"], raw_files["supervision_v1.json"], run_id=run_id
    )["episodes"]
    bundles_plan = build_final_scoring_bundles(
        raw_files, worker_raw, run_id, phase_one_records, scoring_run_ids
    )
    require(len(final_bundle_records) <= len(bundles_plan["batches"]), "excess bundle records")

    cont_by_batch = {}
    for crec in continuation_records:
        b_idx_val = crec["batch_index"]
        cont_by_batch.setdefault(b_idx_val, []).append(crec)

    pos_observations = {}
    protocol_bad = list(protocol_contradictions)
    inc_reasons = list(incomplete_reasons)
    for ep_idx, ep_rec in enumerate(phase_one_records):
        sched_ep = expected_schedule[ep_idx]
        if (
            ep_rec.get("episode_index") != ep_idx
            or ep_rec.get("task_id") != sched_ep["task_id"]
            or ep_rec.get("arm") != sched_ep["arm"]
        ):
            protocol_bad.append("schedule_identity_mismatch")

    if len(phase_one_records) != 24:
        inc_reasons.append("incomplete_phase_one_episodes")
    if len(final_bundle_records) != len(bundles_plan["batches"]):
        inc_reasons.append("incomplete_final_scoring_batches")

    final_exec_ns = 0
    for b_meta, b_rec in zip(bundles_plan["batches"], final_bundle_records):
        batch_cont_list = cont_by_batch.pop(b_meta["batch_index"], [])
        obs_map, full_ok, partial_ok, proto_err = _decode_scoring_batch(
            b_meta, b_rec, expected_runtime
        )
        if proto_err:
            protocol_bad.append("scoring_batch_protocol_mismatch")
            inc_reasons.append("scoring_batch_incomplete")
            if batch_cont_list:
                protocol_bad.append("unexpected_continuation_record")
            continue
        for pos_idx, val in obs_map.items():
            pos_observations[pos_idx] = val
            final_exec_ns += val["elapsed_ns"]

        if full_ok:
            if batch_cont_list:
                protocol_bad.append("unexpected_continuation_record")
            continue
        if not partial_ok:
            inc_reasons.append("scoring_batch_incomplete")
            if batch_cont_list:
                protocol_bad.append("unexpected_continuation_record")
            continue

        remaining_slice = b_meta["positions"][len(obs_map):]
        batch_settled = False
        for expected_sub_idx, crec in enumerate(batch_cont_list, start=1):
            if not remaining_slice:
                protocol_bad.append("excess_continuation_record")
                break
            expected_cont = build_continuation_bundle(
                worker_raw, run_id, b_meta["batch_index"], expected_sub_idx, remaining_slice
            )
            if (
                crec.get("sub_index") != expected_sub_idx
                or crec.get("run_id") != expected_cont["run_id"]
                or crec.get("position_indices") != [pos["position_index"] for pos in remaining_slice]
            ):
                protocol_bad.append("continuation_binding_mismatch")
                break
            c_obs, c_full_ok, c_partial_ok, c_proto_err = _decode_scoring_batch(
                expected_cont,
                crec,
                expected_runtime,
                check_fields=(
                    "batch_index", "sub_index", "run_id", "position_indices", "bundle_raw", "transport"
                ),
            )
            if c_proto_err:
                protocol_bad.append("continuation_protocol_mismatch")
                break
            for pos_idx, val in c_obs.items():
                pos_observations[pos_idx] = val
                final_exec_ns += val["elapsed_ns"]
            if c_full_ok:
                remaining_slice = []
                batch_settled = True
            elif c_partial_ok:
                remaining_slice = remaining_slice[len(c_obs):]
            else:
                break
        if not batch_settled:
            inc_reasons.append("scoring_batch_incomplete")

    if cont_by_batch:
        protocol_bad.append("orphan_continuation_record")

    all_pos = bundles_plan["all_positions"]
    episodes_out = []
    arm_costs = {
        arm_name: {
            "generation_calls": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "forward_calls": 0,
            "generation_wall_ns": 0,
            "public_execution_ns": 0,
            "max_peak_allocated_bytes": 0,
            "max_peak_reserved_bytes": 0,
        }
        for arm_name in ("base", "text")
    }

    raw_episode_entries = []
    for ep_idx, ep_rec in enumerate(phase_one_records):
        tid = ep_rec["task_id"]
        arm = ep_rec["arm"]
        try:
            validated_pub_fb = validate_public_feedback(ep_rec.get("public_feedback"))
        except Exception:
            validated_pub_fb = copy.deepcopy(ep_rec.get("public_feedback"))
            protocol_bad.append("invalid_public_feedback_schema")

        roles_out = {}
        for r_idx, role_name in enumerate(CANDIDATE_ROLES):
            base_pos = (ep_idx * 24) + (r_idx * 8)
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
                    p1_pub_statuses = [item["status"] for item in validated_pub_fb["cases"]]
                    p2_pub_statuses = classified["case_statuses"][:2]
                    for p1_st, p2_st in zip(p1_pub_statuses, p2_pub_statuses):
                        if p2_st == "unknown" or p1_st == p2_st:
                            continue
                        if {p1_st, p2_st} == {"pass", "mismatch"}:
                            protocol_bad.append("phase1_phase2_public_semantic_contradiction")
            roles_out[role_name] = classified

            gcost = ep_rec["generation_costs"][role_name]
            arm_costs[arm]["generation_calls"] += 1
            arm_costs[arm]["input_tokens"] += gcost["input_tokens"]
            arm_costs[arm]["output_tokens"] += gcost["output_tokens"]
            arm_costs[arm]["forward_calls"] += gcost["forward_calls"]
            arm_costs[arm]["generation_wall_ns"] += gcost["wall_ns"]
            if type(gcost.get("peak_allocated_bytes")) is int:
                arm_costs[arm]["max_peak_allocated_bytes"] = max(
                    arm_costs[arm]["max_peak_allocated_bytes"], gcost["peak_allocated_bytes"]
                )
            if type(gcost.get("peak_reserved_bytes")) is int:
                arm_costs[arm]["max_peak_reserved_bytes"] = max(
                    arm_costs[arm]["max_peak_reserved_bytes"], gcost["peak_reserved_bytes"]
                )
        arm_costs[arm]["public_execution_ns"] += ep_rec["public_execution_ns"]
        raw_episode_entries.append((ep_idx, tid, arm, validated_pub_fb, roles_out))

    for ep_idx, tid, arm, validated_pub_fb, roles_out in raw_episode_entries:
        if protocol_bad:
            for role_name in CANDIDATE_ROLES:
                roles_out[role_name]["state"] = "INVALID_EVIDENCE"
        episodes_out.append({
            "episode_index": ep_idx,
            "task_id": tid,
            "arm": arm,
            "public_feedback": validated_pub_fb,
            "roles": roles_out,
        })

    run_complete_flag = bool(not protocol_bad and not inc_reasons)
    arms_summary = {
        arm_name: _arm_summary(
            [ep for ep in episodes_out if ep["arm"] == arm_name],
            family_map,
            run_complete=run_complete_flag,
        )
        for arm_name in ("base", "text")
    }

    cal_reasons = []
    for arm_name in ("base", "text"):
        arm_s = arms_summary[arm_name]
        if arm_s["candidates"]["initial"]["PASS"] < 3:
            cal_reasons.append(arm_name + "_initial_pass_below_3")
        if arm_s["initial_exposure"]["qualifying_semantic_fail_tasks"] < 3:
            cal_reasons.append(arm_name + "_initial_semantic_fail_below_3")
        if arm_s["initial_exposure"]["public_revealed_semantic_fail"] < 2:
            cal_reasons.append(arm_name + "_public_revealed_below_2")
        if len(arm_s["initial_exposure"]["public_revealed_mechanisms"]) < 2:
            cal_reasons.append(arm_name + "_public_mechanisms_below_2")

    if protocol_bad:
        cal_status = "INVALID_PROTOCOL"
    elif inc_reasons:
        cal_status = "INCONCLUSIVE_INCOMPLETE"
    elif cal_reasons:
        cal_status = "INCONCLUSIVE_CALIBRATION"
    else:
        cal_status = "DEVELOPMENT_DIAGNOSTIC_USABLE"

    return {
        "kind": "LORA_RECALIBRATION_DIAGNOSTIC_REPORT",
        "version": 1,
        "data_version": "recalibration_v2_r1",
        "run_id": run_id,
        "file_sha256": dict(FILE_PINS),
        "calibration": {
            "status": cal_status,
            "protocol_contradictions": sorted(set(protocol_bad)),
            "incomplete_reasons": sorted(set(inc_reasons)),
            "reasons": cal_reasons,
        },
        "arms": arms_summary,
        "episodes": episodes_out,
        "costs": {
            "by_arm": arm_costs,
            "final_scoring_execution_ns": final_exec_ns,
            "shared": copy.deepcopy(shared_costs),
        },
        "run_authorized": False,
        "trusted_collection_accepted": False,
        **{flag: False for flag in FLAGS},
    }
