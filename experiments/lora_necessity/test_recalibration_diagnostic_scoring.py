import ast
import copy
import hashlib
import importlib
import io
from pathlib import Path
import re
import tokenize
import unittest
from unittest import mock

from recalibration_data_contract import CASE_IDS, FILE_PINS, TASK_IDS, load_dataset
from validation_contract import digest, encoded


REPO_LORA = Path(__file__).resolve().parent

with mock.patch("subprocess.Popen") as import_spawn, mock.patch("os.killpg") as import_killpg:
    try:
        diag_scoring = importlib.import_module("recalibration_diagnostic_scoring")
    except ModuleNotFoundError as import_err:
        if import_err.name != "recalibration_diagnostic_scoring":
            raise
        diag_scoring = None
    assert not import_spawn.called
    assert not import_killpg.called


class DiagnosticScoringTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.raw_files = {
            fname: (REPO_LORA / "data" / (
                fname if fname == "supervision_v1.json" else "recalibration_v2_r1/" + fname
            )).read_bytes()
            for fname in FILE_PINS
        }
        cls.worker_raw = (REPO_LORA / "validation_task_worker.py").read_bytes()
        cls.dataset = load_dataset(cls.raw_files)
        cls.prompts_by_id = {item["id"]: item for item in cls.dataset["prompts"]}
        cls.oracles_by_id = {item["id"]: item for item in cls.dataset["oracles"]}
        cls.run_id = "12" * 16
        cls.runtime = {
            "hostname": "target-node",
            "uid": 1054,
            "python": "3.12.13",
            "python_root": "/opt/cpython",
            "python_sha256": "2" * 64,
            "bwrap_sha256": "3" * 64,
            "seccomp_sha256": "4" * 64,
            "argv_prefix": ["/usr/bin/bwrap", "--seccomp", "3", "/runtime/bin/python3.12"],
            "no_new_privs": True,
            "host_namespaces": {"mnt": "m", "pid": "p", "net": "n", "user": "u"},
        }

    def setUp(self):
        self.assertIsNotNone(diag_scoring, "recalibration_diagnostic_scoring is absent")

    def test_style_and_no_side_effects(self):
        for fname in (
            "recalibration_diagnostic_scoring.py",
            "test_recalibration_diagnostic_scoring.py",
        ):
            raw = (REPO_LORA / fname).read_bytes()
            tree = ast.parse(raw)
            self.assertEqual(
                [node.id for node in ast.walk(tree) if isinstance(node, ast.Name) and len(node.id) == 1],
                [],
                fname,
            )
            self.assertEqual(
                [tok for tok in tokenize.tokenize(io.BytesIO(raw).readline) if tok.type == tokenize.COMMENT],
                [],
                fname,
            )

    def test_counterbalanced_schedule_72_calls_and_zero_oracle_dependency(self):
        sched = diag_scoring.build_generation_schedule(
            self.raw_files["prompts.json"],
            self.raw_files["supervision_v1.json"],
            run_id=self.run_id,
        )
        self.assertEqual(sched["scheduled_episodes"], 24)
        self.assertEqual(sched["scheduled_generation_calls"], 72)
        self.assertEqual(len(sched["episodes"]), 24)
        base_fb_first = sum(
            ep["arm"] == "base" and ep["branch_order"] == ["feedback", "retry"]
            for ep in sched["episodes"]
        )
        text_fb_first = sum(
            ep["arm"] == "text" and ep["branch_order"] == ["feedback", "retry"]
            for ep in sched["episodes"]
        )
        self.assertEqual((base_fb_first, text_fb_first), (6, 6))
        self.assertEqual(sched["episodes"][0]["task_id"], "C01")
        self.assertEqual(sched["episodes"][0]["arm"], "base")
        self.assertEqual(sched["episodes"][1]["arm"], "text")
        self.assertEqual(sched["episodes"][2]["task_id"], "C02")
        self.assertEqual(sched["episodes"][2]["arm"], "text")
        all_call_ids = [
            cid
            for ep in sched["episodes"]
            for cid in ep["call_ids"].values()
        ]
        self.assertEqual(len(all_call_ids), 72)
        self.assertEqual(len(set(all_call_ids)), 72)
        for cid in all_call_ids:
            self.assertIsNotNone(re.fullmatch("[0-9a-f]{32}", cid), cid)

    def test_candidate_validation_and_continuation_messages_use_generated_initial(self):
        task_c01 = self.prompts_by_id["C01"]
        ref_src = self.oracles_by_id["C01"]["references"][0]["source"]
        self.assertTrue(diag_scoring.candidate_valid(task_c01, ref_src))
        self.assertFalse(diag_scoring.candidate_valid(task_c01, "import os\n" + ref_src))
        self.assertFalse(diag_scoring.candidate_valid(task_c01, "not valid python("))
        oversized_src = ref_src + "\n    padding_str = '" + ("a" * 4100) + "'\n"
        self.assertFalse(diag_scoring.candidate_valid(task_c01, oversized_src))
        escaped_heavy_src = ref_src + "\n    padding_esc = '" + ("\x01" * 900) + "'\n"
        self.assertLessEqual(len(escaped_heavy_src.encode("utf-8")), 4096)
        self.assertFalse(diag_scoring.candidate_valid(task_c01, escaped_heavy_src))

        retry_msgs = diag_scoring.continuation_messages(
            task_c01,
            "base",
            self.raw_files["supervision_v1.json"],
            ref_src,
            "retry",
            feedback=None,
        )
        self.assertEqual(retry_msgs[1], {"role": "assistant", "content": ref_src})
        self.assertIn('{"kind":"results_withheld"}', retry_msgs[2]["content"])

        inv_msgs = diag_scoring.continuation_messages(
            task_c01,
            "base",
            self.raw_files["supervision_v1.json"],
            "bad syntax(",
            "feedback",
            feedback={"kind": "invalid_source"},
        )
        self.assertIn('{"kind":"invalid_source"}', inv_msgs[2]["content"])

    def _simulate_run(self, *, mode="usable"):
        sched = diag_scoring.build_generation_schedule(
            self.raw_files["prompts.json"],
            self.raw_files["supervision_v1.json"],
            run_id=self.run_id,
        )
        phase_one = []
        for ep in sched["episodes"]:
            tid = ep["task_id"]
            arm = ep["arm"]
            task_idx = TASK_IDS.index(tid)
            oracle = self.oracles_by_id[tid]
            ref_src = oracle["references"][0]["source"]
            init_defect_src = oracle["incorrect_programs"][0]["source"]
            priv_defect_src = oracle["incorrect_programs"][1]["source"]

            if mode == "saturated":
                init_cand = ref_src
                fb_cand = ref_src
                rt_cand = ref_src
            elif mode == "invalid_initial" and tid == "C01" and arm == "base":
                init_cand = "def broken(:\n"
                fb_cand = ref_src
                rt_cand = init_defect_src
            else:
                if task_idx < 6:
                    init_cand = ref_src
                    fb_cand = ref_src
                    rt_cand = ref_src
                elif task_idx < 10:
                    init_cand = init_defect_src
                    fb_cand = ref_src
                    rt_cand = init_defect_src
                else:
                    init_cand = priv_defect_src
                    fb_cand = priv_defect_src
                    rt_cand = priv_defect_src

            task_prompt = self.prompts_by_id[tid]
            is_valid = diag_scoring.candidate_valid(task_prompt, init_cand)
            if not is_valid:
                pub_fb = {"kind": "invalid_source"}
            else:
                pub_cases = []
                for pcase in task_prompt["public_cases"]:
                    if init_cand == init_defect_src:
                        wit = oracle["incorrect_programs"][0]["witness"]
                        passed = wit["case_id"] != pcase["id"]
                    else:
                        passed = (init_cand == ref_src) or (init_cand == priv_defect_src)
                    pub_cases.append({
                        "id": pcase["id"],
                        "status": "pass" if passed else "mismatch",
                    })
                pub_fb = {"kind": "visible_tests", "cases": pub_cases}

            phase_one.append({
                "episode_index": ep["episode_index"],
                "task_id": tid,
                "arm": arm,
                "candidates": {
                    "initial": init_cand,
                    "feedback": fb_cand,
                    "retry": rt_cand,
                },
                "public_feedback": pub_fb,
                "generation_costs": {
                    role_name: {
                        "input_tokens": 500,
                        "output_tokens": 80,
                        "forward_calls": 80,
                        "wall_ns": 1000,
                        "peak_allocated_bytes": 1024,
                        "peak_reserved_bytes": 2048,
                    }
                    for role_name in ("initial", "feedback", "retry")
                },
                "public_execution_ns": 0 if not is_valid else 200,
            })

        scoring_run_ids = [("%02x" % idx) + ("ee" * 15) for idx in range(18)]
        bundles_plan = diag_scoring.build_final_scoring_bundles(
            self.raw_files,
            self.worker_raw,
            self.run_id,
            phase_one,
            scoring_run_ids[:len(diag_scoring.plan_final_positions(self.raw_files, phase_one)["batches"])]
            if hasattr(diag_scoring, "plan_final_positions")
            else scoring_run_ids,
        )
        bundle_records = []
        for b_idx, bundle_entry in enumerate(bundles_plan["batches"]):
            events = [{
                "kind": "ready",
                "run_id": bundle_entry["run_id"],
                "bundle_sha256": bundle_entry["bundle_sha256"],
                "worker_sha256": bundle_entry["worker_sha256"],
                "runtime": copy.deepcopy(self.runtime),
            }]
            for ord_idx, pos in enumerate(bundle_entry["positions"]):
                tid = pos["task_id"]
                cid = pos["case_id"]
                src = pos["source"]
                oracle = self.oracles_by_id[tid]
                case_obj = next(item for item in oracle["cases"] if item["id"] == cid)
                ref_src = oracle["references"][0]["source"]
                if src == ref_src:
                    out_obj = case_obj["expected"]
                else:
                    matched_prog = next(
                        (prog for prog in oracle["incorrect_programs"] if prog["source"] == src),
                        None,
                    )
                    if matched_prog and matched_prog["witness"]["case_id"] == cid:
                        out_obj = matched_prog["witness"]["predicted"]
                    else:
                        out_obj = case_obj["expected"]
                events.append({
                    "kind": "row",
                    "run_id": bundle_entry["run_id"],
                    "ordinal": ord_idx,
                    "request_sha256": pos["request_sha256"],
                    "result": {
                        "status": "completed",
                        "returncode": 0,
                        "stdout_hex": encoded(out_obj).hex(),
                        "stderr_hex": "",
                        "elapsed_ns": 50,
                        "process_reaped": True,
                        "stdin_complete": True,
                    },
                })
            events.append({
                "kind": "done",
                "run_id": bundle_entry["run_id"],
                "received": len(bundle_entry["positions"]),
                "stopped": False,
                "elapsed_ns": 500,
            })
            stdout_jsonl = b"".join(encoded(ev) + b"\n" for ev in events)
            bundle_records.append({
                "batch_index": b_idx,
                "bundle_raw": bundle_entry["bundle_raw"],
                "transport": {
                    "status": "completed",
                    "returncode": 0,
                    "stdout_hex": stdout_jsonl.hex(),
                    "stderr_hex": "",
                    "elapsed_ns": 600,
                    "process_reaped": True,
                    "stdin_complete": True,
                },
            })
        return diag_scoring.score_diagnostic_run(
            self.raw_files,
            self.worker_raw,
            self.run_id,
            phase_one,
            bundles_plan["run_ids"],
            bundle_records,
            expected_runtime=self.runtime,
        )

    def test_usable_calibration_transitions_net_gain_and_public_vs_private_exposure(self):
        report = self._simulate_run(mode="usable")
        self.assertEqual(report["calibration"]["status"], "DEVELOPMENT_DIAGNOSTIC_USABLE")
        self.assertEqual(report["calibration"]["reasons"], [])
        base_summary = report["arms"]["base"]
        self.assertEqual(base_summary["candidates"]["initial"]["PASS"], 6)
        self.assertEqual(base_summary["candidates"]["initial"]["SEMANTIC_FAIL"], 6)
        self.assertEqual(base_summary["candidates"]["feedback"]["PASS"], 10)
        self.assertEqual(base_summary["candidates"]["retry"]["PASS"], 6)
        self.assertEqual(base_summary["feedback_vs_retry"]["net_gain"], 4)
        self.assertEqual(base_summary["initial_exposure"]["public_revealed_semantic_fail"], 4)
        self.assertEqual(base_summary["initial_exposure"]["private_only_semantic_fail"], 2)
        for flag in (
            "benchmark_eligible",
            "formal_evaluation_ready",
            "net_benefit_established",
            "isolated",
        ):
            self.assertIs(report[flag], False)

    def test_saturated_and_invalid_initial_source_handling(self):
        sat = self._simulate_run(mode="saturated")
        self.assertEqual(sat["calibration"]["status"], "INCONCLUSIVE_CALIBRATION")
        self.assertTrue(len(sat["calibration"]["reasons"]) >= 1)

        inv = self._simulate_run(mode="invalid_initial")
        self.assertEqual(
            inv["arms"]["base"]["candidates"]["initial"]["EXECUTION_FAIL"],
            1,
        )
        c01_base = next(
            ep for ep in inv["episodes"]
            if ep["task_id"] == "C01" and ep["arm"] == "base"
        )
        self.assertEqual(
            c01_base["roles"]["initial"]["case_statuses"],
            ["not_executed_invalid_source"] * 8,
        )

    def test_mid_batch_program_error_stop_and_continuation_recovery(self):
        sched = diag_scoring.build_generation_schedule(
            self.raw_files["prompts.json"],
            self.raw_files["supervision_v1.json"],
            run_id=self.run_id,
        )
        phase_one = []
        for ep in sched["episodes"]:
            tid = ep["task_id"]
            ref_src = self.oracles_by_id[tid]["references"][0]["source"]
            phase_one.append({
                "episode_index": ep["episode_index"],
                "task_id": tid,
                "arm": ep["arm"],
                "candidates": {"initial": ref_src, "feedback": ref_src, "retry": ref_src},
                "public_feedback": {"kind": "visible_tests", "cases": [{"id": "P01", "status": "pass"}, {"id": "P02", "status": "pass"}]},
                "generation_costs": {
                    role_name: {
                        "input_tokens": 500,
                        "output_tokens": 80,
                        "forward_calls": 80,
                        "wall_ns": 1000,
                        "peak_allocated_bytes": 1024,
                        "peak_reserved_bytes": 2048,
                    }
                    for role_name in ("initial", "feedback", "retry")
                },
                "public_execution_ns": 200,
            })
        scoring_run_ids = [("%02x" % idx) + ("ee" * 15) for idx in range(18)]
        bundles_plan = diag_scoring.build_final_scoring_bundles(
            self.raw_files, self.worker_raw, self.run_id, phase_one, scoring_run_ids
        )
        bundle_records = []
        continuation_records = []
        for b_idx, bundle_entry in enumerate(bundles_plan["batches"]):
            if b_idx == 0:
                first_pos = bundle_entry["positions"][0]
                first_events = [
                    {
                        "kind": "ready",
                        "run_id": bundle_entry["run_id"],
                        "bundle_sha256": bundle_entry["bundle_sha256"],
                        "worker_sha256": bundle_entry["worker_sha256"],
                        "runtime": copy.deepcopy(self.runtime),
                    },
                    {
                        "kind": "row",
                        "run_id": bundle_entry["run_id"],
                        "ordinal": 0,
                        "request_sha256": first_pos["request_sha256"],
                        "result": {
                            "status": "program_error",
                            "returncode": 20,
                            "stdout_hex": "",
                            "stderr_hex": "",
                            "elapsed_ns": 50,
                            "process_reaped": True,
                            "stdin_complete": True,
                        },
                    },
                    {
                        "kind": "done",
                        "run_id": bundle_entry["run_id"],
                        "received": 1,
                        "stopped": True,
                        "elapsed_ns": 100,
                    },
                ]
                stdout_first = b"".join(encoded(ev) + b"\n" for ev in first_events)
                bundle_records.append({
                    "batch_index": 0,
                    "bundle_raw": bundle_entry["bundle_raw"],
                    "transport": {
                        "status": "completed",
                        "returncode": 0,
                        "stdout_hex": stdout_first.hex(),
                        "stderr_hex": "",
                        "elapsed_ns": 120,
                        "process_reaped": True,
                        "stdin_complete": True,
                    },
                })
                cont_meta = diag_scoring.build_continuation_bundle(
                    self.worker_raw,
                    self.run_id,
                    0,
                    1,
                    bundle_entry["positions"][1:],
                )
                cont_events = [{
                    "kind": "ready",
                    "run_id": cont_meta["run_id"],
                    "bundle_sha256": cont_meta["bundle_sha256"],
                    "worker_sha256": cont_meta["worker_sha256"],
                    "runtime": copy.deepcopy(self.runtime),
                }]
                for ord_idx, pos in enumerate(cont_meta["positions"]):
                    oracle = self.oracles_by_id[pos["task_id"]]
                    case_obj = next(item for item in oracle["cases"] if item["id"] == pos["case_id"])
                    cont_events.append({
                        "kind": "row",
                        "run_id": cont_meta["run_id"],
                        "ordinal": ord_idx,
                        "request_sha256": pos["request_sha256"],
                        "result": {
                            "status": "completed",
                            "returncode": 0,
                            "stdout_hex": encoded(case_obj["expected"]).hex(),
                            "stderr_hex": "",
                            "elapsed_ns": 50,
                            "process_reaped": True,
                            "stdin_complete": True,
                        },
                    })
                cont_events.append({
                    "kind": "done",
                    "run_id": cont_meta["run_id"],
                    "received": len(cont_meta["positions"]),
                    "stopped": False,
                    "elapsed_ns": 400,
                })
                stdout_cont = b"".join(encoded(ev) + b"\n" for ev in cont_events)
                continuation_records.append({
                    "batch_index": 0,
                    "sub_index": 1,
                    "run_id": cont_meta["run_id"],
                    "position_indices": [pos["position_index"] for pos in cont_meta["positions"]],
                    "bundle_raw": cont_meta["bundle_raw"],
                    "transport": {
                        "status": "completed",
                        "returncode": 0,
                        "stdout_hex": stdout_cont.hex(),
                        "stderr_hex": "",
                        "elapsed_ns": 450,
                        "process_reaped": True,
                        "stdin_complete": True,
                    },
                })
            else:
                events = [{
                    "kind": "ready",
                    "run_id": bundle_entry["run_id"],
                    "bundle_sha256": bundle_entry["bundle_sha256"],
                    "worker_sha256": bundle_entry["worker_sha256"],
                    "runtime": copy.deepcopy(self.runtime),
                }]
                for ord_idx, pos in enumerate(bundle_entry["positions"]):
                    oracle = self.oracles_by_id[pos["task_id"]]
                    case_obj = next(item for item in oracle["cases"] if item["id"] == pos["case_id"])
                    events.append({
                        "kind": "row",
                        "run_id": bundle_entry["run_id"],
                        "ordinal": ord_idx,
                        "request_sha256": pos["request_sha256"],
                        "result": {
                            "status": "completed",
                            "returncode": 0,
                            "stdout_hex": encoded(case_obj["expected"]).hex(),
                            "stderr_hex": "",
                            "elapsed_ns": 50,
                            "process_reaped": True,
                            "stdin_complete": True,
                        },
                    })
                events.append({
                    "kind": "done",
                    "run_id": bundle_entry["run_id"],
                    "received": len(bundle_entry["positions"]),
                    "stopped": False,
                    "elapsed_ns": 500,
                })
                stdout_jsonl = b"".join(encoded(ev) + b"\n" for ev in events)
                bundle_records.append({
                    "batch_index": b_idx,
                    "bundle_raw": bundle_entry["bundle_raw"],
                    "transport": {
                        "status": "completed",
                        "returncode": 0,
                        "stdout_hex": stdout_jsonl.hex(),
                        "stderr_hex": "",
                        "elapsed_ns": 600,
                        "process_reaped": True,
                        "stdin_complete": True,
                    },
                })
        report = diag_scoring.score_diagnostic_run(
            self.raw_files,
            self.worker_raw,
            self.run_id,
            phase_one,
            bundles_plan["run_ids"],
            bundle_records,
            continuation_records=continuation_records,
            expected_runtime=self.runtime,
        )
        self.assertEqual(report["calibration"]["incomplete_reasons"], [])
        self.assertEqual(report["calibration"]["protocol_contradictions"], [])
        self.assertEqual(report["episodes"][0]["roles"]["initial"]["state"], "EXECUTION_FAIL")
        self.assertEqual(report["episodes"][0]["roles"]["feedback"]["state"], "PASS")
        self.assertEqual(report["costs"]["by_arm"]["base"]["max_peak_allocated_bytes"], 1024)
        self.assertEqual(report["costs"]["by_arm"]["base"]["max_peak_reserved_bytes"], 2048)

    def test_unreaped_worker_on_semantic_fail_and_schedule_public_feedback_cross_checks(self):
        sched = diag_scoring.build_generation_schedule(
            self.raw_files["prompts.json"],
            self.raw_files["supervision_v1.json"],
            run_id=self.run_id,
        )
        phase_one = []
        for ep in sched["episodes"]:
            tid = ep["task_id"]
            oracle = self.oracles_by_id[tid]
            ref_src = oracle["references"][0]["source"]
            init_def = oracle["incorrect_programs"][0]["source"]
            cand_src = init_def if ep["episode_index"] == 0 else ref_src
            wit_cid = oracle["incorrect_programs"][0]["witness"]["case_id"]
            pub_fb = (
                {
                    "kind": "visible_tests",
                    "cases": [
                        {"id": "P01", "status": "mismatch" if wit_cid == "P01" else "pass"},
                        {"id": "P02", "status": "mismatch" if wit_cid == "P02" else "pass"},
                    ],
                }
                if ep["episode_index"] == 0
                else {"kind": "visible_tests", "cases": [{"id": "P01", "status": "pass"}, {"id": "P02", "status": "pass"}]}
            )
            phase_one.append({
                "episode_index": ep["episode_index"],
                "task_id": tid,
                "arm": ep["arm"],
                "candidates": {"initial": cand_src, "feedback": ref_src, "retry": ref_src},
                "public_feedback": pub_fb,
                "generation_costs": {
                    role_name: {
                        "input_tokens": 500,
                        "output_tokens": 80,
                        "forward_calls": 80,
                        "wall_ns": 1000,
                        "peak_allocated_bytes": 1024,
                        "peak_reserved_bytes": 2048,
                    }
                    for role_name in ("initial", "feedback", "retry")
                },
                "public_execution_ns": 200,
            })

        scoring_run_ids = [("%02x" % idx) + ("dd" * 15) for idx in range(18)]
        bundles_plan = diag_scoring.build_final_scoring_bundles(
            self.raw_files, self.worker_raw, self.run_id, phase_one, scoring_run_ids
        )
        b0 = bundles_plan["batches"][0]
        unreaped_events = [{
            "kind": "ready",
            "run_id": b0["run_id"],
            "bundle_sha256": b0["bundle_sha256"],
            "worker_sha256": b0["worker_sha256"],
            "runtime": copy.deepcopy(self.runtime),
        }]
        for ord_idx in range(3):
            pos = b0["positions"][ord_idx]
            oracle = self.oracles_by_id[pos["task_id"]]
            case_obj = next(item for item in oracle["cases"] if item["id"] == pos["case_id"])
            if ord_idx < 2:
                prog = oracle["incorrect_programs"][0]
                out_obj = (
                    prog["witness"]["predicted"]
                    if prog["witness"]["case_id"] == pos["case_id"]
                    else case_obj["expected"]
                )
                unreaped_events.append({
                    "kind": "row",
                    "run_id": b0["run_id"],
                    "ordinal": ord_idx,
                    "request_sha256": pos["request_sha256"],
                    "result": {
                        "status": "completed",
                        "returncode": 0,
                        "stdout_hex": encoded(out_obj).hex(),
                        "stderr_hex": "",
                        "elapsed_ns": 50,
                        "process_reaped": True,
                        "stdin_complete": True,
                    },
                })
            else:
                unreaped_events.append({
                    "kind": "row",
                    "run_id": b0["run_id"],
                    "ordinal": ord_idx,
                    "request_sha256": pos["request_sha256"],
                    "result": {
                        "status": "timeout",
                        "returncode": None,
                        "stdout_hex": "",
                        "stderr_hex": "",
                        "elapsed_ns": 2_000_000_000,
                        "process_reaped": False,
                        "stdin_complete": True,
                    },
                })
        unreaped_events.append({
            "kind": "done",
            "run_id": b0["run_id"],
            "received": 3,
            "stopped": True,
            "elapsed_ns": 2_000_000_200,
        })
        b0_stdout = b"".join(encoded(ev) + b"\n" for ev in unreaped_events)
        unreaped_rep = diag_scoring.score_diagnostic_run(
            self.raw_files,
            self.worker_raw,
            self.run_id,
            phase_one,
            bundles_plan["run_ids"],
            [{
                "batch_index": 0,
                "bundle_raw": b0["bundle_raw"],
                "transport": {
                    "status": "completed",
                    "returncode": 0,
                    "stdout_hex": b0_stdout.hex(),
                    "stderr_hex": "",
                    "elapsed_ns": 2_000_000_300,
                    "process_reaped": True,
                    "stdin_complete": True,
                },
            }],
            expected_runtime=self.runtime,
        )
        self.assertEqual(unreaped_rep["calibration"]["status"], "INCONCLUSIVE_INCOMPLETE")
        self.assertIn("unknown_candidate_outcome", unreaped_rep["calibration"]["incomplete_reasons"])
        self.assertIsNone(unreaped_rep["arms"]["base"]["feedback_vs_retry"]["net_gain"])

        swapped_p1 = copy.deepcopy(phase_one)
        swapped_p1[0]["arm"], swapped_p1[1]["arm"] = swapped_p1[1]["arm"], swapped_p1[0]["arm"]
        swapped_rep = diag_scoring.score_diagnostic_run(
            self.raw_files,
            self.worker_raw,
            self.run_id,
            swapped_p1,
            bundles_plan["run_ids"],
            [],
            expected_runtime=self.runtime,
        )
        self.assertEqual(swapped_rep["calibration"]["status"], "INVALID_PROTOCOL")
        self.assertIn("schedule_identity_mismatch", swapped_rep["calibration"]["protocol_contradictions"])
        self.assertIn("unknown_candidate_outcome", swapped_rep["calibration"]["incomplete_reasons"])


if __name__ == "__main__":
    unittest.main()
