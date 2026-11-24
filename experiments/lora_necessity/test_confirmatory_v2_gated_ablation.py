import ast
import base64
import copy
import importlib
import io
import json
from pathlib import Path
import tokenize
import unittest
from unittest import mock

from confirmatory_v2_contract import (
    CONFIRMATORY_FILE_PINS,
    EVAL_IDS,
    RETENTION_IDS,
    TRANSFER_IDS,
    load_confirmatory_dataset,
)
from validation_contract import digest, encoded


REPO_LORA = Path(__file__).resolve().parent

with mock.patch("subprocess.Popen") as import_spawn, mock.patch("os.killpg") as import_killpg:
    try:
        gated_mod = importlib.import_module("confirmatory_v2_gated_ablation")
    except ModuleNotFoundError as imp_err:
        if imp_err.name != "confirmatory_v2_gated_ablation":
            raise
        gated_mod = None
    assert not import_spawn.called
    assert not import_killpg.called


class GatedAblationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        conf_dir = REPO_LORA / "data/confirmatory_v2_r1"
        cls.raw_files = {
            fname: (conf_dir / fname).read_bytes()
            for fname in CONFIRMATORY_FILE_PINS
        }
        cls.v1_prompts_raw = (REPO_LORA / "data/prompts_v1.json").read_bytes()
        cls.v1_structure_raw = (REPO_LORA / "data/structure_v1.json").read_bytes()
        cls.recal_prompts_raw = (REPO_LORA / "data/recalibration_v2_r1/prompts.json").read_bytes()
        cls.worker_raw = (REPO_LORA / "validation_task_worker.py").read_bytes()
        cls.backend_raw = (REPO_LORA / "validation_remote.py").read_bytes()
        cls.dataset = load_confirmatory_dataset(cls.raw_files)
        cls.expected_runtime = {
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
        cls.preflight_raw = encoded({
            "kind": "LORA_RECALIBRATION_LIVE_PREFLIGHT_RECEIPT",
            "version": 1,
            "status": "PREFLIGHT_PASSED",
            "probes_passed": True,
            "remote_python": "/opt/cpython/bin/python3.12",
            "expected_runtime": cls.expected_runtime,
        })

    def setUp(self):
        self.assertIsNotNone(gated_mod, "confirmatory_v2_gated_ablation is absent")

    def test_style_and_zero_single_letter_identifiers(self):
        for fname in (
            "confirmatory_v2_gated_ablation.py",
            "test_confirmatory_v2_gated_ablation.py",
        ):
            raw = (REPO_LORA / fname).read_bytes()
            tree = ast.parse(raw)
            bad_names = []
            for node_item in ast.walk(tree):
                if isinstance(node_item, ast.Name) and len(node_item.id) == 1:
                    bad_names.append(node_item.id)
                elif isinstance(node_item, ast.arg) and len(node_item.arg) == 1:
                    bad_names.append(node_item.arg)
                elif isinstance(node_item, ast.ExceptHandler) and node_item.name and len(node_item.name) == 1:
                    bad_names.append(node_item.name)
                elif isinstance(node_item, ast.alias):
                    bound_name = node_item.asname or node_item.name
                    if len(bound_name) == 1:
                        bad_names.append(bound_name)
            self.assertEqual(bad_names, [], fname + ": " + repr(bad_names))
            self.assertEqual(
                [tok for tok in tokenize.tokenize(io.BytesIO(raw).readline) if tok.type == tokenize.COMMENT],
                [],
                fname,
            )

    def test_g6a_probe_inputs_and_package_exclude_all_eval_tasks(self):
        loaded = gated_mod.load_g6_calibration_and_probe_inputs(
            self.raw_files["supervision.json"],
            self.v1_prompts_raw,
            self.v1_structure_raw,
            self.recal_prompts_raw,
        )
        self.assertEqual(sorted(loaded["positive"].keys()), ["V%02d" % idx for idx in range(1, 9)])
        self.assertEqual(sorted(loaded["background"].keys()), ["R%02d" % idx for idx in range(1, 17)])
        self.assertEqual(sorted(loaded["dev_positive"].keys()), ["C%02d" % idx for idx in range(1, 13)])
        self.assertEqual(sorted(loaded["dev_negative"].keys()), ["D%02d" % idx for idx in range(1, 5)])

        dep_sources = {
            fname: (REPO_LORA / fname).read_bytes()
            for fname in gated_mod.G6_SOURCE_NAMES
        }
        run_id = "81" * 16
        pkg = gated_mod.build_gate_g6a_probe_package(
            self.raw_files["supervision.json"],
            self.v1_prompts_raw,
            self.v1_structure_raw,
            self.recal_prompts_raw,
            dep_sources,
            self.preflight_raw,
            run_id=run_id,
            ssh_argv_prefix=["/usr/bin/ssh", "-T", "mock-host"],
            ssh_environment={"PATH": "/usr/bin:/bin", "LANG": "C"},
            interpreter="/env/bin/python",
            resolved_interpreter="/opt/cpython/bin/python3.12",
            site_packages="/env/lib/python3.12/site-packages",
            checkpoint_root="/models/Qwen3-8B",
            expected_tokenizer_sha256="7" * 64,
            remote_parent="/tmp/g6a-remote",
            local_run_root="/tmp/g6a-local",
        )
        bundle = json.loads(bytes.fromhex(pkg["private"]["phase_one_payload_hex"]).decode("ascii"))
        self.assertNotIn("prompts_b64", bundle)
        for key_name, val_b64 in bundle.items():
            if key_name == "sources":
                continue
            decoded_text = base64.b64decode(val_b64).decode("utf-8")
            for eval_tid in EVAL_IDS:
                self.assertNotIn('"' + eval_tid + '"', decoded_text)
            self.assertNotIn("H01", decoded_text)

    def test_g6b_package_and_parity_scoring_logic(self):
        dep_sources = {
            fname: (REPO_LORA / fname).read_bytes()
            for fname in gated_mod.G6_SOURCE_NAMES
        }
        run_id = "82" * 16
        probe_sha = "9" * 64
        pkg = gated_mod.build_gate_g6b_ablation_package(
            self.raw_files,
            self.v1_prompts_raw,
            self.v1_structure_raw,
            self.worker_raw,
            self.backend_raw,
            dep_sources,
            self.preflight_raw,
            gate_g6a_probe_sha256=probe_sha,
            run_id=run_id,
            scoring_run_ids=[("%02x" % idx) + ("bb" * 15) for idx in range(13)],
            ssh_argv_prefix=["/usr/bin/ssh", "-T", "mock-host"],
            ssh_environment={"PATH": "/usr/bin:/bin", "LANG": "C"},
            interpreter="/env/bin/python",
            resolved_interpreter="/opt/cpython/bin/python3.12",
            site_packages="/env/lib/python3.12/site-packages",
            checkpoint_root="/models/Qwen3-8B",
            expected_tokenizer_sha256="7" * 64,
            remote_parent="/tmp/g6b-remote",
            local_run_root="/tmp/g6b-local",
        )
        self.assertEqual(pkg["public"]["gate_g6a_probe_sha256"], probe_sha)
        bundle = json.loads(bytes.fromhex(pkg["private"]["phase_one_payload_hex"]).decode("ascii"))
        for key_name, val_b64 in bundle.items():
            if key_name == "sources":
                continue
            if key_name.endswith("_b64") and key_name not in ("backend_b64", "worker_b64"):
                decoded_text = base64.b64decode(val_b64).decode("utf-8")
                self.assertNotIn("H01", decoded_text)
                self.assertNotIn("incorrect_programs", decoded_text)

        sched = gated_mod.build_gated_ablation_schedule(
            self.raw_files["prompts.json"],
            self.raw_files["supervision.json"],
            run_id=run_id,
        )
        self.assertEqual(len(sched["episodes"]), 26)
        self.assertEqual(sched["episodes"][0]["arm"], "base_canary")
        self.assertEqual(sched["episodes"][1]["arm"], "base_canary")
        self.assertEqual(sum(ep["arm"] == "lora_gated" for ep in sched["episodes"]), 24)

        oracles_map = {item["id"]: item for item in self.dataset["oracles"]}
        prompts_map = {item["id"]: item for item in self.dataset["prompts"]}
        g4b_p1 = []
        g5_p1 = []
        g4b_eps = []
        p1_recs = []
        gate_decisions = {}

        for tid in EVAL_IDS:
            ref_a = oracles_map[tid]["references"][0]["source"]
            inc_init = prompts_map[tid]["initial_source"]
            base_src = ref_a if tid in ("WT01", "WT02", "WT03", "WR01", "WR02", "WR03", "WR04") else inc_init
            conv_src = ref_a if tid in ("WT03", "WT05", "WT06", "WT11", "WR04", "WR05") else inc_init
            text_pass = tid in ("WT01", "WT03", "WT05", "WT11", "WR01", "WR02", "WR04", "WR05")
            g4b_p1.append({"task_id": tid, "arm": "base", "candidates": {"initial": base_src, "feedback": base_src}})
            g5_p1.append({"task_id": tid, "arm": "lora_conv", "candidates": {"initial": conv_src, "feedback": conv_src}})
            for arm_n, pass_flag in (
                ("base", base_src == ref_a),
                ("text", text_pass),
                ("lora", False),
            ):
                st_str = "PASS" if pass_flag else "SEMANTIC_FAIL"
                g4b_eps.append({
                    "task_id": tid,
                    "split": prompts_map[tid]["split"],
                    "arm": arm_n,
                    "roles": {
                        "initial": {"state": st_str, "has_unknown_case": False},
                        "feedback": {"state": st_str, "has_unknown_case": False},
                    },
                })
            is_open = tid in TRANSFER_IDS
            gate_decisions[tid] = {
                "score": 0.42 if is_open else -0.08,
                "threshold_tau": 0.15,
                "margin_over_tau": (0.42 if is_open else -0.08) - 0.15,
                "gate_open": is_open,
                "nearest_positive_id": "V01",
            }

        for ep in sched["episodes"]:
            tid = ep["task_id"]
            if ep["arm"] == "base_canary":
                chosen_src = [rec["candidates"]["initial"] for rec in g4b_p1 if rec["task_id"] == tid][0]
                g_open = False
            else:
                g_open = gate_decisions[tid]["gate_open"]
                chosen_src = (
                    [rec["candidates"]["initial"] for rec in g5_p1 if rec["task_id"] == tid][0]
                    if g_open
                    else [rec["candidates"]["initial"] for rec in g4b_p1 if rec["task_id"] == tid][0]
                )
            p1_recs.append({
                "episode_index": ep["episode_index"],
                "task_id": tid,
                "split": ep["split"],
                "arm": ep["arm"],
                "gate_open": g_open,
                "candidates": {"initial": chosen_src, "feedback": chosen_src},
                "public_feedback": {
                    "kind": "visible_tests",
                    "cases": [{"id": "P01", "status": "pass"}, {"id": "P02", "status": "pass"}]
                    if chosen_src == oracles_map[tid]["references"][0]["source"]
                    else [{"id": "P01", "status": "mismatch"}, {"id": "P02", "status": "mismatch"}],
                },
                "generation_costs": {
                    "initial": {"wall_ns": 15_000_000_000, "input_tokens": 450, "output_tokens": 140, "forward_calls": 140, "peak_allocated_bytes": 100, "peak_reserved_bytes": 100},
                    "feedback": {"wall_ns": 15_000_000_000, "input_tokens": 600, "output_tokens": 140, "forward_calls": 140, "peak_allocated_bytes": 100, "peak_reserved_bytes": 100},
                },
                "public_execution_ns": 1000,
            })

        g4b_report = {
            "episodes": g4b_eps,
            "costs": {"by_arm": {"text": {"generation_wall_ns": 6_686_320_000_000}}},
        }
        g5_eps = []
        for tid in EVAL_IDS:
            conv_pass = tid in ("WT03", "WT05", "WT06", "WT11", "WR04", "WR05")
            st_s = "PASS" if conv_pass else "SEMANTIC_FAIL"
            g5_eps.append({
                "task_id": tid,
                "split": prompts_map[tid]["split"],
                "arm": "lora_conv",
                "roles": {
                    "initial": {"state": st_s, "has_unknown_case": False},
                    "feedback": {"state": st_s, "has_unknown_case": False},
                },
            })
        g5_report = {"episodes": g5_eps}

        pos_outcomes = {}
        pos_counter = 0
        for ep_rec in p1_recs:
            tid = ep_rec["task_id"]
            ref_a = oracles_map[tid]["references"][0]["source"]
            for _role_n in ("initial", "feedback"):
                cand_s = ep_rec["candidates"][_role_n]
                for c_obj in oracles_map[tid]["cases"]:
                    if cand_s == ref_a:
                        out_v = copy.deepcopy(c_obj["expected"])
                    else:
                        out_v = {"kind": "return", "value": "__mismatch_sentinel__"}
                    pos_outcomes[pos_counter] = {
                        "status": "completed",
                        "process_reaped": True,
                        "outcome": out_v,
                    }
                    pos_counter += 1

        with mock.patch.object(
            gated_mod,
            "_collect_scored_positions",
            return_value=(pos_outcomes, [], []),
        ):
            scored = gated_mod.score_gated_ablation(
                self.raw_files,
                self.worker_raw,
                run_id,
                p1_recs,
                gate_decisions,
                {"bundle_sha256": "a" * 64, "threshold_tau": 0.15, "estimated_fp32_fingerprint_bytes": 147456},
                [],
                [],
                [],
                self.expected_runtime,
                {"core_wall_ns": 95_000_000_000, "adapter_save_ns": 40_000_000, "adapter_load_ns": 40_000_000, "adapter_bytes": 15335424},
                {"router_calibration_ns": 1_800_000_000, "router_eval_ns": 2_400_000_000},
                g4b_p1,
                g4b_report,
                g5_p1,
                g5_report,
            )

        self.assertTrue(scored["complete_and_verified"])
        self.assertTrue(scored["parity_checks"]["closed_gate_bit_exact_with_g4b_base"])
        self.assertTrue(scored["parity_checks"]["open_gate_bit_exact_with_g5_lora_conv"])
        self.assertEqual(scored["lora_gated_by_split"]["transfer"]["feedback"]["PASS"], 4)
        self.assertEqual(scored["lora_gated_by_split"]["retention"]["feedback"]["PASS"], 4)
        self.assertEqual(scored["total_pass_feedback"], 8)
        self.assertEqual(scored["comparisons_feedback"]["retention"]["lora_gated_vs_lora_conv"]["net_gain"], 2)
        self.assertEqual(scored["comparisons_feedback"]["retention"]["lora_gated_vs_base"]["net_gain"], 0)
        self.assertEqual(scored["comparisons_feedback"]["transfer"]["lora_gated_vs_text"]["net_gain"], 0)
        self.assertFalse(scored["benchmark_eligible"])
        self.assertFalse(scored["formal_evaluation_ready"])


if __name__ == "__main__":
    unittest.main()
