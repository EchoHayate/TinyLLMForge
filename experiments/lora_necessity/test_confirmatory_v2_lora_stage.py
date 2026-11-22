import ast
import copy
import hashlib
import importlib
import importlib.util
import io
import json
import os
from pathlib import Path
import tempfile
import tokenize
import types
import unittest
from unittest import mock

from confirmatory_v2_contract import CONFIRMATORY_FILE_PINS, TRANSFER_IDS, load_confirmatory_dataset
from pilot_protocol import ARMS
from validation_contract import digest, encoded


REPO_LORA = Path(__file__).resolve().parent

with mock.patch("subprocess.Popen") as import_spawn, mock.patch("os.killpg") as import_killpg:
    try:
        conf_train = importlib.import_module("confirmatory_v2_training")
    except ModuleNotFoundError as err_one:
        if err_one.name != "confirmatory_v2_training":
            raise
        conf_train = None
    try:
        conf_scoring = importlib.import_module("confirmatory_v2_scoring")
    except ModuleNotFoundError as err_two:
        if err_two.name != "confirmatory_v2_scoring":
            raise
        conf_scoring = None
    try:
        conf_remote = importlib.import_module("confirmatory_v2_remote")
    except ModuleNotFoundError as err_three:
        if err_three.name != "confirmatory_v2_remote":
            raise
        conf_remote = None
    try:
        conf_ctrl = importlib.import_module("confirmatory_v2_controller")
    except ModuleNotFoundError as err_four:
        if err_four.name != "confirmatory_v2_controller":
            raise
        conf_ctrl = None
    assert not import_spawn.called
    assert not import_killpg.called


class ConfirmatoryLoraStageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        data_dir = REPO_LORA / "data/confirmatory_v2_r1"
        cls.raw_files = {
            fname: (data_dir / fname).read_bytes()
            for fname in CONFIRMATORY_FILE_PINS
        }
        cls.worker_raw = (REPO_LORA / "validation_task_worker.py").read_bytes()
        cls.backend_raw = (REPO_LORA / "validation_remote.py").read_bytes()
        cls.dataset = load_confirmatory_dataset(cls.raw_files)
        cls.oracles_by_id = {item["id"]: item for item in cls.dataset["oracles"]}
        cls.prompts_by_id = {item["id"]: item for item in cls.dataset["prompts"]}
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
        self.assertIsNotNone(conf_train, "confirmatory_v2_training is absent")
        self.assertIsNotNone(conf_scoring, "confirmatory_v2_scoring is absent")
        self.assertIsNotNone(conf_remote, "confirmatory_v2_remote is absent")
        self.assertIsNotNone(conf_ctrl, "confirmatory_v2_controller is absent")

    def test_style_across_all_gate_b_modules(self):
        for fname in (
            "confirmatory_v2_training.py",
            "confirmatory_v2_scoring.py",
            "confirmatory_v2_remote.py",
            "confirmatory_v2_controller.py",
            "test_confirmatory_v2_lora_stage.py",
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

    def test_schedule_96_episodes_192_calls_and_paired_four_arm_scoring_plus_amortization(self):
        run_id = "45" * 16
        sched = conf_scoring.build_confirmatory_schedule(
            self.raw_files["prompts.json"],
            self.raw_files["supervision.json"],
            run_id=run_id,
        )
        self.assertEqual(sched["scheduled_episodes"], 96)
        self.assertEqual(sched["scheduled_generation_calls"], 192)
        first_arms = [sched["episodes"][idx * 4]["arm"] for idx in range(24)]
        for arm_name in ARMS:
            self.assertEqual(first_arms.count(arm_name), 6)

        phase_one = []
        for ep in sched["episodes"]:
            tid = ep["task_id"]
            arm = ep["arm"]
            oracle = self.oracles_by_id[tid]
            ref_src = oracle["references"][0]["source"]
            init_def = oracle["incorrect_programs"][0]["source"]
            is_transfer = tid in TRANSFER_IDS
            if is_transfer:
                task_num = int(tid[2:])
                if arm == "base":
                    pass_flag = task_num <= 5
                elif arm == "text":
                    pass_flag = task_num <= 6
                elif arm == "lora":
                    pass_flag = task_num <= 7
                else:
                    pass_flag = task_num <= 9
            else:
                task_num = int(tid[2:])
                pass_flag = task_num <= 8
            cand_src = ref_src if pass_flag else init_def
            wit_cid = oracle["incorrect_programs"][0]["witness"]["case_id"]
            pub_fb = (
                {"kind": "visible_tests", "cases": [{"id": "P01", "status": "pass"}, {"id": "P02", "status": "pass"}]}
                if pass_flag
                else {
                    "kind": "visible_tests",
                    "cases": [
                        {"id": "P01", "status": "mismatch" if wit_cid == "P01" else "pass"},
                        {"id": "P02", "status": "mismatch" if wit_cid == "P02" else "pass"},
                    ],
                }
            )
            wall_per_call = 50_000_000_000 if arm in ("text", "lora_text") else 10_000_000_000
            phase_one.append({
                "episode_index": ep["episode_index"],
                "task_id": tid,
                "split": ep["split"],
                "arm": arm,
                "candidates": {"initial": cand_src, "feedback": cand_src},
                "public_feedback": pub_fb,
                "generation_costs": {
                    role_name: {
                        "input_tokens": 3000 if arm in ("text", "lora_text") else 500,
                        "output_tokens": 80,
                        "forward_calls": 80,
                        "wall_ns": wall_per_call,
                        "peak_allocated_bytes": 18_000_000_000,
                        "peak_reserved_bytes": 68_000_000_000,
                    }
                    for role_name in ("initial", "feedback")
                },
                "public_execution_ns": 200,
            })

        scoring_run_ids = [("%02x" % idx) + ("cc" * 15) for idx in range(48)]
        bundles_plan = conf_scoring.build_confirmatory_scoring_bundles(
            self.raw_files, self.worker_raw, run_id, phase_one, scoring_run_ids
        )
        self.assertEqual(len(bundles_plan["batches"]), 48)

        bundle_records = []
        for b_idx, b_meta in enumerate(bundles_plan["batches"]):
            events = [{
                "kind": "ready",
                "run_id": b_meta["run_id"],
                "bundle_sha256": b_meta["bundle_sha256"],
                "worker_sha256": b_meta["worker_sha256"],
                "runtime": copy.deepcopy(self.expected_runtime),
            }]
            for ord_idx, pos in enumerate(b_meta["positions"]):
                oracle = self.oracles_by_id[pos["task_id"]]
                case_obj = next(item for item in oracle["cases"] if item["id"] == pos["case_id"])
                ref_src = oracle["references"][0]["source"]
                if pos["source"] == ref_src:
                    out_obj = case_obj["expected"]
                else:
                    prog = oracle["incorrect_programs"][0]
                    out_obj = (
                        prog["witness"]["predicted"]
                        if prog["witness"]["case_id"] == pos["case_id"]
                        else case_obj["expected"]
                    )
                events.append({
                    "kind": "row",
                    "run_id": b_meta["run_id"],
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
                "run_id": b_meta["run_id"],
                "received": len(b_meta["positions"]),
                "stopped": False,
                "elapsed_ns": 400,
            })
            stdout_jsonl = b"".join(encoded(ev) + b"\n" for ev in events)
            bundle_records.append({
                "batch_index": b_idx,
                "bundle_raw": b_meta["bundle_raw"],
                "transport": {
                    "status": "completed",
                    "returncode": 0,
                    "stdout_hex": stdout_jsonl.hex(),
                    "stderr_hex": "",
                    "elapsed_ns": 450,
                    "process_reaped": True,
                    "stdin_complete": True,
                },
            })

        training_meta = {
            "core_wall_ns": 160_000_000_000,
            "adapter_save_ns": 1_000_000_000,
            "adapter_load_ns": 1_000_000_000,
            "adapter_bytes": 18_874_368,
            "initial_loss": 1.85,
            "final_loss": 0.12,
        }
        report = conf_scoring.score_confirmatory_run(
            self.raw_files,
            self.worker_raw,
            run_id,
            phase_one,
            scoring_run_ids,
            bundle_records,
            expected_runtime=self.expected_runtime,
            training_summary=training_meta,
            shared_costs={"model_and_tokenizer_load_ns": 20_000_000_000},
        )
        self.assertEqual(report["verdict"]["status"], "POSITIVE_MATCHED_LORA_GAIN")
        self.assertEqual(
            report["comparisons"]["feedback"]["transfer"]["lora_text_vs_text"]["net_gain"],
            3,
        )
        self.assertEqual(
            report["comparisons"]["feedback"]["retention"]["lora_text_vs_text"]["right_to_wrong"],
            0,
        )
        self.assertGreater(report["costs"]["amortization"]["per_episode_saved_ns_lora_vs_text"], 0)
        self.assertAlmostEqual(
            report["costs"]["amortization"]["break_even_episodes_lora_vs_text"],
            162_000_000_000 / 80_000_000_000,
            places=4,
        )

    def test_package_zero_private_oracle_leak_in_phase_one(self):
        dep_sources = {
            fname: (REPO_LORA / fname).read_bytes()
            for fname in conf_remote.PHASE1_LORA_SOURCE_NAMES
        }
        run_id = "56" * 16
        local_root = "/tmp/conf-runs/" + run_id
        ssh_prefix = ["/usr/bin/ssh", "-T", "rl-server"]
        ssh_env = {"PATH": "/usr/bin:/bin", "LANG": "C"}
        scope_raw = encoded({
            "kind": "LORA_CONFIRMATORY_V2_STAGE_SCOPE",
            "version": 1,
            "run_id": run_id,
            "local_run_root": local_root,
            "remote_parent": "/tmp/conf-remote",
            "ssh_argv_prefix_sha256": digest(ssh_prefix),
            "ssh_environment_sha256": digest(ssh_env),
            "preflight_receipt_sha256": hashlib.sha256(self.preflight_raw).hexdigest(),
        })
        pkg = conf_ctrl.build_confirmatory_lora_package(
            self.raw_files,
            self.worker_raw,
            self.backend_raw,
            dep_sources,
            self.preflight_raw,
            scope_raw,
            run_id=run_id,
            scoring_run_ids=[("%02x" % idx) + ("dd" * 15) for idx in range(48)],
            ssh_argv_prefix=ssh_prefix,
            ssh_environment=ssh_env,
            interpreter="/env/bin/python",
            resolved_interpreter="/opt/cpython/bin/python3.12",
            site_packages="/env/lib/python3.12/site-packages",
            checkpoint_root="/models/Qwen3-8B",
            expected_tokenizer_sha256="7" * 64,
            remote_parent="/tmp/conf-remote",
            local_run_root=local_root,
        )
        p1_str = bytes.fromhex(pkg["private"]["phase_one_payload_hex"]).decode("ascii")
        self.assertNotIn("H01", p1_str)
        self.assertNotIn("incorrect_programs", p1_str)

    @unittest.skipUnless(importlib.util.find_spec("torch"), "requires PyTorch for tiny-model roundtrip")
    def test_real_pytorch_tiny_qwen3_lora_train_save_load_mount_roundtrip(self):
        import torch
        from adapter_artifact import load_adapter, mounted_adapter, save_adapter
        from training_worker import _state, _tensor_digest

        class TinyQwen3(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.config = types.SimpleNamespace(
                    model_type="qwen3", num_hidden_layers=1, return_dict=True, use_cache=False
                )
                self.embedding = torch.nn.Embedding(32, 8)
                self.model = torch.nn.Module()
                layer = torch.nn.Module()
                layer.self_attn = torch.nn.Module()
                layer.self_attn.q_proj = torch.nn.Linear(8, 8)
                layer.self_attn.v_proj = torch.nn.Linear(8, 8)
                self.model.layers = torch.nn.ModuleList([layer])
                self.lm_head = torch.nn.Linear(8, 32)

            def forward(self, input_ids, attention_mask=None, use_cache=False, return_dict=True):
                hidden = self.embedding(input_ids)
                attn_layer = self.model.layers[0].self_attn
                hidden = hidden + attn_layer.q_proj(hidden) + attn_layer.v_proj(hidden)
                logits = self.lm_head(hidden)
                return types.SimpleNamespace(logits=logits)

        torch.manual_seed(7)
        tiny = TinyQwen3().to(dtype=torch.bfloat16).eval().requires_grad_(False)
        before_base = _tensor_digest(_state(tiny))
        before_rng = torch.random.get_rng_state().clone()

        rows = [
            {
                "task_id": "V%02d" % idx,
                "input_ids": [3, 4, 5 + idx, 2],
                "attention_mask": [1, 1, 1, 1],
                "labels": [-100, -100, 5 + idx, 2],
                "prompt_tokens": 2,
                "target_tokens": 2,
                "messages_sha256": "a" * 64,
                "answer_sha256": "b" * 64,
            }
            for idx in range(1, 9)
        ]
        ds = {
            "kind": "LORA_CONFIRMATORY_V2_TRAINING_INPUTS",
            "version": 1,
            "supervision_sha256": CONFIRMATORY_FILE_PINS["supervision.json"],
            "tokenizer_state_sha256": "c" * 64,
            "eos_token_id": 2,
            "pad_token_id": 0,
            "rows": rows,
        }
        ds["dataset_sha256"] = digest(ds)

        train_res = conf_train.train_confirmatory_adapter(tiny, ds)
        self.assertEqual(_tensor_digest(_state(tiny)), before_base)
        self.assertTrue(torch.equal(torch.random.get_rng_state(), before_rng))
        epoch_one_loss = sum(st_item["loss"] for st_item in train_res["receipt"]["steps"][:8])
        epoch_two_loss = sum(st_item["loss"] for st_item in train_res["receipt"]["steps"][8:])
        self.assertLess(epoch_two_loss, epoch_one_loss)

        with tempfile.TemporaryDirectory() as tmp_d:
            art_path = Path(tmp_d) / "adapter.pt"
            saved_meta = save_adapter(art_path, train_res)
            loaded_meta = load_adapter(art_path, expected_sha256=saved_meta["sha256"])
            test_inp = torch.tensor([[3, 4]], dtype=torch.long)
            with torch.no_grad():
                base_logits = tiny(test_inp).logits.clone()
            with mounted_adapter(tiny, loaded_meta["result"]) as mounted_model:
                mounted_logits = mounted_model(test_inp).logits.clone()
            with torch.no_grad():
                after_logits = tiny(test_inp).logits.clone()
            self.assertFalse(torch.equal(base_logits, mounted_logits))
            self.assertTrue(torch.equal(base_logits, after_logits))
            self.assertEqual(_tensor_digest(_state(tiny)), before_base)
            self.assertTrue(torch.equal(torch.random.get_rng_state(), before_rng))


if __name__ == "__main__":
    unittest.main()
