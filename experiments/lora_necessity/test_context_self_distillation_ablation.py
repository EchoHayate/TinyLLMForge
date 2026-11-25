import ast
import base64
import importlib
import importlib.util
import io
import json
from pathlib import Path
import tokenize
import types
import unittest
from unittest import mock

from confirmatory_v2_contract import (
    CONFIRMATORY_FILE_PINS,
    EVAL_IDS,
    load_confirmatory_dataset,
)
from validation_contract import encoded


REPO_LORA = Path(__file__).resolve().parent

with mock.patch("subprocess.Popen") as import_spawn, mock.patch("os.killpg") as import_killpg:
    try:
        csd_mod = importlib.import_module("context_self_distillation_ablation")
    except ModuleNotFoundError as imp_err:
        if imp_err.name != "context_self_distillation_ablation":
            raise
        csd_mod = None
    assert not import_spawn.called
    assert not import_killpg.called


class ContextSelfDistillationTests(unittest.TestCase):
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
        self.assertIsNotNone(csd_mod, "context_self_distillation_ablation is absent")

    def test_pools_and_leave_one_out_teacher_prompt_isolation(self):
        pools = csd_mod.load_g9_distillation_pools(
            self.raw_files["supervision.json"],
            self.v1_prompts_raw,
            self.v1_structure_raw,
            self.recal_prompts_raw,
        )
        self.assertEqual(len(pools["seed_supervision"]), 8)
        self.assertEqual(len(pools["unlabeled_recal"]), 12)
        self.assertEqual(len(pools["unlabeled_v1_transfer"]), 16)
        self.assertEqual(len(pools["background"]), 16)
        self.assertEqual(len(pools["dev_negative"]), 4)
        for u_task in list(pools["unlabeled_recal"].values()) + list(pools["unlabeled_v1_transfer"].values()):
            self.assertNotIn("reference_source", u_task)
            self.assertNotIn("family", u_task)
            self.assertNotIn("split", u_task)

        v01 = pools["seed_supervision"]["V01"]
        loo_msgs = csd_mod.build_teacher_context_messages(
            v01, self.raw_files["supervision.json"], exclude_task_id="V01"
        )
        self.assertNotIn("def lease_occupancy(limit, requests):\n    if limit < 0:\n        raise ValueError(\"limit\")\n    live, out = [], []\n    for st, en, seats in requests:\n        if en <= st or seats < 0:\n            raise ValueError(\"lease\")\n        live = [it for it in live if it[1] > st]", loo_msgs[0]["content"])
        self.assertIn("clipped_band_sum", loo_msgs[0]["content"])

        c01 = pools["unlabeled_recal"]["C01"]
        full_msgs = csd_mod.build_teacher_context_messages(
            c01, self.raw_files["supervision.json"], exclude_task_id=None
        )
        self.assertIn("lease_occupancy", full_msgs[0]["content"])
        self.assertIn("tiered_seat_assign", full_msgs[0]["content"])

    @unittest.skipUnless(importlib.util.find_spec("torch"), "requires PyTorch")
    def test_pure_kl_context_distillation_decreases_teacher_kl_and_anchors_background(self):
        import torch
        from confirmatory_v2_converged_ablation import mounted_converged_adapter
        from training_worker import _state, _tensor_digest

        teacher_logits = torch.tensor([[2.0, -1.0, 0.5, 0.0], [0.0, 3.0, -0.5, 1.0]], dtype=torch.bfloat16)
        identical_student = teacher_logits.float().clone().requires_grad_(True)
        zero_kl = csd_mod.weighted_slice_kl_loss(identical_student, teacher_logits, [1.5, 0.5], torch.device("cpu"))
        self.assertAlmostEqual(float(zero_kl.item()), 0.0, places=6)

        perturbed_student = torch.zeros(2, 4, dtype=torch.float32, requires_grad=True)
        pos_kl = csd_mod.weighted_slice_kl_loss(perturbed_student, teacher_logits, [1.6, 0.4], torch.device("cpu"))
        pos_kl.backward()
        self.assertGreater(float(pos_kl.item()), 0.1)
        self.assertGreater(
            float(torch.linalg.vector_norm(perturbed_student.grad[0]).item()),
            float(torch.linalg.vector_norm(perturbed_student.grad[1]).item()),
        )

        class TinyQwen3(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.config = types.SimpleNamespace(
                    model_type="qwen3", num_hidden_layers=1, return_dict=True, use_cache=False
                )
                self.embedding = torch.nn.Embedding(32, 16)
                self.model = torch.nn.Module()
                layer = torch.nn.Module()
                layer.self_attn = torch.nn.Module()
                layer.self_attn.q_proj = torch.nn.Linear(16, 16)
                layer.self_attn.v_proj = torch.nn.Linear(16, 16)
                self.model.layers = torch.nn.ModuleList([layer])
                self.lm_head = torch.nn.Linear(16, 32)

            def forward(self, input_ids, attention_mask=None, use_cache=False, return_dict=True, past_key_values=None):
                hidden = self.embedding(input_ids)
                attn_layer = self.model.layers[0].self_attn
                hidden = hidden + attn_layer.q_proj(hidden) + attn_layer.v_proj(hidden)
                return types.SimpleNamespace(logits=self.lm_head(hidden), past_key_values=None)

        torch.manual_seed(23)
        tiny = TinyQwen3().to(dtype=torch.bfloat16).eval().requires_grad_(False)
        before_sha = _tensor_digest(_state(tiny))

        pos_trajs = []
        for idx_num in range(10):
            s_ids = [1, 2 + (idx_num % 6), 18, 19, 20]
            target_t = torch.zeros(3, 32, dtype=torch.bfloat16)
            target_t[:, 25] = 6.0
            pos_trajs.append({
                "task_id": "P%02d" % (idx_num + 1),
                "source_pool": "seed" if idx_num < 8 else "unlabeled_verified",
                "student_full_ids": s_ids,
                "student_prompt_len": 2,
                "answer_len": 3,
                "target_weights": [1.5, 1.0, 0.5],
                "teacher_answer_logits_bf16": target_t,
            })

        bg_trajs = []
        with torch.no_grad():
            for bg_idx in range(16):
                bg_ids = [1, 10 + (bg_idx % 6), 22, 23, 24]
                inp_t = torch.tensor([bg_ids[:-1]], dtype=torch.long)
                base_sl = tiny(input_ids=inp_t).logits[0, 1:4].detach().clone()
                bg_trajs.append({
                    "task_id": "R%02d" % (bg_idx + 1),
                    "full_ids": bg_ids,
                    "prompt_len": 2,
                    "answer_len": 3,
                    "base_answer_logits_bf16": base_sl,
                })

        out = csd_mod.train_context_self_distilled_adapter(
            tiny,
            pos_trajs,
            bg_trajs,
            hyperparams={
                **csd_mod.CONTEXT_DISTILL_HYPERPARAMS,
                "learning_rate": 0.02,
                "steps": 40,
            },
        )
        self.assertEqual(_tensor_digest(_state(tiny)), before_sha)
        first_pos_kl = sum(st["pos_distill_kl"] for st in out["steps"][:10]) / 10.0
        last_pos_kl = sum(st["pos_distill_kl"] for st in out["steps"][-10:]) / 10.0
        self.assertLess(last_pos_kl, first_pos_kl * 0.4)
        with mounted_converged_adapter(tiny, out) as adapted_mod:
            bg_audit = csd_mod.audit_kl_restraint_on_trajectories(adapted_mod, bg_trajs[:4])
        self.assertLess(bg_audit["mean_token_kl"], 0.25)

    def test_package_zero_oracle_leak_and_code_hygiene(self):
        for fname in (
            "context_self_distillation_ablation.py",
            "test_context_self_distillation_ablation.py",
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

        dep_sources = {
            fname: (REPO_LORA / fname).read_bytes()
            for fname in csd_mod.G9_SOURCE_NAMES
        }
        run_id = "b9" * 16
        pkg = csd_mod.build_gate_g9_ablation_package(
            self.raw_files,
            self.v1_prompts_raw,
            self.v1_structure_raw,
            self.recal_prompts_raw,
            self.worker_raw,
            self.backend_raw,
            dep_sources,
            self.preflight_raw,
            run_id=run_id,
            scoring_run_ids=[("%02x" % idx) + ("ff" * 15) for idx in range(13)],
            ssh_argv_prefix=["/usr/bin/ssh", "-T", "mock-host"],
            ssh_environment={"PATH": "/usr/bin:/bin", "LANG": "C"},
            interpreter="/env/bin/python",
            resolved_interpreter="/opt/cpython/bin/python3.12",
            site_packages="/env/lib/python3.12/site-packages",
            checkpoint_root="/models/Qwen3-8B",
            expected_tokenizer_sha256="7" * 64,
            remote_parent="/tmp/g9-remote",
            local_run_root="/tmp/g9-local/" + run_id,
        )
        bundle = json.loads(bytes.fromhex(pkg["private"]["phase_one_payload_hex"]).decode("ascii"))
        for key_name, val_b64 in bundle.items():
            if key_name == "sources" or key_name in ("backend_b64", "worker_b64"):
                continue
            decoded_text = base64.b64decode(val_b64).decode("utf-8")
            self.assertNotIn("H01", decoded_text)
            self.assertNotIn("incorrect_programs", decoded_text)


if __name__ == "__main__":
    unittest.main()
