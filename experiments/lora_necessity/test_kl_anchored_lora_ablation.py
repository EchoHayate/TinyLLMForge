import ast
import base64
import copy
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
    TRANSFER_IDS,
    load_confirmatory_dataset,
)
from validation_contract import encoded


REPO_LORA = Path(__file__).resolve().parent

with mock.patch("subprocess.Popen") as import_spawn, mock.patch("os.killpg") as import_killpg:
    try:
        kl_mod = importlib.import_module("kl_anchored_lora_ablation")
    except ModuleNotFoundError as imp_err:
        if imp_err.name != "kl_anchored_lora_ablation":
            raise
        kl_mod = None
    assert not import_spawn.called
    assert not import_killpg.called


class KLAnchoredLoRAAblationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        conf_dir = REPO_LORA / "data/confirmatory_v2_r1"
        cls.raw_files = {
            fname: (conf_dir / fname).read_bytes()
            for fname in CONFIRMATORY_FILE_PINS
        }
        cls.v1_prompts_raw = (REPO_LORA / "data/prompts_v1.json").read_bytes()
        cls.v1_structure_raw = (REPO_LORA / "data/structure_v1.json").read_bytes()
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
        self.assertIsNotNone(kl_mod, "kl_anchored_lora_ablation is absent")

    def test_style_and_zero_single_letter_identifiers(self):
        for fname in (
            "kl_anchored_lora_ablation.py",
            "test_kl_anchored_lora_ablation.py",
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

    def test_schedule_and_package_zero_oracle_leak(self):
        run_id = "91" * 16
        sched = kl_mod.build_kl_anchored_schedule(
            self.raw_files["prompts.json"],
            self.raw_files["supervision.json"],
            run_id=run_id,
        )
        self.assertEqual(len(sched["episodes"]), 26)
        self.assertEqual(sched["episodes"][0]["arm"], "base_canary")
        self.assertEqual(sched["episodes"][1]["arm"], "base_canary")
        self.assertEqual(sum(ep["arm"] == "lora_kl" for ep in sched["episodes"]), 24)

        dep_sources = {
            fname: (REPO_LORA / fname).read_bytes()
            for fname in kl_mod.G7_SOURCE_NAMES
        }
        pkg = kl_mod.build_gate_g7_ablation_package(
            self.raw_files,
            self.v1_prompts_raw,
            self.v1_structure_raw,
            self.worker_raw,
            self.backend_raw,
            dep_sources,
            self.preflight_raw,
            run_id=run_id,
            scoring_run_ids=[("%02x" % idx) + ("cc" * 15) for idx in range(13)],
            ssh_argv_prefix=["/usr/bin/ssh", "-T", "mock-host"],
            ssh_environment={"PATH": "/usr/bin:/bin", "LANG": "C"},
            interpreter="/env/bin/python",
            resolved_interpreter="/opt/cpython/bin/python3.12",
            site_packages="/env/lib/python3.12/site-packages",
            checkpoint_root="/models/Qwen3-8B",
            expected_tokenizer_sha256="7" * 64,
            remote_parent="/tmp/g7-remote",
            local_run_root="/tmp/g7-local/" + run_id,
        )
        bundle = json.loads(bytes.fromhex(pkg["private"]["phase_one_payload_hex"]).decode("ascii"))
        for key_name, val_b64 in bundle.items():
            if key_name == "sources":
                continue
            if key_name.endswith("_b64") and key_name not in ("backend_b64", "worker_b64"):
                decoded_text = base64.b64decode(val_b64).decode("utf-8")
                self.assertNotIn("H01", decoded_text)
                self.assertNotIn("incorrect_programs", decoded_text)

    @unittest.skipUnless(importlib.util.find_spec("torch"), "requires PyTorch")
    def test_pytorch_kl_anchor_suppresses_background_drift_while_fitting_supervision(self):
        import torch
        from confirmatory_v2_converged_ablation import mounted_converged_adapter
        from training_worker import _state, _tensor_digest

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

            def forward(self, input_ids, attention_mask=None, use_cache=False, return_dict=True):
                hidden = self.embedding(input_ids)
                attn_layer = self.model.layers[0].self_attn
                hidden = hidden + attn_layer.q_proj(hidden) + attn_layer.v_proj(hidden)
                return types.SimpleNamespace(logits=self.lm_head(hidden))

        torch.manual_seed(17)
        tiny = TinyQwen3().to(dtype=torch.bfloat16).eval().requires_grad_(False)
        before_sha = _tensor_digest(_state(tiny))

        sup_rows = []
        for idx_num in range(8):
            seq_ids = [1, 2 + idx_num, 18, 19, 20]
            sup_rows.append({
                "task_id": "V%02d" % (idx_num + 1),
                "input_ids": seq_ids,
                "attention_mask": [1] * len(seq_ids),
                "labels": [-100, -100, 18, 19, 20],
                "supervised_tokens": 3,
            })
        sup_dataset = {"rows": sup_rows}

        bg_anchors = []
        with torch.no_grad():
            for bg_idx in range(16):
                bg_ids = [1, 10 + (bg_idx % 6), 22, 23, 24]
                inp_t = torch.tensor([bg_ids], dtype=torch.long)
                base_logits = tiny(input_ids=inp_t).logits[0, 1:4].detach().clone()
                bg_anchors.append({
                    "task_id": "R%02d" % (bg_idx + 1),
                    "full_ids": bg_ids,
                    "prompt_len": 2,
                    "answer_len": 3,
                    "base_answer_logits_bf16": base_logits,
                })

        with mock.patch("kl_anchored_lora_ablation.validate_confirmatory_training", return_value="d" * 64):
            unconstrained = kl_mod.train_kl_anchored_adapter(
                tiny,
                sup_dataset,
                bg_anchors,
                hyperparams={**kl_mod.KL_ANCHORED_HYPERPARAMS, "learning_rate": 0.02, "lambda_kl": 0.0},
            )
            anchored = kl_mod.train_kl_anchored_adapter(
                tiny,
                sup_dataset,
                bg_anchors,
                hyperparams={**kl_mod.KL_ANCHORED_HYPERPARAMS, "learning_rate": 0.02, "lambda_kl": 2.0},
            )

        self.assertEqual(_tensor_digest(_state(tiny)), before_sha)
        self.assertLess(anchored["epoch_mean_sup_losses"][-1], anchored["epoch_mean_sup_losses"][0] * 0.25)

        with mounted_converged_adapter(tiny, unconstrained) as mod_unc:
            unc_audit = kl_mod.audit_kl_restraint_on_trajectories(mod_unc, bg_anchors[:4])
        with mounted_converged_adapter(tiny, anchored) as mod_anc:
            anc_audit = kl_mod.audit_kl_restraint_on_trajectories(mod_anc, bg_anchors[:4])

        self.assertLess(anc_audit["mean_token_kl"], unc_audit["mean_token_kl"] * 0.2)


if __name__ == "__main__":
    unittest.main()
