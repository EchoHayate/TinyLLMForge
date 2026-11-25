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
    load_confirmatory_dataset,
)
from validation_contract import encoded


REPO_LORA = Path(__file__).resolve().parent

with mock.patch("subprocess.Popen") as import_spawn, mock.patch("os.killpg") as import_killpg:
    try:
        hyb_mod = importlib.import_module("mlp_hybrid_lora_ablation")
    except ModuleNotFoundError as imp_err:
        if imp_err.name != "mlp_hybrid_lora_ablation":
            raise
        hyb_mod = None
    assert not import_spawn.called
    assert not import_killpg.called


class MLPHybridLoRAAblationTests(unittest.TestCase):
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
        self.assertIsNotNone(hyb_mod, "mlp_hybrid_lora_ablation is absent")

    def test_qwen3_8b_hybrid_byte_budget_strictly_under_baseline_15mib(self):
        qwen3_8b_bytes = hyb_mod.estimate_qwen3_8b_hybrid_fp32_bytes(rank=4)
        self.assertEqual(qwen3_8b_bytes, 13_959_168)
        self.assertLess(qwen3_8b_bytes, 15_335_424)

    @unittest.skipUnless(importlib.util.find_spec("torch"), "requires PyTorch")
    def test_hybrid_attn_and_mlp_down_proj_train_and_mount_roundtrip(self):
        import torch
        from training_worker import _state, _tensor_digest

        class TinyQwen3WithMLP(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.config = types.SimpleNamespace(
                    model_type="qwen3", num_hidden_layers=3, return_dict=True, use_cache=False
                )
                self.embedding = torch.nn.Embedding(32, 16)
                self.model = torch.nn.Module()
                layers = []
                for _idx in range(3):
                    blk = torch.nn.Module()
                    blk.self_attn = torch.nn.Module()
                    blk.self_attn.q_proj = torch.nn.Linear(16, 16)
                    blk.self_attn.v_proj = torch.nn.Linear(16, 8)
                    blk.mlp = torch.nn.Module()
                    blk.mlp.up_proj = torch.nn.Linear(16, 24)
                    blk.mlp.down_proj = torch.nn.Linear(24, 16)
                    layers.append(blk)
                self.model.layers = torch.nn.ModuleList(layers)
                self.lm_head = torch.nn.Linear(16, 32)

            def forward(self, input_ids, attention_mask=None, use_cache=False, return_dict=True):
                hid = self.embedding(input_ids)
                for blk in self.model.layers:
                    q_out = blk.self_attn.q_proj(hid)
                    v_pad = torch.nn.functional.pad(blk.self_attn.v_proj(hid), (0, 8))
                    mlp_out = blk.mlp.down_proj(torch.nn.functional.silu(blk.mlp.up_proj(hid)))
                    hid = hid + q_out + v_pad + mlp_out
                return types.SimpleNamespace(logits=self.lm_head(hid))

        torch.manual_seed(31)
        tiny = TinyQwen3WithMLP().to(dtype=torch.bfloat16).eval().requires_grad_(False)
        before_sha = _tensor_digest(_state(tiny))

        targets = hyb_mod.discover_hybrid_targets(tiny)
        target_names = [nm for nm, _mod in targets]
        self.assertIn("model.layers.0.self_attn.q_proj", target_names)
        self.assertNotIn("model.layers.0.mlp.down_proj", target_names)
        self.assertIn("model.layers.1.mlp.down_proj", target_names)
        self.assertIn("model.layers.2.mlp.down_proj", target_names)

        rows = []
        for idx_num in range(8):
            seq_ids = [1, 2 + idx_num, 18, 19, 20]
            rows.append({
                "task_id": "V%02d" % (idx_num + 1),
                "input_ids": seq_ids,
                "attention_mask": [1] * 5,
                "labels": [-100, -100, 18, 19, 20],
                "target_weights": [1.5, 1.0, 0.5],
                "prompt_tokens": 2,
                "target_tokens": 3,
                "focused_target_tokens": 1,
            })
        ds = {
            "kind": "LORA_CONFIRMATORY_V2_DIFF_FOCUSED_TRAINING_INPUTS",
            "version": 1,
            "supervision_sha256": CONFIRMATORY_FILE_PINS["supervision.json"],
            "rows": rows,
        }
        from validation_contract import digest
        ds["dataset_sha256"] = digest(ds)

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

        train_out = hyb_mod.train_hybrid_mlp_kl_adapter(
            tiny,
            ds,
            bg_trajs,
            hyperparams={**hyb_mod.MLP_HYBRID_HYPERPARAMS, "learning_rate": 0.02, "steps": 24, "epochs": 3},
        )
        self.assertEqual(_tensor_digest(_state(tiny)), before_sha)
        self.assertLess(
            train_out["epoch_mean_weighted_sup_losses"][-1],
            train_out["epoch_mean_weighted_sup_losses"][0],
        )
        with hyb_mod.mounted_hybrid_adapter(tiny, train_out) as mounted_mod:
            test_in = torch.tensor([[1, 3, 18]], dtype=torch.long)
            adapted_logits = mounted_mod(input_ids=test_in).logits.detach().clone()
        unadapted_logits = tiny(input_ids=test_in).logits.detach().clone()
        self.assertFalse(torch.allclose(adapted_logits, unadapted_logits))
        self.assertEqual(_tensor_digest(_state(tiny)), before_sha)

    def test_package_zero_oracle_leak_and_code_hygiene(self):
        for fname in (
            "mlp_hybrid_lora_ablation.py",
            "test_mlp_hybrid_lora_ablation.py",
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
            for fname in hyb_mod.G10_SOURCE_NAMES
        }
        run_id = "c0" * 16
        pkg = hyb_mod.build_gate_g10_ablation_package(
            self.raw_files,
            self.v1_prompts_raw,
            self.v1_structure_raw,
            self.worker_raw,
            self.backend_raw,
            dep_sources,
            self.preflight_raw,
            run_id=run_id,
            scoring_run_ids=[("%02x" % idx) + ("ab" * 15) for idx in range(13)],
            ssh_argv_prefix=["/usr/bin/ssh", "-T", "mock-host"],
            ssh_environment={"PATH": "/usr/bin:/bin", "LANG": "C"},
            interpreter="/env/bin/python",
            resolved_interpreter="/opt/cpython/bin/python3.12",
            site_packages="/env/lib/python3.12/site-packages",
            checkpoint_root="/models/Qwen3-8B",
            expected_tokenizer_sha256="7" * 64,
            remote_parent="/tmp/g10-remote",
            local_run_root="/tmp/g10-local/" + run_id,
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
