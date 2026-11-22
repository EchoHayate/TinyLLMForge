import ast
import copy
import hashlib
import importlib
import importlib.util
import io
from pathlib import Path
import tempfile
import tokenize
import types
import unittest
from unittest import mock

from confirmatory_v2_contract import CONFIRMATORY_FILE_PINS, load_confirmatory_dataset
from validation_contract import digest, encoded


REPO_LORA = Path(__file__).resolve().parent

with mock.patch("subprocess.Popen") as import_spawn, mock.patch("os.killpg") as import_killpg:
    try:
        conv_mod = importlib.import_module("confirmatory_v2_converged_ablation")
    except ModuleNotFoundError as imp_err:
        if imp_err.name != "confirmatory_v2_converged_ablation":
            raise
        conv_mod = None
    assert not import_spawn.called
    assert not import_killpg.called


class ConvergedAblationTests(unittest.TestCase):
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
        self.assertIsNotNone(conv_mod, "confirmatory_v2_converged_ablation is absent")

    def test_style_and_zero_single_letter_identifiers(self):
        for fname in (
            "confirmatory_v2_converged_ablation.py",
            "test_confirmatory_v2_converged_ablation.py",
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

    def test_converged_schedule_26_episodes_and_zero_oracle_leak(self):
        run_id = "67" * 16
        sched = conv_mod.build_converged_ablation_schedule(
            self.raw_files["prompts.json"],
            self.raw_files["supervision.json"],
            run_id=run_id,
        )
        self.assertEqual(len(sched["episodes"]), 26)
        self.assertEqual(sched["episodes"][0]["arm"], "base_canary")
        self.assertEqual(sched["episodes"][1]["arm"], "base_canary")
        self.assertEqual(sum(ep["arm"] == "lora_conv" for ep in sched["episodes"]), 24)

        dep_sources = {
            fname: (REPO_LORA / fname).read_bytes()
            for fname in conv_mod.ABLATION_SOURCE_NAMES
        }
        pkg = conv_mod.build_converged_ablation_package(
            self.raw_files,
            self.worker_raw,
            self.backend_raw,
            dep_sources,
            self.preflight_raw,
            run_id=run_id,
            scoring_run_ids=[("%02x" % idx) + ("aa" * 15) for idx in range(13)],
            ssh_argv_prefix=["/usr/bin/ssh", "-T", "rl-server"],
            ssh_environment={"PATH": "/usr/bin:/bin", "LANG": "C"},
            interpreter="/env/bin/python",
            resolved_interpreter="/opt/cpython/bin/python3.12",
            site_packages="/env/lib/python3.12/site-packages",
            checkpoint_root="/models/Qwen3-8B",
            expected_tokenizer_sha256="7" * 64,
            remote_parent="/tmp/conv-remote",
            local_run_root="/tmp/conv-local/" + run_id,
        )
        p1_str = bytes.fromhex(pkg["private"]["phase_one_payload_hex"]).decode("ascii")
        self.assertNotIn("H01", p1_str)
        self.assertNotIn("incorrect_programs", p1_str)

    @unittest.skipUnless(importlib.util.find_spec("torch"), "requires PyTorch")
    def test_pytorch_96_step_converged_training_drives_loss_near_zero(self):
        import torch
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

        torch.manual_seed(11)
        tiny = TinyQwen3().to(dtype=torch.bfloat16).eval().requires_grad_(False)
        before_base = _tensor_digest(_state(tiny))
        rows = [
            {
                "task_id": "V%02d" % idx,
                "input_ids": [idx, idx + 8, idx + 16, 2],
                "attention_mask": [1, 1, 1, 1],
                "labels": [-100, -100, idx + 16, 2],
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
        res = conv_mod.train_converged_ablation_adapter(tiny, ds)
        self.assertEqual(_tensor_digest(_state(tiny)), before_base)
        self.assertEqual(len(res["steps"]), 96)
        first_epoch = sum(st["loss"] for st in res["steps"][:8]) / 8.0
        last_epoch = sum(st["loss"] for st in res["steps"][-8:]) / 8.0
        self.assertLess(last_epoch, first_epoch)


if __name__ == "__main__":
    unittest.main()
