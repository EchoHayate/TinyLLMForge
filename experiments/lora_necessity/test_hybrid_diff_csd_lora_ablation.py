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
        hyb_mod = importlib.import_module("hybrid_diff_csd_lora_ablation")
    except ModuleNotFoundError as imp_err:
        if imp_err.name != "hybrid_diff_csd_lora_ablation":
            raise
        hyb_mod = None
    assert not import_spawn.called
    assert not import_killpg.called


class MockCodeTokenizer:
    eos_token_id = 999

    def __init__(self):
        self._str_to_id = {}
        self._id_to_str = {}

    def _tok_char(self, ch_val):
        if ch_val not in self._str_to_id:
            nxt_id = len(self._str_to_id) + 1
            self._str_to_id[ch_val] = nxt_id
            self._id_to_str[nxt_id] = ch_val
        return self._str_to_id[ch_val]

    def encode_answer(self, text_val):
        return [self._tok_char(ch_val) for ch_val in text_val] + [self.eos_token_id]

    def decode(self, token_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False):
        return "".join(self._id_to_str[tok_id] for tok_id in token_ids if tok_id != self.eos_token_id)


class HybridDiffCSDAblationTests(unittest.TestCase):
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
        self.assertIsNotNone(hyb_mod, "hybrid_diff_csd_lora_ablation is absent")

    def test_surgical_zero_scaffold_mask_strictly_isolates_bugfix_lines(self):
        init_src = (
            "def reset_demo(items):\n"
            "    if items is None:\n"
            "        raise ValueError(\"bad items input\")\n"
            "    cache = {}\n"
            "    cache.clear()\n"
            "    return sorted(cache)\n"
        )
        ref_src = (
            "def reset_demo(items):\n"
            "    if items is None:\n"
            "        raise ValueError(\"bad\")\n"
            "    cache = {}\n"
            "    cache.pop(\"k\", None)\n"
            "    return sorted(cache)\n"
        )
        tok = MockCodeTokenizer()
        target_ids = tok.encode_answer(ref_src)
        surg_wts, focused_cnt = hyb_mod.build_surgical_diff_token_weights(
            tok, init_src, ref_src, target_ids
        )
        self.assertEqual(len(surg_wts), len(target_ids))
        self.assertGreater(focused_cnt, 5)
        self.assertAlmostEqual(sum(surg_wts) / float(len(surg_wts)), 1.0, places=6)
        self.assertEqual(surg_wts[-1], 0.0)
        header_len = len("def reset_demo(items):\n")
        for idx_pos in range(header_len):
            self.assertEqual(surg_wts[idx_pos], 0.0)
        tail_start = ref_src.index("    return sorted(cache)\n")
        for idx_pos in range(tail_start, len(ref_src)):
            self.assertEqual(surg_wts[idx_pos], 0.0)
        diff_start = ref_src.index("    cache.pop")
        self.assertGreater(surg_wts[diff_start + 5], 1.0)

    @unittest.skipUnless(importlib.util.find_spec("torch"), "requires PyTorch")
    def test_z_hybrid_surgical_ce_and_soft_kl_training_preserves_base_and_reduces_losses(self):
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

            def forward(self, input_ids, attention_mask=None, use_cache=False, return_dict=True, past_key_values=None):
                hidden = self.embedding(input_ids)
                attn_layer = self.model.layers[0].self_attn
                hidden = hidden + attn_layer.q_proj(hidden) + attn_layer.v_proj(hidden)
                return types.SimpleNamespace(logits=self.lm_head(hidden), past_key_values=None)

        torch.manual_seed(29)
        tiny = TinyQwen3().to(dtype=torch.bfloat16).eval().requires_grad_(False)
        before_sha = _tensor_digest(_state(tiny))

        pos_trajs = []
        for idx_num in range(10):
            s_ids = [1, 2 + (idx_num % 6), 18, 25, 20]
            target_t = torch.zeros(3, 32, dtype=torch.bfloat16)
            target_t[0, 18] = 4.0
            target_t[1, 25] = 6.0
            target_t[2, 20] = 4.0
            is_seed = idx_num < 8
            pos_trajs.append({
                "task_id": "P%02d" % (idx_num + 1),
                "source_pool": "seed" if is_seed else "unlabeled_recal",
                "student_full_ids": s_ids,
                "student_prompt_len": 2,
                "answer_len": 3,
                "target_ids": [18, 25, 20],
                "target_weights": [0.5, 2.0, 0.5],
                "surgical_ce_weights": [0.0, 3.0, 0.0] if is_seed else [0.0, 0.0, 0.0],
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

        out = hyb_mod.train_hybrid_diff_csd_adapter(
            tiny,
            pos_trajs,
            bg_trajs,
            hyperparams={
                **hyb_mod.HYBRID_DIFF_CSD_HYPERPARAMS,
                "learning_rate": 0.02,
                "steps": 40,
            },
        )
        self.assertEqual(_tensor_digest(_state(tiny)), before_sha)
        seed_steps = [st for st in out["steps"] if st["source_pool"] == "seed"]
        self.assertGreater(len(seed_steps), 10)
        self.assertLess(seed_steps[-1]["surgical_ce_loss"], seed_steps[0]["surgical_ce_loss"] * 0.5)
        extra_steps = [st for st in out["steps"] if st["source_pool"] != "seed"]
        self.assertTrue(all(st["surgical_ce_loss"] == 0.0 for st in extra_steps))
        with mounted_converged_adapter(tiny, out) as adapted_mod:
            bg_audit = hyb_mod.audit_kl_restraint_on_trajectories(adapted_mod, bg_trajs[:4])
        self.assertLess(bg_audit["mean_token_kl"], 0.25)

    def test_package_zero_oracle_leak_and_code_hygiene(self):
        for fname in (
            "hybrid_diff_csd_lora_ablation.py",
            "test_hybrid_diff_csd_lora_ablation.py",
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
            for fname in hyb_mod.G12_SOURCE_NAMES
        }
        run_id = "c2" * 16
        pkg = hyb_mod.build_gate_g12_ablation_package(
            self.raw_files,
            self.v1_prompts_raw,
            self.v1_structure_raw,
            self.recal_prompts_raw,
            self.worker_raw,
            self.backend_raw,
            dep_sources,
            self.preflight_raw,
            run_id=run_id,
            scoring_run_ids=[("%02x" % idx) + ("ee" * 15) for idx in range(13)],
            ssh_argv_prefix=["/usr/bin/ssh", "-T", "mock-host"],
            ssh_environment={"PATH": "/usr/bin:/bin", "LANG": "C"},
            interpreter="/env/bin/python",
            resolved_interpreter="/opt/cpython/bin/python3.12",
            site_packages="/env/lib/python3.12/site-packages",
            checkpoint_root="/models/Qwen3-8B",
            expected_tokenizer_sha256="7" * 64,
            remote_parent="/tmp/g12-remote",
            local_run_root="/tmp/g12-local/" + run_id,
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
