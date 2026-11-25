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
        sys_mod = importlib.import_module("bf16_merge_and_kv_benchmark")
    except ModuleNotFoundError as imp_err:
        if imp_err.name != "bf16_merge_and_kv_benchmark":
            raise
        sys_mod = None
    assert not import_spawn.called
    assert not import_killpg.called


class BF16MergeAndKVBenchmarkTests(unittest.TestCase):
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
        self.assertIsNotNone(sys_mod, "bf16_merge_and_kv_benchmark is absent")

    @unittest.skipUnless(importlib.util.find_spec("torch"), "requires PyTorch")
    def test_pack_bf16_and_reversible_inplace_merge_matches_wrapper_and_restores_bits(self):
        import torch
        from confirmatory_v2_converged_ablation import mounted_converged_adapter
        from training_worker import _state, _tensor_digest

        class TinyQwen3(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.config = types.SimpleNamespace(
                    model_type="qwen3", num_hidden_layers=2, return_dict=True, use_cache=False
                )
                self.embedding = torch.nn.Embedding(32, 16)
                self.model = torch.nn.Module()
                layers = []
                for _idx in range(2):
                    blk = torch.nn.Module()
                    blk.self_attn = torch.nn.Module()
                    blk.self_attn.q_proj = torch.nn.Linear(16, 16, bias=False)
                    blk.self_attn.v_proj = torch.nn.Linear(16, 8, bias=False)
                    layers.append(blk)
                self.model.layers = torch.nn.ModuleList(layers)
                self.lm_head = torch.nn.Linear(16, 32, bias=False)

            def forward(self, input_ids, attention_mask=None, use_cache=False, return_dict=True):
                hid = self.embedding(input_ids)
                for blk in self.model.layers:
                    q_out = blk.self_attn.q_proj(hid)
                    v_pad = torch.nn.functional.pad(blk.self_attn.v_proj(hid), (0, 8))
                    hid = hid + q_out + v_pad
                return types.SimpleNamespace(logits=self.lm_head(hid))

        torch.manual_seed(41)
        tiny = TinyQwen3().to(dtype=torch.bfloat16).eval().requires_grad_(False)
        orig_sha = _tensor_digest(_state(tiny))
        orig_param_bytes = sum(par.numel() * par.element_size() for par in tiny.parameters())

        fp32_adapter = {}
        for l_idx in range(2):
            q_p = "model.layers.%d.self_attn.q_proj" % l_idx
            v_p = "model.layers.%d.self_attn.v_proj" % l_idx
            fp32_adapter[q_p + ".adapter_a"] = torch.randn(8, 16, dtype=torch.float32) * 0.1
            fp32_adapter[q_p + ".adapter_b"] = torch.randn(16, 8, dtype=torch.float32) * 0.1
            fp32_adapter[v_p + ".adapter_a"] = torch.randn(8, 16, dtype=torch.float32) * 0.1
            fp32_adapter[v_p + ".adapter_b"] = torch.randn(8, 8, dtype=torch.float32) * 0.1

        fp32_bytes = sum(ten.numel() * ten.element_size() for ten in fp32_adapter.values())
        bf16_pack = sys_mod.pack_adapter_to_bf16(fp32_adapter)
        self.assertEqual(bf16_pack["bf16_bytes"], fp32_bytes // 2)

        train_stub = {
            "base_sha256": orig_sha,
            "adapter": fp32_adapter,
            "adapter_sha256": _tensor_digest(fp32_adapter),
        }
        probe_in = torch.tensor([[1, 5, 9, 14]], dtype=torch.long)
        with mounted_converged_adapter(tiny, train_stub) as wrapped_mod:
            wrapped_logits = wrapped_mod(input_ids=probe_in).logits.float().detach().clone()

        with sys_mod.merged_bf16_adapter(tiny, bf16_pack, alpha=16.0, rank=8) as merged_mod:
            merged_param_bytes = sum(par.numel() * par.element_size() for par in merged_mod.parameters())
            self.assertEqual(merged_param_bytes, orig_param_bytes)
            for _nm, mod_it in merged_mod.named_modules():
                self.assertNotEqual(type(mod_it).__name__, "LoRAProjection")
            merged_logits = merged_mod(input_ids=probe_in).logits.float().detach().clone()

        max_diff = float((wrapped_logits - merged_logits).abs().max().item())
        self.assertLess(max_diff, 0.05)
        self.assertEqual(_tensor_digest(_state(tiny)), orig_sha)

    def test_package_zero_oracle_leak_and_code_hygiene(self):
        for fname in (
            "bf16_merge_and_kv_benchmark.py",
            "test_bf16_merge_and_kv_benchmark.py",
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
            for fname in sys_mod.G11_SOURCE_NAMES
        }
        run_id = "d1" * 16
        pkg = sys_mod.build_gate_g11_system_package(
            self.raw_files,
            self.v1_prompts_raw,
            self.v1_structure_raw,
            self.worker_raw,
            self.backend_raw,
            dep_sources,
            self.preflight_raw,
            run_id=run_id,
            scoring_run_ids=[("%02x" % idx) + ("cd" * 15) for idx in range(13)],
            ssh_argv_prefix=["/usr/bin/ssh", "-T", "mock-host"],
            ssh_environment={"PATH": "/usr/bin:/bin", "LANG": "C"},
            interpreter="/env/bin/python",
            resolved_interpreter="/opt/cpython/bin/python3.12",
            site_packages="/env/lib/python3.12/site-packages",
            checkpoint_root="/models/Qwen3-8B",
            expected_tokenizer_sha256="7" * 64,
            remote_parent="/tmp/g11-remote",
            local_run_root="/tmp/g11-local/" + run_id,
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
