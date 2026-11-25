import ast
import base64
import importlib
import importlib.util
import io
import json
from pathlib import Path
import tokenize
import unittest
from unittest import mock

from confirmatory_v2_contract import (
    CONFIRMATORY_FILE_PINS,
    confirmatory_first_round,
    load_confirmatory_dataset,
)
from validation_contract import encoded


REPO_LORA = Path(__file__).resolve().parent

with mock.patch("subprocess.Popen") as import_spawn, mock.patch("os.killpg") as import_killpg:
    try:
        diff_mod = importlib.import_module("diff_focused_lora_ablation")
    except ModuleNotFoundError as imp_err:
        if imp_err.name != "diff_focused_lora_ablation":
            raise
        diff_mod = None
    assert not import_spawn.called
    assert not import_killpg.called


class DecodableMappingTokenizer:
    chat_template = "synthetic-4char-bpe-v1"
    eos_token_id = 2
    pad_token_id = 0

    def __init__(self, texts_for_vocab=()):
        self._id_to_str = {0: "[PAD]", 2: "[EOS]", 3: "<assistant>"}
        self._str_to_id = {"[PAD]": 0, "[EOS]": 2, "<assistant>": 3}
        for ascii_code in range(128):
            ch_s = chr(ascii_code)
            tok_id = 10 + ascii_code
            self._id_to_str[tok_id] = ch_s
            self._str_to_id[ch_s] = tok_id
        for sample_text in texts_for_vocab:
            self._tokenize_into_chunks(sample_text, allow_new=True)

    def get_vocab(self):
        return dict(self._str_to_id)

    def _intern_chunk(self, chunk_str):
        if chunk_str not in self._str_to_id:
            new_id = len(self._str_to_id) + 200
            self._str_to_id[chunk_str] = new_id
            self._id_to_str[new_id] = chunk_str
        return self._str_to_id[chunk_str]

    def _tokenize_into_chunks(self, text_val, allow_new=False):
        out_ids = []
        eos_parts = text_val.split("[EOS]")
        for eos_idx, eos_part in enumerate(eos_parts):
            if eos_idx > 0:
                out_ids.append(self.eos_token_id)
            ast_parts = eos_part.split("<assistant>")
            for ast_idx, seg in enumerate(ast_parts):
                if ast_idx > 0:
                    out_ids.append(3)
                for off in range(0, len(seg), 4):
                    chunk = seg[off:off + 4]
                    out_ids.append(self._intern_chunk(chunk) if allow_new else self._str_to_id[chunk])
        return out_ids

    def __call__(self, text_val, add_special_tokens=False, truncation=False):
        assert add_special_tokens is False and truncation is False
        return {"input_ids": self._tokenize_into_chunks(text_val, allow_new=False)}

    def apply_chat_template(
        self,
        messages,
        *,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
        truncation=False,
    ):
        assert enable_thinking is False and truncation is False
        rendered = ""
        for msg in messages:
            ending = "[EOS]\n" if msg["role"] == "assistant" else "<end>\n"
            rendered += "<" + msg["role"] + ">" + msg["content"] + ending
        if add_generation_prompt:
            rendered += "<assistant>"
        if not tokenize:
            return rendered
        return self(rendered)["input_ids"]

    def decode(self, token_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False):
        return "".join(self._id_to_str[tok_id] for tok_id in token_ids if tok_id != self.eos_token_id)


class DiffFocusedLoRAAblationTests(unittest.TestCase):
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
        cls.sup_by_id = {item["id"]: item for item in cls.dataset["supervision"]}
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
        self.assertIsNotNone(diff_mod, "diff_focused_lora_ablation is absent")

    def test_line_diff_weights_upweight_repairs_and_downweight_cosmetic_valueerror(self):
        v01 = self.sup_by_id["V01"]
        v01_lines, v01_weights = diff_mod.classify_reference_line_weights(
            v01["initial_source"], v01["reference_source"]
        )
        weight_by_stripped_v01 = {ln.strip(): wt for ln, wt in zip(v01_lines, v01_weights)}
        self.assertEqual(weight_by_stripped_v01["live = [it for it in live if it[1] > st]"], 4.0)
        self.assertEqual(weight_by_stripped_v01["if ok:"], 4.0)
        self.assertEqual(weight_by_stripped_v01["return out"], 1.0)

        v04 = self.sup_by_id["V04"]
        v04_lines, v04_weights = diff_mod.classify_reference_line_weights(
            v04["initial_source"], v04["reference_source"]
        )
        weight_by_stripped_v04 = {ln.strip(): wt for ln, wt in zip(v04_lines, v04_weights)}
        self.assertEqual(weight_by_stripped_v04["cur = 0"], 4.0)

        v06 = self.sup_by_id["V06"]
        v06_lines, v06_weights = diff_mod.classify_reference_line_weights(
            v06["initial_source"], v06["reference_source"]
        )
        weight_by_stripped_v06 = {ln.strip(): wt for ln, wt in zip(v06_lines, v06_weights)}
        self.assertEqual(weight_by_stripped_v06['raise ValueError("thr")'], 0.25)
        self.assertEqual(weight_by_stripped_v06["for tag in {it for it in panel if it}:"], 4.0)
        self.assertEqual(weight_by_stripped_v06["counts = {}"], 1.0)

    def test_prepare_diff_focused_training_aligns_first_round_prompt_and_normalizes_weights(self):
        seed_texts = []
        for item in self.dataset["supervision"]:
            msgs = confirmatory_first_round(item, "lora", self.raw_files["supervision.json"])
            seed_texts.append("<user>" + msgs[0]["content"] + "<end>\n<assistant>")
            seed_texts.append("<user>" + msgs[0]["content"] + "<end>\n<assistant>" + item["reference_source"] + "[EOS]\n")
        fake_tok = DecodableMappingTokenizer(seed_texts)
        prepared = diff_mod.prepare_diff_focused_training(fake_tok, self.raw_files["supervision.json"])
        self.assertEqual(len(prepared["rows"]), 8)

        v01_expected_msgs = confirmatory_first_round(
            self.sup_by_id["V01"], "lora", self.raw_files["supervision.json"]
        )
        self.assertNotIn("Failure:", v01_expected_msgs[0]["content"])
        v01_row = prepared["rows"][0]
        self.assertEqual(v01_row["task_id"], "V01")
        self.assertEqual(len(v01_row["target_weights"]), v01_row["target_tokens"])
        mean_weight = sum(v01_row["target_weights"]) / float(len(v01_row["target_weights"]))
        self.assertAlmostEqual(mean_weight, 1.0, places=6)
        self.assertGreater(max(v01_row["target_weights"]), 2.0 * min(v01_row["target_weights"]))

    @unittest.skipUnless(importlib.util.find_spec("torch"), "requires PyTorch")
    def test_weighted_answer_loss_scales_token_gradients_proportionally(self):
        import torch

        logits = torch.zeros(1, 4, 8, dtype=torch.float32, requires_grad=True)
        labels = torch.tensor([[-100, 2, 3, 4]], dtype=torch.long)
        target_weights = [4.0, 1.0, 0.25]
        mean_w = sum(target_weights) / 3.0
        norm_weights = [val_w / mean_w for val_w in target_weights]
        loss_val = diff_mod.weighted_answer_loss(logits, labels, norm_weights)
        loss_val.backward()
        grad_norms = [
            float(torch.linalg.vector_norm(logits.grad[0, pos_idx]).item())
            for pos_idx in (0, 1, 2)
        ]
        self.assertAlmostEqual(grad_norms[0] / grad_norms[1], 4.0, places=4)
        self.assertAlmostEqual(grad_norms[1] / grad_norms[2], 4.0, places=4)

    def test_package_zero_oracle_leak_and_code_hygiene(self):
        for fname in (
            "diff_focused_lora_ablation.py",
            "test_diff_focused_lora_ablation.py",
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
            for fname in diff_mod.G8_SOURCE_NAMES
        }
        run_id = "a8" * 16
        pkg = diff_mod.build_gate_g8_ablation_package(
            self.raw_files,
            self.v1_prompts_raw,
            self.v1_structure_raw,
            self.worker_raw,
            self.backend_raw,
            dep_sources,
            self.preflight_raw,
            run_id=run_id,
            scoring_run_ids=[("%02x" % idx) + ("dd" * 15) for idx in range(13)],
            ssh_argv_prefix=["/usr/bin/ssh", "-T", "mock-host"],
            ssh_environment={"PATH": "/usr/bin:/bin", "LANG": "C"},
            interpreter="/env/bin/python",
            resolved_interpreter="/opt/cpython/bin/python3.12",
            site_packages="/env/lib/python3.12/site-packages",
            checkpoint_root="/models/Qwen3-8B",
            expected_tokenizer_sha256="7" * 64,
            remote_parent="/tmp/g8-remote",
            local_run_root="/tmp/g8-local/" + run_id,
        )
        bundle = json.loads(bytes.fromhex(pkg["private"]["phase_one_payload_hex"]).decode("ascii"))
        for key_name, val_b64 in bundle.items():
            if key_name == "sources" or key_name in ("backend_b64", "worker_b64"):
                continue
            decoded_text = base64.b64decode(val_b64).decode("utf-8")
            self.assertNotIn("H01", decoded_text)
            self.assertNotIn("incorrect_programs", decoded_text)

    def test_phase2_only_package_binds_sealed_phase1_wire_sha256(self):
        p2_pkg = diff_mod.build_gate_g8_phase2_only_package(
            self.raw_files,
            self.worker_raw,
            self.backend_raw,
            self.preflight_raw,
            source_run_id="3aa1b6b795db51b0307864cdefdcbc4f",
            sealed_phase1_wire_sha256="a" * 64,
            sealed_phase1_result_sha256="b" * 64,
            stage_id="c9" * 16,
            scoring_run_ids=[("%02x" % idx) + ("ee" * 15) for idx in range(13)],
            ssh_argv_prefix=["/usr/bin/ssh", "-T", "mock-host"],
            ssh_environment={"PATH": "/usr/bin:/bin", "LANG": "C"},
            resolved_interpreter="/opt/cpython/bin/python3.12",
            local_run_root="/tmp/g8-p2only/" + ("c9" * 16),
        )
        self.assertEqual(p2_pkg["public"]["sealed_phase1_wire_sha256"], "a" * 64)
        self.assertEqual(p2_pkg["public"]["planned_generations"], 0)
        p2_cmd = p2_pkg["private"]["phase_two_ssh_argv"][-1]
        self.assertNotIn("LORA_REMOTE_VALIDATION_ITEM", p2_cmd)
        self.assertIn("row", p2_cmd)
        self.assertIn("done", p2_cmd)


if __name__ == "__main__":
    unittest.main()
