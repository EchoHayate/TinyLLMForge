import ast
import hashlib
import importlib
import io
import json
from pathlib import Path
import tokenize
import unittest
from unittest import mock

from validation_contract import digest, encoded


REPO_LORA = Path(__file__).resolve().parent

with mock.patch("subprocess.Popen") as import_spawn, mock.patch("os.killpg") as import_killpg:
    try:
        conf_contract = importlib.import_module("confirmatory_v2_contract")
    except ModuleNotFoundError as import_err:
        if import_err.name != "confirmatory_v2_contract":
            raise
        conf_contract = None
    assert not import_spawn.called
    assert not import_killpg.called


class ConfirmatoryV2ContractTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(conf_contract, "confirmatory_v2_contract is absent")

    def test_style_and_zero_single_letter_identifiers(self):
        for fname in ("confirmatory_v2_contract.py", "test_confirmatory_v2_contract.py"):
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

    def test_dataset_splits_disjointness_and_matched_information_rendering(self):
        data_dir = REPO_LORA / "data/confirmatory_v2_r1"
        raw_files = {
            fname: (data_dir / fname).read_bytes()
            for fname in conf_contract.CONFIRMATORY_FILE_PINS
        }
        dataset = conf_contract.load_confirmatory_dataset(raw_files)
        self.assertEqual(len(dataset["supervision"]), 8)
        self.assertEqual(len(dataset["prompts"]), 24)
        self.assertEqual(len(dataset["oracles"]), 32)

        prior_entries = set()
        for rel_name in (
            "supervision_v1.json",
            "prompts_v1.json",
            "recalibration_v2_r1/prompts.json",
            "recalibration_v2/prompts.json",
        ):
            for item in json.loads((REPO_LORA / "data" / rel_name).read_text("utf-8")):
                prior_entries.add(item["entry_point"])

        new_entries = {item["entry_point"] for item in dataset["oracles"]}
        self.assertEqual(len(new_entries), 32)
        self.assertTrue(new_entries.isdisjoint(prior_entries))

        tjobs = conf_contract.confirmatory_training_jobs(raw_files["supervision.json"])
        self.assertEqual(len(tjobs), 8)
        self.assertEqual([job["task_id"] for job in tjobs], list(conf_contract.SUPERVISION_IDS))

        first_eval = dataset["prompts"][0]
        msg_base = conf_contract.confirmatory_first_round(first_eval, "base", raw_files["supervision.json"])
        msg_lora = conf_contract.confirmatory_first_round(first_eval, "lora", raw_files["supervision.json"])
        msg_text = conf_contract.confirmatory_first_round(first_eval, "text", raw_files["supervision.json"])
        msg_lora_text = conf_contract.confirmatory_first_round(first_eval, "lora_text", raw_files["supervision.json"])
        self.assertEqual(encoded(msg_base), encoded(msg_lora))
        self.assertEqual(encoded(msg_text), encoded(msg_lora_text))
        self.assertNotEqual(encoded(msg_base), encoded(msg_text))
        self.assertLessEqual(len(msg_text[0]["content"]), 14500)

    def test_validation_batches_and_tokenizer_projection_counts(self):
        data_dir = REPO_LORA / "data/confirmatory_v2_r1"
        raw_files = {
            fname: (data_dir / fname).read_bytes()
            for fname in conf_contract.CONFIRMATORY_FILE_PINS
        }
        worker_raw = (REPO_LORA / "validation_task_worker.py").read_bytes()
        run_ids = [("%02x" % idx) + ("ab" * 15) for idx in range(40)]
        batches = conf_contract.build_confirmatory_validation_batches(raw_files, worker_raw, run_ids)
        self.assertEqual(len(batches), 40)
        self.assertEqual(sum(len(batch["requests"]) for batch in batches), 1280)

        tok_proj = conf_contract.build_confirmatory_tokenizer_projection(raw_files)
        self.assertEqual(len(tok_proj["rows"]), 200)


if __name__ == "__main__":
    unittest.main()
