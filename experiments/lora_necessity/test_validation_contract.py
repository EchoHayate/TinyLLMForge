import copy
import importlib
import json
from pathlib import Path
import unittest


class ValidationContractTests(unittest.TestCase):
    def setUp(self):
        self.assertTrue(Path(__file__).with_name("validation_contract.py").is_file(),
                        "validation contract not implemented")
        self.module = importlib.import_module("validation_contract")
        self.data = Path(__file__).with_name("data")
        self.source = (self.data / "task_sources_v1.json").read_bytes()
        self.manifest = (self.data / "manifest_v1.json").read_bytes()

    def test_fixed_schedule_and_order(self):
        plan = self.module.build_schedule(self.source, self.manifest)
        self.assertEqual(len(plan["rows"]), 1416)
        self.assertEqual(plan["rows"][0]["task_id"], "D01")
        self.assertEqual(plan["rows"][0]["role"], "reference")
        self.assertEqual([row["ordinal"] for row in plan["rows"]], list(range(1416)))
        self.assertEqual(sum(row["role"] == "reference" for row in plan["rows"]), 354)
        self.assertEqual(sum(row["witness_prediction"] is not None
                             for row in plan["rows"]), 132)

    def test_pin_rejects_tampering_and_reformatted_bytes(self):
        for raw in (self.source + b"\n", self.source.replace(b"count_before", b"count_after")):
            with self.subTest(raw_length=len(raw)), self.assertRaises(ValueError):
                self.module.build_schedule(raw, self.manifest)
        changed = json.loads(self.manifest)
        changed["isolated"] = True
        with self.assertRaises(ValueError):
            self.module.build_schedule(self.source, self.module.encoded(changed))

    def test_work_requests_have_no_private_fields_or_aliases(self):
        requests = self.module.worker_requests(self.source, self.manifest)
        for request in requests:
            self.assertEqual(set(request), {"request_id", "entry_point", "source", "args"})
        original = copy.deepcopy(requests)
        requests[0]["args"].append("sentinel")
        self.assertEqual(self.module.worker_requests(self.source, self.manifest), original)

    def test_reference_jobs_are_private_answer_only_projection(self):
        jobs = self.module.reference_jobs(self.source, self.manifest)
        self.assertEqual(len(jobs), 44)
        self.assertTrue(all(set(job) == {"id", "reference_source"} for job in jobs))
        self.assertEqual(self.module.digest(jobs),
                         "2333003d187a1f12618b4505ca3522b7c2a20e3295c48ae835f3e879ed601cf3")

    def test_strict_json(self):
        for raw in (b'{"a":1,"a":2}', b"NaN", b"Infinity", b"1e999", b"{}\n{}", b"\xff"):
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                self.module.parse_json(raw)
        with self.assertRaises(ValueError):
            self.module.parse_json(b"[]", limit=1)
        with self.assertRaises(ValueError):
            self.module.load_frozen("unknown.json", b"{}")
        self.assertNotEqual(self.module.digest(True), self.module.digest(1))
