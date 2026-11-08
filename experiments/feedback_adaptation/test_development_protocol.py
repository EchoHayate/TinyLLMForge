import copy
import importlib
import json
import pathlib
import tempfile
import unittest


RUN_ID = "8" * 32


def model_event(text="A", **changes):
    return {
        "kind": "model", "status": "completed", "text": text,
        "terminated": True, "input_ids": [1], "output_ids": [2, 3],
        "wall_ns": 5, **changes,
    }


def bind_events(protocol, cell, method, events):
    bound = []
    for event in events:
        if event["kind"] == "model":
            request = protocol.model_request(cell, method, bound, RUN_ID)
            event = {**event, "request_id": request["id"], "request_sha256": protocol.checksum(request)}
        bound.append(event)
    return bound


class ProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.protocol = importlib.import_module("development_protocol")

    def setUp(self):
        self.manifest = self.protocol.build_manifest(RUN_ID)
        self.cell = self.manifest["cells"][0]

    def tool(self, method="active_verify", **changes):
        return {
            "kind": "tool",
            "request_id": self.protocol.tool_request(self.cell, method, RUN_ID)["request_id"],
            "status": "completed", "executed_cases": 2, "wall_ns": 10,
            "results": [
                {**check, "expected": 1, "actual": 1, "status": "ok"}
                for check in self.cell["view"]["check_inputs"]
            ], **changes,
        }

    def step(self, method, events):
        return self.protocol.next_step(self.cell, method, bind_events(self.protocol, self.cell, method, events), RUN_ID)

    def test_manifest_counts_freeze_and_independent_views(self):
        self.assertEqual(len(self.manifest["cells"]), 24)
        self.assertEqual(len(self.manifest["schedule"]), 120)
        self.assertEqual(self.manifest["budget"]["model_calls"], 144)
        self.assertEqual(self.manifest["budget"]["tool_calls"], 48)
        self.assertEqual(len({cell["episode_id"] for cell in self.manifest["cells"]}), 24)
        self.assertEqual(sum(cell["condition"] == "none" for cell in self.manifest["cells"]), 8)
        self.assertEqual(len({tuple(cell["methods"]) for cell in self.manifest["cells"]}), 5)
        original = self.protocol.manifest_digest(self.manifest)
        self.manifest["cells"][0]["view"]["check_inputs"][0]["input"] = "changed"
        with self.assertRaises(ValueError):
            self.protocol.manifest_digest(self.manifest)
        self.assertEqual(original, self.protocol.manifest_digest(self.protocol.build_manifest(RUN_ID)))

    def test_requests_have_only_public_content_and_independent_copy(self):
        request = self.protocol.model_request(self.cell, "active_verify", [], RUN_ID)
        self.assertEqual(set(request), {"id", "messages"})
        self.assertEqual(len(request["messages"]), 1)
        text = request["messages"][0]["content"]
        for hidden in ("family_id", "verification_information", "score_ids", "reference_note",
                       "pass_counts", "dev-001", RUN_ID, "strict-record-highs-v1"):
            self.assertNotIn(hidden, text)
        self.assertIn("VERIFY", text)
        self.assertNotIn('"expected"', text)
        request["messages"][0]["content"] = "mutated"
        self.assertIn("VERIFY", self.protocol.model_request(
            self.cell, "active_verify", [], RUN_ID
        )["messages"][0]["content"])

    def test_keep_and_mandatory_tool(self):
        self.assertTrue(self.step("keep", [])["complete"])
        self.assertEqual(self.step("always_verify", [])["next"], "tool")
        with self.assertRaises(ValueError):
            self.step("always_verify", [model_event()])
        state = self.step("always_verify", [self.tool("always_verify"), model_event()])
        self.assertTrue(state["complete"])
        self.assertEqual(state["model_calls"], 1)
        self.assertEqual(state["tool_calls"], 1)

    def test_direct_and_active_early_stop(self):
        for method in ("direct", "active_verify"):
            state = self.step(method, [model_event("B")])
            self.assertEqual(state["selected"], "B")
            self.assertEqual(state["tool_calls"], 0)
            with self.assertRaises(ValueError):
                self.step(method, [model_event("B"), model_event("A")])

    def test_active_verification_and_failed_tool(self):
        events = [model_event("VERIFY")]
        self.assertEqual(self.step("active_verify", events)["next"], "tool")
        failed = self.tool(status="timeout", results=[], executed_cases=None)
        events.append(failed)
        request = self.protocol.model_request(
            self.cell, "active_verify", bind_events(self.protocol, self.cell, "active_verify", events), RUN_ID
        )
        self.assertIn("timeout", request["messages"][0]["content"])
        self.assertIn("not proof", request["messages"][0]["content"])
        state = self.step("active_verify", events + [model_event("C")])
        self.assertEqual(state["selected"], "C")
        self.assertEqual(state["tool_calls"], 1)
        self.assertIsNone(state["executed_cases"])
        self.assertEqual(self.step("active_verify", events + [model_event("VERIFY")])["status"], "invalid")

    def test_rethink_has_no_new_evidence(self):
        self.assertEqual(self.step("rethink", [model_event()])["next"], "model")
        request = self.protocol.model_request(
            self.cell, "rethink", bind_events(self.protocol, self.cell, "rethink", [model_event()]), RUN_ID
        )
        self.assertIn("tentative", request["messages"][0]["content"])
        self.assertIn("no new evidence", request["messages"][0]["content"])
        self.assertTrue(self.step("rethink", [model_event(), model_event("C")])["complete"])
        with self.assertRaises(ValueError):
            self.step("rethink", [model_event(), self.tool()])

    def test_invalid_truncated_and_error_do_not_retry(self):
        for event, expected in (
            (model_event("Answer: A"), "invalid"),
            (model_event("A", terminated=False), "truncated"),
            (model_event("", status="error", terminated=False, input_ids=None, output_ids=None), "error"),
            (model_event("VERIFY"), "invalid"),
        ):
            state = self.step("direct", [event])
            self.assertTrue(state["complete"])
            self.assertEqual(state["status"], expected)
            self.assertIsNone(state["selected"])
            self.assertIsNotNone(state["deployed"])

    def test_tool_binding_and_typed_rows(self):
        bad = self.tool(request_id="0" * 64)
        with self.assertRaises(ValueError):
            self.step("active_verify", [model_event("VERIFY"), bad])
        for mutate in (
            lambda event: event["results"][0].update(actual=True),
            lambda event: event["results"][0].update(input="wrong"),
            lambda event: event.update(results=[]),
            lambda event: event.update(executed_cases=True),
        ):
            event = self.tool()
            mutate(event)
            with self.assertRaises(ValueError):
                self.step("active_verify", [model_event("VERIFY"), event])

    def test_no_terminal_model_request_or_foreign_method(self):
        with self.assertRaises(ValueError):
            self.protocol.model_request(self.cell, "keep", [], RUN_ID)
        with self.assertRaises(ValueError):
            self.step("invented", [])
        with self.assertRaises(ValueError):
            self.protocol.build_manifest("not-a-run")

    def test_unbound_model_event_is_rejected(self):
        with self.assertRaises(ValueError):
            self.protocol.next_step(self.cell, "direct", [model_event()], RUN_ID)


class ScoreTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.protocol = importlib.import_module("development_protocol")
        import development_cpu
        cls.directory = tempfile.TemporaryDirectory()
        root = pathlib.Path(cls.directory.name) / "audit"
        development_cpu.run(root, "audit")
        cls.audit = {
            "manifest": json.loads((root / "manifest.json").read_text()),
            "observation": json.loads((root / "stdout.json").read_text()),
        }
        cls.manifest = cls.protocol.build_manifest(RUN_ID)
        cls.digest = cls.protocol.manifest_digest(cls.manifest)

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def record(self, cell, method, events, wall=20):
        return {"episode_id": cell["episode_id"], "method": method,
                "manifest_sha256": self.digest, "events": bind_events(self.protocol, cell, method, events),
                "episode_wall_ns": wall}

    def score(self, records):
        return self.protocol.score(self.manifest, self.digest, records, self.audit)

    def test_missing_keeps_scheduled_denominators_and_unknown_cost(self):
        report = self.score([])
        self.assertEqual(report["scheduled_records"], 120)
        for method in report["by_method"].values():
            self.assertEqual(method["scheduled"], 24)
            self.assertEqual(method["missing"], 24)
            self.assertEqual(method["repair_denominator"], 12)
            self.assertEqual(method["regression_denominator"], 12)
            self.assertEqual(method["selection_pass_rate"], 0)
            self.assertIsNone(method["cost"]["model_calls"]["total"])
        self.assertFalse(report["net_benefit_established"])
        self.assertFalse(report["benchmark_eligible"])

    def test_real_pass_maps_and_keep_baseline(self):
        records = [self.record(cell, "keep", []) for cell in self.manifest["cells"]]
        summary = self.score(records)["by_method"]["keep"]
        self.assertEqual(summary["selection_pass_count"], 12)
        self.assertEqual(summary["complete"], 24)
        self.assertEqual(summary["regression_count"], 0)
        self.assertEqual(summary["repair_count"], 0)
        self.assertEqual(summary["cost"]["model_calls"]["total"], 0)

    def test_repairs_regressions_failures_and_partial_are_separate(self):
        records = []
        for cell in self.manifest["cells"]:
            correct = "ABC"[cell["permutation"].index("correct")]
            wrong = "ABC"[cell["permutation"].index("defect")]
            choice = wrong if cell["initial"] == "correct" else correct
            records.append(self.record(cell, "direct", [model_event(choice)]))
            records.append(self.record(cell, "active_verify", [model_event(correct)]))
        report = self.score(records)
        self.assertEqual(report["by_method"]["direct"]["repair_count"], 12)
        self.assertEqual(report["by_method"]["direct"]["regression_count"], 12)
        self.assertEqual(report["by_method"]["active_verify"]["selection_pass_count"], 24)
        self.assertEqual(report["comparisons"]["direct"]["selection_pass_rate_delta"], 0.5)
        cell = self.manifest["cells"][0]
        records = [
            self.record(cell, "direct", [model_event("bad")]),
            self.record(cell, "rethink", [model_event()], wall=None),
        ]
        report = self.score(records)
        self.assertEqual(report["by_method"]["direct"]["deployment_pass_count"], 1)
        self.assertEqual(report["by_method"]["direct"]["selection_pass_count"], 0)
        self.assertEqual(report["by_method"]["rethink"]["incomplete"], 1)
        self.assertEqual(report["by_method"]["rethink"]["cost"]["model_calls"]["observed"], 1)
        self.assertIsNone(report["by_method"]["rethink"]["cost"]["model_calls"]["total"])

    def test_bad_binding_duplicate_and_wrong_audit_reject(self):
        cell = self.manifest["cells"][0]
        record = self.record(cell, "keep", [])
        with self.assertRaises(ValueError):
            self.score([record, record])
        for change in ({"manifest_sha256": "0" * 64}, {"episode_id": "0" * 32},
                       {"episode_wall_ns": True}, {"extra": 1}):
            with self.assertRaises(ValueError):
                self.score([{**record, **change}])
        changed = copy.deepcopy(self.audit)
        changed["observation"]["records"][0]["value"] = 999
        with self.assertRaises(ValueError):
            self.protocol.score(self.manifest, self.digest, [], changed)
        with self.assertRaises(ValueError):
            self.protocol.score(self.manifest, "0" * 64, [], self.audit)

    def test_receipt_reassigned_to_another_cell_or_method_rejects(self):
        cell = self.manifest["cells"][0]
        record = self.record(cell, "direct", [model_event()])
        for change in ({"method": "active_verify"},
                       {"episode_id": self.manifest["cells"][6]["episode_id"]}):
            with self.assertRaisesRegex(ValueError, "model event binding"):
                self.score([{**record, **change}])


if __name__ == "__main__":
    unittest.main()
