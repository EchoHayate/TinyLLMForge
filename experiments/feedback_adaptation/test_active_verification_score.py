import copy
import unittest

from active_verification_contract import METHODS, probe_request
from active_verification_manifest import manifest_digest
from active_verification_score import bind_transcript, score_batch
from test_active_verification_contract import model_event
from test_active_verification_manifest import sample_manifest


def record_for(manifest, index, method, events):
    return bind_transcript(manifest, manifest["cells"][index]["episode_id"], method, events)


def receipt_for(manifest, index, method, status="unavailable"):
    cell = manifest["cells"][index]
    request = probe_request(cell["view"], manifest["run_id"], cell["episode_id"], method)
    return {
        "kind": "tool", "request_id": request["request_id"], "status": status,
        "results": [], "executed_cases": 0, "wall_ns": 7,
    }


class ScoreTests(unittest.TestCase):
    def test_hand_counted_repairs_harms_failures_and_missing(self):
        manifest = sample_manifest()
        records = [
            record_for(manifest, 0, "direct", [model_event("B")]),
            record_for(manifest, 1, "direct", [model_event("invalid")]),
            record_for(manifest, 3, "direct", [model_event("A")]),
            record_for(manifest, 4, "direct", [model_event("C")]),
            record_for(manifest, 5, "direct", []),
        ]
        report = score_batch(manifest, manifest_digest(manifest), records)
        group = report["by_method"]["direct"]
        for field, expected in {
            "scheduled": 6, "observed": 5, "complete": 4, "valid": 3,
            "missing": 1, "incomplete": 1, "selection_pass_count": 1,
            "deployment_pass_count": 2, "deployment_unknown": 2,
            "repair_count": 1, "repair_denominator": 3,
            "regression_count": 1, "regression_denominator": 3,
            "selection_case_repairs": 2, "selection_case_regressions": 2,
        }.items():
            with self.subTest(field=field):
                self.assertEqual(group[field], expected)
        self.assertEqual(group["repair_rate"], 1 / 3)
        self.assertEqual(group["regression_rate"], 1 / 3)
        self.assertEqual(group["cost"]["model_calls"], {
            "total": None, "observed": 4, "unknown_episodes": 2,
        })
        rows = [row for row in report["rows"] if row["method"] == "direct"]
        self.assertEqual((rows[1]["selected"], rows[1]["deployed"]), (None, "A"))
        self.assertIsNone(rows[2]["deployed"])
        self.assertIsNone(rows[5]["deployed"])
        self.assertEqual(report["scheduled_records"], 30)
        self.assertEqual(report["task_count"], 1)
        self.assertFalse(report["complete"])

    def test_keep_is_zero_cost_only_when_its_record_exists(self):
        manifest = sample_manifest()
        records = [record_for(manifest, index, "keep", []) for index in range(6)]
        group = score_batch(manifest, manifest_digest(manifest), records)["by_method"]["keep"]
        self.assertEqual(group["valid"], 6)
        self.assertEqual(group["selection_pass_count"], 3)
        self.assertEqual(group["kept"], 6)
        self.assertEqual(group["repair_count"], 0)
        self.assertEqual(group["cost"]["model_calls"]["total"], 0)
        missing = score_batch(manifest, manifest_digest(manifest), [])["by_method"]["keep"]
        self.assertEqual(missing["missing"], 6)
        self.assertIsNone(missing["cost"]["model_calls"]["total"])
        self.assertIsNone(group["cost"]["episode_wall_ns"]["total"])

    def test_all_five_methods_and_tool_failures(self):
        manifest = sample_manifest()
        records = []
        for index in range(6):
            trajectories = {
                "keep": [], "direct": [model_event("A")],
                "rethink": [model_event("B"), model_event("A")],
                "always_verify": [receipt_for(manifest, index, "always_verify"), model_event("A")],
                "active_verify": [model_event("VERIFY"), receipt_for(manifest, index, "active_verify"), model_event("A")],
            }
            records.extend(record_for(manifest, index, method, events)
                           for method, events in trajectories.items())
        report = score_batch(manifest, manifest_digest(manifest), records)
        self.assertTrue(report["complete"])
        self.assertEqual(set(report["by_method"]), set(METHODS))
        always = report["by_method"]["always_verify"]
        active = report["by_method"]["active_verify"]
        self.assertEqual(always["cost"]["model_calls"]["total"], 6)
        self.assertEqual(active["cost"]["model_calls"]["total"], 12)
        self.assertEqual(active["cost"]["tool_calls"]["total"], 6)
        self.assertEqual(active["verify_requested"], 6)
        self.assertEqual(active["tool_failed"], 6)
        self.assertEqual(always["repair_count"], 3)

    def test_unknown_late_tokens_preserve_known_earlier_tokens(self):
        manifest = sample_manifest()
        failed = model_event(status="error")
        failed.update(input_ids=None, output_ids=None, wall_ns=None)
        record = record_for(manifest, 0, "rethink", [model_event("B"), failed])
        report = score_batch(manifest, manifest_digest(manifest), [record])
        row = next(row for row in report["rows"] if row["method"] == "rethink")
        self.assertEqual(row["cost"]["input_tokens"], {"total": None, "observed": 1, "unknown_episodes": 1})
        self.assertEqual(row["cost"]["component_wall_ns"]["observed"], 10)
        self.assertEqual(row["cost"]["model_calls"]["total"], 2)
        self.assertEqual((row["selected"], row["deployed"]), (None, "A"))

    def test_unknown_duplicate_and_tampered_records_rejected(self):
        manifest = sample_manifest()
        digest = manifest_digest(manifest)
        record = record_for(manifest, 0, "direct", [model_event("A")])
        with self.assertRaises(ValueError):
            score_batch(manifest, digest, [record, record])
        for field, value in (
            ("episode_id", "f" * 32), ("method", "other"), ("run_id", "3" * 32),
            ("manifest_sha256", "0" * 64), ("selected", "A"), ("evidence_kind", "real"),
        ):
            changed = copy.deepcopy(record)
            changed[field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                score_batch(manifest, digest, [changed])
        changed = copy.deepcopy(record)
        changed["events"].append(model_event("A"))
        with self.assertRaises(ValueError):
            score_batch(manifest, digest, [changed])
        with self.assertRaises(ValueError):
            record_for(manifest, 0, "direct", [receipt_for(manifest, 0, "active_verify")])

    def test_frozen_private_revision_requires_original_digest(self):
        manifest = sample_manifest()
        digest = manifest_digest(manifest)
        record = record_for(manifest, 0, "keep", [])
        changed = copy.deepcopy(manifest)
        changed["tasks"][0]["private"]["audit_note"] = "revised"
        with self.assertRaises(ValueError):
            score_batch(changed, digest, [record])

    def test_group_denominators_and_isolation(self):
        manifest = sample_manifest()
        records = [record_for(manifest, 3, "direct", [model_event("A")])]
        before = copy.deepcopy((manifest, records))
        report = score_batch(manifest, manifest_digest(manifest), records)
        group = report["strata"]["initial"]["correct"]["direct"]
        self.assertEqual(group["scheduled"], 3)
        self.assertIsNone(group["repair_rate"])
        self.assertEqual(group["regression_denominator"], 3)
        self.assertEqual(report["strata"]["condition"]["none"]["direct"]["scheduled"], 2)
        self.assertEqual(report["strata"]["family_id"]["family-1"]["direct"]["task_count"], 1)
        self.assertEqual(report["evidence_kind"], "SYNTHETIC_ONLY")
        self.assertFalse(report["benchmark_eligible"])
        self.assertFalse(report["real_execution_eligible"])
        self.assertFalse(report["formal_evaluation_ready"])
        report["rows"][0]["events"].append({"changed": True})
        self.assertEqual((manifest, records), before)

    def test_cross_cell_tool_receipt_is_rejected(self):
        manifest = sample_manifest()
        with self.assertRaises(ValueError):
            record_for(manifest, 1, "active_verify", [
                model_event("VERIFY"), receipt_for(manifest, 0, "active_verify"),
            ])

    def test_comparisons_keep_pair_counts_and_unknown_costs(self):
        manifest = sample_manifest()
        records = []
        for index in range(6):
            records.append(record_for(manifest, index, "always_verify", [
                receipt_for(manifest, index, "always_verify"), model_event("A"),
            ]))
            records.append(record_for(manifest, index, "active_verify", [
                model_event("VERIFY"), receipt_for(manifest, index, "active_verify"),
                model_event("A"),
            ]))
        report = score_batch(manifest, manifest_digest(manifest), records)
        comparison = report["comparisons"]["always_verify"]
        self.assertEqual(comparison["scheduled_pairs"], 6)
        self.assertEqual(comparison["complete_pairs"], 6)
        self.assertEqual(comparison["selection_pass_rate_delta"], 0)
        self.assertEqual(comparison["cost_delta"]["model_calls"], 6)
        self.assertIsNone(comparison["cost_delta"]["episode_wall_ns"])
        self.assertFalse(comparison["net_benefit_established"])
        self.assertEqual(len(comparison["per_task"]), 1)
        incomplete = report["comparisons"]["rethink"]
        self.assertEqual(incomplete["complete_pairs"], 0)
        self.assertIsNone(incomplete["cost_delta"]["model_calls"])
        self.assertIsNone(incomplete["deployment_pass_rate_delta"])
        self.assertEqual(report["by_method"]["always_verify"]["protocol_success_rate"], 1)

    def test_observed_budget_excess_is_reported_not_erased(self):
        manifest = sample_manifest()
        event = model_event("A")
        event["input_ids"] *= 300
        record = record_for(manifest, 0, "direct", [event])
        report = score_batch(manifest, manifest_digest(manifest), [record])
        self.assertIn("input_tokens", report["budget_overruns"])
        self.assertEqual(report["by_method"]["direct"]["status_counts"]["budget_exceeded"], 1)
        self.assertEqual(report["overall"]["cost"]["input_tokens"]["observed"], 300)
        self.assertFalse(report["formal_evaluation_ready"])

    def test_all_permutations_and_terminal_actions_against_role_oracle(self):
        manifest = sample_manifest(True)
        digest = manifest_digest(manifest)
        outcomes = {
            "correct": [True, True], "defect": [True, False], "other": [False, True],
        }
        for action in ("correct", "defect", "other", "invalid", "empty", "truncated", "error", "timeout", "VERIFY"):
            records, expected = [], {}
            for index, cell in enumerate(manifest["cells"]):
                letters = dict(zip(cell["permutation"], "ABC"))
                if action in outcomes:
                    event = model_event(letters[action])
                elif action == "empty":
                    event = model_event("")
                elif action == "truncated":
                    event = model_event("A", terminated=False)
                elif action in ("error", "timeout"):
                    event = model_event(status=action)
                else:
                    event = model_event(action)
                for method in METHODS:
                    if method == "keep":
                        events = []
                    elif method == "direct":
                        events = [event]
                    elif method == "rethink":
                        events = [model_event(letters["other"]), event]
                    elif method == "always_verify":
                        events = [receipt_for(manifest, index, method), event]
                    else:
                        events = [model_event("VERIFY"), receipt_for(manifest, index, method), event]
                    records.append({
                        "evidence_kind": "synthetic", "manifest_sha256": digest,
                        "run_id": manifest["run_id"], "episode_id": cell["episode_id"],
                        "method": method, "events": events,
                    })
                    selected_role = cell["initial"] if method == "keep" else (
                        action if action in outcomes else None
                    )
                    deployed_role = selected_role or cell["initial"]
                    expected[(cell["episode_id"], method)] = {
                        "selected": None if selected_role is None else letters[selected_role],
                        "deployed": letters[deployed_role],
                        "selection_pass_count": int(selected_role == "correct"),
                        "deployment_pass_count": int(deployed_role == "correct"),
                        "repair_count": int(cell["initial"] == "defect" and selected_role == "correct"),
                        "regression_count": int(cell["initial"] == "correct" and selected_role in ("defect", "other")),
                    }
            report = score_batch(manifest, digest, records)
            self.assertEqual(report, score_batch(manifest, digest, list(reversed(records))))
            self.assertEqual((report["task_count"], report["scheduled_records"]), (1, 180))
            for row in report["rows"]:
                for field, value in expected[(row["episode_id"], row["method"])].items():
                    with self.subTest(action=action, episode=row["episode_id"], method=row["method"], field=field):
                        self.assertEqual(row[field], value)
            for field in ("repair_count", "regression_count", "selection_pass_count", "deployment_pass_count"):
                self.assertEqual(report["overall"][field], sum(item[field] for item in expected.values()))


if __name__ == "__main__":
    unittest.main()
