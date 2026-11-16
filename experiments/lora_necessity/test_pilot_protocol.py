import copy
import hashlib
import importlib
import json
from pathlib import Path
import unittest


def task_fixture():
    tasks = []
    for split, count in (("supervision", 8), ("development", 4),
                         ("transfer", 16), ("retention", 16)):
        for index in range(count):
            task_id = f"{split}-{index:02d}"
            tasks.append({
                "id": task_id, "split": split,
                "family": f"{'retain' if split == 'retention' else 'adapt'}-{index % 4}",
                "template_id": f"template-{task_id}",
                "content_sha256": hashlib.sha256(task_id.encode()).hexdigest(),
            })
    return tasks


def record_fixture(manifest):
    records = []
    for task in manifest["tasks"]:
        if task["split"] == "supervision":
            continue
        arms = ("base", "text") if task["split"] == "development" else (
            "base", "text", "lora", "lora_text")
        for arm in arms:
            ordinal = int(task["id"].rsplit("-", 1)[1])
            passed = ordinal % 2 == 0
            if task["split"] == "transfer" and arm == "lora_text" and ordinal in (1, 3):
                passed = True
            records.append({
                "kind": "synthetic", "task_id": task["id"], "arm": arm,
                "status": "pass" if passed else "fail",
                "model_calls": 2, "input_tokens": 20, "output_tokens": 10,
                "episode_ns": 100,
            })
    return records


class PilotProtocolTests(unittest.TestCase):
    def setUp(self):
        self.assertTrue(Path(__file__).with_name("pilot_protocol.py").is_file(),
                        "pilot protocol is not implemented")
        self.protocol = importlib.import_module("pilot_protocol")
        self.manifest = self.protocol.build_manifest(task_fixture())
        self.records = record_fixture(self.manifest)
        self.checks = {
            "protocol_valid": True, "parameters_valid": True,
            "isolation_valid": True, "cleanup_complete": True,
        }

    def test_configuration_copies_and_binding(self):
        original = copy.deepcopy(self.manifest)
        config = self.protocol.configuration()
        self.assertEqual(config["training"]["steps"], 16)
        config["training"]["steps"] = 1
        self.assertEqual(self.protocol.configuration()["training"]["steps"], 16)
        self.assertEqual(self.manifest, original)
        changed = copy.deepcopy(self.manifest)
        changed["config"]["training"]["steps"] = True
        with self.assertRaises(ValueError):
            self.protocol.validate_manifest(changed)

    def test_synthetic_success_cannot_admit_real_experiment(self):
        original = copy.deepcopy(self.records)
        report = self.protocol.score_synthetic(self.manifest, self.records, self.checks)
        self.assertEqual(report["status"], "SYNTHETIC_ONLY")
        self.assertEqual(report["simulated_decision"], "PROMISING_FOR_REPLICATION")
        self.assertEqual(report["pairs"]["transfer"]["net"], 2)
        self.assertEqual(report["pairs"]["retention"]["right_to_wrong"], 0)
        self.assertEqual(report["model_calls"], 272)
        for flag in ("benchmark_eligible", "formal_evaluation_ready",
                     "net_benefit_established", "isolated"):
            self.assertIs(report[flag], False)
        self.assertEqual(self.records, original)

    def test_missing_and_unknown_cost_do_not_pass(self):
        report = self.protocol.score_synthetic(self.manifest, [], self.checks)
        self.assertEqual(report["simulated_decision"], "INCONCLUSIVE_INCOMPLETE")
        self.assertEqual(report["by_split"]["transfer"]["text"]["scheduled"], 16)
        self.assertEqual(report["by_split"]["transfer"]["text"]["passed"], 0)
        self.assertIsNone(report["by_split"]["transfer"]["text"]["episode_ns"])
        changed = copy.deepcopy(self.records)
        changed[-1]["episode_ns"] = None
        report = self.protocol.score_synthetic(self.manifest, changed, self.checks)
        self.assertEqual(report["simulated_decision"], "INCONCLUSIVE_INCOMPLETE")

    def test_retention_regression_blocks_positive_transfer(self):
        changed = copy.deepcopy(self.records)
        for record in changed:
            if record["task_id"] == "retention-00" and record["arm"] == "lora_text":
                record["status"] = "fail"
        report = self.protocol.score_synthetic(self.manifest, changed, self.checks)
        self.assertEqual(report["simulated_decision"], "OBSERVED_REGRESSION")

    def test_duplicate_and_live_records_rejected(self):
        with self.assertRaises(ValueError):
            self.protocol.score_synthetic(self.manifest, self.records + self.records[:1], self.checks)
        changed = copy.deepcopy(self.records)
        changed[0]["kind"] = "live"
        with self.assertRaises(ValueError):
            self.protocol.score_synthetic(self.manifest, changed, self.checks)

    def test_invalid_metadata_and_scalars_rejected(self):
        changed = task_fixture()
        changed[-1]["template_id"] = changed[0]["template_id"]
        with self.assertRaises(ValueError):
            self.protocol.build_manifest(changed)
        for invalid in (True, -1, float("nan"), float("inf"), "10"):
            changed_records = copy.deepcopy(self.records)
            changed_records[0]["episode_ns"] = invalid
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                self.protocol.score_synthetic(self.manifest, changed_records, self.checks)

    def test_split_counts_and_family_contract(self):
        for split in ("supervision", "development", "transfer", "retention"):
            for field, replacement in (("split", "unknown"), ("split", "development"
                                       if split != "development" else "transfer"),
                                       ("family", "extra-family")):
                tasks = task_fixture()
                selected = next(task for task in tasks if task["split"] == split)
                selected[field] = replacement
                with self.subTest(split=split, field=field, replacement=replacement):
                    if split == "development" and field == "family":
                        self.assertEqual(self.protocol.build_manifest(tasks)["tasks"], tasks)
                    else:
                        with self.assertRaises(ValueError):
                            self.protocol.build_manifest(tasks)
            tasks = task_fixture()
            selected = [task for task in tasks if task["split"] == split]
            selected[0]["family"] = selected[1]["family"]
            with self.subTest(split=split, problem="unbalanced"):
                with self.assertRaises(ValueError):
                    self.protocol.build_manifest(tasks)
        for split, family in (("transfer", "other-0"), ("retention", "adapt-0")):
            tasks = task_fixture()
            for task in tasks:
                if task["split"] == split and task["family"].endswith("-0"):
                    task["family"] = family
            with self.subTest(split=split, problem="family-relationship"):
                with self.assertRaises(ValueError):
                    self.protocol.build_manifest(tasks)
        for tasks in (task_fixture()[:-1], task_fixture() + task_fixture()[:1]):
            with self.subTest(count=len(tasks)), self.assertRaises(ValueError):
                self.protocol.build_manifest(tasks)

    def test_duplicate_ids_templates_and_digests(self):
        for field in ("id", "template_id", "content_sha256"):
            tasks = task_fixture()
            tasks[-1][field] = tasks[0][field]
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.protocol.build_manifest(tasks)

    def test_fixed_configuration_and_strict_types(self):
        expected = {
            "arms": ["base", "text", "lora", "lora_text"],
            "splits": {"supervision": 8, "development": 4, "transfer": 16, "retention": 16},
            "training": {
                "rank": 8, "alpha": 16, "dropout": 0, "seed": 42,
                "targets": ["q_proj", "v_proj"], "base_dtype": "bfloat16",
                "adapter_dtype": "float32", "optimizer": "AdamW",
                "learning_rate": 0.00005, "weight_decay": 0,
                "gradient_clip": 1, "batch_size": 1, "accumulation": 1,
                "epochs": 2, "steps": 16, "answer_only_loss": True,
            },
            "limits": {
                "input_tokens": 4096, "output_tokens": 256, "episode_calls": 2,
                "total_calls": 272, "workflow_seconds": 3600, "gpu_count": 1,
            },
            "screen": {"transfer_net_min": 2, "retention_regression_max": 0},
        }
        self.assertEqual(self.protocol.configuration(), expected)
        for section in ("training", "limits", "screen", "splits"):
            for name, value in expected[section].items():
                changed = copy.deepcopy(self.manifest)
                changed["config"][section][name] = None
                with self.subTest(section=section, name=name), self.assertRaises(ValueError):
                    self.protocol.validate_manifest(changed)
                if type(value) is int:
                    for replacement in (True, float(value), value + 1):
                        changed["config"][section][name] = replacement
                        with self.subTest(section=section, name=name, value=replacement):
                            with self.assertRaises(ValueError):
                                self.protocol.validate_manifest(changed)
        for field in ("arms", "training"):
            changed = copy.deepcopy(self.manifest)
            if field == "arms":
                changed["config"][field].reverse()
            else:
                changed["config"][field]["targets"].reverse()
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.protocol.validate_manifest(changed)

    def test_malformed_configuration_returns_value_error(self):
        for invalid in ({1: "mixed-key"}, {"arms": [], 1: "mixed-key"}):
            changed = copy.deepcopy(self.manifest)
            changed["config"].update(invalid)
            with self.subTest(invalid=invalid):
                try:
                    self.protocol.validate_manifest(changed)
                except ValueError:
                    pass
                except Exception as error:
                    self.fail(f"schema must raise ValueError, not {type(error).__name__}")
                else:
                    self.fail("malformed configuration was accepted")
        changed = copy.deepcopy(self.manifest)
        changed["config"]["arms"] = tuple(changed["config"]["arms"])
        with self.assertRaises(ValueError):
            self.protocol.validate_manifest(changed)

    def test_unknown_and_unscheduled_episodes(self):
        for task_id, arm in (("missing", "text"), ("supervision-00", "base"),
                             ("development-00", "lora"), ("development-00", "lora_text"),
                             ("transfer-00", "unknown")):
            changed = copy.deepcopy(self.records)
            changed[0].update(task_id=task_id, arm=arm)
            with self.subTest(task_id=task_id, arm=arm), self.assertRaises(ValueError):
                self.protocol.score_synthetic(self.manifest, changed, self.checks)

    def test_call_count_types_and_completed_episode_contract(self):
        for invalid in (True, False, -1, 3, 2.0, "2", None, [], {}, float("nan"), float("inf")):
            changed = copy.deepcopy(self.records)
            changed[0]["model_calls"] = invalid
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                self.protocol.score_synthetic(self.manifest, changed, self.checks)
        for status in ("pass", "fail", "invalid"):
            for calls in (0, 1):
                changed = copy.deepcopy(self.records)
                changed[0].update(status=status, model_calls=calls, input_tokens=0, output_tokens=0)
                with self.subTest(status=status, calls=calls), self.assertRaises(ValueError):
                    self.protocol.score_synthetic(self.manifest, changed, self.checks)

    def test_cost_types_unknowns_and_zero(self):
        for name in ("input_tokens", "output_tokens", "episode_ns"):
            for invalid in (True, False, -1, 1.0, "1", [], {}, float("nan"), float("inf")):
                changed = copy.deepcopy(self.records)
                changed[0][name] = invalid
                with self.subTest(name=name, invalid=invalid), self.assertRaises(ValueError):
                    self.protocol.score_synthetic(self.manifest, changed, self.checks)
            changed = copy.deepcopy(self.records)
            changed[0][name] = None
            report = self.protocol.score_synthetic(self.manifest, changed, self.checks)
            self.assertEqual(report["simulated_decision"], "INCONCLUSIVE_INCOMPLETE")
            self.assertIsNone(report["by_split"]["development"]["base"][name])
            changed[0][name] = 0
            report = self.protocol.score_synthetic(self.manifest, changed, self.checks)
            self.assertEqual(report["simulated_decision"], "PROMISING_FOR_REPLICATION")

    def test_token_caps_follow_actual_call_count(self):
        for name, cap in (("input_tokens", 4096), ("output_tokens", 256)):
            for calls in (0, 1, 2):
                changed = copy.deepcopy(self.records)
                changed[0].update(status="timeout", model_calls=calls,
                                  input_tokens=0, output_tokens=0)
                changed[0][name] = calls * cap
                report = self.protocol.score_synthetic(self.manifest, changed, self.checks)
                self.assertEqual(report["model_calls"], 270 + calls)
                self.assertEqual(report["simulated_decision"], "INCONCLUSIVE_INCOMPLETE")
                changed[0][name] += 1
                with self.subTest(name=name, calls=calls), self.assertRaises(ValueError):
                    self.protocol.score_synthetic(self.manifest, changed, self.checks)
            changed = copy.deepcopy(self.records)
            changed[0][name] = 2 * cap
            report = self.protocol.score_synthetic(self.manifest, changed, self.checks)
            self.assertEqual(report["simulated_decision"], "PROMISING_FOR_REPLICATION")
            changed[0][name] += 1
            with self.subTest(name=name, status="pass"), self.assertRaises(ValueError):
                self.protocol.score_synthetic(self.manifest, changed, self.checks)

    def test_timeout_and_ordinary_invalid_code_are_distinct(self):
        changed = copy.deepcopy(self.records)
        selected = next(record for record in changed
                        if record["task_id"] == "transfer-01" and record["arm"] == "text")
        selected["status"] = "invalid"
        report = self.protocol.score_synthetic(self.manifest, changed, self.checks)
        self.assertEqual(report["simulated_decision"], "PROMISING_FOR_REPLICATION")
        self.assertEqual(report["by_split"]["transfer"]["text"]["statuses"]["invalid"], 1)
        self.assertEqual(report["pairs"]["transfer"]["wrong_to_right"], 2)
        selected["status"] = "timeout"
        report = self.protocol.score_synthetic(self.manifest, changed, self.checks)
        self.assertEqual(report["simulated_decision"], "INCONCLUSIVE_INCOMPLETE")
        self.assertEqual(report["by_split"]["transfer"]["text"]["statuses"]["timeout"], 1)

    def test_no_gain_and_screen_threshold(self):
        for gains, decision in ((0, "NO_GAIN_IN_THIS_PILOT"),
                                (1, "NO_GAIN_IN_THIS_PILOT"),
                                (2, "PROMISING_FOR_REPLICATION")):
            changed = copy.deepcopy(self.records)
            for record in changed:
                if record["arm"] == "lora_text" and record["task_id"] in (
                        "transfer-01", "transfer-03"):
                    record["status"] = "pass" if record["task_id"] in (
                        "transfer-01", "transfer-03")[:gains] else "fail"
            report = self.protocol.score_synthetic(self.manifest, changed, self.checks)
            with self.subTest(gains=gains):
                self.assertEqual(report["pairs"]["transfer"]["net"], gains)
                self.assertEqual(report["simulated_decision"], decision)
        changed = copy.deepcopy(self.records)
        for record in changed:
            if record["task_id"] == "transfer-00" and record["arm"] == "lora_text":
                record["status"] = "fail"
        report = self.protocol.score_synthetic(self.manifest, changed, self.checks)
        self.assertEqual(report["pairs"]["transfer"]["net"], 1)
        self.assertEqual(report["simulated_decision"], "NO_GAIN_IN_THIS_PILOT")

    def test_calibration_floor_ceiling_and_invalid(self):
        for status in ("pass", "fail", "invalid"):
            changed = copy.deepcopy(self.records)
            for record in changed:
                if record["task_id"].startswith("development-"):
                    record["status"] = status
            report = self.protocol.score_synthetic(self.manifest, changed, self.checks)
            with self.subTest(status=status):
                self.assertEqual(report["simulated_decision"], "INCONCLUSIVE_CALIBRATION")
        changed = copy.deepcopy(self.records)
        changed[0]["status"] = "invalid"
        report = self.protocol.score_synthetic(self.manifest, changed, self.checks)
        self.assertEqual(report["simulated_decision"], "INCONCLUSIVE_CALIBRATION")
        changed[0]["status"] = "timeout"
        report = self.protocol.score_synthetic(self.manifest, changed, self.checks)
        self.assertEqual(report["simulated_decision"], "INCONCLUSIVE_INCOMPLETE")

    def test_failed_unknown_and_malformed_checks(self):
        for name in self.checks:
            for value, decision in ((False, "INVALID_PROTOCOL"),
                                    (None, "INCONCLUSIVE_INCOMPLETE")):
                checks = dict(self.checks, **{name: value})
                report = self.protocol.score_synthetic(self.manifest, self.records, checks)
                with self.subTest(name=name, value=value):
                    self.assertEqual(report["simulated_decision"], decision)
                    for flag in ("benchmark_eligible", "formal_evaluation_ready",
                                 "net_benefit_established", "isolated"):
                        self.assertIs(report[flag], False)
            for invalid in (0, 1, "true", [], {}, float("nan")):
                checks = dict(self.checks, **{name: invalid})
                with self.subTest(name=name, invalid=invalid), self.assertRaises(ValueError):
                    self.protocol.score_synthetic(self.manifest, self.records, checks)
        checks = dict(self.checks, protocol_valid=False, cleanup_complete=None)
        report = self.protocol.score_synthetic(self.manifest, [], checks)
        self.assertEqual(report["simulated_decision"], "INVALID_PROTOCOL")

    def test_record_order_independence_and_pairs(self):
        original = self.protocol.score_synthetic(self.manifest, self.records, self.checks)
        for records in (list(reversed(self.records)), self.records[1:] + self.records[:1]):
            self.assertEqual(self.protocol.score_synthetic(self.manifest, records, self.checks),
                             original)
        self.assertEqual(original["pairs"]["transfer"], {
            "both_right": 8, "both_wrong": 6, "right_to_wrong": 0,
            "wrong_to_right": 2, "missing": 0, "net": 2,
        })
        self.assertEqual(original["scheduled_model_calls"], 272)

    def test_manifest_hash_and_input_output_independence(self):
        tasks = task_fixture()
        original_tasks = copy.deepcopy(tasks)
        manifest = self.protocol.build_manifest(tasks)
        original_manifest = copy.deepcopy(manifest)
        expected = hashlib.sha256(json.dumps(
            manifest, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()).hexdigest()
        self.assertEqual(self.protocol.validate_manifest(manifest), expected)
        reordered = {name: manifest[name] for name in reversed(manifest)}
        self.assertEqual(self.protocol.validate_manifest(reordered), expected)
        report = self.protocol.score_synthetic(manifest, self.records, self.checks)
        self.assertEqual(report["manifest_sha256"], expected)
        self.assertEqual(manifest, original_manifest)
        self.assertEqual(tasks, original_tasks)
        tasks[0]["id"] = "changed"
        self.assertEqual(manifest, original_manifest)
        manifest["tasks"][1]["id"] = "changed-again"
        self.assertEqual(tasks[1], original_tasks[1])
        report["checks"]["protocol_valid"] = False
        self.assertIs(self.checks["protocol_valid"], True)
        config = self.protocol.configuration()
        config["arms"].clear()
        config["training"]["targets"].clear()
        config["limits"]["total_calls"] = 0
        self.assertEqual(self.protocol.configuration(), original_manifest["config"])

    def test_missing_denominators_and_observed_costs(self):
        records = [record for record in self.records
                   if (record["task_id"], record["arm"]) not in {
                       ("transfer-00", "text"), ("transfer-01", "lora_text")}]
        report = self.protocol.score_synthetic(self.manifest, records, self.checks)
        self.assertEqual(report["simulated_decision"], "INCONCLUSIVE_INCOMPLETE")
        for arm in ("base", "text", "lora", "lora_text"):
            summary = report["by_split"]["transfer"][arm]
            self.assertEqual(summary["scheduled"], 16)
            self.assertEqual(sum(summary["statuses"].values()), 16)
        for arm in ("text", "lora_text"):
            summary = report["by_split"]["transfer"][arm]
            self.assertEqual(summary["received"], 15)
            self.assertEqual(summary["statuses"]["missing"], 1)
            for name, cost in (("input_tokens", 20), ("output_tokens", 10), ("episode_ns", 100)):
                self.assertIsNone(summary[name])
                self.assertEqual(summary[f"{name}_observed"], 15 * cost)
        pairs = report["pairs"]["transfer"]
        self.assertEqual(pairs["missing"], 2)
        self.assertEqual(sum(value for name, value in pairs.items() if name != "net"), 16)

    def test_family_summaries_keep_fixed_denominators_and_missing_rows(self):
        for records in (self.records, [], self.records[1:]):
            report = self.protocol.score_synthetic(self.manifest, records, self.checks)
            self.assertTrue("by_family" in report, "per-family summaries are required")
            self.assertEqual(set(report["by_family"]), {"development", "transfer", "retention"})
            for split, arms in report["by_split"].items():
                prefix = "retain" if split == "retention" else "adapt"
                split_families = report["by_family"][split]
                self.assertEqual(set(split_families),
                                 {f"{prefix}-{index}" for index in range(4)})
                for family_arms in split_families.values():
                    self.assertEqual(set(family_arms), set(arms))
                for arm, total in arms.items():
                    families = {family: family_arms[arm]
                                for family, family_arms in split_families.items()}
                    for summary in families.values():
                        self.assertEqual(set(summary), set(total))
                        self.assertEqual(summary["scheduled"], 1 if split == "development" else 4)
                        self.assertEqual(sum(summary["statuses"].values()), summary["scheduled"])
                    for name in ("scheduled", "received", "passed", "input_tokens_observed",
                                 "output_tokens_observed", "episode_ns_observed"):
                        self.assertEqual(sum(summary[name] for summary in families.values()),
                                         total[name])
                    for status in ("pass", "fail", "invalid", "timeout", "missing"):
                        self.assertEqual(sum(summary["statuses"].get(status, 0)
                                             for summary in families.values()),
                                         total["statuses"].get(status, 0))
                    for name in ("input_tokens", "output_tokens", "episode_ns"):
                        values = [summary[name] for summary in families.values()]
                        if None in values:
                            self.assertIsNone(total[name])
                        else:
                            self.assertEqual(sum(values), total[name])
                    if not records:
                        for summary in families.values():
                            self.assertEqual(summary["received"], 0)
                            self.assertEqual(summary["passed"], 0)
                            self.assertEqual(summary["statuses"], {"missing": summary["scheduled"]})
                            for name in ("input_tokens", "output_tokens", "episode_ns"):
                                self.assertIsNone(summary[name])
                                self.assertEqual(summary[f"{name}_observed"], 0)

    def test_family_summaries_bind_manifest_family_and_unknown_cost(self):
        records = copy.deepcopy(self.records)
        records = [record for record in records
                   if (record["task_id"], record["arm"]) != ("transfer-00", "text")]
        selected = next(record for record in records
                        if (record["task_id"], record["arm"]) == ("transfer-01", "lora"))
        selected["episode_ns"] = None
        report = self.protocol.score_synthetic(self.manifest, records, self.checks)
        self.assertTrue("by_family" in report, "per-family summaries are required")
        families = report["by_family"]["transfer"]
        summary = families["adapt-0"]["text"]
        self.assertEqual(summary["scheduled"], 4)
        self.assertEqual(summary["received"], 3)
        self.assertEqual(summary["passed"], 3)
        self.assertEqual(summary["statuses"], {"pass": 3, "missing": 1})
        self.assertIsNone(summary["episode_ns"])
        self.assertEqual(summary["episode_ns_observed"], 300)
        self.assertEqual(families["adapt-1"]["text"]["passed"], 0)
        self.assertEqual(families["adapt-1"]["lora_text"]["passed"], 1)
        self.assertIsNone(families["adapt-1"]["lora"]["episode_ns"])
        self.assertEqual(families["adapt-1"]["lora"]["episode_ns_observed"], 300)
        self.assertEqual(families["adapt-0"]["lora"]["episode_ns"], 400)

    def test_scalar_list_and_dict_substitutions_at_public_boundaries(self):
        invalid_values = (None, True, 1, 1.5, "invalid", [], {})
        for invalid in invalid_values:
            with self.subTest(boundary="manifest", invalid=invalid), self.assertRaises(ValueError):
                self.protocol.validate_manifest(invalid)
            with self.subTest(boundary="score-manifest", invalid=invalid):
                with self.assertRaises(ValueError):
                    self.protocol.score_synthetic(invalid, self.records, self.checks)
            with self.subTest(boundary="tasks", invalid=invalid), self.assertRaises(ValueError):
                self.protocol.build_manifest(invalid)
            with self.subTest(boundary="checks", invalid=invalid), self.assertRaises(ValueError):
                self.protocol.score_synthetic(self.manifest, self.records, invalid)
            if type(invalid) is not list:
                with self.subTest(boundary="records", invalid=invalid), self.assertRaises(ValueError):
                    self.protocol.score_synthetic(self.manifest, invalid, self.checks)
            changed = copy.deepcopy(self.records)
            changed[0] = invalid
            with self.subTest(boundary="record", invalid=invalid), self.assertRaises(ValueError):
                self.protocol.score_synthetic(self.manifest, changed, self.checks)
            tasks = task_fixture()
            tasks[0] = invalid
            with self.subTest(boundary="task", invalid=invalid), self.assertRaises(ValueError):
                self.protocol.build_manifest(tasks)
            for field in self.manifest:
                manifest = copy.deepcopy(self.manifest)
                manifest[field] = invalid
                if field == "version" and type(invalid) is int and invalid == 1:
                    continue
                with self.subTest(boundary="manifest-field", field=field, invalid=invalid):
                    with self.assertRaises(ValueError):
                        self.protocol.validate_manifest(manifest)
            for field in task_fixture()[0]:
                tasks = task_fixture()
                tasks[0][field] = invalid
                if field in ("id", "family", "template_id") and invalid == "invalid":
                    continue
                with self.subTest(boundary="task-field", field=field, invalid=invalid):
                    with self.assertRaises(ValueError):
                        self.protocol.build_manifest(tasks)
            for field in self.records[0]:
                changed = copy.deepcopy(self.records)
                changed[0][field] = invalid
                if field in ("input_tokens", "output_tokens", "episode_ns") and (
                        invalid is None or type(invalid) is int):
                    continue
                if field == "status" and invalid == "invalid":
                    continue
                with self.subTest(boundary="record-field", field=field, invalid=invalid):
                    with self.assertRaises(ValueError):
                        self.protocol.score_synthetic(self.manifest, changed, self.checks)
            for section in self.manifest["config"]:
                manifest = copy.deepcopy(self.manifest)
                manifest["config"][section] = invalid
                with self.subTest(boundary="config-section", section=section, invalid=invalid):
                    with self.assertRaises(ValueError):
                        self.protocol.validate_manifest(manifest)

    def test_missing_and_extra_schema_fields(self):
        for boundary, original in (("manifest", self.manifest), ("task", self.manifest["tasks"][0]),
                                   ("record", self.records[0]), ("checks", self.checks),
                                   ("config", self.manifest["config"])):
            for field in (*original, "unexpected"):
                changed = copy.deepcopy(original)
                if field == "unexpected":
                    changed[field] = True
                else:
                    del changed[field]
                with self.subTest(boundary=boundary, field=field), self.assertRaises(ValueError):
                    if boundary == "manifest":
                        self.protocol.validate_manifest(changed)
                    elif boundary == "task":
                        self.protocol.build_manifest([changed] + self.manifest["tasks"][1:])
                    elif boundary == "record":
                        self.protocol.score_synthetic(self.manifest, [changed], self.checks)
                    elif boundary == "checks":
                        self.protocol.score_synthetic(self.manifest, self.records, changed)
                    else:
                        manifest = copy.deepcopy(self.manifest)
                        manifest["config"] = changed
                        self.protocol.validate_manifest(manifest)

    def test_metadata_lexical_validation(self):
        for field in ("id", "family", "template_id"):
            for invalid in ("", "spaces here", "slash/name", "newline\n", "非ASCII"):
                tasks = task_fixture()
                tasks[0][field] = invalid
                with self.subTest(field=field, invalid=invalid), self.assertRaises(ValueError):
                    self.protocol.build_manifest(tasks)
        for invalid in ("a" * 63, "a" * 65, "A" * 64, "g" * 64):
            tasks = task_fixture()
            tasks[0]["content_sha256"] = invalid
            with self.subTest(digest=invalid), self.assertRaises(ValueError):
                self.protocol.build_manifest(tasks)
