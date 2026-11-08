import copy
import hashlib
import json
import pathlib
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import development_cpu as cpu
from development_tasks import catalog


class DevelopmentCPUTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.storage = tempfile.TemporaryDirectory()
        cls.root = pathlib.Path(cls.storage.name)
        cls.result = cpu.run(cls.root / "audit", "audit")
        cls.manifest = json.loads((cls.root / "audit/manifest.json").read_text())
        cls.observation = json.loads((cls.root / "audit/stdout.json").read_text())

    @classmethod
    def tearDownClass(cls):
        cls.storage.cleanup()

    def test_manifest_has_fixed_budget_and_all_source_bindings(self):
        manifest = cpu.freeze("audit")
        self.assertEqual(len(manifest["schedule"]), 1209)
        self.assertEqual(manifest["budget"], {
            "calls": 1209, "cpu_seconds": 2, "wall_seconds": 5,
            "output_bytes_per_stream": 2 * 1024 * 1024,
        })
        self.assertEqual(set(manifest["source_sha256"]), {
            "development_programs.py", "development_tasks.py", "development_cpu.py",
        })
        for mode, task, initial in (
            ("sealed", None, None), ("audit", "dev-001", "correct"),
            ("probe", "unknown", "correct"), ("probe", "dev-001", "other"),
            ("probe", None, None),
        ):
            with self.assertRaises(ValueError):
                cpu.freeze(mode, task, initial)

    def test_real_audit_is_complete_without_benchmark_or_model_claims(self):
        result = self.result
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["origin"], "controlled_local_invocation")
        self.assertFalse(result["isolated"])
        self.assertFalse(result["benchmark_eligible"])
        self.assertIsNone(result["peak_rss_bytes"])
        self.assertIsNone(result["cpu_time_ns"])
        self.assertGreater(result["controller_wall_ns"], 0)
        comparison = result["comparison"]
        self.assertEqual(comparison["kind"], "FIXED_DEVELOPMENT_COMPARISON")
        self.assertTrue(comparison["admitted"])
        self.assertEqual(comparison["scheduled_calls"], 1209)
        self.assertEqual(comparison["observed_calls"], 1209)
        self.assertEqual(len(comparison["tasks"]), 4)
        for task in comparison["tasks"]:
            self.assertTrue(task["admitted"])
            self.assertEqual(task["pass_counts"]["correct"], task["domain_cases"])
            self.assertLess(task["pass_counts"]["defect"], task["domain_cases"])
            self.assertLess(task["pass_counts"]["other"], task["domain_cases"])
            self.assertTrue(task["feedback_supported"])
            self.assertTrue(task["information_verified"])

    def test_frozen_bytes_raw_transport_and_replay_are_retained(self):
        root = self.root / "audit"
        for name, digest in self.manifest["source_sha256"].items():
            self.assertEqual(hashlib.sha256((root / name).read_bytes()).hexdigest(), digest)
        transport = json.loads((root / "transport.json").read_text())
        self.assertEqual(transport["returncode"], 0)
        self.assertEqual(transport["stdout_sha256"],
                         hashlib.sha256((root / "stdout.json").read_bytes()).hexdigest())
        self.assertEqual(transport["stderr_bytes"], 0)
        self.assertEqual(cpu.compare(self.manifest, self.observation), self.result["comparison"])
        self.assertEqual(json.loads((root / "result.json").read_text()), self.result)

    def test_probe_exposes_only_initial_program_and_two_cases(self):
        for task in catalog():
            for initial in ("correct", "defect"):
                target = self.root / f"{task['task_id']}-{initial}"
                result = cpu.run(target, "probe", task["task_id"], initial)
                self.assertEqual(result["status"], "completed")
                public = result["comparison"]["tool_view"]
                self.assertEqual(set(public), {"initial_source_sha256", "results"})
                self.assertEqual(len(public["results"]), 2)
                for row in public["results"]:
                    self.assertEqual(set(row), {"id", "input", "expected", "actual", "status"})
                    self.assertEqual(row["status"], "ok")
                self.assertNotIn("private", json.dumps(public))
                self.assertNotIn("feedback", json.dumps(public))
                if task["private"]["verification_information"] == "insufficient":
                    self.assertTrue(all(row["actual"] == row["expected"]
                                        for row in public["results"]))
                self.assertEqual(result["comparison"]["observed_calls"], 2)

    def test_observation_mutations_and_missing_denominators_reject(self):
        mutations = (
            lambda value: value["records"].pop(),
            lambda value: value["records"].append(copy.deepcopy(value["records"][0])),
            lambda value: value["records"][0].update(value=True),
            lambda value: value["records"][0].update(elapsed_ns=-1),
            lambda value: value["records"][0].update(program="other"),
            lambda value: value["records"][0].update(input="injected"),
            lambda value: value["records"][0].update(extra="field"),
            lambda value: value.update(worker_sha256="0" * 64),
            lambda value: value.update(run_id="0" * 32),
            lambda value: value.update(uid=True),
            lambda value: value.update(hostname=""),
            lambda value: value.update(extra="field"),
            lambda value: value["limits"].update(cpu_seconds=200),
        )
        for mutate in mutations:
            observation = copy.deepcopy(self.observation)
            mutate(observation)
            with self.subTest(mutation=mutate), self.assertRaises(ValueError):
                cpu.compare(self.manifest, observation)

    def test_private_manifest_drift_and_cross_run_replay_reject(self):
        for mutate in (
            lambda value: value["tasks"][0]["private"]["cases"][0].update(expected=99),
            lambda value: value["schedule"].pop(),
            lambda value: value["budget"].update(calls=2),
            lambda value: value.update(kind="synthetic"),
        ):
            manifest = copy.deepcopy(self.manifest)
            mutate(manifest)
            with self.assertRaises(ValueError):
                cpu.compare(manifest, self.observation)
        with self.assertRaises(ValueError):
            cpu.compare(cpu.freeze("audit"), self.observation)

    def test_reference_disagreement_quarantines_not_relabels(self):
        observation = copy.deepcopy(self.observation)
        observation["records"][0]["value"] = 99
        result = cpu.compare(self.manifest, observation)
        self.assertFalse(result["admitted"])
        self.assertFalse(result["tasks"][0]["admitted"])
        self.assertEqual(result["scheduled_calls"], 1209)

    def test_comparison_results_are_detached_from_frozen_material(self):
        manifest = cpu.freeze("probe", "dev-001", "correct")
        observation = copy.deepcopy(self.observation)
        observation["run_id"] = manifest["run_id"]
        wanted = {row["case_id"] for row in manifest["schedule"]}
        observation["records"] = [
            row for row in observation["records"]
            if row["task_id"] == "dev-001" and row["program"] == "correct"
            and row["case_id"] in wanted
        ]
        original = copy.deepcopy(manifest)
        comparison = cpu.compare(manifest, observation)
        comparison["tool_view"]["results"][0]["input"][0] = 99
        self.assertEqual(manifest, original)
        cpu.compare(manifest, observation)

    def test_audit_results_are_detached_from_frozen_material(self):
        audit = copy.deepcopy(self.manifest)
        original = copy.deepcopy(audit)
        comparison = cpu.compare(audit, self.observation)
        comparison["tasks"][0]["failure_witnesses"]["defect"][0]["input"][0] = 99
        self.assertEqual(audit, original)

    def test_output_directory_collision_prevents_launch(self):
        with mock.patch.object(cpu.subprocess, "run") as launch:
            with self.assertRaises(FileExistsError):
                cpu.run(self.root / "audit", "audit")
            launch.assert_not_called()

    def test_failure_preserves_raw_data_and_never_retries(self):
        failures = (
            ("timeout", subprocess.TimeoutExpired(["fixed"], 5, output=b"partial", stderr=b"err")),
            ("launch_error", OSError("unavailable")),
            ("nonzero_exit", subprocess.CompletedProcess(["fixed"], 7, b"bad", b"err")),
            ("invalid_output", subprocess.CompletedProcess(["fixed"], 0, b'{"x":1,"x":2}', b"")),
            ("output_limit", subprocess.CompletedProcess(
                ["fixed"], 0, b" " * (2 * 1024 * 1024 + 1), b"")),
        )
        for status, failure in failures:
            with self.subTest(status=status):
                target = self.root / status
                options = {"side_effect": failure} if isinstance(failure, Exception) else {
                    "return_value": failure,
                }
                with mock.patch.object(cpu.subprocess, "run", **options) as launch:
                    result = cpu.run(target, "probe", "dev-001", "correct")
                self.assertEqual(result["status"], status)
                self.assertIsNone(result["comparison"])
                self.assertEqual(result["scheduled_calls"], 2)
                self.assertIsNone(result["observed_calls"])
                self.assertTrue((target / "stdout.json").exists())
                self.assertTrue((target / "stderr.txt").exists())
                self.assertEqual(launch.call_count, 1)

    def test_worker_cli_has_no_source_input_or_command_loading(self):
        worker = self.root / "audit/development_programs.py"
        for arguments in (["probe", "dev-001", "other"], ["--source", "print(1)"],
                          ["probe", "dev-001", "correct", "--input", "[]"]):
            process = subprocess.run(
                [sys.executable, "-I", "-B", str(worker), *arguments],
                capture_output=True, timeout=5,
            )
            self.assertNotEqual(process.returncode, 0)
            self.assertEqual(process.stdout, b"")


if __name__ == "__main__":
    unittest.main()
