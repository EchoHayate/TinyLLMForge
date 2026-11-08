import json
import pathlib
import subprocess
import sys
import tempfile
import unittest


class DevelopmentCLITests(unittest.TestCase):
    def test_cli_runs_and_rejects_reusing_destination(self):
        script = pathlib.Path(__file__).with_name("development_cpu.py")
        with tempfile.TemporaryDirectory() as directory:
            output = pathlib.Path(directory) / "probe"
            command = [
                sys.executable, "-B", str(script), "probe", "--task-id", "dev-002",
                "--initial", "defect", "--output-dir", str(output),
            ]
            process = subprocess.run(command, capture_output=True, timeout=10)
            self.assertEqual(process.returncode, 0, process.stderr)
            result = json.loads(process.stdout)
            self.assertEqual(result["status"], "completed")
            self.assertEqual(result["observed_calls"], 2)
            frozen = (output / "manifest.json").read_bytes()
            collision = subprocess.run(command, capture_output=True, timeout=10)
            self.assertNotEqual(collision.returncode, 0)
            self.assertEqual((output / "manifest.json").read_bytes(), frozen)

    def test_cli_help_does_not_run_or_import_a_model(self):
        process = subprocess.run(
            [sys.executable, "-B", str(pathlib.Path(__file__).with_name("development_cpu.py")),
             "--help"],
            capture_output=True, timeout=5,
        )
        self.assertEqual(process.returncode, 0)
        self.assertIn(b"fixed development", process.stdout)


if __name__ == "__main__":
    unittest.main()
