import importlib
import json
from pathlib import Path
import subprocess
import sys
import unittest


class WorkerTests(unittest.TestCase):
    def setUp(self):
        self.assertTrue(Path(__file__).with_name("candidate_execution_worker.py").exists(),
                        "worker implementation missing")
        self.module = importlib.import_module("candidate_execution_worker")

    def payload(self, body="return value", args=None):
        return json.dumps({"source": "def toy(value):\n    " + body + "\n",
                           "entry_point": "toy", "args": [7] if args is None else args}).encode()

    def test_fixed_toy_return(self):
        self.assertEqual(json.loads(self.module.evaluate(self.payload())),
                         {"kind": "return", "value": 7})

    def test_fixed_toy_value_error(self):
        self.assertEqual(json.loads(self.module.evaluate(self.payload("raise ValueError('x')"))),
                         {"kind": "raises", "type": "ValueError"})

    def test_other_exception_is_not_serialized(self):
        with self.assertRaises(RuntimeError):
            self.module.evaluate(self.payload("raise RuntimeError('x')"))

    def test_receipt_shaped_return_stays_a_value(self):
        value = {"status": "completed", "outcome": {"kind": "return", "value": 0}}
        result = json.loads(self.module.evaluate(self.payload(args=[value])))
        self.assertEqual(result, {"kind": "return", "value": value})

    def test_scalar_types_stay_distinct(self):
        for value in (True, 1, 1.0, None, "1", [1], {"a": 1}):
            with self.subTest(value=value):
                actual = json.loads(self.module.evaluate(self.payload(args=[value])))["value"]
                self.assertIs(type(actual), type(value))
                self.assertEqual(actual, value)

    def test_non_json_return_is_rejected(self):
        for body in ("return {1, 2}", "return (1, 2)", "return float('nan')",
                     "return {1: 'a'}", "return object()"):
            with self.subTest(body=body), self.assertRaises(ValueError):
                self.module.evaluate(self.payload(body))

    def test_no_extra_fields_or_duplicate_keys(self):
        for raw in (b'{"source":"x","source":"y"}', b'{}', b'[]',
                    self.payload()[:-1] + b',"expected":7}', b'{"x":NaN}', b'\xff'):
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                self.module.evaluate(raw)

    def test_unsupported_source_is_never_executed(self):
        for source in ("raise RuntimeError('must not run')",
                       "def toy(value=print('must not run')):\n    return value",
                       "@print('must not run')\ndef toy(value):\n    return value",
                       "def other(value):\n    return value",
                       "import os\ndef toy(value):\n    return value",
                       "def toy(value):\n    import os\n    return value"):
            with self.subTest(source=source), self.assertRaises(ValueError):
                self.module.evaluate(json.dumps(
                    {"source": source, "entry_point": "toy", "args": [7]}).encode())

    def test_oversize_result_exposes_only_one_detection_byte(self):
        result = self.module.evaluate(self.payload("return 'x' * 20000"))
        self.assertEqual(len(result), 16385)
        self.assertTrue(result.startswith(b'{"kind": "return", "value": "'))

    def test_large_input_is_not_subject_to_result_byte_limit(self):
        result = self.module.evaluate(self.payload("return 7", ["x" * 20000]))
        self.assertEqual(json.loads(result)["value"], 7)

    def test_many_input_nodes_are_not_subject_to_result_budget(self):
        result = self.module.evaluate(self.payload("return len(value)", [list(range(4096))]))
        self.assertEqual(json.loads(result)["value"], 4096)

    def test_cyclic_result_is_rejected(self):
        with self.assertRaises(ValueError):
            self.module.evaluate(self.payload("value.append(value)\n    return value", [[]]))

    def test_serve_refuses_unqualified_local_runtime_before_ready(self):
        with self.assertRaises(RuntimeError):
            self.module.serve(-1)

    def test_json_module_patch_cannot_replace_function_result(self):
        source = (
            "def toy(value):\n"
            "    __import__('json').encoder._make_iterencode = lambda *args, **kwargs: "
            "lambda *args: iter(['{\"kind\":\"return\",\"value\":999}'])\n"
            "    return 7\n"
        )
        raw = json.dumps({"source": source, "entry_point": "toy", "args": [0]}).encode()
        worker = Path(__file__).with_name("candidate_execution_worker.py")
        program = (
            "import importlib.util,sys\n"
            f"spec=importlib.util.spec_from_file_location('fixed_worker',{str(worker)!r})\n"
            "module=importlib.util.module_from_spec(spec)\nspec.loader.exec_module(module)\n"
            f"sys.stdout.buffer.write(module.evaluate({raw!r}))\n"
        )
        result = subprocess.run([sys.executable, "-I", "-S", "-B", "-c", program],
                                capture_output=True, timeout=5, check=True)
        self.assertEqual(json.loads(result.stdout)["value"], 7)

    def isolated_evaluate(self, body):
        raw = self.payload(body)
        worker = Path(__file__).with_name("candidate_execution_worker.py")
        program = (
            "import importlib.util,sys\n"
            f"spec=importlib.util.spec_from_file_location('fixed_worker',{str(worker)!r})\n"
            "module=importlib.util.module_from_spec(spec)\nspec.loader.exec_module(module)\n"
            f"sys.stdout.buffer.write(module.evaluate({raw!r}))\n"
        )
        return subprocess.run([sys.executable, "-I", "-S", "-B", "-c", program],
                              capture_output=True, timeout=5)

    def test_bytes_builtin_patch_cannot_replace_outcome(self):
        result = self.isolated_evaluate(
            "__import__('builtins').bytes = lambda value: "
            "b'{\"kind\":\"return\",\"value\":999}'\n    return 7")
        self.assertEqual(result.returncode, 0)
        self.assertEqual(json.loads(result.stdout)["value"], 7)

    def test_exception_builtin_patch_cannot_relabel_error(self):
        result = self.isolated_evaluate(
            "__import__('builtins').ValueError = RuntimeError\n"
            "    raise RuntimeError('not a ValueError')")
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(result.stdout, b"")

    def test_type_builtin_patch_cannot_coerce_tuple(self):
        result = self.isolated_evaluate(
            "__import__('builtins').type = lambda value: int\n    return (1, 2)")
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(result.stdout, b"")


if __name__ == "__main__":
    unittest.main()
