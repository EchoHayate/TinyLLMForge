import importlib
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock


class AdmissionTests(unittest.TestCase):
    def setUp(self):
        self.assertTrue(Path(__file__).with_name("candidate_execution_admission.py").exists(),
                        "admission implementation missing")
        self.module = importlib.import_module("candidate_execution_admission")
        self.request = {
            "version": 1, "round": 1, "attempt_id": "a" * 32, "task_id": "D01",
            "arm": "base", "case_id": "P01", "entry_point": "toy",
            "source": "def toy(value):\n    return value\n",
            "args": [7],
        }

    def admit(self, body):
        return self.module.admit_request({
            **self.request, "source": "def toy(value):\n    " + body + "\n"})

    def test_ordinary_calculations(self):
        bodies = (
            "return abs(value) if value < 0 else value",
            "result = []\n    for item in value:\n        result.append(item + 1)\n    return result",
            "return [item * 2 for item in value if item > 0]",
            "return {item: item + 1 for item in value}",
            "return sum(item for item in value)",
            "return value.strip().lower().replace('a', 'b')",
            "while value < 10:\n        value += 1\n    return value",
            "try:\n        return int(value)\n    except ValueError:\n        return 0",
            "if value < 0:\n        raise ValueError('negative')\n    return value",
            "return toy(value - 1) if value > 0 else 0",
        )
        for body in bodies:
            with self.subTest(body=body):
                self.assertEqual(self.admit(body)["kind"], "COMPUTATION_AST_ADMISSION")

    def test_reject_reflection_native_and_process_access(self):
        for body in (
            "return __import__('os').getpid()", "return globals()",
            "return getattr(value, '__class__')", "return value.__class__",
            "return open('/tmp/x').read()", "return eval(value)", "return type(value)",
            "return value.gi_frame", "return value.__class__.__mro__",
            "return object()", "return input()", "return print(value)",
        ):
            with self.subTest(body=body), self.assertRaises(self.module.CandidateAdmissionError):
                self.admit(body)

    def test_reject_indirect_calls_attributes_and_bindings(self):
        for body in (
            "method = value.append\n    return method(1)",
            "return value[0](1)", "return (len if value else sum)(value)",
            "value.append = len\n    return value",
            "__builtins__ = {}\n    return value",
            "return '{}'.format(value)", "return value.encode()",
            "global name\n    return value",
            "return (lambda item: item)(value)",
            "return unknown(value)", "return unknown",
        ):
            with self.subTest(body=body), self.assertRaises(self.module.CandidateAdmissionError):
                self.admit(body)

    def test_no_unsupported_ast_nodes(self):
        for body in (
            "yield value", "with value:\n        pass\n    return 0",
            "return [*value]", "return (other := value)",
            "match value:\n        case 0:\n            return 0\n    return 1",
        ):
            with self.subTest(body=body), self.assertRaises(self.module.CandidateAdmissionError):
                self.admit(body)

    def test_record_binds_complete_request_and_policy(self):
        from validation_contract import digest
        record = self.admit("return value")
        request = {**self.request, "source": "def toy(value):\n    return value\n"}
        self.assertEqual(record, {
            "kind": "COMPUTATION_AST_ADMISSION", "version": 2,
            "policy_sha256":
                "16fb67e48d8adba83ad0cc832053d65d2b4d71a6b0075407d1c4a0e006410859",
            "request_sha256": digest(request),
        })
        self.assertEqual(self.module.POLICY_SHA256, record["policy_sha256"])
        for changes in ({"arm": "text"}, {"task_id": "D02"}, {"case_id": "P02"},
                        {"attempt_id": "b" * 32}, {"args": [8]},
                        {"source": request["source"] + "\n"}):
            changed = self.module.admit_request({**request, **changes})
            self.assertNotEqual(changed["request_sha256"], record["request_sha256"])
            self.assertEqual(changed["policy_sha256"], record["policy_sha256"])

    def test_scope_review_p0_callback_is_rejected_without_execution(self):
        with self.assertRaises(self.module.CandidateAdmissionError):
            self.admit('[eval for eval in []]\n    return list(map(eval, ["1 + 1"]))')

    def test_scope_comprehension_targets_do_not_authorize_outer_builtin_reads(self):
        for name in ("eval", "exec", "compile", "open", "getattr", "globals", "type"):
            for expression in (
                f"[{name} for {name} in []]",
                f"{{{name} for {name} in []}}",
                f"{{{name}: 0 for {name} in []}}",
                f"({name} for {name} in [])",
                f"[[{name} for {name} in []] for item in value]",
            ):
                with self.subTest(name=name, expression=expression):
                    with self.assertRaises(self.module.CandidateAdmissionError):
                        self.admit(f"{expression}\n    return list(map({name}, value))")

    def test_scope_outermost_iterable_uses_enclosing_scope(self):
        for body in (
            "return [eval for eval in eval]",
            "return {eval for eval in eval}",
            "return {eval: 0 for eval in eval}",
            "return (eval for eval in eval)",
            "return [item for item in eval for eval in value]",
            "return [[item for eval in eval] for item in value]",
            "return [item for item in [eval for eval in value] if eval]",
            "return [(eval, [eval for eval in value]) for item in value]",
        ):
            with self.subTest(body=body), self.assertRaises(self.module.CandidateAdmissionError):
                self.admit(body)

    def test_scope_legitimate_comprehension_shadowing_and_nested_iterables(self):
        for body in (
            "return [eval for eval in value]",
            "return {eval for eval in value}",
            "return {eval: eval for eval in value}",
            "return (eval for eval in value)",
            "eval = value\n    return [eval for eval in eval]",
            "return [value for value in value]",
            "return [[item for item in eval] for eval in value]",
            "return [inner for outer in value for inner in outer if inner]",
            "return [[item for item in item] for item in value]",
            "return [[item for item in outer] for outer in value if outer]",
            "return [inner for outer in value for inner in "
            "[eval for eval in outer]]",
            "result = [item for item in value]\n    item = 1\n    return item",
        ):
            with self.subTest(body=body):
                self.assertEqual(self.admit(body)["kind"], "COMPUTATION_AST_ADMISSION")

    def test_scope_reserved_builtin_and_entry_bindings_are_rejected(self):
        for name in sorted(self.module.CALLS | {"toy"}):
            for body in (
                f"{name} = abs\n    return {name}(value)",
                f"{name} = 0\n    return value",
                f"{name} += 1\n    return value",
                f"for {name} in value:\n        pass\n    return value",
                f"return [{name} for {name} in value]",
                f"try:\n        return value\n    except Exception as {name}:\n"
                "        return 0",
                f"{name}, other = value\n    return other",
            ):
                with self.subTest(name=name, body=body):
                    with self.assertRaises(self.module.CandidateAdmissionError):
                        self.admit(body)
            for signature in (name, f"*{name}", f"**{name}", f"*, {name}", f"{name}, /"):
                with self.subTest(signature=signature):
                    with self.assertRaises(self.module.CandidateAdmissionError):
                        self.module.admit_request({
                            **self.request,
                            "source": f"def toy({signature}):\n    return 0\n",
                        })
        for name in sorted(self.module.CALLS):
            with self.subTest(entry=name), self.assertRaises(self.module.CandidateAdmissionError):
                self.module.admit_request({
                    **self.request, "entry_point": name,
                    "source": f"def {name}(value):\n    return value\n",
                })

    def test_scope_builtin_laundering_and_indirect_aliases_are_rejected(self):
        for body in (
            "[eval for eval in []]\n    abs = eval\n    return abs(value)",
            "[globals for globals in []]\n    abs = globals\n    return abs()",
            "abs = len\n    return abs(value)",
            "toy = abs\n    return toy(value)",
            "helper = abs\n    return helper(value)",
            "helper = toy\n    return helper(value)",
            "abs = len\n    return list(map(abs, value))",
        ):
            with self.subTest(body=body), self.assertRaises(self.module.CandidateAdmissionError):
                self.admit(body)

    def test_scope_nonreserved_function_bindings_remain_ordinary_locals(self):
        for body in (
            "eval = value\n    return eval",
            "for eval in value:\n        pass\n    return eval",
            "try:\n        return value\n    except ValueError as problem:\n"
            "        return str(problem)",
            "first, second = value\n    return first + second",
            "return list(map(abs, value))",
            "return sorted(value, key=abs)",
            "return toy(value - 1) if value else 0",
        ):
            with self.subTest(body=body):
                self.assertEqual(self.admit(body)["kind"], "COMPUTATION_AST_ADMISSION")

    def test_declared_method_families_are_admitted_only_at_call_sites(self):
        for method in sorted(self.module.METHODS):
            with self.subTest(method=method):
                self.admit(f"return value.{method}()")
                with self.assertRaises(self.module.CandidateAdmissionError):
                    self.admit(f"return value.{method}")

    def test_declared_statement_expression_forms_are_traversed(self):
        for body in (
            "assert value\n    return value[1:-1:2]",
            "result = 0\n    for item in value:\n        if item:\n"
            "            continue\n        break\n    else:\n        result = 1\n"
            "    return result",
            "try:\n        assert value\n    except (ValueError, TypeError) as problem:\n"
            "        raise RuntimeError('failure') from problem\n    else:\n"
            "        value = 1\n    finally:\n        value += 1\n    return value",
            "while value:\n        value -= 1\n    else:\n        pass\n    return value",
            "return {value, 1}",
            "return f'{value!r:>10}'",
            "value[0] = 1\n    return value",
            "return not value or value and ~value",
            "return (+value, -value, value is None, value is not None, "
            "value in [1], value not in [2])",
        ):
            with self.subTest(body=body):
                self.admit(body)
        for operator in ("+", "-", "*", "@", "/", "//", "%", "**", "<<", ">>",
                         "|", "^", "&", "==", "!=", "<", "<=", ">", ">="):
            with self.subTest(operator=operator):
                self.admit(f"return value {operator} 2")

    def test_forbidden_fstring_exception_and_unpacking_paths(self):
        for body in (
            "return f'{globals()}'", "return f'{value:{globals()}}'",
            "return f'{value.__class__}'", "return f'{value:{unknown}}'",
            "try:\n        return value\n    except Exception as __private:\n"
            "        return 0",
            "return dict(**value)", "return dict(__private=value)",
            "return {**value}", "return (*value,)", "return {*value}",
            "return len(*value)", "first, *rest = value\n    return first",
            "return value.format_map({})", "return value.decode()",
            "return [item async for item in value]",
        ):
            with self.subTest(body=body), self.assertRaises(self.module.CandidateAdmissionError):
                self.admit(body)

    def test_hostile_builtin_subclasses_rejected_without_hooks(self):
        import candidate_execution_controller as controller
        hooks = []

        def hostile(*arguments, **keywords):
            hooks.append("called")
            raise AssertionError("host object hook invoked")

        overrides = {name: hostile for name in (
            "__iter__", "__getitem__", "__len__", "__repr__", "__str__", "__eq__",
            "__int__", "__float__", "items", "keys", "encode",
        )}
        hostile_list = type("HostileList", (list,), overrides)
        hostile_dict = type("HostileDict", (dict,), overrides)
        hostile_int = type("HostileInt", (int,), overrides)
        hostile_float = type("HostileFloat", (float,), overrides)
        hostile_str = type("HostileStr", (str,), {**overrides, "__hash__": str.__hash__})
        keyed = dict(self.request)
        del keyed["args"]
        keyed[hostile_str("args")] = [7]
        requests = [
            hostile_dict(self.request), keyed,
            {**self.request, "source": hostile_str(self.request["source"])},
            {**self.request, "args": hostile_list([7])},
        ]
        for value in (hostile_list([7]), hostile_dict({"item": 7}), hostile_int(7),
                      hostile_float(7), hostile_str("text"),
                      {hostile_str("item"): 7}):
            requests.append({**self.request, "args": [{"nested": [value]}]})
        for ordinal, request in enumerate(requests):
            for adapter in (False, True):
                with self.subTest(ordinal=ordinal, adapter=adapter):
                    hooks.clear()
                    with tempfile.TemporaryDirectory() as folder:
                        with mock.patch.object(controller, "run_request") as launch:
                            with self.assertRaises(self.module.CandidateAdmissionError):
                                if adapter:
                                    self.module.run_candidate_request(
                                        request, [], artifact_dir=folder, cwd=folder, env={})
                                else:
                                    self.module.admit_request(request)
                        launch.assert_not_called()
                        self.assertEqual(list(Path(folder).iterdir()), [])
                    self.assertEqual(hooks, [])

    def test_nested_snapshot_has_no_shared_containers(self):
        import candidate_execution_controller as controller
        from validation_contract import digest
        request = {**self.request, "args": [{"nested": [{"values": [7]}]}]}
        expected = digest(request)

        def launched(snapshot, argv, **options):
            self.assertIsNot(snapshot, request)
            self.assertIsNot(snapshot["args"], request["args"])
            self.assertIsNot(snapshot["args"][0], request["args"][0])
            self.assertIsNot(snapshot["args"][0]["nested"], request["args"][0]["nested"])
            self.assertIsNot(snapshot["args"][0]["nested"][0],
                             request["args"][0]["nested"][0])
            self.assertIsNot(snapshot["args"][0]["nested"][0]["values"],
                             request["args"][0]["nested"][0]["values"])
            request["args"][0]["nested"][0]["values"].append(99)
            request["args"][0]["extra"] = True
            self.assertEqual(snapshot["args"], [{"nested": [{"values": [7]}]}])
            return {"receipt": {"request_sha256": expected}, "audit": {}}

        with tempfile.TemporaryDirectory() as folder:
            with mock.patch.object(controller, "run_request", side_effect=launched) as launch:
                result = self.module.run_candidate_request(
                    request, [], artifact_dir=folder, cwd=folder, env={})
            launch.assert_called_once()
            self.assertEqual(result["admission"]["request_sha256"], expected)

    def test_persistence_write_and_close_failures_never_launch(self):
        import candidate_execution_controller as controller
        original_fdopen = os.fdopen
        for phase in ("write", "close", "short_write"):
            with self.subTest(phase=phase), tempfile.TemporaryDirectory() as folder:
                failure = OSError(f"{phase} failed")
                streams = []

                def opened(descriptor, mode):
                    stream = original_fdopen(descriptor, mode)
                    streams.append(stream)
                    output = mock.MagicMock(spec=type(stream), wraps=stream)
                    output.__enter__.return_value = output
                    if phase == "write":
                        output.write.side_effect = failure
                    elif phase == "short_write":
                        output.write.return_value = 0

                    def closed(*arguments):
                        stream.close()
                        if phase == "close":
                            raise failure
                        return False

                    output.__exit__.side_effect = closed
                    return output

                with mock.patch.object(controller, "run_request") as launch:
                    with mock.patch.object(self.module.os, "fdopen", side_effect=opened):
                        with self.assertRaises(self.module.CandidateAdmissionError) as raised:
                            self.module.run_candidate_request(
                                self.request, [], artifact_dir=folder, cwd=folder, env={})
                launch.assert_not_called()
                self.assertTrue(all(stream.closed for stream in streams))
                self.assertFalse((Path(folder) / "claim.json").exists())
                if phase != "short_write":
                    self.assertIs(raised.exception.__cause__, failure)

    def test_controller_infrastructure_failure_retains_admission_without_retry(self):
        import candidate_execution_controller as controller
        expected = self.module.admit_request(self.request)
        failure = self.module.InfrastructureError("collector failed")
        with tempfile.TemporaryDirectory() as folder:
            with mock.patch.object(controller, "run_request", side_effect=failure) as launch:
                with self.assertRaises(self.module.InfrastructureError) as raised:
                    self.module.run_candidate_request(
                        self.request, [], artifact_dir=folder, cwd=folder, env={})
            self.assertIs(raised.exception, failure)
            launch.assert_called_once()
            self.assertEqual(json.loads((Path(folder) / "admission.json").read_bytes()),
                             expected)
            self.assertEqual(sorted(path.name for path in Path(folder).iterdir()),
                             ["admission.json"])

    def test_denial_precedes_launch_and_claim(self):
        import candidate_execution_controller as controller
        with tempfile.TemporaryDirectory() as folder:
            request = {**self.request, "source": "def toy(value):\n    return open(value)\n"}
            with mock.patch.object(controller, "run_request") as launch:
                with self.assertRaises(self.module.CandidateAdmissionError):
                    self.module.run_candidate_request(
                        request, [], artifact_dir=folder, cwd=folder, env={})
            launch.assert_not_called()
            self.assertEqual(list(Path(folder).iterdir()), [])

    def test_admission_is_persisted_and_request_is_not_shared(self):
        import candidate_execution_controller as controller
        from validation_contract import digest
        original = dict(self.request)
        expected = digest(original)

        def launched(snapshot, argv, **options):
            self.request["args"][0] = 99
            self.assertEqual(snapshot["args"], [7])
            return {"receipt": {"request_sha256": expected, "status": "program_error"},
                    "audit": {}}

        with tempfile.TemporaryDirectory() as folder:
            with mock.patch.object(controller, "run_request", side_effect=launched):
                result = self.module.run_candidate_request(
                    self.request, [], artifact_dir=folder, cwd=folder, env={})
            saved = json.loads((Path(folder) / "admission.json").read_bytes())
            self.assertEqual(saved, result["admission"])
            self.assertEqual(saved["request_sha256"], expected)

    def test_bad_request_does_not_become_candidate_feedback(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaises(self.module.CandidateAdmissionError):
                self.module.run_candidate_request(
                    {**self.request, "round": 2}, [], artifact_dir=folder, cwd=folder, env={})
            self.assertEqual(list(Path(folder).iterdir()), [])

    def test_persisted_admission_precedes_launch_and_is_exclusive(self):
        import candidate_execution_controller as controller
        from validation_contract import digest

        def launched(snapshot, argv, **options):
            record = json.loads((Path(options["artifact_dir"]) / "admission.json").read_bytes())
            self.assertEqual(record["request_sha256"], digest(snapshot))
            return {"receipt": {"request_sha256": digest(snapshot)}, "audit": {}}

        with tempfile.TemporaryDirectory() as folder:
            with mock.patch.object(controller, "run_request", side_effect=launched) as launch:
                self.module.run_candidate_request(
                    self.request, [], artifact_dir=folder, cwd=folder, env={})
                before = (Path(folder) / "admission.json").read_bytes()
                with self.assertRaises(self.module.CandidateAdmissionError):
                    self.module.run_candidate_request(
                        self.request, [], artifact_dir=folder, cwd=folder, env={})
                self.assertEqual(launch.call_count, 1)
            self.assertEqual((Path(folder) / "admission.json").read_bytes(), before)
            self.assertEqual((Path(folder) / "admission.json").stat().st_mode & 0o777, 0o600)

    def test_request_digest_mismatch_is_protocol_abort(self):
        import candidate_execution_controller as controller
        with tempfile.TemporaryDirectory() as folder:
            with mock.patch.object(controller, "run_request", return_value={
                    "receipt": {"request_sha256": "0" * 64}, "audit": {}}):
                with self.assertRaises(self.module.CandidateAdmissionError):
                    self.module.run_candidate_request(
                        self.request, [], artifact_dir=folder, cwd=folder, env={})

    def test_real_controller_with_fixed_trusted_helpers(self):
        prefix = (
            "import os,sys,time\n"
            "control=int(sys.argv[1])\n"
            "os.write(control,b'TLF_CANDIDATE_READY_V1\\n')\n"
            "payload=sys.stdin.buffer.read()\n"
            "os.write(control,b'{\"kind\":\"return\",\"value\":7}\\n')\n"
        )
        for suffix, status in (
            ("", "completed"),
            ("sys.exit(9)\n", "program_error"),
            ("time.sleep(30)\n", "timeout"),
            ("while True:\n    os.write(1,b'x'*8192)\n", "output_limit"),
        ):
            with self.subTest(status=status), tempfile.TemporaryDirectory() as folder:
                result = self.module.run_candidate_request(
                    self.request,
                    [sys.executable, "-I", "-S", "-B", "-c", prefix + suffix, "{control_fd}"],
                    artifact_dir=folder, cwd=folder, env={"PATH": os.defpath},
                    startup_timeout=2.0, candidate_timeout=0.2, output_limit=4096)
                self.assertEqual(result["receipt"]["status"], status)
                self.assertTrue(result["audit"]["ready"])
                self.assertTrue(result["audit"]["input_complete"])
                self.assertTrue(result["audit"]["reaped"])
                for name in ("admission", "receipt"):
                    saved = json.loads((Path(folder) / (name + ".json")).read_bytes())
                    self.assertEqual(saved, result[name])
                self.assertEqual(result["admission"]["request_sha256"],
                                 result["receipt"]["request_sha256"])


if __name__ == "__main__":
    unittest.main()
