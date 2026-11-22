import base64
import builtins
import copy
import hashlib
import importlib
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import time
import types
import unittest
from unittest import mock

import test_development_coordinator as coordinator_fixtures
import test_development_startup_evidence as startup_fixtures
from tokenizer_measurement import _state, measure
from validation_contract import digest, encoded, parse_json


STAGE = Path("/tmp/tlf-candidate-executor-20261117-gb1gZX")
WORKER = (STAGE / "remote-admission-v1/sandbox_worker.py").read_bytes()
LAUNCHER = (STAGE / "remote-admission-v1/host_launcher.py").read_bytes()


class Tokenizer:
    chat_template = "synthetic template"
    eos_token_id = 0
    pad_token_id = 0
    count = 10

    def get_vocab(self):
        return {"eos": 0, "source": 1}

    def apply_chat_template(self, messages, *, tokenize, **keywords):
        return [1] * self.count if tokenize else "synthetic prompt"

    def __call__(self, text, **keywords):
        return {"input_ids": [1] * self.count}

    def decode(self, values, *, skip_special_tokens, clean_up_tokenization_spaces):
        if values != [1] or skip_special_tokens or clean_up_tokenization_spaces:
            raise AssertionError("decode contract")
        return "def toy(value):\n    return value\n"


class DriverFixture(unittest.TestCase):
    def setUp(self):
        self.assertTrue(Path(__file__).with_name("development_remote_driver.py").is_file(),
                        "fixed driver implementation missing")
        self.driver = importlib.import_module("development_remote_driver")
        self.fixture = coordinator_fixtures.RuntimeFixture()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.parent = Path(self.fixture.folder.name)
        self.root = self.parent / ("development-" + self.fixture.run_id)
        self.tokenizer = Tokenizer()
        self.model = object()
        self.bindings = self.fixture.bindings
        self.bindings["tokenizer_state"] = _state(self.tokenizer)[0]
        self.frames = []
        self.state = self.driver._new_state(self.fixture.run_id, time.monotonic_ns() + 10 ** 12)
        original_import = builtins.__import__

        def guarded(name, *arguments, **keywords):
            if name.split(".")[0] in ("torch", "transformers"):
                raise AssertionError("real model import forbidden")
            return original_import(name, *arguments, **keywords)

        for patch in (
            mock.patch("builtins.__import__", side_effect=guarded),
            mock.patch.object(subprocess, "Popen", side_effect=AssertionError("launch forbidden")),
            mock.patch.object(socket, "socket", side_effect=AssertionError("network forbidden")),
        ):
            patch.start()
            self.addCleanup(patch.stop)

    def case(self, name="case"):
        path = self.parent / name
        path.mkdir()
        return path

    def observer(self, model, tokenizer, messages, **keywords):
        self.assertIs(model, self.model)
        self.assertIs(tokenizer, self.tokenizer)
        result = self.fixture.generate(messages, **keywords)
        generation = result["generation"]
        generation.update(device_type="cuda", peak_allocated_bytes=10, peak_reserved_bytes=20,
                          prompt_sha256=hashlib.sha256(b"synthetic prompt").hexdigest())
        (keywords["artifact_dir"] / "generation.json").write_bytes(encoded(generation))
        return result

    def callbacks(self):
        return self.driver.make_callbacks(
            self.model, self.tokenizer, bindings=self.bindings,
            launcher=LAUNCHER, worker=WORKER, driver_state=self.state)

    def call_generation(self, callback, path):
        return callback(
            [{"role": "user", "content": "synthetic"}],
            expected_tokenizer_sha256=digest(self.bindings["tokenizer_state"]),
            call_id="b" * 32, artifact_dir=path)

    def runner(self, request, argv, **keywords):
        self.assertEqual(keywords["cwd"], keywords["artifact_dir"])
        self.assertEqual(keywords["env"], {"PATH": "/usr/bin:/bin", "LC_ALL": "C"})
        phase = "final" if request["round"] == 2 else "public"
        self.assertEqual(argv, self.bindings["execution"][phase]["argv"])
        self.assertEqual({name: keywords[name] for name in self.driver.CHILD_LIMITS},
                         {name: self.bindings["execution"][phase][name] for name in self.driver.CHILD_LIMITS})
        directory = keywords["artifact_dir"]
        result = self.fixture.execute(request, artifact_dir=directory)
        raw = startup_fixtures.startup_bytes()
        (directory / "stdout.bin").write_bytes(raw)
        result["audit"]["stream_sha256"]["stdout"] = hashlib.sha256(raw).hexdigest()
        result["audit"]["output_bytes_observed"] += len(raw)
        (directory / "observation.json").write_bytes(encoded(result["audit"]))
        return result


class CallbackTests(DriverFixture):
    def test_genuine_input_limit_keeps_failed_observer_with_no_forward(self):
        observed = importlib.import_module("observed_generation")
        self.tokenizer.count = 4097
        self.model = mock.Mock()
        path = self.case()
        with mock.patch.object(self.driver, "generate_observed", wraps=observed.generate_observed) as observer:
            with self.assertRaises(observed.GenerationAttemptError) as raised:
                self.call_generation(self.callbacks()[0], path)
        self.assertEqual(observer.call_count, 1)
        self.assertEqual(str(raised.exception.__cause__), "input token limit exceeded")
        record = parse_json((path / "observation.json").read_bytes())
        self.assertEqual((record["forward_started"], record["forward_completed"]), (0, 0))
        self.assertEqual(record["status"], "failed")
        self.assertTrue((path / "claim.json").is_file())
        self.model.assert_not_called()

    def test_unreadable_startup_preserves_failure_but_rejects_success_in_both_domains(self):
        self.fixture.boundary_state()
        original = Path.read_bytes
        for phase, index in (("public", 1), ("final", 2)):
            for failed in (False, True):
                with self.subTest(phase=phase, failed=failed):
                    path = self.case(phase + str(failed))
                    primary = RuntimeError("runner primary")
                    unreadable = PermissionError("stdout unavailable")
                    module = self.driver.candidate_execution_admission if index == 1 \
                        else self.driver.development_final_admission

                    def read(current):
                        if current == path / "stdout.bin":
                            raise unreadable
                        return original(current)

                    with mock.patch.object(module, "run_candidate_request",
                                           side_effect=primary if failed else None, return_value={}):
                        with mock.patch.object(Path, "read_bytes", read):
                            with self.assertRaises(BaseException) as raised:
                                self.callbacks()[index](self.fixture.request(phase), artifact_dir=path)
                    self.assertIs(raised.exception, primary if failed else unreadable)
                    self.assertEqual(self.state["violations"], [])
                    row = parse_json((path / "driver-callback.json").read_bytes())
                    self.assertEqual(row["collection_errors"][0]["error"]["type"], "PermissionError")

    def test_malformed_cleanup_never_replaces_observer_primary(self):
        primary = RuntimeError("original")
        primary.record_cleanup = {"bad": object()}
        with mock.patch.object(self.driver, "generate_observed", side_effect=primary):
            with self.assertRaises(BaseException) as raised:
                self.call_generation(self.callbacks()[0], self.case())
        self.assertIs(raised.exception, primary)
        self.assertTrue(self.state["secondary_errors"])

    def test_measurement_mutation_stops_before_observer(self):
        def mutate(tokenizer, messages, **keywords):
            measured = measure(tokenizer, messages, **keywords)
            messages[0]["content"] = "mutated"
            return measured

        with mock.patch.object(self.driver, "measure", side_effect=mutate):
            with mock.patch.object(self.driver, "generate_observed", side_effect=RuntimeError("entered")) as observer:
                with self.assertRaises(BaseException):
                    self.call_generation(self.callbacks()[0], self.case())
        observer.assert_not_called()

    def test_generation_measures_once_and_preserves_observer_envelope(self):
        path = self.case()
        with mock.patch.object(self.driver, "generate_observed", side_effect=self.observer) as observer:
            with mock.patch.object(self.driver, "measure", wraps=measure) as measured:
                result = self.call_generation(self.callbacks()[0], path)
        self.assertEqual(observer.call_count, 1)
        self.assertEqual(measured.call_count, 1)
        self.assertEqual(set(result), {"generation", "observation", "record_cleanup"})
        saved = parse_json((path / "driver-measurement.json").read_bytes())
        self.assertEqual(saved["measurement"]["input"]["input_tokens"], 10)
        self.assertEqual(saved["messages_sha256"], result["generation"]["messages_sha256"])
        self.assertGreaterEqual(saved["elapsed_ns"], 0)
        self.assertTrue((path / "driver-callback.json").is_file())

    def test_input_limit_reaches_observer_once_with_original_refusal(self):
        for count in (4097, 5000):
            with self.subTest(count=count):
                path = self.case(str(count))
                self.tokenizer.count = count
                primary = RuntimeError("observer original refusal")

                def reject(model, tokenizer, messages, **keywords):
                    claim = {"call_id": keywords["call_id"]}
                    (path / "claim.json").write_bytes(encoded(claim))
                    (path / "observation.json").write_bytes(encoded({"status": "failed"}))
                    raise primary

                with mock.patch.object(self.driver, "generate_observed", side_effect=reject) as observer:
                    with self.assertRaises(RuntimeError) as raised:
                        self.call_generation(self.callbacks()[0], path)
                self.assertIs(raised.exception, primary)
                self.assertEqual(observer.call_count, 1)
                saved = parse_json((path / "driver-measurement.json").read_bytes())
                self.assertEqual(saved["measurement"]["input"]["input_tokens"], count)
                self.assertEqual(saved["measurement"]["status"], "INPUT_LIMIT")
                self.assertEqual(parse_json((path / "observation.json").read_bytes()), {"status": "failed"})

    def test_measurement_failure_and_save_failure_never_enter_observer(self):
        for failing in ("measure", "save"):
            with self.subTest(failing=failing):
                path = self.case(failing)
                primary = OSError("measurement unavailable")
                target = "measure" if failing == "measure" else "save_measurement"
                with mock.patch.object(self.driver, target, side_effect=primary):
                    with mock.patch.object(self.driver, "generate_observed") as observer:
                        with self.assertRaises(OSError) as raised:
                            self.call_generation(self.callbacks()[0], path)
                self.assertIs(raised.exception, primary)
                observer.assert_not_called()
                self.assertFalse((path / "claim.json").exists())
                row = parse_json((path / "driver-callback.json").read_bytes())
                self.assertFalse(row["observer_entered"])

    def test_successful_generation_mutations_rejected_and_cleanup_preserved(self):
        mutations = (
            lambda result: result["generation"].update(prompt_sha256="f" * 64),
            lambda result: result["generation"].update(input_tokens=9),
            lambda result: result["generation"].update(output_ids=[8, 0]),
            lambda result: result["generation"].update(device_type="cpu"),
            lambda result: result["generation"].update(peak_allocated_bytes=0),
            lambda result: result["generation"].update(candidate="changed"),
        )
        for ordinal, mutate in enumerate(mutations):
            with self.subTest(mutation=ordinal):
                path = self.case(str(ordinal))

                def changed(*arguments, **keywords):
                    result = self.observer(*arguments, **keywords)
                    mutate(result)
                    return result

                with mock.patch.object(self.driver, "generate_observed", side_effect=changed):
                    with self.assertRaises(self.driver.BoundaryProtocolError) as raised:
                        self.call_generation(self.callbacks()[0], path)
                self.assertEqual(raised.exception.record_cleanup, {"failed_count": 0, "failures": []})

    def test_startup_uses_only_exact_first_four_lines_and_pinned_worker(self):
        good = startup_fixtures.startup_bytes()
        self.assertEqual(self.driver.verify_startup(good + b"candidate stdout\n", WORKER),
                         startup_fixtures.startup_record())
        for raw, worker in ((b"wrong\n" + good, WORKER), (good, WORKER + b"\n"),
                            (good.replace(b"true", b"false", 1) + good, WORKER)):
            with self.subTest(raw=raw[:20]), self.assertRaises(self.driver.BoundaryProtocolError):
                self.driver.verify_startup(raw, worker)

    def test_both_execution_closures_use_original_domain_and_limits(self):
        self.fixture.boundary_state()
        for phase, index in (("public", 1), ("final", 2)):
            with self.subTest(phase=phase):
                path = self.case(phase)
                module = self.driver.candidate_execution_admission if index == 1 \
                    else self.driver.development_final_admission
                with mock.patch.object(module, "run_candidate_request", side_effect=self.runner) as runner:
                    result = self.callbacks()[index](self.fixture.request(phase), artifact_dir=path)
                self.assertEqual(runner.call_count, 1)
                self.assertEqual(set(result), {"admission", "audit", "receipt"})
                self.assertTrue((path / "driver-callback.json").is_file())

    def test_failed_execution_missing_partial_and_contradictory_startup_keep_primary(self):
        self.fixture.boundary_state()
        for phase, index in (("public", 1), ("final", 2)):
            for evidence in ("missing", "partial", "contradictory"):
                with self.subTest(phase=phase, evidence=evidence):
                    path = self.case(phase + evidence)
                    primary = RuntimeError("runner failed")

                    def fail(*arguments, **keywords):
                        if evidence != "missing":
                            raw = b"USERNS_POLICY_enabled\n" if evidence == "partial" else \
                                startup_fixtures.startup_bytes().replace(b"true", b"false", 1)
                            (path / "stdout.bin").write_bytes(raw)
                        raise primary

                    module = self.driver.candidate_execution_admission if index == 1 \
                        else self.driver.development_final_admission
                    before = len(self.state["violations"])
                    with mock.patch.object(module, "run_candidate_request", side_effect=fail):
                        with self.assertRaises(RuntimeError) as raised:
                            self.callbacks()[index](self.fixture.request(phase), artifact_dir=path)
                    self.assertIs(raised.exception, primary)
                    self.assertEqual(len(self.state["violations"]) - before,
                                     int(evidence == "contradictory"))


class ExportTests(DriverFixture):
    def test_unreadable_file_is_incomplete_without_losing_readable_file(self):
        root = self.case()
        (root / "good").write_bytes(b"evidence")
        (root / "bad").write_bytes(b"unreadable")
        original = self.driver._read_regular
        primary = PermissionError("unreadable")

        def read(directory, relative):
            if relative == "bad":
                raise primary
            return original(directory, relative)

        with mock.patch.object(self.driver, "_read_regular", side_effect=read):
            with self.assertRaises(self.driver.ExportError) as raised:
                self.driver.export_evidence(root, run_id=self.fixture.run_id, emit=self.frames.append)
        self.assertIs(raised.exception.__cause__, primary)
        self.assertEqual(raised.exception.receipt["exported_files"], 1)
        self.assertEqual(raised.exception.receipt["errors"][0]["path"], "bad")

    def test_mutation_during_chunk_emission_is_incomplete(self):
        root = self.case()
        (root / "record").write_bytes(b"before")

        def emit(frame):
            if frame["phase"] == "FILE_CHUNK":
                (root / "record").write_bytes(b"after")

        with self.assertRaises(self.driver.ExportError):
            self.driver.export_evidence(root, run_id=self.fixture.run_id, emit=emit)

    def test_export_exact_order_chunk_bytes_zero_files_and_no_mutation(self):
        root = self.case()
        (root / "b").write_bytes(b"z" * 65537)
        (root / "a").write_bytes(b"")
        before = {path.name: path.read_bytes() for path in root.iterdir()}
        receipt = self.driver.export_evidence(root, run_id=self.fixture.run_id, emit=self.frames.append)
        self.assertEqual(receipt["status"], "completed")
        self.assertEqual([item["path"] for item in self.frames[0]["inventory"]], ["a", "b"])
        chunks = [frame for frame in self.frames if frame["phase"] == "FILE_CHUNK"]
        self.assertEqual([frame["bytes"] for frame in chunks], [65536, 1])
        self.assertEqual(b"".join(base64.b64decode(frame["base64"]) for frame in chunks), before["b"])
        self.assertEqual({path.name: path.read_bytes() for path in root.iterdir()}, before)
        self.assertTrue(all(frame["run_id"] == self.fixture.run_id for frame in self.frames))
        self.assertEqual(self.frames[-1]["phase"], "EXPORT_END")

    def test_export_symlinks_and_file_changes_are_incomplete(self):
        for kind in ("symlink", "changed"):
            with self.subTest(kind=kind):
                root = self.case(kind)
                (root / "good").write_bytes(b"original")
                if kind == "symlink":
                    (root / "escape").symlink_to(self.parent)

                def emit(frame):
                    self.frames.append(frame)
                    if kind == "changed" and frame["phase"] == "EXPORT_BEGIN":
                        (root / "good").write_bytes(b"changed!")

                with self.assertRaises(self.driver.ExportError) as raised:
                    self.driver.export_evidence(root, run_id=self.fixture.run_id, emit=emit)
                self.assertEqual(raised.exception.receipt["status"], "incomplete")
                self.assertTrue(raised.exception.receipt["errors"])

    def test_export_emit_failure_keeps_exact_cause_and_existing_files(self):
        root = self.case()
        (root / "record").write_bytes(b"evidence")
        primary = BrokenPipeError("stream broken")
        with self.assertRaises(self.driver.ExportError) as raised:
            self.driver.export_evidence(root, run_id=self.fixture.run_id,
                                        emit=mock.Mock(side_effect=primary))
        self.assertIs(raised.exception.__cause__, primary)
        self.assertEqual((root / "record").read_bytes(), b"evidence")


class FlowTests(DriverFixture):
    def test_primary_survives_export_failure(self):
        primary = KeyboardInterrupt("interrupted preparation")
        export_error = BrokenPipeError("export channel")

        def emit(frame):
            if frame["phase"] == "FILE_CHUNK":
                raise export_error
            self.frames.append(frame)

        with self.assertRaises(self.driver.DriverRunError) as raised:
            self.run_driver(prepare=mock.Mock(side_effect=primary), emit=emit)
        self.assertIs(raised.exception.__cause__, primary)
        self.assertEqual(raised.exception.envelope["export"]["status"], "incomplete")
        self.assertTrue(any(row["phase"] == "export" for row in raised.exception.envelope["secondary_errors"]))

    def test_terminal_precommit_failure_uses_failure_record_without_reentry(self):
        primary = OSError("terminal link")
        original = os.link

        def link(source, destination, **keywords):
            if Path(destination).name == "driver-result.json":
                raise primary
            return original(source, destination, **keywords)

        with mock.patch.object(os, "link", side_effect=link):
            with self.assertRaises(self.driver.DriverRunError) as raised:
                self.run_driver()
        self.assertIs(raised.exception.__cause__, primary)
        self.assertEqual(raised.exception.envelope["terminal_path"], "driver-failure.json")
        self.assertEqual(len(self.fixture.calls), 16)
        self.assertFalse((self.root / "driver-result.json").exists())

    def test_nested_observer_publication_failure_survives_export(self):
        observed = importlib.import_module("observed_generation")
        self.tokenizer.count = 4097
        self.model = mock.Mock()
        publication = OSError("failed observation publication")
        cleanup = {"failed_count": 1, "failures": [{"record": "observation.json", "published": False}]}
        caught = []

        def save(root, name, record, *, cleanup):
            if name == "observation.json":
                cleanup.update(copy.deepcopy(expected_cleanup))
                raise publication
            return original_save(root, name, record, cleanup=cleanup)

        def observer(*arguments, **keywords):
            try:
                return observed.generate_observed(*arguments, **keywords)
            except observed.GenerationAttemptError as error:
                caught.append(error)
                raise

        expected_cleanup = cleanup
        original_save = observed._save
        with mock.patch.object(observed, "_save", side_effect=save):
            with self.assertRaises(self.driver.DriverRunError) as raised:
                self.run_driver(observer=observer)
        error = raised.exception
        self.assertEqual(len(caught), 1)
        self.assertIs(error.__cause__.__cause__, caught[0])
        self.assertIs(caught[0].observation_error, publication)
        details = error.envelope["failure"]["cause"]
        self.assertEqual(details.get("observation_error", {}).get("message"), str(publication))
        self.assertEqual(details.get("cause", {}).get("message"), str(caught[0].__cause__))
        self.assertEqual(details.get("record_cleanup"), cleanup)
        row = error.envelope["costs"]["callbacks"][0]
        self.assertEqual(row["failure"]["observation_error"]["message"], str(publication))
        self.assertFalse(list((self.root / "coordinator").rglob("observation.json")))
        saved = parse_json((self.root / error.envelope["terminal_path"]).read_bytes())
        self.assertEqual(saved["failure"], error.envelope["failure"])

    def test_error_details_are_bounded_cycle_safe_and_keep_cleanup(self):
        outer = RuntimeError("outer")
        inner = ValueError("inner")
        outer.__cause__ = inner
        inner.__cause__ = outer
        inner.observation_error = OSError("observer")
        inner.record_cleanup = {"failed_count": 0, "failures": []}
        details = self.driver._error_details(outer)
        self.assertEqual(details["cause"].get("observation_error", {}).get("message"), "observer")
        self.assertEqual(details["cause"]["cause"]["truncated"], "cycle")
        for unused in range(100):
            parent = RuntimeError("parent")
            parent.__cause__ = outer
            outer = parent
        self.assertLess(len(encoded(self.driver._error_details(outer))), 20000)

    def test_coordinator_event_failure_after_success_is_primary(self):
        primary = OSError("event transport failed")

        def emit(frame):
            if frame["phase"] == "COORDINATOR_ENTERED":
                raise primary
            self.frames.append(frame)

        with self.assertRaises(self.driver.DriverRunError) as raised:
            self.run_driver(emit=emit)
        self.assertIs(raised.exception.__cause__, primary)
        self.assertEqual(raised.exception.envelope["coordinator"]["report"]["status"], "completed")

    def package(self):
        self.assertTrue(callable(getattr(self.driver, "run_driver", None)), "driver flow missing")
        executable = self.parent / "python"
        if not executable.exists():
            executable.write_bytes(b"synthetic interpreter")
        model_root = self.parent / "checkpoint"
        model_root.mkdir(exist_ok=True)
        (model_root / "weights").write_bytes(b"synthetic weights")
        manifest = {
            "hostname": "synthetic-host", "uid": 1054, "python": "3.12.13",
            "executable": str(executable), "executable_sha256": hashlib.sha256(executable.read_bytes()).hexdigest(),
            "packages": str(self.parent / "packages"), "versions": {"torch": "fake", "transformers": "fake"},
            "gpu_uuid": "GPU-synthetic", "model_root": str(model_root),
            "checkpoint_files": {"weights": {"bytes": 17, "sha256": hashlib.sha256(b"synthetic weights").hexdigest()}},
            "artifact_parent": str(self.parent), "expected_base_sha256": "f" * 64,
            "expected_tokenizer_sha256": digest(self.bindings["tokenizer_state"]),
        }
        return {
            "kind": "LORA_DEVELOPMENT_PACKAGE", "version": 1, "run_id": self.fixture.run_id,
            "runtime_manifest": manifest,
            "limits": {
                "public": dict(zip(self.driver.CHILD_LIMITS, (10, 10, 65536, 16384, 2))),
                "final": dict(zip(self.driver.CHILD_LIMITS, (10, 10, 65536, 16384, 2))),
                "driver_timeout_seconds": 3600, "export_reserve_seconds": 120,
                "transport_timeout_seconds": 3900,
            },
            "module_order": list(self.driver.MODULE_ORDER),
            "modules": {name: Path(__file__).with_name(name + ".py").read_bytes()
                        for name in self.driver.MODULE_ORDER},
            "data": dict(zip(self.driver.DATA_NAMES, self.fixture.raw)),
            "launcher": LAUNCHER, "worker": WORKER,
        }

    def run_driver(self, *, prepare=None, integrity=None, observer=None, emit=None):
        package = self.package()
        self.bindings["execution"] = self.driver._bindings_for(package, self.tokenizer)["execution"]
        with mock.patch.object(self.driver, "_prepare_base", side_effect=prepare) as preparation:
            preparation.return_value = (self.model, self.tokenizer)
            with mock.patch.object(self.driver, "_final_integrity", side_effect=integrity) as final:
                final.return_value = {"verified": True}
                with mock.patch.object(self.driver, "_alarm"):
                    with mock.patch.object(self.driver, "generate_observed", side_effect=observer or self.observer):
                        with mock.patch.object(self.driver.candidate_execution_admission,
                                               "run_candidate_request", side_effect=self.runner):
                            with mock.patch.object(self.driver.development_final_admission,
                                                   "run_candidate_request", side_effect=self.runner):
                                return self.driver.run_driver(
                                    package, artifact_root=self.root,
                                    deadline_ns=time.monotonic_ns() + 10 ** 12,
                                    emit=emit or self.frames.append)

    def test_complete_driver_uses_real_coordinator_and_exports_immutable_terminal(self):
        result = self.run_driver()
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["coordinator"]["report"]["counts"]["completed_episodes"], 8)
        self.assertEqual(len(self.fixture.calls), 16)
        self.assertEqual(len(self.fixture.public_calls), 16)
        self.assertEqual(len(self.fixture.final_calls), 24)
        saved_raw = (self.root / result["terminal_path"]).read_bytes()
        saved = parse_json(saved_raw)
        self.assertIsNone(saved["export"])
        self.assertIsNone(saved["costs"]["driver_wall_ns"])
        self.assertIsNone(saved["costs"]["export_ns"])
        self.assertGreaterEqual(result["costs"]["driver_wall_ns"], result["costs"]["work_wall_ns"])
        self.assertGreaterEqual(result["costs"]["export_ns"], 0)
        self.assertEqual(hashlib.sha256(saved_raw).hexdigest(), result["terminal_sha256"])
        self.assertEqual(result["export"]["status"], "completed")
        self.assertEqual(self.frames[-1]["phase"], "DONE")
        self.assertIsNone(self.frames[-1]["envelope"]["costs"]["driver_wall_ns"])
        self.assertIsNone(self.frames[-1]["envelope"]["costs"]["terminal_emit_ns"])
        self.assertGreaterEqual(result["costs"]["driver_wall_ns"], result["costs"]["export_complete_wall_ns"])
        self.assertGreaterEqual(result["costs"]["terminal_emit_ns"], 0)
        self.assertFalse((self.root / "driver-failure.json").exists())
        exported = {}
        for frame in self.frames:
            if frame["phase"] == "FILE_CHUNK":
                exported.setdefault(frame["path"], bytearray()).extend(base64.b64decode(frame["base64"]))
        self.assertEqual(bytes(exported[result["terminal_path"]]), saved_raw)
        self.assertEqual(bytes(exported["coordinator/development-report.json"]),
                         (self.root / "coordinator/development-report.json").read_bytes())

    def test_partial_setup_preserves_exact_primary_and_exports_failure(self):
        self.package()
        primary = OSError("partial preparation")
        with self.assertRaises(self.driver.DriverRunError) as raised:
            self.run_driver(prepare=mock.Mock(side_effect=primary))
        self.assertIs(raised.exception.__cause__, primary)
        report = raised.exception.envelope
        self.assertIsNone(report["coordinator"])
        self.assertIsNone(report["costs"]["model_load_ns"])
        self.assertEqual(report["export"]["status"], "completed")
        self.assertEqual(self.fixture.calls, [])
        self.assertEqual(self.frames[-1]["phase"], "FAILED")

    def test_callback_failure_keeps_coordinator_envelope_and_secondary_integrity(self):
        self.package()
        primary = RuntimeError("observer failed")
        secondary = RuntimeError("final integrity failed")
        with self.assertRaises(self.driver.DriverRunError) as raised:
            self.run_driver(observer=mock.Mock(side_effect=primary), integrity=mock.Mock(side_effect=secondary))
        error = raised.exception
        self.assertIsInstance(error.__cause__, self.driver.DevelopmentRunError)
        self.assertIs(error.__cause__.__cause__, primary)
        self.assertEqual(error.envelope["coordinator"]["report"], error.__cause__.report)
        self.assertTrue(any(item["error"]["message"] == str(secondary)
                            for item in error.envelope["secondary_errors"]))
        self.assertEqual(error.envelope["export"]["status"], "completed")

    def test_late_integrity_failure_never_rewrites_coordinator_success(self):
        self.package()
        primary = RuntimeError("model state drift")
        with self.assertRaises(self.driver.DriverRunError) as raised:
            self.run_driver(integrity=mock.Mock(side_effect=primary))
        self.assertIs(raised.exception.__cause__, primary)
        self.assertEqual(raised.exception.envelope["coordinator"]["report"]["status"], "completed")
        self.assertTrue((self.root / "coordinator/development-report.json").is_file())
        self.assertFalse((self.root / "coordinator/partial-report.json").exists())

    def test_export_failure_does_not_rewrite_committed_driver_terminal(self):
        self.package()
        primary = BrokenPipeError("export disconnected")

        def emit(frame):
            if frame["phase"] == "FILE_CHUNK":
                raise primary
            self.frames.append(frame)

        with self.assertRaises(self.driver.DriverRunError) as raised:
            self.run_driver(emit=emit)
        self.assertIs(raised.exception.__cause__, primary)
        self.assertEqual(raised.exception.envelope["export"]["status"], "incomplete")
        self.assertIsNone(parse_json((self.root / "driver-result.json").read_bytes())["export"])
        self.assertFalse((self.root / "driver-failure.json").exists())
        self.assertNotIn("DONE", [frame["phase"] for frame in self.frames])

    def test_exclusive_root_and_expired_deadline_never_prepare_model(self):
        package = self.package()
        self.root.mkdir()
        with mock.patch.object(self.driver, "_prepare_base") as prepare:
            for root, deadline in ((self.root, time.monotonic_ns() + 10 ** 12),
                                   (self.parent / "wrong-root", 0)):
                with self.subTest(root=root), self.assertRaises(self.driver.DriverRunError):
                    self.driver.run_driver(package, artifact_root=root, deadline_ns=deadline, emit=self.frames.append)
            prepare.assert_not_called()
        self.assertEqual(list(self.root.iterdir()), [])

    def test_postcommit_unlink_is_nonfatal_and_export_contains_only_frozen_files(self):
        self.package()
        original = os.unlink

        def unlink(path, *arguments, **keywords):
            if Path(path).name.startswith(".driver-result.json."):
                raise OSError("postcommit cleanup")
            return original(path, *arguments, **keywords)

        with mock.patch.object(self.driver.os, "unlink", side_effect=unlink):
            result = self.run_driver()
        self.assertEqual(result["status"], "completed")
        self.assertGreater(result["record_cleanup"]["failed_count"], 0)
        self.assertEqual(result["export"]["status"], "completed")


class RuntimeTests(DriverFixture):
    package = FlowTests.package

    def test_loaded_source_verification_accepts_driver_contextmanager(self):
        package = self.package()
        for name in self.driver.MODULE_ORDER:
            if name not in ("training_inputs", "training_worker"):
                importlib.import_module(name)
        try:
            result = self.driver._verify_sources(package, deferred=("training_inputs", "training_worker"))
        except self.driver.BoundaryProtocolError as error:
            self.fail(str(error))
        self.assertEqual(result["development_remote_driver"],
                         hashlib.sha256(package["modules"]["development_remote_driver"]).hexdigest())

    def test_loaded_source_rebinding_or_removal_is_rejected(self):
        package = self.package()
        deferred = tuple(name for name in self.driver.MODULE_ORDER if name != "generation_core")
        module = importlib.import_module("generation_core")
        for value in (None, lambda: None):
            with self.subTest(value=value):
                with mock.patch.object(module, "generate_candidate", value):
                    with self.assertRaises(self.driver.BoundaryProtocolError):
                        self.driver._verify_sources(package, deferred=deferred)

    def test_contextmanager_wrapper_and_original_drift_are_rejected(self):
        package = self.package()
        deferred = tuple(name for name in self.driver.MODULE_ORDER if name != "development_remote_driver")
        original = self.driver._alarm
        for kind in ("wrapped", "wrapper"):
            with self.subTest(kind=kind):
                changed = self.driver.contextmanager(lambda: None)
                changed.__module__ = self.driver.__name__
                if kind == "wrapper":
                    changed = lambda unused: None
                    changed.__wrapped__ = original.__wrapped__
                    changed.__module__ = self.driver.__name__
                with mock.patch.object(self.driver, "_alarm", changed):
                    with self.assertRaises(self.driver.BoundaryProtocolError):
                        self.driver._verify_sources(package, deferred=deferred)

    def test_gpu_second_query_failure_keeps_first_output(self):
        primary = OSError("second GPU query")
        output = "0, GPU-synthetic, NVIDIA A100, 80000, 0\n"
        with mock.patch.object(subprocess, "run", side_effect=[
            types.SimpleNamespace(stdout=output), primary,
        ]):
            with self.assertRaises(OSError) as raised:
                self.driver._gpu_outputs(self.package()["runtime_manifest"])
        self.assertIs(raised.exception, primary)
        self.assertEqual(getattr(primary, "gpu_outputs", None), {"gpu": output})

    def test_partial_preparation_preserves_phase_evidence_and_durations(self):
        package = self.package()
        manifest = package["runtime_manifest"]
        facts = {name: manifest[name] for name in (
            "hostname", "uid", "python", "executable", "executable_sha256", "versions")}
        facts.update(isolated=True, no_site=True, dont_write_bytecode=True)
        for phase in ("gpu", "checkpoint"):
            with self.subTest(phase=phase):
                current = copy.deepcopy(package)
                primary = OSError("GPU query failed")
                primary.gpu_outputs = {"gpu": "partial GPU stdout"}
                if phase == "checkpoint":
                    current["runtime_manifest"]["checkpoint_files"]["weights"]["sha256"] = "0" * 64
                root = self.case("partial-" + phase)
                state = self.driver._new_state(self.fixture.run_id, time.monotonic_ns() + 10 ** 12)
                state.update(root=root, emit=self.frames.append)
                with mock.patch.object(self.driver, "_verify_sources", return_value={}), \
                     mock.patch.object(self.driver, "_runtime_facts", return_value=facts), \
                     mock.patch.object(self.driver, "_gpu_outputs", side_effect=primary if phase == "gpu" else None,
                                       return_value={"gpu": "0, GPU-synthetic, A100, 80000, 0", "apps": ""}), \
                     mock.patch.object(self.driver, "_load_runtime") as load, mock.patch.dict(os.environ):
                    with self.assertRaises(BaseException):
                        self.driver._prepare_base(current, root, state)
                    load.assert_not_called()
                self.assertIsNotNone(state["costs"][phase + "_ns"])
                path = root / ("gpu-before.json" if phase == "gpu" else "checkpoint-observation.json")
                self.assertTrue(path.exists())
                record = parse_json(path.read_bytes())
                self.assertIn("gpu" if phase == "gpu" else "weights", record)

    def test_degraded_coordinator_envelope_is_preserved_without_reduction(self):
        error = self.driver.DevelopmentRunError.__new__(self.driver.DevelopmentRunError)
        for name, value in {
            "report": {"counts": None, "costs": None, "by_arm": None, "episodes": [{"position": 0}]},
            "terminal_path": "partial-report.json", "terminal_sha256": "a" * 64,
            "record_cleanup": {"failed_count": 0, "failures": []},
            "record_cleanup_complete": True, "persistence_errors": [],
        }.items():
            setattr(error, name, value)
        envelope = self.driver._coordinator_envelope(error)
        self.assertEqual(envelope["report"], error.report)
        self.assertIsNot(envelope["report"], error.report)

    def fake_runtime(self):
        parameter = types.SimpleNamespace(dtype="bf16", device=types.SimpleNamespace(type="cuda"),
                                          requires_grad=False, grad=None)
        model = mock.Mock()
        model.parameters.return_value = [parameter]
        model.modules.return_value = [types.SimpleNamespace(training=False)]
        model.model.layers = [None] * 36
        model.config = types.SimpleNamespace(use_cache=False, _attn_implementation="eager")
        model.to.return_value = model
        model.eval.return_value = model
        model.requires_grad_.return_value = model
        rng = mock.Mock()
        rng.clone.return_value = rng
        torch = mock.Mock()
        torch.bfloat16 = "bf16"
        torch.random.get_rng_state.return_value = rng
        torch.cuda.get_rng_state.return_value = rng
        torch.equal.return_value = True
        torch.cuda.max_memory_allocated.return_value = 10
        torch.cuda.max_memory_reserved.return_value = 20
        transformers = mock.Mock()
        transformers.AutoTokenizer.from_pretrained.return_value = self.tokenizer
        transformers.AutoModelForCausalLM.from_pretrained.return_value = model
        return torch, transformers, model, parameter

    def test_integrity_state_gradient_tokenizer_and_rng_drift(self):
        manifest = self.package()["runtime_manifest"]
        torch, unused, model, parameter = self.fake_runtime()
        state = self.driver._new_state(self.fixture.run_id, time.monotonic_ns() + 10 ** 12)
        state.update(_torch=torch, _manifest=manifest)
        with mock.patch.object(self.driver, "_state_hash", return_value=manifest["expected_base_sha256"]) as base:
            initial = self.driver._integrity(model, self.tokenizer, state, initial=True)
            self.assertTrue(initial["verified"])
            for kind in ("base", "gradient", "tokenizer", "rng", "layers", "dtype"):
                with self.subTest(kind=kind):
                    base.return_value = "0" * 64 if kind == "base" else manifest["expected_base_sha256"]
                    parameter.grad = object() if kind == "gradient" else None
                    parameter.dtype = "fp32" if kind == "dtype" else "bf16"
                    self.tokenizer.chat_template = "changed" if kind == "tokenizer" else "synthetic template"
                    torch.equal.return_value = kind != "rng"
                    model.model.layers = [None] * (35 if kind == "layers" else 36)
                    with self.assertRaises(self.driver.BoundaryProtocolError):
                        self.driver._final_integrity(model, self.tokenizer, state)

    def test_base_load_kwargs_and_partial_load_timings(self):
        package = self.package()
        manifest = package["runtime_manifest"]
        facts = {name: manifest[name] for name in (
            "hostname", "uid", "python", "executable", "executable_sha256", "versions")}
        facts.update(isolated=True, no_site=True, dont_write_bytecode=True)
        for failing in (None, "tokenizer", "model"):
            with self.subTest(failing=failing):
                torch, transformers, model, unused = self.fake_runtime()
                primary = OSError("loader failed")
                if failing is not None:
                    loader = transformers.AutoTokenizer if failing == "tokenizer" else transformers.AutoModelForCausalLM
                    loader.from_pretrained.side_effect = primary
                root = self.case("base-" + str(failing))
                state = self.driver._new_state(self.fixture.run_id, time.monotonic_ns() + 10 ** 12)
                state.update(root=root, emit=self.frames.append)
                with mock.patch.object(self.driver, "_verify_sources", return_value={}), \
                     mock.patch.object(self.driver, "_runtime_facts", return_value=facts), \
                     mock.patch.object(self.driver, "_gpu_outputs", return_value={
                         "gpu": "0, GPU-synthetic, NVIDIA A100, 80000, 0\n", "apps": ""}), \
                     mock.patch.object(self.driver, "_load_runtime", return_value=(torch, transformers)), \
                     mock.patch.object(self.driver, "_state_hash", return_value=manifest["expected_base_sha256"]), \
                     mock.patch.dict(os.environ):
                    if failing is None:
                        result = self.driver._prepare_base(package, root, state)
                        self.assertEqual(result, (model, self.tokenizer))
                    else:
                        with self.assertRaises(OSError) as raised:
                            self.driver._prepare_base(package, root, state)
                        self.assertIs(raised.exception, primary)
                        self.assertIsNotNone(state["costs"][failing + "_load_ns"])
                transformers.AutoTokenizer.from_pretrained.assert_called_once_with(
                    manifest["model_root"], local_files_only=True, trust_remote_code=False)
                if failing != "tokenizer":
                    transformers.AutoModelForCausalLM.from_pretrained.assert_called_once_with(
                        manifest["model_root"], local_files_only=True, trust_remote_code=False,
                        use_safetensors=True, torch_dtype="bf16", attn_implementation="eager")

    def test_decoded_package_contract_and_source_manifest_are_exact(self):
        package = self.package()
        snapshot = self.driver._validate_package(package)
        self.assertEqual(snapshot["modules"], package["modules"])
        snapshot["runtime_manifest"]["hostname"] = "changed"
        self.assertNotEqual(snapshot["runtime_manifest"], package["runtime_manifest"])
        for field in ("module_order", "limits", "runtime_manifest"):
            changed = copy.deepcopy(package)
            changed[field] = {}
            with self.subTest(field=field), self.assertRaises(self.driver.BoundaryProtocolError):
                self.driver._validate_package(changed)

    def test_runtime_gpu_and_checkpoint_mutations_fail_before_model_load(self):
        package = self.package()
        manifest = package["runtime_manifest"]
        facts = {name: manifest[name] for name in (
            "hostname", "uid", "python", "executable", "executable_sha256", "versions")}
        facts.update(isolated=True, no_site=True, dont_write_bytecode=True)
        for kind in ("runtime", "gpu", "checkpoint"):
            with self.subTest(kind=kind):
                current = copy.deepcopy(package)
                observed = copy.deepcopy(facts)
                gpu = {"gpu": "0, GPU-synthetic, NVIDIA A100, 80000, 0\n", "apps": ""}
                if kind == "runtime":
                    observed["uid"] = 7
                elif kind == "gpu":
                    gpu["apps"] = "GPU-synthetic, 999, 1024\n"
                else:
                    current["runtime_manifest"]["checkpoint_files"]["weights"]["sha256"] = "0" * 64
                root = self.case("runtime-case-" + kind)
                state = self.driver._new_state(self.fixture.run_id, time.monotonic_ns() + 10 ** 12)
                state.update(root=root, emit=self.frames.append)
                with mock.patch.object(self.driver, "_verify_sources"):
                    with mock.patch.object(self.driver, "_runtime_facts", return_value=observed):
                        with mock.patch.object(self.driver, "_gpu_outputs", return_value=gpu):
                            with mock.patch.object(self.driver, "_load_runtime") as load:
                                with self.assertRaises(self.driver.BoundaryProtocolError):
                                    self.driver._prepare_base(current, root, state)
                                load.assert_not_called()


class ReviewFixTests(DriverFixture):
    package = FlowTests.package
    fake_runtime = RuntimeTests.fake_runtime

    def test_f1_cancellation_stops_inventory_and_both_read_passes(self):
        for kind in (TimeoutError, KeyboardInterrupt):
            for read_pass in (1, 2, 3):
                with self.subTest(kind=kind.__name__, read_pass=read_pass):
                    root = self.case(kind.__name__ + str(read_pass))
                    (root / "a").write_bytes(b"a" * 65537)
                    (root / "b").write_bytes(b"later")
                    primary = kind("export cancelled")
                    original = self.driver._read_regular
                    reads = 0
                    cancelled = False
                    work_after = []
                    frames = []

                    def read(directory, relative):
                        nonlocal reads, cancelled
                        if cancelled:
                            work_after.append(("read", relative))
                        if relative == "a":
                            reads += 1
                            if reads == read_pass:
                                cancelled = True
                                raise primary
                        return original(directory, relative)

                    def emit(frame):
                        frames.append(frame)
                        if cancelled and frame["phase"] != "EXPORT_END":
                            work_after.append(("frame", frame["phase"]))

                    with mock.patch.object(self.driver, "_read_regular", side_effect=read):
                        with self.assertRaises(self.driver.ExportError) as raised:
                            self.driver.export_evidence(root, run_id=self.fixture.run_id, emit=emit)
                    self.assertIs(raised.exception.__cause__, primary)
                    self.assertEqual(work_after, [])
                    self.assertEqual(raised.exception.receipt["status"], "incomplete")
                    self.assertEqual(len(raised.exception.receipt["inventory"]), 0 if read_pass == 1 else 2)
                    self.assertEqual([frame["phase"] for frame in frames].count("EXPORT_END"), 1)

    def test_f1_directory_cancellation_stops_walk_and_keeps_partial_receipt(self):
        for kind in (TimeoutError, KeyboardInterrupt):
            with self.subTest(kind=kind.__name__):
                root = self.case(kind.__name__)
                (root / "a").write_bytes(b"retained")
                (root / "b").mkdir()
                (root / "c").write_bytes(b"not visited")
                primary = kind("directory cancellation")
                original = Path.iterdir
                original_read = self.driver._read_regular
                cancelled = False
                work_after = []

                def entries(path):
                    nonlocal cancelled
                    if cancelled:
                        work_after.append(str(path))
                    if path == root / "b":
                        cancelled = True
                        raise primary
                    return original(path)

                def read(directory, relative):
                    if cancelled:
                        work_after.append(relative)
                    return original_read(directory, relative)

                with mock.patch.object(Path, "iterdir", entries), \
                     mock.patch.object(self.driver, "_read_regular", side_effect=read):
                    try:
                        self.driver.export_evidence(root, run_id=self.fixture.run_id, emit=self.frames.append)
                    except BaseException as error:
                        self.assertIsInstance(error, self.driver.ExportError)
                        self.assertIs(error.__cause__, primary)
                        self.assertEqual([item["path"] for item in error.receipt["inventory"]], ["a"])
                    else:
                        self.fail("cancellation must abort export")
                self.assertEqual(work_after, [])

    def test_f1_chunk_cancellation_and_failed_terminal_do_not_retry(self):
        for kind in (TimeoutError, KeyboardInterrupt):
            with self.subTest(kind=kind.__name__):
                root = self.case(kind.__name__)
                (root / "a").write_bytes(b"a" * 65537)
                (root / "b").write_bytes(b"later")
                primary = kind("chunk cancellation")
                phases = []

                def emit(frame):
                    phases.append(frame["phase"])
                    if frame["phase"] == "FILE_CHUNK":
                        raise primary
                    if frame["phase"] == "EXPORT_END":
                        raise BrokenPipeError("terminal unavailable")

                with self.assertRaises(self.driver.ExportError) as raised:
                    self.driver.export_evidence(root, run_id=self.fixture.run_id, emit=emit)
                self.assertIs(raised.exception.__cause__, primary)
                self.assertEqual(phases, ["EXPORT_BEGIN", "FILE_CHUNK", "EXPORT_END"])
                self.assertEqual(raised.exception.receipt["exported_files"], 0)
                self.assertEqual(len(raised.exception.receipt["errors"]), 2)

    def test_f1_export_cancellation_preserves_existing_driver_primary(self):
        primary = RuntimeError("preparation failed")
        cancellation = TimeoutError("export deadline")
        with mock.patch.object(self.driver, "_read_regular", side_effect=cancellation) as read:
            with self.assertRaises(self.driver.DriverRunError) as raised:
                FlowTests.run_driver(self, prepare=mock.Mock(side_effect=primary))
        self.assertIs(raised.exception.__cause__, primary)
        self.assertEqual(read.call_count, 1)
        self.assertEqual(raised.exception.envelope["export"]["status"], "incomplete")
        self.assertEqual(raised.exception.envelope["export"]["errors"][0]["error"]["type"], "TimeoutError")
        self.assertTrue(any(item["phase"] == "export" and item["error"]["type"] == "TimeoutError"
                            for item in raised.exception.envelope["secondary_errors"]))
        self.assertEqual(self.frames[-1]["phase"], "EXPORT_INCOMPLETE")

    def test_f2_consumer_alias_drift_is_rejected_without_invoking_replacement(self):
        package = self.package()
        for name in self.driver.MODULE_ORDER:
            if name not in ("training_inputs", "training_worker"):
                importlib.import_module(name)
        slots = (
            ("development_remote_driver", "generate_observed", "observed_generation", "generate_observed"),
            ("development_remote_driver", "measure", "tokenizer_measurement", "measure"),
            ("development_remote_driver", "tokenizer_state", "tokenizer_measurement", "_state"),
            ("development_remote_driver", "_save", "observed_generation", "_save"),
            ("development_remote_driver", "candidate_execution_admission", "candidate_execution_admission", None),
            ("development_remote_driver", "development_final_admission", "development_final_admission", None),
            ("development_remote_driver", "coordinator", "development_coordinator", None),
            ("generation_core", "measure", "tokenizer_measurement", "measure"),
            ("observed_generation", "generation_core", "generation_core", None),
        )
        for consumer_name, alias, provider_name, definition in slots:
            with self.subTest(consumer=consumer_name, alias=alias):
                consumer = sys.modules[consumer_name]
                provider = sys.modules[provider_name]
                original = provider if definition is None else getattr(provider, definition)
                replacement = mock.Mock(side_effect=AssertionError("replacement must not execute"))
                with mock.patch.object(consumer, alias, replacement):
                    with self.assertRaises(self.driver.BoundaryProtocolError):
                        self.driver._verify_sources(package, deferred=("training_inputs", "training_worker"))
                    self.assertIs(provider if definition is None else getattr(provider, definition), original)
                replacement.assert_not_called()

    def test_f3_runtime_and_initial_integrity_failures_keep_elapsed_and_primary(self):
        package = self.package()
        manifest = package["runtime_manifest"]
        for phase in ("runtime", "initial_integrity"):
            with self.subTest(phase=phase):
                root = self.case(phase)
                state = self.driver._new_state(self.fixture.run_id, time.monotonic_ns() + 10 ** 12)
                state.update(root=root, emit=self.frames.append)
                facts = {name: manifest[name] for name in (
                    "hostname", "uid", "python", "executable", "executable_sha256", "versions")}
                facts.update(isolated=True, no_site=True, dont_write_bytecode=True)
                if phase == "runtime":
                    facts["uid"] += 1
                torch, transformers, model, unused = self.fake_runtime()
                failures = []
                original_same = self.driver.coordinator._same
                original_integrity = self.driver._integrity

                def capture(function, *arguments, **keywords):
                    try:
                        return function(*arguments, **keywords)
                    except self.driver.BoundaryProtocolError as error:
                        failures.append(error)
                        raise

                with mock.patch.object(self.driver, "_verify_sources", return_value={}), \
                     mock.patch.object(self.driver, "_runtime_facts", return_value=facts), \
                     mock.patch.object(self.driver, "_gpu_outputs", return_value={
                         "gpu": "0, GPU-synthetic, A100, 80000, 0", "apps": ""}), \
                     mock.patch.object(self.driver, "_load_runtime", return_value=(torch, transformers)), \
                     mock.patch.object(self.driver, "_state_hash", return_value="0" * 64), \
                     mock.patch.object(self.driver.coordinator, "_same",
                                       side_effect=lambda *args, **kwargs: capture(original_same, *args, **kwargs)), \
                     mock.patch.object(self.driver, "_integrity",
                                       side_effect=lambda *args, **kwargs: capture(original_integrity, *args, **kwargs)), \
                     mock.patch.dict(os.environ):
                    with self.assertRaises(self.driver.BoundaryProtocolError) as raised:
                        self.driver._prepare_base(package, root, state)
                self.assertIs(raised.exception, failures[-1])
                self.assertIsNotNone(state["costs"][phase + "_ns"])
                self.assertGreaterEqual(state["costs"][phase + "_ns"], 0)
                self.assertIsNone(state["integrity"]["initial"])
                phases = [item["phase"] for item in state["events"]]
                self.assertNotIn("MODEL_VERIFIED", phases)
                if phase == "runtime":
                    self.assertNotIn("RUNTIME_VERIFIED", phases)
                    self.assertIsNone(state["costs"]["initial_integrity_ns"])


class ColdImportTests(unittest.TestCase):
    def test_cold_import_is_stdlib_and_pure_dependencies_only(self):
        directory = str(Path(__file__).parent)
        script = f"""
import builtins, socket, subprocess, sys
sys.path.insert(0, {directory!r})
original = builtins.__import__
def guarded(name, *arguments, **keywords):
    if name.split(".")[0] in ("torch", "transformers", "training_worker"):
        raise AssertionError("forbidden cold import: " + name)
    return original(name, *arguments, **keywords)
def forbidden(*arguments, **keywords):
    raise AssertionError("forbidden cold effect")
builtins.__import__ = guarded
socket.socket = forbidden
subprocess.Popen = forbidden
import development_remote_driver
assert not any(name in sys.modules for name in ("torch", "transformers", "training_worker"))
print("COLD_IMPORT_OK")
"""
        result = subprocess.run([sys.executable, "-I", "-S", "-B", "-c", script],
                                capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout, "COLD_IMPORT_OK\n")


if __name__ == "__main__":
    unittest.main()
