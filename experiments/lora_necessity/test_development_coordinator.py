import copy
import hashlib
import importlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

from candidate_execution_admission import CandidateAdmissionError
import candidate_execution_admission
import candidate_execution_protocol
import development_final_admission
import development_final_protocol
import development_final_scoring
import candidate_feedback
import prompt_rendering
import observed_generation
from development_schedule import build_schedule
from pilot_protocol import FLAGS
import test_development_final_scoring as scoring_fixtures
from validation_contract import PINS, digest, encoded, parse_json


ORDER = [
    ("D01", "base"), ("D01", "text"), ("D02", "text"), ("D02", "base"),
    ("D03", "base"), ("D03", "text"), ("D04", "text"), ("D04", "base"),
]

COMPONENTS = (
    "development_coordinator.py", "development_schedule.py",
    "development_final_scoring.py", "development_final_protocol.py",
    "development_final_admission.py", "development_final_controller.py",
    "candidate_execution_admission.py", "candidate_execution_controller.py",
    "candidate_execution_protocol.py", "candidate_feedback.py", "prompt_rendering.py",
    "observed_generation.py", "generation_core.py", "tokenizer_measurement.py",
    "validation_contract.py", "pilot_protocol.py",
)


def bindings_fixture():
    execution = {
        "argv": ["synthetic-not-launched", "--control={control_fd}"],
        "startup_timeout": 10.0, "candidate_timeout": 10,
        "output_limit": 65536, "result_limit": 16384, "reap_timeout": 2.0,
    }
    return {
        "component_sha256": dict.fromkeys(COMPONENTS, "1" * 64),
        "runtime_sha256": "2" * 64, "model_sha256": "3" * 64,
        "tokenizer_state": {
            "template_sha256": "4" * 64, "vocabulary_sha256": "5" * 64,
            "eos_token_id": 1, "pad_token_id": 0,
        },
        "execution": {"public": copy.deepcopy(execution), "final": copy.deepcopy(execution)},
    }


def synthetic_views():
    tasks = scoring_fixtures.synthetic_tasks()
    for task in tasks:
        task["spec"] = "Synthetic identity task"
        task["initial_witness"] = {
            "public_case_id": "P01", "predicted_initial": {"kind": "return", "value": 99},
        }
        if task["split"] == "supervision":
            task["reference_source"] = "def toy(value):\n    return value\n"

    def public(task):
        return {
            **{name: task[name] for name in ("id", "entry_point", "initial_source", "spec")},
            "public_cases": [
                {name: case[name] for name in ("id", "args", "expected")}
                for case in task["public_cases"]
            ],
        }

    supervision = [
        {**public(task), "reference_source": task["reference_source"],
         "initial_witness": task["initial_witness"]}
        for task in tasks if task["split"] == "supervision"
    ]
    prompts = [public(task) for task in tasks if task["split"] != "supervision"]
    return (encoded(supervision), encoded(prompts), *scoring_fixtures.frozen_bytes(tasks))


class CoordinatorTests(unittest.TestCase):
    def setUp(self):
        self.assertTrue(Path(__file__).with_name("development_coordinator.py").is_file(),
                        "coordinator implementation missing")
        self.module = importlib.import_module("development_coordinator")
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.root = Path(self.folder.name) / "run"
        self.run_id = "a" * 32

    def test_fixed_ledger_retains_all_positions_and_unknown_denominators(self):
        state = self.module._new_state(self.run_id)
        self.assertIsInstance(state, dict)
        self.assertEqual(state["run_id"], self.run_id)
        self.assertIsNone(state["schedule"])
        self.assertIsNone(state["active"])
        self.assertEqual(state["record_cleanup"], {"failed_count": 0, "failures": []})
        self.assertEqual(state["persistence_errors"], [])
        self.assertEqual(state["protocol_errors"], [])
        self.assertIsNone(state["failure"])
        self.assertEqual(state["final_identity"], {})
        self.assertEqual([(row["task_id"], row["arm"]) for row in state["episodes"]],
                         ORDER)
        for ordinal, row in enumerate(state["episodes"]):
            self.assertEqual(row, {
                "ordinal": ordinal, "task_id": ORDER[ordinal][0], "arm": ORDER[ordinal][1],
                "scheduled_final_cases": None, "first_round_valid": None,
                "final_round_valid": None, "feedback": None, "feedback_opportunities": 0,
                "generations": [], "public_executions": [], "final_executions": [],
                "final_score": None, "completed": False,
            })

    def test_ledgers_do_not_share_mutable_containers(self):
        state = self.module._new_state(self.run_id)
        other = self.module._new_state(self.run_id)
        self.assertIsInstance(state, dict)
        self.assertIsInstance(other, dict)
        state["episodes"][0]["generations"].append({"prepared": True})
        state["record_cleanup"]["failures"].append({})
        state["final_identity"][0] = {}
        self.assertEqual(state["episodes"][1]["generations"], [])
        self.assertEqual(other["episodes"][0]["generations"], [])
        self.assertEqual(other["record_cleanup"]["failures"], [])
        self.assertEqual(other["final_identity"], {})

    def test_clone_is_an_independent_native_snapshot(self):
        value = {"messages": [{"role": "user", "content": "hello"}], "cost": None}
        result = self.module._clone(value)
        self.assertEqual(result, value)
        value["messages"][0]["content"] = "changed"
        self.assertEqual(result["messages"][0]["content"], "hello")

    def test_clone_rejects_non_native_and_nonfinite_values(self):
        class Mapping(dict):
            pass

        for value in ((1,), Mapping(), {1: "key"}, object(),
                      float("nan"), float("inf")):
            with self.subTest(kind=type(value).__name__), self.assertRaises(
                (CandidateAdmissionError, ValueError)
            ):
                self.module._clone(value)

    def test_typed_error_preserves_failure_envelope_and_cause(self):
        envelope = {
            "report": {"status": "aborted"}, "terminal_path": None,
            "terminal_sha256": None, "record_cleanup": {"failed_count": 0, "failures": []},
            "record_cleanup_complete": True, "persistence_errors": [],
        }
        primary = OSError("first failure")
        with self.assertRaises(self.module.DevelopmentRunError) as raised:
            raise self.module.DevelopmentRunError(envelope) from primary
        self.assertIs(raised.exception.__cause__, primary)
        for name, value in envelope.items():
            self.assertTrue(hasattr(raised.exception, name), name)
            self.assertEqual(getattr(raised.exception, name), value)
        self.assertIn("preserved evidence", str(raised.exception))
        self.assertTrue(issubclass(self.module.BoundaryProtocolError, ValueError))

    def test_checked_wraps_contract_errors_and_preserves_typed_identity(self):
        for primary in (ValueError("value"), KeyError("key"), TypeError("type"),
                        RecursionError("depth")):
            with self.subTest(kind=type(primary).__name__):
                with self.assertRaises(self.module.BoundaryProtocolError) as raised:
                    self.module._checked(mock.Mock(side_effect=primary))
                self.assertIs(raised.exception.__cause__, primary)
        for primary in (self.module.BoundaryProtocolError("protocol"),
                        OSError("disk"), KeyboardInterrupt()):
            with self.subTest(kind=type(primary).__name__):
                with self.assertRaises(type(primary)) as raised:
                    self.module._checked(mock.Mock(side_effect=primary))
                self.assertIs(raised.exception, primary)
        self.assertEqual(self.module._checked(lambda value, *, increment: value + increment,
                                             2, increment=3), 5)

    def test_need_rejects_false_conditions(self):
        with self.assertRaisesRegex(self.module.BoundaryProtocolError, "identity"):
            self.module._need(False, "identity")
        self.assertIsNone(self.module._need(True, "identity"))

    def test_record_publication_is_exclusive_and_native(self):
        state = self.module._new_state(self.run_id)
        self.root.mkdir(mode=0o700)
        value = {"messages": [{"content": "original"}]}
        self.module._publish(self.root, "record.json", value, state)
        self.assertTrue((self.root / "record.json").is_file())
        saved = (self.root / "record.json").read_bytes()
        self.assertEqual(saved, encoded(value))
        self.assertEqual((self.root / "record.json").stat().st_mode & 0o777, 0o600)
        value["messages"][0]["content"] = "mutated"
        with self.assertRaises(FileExistsError):
            self.module._publish(self.root, "record.json", value, state)
        self.assertEqual((self.root / "record.json").read_bytes(), saved)
        self.assertEqual(list(self.root.iterdir()), [self.root / "record.json"])
        with self.assertRaises(CandidateAdmissionError):
            self.module._publish(self.root, "invalid.json", {"tuple": ()}, state)
        self.assertFalse((self.root / "invalid.json").exists())

    def test_record_publish_unlink_failure_is_postcommit_diagnostic(self):
        state = self.module._new_state(self.run_id)
        self.root.mkdir(mode=0o700)
        with mock.patch("observed_generation.os.unlink", side_effect=OSError("cleanup")):
            self.module._publish(self.root, "terminal.json", {"status": "completed"}, state)
        self.assertTrue((self.root / "terminal.json").is_file())
        self.assertEqual(parse_json((self.root / "terminal.json").read_bytes()),
                         {"status": "completed"})
        self.assertEqual(state["record_cleanup"]["failed_count"], 1)
        self.assertTrue(state["record_cleanup"]["failures"][0]["published"])
        self.assertEqual(state["persistence_errors"], [])

    def test_record_publish_preserves_link_failure_when_unlink_also_fails(self):
        state = self.module._new_state(self.run_id)
        self.root.mkdir(mode=0o700)
        primary = OSError("primary link failure")
        with mock.patch("observed_generation.os.link", side_effect=primary), \
                mock.patch("observed_generation.os.unlink", side_effect=OSError("cleanup")):
            with self.assertRaises(OSError) as raised:
                self.module._publish(self.root, "terminal.json", {}, state)
        self.assertIs(raised.exception, primary)
        self.assertFalse((self.root / "terminal.json").exists())
        self.assertFalse(state["record_cleanup"]["failures"][0]["published"])

    def test_record_publish_preserves_write_failure_when_unlink_also_fails(self):
        state = self.module._new_state(self.run_id)
        self.root.mkdir(mode=0o700)
        primary = OSError("primary write failure")
        original_write = observed_generation._write

        def fail_write(descriptor, record):
            original_write(descriptor, record)
            raise primary

        with mock.patch("observed_generation._write", side_effect=fail_write), \
                mock.patch("observed_generation.os.unlink", side_effect=OSError("cleanup")):
            with self.assertRaises(OSError) as raised:
                self.module._publish(self.root, "terminal.json", {}, state)
        self.assertIs(raised.exception, primary)
        self.assertFalse((self.root / "terminal.json").exists())
        self.assertFalse(state["record_cleanup"]["failures"][0]["published"])

    def test_persistence_errors_are_separate_and_use_observer_formatter(self):
        state = self.module._new_state(self.run_id)
        primary = OSError("initial error")
        state["failure"] = observed_generation._failure(primary)
        self.module._save_error(state, "partial-report.json", OSError("disk full"))
        self.assertEqual(state["persistence_errors"], [{
            "path": "partial-report.json",
            "error": {"type": "OSError", "message": "disk full"},
        }])
        self.assertEqual(state["failure"], observed_generation._failure(primary))
        self.assertEqual(state["record_cleanup"], {"failed_count": 0, "failures": []})

    def test_preflight_snapshots_declarations_without_entering_callbacks(self):
        bindings = bindings_fixture()
        callback = mock.Mock(side_effect=AssertionError("callback must not enter"))
        result = self.module._preflight(self.run_id, (b"{}",) * 4,
                                        bindings, (callback,) * 3)
        self.assertEqual(result, bindings)
        bindings["execution"]["public"]["argv"].append("changed")
        self.assertEqual(len(result["execution"]["public"]["argv"]), 2)
        self.assertEqual(result["component_sha256"]["development_schedule.py"], "1" * 64)
        callback.assert_not_called()
        self.assertFalse(self.root.exists())

    def test_invalid_run_identity_is_never_serialized_in_ledger(self):
        class Untrusted:
            def __str__(self):
                raise AssertionError("must not stringify run identity")

        class String(str):
            pass

        for run_id in (None, 1, "A" * 32, "a" * 31, "../escape",
                       String("a" * 32), Untrusted()):
            with self.subTest(kind=type(run_id).__name__):
                state = self.module._new_state(run_id)
                self.assertIsNone(state["run_id"])
                self.assertEqual(len(parse_json(encoded(state))["episodes"]), 8)
                with self.assertRaises(self.module.BoundaryProtocolError):
                    self.module._preflight(run_id, (b"{}",) * 4, bindings_fixture(),
                                            (lambda: None,) * 3)

    def test_preflight_rejects_nonbytes_and_bad_callback_arity(self):
        class Bytes(bytes):
            pass

        callback = mock.Mock()
        for raw in ((b"{}",) * 3, [b"{}"] * 4,
                    (bytearray(b"{}"), b"{}", b"{}", b"{}"),
                    (b"{}", b"{}", "{}", b"{}"),
                    (b"{}", b"{}", b"{}", Bytes(b"{}"))):
            with self.subTest(raw_type=type(raw).__name__), \
                    self.assertRaises(self.module.BoundaryProtocolError):
                self.module._preflight(self.run_id, raw, bindings_fixture(), (callback,) * 3)
        for callbacks in ((callback,) * 2, (callback, None, callback),
                          (callback, callback, "not callable")):
            with self.subTest(callbacks=callbacks), \
                    self.assertRaises(self.module.BoundaryProtocolError):
                self.module._preflight(self.run_id, (b"{}",) * 4,
                                        bindings_fixture(), callbacks)
        callback.assert_not_called()

    def test_preflight_rejects_inexact_binding_shapes_and_native_types(self):
        class Mapping(dict):
            pass

        mutations = [
            lambda value: value.update(extra=True),
            lambda value: value.pop("runtime_sha256"),
            lambda value: value.update(runtime_sha256="A" * 64),
            lambda value: value.update(model_sha256="short"),
            lambda value: value.update(component_sha256=Mapping(value["component_sha256"])),
            lambda value: value["component_sha256"].pop("development_schedule.py"),
            lambda value: value["component_sha256"].update(extra="0" * 64),
            lambda value: value["component_sha256"].update(**{"pilot_protocol.py": 12}),
            lambda value: value["tokenizer_state"].update(template_sha256="X" * 64),
            lambda value: value["tokenizer_state"].update(vocabulary_sha256=None),
            lambda value: value["tokenizer_state"].update(eos_token_id=True),
            lambda value: value["tokenizer_state"].update(pad_token_id=-1),
            lambda value: value["tokenizer_state"].update(extra=0),
            lambda value: value["execution"].pop("final"),
            lambda value: value["execution"].update(other={}),
            lambda value: value["execution"]["final"].update(extra=True),
        ]
        callback = mock.Mock()
        for ordinal, mutate in enumerate(mutations):
            bindings = bindings_fixture()
            mutate(bindings)
            with self.subTest(mutation=ordinal), \
                    self.assertRaises(self.module.BoundaryProtocolError):
                self.module._preflight(self.run_id, (b"{}",) * 4, bindings, (callback,) * 3)
        callback.assert_not_called()

    def test_preflight_enforces_controller_deadlines_byte_limits_and_argv(self):
        invalid = {
            "startup_timeout": [False, 0, -1, float("inf"), float("nan"), "10"],
            "candidate_timeout": [False, 0, -1, float("inf"), float("nan"), "10"],
            "reap_timeout": [False, 0, -1, float("inf"), float("nan"), "10"],
            "output_limit": [True, 0, 65537, 1.0],
            "result_limit": [True, 0, 16385, 1.0],
            "argv": [[], ("{control_fd}",), ["none"], [1, "{control_fd}"],
                     ["\0", "{control_fd}"], ["{control_fd}{control_fd}"],
                     ["{control_fd}", "{control_fd}"]],
        }
        for phase in ("public", "final"):
            for name, values in invalid.items():
                for value in values:
                    bindings = bindings_fixture()
                    bindings["execution"][phase][name] = value
                    with self.subTest(phase=phase, name=name, value=value), \
                            self.assertRaises(self.module.BoundaryProtocolError):
                        self.module._preflight(self.run_id, (b"{}",) * 4,
                                                bindings, (lambda: None,) * 3)

    def test_preflight_normalizes_deadline_overflow_with_original_cause(self):
        original_isfinite = self.module.math.isfinite
        for phase in ("public", "final"):
            for name in ("startup_timeout", "candidate_timeout", "reap_timeout"):
                with self.subTest(phase=phase, deadline=name):
                    bindings = bindings_fixture()
                    bindings["execution"][phase][name] = 10**400
                    overflows = []
                    callback = mock.Mock()

                    def observe_isfinite(value):
                        try:
                            return original_isfinite(value)
                        except OverflowError as error:
                            overflows.append(error)
                            raise

                    with mock.patch.object(self.module.math, "isfinite",
                                           side_effect=observe_isfinite):
                        with self.assertRaises((self.module.BoundaryProtocolError,
                                                OverflowError)) as raised:
                            self.module._preflight(self.run_id, (b"{}",) * 4,
                                                   bindings, (callback,) * 3)
                    callback.assert_not_called()
                    self.assertEqual(list(Path(self.folder.name).iterdir()), [])
                    self.assertEqual(len(overflows), 1)
                    self.assertIsInstance(raised.exception, self.module.BoundaryProtocolError)
                    self.assertIs(raised.exception.__cause__, overflows[0])

    def schedule_inputs(self):
        raw = synthetic_views()
        original_pins = dict(PINS)
        self.addCleanup(self.assertEqual, PINS, original_pins)
        patch = mock.patch.dict(PINS, {
            name: hashlib.sha256(value).hexdigest()
            for name, value in zip((
                "supervision_v1.json", "prompts_v1.json",
                "task_sources_v1.json", "manifest_v1.json",
            ), raw)
        }, clear=True)
        patch.start()
        self.addCleanup(patch.stop)
        return raw

    def test_claim_uses_real_schedule_and_commits_all_denominators_before_publication(self):
        raw = self.schedule_inputs()
        state = self.module._new_state(self.run_id)
        bindings = bindings_fixture()
        expected = build_schedule(*raw, run_id=self.run_id)
        self.root.mkdir()
        published = []
        publish = self.module._publish

        def observe_publication(root, name, value, current):
            self.assertEqual([row["scheduled_final_cases"] for row in current["episodes"]],
                             [3] * 8)
            if name == "schedule.json":
                self.assertTrue((root / "claim.json").is_file())
            publish(root, name, value, current)
            published.append(name)

        with mock.patch.object(self.module, "_publish", side_effect=observe_publication):
            result = self.module._claim(self.root, state, raw, bindings)
        self.assertEqual(published, ["claim.json", "schedule.json"])
        self.assertEqual(result, expected)
        self.assertEqual(state["schedule"], expected)
        self.assertEqual(parse_json((self.root / "schedule.json").read_bytes()), expected)
        self.assertEqual(parse_json((self.root / "claim.json").read_bytes()), {
            "kind": "LORA_DEVELOPMENT_CLAIM", "version": 1, "run_id": self.run_id,
            "schedule_sha256": digest(expected), "data_sha256": dict(PINS),
            "bindings": bindings, "bindings_authenticity": "CALLER_DECLARED",
            "scheduled_episodes": 8, "scheduled_generation_calls": 16,
            "input_token_limit": 4096, "output_token_limit": 256,
            "episode_order": [{"task_id": task_id, "arm": arm} for task_id, arm in ORDER],
            "calibration_precedence": [
                "INVALID_PROTOCOL", "INCONCLUSIVE_INCOMPLETE",
                "INCONCLUSIVE_CALIBRATION", "DEVELOPMENT_CHECKS_SATISFIED",
            ],
        })
        self.assertEqual(state["final_identity"], {})
        for row, job in zip(state["episodes"], expected["episodes"]):
            self.assertEqual(row["generations"], [])
            self.assertEqual(row["scheduled_final_cases"], job["scheduled_final_cases"])
        self.assertNotIn(b"REFERENCE_MUST_NOT_LEAK", (self.root / "schedule.json").read_bytes())

    def test_claim_rejects_each_raw_hash_drift_before_any_publication(self):
        raw = self.schedule_inputs()
        self.root.mkdir()
        for ordinal in range(4):
            changed = list(raw)
            changed[ordinal] += b"\n"
            state = self.module._new_state(self.run_id)
            with self.subTest(ordinal=ordinal), \
                    self.assertRaises(self.module.BoundaryProtocolError):
                self.module._claim(self.root, state, tuple(changed), bindings_fixture())
            self.assertEqual(list(self.root.iterdir()), [])
            self.assertIsNone(state["schedule"])
            self.assertTrue(all(row["scheduled_final_cases"] is None
                                for row in state["episodes"]))

    def test_claim_validates_complete_schedule_before_publishing_claim(self):
        raw = self.schedule_inputs()
        expected = build_schedule(*raw, run_id=self.run_id)
        mutations = [
            lambda value: value.update(extra=True),
            lambda value: value.update(kind="other"),
            lambda value: value.update(version=True),
            lambda value: value.update(run_id="b" * 32),
            lambda value: value.update(scheduled_episodes=7),
            lambda value: value.update(scheduled_generation_calls=17),
            lambda value: value["data_sha256"].update(**{"prompts_v1.json": "0" * 64}),
            lambda value: value["episodes"].pop(),
            lambda value: value["episodes"].reverse(),
            lambda value: value["episodes"][7].update(ordinal=0),
            lambda value: value["episodes"][7].update(task_id="D01"),
            lambda value: value["episodes"][7].update(arm="lora"),
            lambda value: value["episodes"][7].update(extra=True),
            lambda value: value["episodes"][7].update(messages_sha256="0" * 64),
            lambda value: value["episodes"][7].update(task_sha256="short"),
            lambda value: value["episodes"][7].update(examples_sha256="short"),
            lambda value: value["episodes"][7].update(first_call_id="bad"),
            lambda value: value["episodes"][7].update(
                public_attempt_id=value["episodes"][0]["first_call_id"]),
            lambda value: value["episodes"][7].update(scheduled_final_cases=True),
            lambda value: value["episodes"][7].update(scheduled_final_cases=2),
            lambda value: value["episodes"][7]["final_cases"][0].update(expected="PRIVATE"),
            lambda value: value["episodes"][7]["final_cases"][0].update(input_sha256="bad"),
            lambda value: value["episodes"][7]["final_cases"][0].update(case_group="private"),
            lambda value: value["episodes"][7]["final_cases"][0].update(case_id="P02"),
        ]
        self.root.mkdir()
        for ordinal, mutate in enumerate(mutations):
            changed = copy.deepcopy(expected)
            mutate(changed)
            state = self.module._new_state(self.run_id)
            with self.subTest(mutation=ordinal), \
                    mock.patch.object(self.module, "build_schedule", return_value=changed), \
                    self.assertRaises(self.module.BoundaryProtocolError):
                self.module._claim(self.root, state, raw, bindings_fixture())
            self.assertEqual(list(self.root.iterdir()), [])
            self.assertIsNone(state["schedule"])

    def test_claim_snapshots_schedule_and_bindings_without_regenerating_ids(self):
        raw = self.schedule_inputs()
        scheduled = build_schedule(*raw, run_id=self.run_id)
        state = self.module._new_state(self.run_id)
        bindings = bindings_fixture()
        self.root.mkdir()
        with mock.patch.object(self.module, "build_schedule", return_value=scheduled):
            result = self.module._claim(self.root, state, raw, bindings)
        self.assertEqual(result, scheduled)
        saved = (self.root / "claim.json").read_bytes()
        scheduled["episodes"][0]["messages"][0]["content"] = "mutated"
        bindings["model_sha256"] = "f" * 64
        self.assertNotEqual(state["schedule"]["episodes"][0]["messages"],
                            scheduled["episodes"][0]["messages"])
        self.assertEqual((self.root / "claim.json").read_bytes(), saved)

    def attempt_fixture(self):
        state = self.module._new_state(self.run_id)
        self.root.mkdir(mode=0o700)
        relative = "generation-1"
        row = {
            "phase": "generation", "relative_dir": relative, "round": 1,
            "call_id": "b" * 32, "messages_sha256": digest([{"content": "synthetic"}]),
            "prepared": False, "callback_entered": False, "callback_returned": False,
            "claim_verified": False, "observation_verified": False,
            "observation_status": None, "generation_verified": False,
            "receipt_verified": False, "process_started": None,
            "forward_started": None, "forward_completed": None,
            "input_tokens": None, "output_tokens": None, "generation_timings_ns": None,
            "observer_elapsed_ns": None, "execution_elapsed_seconds": None,
            "peak_allocated_bytes": None, "peak_reserved_bytes": None,
            "record_cleanup": None, "receipt": None, "files": {},
            "failure": None, "evidence_errors": [],
        }
        state["episodes"][0]["generations"].append(row)
        prepared = {
            "kind": "LORA_DEVELOPMENT_PREPARED", "version": 1, "run_id": self.run_id,
            "phase": "generation", "episode_ordinal": 0, "round": 1,
            "call_id": row["call_id"], "messages_sha256": row["messages_sha256"],
            "relative_dir": relative,
        }
        return state, relative, row, prepared

    def test_owned_root_is_new_private_directory_with_existing_parent(self):
        result = self.module._owned_root(self.root)
        self.assertEqual(result, self.root)
        self.assertTrue(self.root.is_dir())
        self.assertEqual(self.root.stat().st_mode & 0o777, 0o700)
        missing = Path(self.folder.name) / "absent" / "run"
        with self.assertRaises(FileNotFoundError):
            self.module._owned_root(missing)
        self.assertFalse(missing.parent.exists())

    def test_owned_root_never_adopts_existing_directory_file_or_symlink(self):
        self.root.mkdir()
        sentinel = self.root / "sentinel"
        sentinel.write_bytes(b"preserve")
        before = {path: (path.stat(), path.read_bytes()) for path in self.root.iterdir()}
        callback = mock.Mock()
        with self.assertRaises(FileExistsError):
            self.module._owned_root(self.root)
            callback()
        self.assertEqual({path: (path.stat(), path.read_bytes())
                          for path in self.root.iterdir()}, before)
        callback.assert_not_called()
        for target in (sentinel, self.root):
            link = Path(self.folder.name) / ("link-" + target.name)
            link.symlink_to(target)
            with self.subTest(target=target.name), self.assertRaises(FileExistsError):
                self.module._owned_root(link)
            self.assertTrue(link.is_symlink())
        with self.assertRaises(FileExistsError):
            self.module._owned_root(sentinel)
        self.assertEqual(sentinel.read_bytes(), b"preserve")

    def test_attempt_publishes_prepared_before_entry_then_inspects_and_associates(self):
        state, relative, row, prepared = self.attempt_fixture()
        events = []
        result_value = {"verified": True}

        def invoke(directory):
            events.append("invoke")
            self.assertEqual(parse_json((directory / "prepared.json").read_bytes()), prepared)
            self.assertTrue(row["prepared"])
            self.assertTrue(row["callback_entered"])
            self.assertFalse(row["callback_returned"])
            self.assertEqual(directory.stat().st_mode & 0o777, 0o700)
            return {"synthetic": True}

        def inspect(directory, returned, primary, current):
            events.append("inspect")
            self.assertEqual(directory, self.root / relative)
            self.assertEqual(returned, {"synthetic": True})
            self.assertIsNone(primary)
            self.assertIs(current, row)
            self.assertTrue(current["callback_returned"])
            return result_value

        result = self.module._attempt(self.root, relative, prepared, row, state, invoke, inspect)
        self.assertEqual(events, ["invoke", "inspect"])
        self.assertIs(result, result_value)
        self.assertEqual(state["active"], relative)
        self.assertEqual(parse_json((self.root / relative / "association.json").read_bytes()), row)
        self.assertFalse(row["claim_verified"])
        self.assertIsNone(row["forward_started"])
        self.assertIsNone(row["input_tokens"])

    def test_prepared_write_failure_keeps_missing_attempt_without_callback_or_inspection(self):
        state, relative, row, prepared = self.attempt_fixture()
        primary = OSError("prepared write failed")
        invoke = mock.Mock()
        inspect = mock.Mock()
        with mock.patch("observed_generation.os.link", side_effect=primary):
            with self.assertRaises(OSError) as raised:
                self.module._attempt(self.root, relative, prepared, row, state, invoke, inspect)
        self.assertIs(raised.exception, primary)
        invoke.assert_not_called()
        inspect.assert_not_called()
        self.assertIs(state["episodes"][0]["generations"][0], row)
        for name in ("prepared", "callback_entered", "callback_returned", "claim_verified",
                     "observation_verified", "generation_verified"):
            self.assertFalse(row[name])
        for name in ("forward_started", "forward_completed", "input_tokens", "output_tokens"):
            self.assertIsNone(row[name])
        self.assertEqual(len(state["episodes"]), 8)
        self.assertFalse((self.root / relative / "association.json").exists())

    def test_attempt_directory_collision_preserves_original_records_without_entry(self):
        state, relative, row, prepared = self.attempt_fixture()
        directory = self.root / relative
        directory.mkdir()
        (directory / "prepared.json").write_bytes(b"original")
        invoke = mock.Mock()
        inspect = mock.Mock()
        with self.assertRaises(FileExistsError):
            self.module._attempt(self.root, relative, prepared, row, state, invoke, inspect)
        self.assertEqual((directory / "prepared.json").read_bytes(), b"original")
        invoke.assert_not_called()
        inspect.assert_not_called()
        self.assertFalse(row["prepared"])
        self.assertFalse(row["callback_entered"])

    def test_attempt_inspects_saved_evidence_after_callback_baseexception(self):
        state, relative, row, prepared = self.attempt_fixture()
        primary = KeyboardInterrupt("first failure")
        entered = []

        def invoke(directory):
            entered.append(directory)
            (directory / "observation.json").write_bytes(encoded({"forward_started": 2}))
            raise primary

        def inspect(directory, returned, error, current):
            self.assertIsNone(returned)
            self.assertIs(error, primary)
            current["forward_started"] = parse_json(
                (directory / "observation.json").read_bytes())["forward_started"]
            current["observation_verified"] = True

        with self.assertRaises(KeyboardInterrupt) as raised:
            self.module._attempt(self.root, relative, prepared, row, state, invoke, inspect)
        self.assertIs(raised.exception, primary)
        self.assertEqual(entered, [self.root / relative])
        self.assertFalse(row["callback_returned"])
        self.assertEqual(row["forward_started"], 2)
        self.assertIsNone(row["input_tokens"])
        self.assertEqual(row["failure"], observed_generation._failure(primary))
        self.assertEqual(parse_json((self.root / relative / "association.json").read_bytes()), row)

    def test_attempt_preserves_first_error_and_separates_evidence_and_persistence_errors(self):
        state, relative, row, prepared = self.attempt_fixture()
        primary = RuntimeError("callback failed")
        evidence = self.module.BoundaryProtocolError("readable contradictory record")
        persistence = OSError("association failed")
        publish = self.module._publish

        def fail_association(directory, name, value, current):
            if name == "association.json":
                raise persistence
            return publish(directory, name, value, current)

        with mock.patch.object(self.module, "_publish", side_effect=fail_association):
            with self.assertRaises(RuntimeError) as raised:
                self.module._attempt(
                    self.root, relative, prepared, row, state,
                    mock.Mock(side_effect=primary), mock.Mock(side_effect=evidence))
        self.assertIs(raised.exception, primary)
        self.assertEqual(row["failure"], observed_generation._failure(primary))
        self.assertEqual(row["evidence_errors"], [observed_generation._failure(evidence)])
        self.assertEqual(state["protocol_errors"], [observed_generation._failure(evidence)])
        self.assertEqual(state["persistence_errors"], [{
            "path": relative + "/association.json",
            "error": observed_generation._failure(persistence),
        }])

    def test_attempt_raises_inspection_error_after_return_without_reclassifying_infrastructure(self):
        state, relative, row, prepared = self.attempt_fixture()
        primary = OSError("unreadable")
        with self.assertRaises(OSError) as raised:
            self.module._attempt(self.root, relative, prepared, row, state,
                                 lambda directory: {}, mock.Mock(side_effect=primary))
        self.assertIs(raised.exception, primary)
        self.assertTrue(row["callback_returned"])
        self.assertEqual(row["failure"], observed_generation._failure(primary))
        self.assertEqual(row["evidence_errors"], [observed_generation._failure(primary)])
        self.assertEqual(state["protocol_errors"], [])
        self.assertTrue((self.root / relative / "association.json").is_file())

    def test_attempt_association_failure_becomes_primary_only_without_earlier_failure(self):
        state, relative, row, prepared = self.attempt_fixture()
        primary = OSError("association link failed")
        original_link = observed_generation.os.link

        def fail_association(source, target):
            if target.name == "association.json":
                raise primary
            return original_link(source, target)

        invoke = mock.Mock(return_value={})
        inspect = mock.Mock(return_value={"verified": True})
        with mock.patch("observed_generation.os.link", side_effect=fail_association):
            with self.assertRaises(OSError) as raised:
                self.module._attempt(self.root, relative, prepared, row, state, invoke, inspect)
        self.assertIs(raised.exception, primary)
        invoke.assert_called_once()
        inspect.assert_called_once()
        self.assertTrue(row["callback_returned"])
        self.assertEqual(state["persistence_errors"], [{
            "path": relative + "/association.json",
            "error": observed_generation._failure(primary),
        }])

    def test_attempt_postcommit_cleanup_does_not_retry_or_turn_success_into_failure(self):
        state, relative, row, prepared = self.attempt_fixture()
        invoke = mock.Mock(return_value={})
        inspect = mock.Mock(return_value={"verified": True})
        with mock.patch("observed_generation.os.unlink", side_effect=OSError("cleanup")):
            result = self.module._attempt(
                self.root, relative, prepared, row, state, invoke, inspect)
        self.assertEqual(result, {"verified": True})
        invoke.assert_called_once()
        inspect.assert_called_once()
        self.assertEqual(state["record_cleanup"]["failed_count"], 2)
        self.assertTrue(all(item["published"] for item in state["record_cleanup"]["failures"]))
        self.assertEqual(state["persistence_errors"], [])
        self.assertIsNone(row["failure"])
        self.assertTrue((self.root / relative / "association.json").is_file())


def generation_records(messages, call_id, expected_sha256, candidate):
    generation = {
        "kind": "LORA_GREEDY_CANDIDATE_UNATTESTED", "device_type": "cpu",
        "tokenizer_state_sha256": expected_sha256, "messages_sha256": digest(messages),
        "prompt_sha256": "f" * 64, "input_tokens": 10, "output_tokens": 2,
        "output_ids": [1, 0], "candidate": candidate,
        "candidate_sha256": hashlib.sha256(candidate.encode()).hexdigest(),
        "stop_reason": "eos", "model_calls": 1, "forward_calls": 2,
        "timings_ns": {"preparation": 1, "generation": 2, "decoding": 1, "wall": 5},
        "peak_allocated_bytes": None, "peak_reserved_bytes": None,
        **dict.fromkeys(FLAGS, False),
    }
    observation = {
        "kind": "LORA_GENERATION_OBSERVATION", "call_id": call_id, "status": "completed",
        "messages_sha256": digest(messages), "forward_started": 2,
        "forward_completed": 2, "elapsed_ns": 6, "failure": None,
    }
    claim = {
        "kind": "LORA_GENERATION_CLAIM", "call_id": call_id,
        "messages_sha256": digest(messages), "expected_tokenizer_sha256": expected_sha256,
    }
    return claim, generation, observation


def save_json(directory, name, value, *, pretty=False):
    raw = json.dumps(value, sort_keys=True, indent=2, allow_nan=False).encode() \
        if pretty else encoded(value)
    with (directory / name).open("xb") as output:
        output.write(raw)


def altered(callback, mutate):
    def invoke(*arguments, **keywords):
        result = callback(*arguments, **keywords)
        mutate(result, keywords["artifact_dir"])
        return result
    return invoke


class RuntimeFixture(unittest.TestCase):
    def setUp(self):
        self.module = importlib.import_module("development_coordinator")
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.root = Path(self.folder.name) / "run"
        self.run_id = "a" * 32
        self.raw = synthetic_views()
        pins = {
            name: hashlib.sha256(raw).hexdigest()
            for name, raw in zip(("supervision_v1.json", "prompts_v1.json",
                                  "task_sources_v1.json", "manifest_v1.json"), self.raw)
        }
        patch = mock.patch.dict(PINS, pins, clear=True)
        patch.start()
        self.addCleanup(patch.stop)
        self.bindings = bindings_fixture()
        self.bindings["tokenizer_state"]["eos_token_id"] = 0
        self.calls = []
        self.public_calls = []
        self.final_calls = []
        self.events = []
        self.candidates = ["def toy(value):\n    return value\n"] * 16
        self.outcomes = {}

    def boundary_state(self):
        self.state = self.module._new_state(self.run_id)
        self.module._owned_root(self.root)
        self.schedule = self.module._claim(self.root, self.state, self.raw, self.bindings)
        self.job = self.schedule["episodes"][0]
        self.episode = self.state["episodes"][0]
        self.episode_dir = "episodes/00-D01-base"
        (self.root / "episodes").mkdir()
        (self.root / self.episode_dir).mkdir()

    def generate(self, messages, *, expected_tokenizer_sha256, call_id, artifact_dir):
        self.calls.append({"messages": copy.deepcopy(messages), "call_id": call_id,
                           "expected_tokenizer_sha256": expected_tokenizer_sha256,
                           "artifact_dir": artifact_dir})
        self.events.append(("generation", call_id))
        claim, generation, observation = generation_records(
            messages, call_id, expected_tokenizer_sha256, self.candidates[len(self.calls) - 1])
        for name, value in (("claim.json", claim), ("generation.json", generation),
                            ("observation.json", observation)):
            save_json(artifact_dir, name, value)
        return {"generation": generation, "observation": observation,
                "record_cleanup": {"failed_count": 0, "failures": []}}

    def execute(self, request, *, artifact_dir):
        final = request["round"] == 2
        phase = "final" if final else "public"
        calls = self.final_calls if final else self.public_calls
        calls.append(copy.deepcopy(request))
        self.events.append((phase, request["task_id"], request["arm"], request["case_id"]))
        protocol = development_final_protocol if final else candidate_execution_protocol
        admission_module = development_final_admission if final else candidate_execution_admission
        admission = admission_module.admit_request(request)
        request_sha256, payload = protocol.prepare_request(request)
        execution = self.bindings["execution"][phase]
        value = self.outcomes.get((phase, request["task_id"], request["arm"], request["case_id"]),
                                  {"P01": 1, "P02": 2, "H01": 3}[request["case_id"]])
        control = candidate_execution_protocol.READY + encoded(
            {"kind": "return", "value": value}) + b"\n"
        claim = {
            "request_sha256": request_sha256, "argv": execution["argv"],
            "payload_sha256": hashlib.sha256(payload).hexdigest(),
            **{name: execution[name] for name in
               ("startup_timeout", "candidate_timeout", "output_limit", "result_limit")},
        }
        audit = {
            "request_sha256": request_sha256, "ready": True, "input_complete": True,
            "input_bytes_sent": len(payload), "reaped": True, "eof": True,
            "returncode": 0, "termination": None, "elapsed_seconds": 0.01,
            "send_started_seconds": 0.001, "output_bytes_observed": len(control),
            "failure": None,
            "stream_sha256": {
                "stdout": hashlib.sha256(b"").hexdigest(),
                "stderr": hashlib.sha256(b"").hexdigest(),
                "control": hashlib.sha256(control).hexdigest(),
            },
        }
        receipt = candidate_execution_protocol.make_receipt(
            request_sha256, returncode=0, ready=True, input_complete=True,
            reaped=True, eof=True, termination=None,
            result_raw=control[len(candidate_execution_protocol.READY):])
        for name, record in (("admission.json", admission), ("claim.json", claim),
                             ("observation.json", audit), ("receipt.json", receipt)):
            save_json(artifact_dir, name, record, pretty=name != "admission.json")
        for name, raw in (("stdout.bin", b""), ("stderr.bin", b""), ("control.bin", control)):
            with (artifact_dir / name).open("xb") as output:
                output.write(raw)
        return {"admission": admission, "audit": audit, "receipt": receipt}

    def generation(self, callback=None, round_number=1):
        return self.module._generation(
            self.root, self.episode_dir, self.episode["generations"], self.job["messages"],
            round_number, self.job, self.bindings, callback or self.generate, self.state)

    def request(self, phase="public"):
        if phase == "public":
            return {
                "version": 1, "round": 1, "attempt_id": self.job["public_attempt_id"],
                "task_id": "D01", "arm": "base", "case_id": "P02", "entry_point": "toy",
                "source": self.candidates[0], "args": [2],
            }
        unused_claim, generation, unused_observation = generation_records(
            self.job["messages"], self.job["second_call_id"],
            digest(self.bindings["tokenizer_state"]), self.candidates[1])
        raw = encoded(generation)
        return development_final_scoring.build_final_plan(
            self.raw[2], self.raw[3], run_id=self.run_id, task_id="D01", arm="base",
            generation_call_id=self.job["second_call_id"], generation_raw=raw,
            expected_generation_sha256=hashlib.sha256(raw).hexdigest())["requests"][0]

    def execution(self, phase="public", callback=None, request=None):
        return self.module._execution(
            self.root, self.episode_dir, self.episode[phase + "_executions"],
            request or self.request(phase), phase, self.bindings,
            callback or self.execute, self.state)

    def new_boundary(self, ordinal):
        self.root = Path(self.folder.name) / f"boundary-{ordinal}"
        self.calls.clear()
        self.public_calls.clear()
        self.final_calls.clear()
        self.boundary_state()


class BoundaryTests(RuntimeFixture):
    def test_generation_verifies_saved_native_evidence_and_costs(self):
        self.boundary_state()
        raw = self.generation()
        self.assertIsInstance(raw, bytes)
        row = self.episode["generations"][0]
        self.assertEqual(raw, (self.root / row["relative_dir"] / "generation.json").read_bytes())
        self.assertTrue(row["claim_verified"])
        self.assertTrue(row["observation_verified"])
        self.assertTrue(row["generation_verified"])
        self.assertEqual(row["observation_status"], "completed")
        self.assertEqual(row["input_tokens"], 10)
        self.assertEqual(row["output_tokens"], 2)
        self.assertEqual(row["observer_elapsed_ns"], 6)
        self.assertEqual(row["forward_started"], 2)
        self.assertEqual(set(row["files"]), {"claim.json", "observation.json", "generation.json"})
        self.assertEqual(row["record_cleanup"], {"failed_count": 0, "failures": []})

    def test_generation_mutations_are_protocol_failures_with_all_file_hashes(self):
        mutations = [
            ("claim.json", lambda record: record.update(call_id="f" * 32)),
            ("claim.json", lambda record: record.update(expected_tokenizer_sha256="f" * 64)),
            ("observation.json", lambda record: record.update(messages_sha256="f" * 64)),
            ("observation.json", lambda record: record.update(forward_started=True)),
            ("observation.json", lambda record: record.update(forward_completed=3)),
            ("observation.json", lambda record: record.update(elapsed_ns=-1)),
            ("observation.json", lambda record: record.update(extra=1)),
            ("generation.json", lambda record: record.update(candidate_sha256="f" * 64)),
            ("generation.json", lambda record: record.update(tokenizer_state_sha256="f" * 64)),
            ("generation.json", lambda record: record.update(input_tokens=4097)),
            ("generation.json", lambda record: record.update(input_tokens=True)),
            ("generation.json", lambda record: record.update(output_ids=[0, 0])),
            ("generation.json", lambda record: record.update(stop_reason="length")),
            ("generation.json", lambda record: record.update(forward_calls=1)),
            ("generation.json", lambda record: record.update(isolated=True)),
            ("generation.json", lambda record: record.update(peak_allocated_bytes=0)),
            ("generation.json", lambda record: record["timings_ns"].update(wall=7)),
        ]
        for ordinal, (name, mutate) in enumerate(mutations):
            with self.subTest(record=name, mutation=ordinal):
                self.new_boundary(ordinal)

                def change(result, directory):
                    value = parse_json((directory / name).read_bytes())
                    mutate(value)
                    (directory / name).write_bytes(encoded(value))

                with self.assertRaises(self.module.BoundaryProtocolError):
                    self.generation(altered(self.generate, change))
                row = self.episode["generations"][0]
                self.assertEqual(len(self.calls), 1)
                self.assertEqual(len(row["files"]), 3)
                self.assertTrue(self.state["protocol_errors"])

    def test_generation_returned_disagreement_and_missing_success_record(self):
        for ordinal, mutation in enumerate((
            lambda result, directory: result["generation"].update(candidate="changed"),
            lambda result, directory: (directory / "generation.json").unlink(),
            lambda result, directory: result.update(extra=True),
            lambda result, directory: result["record_cleanup"].update(failed_count=True),
        )):
            with self.subTest(mutation=ordinal):
                self.new_boundary(ordinal)
                with self.assertRaises(self.module.BoundaryProtocolError):
                    self.generation(altered(self.generate, mutation))
                self.assertEqual(len(self.calls), 1)

    def test_failed_generation_retains_observation_without_inventing_tokens_or_protocol(self):
        self.boundary_state()
        primary = RuntimeError("generation failed")

        def failed(messages, *, expected_tokenizer_sha256, call_id, artifact_dir):
            claim, unused_generation, observation = generation_records(
                messages, call_id, expected_tokenizer_sha256, self.candidates[0])
            observation.update(status="failed", failure={"type": "RuntimeError", "message": "x"},
                               forward_started=3, forward_completed=2)
            save_json(artifact_dir, "claim.json", claim)
            save_json(artifact_dir, "observation.json", observation)
            raise primary

        with self.assertRaises(RuntimeError) as raised:
            self.generation(failed)
        self.assertIs(raised.exception, primary)
        row = self.episode["generations"][0]
        self.assertEqual(row["forward_started"], 3)
        self.assertEqual(row["forward_completed"], 2)
        self.assertIsNone(row["input_tokens"])
        self.assertFalse(row["generation_verified"])
        self.assertEqual(self.state["protocol_errors"], [])
        self.assertIsNone(row["record_cleanup"])

    def test_generation_saved_success_then_exception_keeps_verified_records(self):
        self.boundary_state()
        primary = RuntimeError("after records")

        def fail_after(result, directory):
            raise primary

        with self.assertRaises(RuntimeError) as raised:
            self.generation(altered(self.generate, fail_after))
        self.assertIs(raised.exception, primary)
        row = self.episode["generations"][0]
        self.assertTrue(row["generation_verified"])
        self.assertFalse(row["callback_returned"])
        self.assertEqual(row["input_tokens"], 10)
        self.assertEqual(self.state["protocol_errors"], [])

    def test_generation_never_enters_seventeenth_callback(self):
        self.boundary_state()
        self.episode["generations"].extend({"callback_entered": True} for ordinal in range(16))
        with self.assertRaises(self.module.BoundaryProtocolError):
            self.generation()
        self.assertEqual(self.calls, [])

    def test_execution_verifies_both_protocols_and_pretty_printed_saved_records(self):
        for ordinal, phase in enumerate(("public", "final")):
            with self.subTest(phase=phase):
                self.new_boundary(ordinal)
                receipt = self.execution(phase)
                self.assertIsInstance(receipt, dict)
                self.assertEqual(receipt["status"], "completed")
                row = self.episode[phase + "_executions"][0]
                self.assertTrue(row["receipt_verified"])
                self.assertTrue(row["process_started"])
                self.assertEqual(row["execution_elapsed_seconds"], 0.01)
                self.assertEqual(len(row["files"]), 7)
                saved = self.root / row["relative_dir"] / "observation.json"
                self.assertNotEqual(saved.read_bytes(), encoded(parse_json(saved.read_bytes())))
                self.assertEqual(row["files"]["observation.json"]["sha256"],
                                 hashlib.sha256(saved.read_bytes()).hexdigest())

    def test_execution_mutations_abort_without_retry_in_both_domains(self):
        mutations = [
            ("admission.json", lambda record: record.update(kind="other")),
            ("claim.json", lambda record: record.update(argv=["wrong"])),
            ("claim.json", lambda record: record.update(output_limit=1)),
            ("observation.json", lambda record: record.update(reaped=False)),
            ("observation.json", lambda record: record.update(input_complete=False)),
            ("observation.json", lambda record: record.update(returncode=True)),
            ("observation.json", lambda record: record.update(elapsed_seconds=-1)),
            ("observation.json", lambda record: record.update(output_bytes_observed=True)),
            ("observation.json", lambda record: record.update(extra=True)),
            ("observation.json", lambda record: record["stream_sha256"].update(control="f" * 64)),
            ("receipt.json", lambda record: record.update(request_sha256="f" * 64)),
        ]
        for domain, phase in enumerate(("public", "final")):
            for ordinal, (name, mutate) in enumerate(mutations):
                with self.subTest(phase=phase, mutation=ordinal):
                    self.new_boundary(domain * 100 + ordinal)

                    def change(result, directory):
                        record = parse_json((directory / name).read_bytes())
                        mutate(record)
                        (directory / name).write_bytes(encoded(record))

                    with self.assertRaises(self.module.BoundaryProtocolError):
                        self.execution(phase, altered(self.execute, change))
                    row = self.episode[phase + "_executions"][0]
                    self.assertEqual(len(row["files"]), 7)
                    self.assertEqual(len(self.public_calls) + len(self.final_calls), 1)

    def test_execution_missing_and_duplicate_frames_are_ordinary_program_errors(self):
        for ordinal, result_raw in enumerate((b"", b"{}\n{}\n", b"not json")):
            with self.subTest(frame=result_raw):
                self.new_boundary(ordinal)

                def change(result, directory):
                    control = candidate_execution_protocol.READY + result_raw
                    (directory / "control.bin").write_bytes(control)
                    result["audit"]["output_bytes_observed"] = len(control)
                    result["audit"]["stream_sha256"]["control"] = hashlib.sha256(control).hexdigest()
                    result["receipt"] = {"request_sha256": result["audit"]["request_sha256"],
                                         "status": "program_error"}
                    (directory / "observation.json").write_bytes(encoded(result["audit"]))
                    (directory / "receipt.json").write_bytes(encoded(result["receipt"]))

                receipt = self.execution(callback=altered(self.execute, change))
                self.assertIsInstance(receipt, dict)
                self.assertEqual(receipt["status"], "program_error")
                self.assertEqual(self.state["protocol_errors"], [])

    def test_local_policy_denial_is_typed_but_callback_admission_write_failure_is_not(self):
        self.boundary_state()
        request = self.request()
        request["source"] = "def toy(value):\n    return open('/tmp/x').read()\n"
        with self.assertRaises(self.module.BoundaryProtocolError) as raised:
            self.execution(request=request)
        self.assertIsInstance(raised.exception.__cause__, CandidateAdmissionError)
        self.assertEqual(self.public_calls, [])
        primary = CandidateAdmissionError("admission record could not be persisted")
        with self.assertRaises(CandidateAdmissionError) as raised:
            self.execution(callback=mock.Mock(side_effect=primary))
        self.assertIs(raised.exception, primary)
        self.assertEqual(self.state["protocol_errors"], [])

    def test_failed_execution_retains_valid_audit_and_unknown_missing_receipt(self):
        self.boundary_state()
        primary = RuntimeError("after audit")

        def fail(result, directory):
            (directory / "receipt.json").unlink()
            result["audit"]["failure"] = "infrastructure failure"
            (directory / "observation.json").write_bytes(encoded(result["audit"]))
            raise primary

        with self.assertRaises(RuntimeError) as raised:
            self.execution(callback=altered(self.execute, fail))
        self.assertIs(raised.exception, primary)
        row = self.episode["public_executions"][0]
        self.assertEqual(row["execution_elapsed_seconds"], 0.01)
        self.assertTrue(row["process_started"])
        self.assertFalse(row["receipt_verified"])
        self.assertEqual(self.state["protocol_errors"], [])


class FlowTests(RuntimeFixture):
    def run_development(self, **changes):
        parameters = {
            "run_id": self.run_id, "artifact_dir": self.root, "bindings": self.bindings,
            "generate": self.generate, "execute_public": self.execute, "execute_final": self.execute,
        }
        parameters.update(changes)
        result = self.module.run_development(*self.raw, **parameters)
        self.assertIsInstance(result, dict)
        return result

    def test_all_public_pass_still_generates_sixteen_times(self):
        result = self.run_development()
        report = result["report"]
        self.assertEqual(set(result), {
            "report", "terminal_path", "terminal_sha256", "record_cleanup",
            "record_cleanup_complete", "persistence_errors",
        })
        self.assertEqual(len(self.calls), 16)
        self.assertEqual(report["counts"]["completed_episodes"], 8)
        self.assertEqual(report["counts"]["completed_generation_calls"], 16)
        self.assertEqual(report["counts"]["public_verified_receipts"], 16)
        self.assertEqual(report["counts"]["final_verified_receipts"], 24)
        self.assertEqual(report["counts"]["missing_episode_positions"], [])
        self.assertEqual(report["counts"]["unexecuted_final_cases"], 0)
        self.assertEqual(report["costs"]["input_tokens"]["total"], 160)
        self.assertEqual(report["costs"]["output_tokens"]["total"], 32)
        self.assertEqual(report["calibration"]["status"], "INCONCLUSIVE_CALIBRATION")
        self.assertEqual(report["by_arm"]["base"]["passed"], 4)
        self.assertEqual(report["by_arm"]["text"]["passed"], 4)
        self.assertTrue(all(report[flag] is False for flag in FLAGS))
        self.assertEqual(result["terminal_sha256"], digest(report))
        self.assertEqual((self.root / result["terminal_path"]).read_bytes(), encoded(report))
        self.assertFalse((self.root / "failure.json").exists())
        self.assertFalse((self.root / "partial-report.json").exists())
        self.assertNotIn("final_identity", report)
        self.assertEqual(report["boundary_authenticity"], "NOT_ATTESTED")

    def test_exact_schedule_order_and_ids(self):
        self.run_development()
        schedule = build_schedule(*self.raw, run_id=self.run_id)
        self.assertEqual([call["call_id"] for call in self.calls], [
            job[name] for job in schedule["episodes"] for name in ("first_call_id", "second_call_id")
        ])
        expected = []
        for job in schedule["episodes"]:
            expected.append(("generation", job["first_call_id"]))
            expected.extend(("public", job["task_id"], job["arm"], case) for case in ("P02", "P01"))
            expected.append(("generation", job["second_call_id"]))
            expected.extend(("final", job["task_id"], job["arm"], case)
                            for case in ("P01", "P02", "H01"))
        self.assertEqual(self.events, expected)

    def test_second_messages_only_original_public_projection(self):
        result = self.run_development()
        for ordinal, episode in enumerate(result["report"]["episodes"]):
            expected = prompt_rendering.second_round(
                self.raw[0], self.raw[1], episode["task_id"], episode["arm"],
                self.candidates[ordinal * 2], episode["feedback"])
            self.assertEqual(encoded(self.calls[ordinal * 2 + 1]["messages"]), encoded(expected))
            messages = encoded(self.calls[ordinal * 2 + 1]["messages"])
            for sentinel in (b"REASON_MUST_NOT_LEAK", b"REFERENCE_MUST_NOT_LEAK",
                             b"expected_outcomes", b"final_score"):
                self.assertNotIn(sentinel, messages)

    def test_execution_callbacks_receive_no_expectations(self):
        self.run_development()
        public = {"version", "round", "attempt_id", "task_id", "arm", "case_id",
                  "entry_point", "source", "args"}
        final = public | {"kind", "run_id", "generation_call_id", "generation_sha256",
                          "case_set_sha256", "case_group"}
        for request in self.public_calls:
            self.assertEqual(set(request), public)
        for request in self.final_calls:
            self.assertEqual(set(request), final)
            self.assertNotIn("expected", request)

    def test_first_invalid_can_recover_without_erasing_warning(self):
        self.candidates[0] = "not valid python"
        result = self.run_development()
        episode = result["report"]["episodes"][0]
        self.assertIs(episode["first_round_valid"], False)
        self.assertEqual(episode["public_executions"], [])
        self.assertEqual(episode["feedback"], {"kind": "invalid_source"})
        self.assertEqual(episode["final_score"]["status"], "pass")
        self.assertEqual(len(self.calls), 16)
        self.assertIn("first_round_invalid_source",
                      result["report"]["calibration"]["reasons"]["INCONCLUSIVE_CALIBRATION"])

    def test_final_invalid_keeps_full_case_denominator(self):
        self.candidates[1] = "not valid python"
        report = self.run_development()["report"]
        episode = report["episodes"][0]
        self.assertEqual(episode["final_executions"], [])
        self.assertEqual(episode["final_score"]["scheduled_cases"], 3)
        self.assertEqual(episode["final_score"]["received_cases"], 0)
        self.assertEqual(episode["final_score"]["status"], "invalid_source")
        self.assertEqual(report["counts"]["scheduled_final_cases"], 24)
        self.assertEqual(report["counts"]["invalid_final_cases"], 3)
        self.assertEqual(report["counts"]["missing_final_cases"], 0)
        self.assertEqual(report["counts"]["unexecuted_final_cases"], 3)

    def test_final_failure_preserves_prefix(self):
        primary = RuntimeError("second final callback")
        entered = []

        def fail(request, *, artifact_dir):
            entered.append(request["case_id"])
            if len(entered) == 2:
                raise primary
            return self.execute(request, artifact_dir=artifact_dir)

        with self.assertRaises(self.module.DevelopmentRunError) as raised:
            self.run_development(execute_final=fail)
        self.assertIs(raised.exception.__cause__, primary)
        report = raised.exception.report
        self.assertEqual(report["episodes"][0]["final_score"]["received_cases"], 1)
        self.assertEqual(report["counts"]["missing_final_cases"], 23)
        self.assertEqual(report["counts"]["final_callbacks_entered"], 2)
        self.assertEqual(report["counts"]["unexecuted_final_cases"], 22)
        self.assertEqual(report["counts"]["final_unknown_child_starts"], 1)
        self.assertEqual(report["counts"]["missing_episode_positions"], list(range(8)))
        self.assertEqual(len(self.calls), 2)
        self.assertEqual(report["calibration"]["status"], "INCONCLUSIVE_INCOMPLETE")
        self.assertEqual(len(list((self.root / "episodes").iterdir())), 1)

    def test_public_failure_has_no_feedback_retry(self):
        primary = RuntimeError("second public")
        entered = []

        def fail(request, *, artifact_dir):
            entered.append(request["case_id"])
            if len(entered) == 2:
                raise primary
            return self.execute(request, artifact_dir=artifact_dir)

        with self.assertRaises(self.module.DevelopmentRunError) as raised:
            self.run_development(execute_public=fail)
        self.assertIs(raised.exception.__cause__, primary)
        report = raised.exception.report
        self.assertEqual(len(self.calls), 1)
        self.assertEqual(report["counts"]["public_verified_receipts"], 1)
        self.assertEqual(report["counts"]["public_feedback_opportunities"], 0)
        self.assertIsNone(report["episodes"][0]["feedback"])
        self.assertEqual(report["counts"]["unexecuted_final_cases"], 24)

    def test_unknown_callback_error_is_incomplete(self):
        primary = RuntimeError("ordinary infrastructure")
        with self.assertRaises(self.module.DevelopmentRunError) as raised:
            self.run_development(generate=mock.Mock(side_effect=primary))
        self.assertIs(raised.exception.__cause__, primary)
        report = raised.exception.report
        self.assertEqual(report["calibration"]["status"], "INCONCLUSIVE_INCOMPLETE")
        self.assertEqual(report["calibration"]["reasons"]["INVALID_PROTOCOL"], [])
        self.assertEqual(report["counts"]["generation_callbacks_entered"], 1)
        self.assertEqual(report["counts"]["verified_generation_claims"], 0)
        self.assertIsNone(report["costs"]["input_tokens"]["observed_subtotal"])
        self.assertIsNone(raised.exception.record_cleanup_complete)

    def test_confirmed_driver_violation_has_priority(self):
        primary = self.module.BoundaryProtocolError("driver violation")
        with self.assertRaises(self.module.DevelopmentRunError) as raised:
            self.run_development(generate=mock.Mock(side_effect=primary))
        self.assertIs(raised.exception.__cause__, primary)
        calibration = raised.exception.report["calibration"]
        self.assertEqual(calibration["status"], "INVALID_PROTOCOL")
        self.assertIn("run_aborted", calibration["reasons"]["INCONCLUSIVE_INCOMPLETE"])

    def test_source_hash_failure_never_enters_callback(self):
        self.raw = (self.raw[0] + b"\n", *self.raw[1:])
        with self.assertRaises(self.module.DevelopmentRunError) as raised:
            self.run_development()
        report = raised.exception.report
        self.assertEqual(report["counts"]["missing_episode_positions"], list(range(8)))
        self.assertIsNone(report["counts"]["scheduled_final_cases"])
        self.assertIsNone(report["counts"]["unexecuted_final_cases"])
        self.assertEqual(self.calls, [])
        self.assertEqual(report["calibration"]["status"], "INVALID_PROTOCOL")

    def test_preflight_and_existing_root_errors_never_write_unowned_artifacts(self):
        with self.assertRaises(self.module.DevelopmentRunError) as raised:
            self.run_development(run_id=object())
        self.assertIsNone(raised.exception.report["run_id"])
        self.assertIsNone(raised.exception.terminal_path)
        self.assertFalse(self.root.exists())
        self.root.mkdir()
        (self.root / "sentinel").write_bytes(b"original")
        with self.assertRaises(self.module.DevelopmentRunError) as raised:
            self.run_development()
        self.assertIsInstance(raised.exception.__cause__, FileExistsError)
        self.assertIsNone(raised.exception.terminal_path)
        self.assertEqual(list(self.root.iterdir()), [self.root / "sentinel"])
        self.assertEqual((self.root / "sentinel").read_bytes(), b"original")
        self.assertEqual(self.calls, [])

    def test_failure_save_errors_keep_original_cause(self):
        primary = RuntimeError("first callback")
        original_publish = self.module._publish

        def publish(root, name, value, state):
            if name in ("association.json", "failure.json", "partial-report.json"):
                raise OSError(name)
            return original_publish(root, name, value, state)

        with mock.patch.object(self.module, "_publish", side_effect=publish):
            with self.assertRaises(self.module.DevelopmentRunError) as raised:
                self.run_development(generate=mock.Mock(side_effect=primary))
        self.assertIs(raised.exception.__cause__, primary)
        self.assertEqual(len(raised.exception.persistence_errors), 3)
        self.assertIsNone(raised.exception.terminal_path)
        self.assertEqual(raised.exception.report["counts"]["generation_callbacks_entered"], 1)
        self.assertEqual(raised.exception.report["persistence_errors"],
                         raised.exception.persistence_errors)

    def test_success_terminal_unlink_failure_stays_committed_success(self):
        original_unlink = observed_generation.os.unlink

        def unlink(path, *arguments, **keywords):
            if Path(path).name.startswith(".development-report.json."):
                raise OSError("postcommit cleanup")
            return original_unlink(path, *arguments, **keywords)

        with mock.patch("observed_generation.os.unlink", side_effect=unlink):
            result = self.run_development()
        self.assertEqual(result["report"]["status"], "completed")
        self.assertEqual(result["record_cleanup"]["failed_count"], 1)
        self.assertFalse(result["record_cleanup_complete"])
        self.assertTrue(result["report"]["record_cleanup_complete_before_terminal"])
        self.assertEqual((self.root / "development-report.json").read_bytes(), encoded(result["report"]))
        self.assertFalse((self.root / "failure.json").exists())
        self.assertEqual(len(self.calls), 16)

    def test_success_terminal_link_failure_aborts_without_reentering_callbacks(self):
        original_link = observed_generation.os.link
        primary = OSError("terminal link")

        def link(source, target):
            if target.name == "development-report.json":
                raise primary
            return original_link(source, target)

        with mock.patch("observed_generation.os.link", side_effect=link):
            with self.assertRaises(self.module.DevelopmentRunError) as raised:
                self.run_development()
        self.assertIs(raised.exception.__cause__, primary)
        self.assertEqual(len(self.calls), 16)
        self.assertEqual(raised.exception.report["counts"]["completed_episodes"], 8)
        self.assertEqual(raised.exception.report["status"], "aborted")
        self.assertEqual(raised.exception.terminal_path, "partial-report.json")
        self.assertFalse((self.root / "development-report.json").exists())

    def test_nullable_cost_reducer_distinguishes_unknown_from_observed_zero(self):
        self.assertEqual(self.module._cost([], "value"), {
            "total": None, "observed_subtotal": None, "known_calls": 0, "scheduled_calls": 16,
        })
        self.assertEqual(self.module._cost([{"value": 0}, {"value": None}], "value"), {
            "total": None, "observed_subtotal": 0, "known_calls": 1, "scheduled_calls": 16,
        })
        self.assertEqual(self.module._cost([{"value": 2}] * 16, "value")["total"], 32)

    def test_partial_scoring_stops_at_first_unverified_receipt(self):
        self.boundary_state()
        unused_claim, generation, unused_observation = generation_records(
            self.job["messages"], self.job["second_call_id"],
            digest(self.bindings["tokenizer_state"]), self.candidates[1])
        raw = encoded(generation)
        identity = {
            "run_id": self.run_id, "task_id": "D01", "arm": "base",
            "generation_call_id": self.job["second_call_id"], "generation_raw": raw,
            "expected_generation_sha256": hashlib.sha256(raw).hexdigest(),
        }
        self.state["final_identity"][0] = identity
        plan = development_final_scoring.build_final_plan(self.raw[2], self.raw[3], **identity)
        self.episode["final_executions"] = [
            {"receipt_verified": ordinal != 1,
             "receipt": {"request_sha256": digest(request), "status": "completed",
                         "outcome": {"kind": "return", "value": ordinal + 1}}}
            for ordinal, request in enumerate(plan["requests"])
        ]
        self.module._partial_score(self.episode, self.state, self.raw[2], self.raw[3])
        self.assertIsInstance(self.episode["final_score"], dict)
        self.assertEqual(self.episode["final_score"]["received_cases"], 1)
        self.assertEqual([case["status"] for case in self.episode["final_score"]["cases"]],
                         ["pass", "missing", "missing"])


class MatrixTests(RuntimeFixture):
    run_development = FlowTests.run_development

    def failed_generation(self, primary, started, completed, *, claim=True, observation=True):
        def fail(messages, *, expected_tokenizer_sha256, call_id, artifact_dir):
            self.calls.append({"messages": copy.deepcopy(messages), "call_id": call_id})
            claim_record, unused_generation, observed = generation_records(
                messages, call_id, expected_tokenizer_sha256, self.candidates[0])
            observed.update(status="failed", failure={"type": "RuntimeError", "message": "synthetic"},
                            forward_started=started, forward_completed=completed)
            if claim:
                save_json(artifact_dir, "claim.json", claim_record)
            if observation:
                save_json(artifact_dir, "observation.json", observed)
            raise primary
        return fail

    def execution_status(self, status):
        def change(result, directory):
            audit = result["audit"]
            audit["returncode"] = 1 if status == "program_error" else -9
            audit["termination"] = None if status == "program_error" else status
            control = (directory / "control.bin").read_bytes()
            result["receipt"] = candidate_execution_protocol.make_receipt(
                audit["request_sha256"], returncode=audit["returncode"], ready=audit["ready"],
                input_complete=audit["input_complete"], reaped=audit["reaped"], eof=audit["eof"],
                termination=audit["termination"],
                result_raw=control[len(candidate_execution_protocol.READY):])
            (directory / "observation.json").write_bytes(encoded(audit))
            (directory / "receipt.json").write_bytes(encoded(result["receipt"]))
        return altered(self.execute, change)

    def test_input_4097_rejection_preserves_first_cost(self):
        primary = ValueError("input token limit exceeded: 4097")
        failed = self.failed_generation(primary, 0, 0)

        def generate(messages, **keywords):
            return self.generate(messages, **keywords) if not self.calls else failed(messages, **keywords)

        with self.assertRaises(self.module.DevelopmentRunError) as raised:
            self.run_development(generate=generate)
        self.assertIs(raised.exception.__cause__, primary)
        report = raised.exception.report
        self.assertEqual(len(self.calls), 2)
        self.assertEqual(report["counts"]["completed_generation_calls"], 1)
        self.assertEqual(report["costs"]["input_tokens"]["observed_subtotal"], 10)
        self.assertIsNone(report["costs"]["input_tokens"]["total"])
        second = report["episodes"][0]["generations"][1]
        self.assertIsNone(second["input_tokens"])
        self.assertEqual(second["forward_started"], 0)
        self.assertEqual(report["calibration"]["status"], "INCONCLUSIVE_INCOMPLETE")

    def test_generation_failure_after_forward_is_not_zero_cost(self):
        primary = RuntimeError("forward failure")
        with self.assertRaises(self.module.DevelopmentRunError) as raised:
            self.run_development(generate=self.failed_generation(primary, 3, 2))
        costs = raised.exception.report["costs"]
        self.assertEqual(costs["forward_started"]["observed_subtotal"], 3)
        self.assertEqual(costs["forward_completed"]["observed_subtotal"], 2)
        self.assertIsNone(costs["input_tokens"]["observed_subtotal"])
        self.assertIsNone(costs["output_tokens"]["total"])
        self.assertIs(raised.exception.__cause__, primary)

    def test_generation_exception_after_saved_success_still_aborts(self):
        primary = RuntimeError("after saved success")

        def fail(result, directory):
            raise primary

        with self.assertRaises(self.module.DevelopmentRunError) as raised:
            self.run_development(generate=altered(self.generate, fail))
        counts = raised.exception.report["counts"]
        self.assertEqual(counts["generation_callbacks_entered"], 1)
        self.assertEqual(counts["generation_callbacks_returned"], 0)
        self.assertEqual(counts["completed_generation_calls"], 1)
        self.assertEqual(counts["verified_generation_records"], 1)
        self.assertIs(raised.exception.__cause__, primary)
        self.assertEqual(len(self.calls), 1)

    def test_policy_denial_is_not_model_error(self):
        self.candidates[0] = "def toy(value):\n    return open('/tmp/x').read()\n"
        with self.assertRaises(self.module.DevelopmentRunError) as raised:
            self.run_development()
        self.assertIsInstance(raised.exception.__cause__, self.module.BoundaryProtocolError)
        report = raised.exception.report
        self.assertEqual(report["calibration"]["status"], "INVALID_PROTOCOL")
        self.assertEqual(report["counts"]["public_callbacks_entered"], 0)
        self.assertEqual(self.public_calls, [])
        row = report["episodes"][0]["public_executions"][0]
        self.assertEqual(report["failure"]["active"], row["relative_dir"])

    def test_callback_mutation_cannot_change_schedule(self):
        for ordinal, phase in enumerate(("generation", "public", "final")):
            with self.subTest(phase=phase):
                self.root = Path(self.folder.name) / f"mutation-{ordinal}"
                self.calls.clear()
                self.public_calls.clear()
                self.final_calls.clear()
                if phase == "generation":
                    def callback(messages, **keywords):
                        result = self.generate(messages, **keywords)
                        messages[0]["content"] = "mutated"
                        return result
                    changes = {"generate": callback}
                else:
                    def callback(request, **keywords):
                        result = self.execute(request, **keywords)
                        request["args"].append("mutated")
                        return result
                    changes = {"execute_" + phase: callback}
                with self.assertRaises(self.module.DevelopmentRunError) as raised:
                    self.run_development(**changes)
                self.assertEqual(raised.exception.report["calibration"]["status"], "INVALID_PROTOCOL")
                schedule = parse_json((self.root / "schedule.json").read_bytes())
                self.assertEqual(schedule, build_schedule(*self.raw, run_id=self.run_id))
                prepared = list((self.root / "episodes").rglob("prepared.json"))
                for path in prepared:
                    self.assertNotIn(b"mutated", path.read_bytes())

    def test_prepared_and_claim_are_not_completed_work(self):
        with self.assertRaises(self.module.DevelopmentRunError) as raised:
            self.run_development(generate=self.failed_generation(
                RuntimeError("after claim"), 0, 0, observation=False))
        report = raised.exception.report
        self.assertEqual(report["counts"]["generation_callbacks_entered"], 1)
        self.assertEqual(report["counts"]["verified_generation_claims"], 1)
        self.assertEqual(report["counts"]["completed_generation_calls"], 0)
        self.assertIsNone(report["costs"]["forward_started"]["observed_subtotal"])
        self.assertIsNone(report["costs"]["input_tokens"]["total"])

    def test_final_receipt_is_not_first_round_feedback(self):
        self.outcomes[("final", "D01", "base", "H01")] = 99
        report = self.run_development()["report"]
        episode = report["episodes"][0]
        self.assertEqual(episode["final_score"]["status"], "fail")
        self.assertTrue(all(case["status"] == "pass" for case in episode["feedback"]["cases"]))
        schedule = build_schedule(*self.raw, run_id=self.run_id)
        self.assertEqual(self.calls[2]["messages"], schedule["episodes"][1]["messages"])
        self.assertNotIn("final_score", encoded(self.calls[2]["messages"]).decode())

    def test_timeout_priority_keeps_invalid_warning(self):
        self.candidates[0] = "not valid python"
        report = self.run_development(execute_final=self.execution_status("timeout"))["report"]
        self.assertEqual(len(self.calls), 16)
        self.assertEqual(report["calibration"]["status"], "INCONCLUSIVE_INCOMPLETE")
        reasons = report["calibration"]["reasons"]
        self.assertIn("timeout", reasons["INCONCLUSIVE_INCOMPLETE"])
        self.assertIn("first_round_invalid_source", reasons["INCONCLUSIVE_CALIBRATION"])
        self.assertEqual(report["status"], "completed")

    def test_program_and_output_errors_are_not_infrastructure(self):
        for ordinal, status in enumerate(("program_error", "output_limit")):
            with self.subTest(status=status):
                self.root = Path(self.folder.name) / f"ordinary-{ordinal}"
                self.calls.clear()
                report = self.run_development(execute_final=self.execution_status(status))["report"]
                self.assertEqual(len(self.calls), 16)
                self.assertEqual(report["status"], "completed")
                self.assertEqual(report["calibration"]["status"], "INCONCLUSIVE_CALIBRATION")
                self.assertIn(status, report["calibration"]["reasons"]["INCONCLUSIVE_CALIBRATION"])

    def saturated_run(self, base, text, ordinal):
        self.root = Path(self.folder.name) / f"calibration-{ordinal}"
        self.calls.clear()
        self.outcomes.clear()
        for arm, passed in (("base", base), ("text", text)):
            for position in range(passed + 1, 5):
                self.outcomes[("final", f"D0{position}", arm, "H01")] = 99
        return self.run_development()["report"]

    def test_only_both_saturated_arms_trigger_calibration(self):
        for ordinal, (base, text) in enumerate(((4, 2), (0, 2), (2, 3))):
            with self.subTest(base=base, text=text):
                report = self.saturated_run(base, text, ordinal)
                self.assertEqual(report["by_arm"]["base"]["passed"], base)
                self.assertEqual(report["by_arm"]["text"]["passed"], text)
                self.assertEqual(report["calibration"]["status"], "DEVELOPMENT_CHECKS_SATISFIED")

    def test_both_zero_and_both_four_trigger_calibration(self):
        for ordinal, passed in enumerate((0, 4)):
            with self.subTest(passed=passed):
                report = self.saturated_run(passed, passed, ordinal)
                self.assertEqual(report["calibration"]["status"], "INCONCLUSIVE_CALIBRATION")

    def test_unreadable_records_are_io_failures_not_invented_protocol_violations(self):
        original_read = Path.read_bytes
        for ordinal, phase in enumerate(("generation", "execution")):
            with self.subTest(phase=phase):
                self.root = Path(self.folder.name) / f"unreadable-{ordinal}"
                self.calls.clear()
                primary = OSError("read failed")
                suffix = "generation-1/generation.json" if phase == "generation" \
                    else "public-00-P02/stdout.bin"

                def read(path):
                    if str(path).endswith(suffix):
                        raise primary
                    return original_read(path)

                with mock.patch.object(Path, "read_bytes", read):
                    with self.assertRaises(self.module.DevelopmentRunError) as raised:
                        self.run_development()
                self.assertIs(raised.exception.__cause__, primary)
                calibration = raised.exception.report["calibration"]
                self.assertEqual(calibration["reasons"]["INVALID_PROTOCOL"], [])
                self.assertEqual(calibration["status"], "INCONCLUSIVE_INCOMPLETE")

    def test_failed_callback_contradiction_preserves_first_cause_and_independent_cost(self):
        primary = RuntimeError("first callback failure")

        def fail(result, directory):
            claim = parse_json((directory / "claim.json").read_bytes())
            claim["call_id"] = "f" * 32
            (directory / "claim.json").write_bytes(encoded(claim))
            raise primary

        with self.assertRaises(self.module.DevelopmentRunError) as raised:
            self.run_development(generate=altered(self.generate, fail))
        self.assertIs(raised.exception.__cause__, primary)
        report = raised.exception.report
        self.assertEqual(report["calibration"]["status"], "INVALID_PROTOCOL")
        self.assertEqual(report["costs"]["input_tokens"]["observed_subtotal"], 10)
        self.assertFalse(report["episodes"][0]["generations"][0]["claim_verified"])

    def test_generation_cleanup_diagnostics_are_separate_from_completed_status(self):
        def cleanup(result, directory):
            result["record_cleanup"] = {
                "failed_count": 1, "failures": [{
                    "record": "generation.json", "temporary": ".synthetic",
                    "published": True, "error": {"type": "OSError", "message": "cleanup"},
                }],
            }

        result = self.run_development(generate=altered(self.generate, cleanup))
        self.assertEqual(result["report"]["status"], "completed")
        self.assertFalse(result["record_cleanup_complete"])
        self.assertEqual(result["record_cleanup"]["failed_count"], 0)
        self.assertEqual(len(self.calls), 16)

    def test_success_and_failure_terminal_link_errors_keep_first_cause(self):
        original_link = observed_generation.os.link
        primary = OSError("success terminal")

        def link(source, target):
            if target.name == "development-report.json":
                raise primary
            if target.name in ("failure.json", "partial-report.json"):
                raise OSError(target.name)
            return original_link(source, target)

        with mock.patch("observed_generation.os.link", side_effect=link):
            with self.assertRaises(self.module.DevelopmentRunError) as raised:
                self.run_development()
        self.assertIs(raised.exception.__cause__, primary)
        self.assertIsNone(raised.exception.terminal_path)
        self.assertEqual(len(raised.exception.persistence_errors), 3)
        self.assertEqual(len(self.calls), 16)
        self.assertEqual(raised.exception.report["counts"]["completed_episodes"], 8)

    def test_claim_schedule_and_prepared_publication_failures_have_no_false_entries(self):
        publish = self.module._publish
        for ordinal, filename in enumerate(("claim.json", "schedule.json", "prepared.json")):
            with self.subTest(filename=filename):
                self.root = Path(self.folder.name) / f"precommit-{ordinal}"
                self.calls.clear()
                primary = OSError(filename)

                def fail(root, name, value, state):
                    if name == filename:
                        raise primary
                    return publish(root, name, value, state)

                with mock.patch.object(self.module, "_publish", side_effect=fail):
                    with self.assertRaises(self.module.DevelopmentRunError) as raised:
                        self.run_development()
                self.assertIs(raised.exception.__cause__, primary)
                counts = raised.exception.report["counts"]
                self.assertEqual(counts["scheduled_final_cases"], 24)
                self.assertEqual(counts["generation_callbacks_entered"], 0)
                self.assertEqual(counts["verified_generation_claims"], 0)
                self.assertEqual(self.calls, [])

    def test_report_is_side_effect_free_and_keeps_transient_bytes_out(self):
        self.boundary_state()
        self.state["final_identity"][0] = {"generation_raw": b"not JSON"}
        before = copy.deepcopy(self.state)
        report = self.module._report(self.state, 123, aborted=True)
        self.assertEqual(self.state, before)
        self.assertNotIn("final_identity", report)
        self.assertEqual(report["costs"]["coordinator_wall_ns"], 123)
        report["episodes"][0]["completed"] = True
        self.assertFalse(self.state["episodes"][0]["completed"])

    def test_cold_import_has_no_model_or_launcher_side_effects(self):
        script = """
import importlib.abc
import subprocess
import sys
class Guard(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in ('torch', 'transformers'):
            raise AssertionError('model import: ' + fullname)
def launch(*arguments, **keywords):
    raise AssertionError('launcher invoked')
sys.meta_path.insert(0, Guard())
subprocess.Popen = launch
sys.path.insert(0, sys.argv[1])
import development_coordinator
assert callable(development_coordinator.run_development)
assert 'torch' not in sys.modules and 'transformers' not in sys.modules
print('cold import safe')
"""
        result = subprocess.run(
            [sys.executable, "-B", "-W", "error", "-I", "-c", script, str(Path(__file__).parent)],
            capture_output=True, text=True, timeout=10, check=False)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout, "cold import safe\n")


class EvidenceEdgeTests(RuntimeFixture):
    run_development = FlowTests.run_development

    def test_io_failure_does_not_hide_readable_success_contract_contradictions(self):
        original_read = Path.read_bytes
        for ordinal, kind in enumerate(("failed_generation", "failed_audit", "unreaped_audit")):
            with self.subTest(kind=kind):
                self.root = Path(self.folder.name) / f"contradiction-{ordinal}"
                self.calls.clear()
                primary = OSError("unreadable secondary file")
                suffix = "generation-1/claim.json" if kind == "failed_generation" \
                    else "public-00-P02/stdout.bin"

                def read(path):
                    if str(path).endswith(suffix):
                        raise primary
                    return original_read(path)

                def mutate(result, directory):
                    if kind == "failed_generation":
                        result["observation"].update(
                            status="failed", failure={"type": "RuntimeError", "message": "failed"})
                        (directory / "observation.json").write_bytes(encoded(result["observation"]))
                    else:
                        result["audit"].update(
                            {"failure": "failed"} if kind == "failed_audit" else {"reaped": False})
                        (directory / "observation.json").write_bytes(encoded(result["audit"]))

                changes = {"generate": altered(self.generate, mutate)} if kind == "failed_generation" \
                    else {"execute_public": altered(self.execute, mutate)}
                with mock.patch.object(Path, "read_bytes", read):
                    with self.assertRaises(self.module.DevelopmentRunError) as raised:
                        self.run_development(**changes)
                self.assertIs(raised.exception.__cause__, primary)
                self.assertEqual(raised.exception.report["calibration"]["status"], "INVALID_PROTOCOL")

    def test_each_missing_execution_record_is_a_success_contract_violation(self):
        names = ("admission.json", "claim.json", "observation.json", "receipt.json",
                 "stdout.bin", "stderr.bin", "control.bin")
        for domain, phase in enumerate(("public", "final")):
            for ordinal, name in enumerate(names):
                with self.subTest(phase=phase, missing=name):
                    self.new_boundary(domain * 100 + ordinal)
                    with self.assertRaises(self.module.BoundaryProtocolError):
                        self.execution(phase, altered(
                            self.execute, lambda result, directory: (directory / name).unlink()))
                    row = self.episode[phase + "_executions"][0]
                    self.assertEqual(len(row["files"]), 6)
                    self.assertNotIn(name, row["files"])

    def test_all_malformed_record_bytes_are_hashed_before_interpretation(self):
        names = ("admission.json", "claim.json", "observation.json", "receipt.json")
        for ordinal, name in enumerate(names):
            with self.subTest(record=name):
                self.new_boundary(ordinal)
                with self.assertRaises(self.module.BoundaryProtocolError):
                    self.execution(callback=altered(
                        self.execute, lambda result, directory: (directory / name).write_bytes(b"{")))
                row = self.episode["public_executions"][0]
                self.assertEqual(len(row["files"]), 7)
                self.assertEqual(row["files"][name], {
                    "sha256": hashlib.sha256(b"{").hexdigest(), "bytes": 1,
                })
                self.assertEqual((self.root / row["relative_dir"] / name).read_bytes(), b"{")

    def test_execution_control_mutation_is_not_reinterpreted_as_program_error(self):
        self.boundary_state()
        with self.assertRaises(self.module.BoundaryProtocolError):
            self.execution(callback=altered(
                self.execute, lambda result, directory: (directory / "control.bin").write_bytes(b"bad")))
        self.assertTrue(self.state["protocol_errors"])
        self.assertFalse(self.episode["public_executions"][0]["receipt_verified"])

    def test_nonnative_returned_envelopes_and_bool_audits_are_rejected(self):
        class Mapping(dict):
            pass

        for ordinal, phase in enumerate(("generation", "public", "final")):
            with self.subTest(phase=phase):
                self.new_boundary(ordinal)
                if phase == "generation":
                    def callback(messages, **keywords):
                        return Mapping(self.generate(messages, **keywords))
                    operation = lambda: self.generation(callback)
                else:
                    def callback(request, **keywords):
                        return Mapping(self.execute(request, **keywords))
                    operation = lambda: self.execution(phase, callback)
                with self.assertRaises(self.module.BoundaryProtocolError):
                    operation()
        self.new_boundary(10)

        def bad_audit(result, directory):
            result["audit"]["elapsed_seconds"] = True
            (directory / "observation.json").write_bytes(encoded(result["audit"]))

        with self.assertRaises(self.module.BoundaryProtocolError):
            self.execution(callback=altered(self.execute, bad_audit))

    def test_nonfinite_audit_and_generation_observation_are_protocol_errors(self):
        for ordinal, phase in enumerate(("generation", "execution")):
            with self.subTest(phase=phase):
                self.new_boundary(ordinal)

                def mutate(result, directory):
                    filename = directory / "observation.json"
                    record = parse_json(filename.read_bytes())
                    record["elapsed_ns" if phase == "generation" else "elapsed_seconds"] = float("nan")
                    filename.write_bytes(json.dumps(record).encode())

                with self.assertRaises(self.module.BoundaryProtocolError):
                    if phase == "generation":
                        self.generation(altered(self.generate, mutate))
                    else:
                        self.execution(callback=altered(self.execute, mutate))

    def test_failed_observation_can_retain_valid_generation_cost_without_completion(self):
        primary = RuntimeError("post-generation failure")
        primary.record_cleanup = {"failed_count": 0, "failures": []}

        def fail(result, directory):
            result["observation"].update(
                status="failed", failure={"type": "RuntimeError", "message": "synthetic"})
            (directory / "observation.json").write_bytes(encoded(result["observation"]))
            raise primary

        with self.assertRaises(self.module.DevelopmentRunError) as raised:
            self.run_development(generate=altered(self.generate, fail))
        report = raised.exception.report
        self.assertEqual(report["counts"]["verified_generation_records"], 1)
        self.assertEqual(report["counts"]["completed_generation_calls"], 0)
        self.assertEqual(report["costs"]["input_tokens"]["observed_subtotal"], 10)
        self.assertTrue(raised.exception.record_cleanup_complete)
        self.assertIs(raised.exception.__cause__, primary)

    def test_saved_execution_success_after_callback_error_is_retained_without_retry(self):
        primary = RuntimeError("after execution records")

        def fail(result, directory):
            raise primary

        with self.assertRaises(self.module.DevelopmentRunError) as raised:
            self.run_development(execute_final=altered(self.execute, fail))
        report = raised.exception.report
        self.assertEqual(report["counts"]["received_final_cases"], 1)
        self.assertEqual(report["counts"]["final_callbacks_entered"], 1)
        self.assertEqual(report["counts"]["unexecuted_final_cases"], 23)
        self.assertEqual(report["counts"]["final_verified_child_starts"], 1)
        self.assertEqual(len(self.calls), 2)
        self.assertIs(raised.exception.__cause__, primary)

    def test_valueerror_outcome_is_a_supported_completed_observation(self):
        self.boundary_state()

        def change(result, directory):
            outcome = {"kind": "raises", "type": "ValueError"}
            control = candidate_execution_protocol.READY + encoded(outcome) + b"\n"
            (directory / "control.bin").write_bytes(control)
            result["audit"]["output_bytes_observed"] = len(control)
            result["audit"]["stream_sha256"]["control"] = hashlib.sha256(control).hexdigest()
            result["receipt"]["outcome"] = outcome
            (directory / "observation.json").write_bytes(encoded(result["audit"]))
            (directory / "receipt.json").write_bytes(encoded(result["receipt"]))

        receipt = self.execution(callback=altered(self.execute, change))
        self.assertEqual(receipt["status"], "completed")
        self.assertEqual(receipt["outcome"], {"kind": "raises", "type": "ValueError"})

    def test_partial_score_failure_is_secondary_to_original_callback_error(self):
        primary = RuntimeError("callback")
        secondary = self.module.BoundaryProtocolError("scorer failure")
        with mock.patch.object(self.module, "_partial_score", side_effect=secondary):
            with self.assertRaises(self.module.DevelopmentRunError) as raised:
                self.run_development(execute_final=mock.Mock(side_effect=primary))
        self.assertIs(raised.exception.__cause__, primary)
        self.assertEqual(raised.exception.report["calibration"]["status"], "INVALID_PROTOCOL")
        self.assertEqual(raised.exception.report["failure"]["partial_score_error"],
                         observed_generation._failure(secondary))

    def test_valid_length_and_cuda_records_keep_exact_counts_and_memory(self):
        self.boundary_state()

        def change(result, directory):
            result["generation"].update(
                device_type="cuda", peak_allocated_bytes=8, peak_reserved_bytes=16,
                stop_reason="length", output_ids=[1] * 256, output_tokens=256, forward_calls=256)
            result["observation"].update(forward_started=256, forward_completed=256)
            for key in ("generation", "observation"):
                (directory / (key + ".json")).write_bytes(encoded(result[key]))

        raw = self.generation(altered(self.generate, change))
        self.assertEqual(parse_json(raw)["output_tokens"], 256)
        row = self.episode["generations"][0]
        self.assertEqual(row["forward_completed"], 256)
        self.assertEqual(row["peak_allocated_bytes"], 8)
        self.assertEqual(row["peak_reserved_bytes"], 16)

    def test_failed_callback_missing_claim_does_not_erase_independent_observation(self):
        primary = RuntimeError("missing claim after failure")

        def fail(result, directory):
            (directory / "claim.json").unlink()
            raise primary

        with self.assertRaises(self.module.DevelopmentRunError) as raised:
            self.run_development(generate=altered(self.generate, fail))
        report = raised.exception.report
        self.assertEqual(report["counts"]["verified_generation_claims"], 0)
        self.assertEqual(report["counts"]["completed_generation_calls"], 1)
        self.assertEqual(report["costs"]["input_tokens"]["observed_subtotal"], 10)
        self.assertEqual(report["calibration"]["status"], "INCONCLUSIVE_INCOMPLETE")
        self.assertIs(raised.exception.__cause__, primary)

    def test_malformed_failed_cleanup_preserves_primary_and_records_protocol_evidence(self):
        primary = RuntimeError("callback")
        primary.record_cleanup = {"failed_count": True, "failures": []}
        with self.assertRaises(self.module.DevelopmentRunError) as raised:
            self.run_development(generate=mock.Mock(side_effect=primary))
        self.assertIs(raised.exception.__cause__, primary)
        self.assertEqual(raised.exception.report["calibration"]["status"], "INVALID_PROTOCOL")
        self.assertIsNone(raised.exception.record_cleanup_complete)

    def test_final_missing_result_frame_is_a_program_error_not_protocol(self):
        self.boundary_state()

        def change(result, directory):
            control = candidate_execution_protocol.READY
            (directory / "control.bin").write_bytes(control)
            result["audit"]["output_bytes_observed"] = len(control)
            result["audit"]["stream_sha256"]["control"] = hashlib.sha256(control).hexdigest()
            result["receipt"] = {"request_sha256": result["audit"]["request_sha256"],
                                 "status": "program_error"}
            (directory / "observation.json").write_bytes(encoded(result["audit"]))
            (directory / "receipt.json").write_bytes(encoded(result["receipt"]))

        receipt = self.execution("final", altered(self.execute, change))
        self.assertEqual(receipt["status"], "program_error")
        self.assertEqual(self.state["protocol_errors"], [])

    def test_final_inventory_binding_detects_task_count_order_expectation_and_input_drift(self):
        self.boundary_state()
        unused_claim, generation, unused_observation = generation_records(
            self.job["messages"], self.job["second_call_id"],
            digest(self.bindings["tokenizer_state"]), self.candidates[1])
        raw = encoded(generation)
        plan = development_final_scoring.build_final_plan(
            self.raw[2], self.raw[3], run_id=self.run_id, task_id="D01", arm="base",
            generation_call_id=self.job["second_call_id"], generation_raw=raw,
            expected_generation_sha256=hashlib.sha256(raw).hexdigest())
        mutations = [
            lambda value: value.update(task_sha256="f" * 64),
            lambda value: value.update(scheduled_final_cases=4),
            lambda value: value["final_cases"].reverse(),
            lambda value: value["final_cases"][0].update(expected_sha256="f" * 64),
            lambda value: value["final_cases"][0].update(input_sha256="f" * 64),
        ]
        for ordinal, mutate in enumerate(mutations):
            job = copy.deepcopy(self.job)
            mutate(job)
            with self.subTest(mutation=ordinal), self.assertRaises(self.module.BoundaryProtocolError):
                self.module._match_inventory(job, plan)

    def test_result_publication_failure_preserves_score_but_position_is_missing(self):
        original_publish = self.module._publish
        primary = OSError("result publication")

        def publish(root, name, value, state):
            if name == "result.json":
                raise primary
            return original_publish(root, name, value, state)

        with mock.patch.object(self.module, "_publish", side_effect=publish):
            with self.assertRaises(self.module.DevelopmentRunError) as raised:
                self.run_development()
        report = raised.exception.report
        self.assertEqual(report["episodes"][0]["final_score"]["status"], "pass")
        self.assertFalse(report["episodes"][0]["completed"])
        self.assertEqual(report["counts"]["missing_episode_positions"], list(range(8)))
        self.assertEqual(report["counts"]["received_final_cases"], 3)
        self.assertEqual(report["counts"]["unexecuted_final_cases"], 21)
        self.assertIs(raised.exception.__cause__, primary)

    def test_contextual_receipt_infrastructure_error_is_not_arbitrary_callback_reclassification(self):
        self.boundary_state()
        original_receipt = candidate_execution_protocol.make_receipt
        primary = candidate_execution_protocol.InfrastructureError("reconstruction lifecycle")
        calls = []

        def receipt(*arguments, **keywords):
            calls.append(None)
            if len(calls) == 2:
                raise primary
            return original_receipt(*arguments, **keywords)

        with mock.patch.object(candidate_execution_protocol, "make_receipt", side_effect=receipt):
            with self.assertRaises(self.module.BoundaryProtocolError) as raised:
                self.execution()
        self.assertIs(raised.exception.__cause__, primary)
        self.assertEqual(len(calls), 2)
        self.new_boundary(1)
        with self.assertRaises(candidate_execution_protocol.InfrastructureError) as raised:
            self.execution(callback=mock.Mock(side_effect=primary))
        self.assertIs(raised.exception, primary)
        self.assertEqual(self.state["protocol_errors"], [])

    def test_noncanonical_successful_observer_bytes_do_not_match_returned_envelope(self):
        self.boundary_state()

        def change(result, directory):
            (directory / "generation.json").write_bytes(
                json.dumps(result["generation"], indent=2).encode())

        with self.assertRaises(self.module.BoundaryProtocolError):
            self.generation(altered(self.generate, change))
        self.assertTrue(self.episode["generations"][0]["generation_verified"])

    def test_exact_report_and_count_field_sets(self):
        report = self.run_development()["report"]
        self.assertEqual(set(report), {
            "kind", "version", "run_id", "status", "schedule_sha256", "bindings_authenticity",
            "scheduled_episodes", "scheduled_generation_calls", "episodes", "counts",
            "by_arm", "costs", "calibration", "failure", "persistence_errors",
            "record_cleanup_before_terminal", "record_cleanup_complete_before_terminal",
            "boundary_authenticity", *FLAGS,
        })
        self.assertEqual(set(report["counts"]), {
            "completed_episodes", "generation_callbacks_entered", "generation_callbacks_returned",
            "verified_generation_claims", "verified_generation_records", "completed_generation_calls",
            "public_feedback_opportunities", "public_callbacks_entered", "final_callbacks_entered",
            "public_verified_child_starts", "final_verified_child_starts",
            "public_unknown_child_starts", "final_unknown_child_starts", "public_verified_receipts",
            "final_verified_receipts", "scheduled_final_cases", "received_final_cases",
            "passed_final_cases", "missing_final_cases", "invalid_final_cases",
            "unexecuted_final_cases", "missing_episode_positions",
        })
        for name in ("driver_model_load_ns", "driver_shared_preparation_ns", "driver_outer_wall_ns"):
            self.assertIsNone(report["costs"][name])
        for arm in ("base", "text"):
            self.assertEqual([row["task_id"] for row in report["by_arm"][arm]["episodes"]],
                             ["D01", "D02", "D03", "D04"])


class RuntimeReviewTests(RuntimeFixture):
    run_development = FlowTests.run_development

    def reset_run(self, name):
        self.root = Path(self.folder.name) / name
        self.calls.clear()
        self.public_calls.clear()
        self.final_calls.clear()
        self.events.clear()

    def run_error(self, **changes):
        try:
            self.run_development(**changes)
        except BaseException as error:
            self.assertIsInstance(error, self.module.DevelopmentRunError)
            return error
        self.fail("expected a typed development failure")

    def large_elapsed(self, phase, primary=None):
        entered = []

        def execute(request, *, artifact_dir):
            entered.append(request["case_id"])
            result = self.execute(request, artifact_dir=artifact_dir)
            if len(entered) <= 2:
                result["audit"]["elapsed_seconds"] = 1e308
                (artifact_dir / "observation.json").write_bytes(encoded(result["audit"]))
            if len(entered) == 2 and primary is not None:
                raise primary
            return result

        return {"execute_" + phase: execute}

    def assert_overflow_evidence(self, report, phase):
        elapsed = report["costs"]["execution_elapsed_seconds"]
        self.assertIsNone(elapsed["total"])
        self.assertIsNone(elapsed["observed_subtotal"])
        self.assertEqual(elapsed["aggregation_error"]["type"], "OverflowError")
        self.assertIn("finite", elapsed["aggregation_error"]["message"])
        self.assertEqual(elapsed["unknown_attempts"], 0)
        rows = report["episodes"][0][phase + "_executions"]
        self.assertEqual([row["execution_elapsed_seconds"] for row in rows[:2]], [1e308, 1e308])
        self.assertTrue(all(row["observation_verified"] for row in rows[:2]))
        for row in rows[:2]:
            raw = (self.root / row["relative_dir"] / "observation.json").read_bytes()
            self.assertEqual(row["files"]["observation.json"]["sha256"], hashlib.sha256(raw).hexdigest())
            self.assertEqual(parse_json(raw)["elapsed_seconds"], 1e308)
        self.assertEqual(report["calibration"]["status"], "INCONCLUSIVE_INCOMPLETE")
        self.assertIn("unknown_necessary_cost",
                      report["calibration"]["reasons"]["INCONCLUSIVE_INCOMPLETE"])
        self.assertEqual(report["calibration"]["reasons"]["INVALID_PROTOCOL"], [])
        self.assertEqual(parse_json(encoded(report)), report)

    def test_f1_overflow_after_saved_failure_preserves_primary_in_both_domains(self):
        for phase in ("public", "final"):
            with self.subTest(phase=phase):
                self.reset_run("overflow-failed-" + phase)
                primary = RuntimeError("second saved " + phase)
                error = self.run_error(**self.large_elapsed(phase, primary))
                self.assertIs(error.__cause__, primary)
                self.assertEqual(error.report["status"], "aborted")
                self.assert_overflow_evidence(error.report, phase)
                self.assertEqual(error.report["counts"][phase + "_callbacks_entered"], 2)
                self.assertEqual(error.report["counts"][phase + "_verified_receipts"], 2)
                self.assertEqual(len(self.calls), 1 if phase == "public" else 2)
                self.assertEqual(error.terminal_path, "partial-report.json")
                self.assertEqual((self.root / error.terminal_path).read_bytes(), encoded(error.report))
                self.assertEqual(parse_json((self.root / "failure.json").read_bytes())["error"],
                                 {"type": "RuntimeError", "message": str(primary)})

    def test_f1_successful_flow_with_overflow_keeps_all_callbacks_and_null_aggregate(self):
        for phase in ("public", "final"):
            with self.subTest(phase=phase):
                self.reset_run("overflow-complete-" + phase)
                try:
                    result = self.run_development(**self.large_elapsed(phase))
                except BaseException as error:
                    self.fail(f"finite evidence must remain reportable: {type(error).__name__}: {error}")
                report = result["report"]
                self.assert_overflow_evidence(report, phase)
                self.assertEqual(report["status"], "completed")
                self.assertEqual(report["counts"]["completed_episodes"], 8)
                self.assertEqual(report["counts"]["generation_callbacks_entered"], 16)
                self.assertEqual(report["counts"]["public_verified_receipts"], 16)
                self.assertEqual(report["counts"]["final_verified_receipts"], 24)
                self.assertEqual(report["costs"]["execution_elapsed_seconds"]["known_attempts"], 40)
                self.assertEqual(result["terminal_path"], "development-report.json")
                self.assertEqual((self.root / result["terminal_path"]).read_bytes(), encoded(report))
                self.assertFalse((self.root / "failure.json").exists())

    def test_f1_overflow_report_is_pure_and_retains_individual_evidence(self):
        self.boundary_state()
        callback = self.large_elapsed("public")["execute_public"]
        self.execution(callback=callback)
        self.execution(callback=callback)
        before = copy.deepcopy(self.state)
        try:
            report = self.module._report(self.state, 123, aborted=True)
        except BaseException as error:
            self.fail(f"finite aggregate must be reportable: {type(error).__name__}: {error}")
        self.assertEqual(self.state, before)
        self.assert_overflow_evidence(report, "public")
        report["episodes"][0]["public_executions"][0]["execution_elapsed_seconds"] = 0
        self.assertEqual(self.state, before)

    def test_f1_unexpected_report_error_preserves_callback_primary_without_reduction_retry(self):
        for phase in ("public", "final"):
            with self.subTest(phase=phase):
                self.reset_run("reducer-failed-" + phase)
                primary = RuntimeError("callback primary")
                reduction = ValueError("report reduction unavailable")
                with mock.patch.object(self.module, "_report", side_effect=reduction) as report_call:
                    error = self.run_error(**self.large_elapsed(phase, primary))
                self.assertIs(error.__cause__, primary)
                self.assertEqual(report_call.call_count, 1)
                self.assertEqual(error.report["failure"]["report_error"],
                                 {"type": "ValueError", "message": str(reduction)})
                self.assertEqual(error.report["status"], "aborted")
                for name in ("counts", "costs", "by_arm"):
                    self.assertIsNone(error.report[name])
                self.assertEqual(len(error.report["episodes"]), 8)
                rows = error.report["episodes"][0][phase + "_executions"]
                self.assertEqual([row["execution_elapsed_seconds"] for row in rows], [1e308, 1e308])
                self.assertEqual(error.terminal_path, "partial-report.json")
                self.assertEqual((self.root / error.terminal_path).read_bytes(), encoded(error.report))
                self.assertTrue((self.root / "failure.json").is_file())

    def test_f1_success_report_reduction_error_is_primary_and_is_not_retried(self):
        reduction = ValueError("first report reduction")
        with mock.patch.object(self.module, "_report", side_effect=reduction) as report_call:
            error = self.run_error()
        self.assertIs(error.__cause__, reduction)
        self.assertEqual(report_call.call_count, 1)
        self.assertEqual(len(self.calls), 16)
        self.assertEqual(len(self.final_calls), 24)
        self.assertEqual(error.report["failure"]["report_error"],
                         {"type": "ValueError", "message": str(reduction)})
        self.assertEqual(error.terminal_path, "partial-report.json")
        self.assertFalse((self.root / "development-report.json").exists())

    def test_f1_elapsed_reducer_distinguishes_zero_unknown_and_integer_overflow(self):
        cases = (
            ([], None, None, 0, 0, None),
            ([None], None, None, 0, 1, None),
            ([0], 0, 0, 1, 0, None),
            ([0.01, None], None, 0.01, 1, 1, None),
            ([1e308, 1e308, None], None, None, 2, 1, "OverflowError"),
            ([10 ** 308, 10 ** 308], 2 * 10 ** 308, 2 * 10 ** 308, 2, 0, None),
            ([10 ** 308, 10 ** 308, 0.01], None, None, 3, 0, "OverflowError"),
        )
        for ordinal, (values, total, subtotal, known, unknown, error_type) in enumerate(cases):
            with self.subTest(case=ordinal):
                rows = [{"execution_elapsed_seconds": value} for value in values]
                before = copy.deepcopy(rows)
                result = self.module._execution_cost(rows)
                self.assertEqual(result["total"], total)
                self.assertEqual(result["observed_subtotal"], subtotal)
                self.assertEqual(result["known_attempts"], known)
                self.assertEqual(result["unknown_attempts"], unknown)
                self.assertEqual(result["entered_attempts"], len(values))
                if error_type is None:
                    self.assertIsNone(result["aggregation_error"])
                else:
                    self.assertEqual(result["aggregation_error"]["type"], error_type)
                self.assertEqual(parse_json(encoded(result)), result)
                self.assertEqual(rows, before)

    def failed_input(self, complete, sent_mode, primary):
        def execute(request, *, artifact_dir):
            result = self.execute(request, artifact_dir=artifact_dir)
            audit = result["audit"]
            sent = audit["input_bytes_sent"]
            amounts = {"full": sent, "partial": sent // 2, "zero": 0}
            audit.update(input_complete=complete,
                         input_bytes_sent=amounts[sent_mode],
                         reaped=False, eof=False, returncode=None, failure="input interrupted")
            control = candidate_execution_protocol.READY
            audit["stream_sha256"]["control"] = hashlib.sha256(control).hexdigest()
            audit["output_bytes_observed"] = len(control)
            (artifact_dir / "control.bin").write_bytes(control)
            (artifact_dir / "observation.json").write_bytes(encoded(audit))
            (artifact_dir / "receipt.json").unlink()
            raise primary

        return execute

    def test_f2_failed_audit_rejects_both_input_contradictions_in_both_domains(self):
        for phase in ("public", "final"):
            for complete, sent_mode in ((True, "zero"), (False, "full")):
                with self.subTest(phase=phase, complete=complete):
                    self.reset_run(f"input-contradiction-{phase}-{complete}")
                    primary = RuntimeError("failed input")
                    error = self.run_error(**{
                        "execute_" + phase: self.failed_input(complete, sent_mode, primary)})
                    self.assertIs(error.__cause__, primary)
                    row = error.report["episodes"][0][phase + "_executions"][0]
                    self.assertFalse(row["observation_verified"])
                    self.assertTrue(row["claim_verified"])
                    self.assertIsNone(row["execution_elapsed_seconds"])
                    self.assertFalse(row["receipt_verified"])
                    self.assertEqual(len(row["files"]), 6)
                    self.assertEqual(row["failure"], {"type": "RuntimeError", "message": str(primary)})
                    self.assertTrue(any(item["type"] == "BoundaryProtocolError"
                                        for item in row["evidence_errors"]))
                    self.assertEqual(error.report["calibration"]["status"], "INVALID_PROTOCOL")
                    self.assertEqual(error.report["counts"][phase + "_callbacks_entered"], 1)
                    self.assertEqual(error.terminal_path, "partial-report.json")

    def test_f2_coherent_partial_input_retains_verified_failed_evidence_in_both_domains(self):
        for phase in ("public", "final"):
            with self.subTest(phase=phase):
                self.reset_run("input-partial-" + phase)
                primary = RuntimeError("coherent partial input")
                error = self.run_error(**{
                    "execute_" + phase: self.failed_input(False, "partial", primary)})
                self.assertIs(error.__cause__, primary)
                row = error.report["episodes"][0][phase + "_executions"][0]
                self.assertTrue(row["observation_verified"])
                self.assertTrue(row["claim_verified"])
                self.assertTrue(row["process_started"])
                self.assertEqual(row["execution_elapsed_seconds"], 0.01)
                self.assertFalse(row["receipt_verified"])
                self.assertEqual(row["evidence_errors"], [])
                self.assertEqual(error.report["calibration"]["status"], "INCONCLUSIVE_INCOMPLETE")
                self.assertEqual(error.report["calibration"]["reasons"]["INVALID_PROTOCOL"], [])
                self.assertEqual(error.report["counts"][phase + "_callbacks_entered"], 1)


if __name__ == "__main__":
    unittest.main()
