# Owned Tokenizer Session Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development. Execute the two tasks sequentially with fresh implementers and independent reviews. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement a one-shot tokenizer-only loading session that owns its instance, binds measurements to observed state, and invalidates provisional results on integrity failure.

**Architecture:** A process-local guard establishes cold-start/offline import rules. A session creates the tokenizer internally from the approved snapshot, observes its identity, delegates counting to the sealed neutral measurement module, and checks integrity before finalization. The later audit controller must consume both provisional measurements and successful finalization; neither is a complete audit receipt.

**Tech Stack:** Python 3.12 standard library, synthetic unittest fixtures. Production loading targets the previously specified transformers API; actual target-version import/load compatibility remains untested and requires separate admission.

## Global Constraints

- Binding parent: `docs/superpowers/specs/2026-11-12-lora-tokenizer-admission-design.md`, especially sections 3, 5, 6 and 7. Do not edit it.
- Preserve `/Users/bytedance/dev/TinyLLMForge`, branch `experiments/feedback-active-verification`, HEAD `93b4e8e0dc5174db0426a5da9e29cc360ed331a7`.
- No commits, pushes, branch/worktree creation, SSH/network, installs, telemetry, real metadata discovery, real tokenizer/model loading, weights access, candidate execution, GPU work or training during local implementation.
- The real launch path is code under test, not permission to invoke it with installed third-party packages. Tests substitute the import boundary, use synthetic bytes and metadata, and must not emit a real audit receipt.
- Use apply_patch; no inline code comments or one-letter identifiers. Keep completed source/tests, data, READMEs, previous plans/ledgers and sealed evidence unchanged.
- Previous closeout: `/tmp/tlf-tokenizer-runtime-20261112-TeMBri/final-verification.json`. Do not execute its consumed helpers or write into its evidence root.
- Only the five pinned files enter a snapshot. Use the sealed `validate_profile`, `inspect_runtime`, `verify_snapshot` and `measure` interfaces; do not weaken them or reuse the selection model preflight.
- Python 3.12.13 and transformers 4.57.1 remain future-run requirements. Other real versions remain unknown; synthetic fixture strings do not approve them.
- `benchmark_eligible`, `formal_evaluation_ready`, `net_benefit_established`, `isolated`, and `all_possible_second_rounds_validated` remain false.
- Do not change the 324 audit rows, 4096 input/256 answer token limits, complete eight examples, comparison arms, loss/training settings or private/public data boundary.
- Standard-library unittest/TDD and explicit source/test ownership replace incompatible plugin bootstrap, telemetry, flush and test-only defaults.

## Design Decisions

Three routes were considered:

1. Accept a caller-provided tokenizer or identity: rejected. It cannot establish that the instance was loaded from this run's approved snapshot.
2. Reuse the old selection preflight or a reusable application-wide loader: rejected. The former explicitly imports torch and generation machinery; the latter cannot undo already-imported backend state or give an unambiguous single-attempt lifetime.
3. Internally load once in a dedicated fresh process: selected under the existing one-run parent contract. This component enforces its own process-local attempt and guards; process creation, hard limits and termination remain the wrapper's responsibility.

The module import itself is pure. After the main-thread entry prerequisite succeeds, calling `open_session` claims the process's single attempt, including failed attempts, before any third-party import. An off-main-thread call is rejected before claiming that attempt, matching `RuntimeGuard.start()` below. No retry of a claimed attempt, reset API, fallback tokenizer, environment restoration or model backend activation is provided. Tests reset private state only inside explicit patches; that is not a production capability.

The guard rejects already-imported tokenizer-stack modules on entry. It sets offline/backend-disable variables before loading and installs a first-position import finder. A prohibited import attempt poisons the guard even if library code catches the ImportError. Checks also detect cached forbidden modules, environment drift and removal/reordering of the finder. This is fail-closed instrumentation for a trusted dedicated process, not an adversarial Python sandbox or universal network/OS isolation.

`open_session(root, raw, approved_sha256)` accepts no tokenizer, loader callback or identity object. It validates runtime metadata and snapshot, calls AutoTokenizer internally with explicit fast/local/no-remote-code arguments, verifies membership in the library's fast-tokenizer base class, and observes state. The actual class is recorded, not guessed from a model family name. Compatibility with the admitted historical environment is a future test, not inferred from a different local installation.

Observed state covers exact template text/hash, vocabulary hash, special-token strings/IDs, backend serialization hash, actual qualified class and relevant encode settings. The observer also reconstructs the sealed measurement module's four-field state hash so a returned count must agree with the owned instance. Backend serialization is stronger than checking only vocabulary/template, but still does not authenticate package code.

### Session State And Outputs

| State | Allowed operation | Transition |
| --- | --- | --- |
| ACTIVE | measure | ACTIVE on OK; STOPPED on input/output limit |
| ACTIVE or STOPPED | finish | CLOSED only after final integrity checks |
| ACTIVE or STOPPED | abort | ABORTED, no finalization |
| Any live operation fails | none afterward | FAILED, discard instance and close guard |
| CLOSED, ABORTED, FAILED | no measure/finish | reject; never reopen |

Reentrant calls are errors, including when a callback catches the inner error. Guard checks require the main Python thread; this is not a native-thread resource limit.

Every measurement remains `TOKENIZER_SESSION_MEASUREMENT_PROVISIONAL`, wrapping the unchanged `TOKENIZER_MEASUREMENT_UNATTESTED` result. It carries run/profile/observed-state digests, ordinal and rolling chain hash. Identity/environment are rechecked per measurement. Metadata and snapshot bytes are rechecked before and after loading and at finalization. An intervening persistent file/version drift therefore invalidates the entire provisional sequence, not just the last row.

`finish` binds count, final chain, baseline observed identity, runtime metadata and snapshot into `TOKENIZER_SESSION_FINALIZATION`. It can finalize a STOPPED session, but retains its budget stop reason and never says the inventory passed. `tokenizer_loaded=true` describes the internally loaded production instance only; this flag in a mocked unit-test result is not actual loading evidence. All five admission/scientific flags remain false.

The initial chain is `digest([])`. For each record before adding its `chain_sha256`, the next chain is `digest({"previous": previous_chain, "record": record})`. A later controller must reconstruct this chain, verify ordinals and require the matching finalization. These are consistency bindings, not signatures or protection against a malicious controlling process.

Exceptions exposed by the session are sanitized; no tokenizer exception message or private prompt/answer is included in its public error. Cancellation is re-raised after invalidation. Aborting or dropping the Python object is not independent process/GPU/resource cleanup evidence.

## Ownership And Entry

| Owner | New files |
| --- | --- |
| Task 1 | `experiments/lora_necessity/tokenizer_runtime_guard.py`, `test_tokenizer_runtime_guard.py` |
| Task 2 | `experiments/lora_necessity/tokenizer_session.py`, `test_tokenizer_session.py` |
| Main | This plan, `.superpowers/sdd/lora-tokenizer-session-20261112-progress.md`, fresh evidence root |

- [ ] Verify previous receipt, source/document/evidence hashes and its protected manifest; capture a fresh preservation manifest including prior stage ledgers.
- [ ] Verify branch/HEAD and empty tracked/index diffs. Preserve unrelated untracked work.
- [ ] Verify all four implementation/test paths are absent.
- [ ] Run the fresh 83-test baseline, retaining command, log, exit and source hashes:

```bash
PYTHONDONTWRITEBYTECODE=1 /opt/homebrew/opt/python@3.12/bin/python3.12 \
  -m unittest discover -s experiments/lora_necessity -p 'test_*.py' -v
```

## Task 1: One-Shot Runtime Guard

Create `tokenizer_runtime_guard.py` and its colocated test file. The only consumer is the new session. `RuntimeGuard.start()`, `.check()` and `.close()` manage the process-local lifetime; there is no production reset.

- [ ] Write the complete tests first using apply_patch.

```python
import importlib
import importlib.util
from pathlib import Path
import sys
from types import ModuleType
import unittest
from unittest.mock import patch


class RuntimeGuardTests(unittest.TestCase):
    def setUp(self):
        self.assertTrue(Path(__file__).with_name("tokenizer_runtime_guard.py").is_file())
        self.module = importlib.import_module("tokenizer_runtime_guard")
        for context in (
            patch.object(self.module, "_USED", False),
            patch.object(sys, "meta_path", list(sys.meta_path)),
            patch.dict(self.module.os.environ),
        ):
            context.start()
            self.addCleanup(context.stop)
        self.guard = self.module.RuntimeGuard()
        self.addCleanup(self.guard.close)

    def test_start_check_close_and_no_retry(self):
        self.guard.start()
        self.guard.check()
        for name, expected in self.module.ENVIRONMENT.items():
            self.assertEqual(self.module.os.environ[name], expected)
        self.guard.close()
        self.assertNotIn(self.guard, sys.meta_path)
        with self.assertRaises(ValueError):
            self.guard.check()
        with self.assertRaises(ValueError):
            self.module.RuntimeGuard().start()

    def test_preloaded_stack_rejected_before_environment_change(self):
        before = dict(self.module.os.environ)
        with patch.dict(sys.modules, {"transformers.fixture": ModuleType("fixture")}):
            with self.assertRaises(ValueError):
                self.guard.start()
        self.assertEqual(dict(self.module.os.environ), before)
        self.assertNotIn(self.guard, sys.meta_path)
        with self.assertRaises(ValueError):
            self.module.RuntimeGuard().start()

    def test_caught_forbidden_import_still_poisons_guard(self):
        self.guard.start()
        with self.assertRaises(ImportError):
            self.guard.find_spec("transformers.generation.utils", None)
        with self.assertRaises(ValueError):
            self.guard.check()
        self.assertIsNone(self.guard.find_spec("tokenizers.fixture", None))
        with self.assertRaises(ValueError):
            self.guard.check()

    def test_cached_backend_and_environment_drift_are_sticky(self):
        self.guard.start()
        with patch.dict(sys.modules, {"torch.fixture": ModuleType("fixture")}):
            with self.assertRaises(ValueError):
                self.guard.check()
        with self.assertRaises(ValueError):
            self.guard.check()
        self.guard.close()
        with patch.object(self.module, "_USED", False):
            another = self.module.RuntimeGuard()
            self.addCleanup(another.close)
            another.start()
            self.module.os.environ["USE_TORCH"] = "1"
            with self.assertRaises(ValueError):
                another.check()
            self.module.os.environ["USE_TORCH"] = "0"
            with self.assertRaises(ValueError):
                another.check()

    def test_finder_removal_and_wrong_thread_rejected(self):
        self.guard.start()
        sys.meta_path.remove(self.guard)
        with self.assertRaises(ValueError):
            self.guard.check()
        with patch.object(self.module, "_USED", False), patch.object(
                self.module.threading, "current_thread", return_value=object()):
            with self.assertRaises(ValueError):
                self.module.RuntimeGuard().start()

    def test_import_dispatch_blocks_synthetic_backend_body(self):
        executed = []

        class SyntheticBackend:
            def find_spec(self, fullname, path, target=None):
                if fullname == "torch":
                    return importlib.util.spec_from_loader(fullname, self)
                return None

            def create_module(self, spec):
                return None

            def exec_module(self, module):
                executed.append(module.__name__)

        sys.meta_path.insert(0, SyntheticBackend())
        with patch.dict(sys.modules):
            self.guard.start()
            with self.assertRaises(ImportError):
                importlib.import_module("torch")
            self.assertEqual(executed, [])
            self.assertNotIn("torch", sys.modules)
            with self.assertRaises(ValueError):
                self.guard.check()
```

- [ ] Run focused RED; retain actual missing-file setup failures without claiming behavioral sensitivity:

```bash
PYTHONDONTWRITEBYTECODE=1 /opt/homebrew/opt/python@3.12/bin/python3.12 \
  -m unittest discover -s experiments/lora_necessity -p 'test_tokenizer_runtime_guard.py' -v
```

- [ ] Implement this source using apply_patch.

```python
import os
import sys
import threading

from pilot_protocol import require


ENVIRONMENT = {
    "USE_TORCH": "0", "USE_TF": "0", "USE_FLAX": "0",
    "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
    "HF_HUB_DISABLE_TELEMETRY": "1", "TOKENIZERS_PARALLELISM": "false",
    "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
}
BACKENDS = {"torch", "tensorflow", "jax", "jaxlib", "flax", "safetensors",
            "triton", "cupy", "cuda"}
COLD_STACK = {"transformers", "tokenizers", "huggingface_hub", "jinja2"}
_USED = False


def _forbidden(name):
    parts = name.split(".")
    return parts[0] in BACKENDS or (
        parts[0] == "transformers" and
        ("generation" in parts or any(part.startswith("modeling_") for part in parts)))


class RuntimeGuard:
    def __init__(self):
        self.active = False
        self.failed = False

    def start(self):
        global _USED
        require(threading.current_thread() is threading.main_thread(), "main thread required")
        require(not _USED, "runtime attempt already consumed")
        _USED = True
        require(not any(_forbidden(name) or name.split(".")[0] in COLD_STACK
                        for name in sys.modules), "cold tokenizer process required")
        os.environ.update(ENVIRONMENT)
        sys.meta_path.insert(0, self)
        self.active = True
        self.check()

    def find_spec(self, fullname, path, target=None):
        if _forbidden(fullname):
            self.failed = True
            raise ImportError("prohibited tokenizer backend import")
        return None

    def check(self):
        try:
            require(self.active and not self.failed, "runtime guard inactive or failed")
            require(threading.current_thread() is threading.main_thread(), "main thread required")
            require(bool(sys.meta_path) and sys.meta_path[0] is self, "runtime import guard changed")
            require(all(os.environ.get(name) == value for name, value in ENVIRONMENT.items()),
                    "runtime environment changed")
            require(not any(_forbidden(name) for name in sys.modules), "prohibited backend cached")
        except BaseException:
            self.failed = True
            raise

    def close(self):
        sys.meta_path[:] = [finder for finder in sys.meta_path if finder is not self]
        self.active = False
```

- [ ] Run focused GREEN and full local suite; planned counts are six and 89, not observed results.
- [ ] The sixth test must exercise real import dispatch in the fresh unittest process. Its lower-priority synthetic loader prevents execution of an installed package even if the guard regresses. Retain source/log/exit evidence; do not replace it with an unguarded installed-package import.
- [ ] Independently review both full additions, including lack of import-time mutations, cached-module rejection and test patch cleanup.

## Task 2: Internally Owned Session

Create `tokenizer_session.py` and its colocated tests.

Consumes `RuntimeGuard`, `inspect_runtime(raw, approved_sha256)`, `verify_snapshot(root, raw, approved_sha256)`, `measure(tokenizer, messages, answer=None)`, `digest`, `require` and `FLAGS`.

Public entry: `open_session(root, raw, approved_sha256)`. Returned object exposes `.measure(messages, answer=None)`, `.finish()` and `.abort()`. `_observe`, `_load_tokenizer` and `_Session` are implementation details, not alternative admission interfaces.

- [ ] Write these complete tests first.

```python
import copy
import hashlib
import importlib
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from test_tokenizer_measurement import MappingTokenizer
from validation_contract import digest, encoded


class SessionFixtureTokenizer(MappingTokenizer):
    def __init__(self):
        self.is_fast = True
        self.eos_token = "[EOS]"
        self.pad_token = "[PAD]"
        self.special_tokens_map = {"eos_token": "[EOS]", "pad_token": "[PAD]"}
        self.all_special_tokens = ["[EOS]", "[PAD]"]
        self.all_special_ids = [2, 0]
        self.backend_payload = '{"synthetic":1}'
        self.backend_tokenizer = SimpleNamespace(to_str=lambda: self.backend_payload)
        self.padding_side = "right"
        self.truncation_side = "right"
        self.model_max_length = 4096
        self.split_special_tokens = False


class TokenizerSessionTests(unittest.TestCase):
    def setUp(self):
        self.assertTrue(Path(__file__).with_name("tokenizer_session.py").is_file())
        self.module = importlib.import_module("tokenizer_session")
        self.profile_module = importlib.import_module("tokenizer_runtime_profile")
        self.guard_module = importlib.import_module("tokenizer_runtime_guard")
        self.snapshot_module = importlib.import_module("tokenizer_snapshot")
        self.temporary = tempfile.TemporaryDirectory()
        self.directory = Path(self.temporary.name).resolve()
        self.addCleanup(self.cleanup_files)
        source = self.directory / "source"
        source.mkdir()
        payloads = {name: name.encode("ascii") for name in self.profile_module.FILE_PINS}
        pins = {name: hashlib.sha256(raw).hexdigest() for name, raw in payloads.items()}
        for name, raw in payloads.items():
            (source / name).write_bytes(raw)
        self.versions = {"python": "3.12.13", "transformers": "4.57.1",
                         "tokenizers": "fixture", "huggingface-hub": "fixture", "jinja2": "fixture"}
        profile = {
            "kind": "TOKENIZER_RUNTIME_PROFILE", "version": 1, "run_id": "a" * 32,
            "revision": self.profile_module.REVISION, "file_sha256": pins,
            "versions": self.versions,
        }
        self.raw = encoded(profile)
        self.approved = hashlib.sha256(self.raw).hexdigest()
        self.root = self.directory / profile["run_id"]
        for context in (
            patch.object(self.profile_module, "FILE_PINS", pins),
            patch.object(self.profile_module, "runtime_inventory", return_value=self.versions),
            patch.object(self.guard_module, "_USED", False),
            patch.object(sys, "meta_path", list(sys.meta_path)),
            patch.dict(self.guard_module.os.environ),
        ):
            context.start()
            self.addCleanup(context.stop)
        self.snapshot_module.create_snapshot(source, self.directory, self.raw, self.approved)
        self.tokenizer = SessionFixtureTokenizer()
        self.load_calls = []

        def from_pretrained(*arguments, **keywords):
            self.load_calls.append((arguments, keywords))
            return self.tokenizer

        self.library = SimpleNamespace(
            AutoTokenizer=SimpleNamespace(from_pretrained=from_pretrained),
            PreTrainedTokenizerFast=SessionFixtureTokenizer)
        context = patch.object(self.module.importlib, "import_module", return_value=self.library)
        self.importer = context.start()
        self.addCleanup(context.stop)
        self.messages = [{"role": "user", "content": "question"}]

    def cleanup_files(self):
        for path in self.directory.rglob("*"):
            if path.is_dir() and not path.is_symlink():
                path.chmod(0o700)
        self.temporary.cleanup()

    def open(self):
        session = self.module.open_session(self.root, self.raw, self.approved)
        self.addCleanup(session.abort)
        return session

    def test_owned_loading_measurement_chain_and_finalization(self):
        session = self.open()
        self.importer.assert_called_once_with("transformers")
        self.assertEqual(self.load_calls, [((str(self.root / "snapshot"),), {
            "use_fast": True, "local_files_only": True, "trust_remote_code": False,
        })])
        record = session.measure(self.messages, "answer")
        chain = record.pop("chain_sha256")
        self.assertEqual(chain, digest({"previous": digest([]), "record": record}))
        self.assertEqual(record["kind"], "TOKENIZER_SESSION_MEASUREMENT_PROVISIONAL")
        self.assertEqual(record["measurement"]["kind"], "TOKENIZER_MEASUREMENT_UNATTESTED")
        self.assertEqual(record["measurement"]["target"]["target_tokens"], 7)
        record["measurement"]["input"]["input_ids"].clear()
        final = session.finish()
        self.assertEqual(final["chain_sha256"], chain)
        self.assertEqual(final["measurements"], 1)
        self.assertIsNone(final["stop_reason"])
        self.assertTrue(final["tokenizer_loaded"])
        for name in (*self.module.FLAGS, "all_possible_second_rounds_validated"):
            self.assertIs(final[name], False)
        with self.assertRaises(self.module.SessionError):
            session.measure(self.messages)
        with self.assertRaises(self.module.SessionError):
            session.finish()

    def test_bad_approval_never_loads_and_consumes_attempt(self):
        with self.assertRaises(self.module.SessionError):
            self.module.open_session(self.root, self.raw, "0" * 64)
        self.importer.assert_not_called()
        with self.assertRaises(self.module.SessionError):
            self.open()
        self.importer.assert_not_called()

    def test_load_failure_is_sanitized_and_no_retry(self):
        self.importer.side_effect = RuntimeError("private fixture text")
        with self.assertRaisesRegex(self.module.SessionError, "^tokenizer session failed$"):
            self.open()
        with self.assertRaises(self.module.SessionError):
            self.open()
        self.assertEqual(self.importer.call_count, 1)

    def test_slow_mode_and_wrong_special_mapping_rejected(self):
        self.tokenizer.is_fast = False
        with self.assertRaises(ValueError):
            self.module._observe(self.tokenizer)
        self.tokenizer.is_fast = True
        self.tokenizer.eos_token = "[PAD]"
        with self.assertRaises(ValueError):
            self.module._observe(self.tokenizer)
        with self.assertRaises(self.module.SessionError):
            self.open()

    def test_observed_state_drift_invalidates_session(self):
        session = self.open()
        self.tokenizer.backend_payload = '{"synthetic":2}'
        with self.assertRaises(self.module.SessionError):
            session.measure(self.messages)
        self.tokenizer.backend_payload = '{"synthetic":1}'
        with self.assertRaises(self.module.SessionError):
            session.measure(self.messages)
        with self.assertRaises(self.module.SessionError):
            session.finish()

    def test_metadata_drift_prevents_finalization(self):
        session = self.open()
        session.measure(self.messages)
        self.versions["jinja2"] = "changed"
        with self.assertRaises(self.module.SessionError):
            session.finish()
        self.assertIsNone(session._tokenizer)

    def test_snapshot_drift_prevents_finalization(self):
        session = self.open()
        session.measure(self.messages)
        target = self.root / "snapshot" / "config.json"
        target.chmod(0o600)
        target.write_bytes(b"changed")
        target.chmod(0o400)
        with self.assertRaises(self.module.SessionError):
            session.finish()
        self.assertIsNone(session._tokenizer)

    def test_input_limit_stops_measurement_but_finalizes_stop(self):
        session = self.open()
        record = session.measure([{"role": "user", "content": "x" * 4097}])
        self.assertEqual(record["measurement"]["status"], "INPUT_LIMIT")
        final = session.finish()
        self.assertEqual(final["stop_reason"], "INPUT_LIMIT")
        self.assertEqual(final["measurements"], 1)

    def test_output_limit_rejects_next_measurement(self):
        session = self.open()
        record = session.measure(self.messages, "x" * 256)
        self.assertEqual(record["measurement"]["status"], "OUTPUT_LIMIT")
        with self.assertRaises(self.module.SessionError):
            session.measure(self.messages)
        with self.assertRaises(self.module.SessionError):
            session.finish()

    def test_callback_message_mutation_and_reentry_fail_closed(self):
        session = self.open()
        original = SessionFixtureTokenizer.__call__

        def changed(tokenizer, text, **keywords):
            self.messages[0]["content"] = "changed"
            try:
                session.measure([{"role": "user", "content": "nested"}])
            except self.module.SessionError:
                pass
            return original(tokenizer, text, **keywords)

        with patch.object(SessionFixtureTokenizer, "__call__", changed):
            with self.assertRaises(self.module.SessionError):
                session.measure(self.messages)
        with self.assertRaises(self.module.SessionError):
            session.finish()

    def test_abort_and_environment_drift_do_not_finalize(self):
        session = self.open()
        self.guard_module.os.environ["USE_TORCH"] = "1"
        with self.assertRaises(self.module.SessionError):
            session.measure(self.messages)
        session.abort()
        self.assertIsNone(session._tokenizer)
        with self.assertRaises(self.module.SessionError):
            session.finish()

    def test_message_mutation_without_reentry_rejected(self):
        session = self.open()
        original = SessionFixtureTokenizer.__call__

        def changed(tokenizer, text, **keywords):
            self.messages[0]["content"] = "changed"
            return original(tokenizer, text, **keywords)

        with patch.object(SessionFixtureTokenizer, "__call__", changed):
            with self.assertRaises(self.module.SessionError):
                session.measure(self.messages)
        with self.assertRaises(self.module.SessionError):
            session.finish()

    def test_backend_drift_during_encoding_rejected(self):
        session = self.open()
        original = SessionFixtureTokenizer.__call__

        def changed(tokenizer, text, **keywords):
            tokenizer.backend_payload = '{"synthetic":2}'
            return original(tokenizer, text, **keywords)

        with patch.object(SessionFixtureTokenizer, "__call__", changed):
            with self.assertRaises(self.module.SessionError):
                session.measure(self.messages)
        with self.assertRaises(self.module.SessionError):
            session.finish()

    def test_active_abort_releases_instance_without_finalization(self):
        session = self.open()
        session.abort()
        session.abort()
        self.assertIsNone(session._tokenizer)
        with self.assertRaises(self.module.SessionError):
            session.finish()

    def test_caught_guard_failure_in_loader_rejected(self):
        def loader(*arguments, **keywords):
            guard = next(finder for finder in sys.meta_path
                         if isinstance(finder, self.guard_module.RuntimeGuard))
            try:
                guard.find_spec("torch", None)
            except ImportError:
                pass
            return self.tokenizer

        self.library.AutoTokenizer.from_pretrained = loader
        with self.assertRaises(self.module.SessionError):
            self.open()

    def test_wrong_exported_fast_class_rejected(self):
        self.library.PreTrainedTokenizerFast = str
        with self.assertRaises(self.module.SessionError):
            self.open()

    def test_runtime_drift_during_load_rejected(self):
        def loader(*arguments, **keywords):
            self.versions["jinja2"] = "changed"
            return self.tokenizer

        self.library.AutoTokenizer.from_pretrained = loader
        with self.assertRaises(self.module.SessionError):
            self.open()
```

- [ ] Run focused RED and retain the actual missing-file failures:

```bash
PYTHONDONTWRITEBYTECODE=1 /opt/homebrew/opt/python@3.12/bin/python3.12 \
  -m unittest discover -s experiments/lora_necessity -p 'test_tokenizer_session.py' -v
```

- [ ] Implement this complete source using apply_patch.

```python
import copy
import hashlib
import importlib
from pathlib import Path

from pilot_protocol import FLAGS, require
from tokenizer_measurement import measure
from tokenizer_runtime_guard import RuntimeGuard
from tokenizer_runtime_profile import inspect_runtime
from tokenizer_snapshot import verify_snapshot
from validation_contract import digest


class SessionError(ValueError):
    pass


def _raise_failure(error):
    if isinstance(error, (KeyboardInterrupt, SystemExit)):
        raise error from None
    raise SessionError("tokenizer session failed") from None


def _text_hash(value):
    require(type(value) is str and bool(value), "nonempty observed text")
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _observe(tokenizer):
    require(tokenizer.is_fast is True, "fast tokenizer required")
    template = tokenizer.chat_template
    vocabulary = tokenizer.get_vocab()
    require(type(vocabulary) is dict and bool(vocabulary), "observed vocabulary")
    require(all(type(name) is str and type(value) is int and value >= 0
                for name, value in vocabulary.items()), "observed vocabulary entries")
    special_map = copy.deepcopy(tokenizer.special_tokens_map)
    require(type(special_map) is dict, "special token map")
    mapped = []
    for name, value in special_map.items():
        require(type(name) is str, "special token name")
        values = value if type(value) is list else [value]
        require(all(type(token) is str and token in vocabulary for token in values),
                "special token strings")
        mapped.extend(values)
    tokens = tokenizer.all_special_tokens
    identifiers = tokenizer.all_special_ids
    require(type(tokens) is list and type(identifiers) is list, "special token lists")
    require(all(type(token) is str for token in tokens) and
            len(tokens) == len(set(tokens)) and set(tokens) == set(mapped),
            "complete special tokens")
    require(len(tokens) == len(identifiers) and
            all(type(value) is int and value == vocabulary[token]
                for token, value in zip(tokens, identifiers)), "special token IDs")
    for name in ("eos", "pad"):
        token = getattr(tokenizer, name + "_token")
        identifier = getattr(tokenizer, name + "_token_id")
        require(type(token) is str and special_map.get(name + "_token") == token,
                "required special token string")
        require(type(identifier) is int and vocabulary.get(token) == identifier,
                "required special token ID")
    settings = {
        "padding_side": tokenizer.padding_side, "truncation_side": tokenizer.truncation_side,
        "model_max_length": tokenizer.model_max_length,
        "split_special_tokens": tokenizer.split_special_tokens,
    }
    require(settings["padding_side"] in ("left", "right") and
            settings["truncation_side"] in ("left", "right"), "tokenizer sides")
    require(type(settings["model_max_length"]) is int and settings["model_max_length"] > 0,
            "model max length")
    require(type(settings["split_special_tokens"]) is bool, "split special tokens")
    state = {
        "template_sha256": _text_hash(template), "vocabulary_sha256": digest(vocabulary),
        "eos_token_id": tokenizer.eos_token_id, "pad_token_id": tokenizer.pad_token_id,
    }
    return {
        "kind": "TOKENIZER_OBSERVED_STATE_UNATTESTED",
        "class": type(tokenizer).__module__ + "." + type(tokenizer).__qualname__,
        "is_fast": True, "chat_template": template, **state,
        "measurement_state_sha256": digest(state),
        "special_tokens_map": special_map,
        "special_token_ids": dict(zip(tokens, identifiers)),
        "backend_sha256": _text_hash(tokenizer.backend_tokenizer.to_str()),
        "settings": settings,
    }


def _load_tokenizer(root):
    library = importlib.import_module("transformers")
    tokenizer = library.AutoTokenizer.from_pretrained(
        str(root / "snapshot"), use_fast=True, local_files_only=True, trust_remote_code=False)
    require(isinstance(tokenizer, library.PreTrainedTokenizerFast), "fast tokenizer class")
    return tokenizer


class _Session:
    def __init__(self, root, raw, approved, guard, tokenizer, runtime, snapshot):
        self._root = root
        self._raw = raw
        self._approved = approved
        self._guard = guard
        self._tokenizer = tokenizer
        self._runtime = copy.deepcopy(runtime)
        self._snapshot = copy.deepcopy(snapshot)
        self._identity = _observe(tokenizer)
        self._state = "ACTIVE"
        self._busy = False
        self._count = 0
        self._chain = digest([])
        self._stop = None

    def _check(self, heavy=False):
        self._guard.check()
        if heavy:
            require(inspect_runtime(self._raw, self._approved) == self._runtime,
                    "runtime metadata changed")
            require(verify_snapshot(self._root, self._raw, self._approved) == self._snapshot,
                    "snapshot changed")
        require(_observe(self._tokenizer) == self._identity, "observed tokenizer changed")
        self._guard.check()

    def _begin(self, allowed):
        require(self._state in allowed and not self._busy, "session state")
        self._busy = True

    def _invalidate(self):
        if self._state != "CLOSED":
            self._state = "FAILED"
        self._tokenizer = None
        self._guard.close()

    def measure(self, messages, answer=None):
        try:
            self._begin(("ACTIVE",))
            self._check()
            copied = copy.deepcopy(messages)
            original = digest(copied)
            measured = measure(self._tokenizer, copied, answer)
            self._check()
            require(digest(copied) == original and digest(messages) == original,
                    "session messages changed")
            require(measured["state_sha256"] == self._identity["measurement_state_sha256"],
                    "measurement state binding")
            require(measured["status"] in ("OK", "INPUT_LIMIT", "OUTPUT_LIMIT"),
                    "measurement status")
            record = {
                "kind": "TOKENIZER_SESSION_MEASUREMENT_PROVISIONAL",
                "run_id": self._snapshot["run_id"], "profile_sha256": self._approved,
                "identity_sha256": digest(self._identity), "ordinal": self._count,
                "measurement": measured,
            }
            self._chain = digest({"previous": self._chain, "record": record})
            self._count += 1
            if measured["status"] != "OK":
                self._state = "STOPPED"
                self._stop = measured["status"]
            return {**record, "chain_sha256": self._chain}
        except BaseException as error:
            self._invalidate()
            _raise_failure(error)
        finally:
            self._busy = False

    def finish(self):
        try:
            self._begin(("ACTIVE", "STOPPED"))
            self._check(heavy=True)
            result = {
                "kind": "TOKENIZER_SESSION_FINALIZATION",
                "run_id": self._snapshot["run_id"], "profile_sha256": self._approved,
                "identity": copy.deepcopy(self._identity),
                "runtime": copy.deepcopy(self._runtime), "snapshot": copy.deepcopy(self._snapshot),
                "measurements": self._count, "chain_sha256": self._chain,
                "stop_reason": self._stop, "tokenizer_loaded": True,
                "all_possible_second_rounds_validated": False,
                **{flag: False for flag in FLAGS},
            }
            self._tokenizer = None
            self._guard.close()
            self._state = "CLOSED"
            return result
        except BaseException as error:
            self._invalidate()
            _raise_failure(error)
        finally:
            self._busy = False

    def abort(self):
        if self._state in ("CLOSED", "FAILED", "ABORTED"):
            return
        self._state = "ABORTED"
        self._tokenizer = None
        self._guard.close()


def open_session(root, raw, approved_sha256):
    guard = RuntimeGuard()
    try:
        guard.start()
        runtime = inspect_runtime(raw, approved_sha256)
        snapshot = verify_snapshot(root, raw, approved_sha256)
        canonical_root = Path(root)
        tokenizer = _load_tokenizer(canonical_root)
        guard.check()
        session = _Session(canonical_root, raw, approved_sha256, guard, tokenizer,
                           runtime, snapshot)
        session._check(heavy=True)
        return session
    except BaseException as error:
        guard.close()
        _raise_failure(error)
```

- [ ] Run focused GREEN and the full suite. Planned counts are 17 focused and 106 total; adjust only for explicitly recorded regression additions, not removed cases.
- [ ] Check that the separate message-mutation, during-encoding backend drift, active-abort, class-membership, load-time metadata drift and swallowed-guard-failure tests exercise their own cause. The retained combined reentrancy/mutation example is not their substitute.
- [ ] Independently review complete session/test additions and their guard/profile/snapshot/neutral-measurement interfaces. Resolve concrete findings with retained RED/GREEN evidence.

## Combined Acceptance And Future Gates

- [ ] Main runs a fresh final-source suite and rechecks the complete protection manifest, branch/HEAD and tracked/index state.
- [ ] Verify importing both new modules performs no environment mutation, metadata enumeration, third-party import, snapshot read or load attempt.
- [ ] Verify real import dispatch is intercepted before a synthetic forbidden module body runs, cached modules are rejected and swallowed failure remains sticky. This is an import-policy probe, not installation or backend execution.
- [ ] Verify test fixtures never use installed tokenizer packages or real metadata. Logs must identify fixtures as synthetic and must not persist mocked finalizations as real audit evidence.
- [ ] Inspect full untracked additions with an independent whole-component reviewer, not empty ordinary Git diff.
- [ ] Record final source/test/document/review/probe hashes and all remaining gates in a fresh exclusive receipt and this stage's new ledger.
- [ ] Keep local/unpublished. No real launch, real profile approval, private transfer, tokenizer loading or task execution follows automatically.

| Parent obligation | This plan | Still required afterward |
| --- | --- | --- |
| Approved profile and five-file snapshot | Consume sealed checks | Real metadata discovery/approval |
| Disable model backends before import | One-shot guard and explicit settings | Target-version startup compatibility and wrapper enforcement |
| No externally injected trusted instance | Internal factory-only public entry | Trusted process/package provenance |
| Actual class, fast mode and measurement state | Observed identity and row binding | Real target-load observation |
| Drift and lifetime | Per-row identity/guard plus final snapshot/metadata checks | Controller rejects all unfinalized chains |
| Limits and stopped rows | Count existing neutral limits; STOPPED state | Ordered 324-row controller with full denominator and NOT_RUN rows |
| Private evidence | No logging, sanitized session errors | Private/public storage, transfer and derived-ID handling |
| Run cleanup and budgets | Guard release only, no retry | OS hard limits, deadline, process-group release and independent cleanup |
| Scientific benefit | No claim; all five flags false | Real generation/training and equal-information LoRA/text comparison |

The proposed implementation is not target-runtime compatibility evidence. In particular, fast-tokenizer serialization stability and tokenizer-only imports in transformers 4.57.1 must be observed in an admitted environment; failures stop the run rather than changing pins, importing models or relaxing guards.

## Plan Self-Review

Before implementation, parse all four Python blocks without executing them; check comments, one-letter identifiers, imports, exact public signatures and test inventory. Verify no source paths already exist and no sealed source or document changed.

Do not execute snippets or construct a synthetic runtime receipt as proof of real identity. This document is a plan. Its code and test counts are prospective until actual TDD execution and independent review.

## Execution Checkpoint

2026-11-12: both source/test pairs are implemented; Task 1 and Task 2 reviews passed.
The whole-component review found F1, a persistent identity change during the final
observation. Its bounded double-observation repair and five regressions are written.
Main's corrected-source suite passes 144 tests; static/import and preservation
checks pass. The earlier nested-open finding T2-R1 has an independent closure.

After two primary reviewer capacity failures and one alternate-route HTTP 401,
the user's renewed continuation led to a fresh successful independent re-review:
Spec PASS / Quality APPROVED, findings `[]`. F1 and T2-R1 are closed for their
concrete tested paths. Main's corrected-source 144-test run and final source-bound
cold-import probe pass; all 227 protected file hashes match.

This local component plan is complete. Final evidence is in
`/tmp/tlf-tokenizer-session-20261112-35gebi`, with `final-rereview.md`,
`main-final-fixed-tests.json`, `main-seal-import-probe.json` and the exclusive
`final-verification.json` receipt. The ledger records the full repair and provider
failure history. Two observations do not establish adversarial atomicity;
real runtime approval, the ordered 324-row controller, process wrapper/cleanup
and scientific evaluation remain separate gates. No real load or publication
was performed.
