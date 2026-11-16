import copy
import hashlib
import importlib
from pathlib import Path
import threading
import weakref

from pilot_protocol import FLAGS, require
from tokenizer_measurement import _VocabularyDigest, measure
from tokenizer_runtime_guard import RuntimeGuard
from tokenizer_runtime_profile import inspect_runtime
from tokenizer_snapshot import verify_snapshot
from validation_contract import digest


_OWNER = None


class SessionError(ValueError):
    open_diagnostic = None


def _raise_failure(error, open_diagnostic=None):
    if not isinstance(error, Exception):
        raise error from None
    failure = SessionError("tokenizer session failed")
    failure.open_diagnostic = open_diagnostic
    raise failure from None


def _text_hash(value):
    require(type(value) is str and bool(value), "nonempty observed text")
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _observe(tokenizer, vocabulary_digest=None):
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
        "template_sha256": _text_hash(template),
        "vocabulary_sha256": (vocabulary_digest or digest)(vocabulary),
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
        self._vocabulary_digest = _VocabularyDigest()
        self._identity = _observe(tokenizer, self._vocabulary_digest)
        self._state = "ACTIVE"
        self._busy = False
        self._count = 0
        self._chain = digest([])
        self._stop = None

    def _check_files(self):
        require(inspect_runtime(self._raw, self._approved) == self._runtime,
                "runtime metadata changed")
        require(verify_snapshot(self._root, self._raw, self._approved) == self._snapshot,
                "snapshot changed")

    def _check(self, heavy=False):
        self._guard.check()
        observed = _observe(self._tokenizer, self._vocabulary_digest)
        require(observed == self._identity, "observed tokenizer changed")
        self._guard.check()
        require(_observe(self._tokenizer, self._vocabulary_digest) == observed,
                "tokenizer observation unstable")
        self._guard.check()
        if heavy:
            self._check_files()
            self._guard.check()

    def _begin(self, allowed):
        require(self._state in allowed and not self._busy, "session state")
        self._busy = True

    def _invalidate(self):
        if self._state != "CLOSED":
            self._state = "FAILED"
        self._tokenizer = None
        self._vocabulary_digest = None
        self._guard.close()

    def measure(self, messages, answer=None):
        entered = False
        try:
            self._begin(("ACTIVE",))
            entered = True
            copied = copy.deepcopy(messages)
            original = digest(copied)
            self._check()
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
            if entered:
                self._busy = False

    def finish(self):
        entered = False
        try:
            self._begin(("ACTIVE", "STOPPED"))
            entered = True
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
            self._vocabulary_digest = None
            self._guard.check()
            self._check_files()
            self._guard.check()
            require(self._state in ("ACTIVE", "STOPPED"), "session invalidated during release")
            self._guard.close()
            self._state = "CLOSED"
            return result
        except BaseException as error:
            self._invalidate()
            _raise_failure(error)
        finally:
            if entered:
                self._busy = False

    def abort(self):
        if self._state in ("CLOSED", "FAILED", "ABORTED") and not self._busy:
            return
        entered = False
        try:
            self._begin(("ACTIVE", "STOPPED"))
            entered = True
            self._guard.check()
            self._state = "ABORTED"
            self._tokenizer = None
            self._vocabulary_digest = None
            self._guard.close()
            require(self._state == "ABORTED", "session invalidated during release")
        except BaseException as error:
            self._invalidate()
            _raise_failure(error)
        finally:
            if entered:
                self._busy = False


def open_session(root, raw, approved_sha256):
    global _OWNER
    guard = RuntimeGuard()
    session = None
    tokenizer = None
    stage = "GUARD_START"
    try:
        guard.start()
        _OWNER = (weakref.ref(guard), None)
        stage = "RUNTIME_METADATA"
        runtime = inspect_runtime(raw, approved_sha256)
        stage = "SNAPSHOT_VERIFY"
        snapshot = verify_snapshot(root, raw, approved_sha256)
        canonical_root = Path(root)
        stage = "TOKENIZER_LOAD"
        tokenizer = _load_tokenizer(canonical_root)
        guard.check()
        stage = "OBSERVE"
        session = _Session(canonical_root, raw, approved_sha256, guard, tokenizer,
                           runtime, snapshot)
        _OWNER = (weakref.ref(guard), weakref.ref(session))
        stage = "FINAL_CHECK"
        session._check(heavy=True)
        return session
    except BaseException as error:
        error_types = {
            ImportError: "ImportError", ModuleNotFoundError: "ModuleNotFoundError",
            ValueError: "ValueError", RuntimeError: "RuntimeError", OSError: "OSError",
            TypeError: "TypeError", MemoryError: "MemoryError",
        }
        diagnostic = {
            "stage": stage, "error_type": error_types.get(type(error), "OTHER"),
            "guard_failed": guard.failed,
        }
        if threading.current_thread() is threading.main_thread() and _OWNER is not None:
            owner_guard, owner_session = _OWNER
            owner_session = owner_session() if owner_session is not None else None
            if owner_session is not None and (
                    owner_session._state in ("ACTIVE", "STOPPED") or owner_session._busy):
                owner_session._invalidate()
            owner_guard = owner_guard()
            if owner_guard is not None:
                owner_guard.close()
        if session is not None:
            session._invalidate()
        tokenizer = None
        guard.close()
        _raise_failure(error, diagnostic)
