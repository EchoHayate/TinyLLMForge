import copy
from functools import wraps
import hashlib
import re

from pilot_protocol import ARMS, FLAGS, fields, require
from tokenizer_inventory import REFERENCE_SHA256
from tokenizer_runtime_profile import validate_profile
from validation_contract import PINS, digest, encoded, parse_json


_FLAGS = (*FLAGS, "all_possible_second_rounds_validated")
_CATEGORIES = ("training", "first", "references", "sampled_second")
_ORDER = ("training",) * 8 + ("first",) * 136 + ("references",) * 44 + (
    "sampled_second",) * 136
_REASONS = {
    "OPEN": "OPEN_FAILED", "MEASURE": "MEASUREMENT_FAILED",
    "RECEIPT": "RECEIPT_INVALID", "FINALIZE": "FINALIZATION_FAILED",
}


def _public_validation(function):
    @wraps(function)
    def checked(*arguments, **keywords):
        try:
            return function(*arguments, **keywords)
        except Exception:
            raise ValueError("tokenizer audit evidence invalid") from None
    return checked


def _text(value, nonempty=False):
    require(type(value) is str and (not nonempty or bool(value)), "text")
    value.encode("utf-8")


def _primitive(value):
    if type(value) is dict:
        for name, item in value.items():
            _text(name)
            _primitive(item)
    elif type(value) is list:
        for item in value:
            _primitive(item)
    elif type(value) is str:
        _text(value)
    else:
        require(value is None or type(value) in (int, bool), "primitive evidence")


def _hex(value, length=64):
    require(type(value) is str and re.fullmatch("[0-9a-f]{" + str(length) + "}", value),
            "digest text")


def _integer(value, minimum=0):
    require(type(value) is int and value >= minimum, "integer")


def _ids(value, nonempty=True):
    require(type(value) is list and (not nonempty or bool(value)), "token list")
    for identifier in value:
        _integer(identifier)


def _flags(value):
    for name in _FLAGS:
        require(value[name] is False, "unadmitted flag")


def _text_hash(value):
    _text(value)
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _row(row):
    fields(row, ("ordinal", "category", "task_id", "arm", "messages", "answer",
                 "sample", "row_sha256"))
    _integer(row["ordinal"])
    require(row["ordinal"] < len(_ORDER) and row["category"] == _ORDER[row["ordinal"]],
            "category order")
    require(type(row["task_id"]) is str and
            re.fullmatch("[A-Za-z0-9_-]+", row["task_id"]), "safe task ID")
    category = row["category"]
    if category in ("training", "references"):
        require(row["arm"] is None, "answer arm")
        _text(row["answer"], nonempty=True)
    else:
        require(type(row["arm"]) is str and row["arm"] in ARMS, "episode arm")
        require(row["answer"] is None, "episode answer")
    if category == "sampled_second":
        fields(row["sample"], ("kind", "candidate_sha256", "feedback_sha256"))
        require(row["sample"]["kind"] == "INITIAL_SOURCE_SYNTHETIC_FEEDBACK",
                "sample kind")
        _hex(row["sample"]["candidate_sha256"])
        _hex(row["sample"]["feedback_sha256"])
    else:
        require(row["sample"] is None, "unexpected sample")
    require(type(row["messages"]) is list, "messages")
    roles = []
    for message in row["messages"]:
        fields(message, ("role", "content"))
        _text(message["role"])
        _text(message["content"])
        roles.append(message["role"])
    expected = ["user", "assistant", "user"] if category == "sampled_second" else ["user"]
    require(roles == expected, "message roles")
    _hex(row["row_sha256"])
    require(row["row_sha256"] == digest(
        {name: value for name, value in row.items() if name != "row_sha256"}), "row digest")


@_public_validation
def validate_inventory(raw, approved_sha256):
    _hex(approved_sha256)
    require(type(raw) is bytes and hashlib.sha256(raw).hexdigest() == approved_sha256,
            "inventory approval")
    inventory = parse_json(raw, limit=8388608)
    require(encoded(inventory) == raw, "canonical inventory bytes")
    fields(inventory, ("kind", "version", "data_sha256", "references_sha256", "rows", *_FLAGS))
    require(inventory["kind"] == "LORA_TOKENIZER_INVENTORY", "inventory kind")
    require(type(inventory["version"]) is int and inventory["version"] == 1, "version")
    require(inventory["data_sha256"] == PINS, "data pins")
    require(inventory["references_sha256"] == REFERENCE_SHA256, "reference pin")
    _flags(inventory)
    rows = inventory["rows"]
    require(type(rows) is list and len(rows) == len(_ORDER), "inventory denominator")
    for ordinal, row in enumerate(rows):
        _row(row)
        require(row["ordinal"] == ordinal, "row order")
    references = [{"id": row["task_id"], "reference_source": row["answer"]}
                  for row in rows if row["category"] == "references"]
    require(digest(references) == REFERENCE_SHA256, "reference projection")
    return inventory


@_public_validation
def validate_record(row, record, run_id, profile_sha256, previous_chain, identity_sha256=None):
    _primitive(row)
    _primitive(record)
    _row(row)
    _hex(run_id, 32)
    _hex(profile_sha256)
    _hex(previous_chain)
    if identity_sha256 is not None:
        _hex(identity_sha256)
    fields(record, ("kind", "run_id", "profile_sha256", "identity_sha256", "ordinal",
                    "measurement", "chain_sha256"))
    require(record["kind"] == "TOKENIZER_SESSION_MEASUREMENT_PROVISIONAL", "record kind")
    require(record["run_id"] == run_id and record["profile_sha256"] == profile_sha256,
            "session binding")
    _integer(record["ordinal"])
    require(record["ordinal"] == row["ordinal"], "record ordinal")
    _hex(record["identity_sha256"])
    require(identity_sha256 is None or record["identity_sha256"] == identity_sha256,
            "identity continuity")
    measurement = record["measurement"]
    fields(measurement, ("kind", "state_sha256", "status", "input", "target"))
    require(measurement["kind"] == "TOKENIZER_MEASUREMENT_UNATTESTED", "measurement kind")
    _hex(measurement["state_sha256"])
    observed = measurement["input"]
    fields(observed, ("messages_sha256", "prompt", "prompt_sha256", "input_ids",
                      "input_tokens", "remaining_input_tokens"))
    require(observed["messages_sha256"] == digest(row["messages"]), "messages binding")
    require(observed["prompt_sha256"] == _text_hash(observed["prompt"]), "prompt digest")
    _ids(observed["input_ids"])
    _integer(observed["input_tokens"])
    require(observed["input_tokens"] == len(observed["input_ids"]), "input count")
    require(type(observed["remaining_input_tokens"]) is int and
            observed["remaining_input_tokens"] == 4096 - observed["input_tokens"],
            "remaining input")
    target = measurement["target"]
    status = "OK"
    if observed["input_tokens"] > 4096:
        require(target is None, "input limit target")
        status = "INPUT_LIMIT"
    elif row["answer"] is None:
        require(target is None, "answerless target")
    else:
        fields(target, ("answer_sha256", "full_prompt_sha256", "full_sequence_ids",
                        "full_sequence_tokens", "target_start", "target_ids",
                        "target_tokens", "template_tail_ids"))
        require(target["answer_sha256"] == _text_hash(row["answer"]), "answer binding")
        _hex(target["full_prompt_sha256"])
        for name in ("full_sequence_ids", "target_ids", "template_tail_ids"):
            _ids(target[name], nonempty=name != "template_tail_ids")
        for name in ("full_sequence_tokens", "target_start", "target_tokens"):
            _integer(target[name])
        require(target["target_start"] == observed["input_tokens"], "target start")
        require(target["full_sequence_tokens"] == len(target["full_sequence_ids"]) and
                target["target_tokens"] == len(target["target_ids"]), "target counts")
        require(target["full_sequence_ids"] == observed["input_ids"] +
                target["target_ids"] + target["template_tail_ids"], "sequence partition")
        if target["target_tokens"] > 256:
            status = "OUTPUT_LIMIT"
    require(measurement["status"] == status, "measurement status")
    _hex(record["chain_sha256"])
    unsigned = {name: value for name, value in record.items() if name != "chain_sha256"}
    require(record["chain_sha256"] == digest({"previous": previous_chain, "record": unsigned}),
            "receipt chain")
    return copy.deepcopy(record)


def _identity(identity):
    fields(identity, ("kind", "class", "is_fast", "chat_template", "template_sha256",
                      "vocabulary_sha256", "eos_token_id", "pad_token_id",
                      "measurement_state_sha256", "special_tokens_map", "special_token_ids",
                      "backend_sha256", "settings"))
    require(identity["kind"] == "TOKENIZER_OBSERVED_STATE_UNATTESTED", "identity kind")
    require(identity["is_fast"] is True, "fast tokenizer")
    _text(identity["class"], nonempty=True)
    _text(identity["chat_template"], nonempty=True)
    require(identity["template_sha256"] == _text_hash(identity["chat_template"]),
            "template digest")
    for name in ("vocabulary_sha256", "backend_sha256"):
        _hex(identity[name])
    for name in ("eos_token_id", "pad_token_id"):
        _integer(identity[name])
    state = {name: identity[name] for name in
             ("template_sha256", "vocabulary_sha256", "eos_token_id", "pad_token_id")}
    require(identity["measurement_state_sha256"] == digest(state), "state digest")
    special_map = identity["special_tokens_map"]
    special_ids = identity["special_token_ids"]
    require(type(special_map) is dict and type(special_ids) is dict, "special token mappings")
    mapped = set()
    for name, value in special_map.items():
        _text(name)
        values = value if type(value) is list else [value]
        for token in values:
            _text(token)
            mapped.add(token)
    require(set(special_ids) == mapped, "complete special IDs")
    for token, identifier in special_ids.items():
        _text(token)
        _integer(identifier)
    for name in ("eos", "pad"):
        token = special_map.get(name + "_token")
        require(type(token) is str and special_ids.get(token) == identity[name + "_token_id"],
                "required special token")
    settings = identity["settings"]
    fields(settings, ("padding_side", "truncation_side", "model_max_length",
                      "split_special_tokens"))
    require(settings["padding_side"] in ("left", "right") and
            settings["truncation_side"] in ("left", "right"), "tokenizer sides")
    _integer(settings["model_max_length"], minimum=1)
    require(type(settings["split_special_tokens"]) is bool, "special token setting")


def _finalization(finalization, profile, approved, records, chain, stop):
    _primitive(finalization)
    fields(finalization, ("kind", "run_id", "profile_sha256", "identity", "runtime",
                          "snapshot", "measurements", "chain_sha256", "stop_reason",
                          "tokenizer_loaded", *_FLAGS))
    require(finalization["kind"] == "TOKENIZER_SESSION_FINALIZATION", "finalization kind")
    require(finalization["run_id"] == profile["run_id"] and
            finalization["profile_sha256"] == approved, "finalization binding")
    _integer(finalization["measurements"])
    require(finalization["measurements"] == len(records) and
            finalization["chain_sha256"] == chain and finalization["stop_reason"] == stop,
            "finalization transcript")
    require(finalization["tokenizer_loaded"] is True, "tokenizer loaded")
    _flags(finalization)
    identity = finalization["identity"]
    _identity(identity)
    for record in records:
        require(record["identity_sha256"] == digest(identity) and
                record["measurement"]["state_sha256"] == identity["measurement_state_sha256"],
                "observed identity binding")
        target = record["measurement"]["target"]
        if target is not None:
            eos = identity["eos_token_id"]
            require(target["target_ids"][-1] == eos and target["target_ids"].count(eos) == 1 and
                    eos not in target["template_tail_ids"], "unique target EOS")
    runtime = finalization["runtime"]
    fields(runtime, ("kind", "run_id", "profile_sha256", "versions", "tokenizer_loaded", *_FLAGS))
    require(runtime["kind"] == "TOKENIZER_RUNTIME_METADATA_ONLY" and
            runtime["run_id"] == profile["run_id"] and runtime["profile_sha256"] == approved and
            runtime["versions"] == profile["versions"], "runtime profile binding")
    require(runtime["tokenizer_loaded"] is False, "runtime metadata only")
    _flags(runtime)
    snapshot = finalization["snapshot"]
    fields(snapshot, ("kind", "run_id", "profile_sha256", "revision_declaration",
                      "file_sha256", "file_bytes", "tokenizer_loaded", *_FLAGS))
    require(snapshot["kind"] == "TOKENIZER_FILE_SNAPSHOT_ONLY" and
            snapshot["run_id"] == profile["run_id"] and
            snapshot["profile_sha256"] == approved and
            snapshot["revision_declaration"] == profile["revision"] and
            snapshot["file_sha256"] == profile["file_sha256"], "snapshot profile binding")
    require(snapshot["tokenizer_loaded"] is False, "snapshot files only")
    _flags(snapshot)
    fields(snapshot["file_bytes"], profile["file_sha256"])
    for count in snapshot["file_bytes"].values():
        _integer(count)
    require(sum(snapshot["file_bytes"].values()) <= 128 * 1024 * 1024, "snapshot byte ceiling")


def _failure(failure, abort_failed, records, finalization, stop):
    require(type(abort_failed) is bool, "abort result")
    if failure is None:
        require(not abort_failed and finalization is not None, "successful termination")
        require(len(records) == len(_ORDER) or stop is not None, "incomplete transcript")
        return None
    fields(failure, ("phase", "ordinal"))
    phase = failure["phase"]
    require(type(phase) is str and phase in _REASONS, "failure phase")
    require(finalization is None, "failed finalization")
    ordinal = failure["ordinal"]
    if phase == "OPEN":
        require(ordinal is None and not records and not abort_failed, "open failure")
    elif phase in ("MEASURE", "RECEIPT"):
        _integer(ordinal)
        require(ordinal == len(records) < len(_ORDER) and stop is None, "row failure")
    else:
        require(ordinal is None and (len(records) == len(_ORDER) or stop is not None),
                "finalize failure")
    return {"phase": phase, "ordinal": ordinal, "reason": _REASONS[phase]}


@_public_validation
def build_report(inventory_raw, approved_inventory_sha256, profile_raw,
                 approved_profile_sha256, records, finalization, failure=None,
                 abort_failed=False, open_diagnostic=None):
    inventory = validate_inventory(inventory_raw, approved_inventory_sha256)
    profile = validate_profile(profile_raw, approved_profile_sha256)
    require(type(records) is list and len(records) <= len(_ORDER), "records")
    checked = []
    chain = digest([])
    identity_sha256 = None
    state_sha256 = None
    stop = None
    stop_ordinal = None
    for row, record in zip(inventory["rows"], records):
        require(stop is None, "record after stop")
        record = validate_record(row, record, profile["run_id"], approved_profile_sha256,
                                 chain, identity_sha256)
        identity_sha256 = record["identity_sha256"]
        measurement = record["measurement"]
        require(state_sha256 is None or state_sha256 == measurement["state_sha256"],
                "state continuity")
        state_sha256 = measurement["state_sha256"]
        chain = record["chain_sha256"]
        checked.append(record)
        if measurement["status"] != "OK":
            stop, stop_ordinal = measurement["status"], row["ordinal"]
    public_failure = _failure(failure, abort_failed, checked, finalization, stop)
    if open_diagnostic is not None:
        require(public_failure is not None and public_failure["phase"] == "OPEN",
                "diagnostic requires open failure")
        fields(open_diagnostic, ("stage", "error_type", "guard_failed"))
        require(type(open_diagnostic["stage"]) is str and open_diagnostic["stage"] in (
            "GUARD_START", "RUNTIME_METADATA", "SNAPSHOT_VERIFY", "TOKENIZER_LOAD",
            "OBSERVE", "FINAL_CHECK"), "diagnostic stage")
        require(type(open_diagnostic["error_type"]) is str and open_diagnostic["error_type"] in (
            "ImportError", "ModuleNotFoundError", "ValueError", "RuntimeError",
            "OSError", "TypeError", "MemoryError", "OTHER"), "diagnostic error type")
        require(type(open_diagnostic["guard_failed"]) is bool, "diagnostic guard state")
    if public_failure is None:
        _finalization(finalization, profile, approved_profile_sha256, checked, chain, stop)
    if stop is None and public_failure is not None:
        stop, stop_ordinal = public_failure["reason"], public_failure["ordinal"]
    rows = []
    by_category = {category: {"scheduled": 0, "checked": 0, "not_checked": 0, "over_limit": 0}
                   for category in _CATEGORIES}
    for planned in inventory["rows"]:
        ordinal = planned["ordinal"]
        measurement = checked[ordinal]["measurement"] if ordinal < len(checked) else None
        status = "NOT_RUN_AFTER_STOP"
        if measurement is not None:
            status = measurement["status"]
        elif public_failure is not None and public_failure["ordinal"] == ordinal:
            status = public_failure["reason"]
        target = None if measurement is None else measurement["target"]
        rows.append({
            **{name: planned[name] for name in
               ("ordinal", "category", "task_id", "arm", "row_sha256")},
            "status": status,
            "input_tokens": None if measurement is None else measurement["input"]["input_tokens"],
            "target_tokens": None if target is None else target["target_tokens"],
            "full_sequence_tokens": None if target is None else target["full_sequence_tokens"],
        })
        counts = by_category[planned["category"]]
        counts["scheduled"] += 1
        counts["checked" if measurement is not None else "not_checked"] += 1
        counts["over_limit"] += int(status in ("INPUT_LIMIT", "OUTPUT_LIMIT"))
    public = {
        "kind": "LORA_TOKENIZER_AUDIT_PENDING_CLEANUP",
        "run_id": profile["run_id"], "profile_sha256": approved_profile_sha256,
        "inventory_sha256": approved_inventory_sha256, "rows": rows, "by_category": by_category,
        "stop_reason": stop, "stop_ordinal": stop_ordinal, "failure": public_failure,
        "abort_failed": abort_failed, "session_finalized": public_failure is None,
        "all_rows_within_budget": len(checked) == len(_ORDER) and stop is None,
        "checked_inventory_within_budget": False, "cleanup_status": "CLEANUP_UNVERIFIED",
        **{name: False for name in _FLAGS},
    }
    if open_diagnostic is not None:
        public["open_diagnostic"] = copy.deepcopy(open_diagnostic)
    return {"public": public, "private": {
        "inventory": inventory, "records": checked, "finalization": copy.deepcopy(finalization),
    }}
