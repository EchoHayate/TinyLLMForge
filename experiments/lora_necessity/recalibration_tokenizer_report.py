import copy

from pilot_protocol import FLAGS, fields, require
from recalibration_data_contract import _plain
from recalibration_tokenizer_inventory import PLANNED_COUNTS, build_synthetic_inventory
from tokenizer_audit_report import _finalization, _hex, _ids, _integer, _public_validation, _text_hash
from tokenizer_runtime_profile import validate_profile
from validation_contract import digest


REAL_FLAGS = (*FLAGS, "all_possible_second_rounds_validated", "tokenizer_length_validated",
              "execution_provenance_verified", "real_dynamic_validation_complete",
              "run_authorized", "tokenizer_dispatch_authorized")
INPUT_FIELDS = ("raw_files", "worker_raw", "run_ids", "records", "rendering_raw", "expected_runtime")
REASONS = {"OPEN": "OPEN_FAILED", "MEASURE": "MEASUREMENT_FAILED",
           "RECEIPT": "RECEIPT_INVALID", "FINALIZE": "FINALIZATION_FAILED"}


@_public_validation
def prepare_synthetic_inputs(validation_inputs, profile_raw, approved_profile_sha256):
    fields(validation_inputs, INPUT_FIELDS)
    require(all(type(name) is str for name in validation_inputs), "input keys")
    inventory = build_synthetic_inventory(**validation_inputs)
    profile = validate_profile(profile_raw, approved_profile_sha256)
    return inventory, profile


def _measurement(row, measurement):
    fields(measurement, ("kind", "state_sha256", "status", "input", "target"))
    require(measurement["kind"] == "TOKENIZER_MEASUREMENT_UNATTESTED", "measurement kind")
    _hex(measurement["state_sha256"])
    observed = measurement["input"]
    fields(observed, ("messages_sha256", "prompt", "prompt_sha256", "input_ids",
                      "input_tokens", "remaining_input_tokens"))
    require(observed["messages_sha256"] == digest(row["messages"]), "message binding")
    require(observed["prompt_sha256"] == _text_hash(observed["prompt"]), "prompt binding")
    _ids(observed["input_ids"])
    _integer(observed["input_tokens"], minimum=1)
    require(observed["input_tokens"] == len(observed["input_ids"]), "input count")
    require(type(observed["remaining_input_tokens"]) is int and
            observed["remaining_input_tokens"] == 4096 - observed["input_tokens"], "remaining input")
    target = measurement["target"]
    status = "INPUT_LIMIT" if observed["input_tokens"] > 4096 else "OK"
    if status == "INPUT_LIMIT" or row["answer"] is None:
        require(target is None, "unexpected target")
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
        require(target["target_start"] == observed["input_tokens"], "target boundary")
        require(target["full_sequence_tokens"] == len(target["full_sequence_ids"]) and
                target["target_tokens"] == len(target["target_ids"]), "target counts")
        require(target["full_sequence_ids"] == observed["input_ids"] + target["target_ids"] +
                target["template_tail_ids"], "sequence partition")
        if target["target_tokens"] > 256:
            status = "OUTPUT_LIMIT"
    require(measurement["status"] == status, "measurement status")


@_public_validation
def _validate_record(row, record, run_id, profile_sha256, previous_chain,
                     identity_sha256=None, state_sha256=None):
    """Validate a receipt against a row reconstructed by prepare_synthetic_inputs."""
    _plain(row)
    _plain(record)
    _hex(run_id, 32)
    for value in (profile_sha256, previous_chain):
        _hex(value)
    for value in (identity_sha256, state_sha256):
        if value is not None:
            _hex(value)
    fields(record, ("kind", "run_id", "profile_sha256", "identity_sha256",
                    "ordinal", "measurement", "chain_sha256"))
    require(record["kind"] == "TOKENIZER_SESSION_MEASUREMENT_PROVISIONAL", "record kind")
    require(record["run_id"] == run_id and record["profile_sha256"] == profile_sha256,
            "session binding")
    _integer(record["ordinal"])
    require(record["ordinal"] == row["ordinal"], "record position")
    _hex(record["identity_sha256"])
    require(identity_sha256 is None or record["identity_sha256"] == identity_sha256,
            "identity continuity")
    _measurement(row, record["measurement"])
    require(state_sha256 is None or record["measurement"]["state_sha256"] == state_sha256,
            "state continuity")
    unsigned = {name: value for name, value in record.items() if name != "chain_sha256"}
    require(record["chain_sha256"] == digest({"previous": previous_chain, "record": unsigned}),
            "record chain")
    return copy.deepcopy(record)


def _failure(failure, abort_failed, count, finalization, stop):
    require(type(abort_failed) is bool, "abort flag")
    if failure is None:
        require(not abort_failed and finalization is not None, "missing finalization")
        require(count == 120 or stop is not None, "incomplete measurements")
        return None
    _plain(failure)
    fields(failure, ("phase", "ordinal"))
    phase, ordinal = failure["phase"], failure["ordinal"]
    require(type(phase) is str and phase in REASONS and finalization is None, "failure phase")
    if phase == "OPEN":
        require(ordinal is None and count == 0 and not abort_failed, "open failure")
    elif phase == "FINALIZE":
        require(ordinal is None and (count == 120 or stop is not None), "finish failure")
    else:
        require(type(ordinal) is int and ordinal == count < 120 and stop is None, "row failure")
    return {"phase": phase, "ordinal": ordinal, "reason": REASONS[phase]}


@_public_validation
def build_synthetic_report(validation_inputs, profile_raw, approved_profile_sha256,
                           records, finalization, *, failure=None, abort_failed=False):
    inventory, profile = prepare_synthetic_inputs(
        validation_inputs, profile_raw, approved_profile_sha256)
    _plain(records)
    _plain(finalization)
    require(type(records) is list and len(records) <= 120 and type(abort_failed) is bool,
            "record denominator")
    ready = inventory["status"] == "SYNTHETIC_ONLY"
    checked, chain, identity, state, stop, stop_ordinal = [], digest([]), None, None, None, None
    public_failure = None
    if not ready:
        require(not records and finalization is None and failure is None and not abort_failed,
                "unsatisfied dependency evidence")
    else:
        for row, record in zip(inventory["rows"], records):
            require(stop is None, "measurement after stop")
            record = _validate_record(row, record, profile["run_id"], approved_profile_sha256,
                                      chain, identity, state)
            chain, identity = record["chain_sha256"], record["identity_sha256"]
            state = record["measurement"]["state_sha256"]
            checked.append(record)
            if record["measurement"]["status"] != "OK":
                stop, stop_ordinal = record["measurement"]["status"], row["ordinal"]
        public_failure = _failure(failure, abort_failed, len(checked), finalization, stop)
        if public_failure is None:
            _finalization(finalization, profile, approved_profile_sha256, checked, chain, stop)
        elif stop is None:
            stop, stop_ordinal = public_failure["reason"], public_failure["ordinal"]
    counts = {name: {"scheduled": total, "checked": 0, "not_checked": total, "over_limit": 0}
              for name, total in PLANNED_COUNTS.items()}
    rows = []
    for row in inventory["rows"]:
        ordinal = row["ordinal"]
        measurement = checked[ordinal]["measurement"] if ordinal < len(checked) else None
        status = "NOT_RUN_AFTER_STOP"
        if measurement is not None:
            status = measurement["status"]
        elif public_failure is not None and public_failure["ordinal"] == ordinal:
            status = public_failure["reason"]
        target = None if measurement is None else measurement["target"]
        rows.append({
            **{name: row[name] for name in ("ordinal", "category", "task_id", "arm",
                                          "reference_id", "branch", "row_sha256")},
            "status": status,
            "input_tokens": None if measurement is None else measurement["input"]["input_tokens"],
            "target_tokens": None if target is None else target["target_tokens"],
            "full_sequence_tokens": None if target is None else target["full_sequence_tokens"],
        })
        if measurement is not None:
            counts[row["category"]]["checked"] += 1
            counts[row["category"]]["not_checked"] -= 1
        counts[row["category"]]["over_limit"] += int(status in ("INPUT_LIMIT", "OUTPUT_LIMIT"))
    status = "DEPENDENCY_UNSATISFIED"
    if ready:
        status = "SYNTHETIC_FAILED" if public_failure else (
            "SYNTHETIC_STOPPED" if stop else "SYNTHETIC_COMPLETE")
    public = {
        "kind": "LORA_RECALIBRATION_TOKENIZER_REPORT", "version": 1,
        "data_version": "recalibration_v2_r1", "status": status,
        "synthetic_only": True, "evidence_kind": "synthetic",
        "run_id": profile["run_id"], "profile_sha256": approved_profile_sha256,
        "inventory_sha256": inventory["inventory_sha256"],
        "validation_evidence_sha256": inventory["dynamic_evidence_sha256"],
        "planned": 120, "checked": len(checked), "not_checked": 120 - len(checked),
        "over_limit": sum(count["over_limit"] for count in counts.values()),
        "rows": rows, "by_category": counts, "stop_reason": stop, "stop_ordinal": stop_ordinal,
        "failure": public_failure, "abort_failed": abort_failed,
        "session_finalized": ready and public_failure is None,
        "synthetic_all_rows_within_budget": ready and public_failure is None and
                                           len(checked) == 120 and stop is None,
        **{flag: False for flag in REAL_FLAGS},
    }
    return {"public": public, "private": {
        "inventory": inventory, "records": checked, "finalization": copy.deepcopy(finalization)}}
