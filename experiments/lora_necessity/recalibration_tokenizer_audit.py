import copy

from pilot_protocol import require
from recalibration_tokenizer_report import (
    _validate_record, build_synthetic_report, prepare_synthetic_inputs,
)
from tokenizer_audit_report import _public_validation
from validation_contract import digest


@_public_validation
def run_synthetic_audit(validation_inputs, profile_raw, approved_profile_sha256, *, session_factory):
    """Coordinate explicitly constructed fake sessions, never a real opener.

    The caller-supplied factory is not an authentication or sandbox boundary.
    """
    inventory, profile = prepare_synthetic_inputs(
        validation_inputs, profile_raw, approved_profile_sha256)
    require(callable(session_factory), "explicit synthetic session factory")
    frozen_inputs = copy.deepcopy(validation_inputs)
    if inventory["status"] == "DEPENDENCY_UNSATISFIED":
        return build_synthetic_report(
            frozen_inputs, profile_raw, approved_profile_sha256, [], None)
    session, phase, ordinal = None, "OPEN", None
    records, chain, identity, state = [], digest([]), None, None
    try:
        session = session_factory(profile_raw, approved_profile_sha256)
        for row in inventory["rows"]:
            phase, ordinal = "MEASURE", row["ordinal"]
            messages = copy.deepcopy(row["messages"])
            record = session.measure(messages, row["answer"])
            phase = "RECEIPT"
            require(digest(messages) == digest(row["messages"]), "messages mutated")
            record = _validate_record(
                row, record, profile["run_id"], approved_profile_sha256, chain, identity, state)
            chain, identity = record["chain_sha256"], record["identity_sha256"]
            state = record["measurement"]["state_sha256"]
            records.append(record)
            if record["measurement"]["status"] != "OK":
                break
        phase, ordinal = "FINALIZE", None
        finalization = session.finish()
        return build_synthetic_report(
            frozen_inputs, profile_raw, approved_profile_sha256, records, finalization)
    except BaseException as error:
        abort_failed = False
        if session is not None:
            try:
                session.abort()
            except BaseException as abort_error:
                if isinstance(error, Exception) and not isinstance(abort_error, Exception):
                    raise
                abort_failed = True
        if not isinstance(error, Exception):
            raise
        return build_synthetic_report(
            frozen_inputs, profile_raw, approved_profile_sha256, records, None,
            failure={"phase": phase, "ordinal": ordinal}, abort_failed=abort_failed)
