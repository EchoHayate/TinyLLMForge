import copy

from tokenizer_audit_report import build_report, validate_inventory, validate_record
from tokenizer_runtime_profile import validate_profile
import tokenizer_session
from validation_contract import digest


def run_audit(root, profile_raw, approved_profile_sha256,
              inventory_raw, approved_inventory_sha256):
    try:
        inventory = validate_inventory(inventory_raw, approved_inventory_sha256)
        profile = validate_profile(profile_raw, approved_profile_sha256)
    except Exception:
        raise ValueError("tokenizer audit evidence invalid") from None

    session = None
    records = []
    chain = digest([])
    identity_sha256 = None
    state_sha256 = None
    phase = "OPEN"
    ordinal = None
    try:
        session = tokenizer_session.open_session(root, profile_raw, approved_profile_sha256)
        for row in inventory["rows"]:
            phase = "MEASURE"
            ordinal = row["ordinal"]
            record = session.measure(copy.deepcopy(row["messages"]), row["answer"])
            phase = "RECEIPT"
            record = validate_record(
                row, record, profile["run_id"], approved_profile_sha256, chain, identity_sha256)
            measurement = record["measurement"]
            if state_sha256 is not None and state_sha256 != measurement["state_sha256"]:
                raise ValueError("tokenizer audit evidence invalid")
            state_sha256 = measurement["state_sha256"]
            identity_sha256 = record["identity_sha256"]
            chain = record["chain_sha256"]
            records.append(record)
            if measurement["status"] != "OK":
                break
        phase = "FINALIZE"
        ordinal = None
        finalization = session.finish()
        return build_report(
            inventory_raw, approved_inventory_sha256, profile_raw, approved_profile_sha256,
            records, finalization)
    except BaseException as error:
        abort_failed = False
        if session is not None:
            try:
                session.abort()
            except BaseException as cleanup_error:
                if isinstance(error, Exception) and not isinstance(cleanup_error, Exception):
                    raise
                abort_failed = True
        if not isinstance(error, Exception):
            raise
        diagnostic = None
        if phase == "OPEN" and isinstance(error, tokenizer_session.SessionError):
            diagnostic = error.open_diagnostic
        return build_report(
            inventory_raw, approved_inventory_sha256, profile_raw, approved_profile_sha256,
            records, None, failure={"phase": phase, "ordinal": ordinal},
            abort_failed=abort_failed, open_diagnostic=diagnostic)
