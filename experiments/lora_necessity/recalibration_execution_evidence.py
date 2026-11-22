import copy
import hashlib
import re

from pilot_protocol import FLAGS, fields, require
from recalibration_data_contract import _plain
from recalibration_execution_package import BACKEND_SHA256, validate_preparation
from recalibration_execution_wire import FRAME_NAMES, HEADER, MAGIC, decode_wire
from recalibration_validation import WORKER_SHA256, build_report
from validation_contract import digest, encoded, parse_json
from validation_real_report import _process_result


READY_FIELDS = (
    "kind", "version", "stage_id", "batch_index", "run_id", "preparation_sha256",
    "package_sha256", "config_sha256", "bundle_sha256", "backend_sha256",
    "worker_sha256", "hostname", "uid", "stage_root", "batch_root",
    "expected_runtime_sha256",
)
DONE_FIELDS = (
    "kind", "version", "stage_id", "batch_index", "run_id", "package_sha256",
    "config_sha256", "ready_sha256", "payload_sha256", "prior_frames_sha256",
    "status", "payload_received", "backend_pid", "backend", "work_elapsed_ns",
    "cleanup_elapsed_ns",
)
PROCESS_FIELDS = ("status", "returncode", "elapsed_ns", "process_reaped", "stdin_complete")
RUNTIME_FIELDS = (
    "hostname", "uid", "python", "python_root", "python_sha256", "bwrap_sha256",
    "seccomp_sha256", "argv_prefix", "no_new_privs", "host_namespaces",
)
MISSING_BINDINGS = (
    "full_launch_package", "live_preflight", "actual_controller_and_claims",
    "independent_collection_acceptance",
)
FALSE_FLAGS = (
    *FLAGS, "binding_complete", "trusted_collection_accepted", "dynamic_checks_passed",
    "launch_ready", "run_authorized", "tokenizer_length_validated",
)


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _hex(value, size):
    require(type(value) is str and re.fullmatch("[0-9a-f]{%d}" % size, value) is not None,
            "hex identity")


def _integer(value, minimum=0):
    require(type(value) is int and value >= minimum, "integer range")


def _text(value):
    require(type(value) is str and 0 < len(value) <= 4096, "nonempty bounded text")


def _stage_root(value, stage_id):
    require(type(value) is str and len(value) <= 256 and value.startswith("/"),
            "stage root")
    components = value[1:].split("/")
    require(all(re.fullmatch("[A-Za-z0-9_.-]+", part) is not None
                and part not in (".", "..") for part in components)
            and components[-1] == stage_id, "canonical stage root")


def _expected(expected):
    _plain(expected)
    fields(expected, READY_FIELDS)
    require(expected["kind"] == "LORA_RECALIBRATION_READY"
            and type(expected["version"]) is int and expected["version"] == 1,
            "ready kind/version")
    for name in ("stage_id", "run_id"):
        _hex(expected[name], 32)
    require(expected["stage_id"] != expected["run_id"], "distinct stage/run IDs")
    _integer(expected["batch_index"])
    require(expected["batch_index"] < 15, "batch index")
    for name in READY_FIELDS:
        if name.endswith("_sha256"):
            _hex(expected[name], 64)
    require(expected["backend_sha256"] == BACKEND_SHA256
            and expected["worker_sha256"] == WORKER_SHA256, "fixed execution sources")
    _text(expected["hostname"])
    _integer(expected["uid"])
    _stage_root(expected["stage_root"], expected["stage_id"])
    require(expected["batch_root"] == expected["stage_root"]
            + "/batch-%02d-" % expected["batch_index"] + expected["run_id"],
            "batch root")


def validate_ready(ready_raw, *, expected):
    _expected(expected)
    observed = parse_json(ready_raw, limit=4096)
    _plain(observed)
    fields(observed, (*READY_FIELDS, "ready_elapsed_ns"))
    _integer(observed["ready_elapsed_ns"])
    require(observed["ready_elapsed_ns"] <= 10_000_000_000, "READY deadline")
    actual = {name: observed[name] for name in READY_FIELDS}
    _expected(actual)
    require(encoded(actual) == encoded(expected), "READY context binding")
    return observed


def decode_receipt(wire_raw, *, expected_sha256, expected):
    decoded = decode_wire(wire_raw, expected_sha256=expected_sha256)
    ready = validate_ready(decoded["ready_raw"], expected=expected)
    done = parse_json(decoded["done_raw"], limit=4096)
    _plain(done)
    fields(done, DONE_FIELDS)
    require(done["kind"] == "LORA_RECALIBRATION_DONE"
            and type(done["version"]) is int and done["version"] == 1,
            "DONE kind/version")
    for name in ("stage_id", "batch_index", "run_id", "package_sha256", "config_sha256"):
        require(encoded(done[name]) == encoded(ready[name]), "DONE identity binding")
    require(done["ready_sha256"] == _sha(decoded["ready_raw"]), "DONE READY digest")
    prior = [
        _sha(HEADER.pack(MAGIC, channel, len(decoded[name])) + decoded[name])
        for channel, name in enumerate(FRAME_NAMES[:4], 1)
    ]
    require(type(done["prior_frames_sha256"]) is list
            and done["prior_frames_sha256"] == prior, "DONE prior frame digests")
    require(type(done["payload_received"]) is bool, "payload received flag")
    require(done["payload_sha256"] == (
        ready["bundle_sha256"] if done["payload_received"] else None), "payload digest")
    for name in ("work_elapsed_ns", "cleanup_elapsed_ns"):
        _integer(done[name])
    require(type(done["status"]) is str and done["status"] in (
        "completed", "program_error", "timeout", "output_limit", "launch_failed",
        "input_error", "cancelled"), "supervisor status")
    backend = done["backend"]
    transport = None
    if backend is None:
        require(done["backend_pid"] is None and not decoded["backend_stdout_raw"]
                and not decoded["backend_stderr_raw"], "absent backend observation")
        require(done["status"] in ("input_error", "launch_failed", "timeout", "cancelled"),
                "absent backend status")
        require(done["status"] != "launch_failed" or done["payload_received"],
                "backend launch before payload")
    else:
        fields(backend, PROCESS_FIELDS)
        _integer(done["backend_pid"], 1)
        require(done["payload_received"], "backend requires payload")
        transport = {
            **backend, "stdout_hex": decoded["backend_stdout_raw"].hex(),
            "stderr_hex": decoded["backend_stderr_raw"].hex(),
        }
        _process_result(transport, outer=True)
        require(backend["status"] == done["status"]
                and backend["status"] != "launch_failed", "supervisor/backend status")
        require(backend["process_reaped"] == (backend["returncode"] is not None),
                "backend reaped/returncode")
        require(backend["elapsed_ns"] <= done["work_elapsed_ns"] + done["cleanup_elapsed_ns"],
                "backend elapsed containment")
    if done["status"] == "completed":
        require(done["work_elapsed_ns"] <= 95_000_000_000
                and done["cleanup_elapsed_ns"] <= 2_000_000_000, "supervisor deadlines")
    return {
        "ready": ready, "done": done, "transport": transport,
        "receipt_binding_consistent": True,
        "backend_lifecycle_complete": backend is not None and backend["process_reaped"],
        "payload_received": done["payload_received"], "backend_started": backend is not None,
    }


def _context(context, stage_id):
    _plain(context)
    fields(context, ("package_sha256", "config_sha256", "stage_root", "expected_runtime"))
    for name in ("package_sha256", "config_sha256"):
        _hex(context[name], 64)
    _stage_root(context["stage_root"], stage_id)
    runtime = context["expected_runtime"]
    fields(runtime, RUNTIME_FIELDS)
    for name in ("hostname", "python", "python_root"):
        _text(runtime[name])
    _integer(runtime["uid"])
    for name in ("python_sha256", "bwrap_sha256", "seccomp_sha256"):
        _hex(runtime[name], 64)
    require(runtime["no_new_privs"] is True, "runtime no_new_privs")
    require(type(runtime["argv_prefix"]) is list and bool(runtime["argv_prefix"]),
            "runtime argv")
    for argument in runtime["argv_prefix"]:
        _text(argument)
    fields(runtime["host_namespaces"], ("mnt", "pid", "net", "user"))
    for namespace in runtime["host_namespaces"].values():
        _text(namespace)


def _ready_context(public, batch, context):
    return {
        "kind": "LORA_RECALIBRATION_READY", "version": 1, "stage_id": public["stage_id"],
        **{name: batch[name] for name in ("batch_index", "run_id", "bundle_sha256")},
        **{name: public[name] for name in (
            "preparation_sha256", "backend_sha256", "worker_sha256")},
        **{name: context[name] for name in ("package_sha256", "config_sha256", "stage_root")},
        "hostname": context["expected_runtime"]["hostname"],
        "uid": context["expected_runtime"]["uid"],
        "batch_root": context["stage_root"]
        + "/batch-%02d-" % batch["batch_index"] + batch["run_id"],
        "expected_runtime_sha256": digest(context["expected_runtime"]),
    }


def _record(record, position, public, context, bundle_raw):
    fields(record, ("batch_index", "bundle_raw", "ssh", "wire_sha256"))
    require(all(type(name) is str for name in record), "record key type")
    require(type(record["batch_index"]) is int and record["batch_index"] == position,
            "record order")
    require(type(record["bundle_raw"]) is bytes and len(record["bundle_raw"]) <= 131072,
            "record bundle bytes")
    _plain(record["ssh"])
    wire_raw = _process_result(record["ssh"], outer=True)
    _hex(record["wire_sha256"], 64)
    ssh = record["ssh"]
    require(ssh["process_reaped"] == (ssh["returncode"] is not None),
            "SSH reaped/returncode")
    observation = {
        "batch_index": position, "ready": None, "done": None, "transport": None,
        "receipt_binding_consistent": False, "backend_lifecycle_complete": False,
        "payload_received": None, "backend_started": None, "receipt_error": None,
        "ssh_lifecycle_complete": ssh["process_reaped"], "can_continue": False,
        "record_sha256": digest({
            "batch_index": position, "bundle_sha256": _sha(record["bundle_raw"]),
            "ssh_sha256": digest(ssh), "wire_sha256": record["wire_sha256"],
        }),
    }
    reason = None
    try:
        require(record["bundle_raw"] == bundle_raw, "exact dispatched bundle")
        observation.update(decode_receipt(
            wire_raw, expected_sha256=record["wire_sha256"],
            expected=_ready_context(public, public["batches"][position], context)))
    except ValueError as error:
        reason = "receipt_error"
        observation["receipt_error"] = str(error)
    if reason is None and observation["done"]["status"] != "completed":
        reason = "supervisor_" + observation["done"]["status"]
    if reason is None and ssh["status"] != "completed":
        reason = "ssh_" + ssh["status"]
    if reason is None and ssh["elapsed_ns"] > 112_000_000_000:
        reason = "ssh_deadline"
    observation["stop_reason"] = reason
    return observation


def build_evidence(preparation, raw_files, worker_raw, backend_raw, source_raw,
                   records, *, stage_id, run_ids, context):
    validate_preparation(preparation, raw_files, worker_raw, backend_raw, source_raw,
                         stage_id=stage_id, run_ids=run_ids)
    _context(context, stage_id)
    require(type(records) is list and len(records) <= 15, "stage records")
    public = preparation["public"]
    observations = []
    mapped = []
    for position, record in enumerate(records):
        require(not observations or observations[-1]["stop_reason"] is None,
                "record after outer stop")
        bundle_raw = bytes.fromhex(preparation["private"]["bundles_hex"][position])
        observation = _record(record, position, public, context, bundle_raw)
        observations.append(observation)
        mapped.append({
            "batch_index": position, "bundle_raw": bundle_raw,
            "transport": observation["transport"],
        })
    content = build_report(raw_files, worker_raw, run_ids, mapped,
                           expected_runtime=context["expected_runtime"])
    first_stop = None
    for observation, batch in zip(observations, content["batches"]):
        reason = observation["stop_reason"] or batch["stop_reason"]
        observation["stop_reason"] = reason
        observation["can_continue"] = reason is None
        if batch["stop_reason"] == "evidence_error":
            observation["receipt_binding_consistent"] = False
        if reason is not None:
            first_stop = {
                "batch_index": batch["batch_index"], "ordinal": batch["stop_ordinal"],
                "reason": reason,
            }
    complete = len(observations) == 15 and all(row["can_continue"] for row in observations)
    if first_stop is None and not complete:
        first_stop = {
            "batch_index": len(observations), "ordinal": 32 * len(observations),
            "reason": "records_exhausted",
        }
    return {
        "kind": "LORA_RECALIBRATION_EXECUTION_EVIDENCE", "version": 1,
        "stage_id": stage_id, "preparation_sha256": public["preparation_sha256"],
        "declared_context_sha256": digest(context), "declared_context": copy.deepcopy(context),
        "records_sha256": digest([row["record_sha256"] for row in observations]),
        "content_report": content, "batches": observations, "first_stop": first_stop,
        "receipt_binding_consistent": bool(observations) and all(
            row["receipt_binding_consistent"] for row in observations),
        "collection_complete": complete, "missing_bindings": list(MISSING_BINDINGS),
        **{flag: False for flag in FALSE_FLAGS},
    }
