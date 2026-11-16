"""Retain failed-stage evidence while comparing authorized cross-stage receipts."""

import copy
import re

from pilot_protocol import fields, require
from validation_contract import parse_json
from validation_matrix import aggregate_matrix


def _authorization(authorization):
    fields(authorization, (
        "kind", "prior_stage_id", "stage_id", "prior_seal_sha256",
        "failed_run_id", "failed_transport_sha256", "resubmit_start",
        "resubmit_stop", "tail_stop", "user_approved"))
    require(type(authorization["kind"]) is str
            and authorization["kind"] == "LORA_TAIL_AUTHORIZATION",
            "tail authorization kind")
    require(authorization["user_approved"] is True, "explicit resubmission approval required")
    for name in ("prior_stage_id", "stage_id", "failed_run_id"):
        require(type(authorization[name]) is str
                and re.fullmatch(r"[0-9a-f]{32}", authorization[name]),
                "authorization ID")
    for name in ("prior_seal_sha256", "failed_transport_sha256"):
        require(type(authorization[name]) is str
                and re.fullmatch(r"[0-9a-f]{64}", authorization[name]),
                "authorization hash")
    require(authorization["stage_id"] != authorization["prior_stage_id"],
            "distinct continuation stage required")
    for name, expected in (("resubmit_start", 1088), ("resubmit_stop", 1120),
                           ("tail_stop", 1416)):
        require(type(authorization[name]) is int and authorization[name] == expected,
                "authorization bounds")


def _jump_rejection(transport):
    stderr = bytes.fromhex(transport["stderr_hex"])
    require(re.search(
        rb"(?m)^[^\s@]+@jump[-.][A-Za-z0-9.-]+: "
        rb"Permission denied \([^\r\n()]+\)\.\r?$", stderr),
        "saved jump authentication rejection required")
    lower = stderr.lower()
    require(not any(marker in lower for marker in (
        b"authenticated to ", b"authentication succeeded",
        b"sending command:", b"request exec", b"exec request accepted",
        b"entering interactive session", b"sending subsystem:",
        b"request subsystem", b"request shell",
    )), "authentication or target command launch contradicts jump rejection")


def aggregate_tail(source_raw, manifest_raw, d01_bundle_raw, d01_transport,
                   previous_records, tail_records, *, expected_runtime,
                   d01_expected_runtime, authorization):
    """Revalidate receipt lineage; the caller authenticates approval and seal pins."""
    _authorization(authorization)
    require(type(previous_records) is list and len(previous_records) == 34,
            "exactly 34 previous attempts required")
    require(type(tail_records) is list and len(tail_records) <= 11,
            "ordered tail records required")
    prior = aggregate_matrix(
        source_raw, manifest_raw, d01_bundle_raw, d01_transport, previous_records,
        expected_runtime=expected_runtime, d01_expected_runtime=d01_expected_runtime)
    failed = prior["batches"][-1]
    transport = failed["transport"]
    require(all(batch["batch_complete"] for batch in prior["batches"][:-1])
            and failed["start"] == 1088 and failed["received"] == 0
            and failed["runtime"] is None and failed["done"] is None
            and not failed["batch_complete"] and prior["received"] == 1088
            and prior["remaining"] == 328 and not prior["matrix_complete"],
            "prior must retain 33 complete batches and the empty start1088 failure")
    require(transport["status"] == "program_error" and transport["returncode"] == 255
            and transport["stdout_hex"] == "", "prior jump transport failure required")
    require(failed["run_id"] == authorization["failed_run_id"]
            and failed["transport_sha256"] == authorization["failed_transport_sha256"],
            "authorization does not bind failed attempt")
    _jump_rejection(transport)
    stage_ids = {authorization["prior_stage_id"], authorization["stage_id"]}
    run_ids = {prior["d01"]["run_id"], *(batch["run_id"] for batch in prior["batches"])}
    require(run_ids.isdisjoint(stage_ids), "stage ID reused as run ID")
    for position, record in enumerate(tail_records):
        fields(record, ("start", "bundle_raw", "transport"))
        require(type(record["start"]) is int and record["start"] == 1088 + 32 * position,
                "tail batch gap, duplicate or reordering")
        bundle = parse_json(record["bundle_raw"])
        require(type(bundle) is dict, "tail bundle object")
        run_id = bundle.get("run_id")
        require(type(run_id) is str and re.fullmatch(r"[0-9a-f]{32}", run_id),
                "tail bundle run ID")
        require(run_id not in run_ids and run_id not in stage_ids,
                "tail run ID reused")
        run_ids.add(run_id)
    combined = aggregate_matrix(
        source_raw, manifest_raw, d01_bundle_raw, d01_transport,
        previous_records[:-1] + tail_records, expected_runtime=expected_runtime,
        d01_expected_runtime=d01_expected_runtime)
    combined["kind"] = "LORA_MATRIX_CROSS_STAGE_OBSERVATIONS"
    return {
        "kind": "LORA_MATRIX_CONTINUATION_REPORT",
        "authorization": copy.deepcopy(authorization),
        "prior": prior, "combined": combined,
        "selected_batches": [
            {"stage_id": authorization[
                "prior_stage_id" if batch["start"] < 1088 else "stage_id"],
             "start": batch["start"], "run_id": batch["run_id"]}
            for batch in combined["batches"]
        ],
    }
