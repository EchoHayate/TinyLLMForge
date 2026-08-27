#!/usr/bin/env python3
"""Run the Qwen3.8 verifier locally while staging one remote trace at a time."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time


REPO_ROOT = Path(__file__).resolve().parents[4]
TOOLS_ROOT = REPO_ROOT / "tools"
sys.path.insert(0, str(TOOLS_ROOT))

import verify_qwen38_tp4_communication_profile as verifier  # noqa: E402


def _load_resume_records(
    path: Path | None,
    *,
    recorded: dict[str, str],
    ordered_trace_names: list[str],
) -> dict[str, str]:
    if path is None:
        return {}
    completed = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if (
            len(fields) != 4
            or fields[0] not in {
                "TRACE_VERIFY_PASS",
                "TRACE_VERIFY_RESUME_PASS",
            }
        ):
            continue
        index_text, relative, digest = fields[1:]
        index_parts = index_text.split("/", maxsplit=1)
        if (
            len(index_parts) != 2
            or not all(part.isdigit() for part in index_parts)
        ):
            raise ValueError(f"invalid resume trace index: {index_text}")
        index, total = (int(part) for part in index_parts)
        if total != len(ordered_trace_names) or not 1 <= index <= total:
            raise ValueError(f"resume trace index out of range: {index_text}")
        if ordered_trace_names[index - 1] != relative:
            raise ValueError(
                f"resume trace order mismatch at {index_text}: {relative}"
            )
        if recorded.get(relative) != digest:
            raise ValueError(f"resume digest mismatch for {relative}")
        if relative in completed:
            raise ValueError(f"duplicate resume trace: {relative}")
        completed[relative] = digest
    return completed


def _validate_one_trace(
    trace_path: Path,
    *,
    workload: str,
    repetition: int,
    profiles: dict,
) -> None:
    rank_rows = profiles["cases"][(workload, "measured", repetition)]
    parsed = verifier.parse_nsys_sqlite(
        trace_path,
        verifier._structured_rows(rank_rows),
    )
    if parsed.get("classification") != "COMPLETE":
        raise ValueError(
            "Nsight correlation is incomplete: "
            f"{parsed.get('coverage_errors')}"
        )

    metric_fields = (
        "step_critical_interval_ns",
        *verifier.LAYER_METRICS,
        "cpu_global_tids",
        "stream_ids",
    )
    parsed_rows = {
        (
            row["rank"],
            row["decode_ordinal"],
            row["layer_index"],
            row["layer_role"],
        ): row
        for row in parsed["rows"]
    }
    expected_rows = {}
    for rank, profile in rank_rows.items():
        for step in profile["steps"]:
            for layer in step["layers"]:
                expected_rows[
                    (
                        rank,
                        step["decode_ordinal"],
                        layer["layer_index"],
                        layer["layer_role"],
                    )
                ] = layer
    if set(parsed_rows) != set(expected_rows):
        raise ValueError("Nsight/profile layer inventory mismatch")
    for key, expected in expected_rows.items():
        actual = parsed_rows[key]
        if any(actual[field] != expected[field] for field in metric_fields):
            raise ValueError("Nsight/profile interval arithmetic mismatch")

    parsed_step_rows = {
        (row["rank"], row["decode_ordinal"]): row
        for row in parsed["step_rows"]
    }
    expected_step_rows = {
        (rank, step["decode_ordinal"]): step
        for rank, profile in rank_rows.items()
        for step in profile["steps"]
    }
    if set(parsed_step_rows) != set(expected_step_rows):
        raise ValueError("Nsight/profile step inventory mismatch")
    for key, expected in expected_step_rows.items():
        if (
            parsed_step_rows[key]["final_required_offset_ns"]
            != expected["final_required_offset_ns"]
        ):
            raise ValueError("Nsight/profile rank-step offset mismatch")

    critical_rows = {
        row["decode_ordinal"]: row for row in parsed["critical_rows"]
    }
    for step_index, step in enumerate(rank_rows[0]["steps"]):
        actual = critical_rows.get(step["decode_ordinal"])
        critical_rank = step["critical_rank"]
        critical_step = rank_rows[critical_rank]["steps"][step_index]
        if (
            actual is None
            or actual["critical_rank"] != critical_rank
            or actual["final_required_offset_ns"]
            != critical_step["final_required_offset_ns"]
        ):
            raise ValueError("Nsight/profile critical-rank mismatch")


def _copy_trace(
    *,
    ssh_target: str,
    ssh_control_path: str | None,
    remote_trace_root: str,
    relative: str,
    destination: Path,
    max_attempts: int,
    retry_delay_seconds: float,
) -> None:
    remote_path = f"{remote_trace_root.rstrip('/')}/{Path(relative).name}"
    ssh_options = [
        "ssh",
        "-o",
        "BatchMode=yes",
        "-o",
        "GSSAPIAuthentication=yes",
        "-o",
        "ConnectTimeout=20",
    ]
    if ssh_control_path:
        ssh_options.extend(
            ["-S", ssh_control_path, "-o", "ControlMaster=no"]
        )
    ssh_options.append(ssh_target)
    command = [
        "rclone",
        "copyto",
        f":sftp:{remote_path}",
        str(destination),
        "--sftp-ssh",
        " ".join(ssh_options),
        "--sftp-disable-hashcheck",
        "--multi-thread-cutoff",
        "1M",
        "--multi-thread-streams",
        "2",
    ]
    environment = dict(os.environ)
    environment["KRB5CCNAME"] = "FILE:/Users/bytedance/krb5cc_sitian"
    for attempt in range(1, max_attempts + 1):
        try:
            subprocess.run(command, check=True, env=environment)
            return
        except subprocess.CalledProcessError:
            destination.unlink(missing_ok=True)
            for partial in destination.parent.glob(
                f"{destination.name}.*.partial"
            ):
                partial.unlink()
            if attempt == max_attempts:
                raise
            print(
                f"TRACE_COPY_RETRY {attempt}/{max_attempts} {relative}",
                file=sys.stderr,
                flush=True,
            )
            time.sleep(retry_delay_seconds)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", required=True, type=Path)
    parser.add_argument("--remote-trace-root", required=True)
    parser.add_argument(
        "--ssh-target",
        default="sitian@10.232.195.203",
    )
    parser.add_argument(
        "--ssh-control-path",
        default="/tmp/ssh-sitian-10.232.195.203",
    )
    parser.add_argument("--resume-log", type=Path)
    parser.add_argument("--copy-attempts", type=int, default=5)
    parser.add_argument("--retry-delay-seconds", type=float, default=5.0)
    parser.add_argument("--remote-manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.copy_attempts < 1:
        raise ValueError("copy-attempts must be positive")
    if args.retry_delay_seconds < 0:
        raise ValueError("retry-delay-seconds must be non-negative")

    root = args.bundle.resolve()
    final_manifest = verifier._load_json(args.remote_manifest)
    if (
        final_manifest.get("schema_version") != verifier.MANIFEST_SCHEMA
        or not isinstance(final_manifest.get("artifacts"), dict)
    ):
        raise ValueError("manifest schema mismatch")
    recorded = final_manifest["artifacts"]
    trace_names = verifier._expected_trace_names()
    logical_inventory = (
        verifier.BASE_FILES
        | trace_names
        | {"independent_verification.json"}
    )
    if set(recorded) != logical_inventory:
        raise ValueError("manifest artifact inventory mismatch")
    ordered_trace_names = sorted(trace_names)
    completed_traces = _load_resume_records(
        args.resume_log,
        recorded=recorded,
        ordered_trace_names=ordered_trace_names,
    )

    for relative, expected in sorted(recorded.items()):
        verifier._sha256(expected, f"manifest digest for {relative}")
        if relative.startswith("nsys/") or relative == (
            "independent_verification.json"
        ):
            continue
        actual = verifier._sha256_file(root / relative)
        if actual != expected:
            raise ValueError(f"manifest digest mismatch for {relative}")
        print(f"LOCAL_BASE_HASH_PASS {relative}", flush=True)

    identity = verifier._validate_identity(root)
    profiles = verifier._validate_profiles(
        verifier._load_jsonl(root / "profile_rows.jsonl"),
        identity,
    )

    with tempfile.TemporaryDirectory(
        prefix="qwen38-streaming-verifier-",
        dir="/tmp",
    ) as temporary:
        staging_root = Path(temporary)
        for index, relative in enumerate(ordered_trace_names, start=1):
            if relative in completed_traces:
                print(
                    f"TRACE_VERIFY_RESUME_PASS {index}/25 {relative} "
                    f"{completed_traces[relative]}",
                    flush=True,
                )
                continue
            free = shutil.disk_usage(staging_root).free
            if free < 10 * 1024**3:
                raise RuntimeError(
                    f"local free space below 10 GiB before {relative}"
                )
            staged = staging_root / Path(relative).name
            print(
                f"TRACE_STAGE_BEGIN {index}/25 {relative} "
                f"free_bytes={free}",
                flush=True,
            )
            _copy_trace(
                ssh_target=args.ssh_target,
                ssh_control_path=args.ssh_control_path,
                remote_trace_root=args.remote_trace_root,
                relative=relative,
                destination=staged,
                max_attempts=args.copy_attempts,
                retry_delay_seconds=args.retry_delay_seconds,
            )
            actual = verifier._sha256_file(staged)
            expected = recorded[relative]
            if actual != expected:
                raise ValueError(f"manifest digest mismatch for {relative}")
            workload, repetition_text = staged.stem.split("-r", maxsplit=1)
            _validate_one_trace(
                staged,
                workload=workload,
                repetition=int(repetition_text),
                profiles=profiles,
            )
            staged.unlink()
            print(
                f"TRACE_VERIFY_PASS {index}/25 {relative} {actual}",
                flush=True,
            )

    summary = verifier._reconstruct_summary(root, profiles, identity)
    verifier._verify_producer_outputs(root, summary, identity)
    result = {
        "schema_version": verifier.VERIFICATION_SCHEMA,
        "status": "PASS",
        "source_revision": identity["source_revision"],
        "source_tree_sha256": identity["source_tree_sha256"],
        "model_revision": identity["model_revision"],
        "rank_inventory": list(verifier.RANKS),
        "gpu_uuids": identity["gpu_uuids"],
        "profile_row_count": len(profiles["rows"]),
        "correctness_row_count": summary["correctness"]["row_count"],
        "nsys_trace_count": len(trace_names),
        "artifact_hashes_verified": True,
        "complete_four_rank_alignment": True,
        "trace_coverage_complete": True,
        "correctness_valid": True,
        "cleanup_valid": identity["cleanup_valid"],
        "strict_clean_worker_entry_count": identity[
            "strict_clean_worker_entry_count"
        ],
        "profiler_overhead_ratio": summary["profiler_overhead_ratio"],
        "producer_classification": summary["classification"],
        "reconstructed_classification": summary["classification"],
        "workloads": {
            workload: {
                "median_exposed_communication_ratio": payload[
                    "median_exposed_communication_ratio"
                ],
                "median_overlap_headroom_lower_bound": payload[
                    "median_overlap_headroom_lower_bound"
                ],
                "representative_repetition": payload[
                    "representative_repetition"
                ],
            }
            for workload, payload in summary["workloads"].items()
        },
    }
    verifier._write_json_atomic(args.output, result)
    expected_result_hash = recorded["independent_verification.json"]
    actual_result_hash = verifier._sha256_file(args.output)
    if actual_result_hash != expected_result_hash:
        raise ValueError(
            "local/remote independent-verification digest mismatch"
        )
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    print(
        "LOCAL_STREAMING_MANIFEST_PASS "
        f"artifacts={len(recorded)} "
        f"result_sha256={actual_result_hash}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
