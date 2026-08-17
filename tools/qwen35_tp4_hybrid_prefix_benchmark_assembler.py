from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile

import qwen35_tp4_hybrid_prefix_benchmark_contract as contract


COMPLETION_MARKER = "QWEN35_TP4_BENCHMARK_WORKER_COMPLETE"


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _regular_file(path, label):
    path = Path(path)
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"{label} must be a regular file")
    return path


def _load_json(path, label):
    path = _regular_file(path, label)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"{label} JSON is invalid") from error
    return value


def _load_jsonl(path, label):
    path = _regular_file(path, label)
    rows = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        1,
    ):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as error:
            raise ValueError(
                f"{label} JSONL row {line_number} is invalid"
            ) from error
    return rows


def _write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(
            value,
            handle,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        handle.write("\n")


def _write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        for row in rows:
            handle.write(contract.canonical_json_bytes(row) + b"\n")


def _safe_relative_file(root, relative, label):
    if not isinstance(relative, str) or not relative:
        raise ValueError(f"{label} path is invalid")
    value = Path(relative)
    if value.is_absolute() or ".." in value.parts:
        raise ValueError(f"{label} path is unsafe")
    path = Path(root) / value
    return _regular_file(path, label)


def _validate_document(value, required, label):
    if not isinstance(value, dict) or set(value) != set(required):
        raise ValueError(f"{label} schema is invalid")
    if value["schema_version"] != contract.SCHEMA_VERSION:
        raise ValueError(f"{label} schema version mismatch")
    return value


def _validate_commands(commands, matrix):
    fields = {
        "case_id",
        "policy",
        "workload",
        "phase",
        "repetition",
        "dist_port",
        "master_port",
    }
    if not isinstance(commands, list) or len(commands) != len(matrix):
        raise ValueError("command inventory is invalid")
    used_ports = set()
    for command, case in zip(commands, matrix):
        if not isinstance(command, dict) or set(command) != fields:
            raise ValueError("command inventory is invalid")
        for name in (
            "case_id",
            "policy",
            "workload",
            "phase",
            "repetition",
        ):
            if command[name] != getattr(case, name):
                raise ValueError("command inventory is not canonical")
        ports = (command["dist_port"], command["master_port"])
        if (
            any(
                isinstance(port, bool)
                or not isinstance(port, int)
                or port <= 0
                for port in ports
            )
            or ports[0] == ports[1]
            or any(port in used_ports for port in ports)
        ):
            raise ValueError("command port inventory is invalid")
        used_ports.update(ports)


def _validate_worker_logs(worker_logs, matrix):
    if (
        not isinstance(worker_logs, dict)
        or set(worker_logs) != {case.case_id for case in matrix}
    ):
        raise ValueError("worker log inventory is invalid")
    validated = {}
    for case in matrix:
        path = _regular_file(
            worker_logs[case.case_id],
            f"{case.case_id} worker log",
        )
        text = path.read_text(encoding="utf-8", errors="replace")
        if (
            "Traceback (most recent call last)" in text
            or COMPLETION_MARKER not in text
        ):
            raise ValueError(f"{case.case_id} worker log is invalid")
        validated[case.case_id] = path
    return validated


def _copy_prerequisites(source, destination_root):
    payload = _load_json(source, "correctness prerequisites")
    status = contract.validate_prerequisites(source)
    if not status.authorized:
        raise ValueError(
            "correctness prerequisites are not authorized: "
            + "; ".join(status.reasons)
        )
    shutil.copyfile(
        source,
        destination_root / "correctness_prerequisites.json",
    )
    for name in (
        "tp4_root_logit",
        "cached_continuation",
        "engine_correctness",
    ):
        row = payload[name]
        for field, label in (
            ("artifact_path", "artifact"),
            (
                "independent_verification_path",
                "independent verification",
            ),
            ("provenance_path", "provenance"),
        ):
            relative = row[field]
            if (
                not isinstance(relative, str)
                or not relative.startswith("prerequisites/")
            ):
                raise ValueError(
                    f"{name} prerequisite path is not canonical"
                )
            source_path = _safe_relative_file(
                source.parent,
                relative,
                f"{name} {label}",
            )
            destination = destination_root / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source_path, destination)
        provenance = _load_json(
            source.parent / row["provenance_path"],
            f"{name} provenance",
        )
        provenance_root = (
            source.parent / row["provenance_path"]
        ).parent
        for field, label in (
            ("plan_path", "execution plan"),
            (
                "authorization_path",
                "consumed authorization",
            ),
            ("receipt_path", "execution receipt"),
        ):
            relative = provenance[field]
            source_path = _safe_relative_file(
                provenance_root,
                relative,
                f"{name} {label}",
            )
            destination = (
                destination_root
                / Path(row["provenance_path"]).parent
                / relative
            )
            shutil.copyfile(source_path, destination)
    return _sha256(source)


def _validate_case_rows(
    rows,
    *,
    case,
    source_tree_sha256,
    workload_manifest_sha256,
    prerequisites_sha256,
):
    expected_count = contract.WORKLOAD_SPECS[
        case.workload
    ]["continuations"]
    if len(rows) != expected_count:
        raise ValueError(f"{case.case_id} case row count mismatch")
    expected_ids = [
        f"{case.case_id}__request-{index}"
        for index in range(expected_count)
    ]
    if [row.get("row_id") for row in rows] != expected_ids:
        raise ValueError(f"{case.case_id} case row order mismatch")
    for row in rows:
        if (
            not isinstance(row, dict)
            or set(row) != set(contract.CASE_ROW_FIELDS)
            or any(
                row[name] != getattr(case, name)
                for name in (
                    "case_id",
                    "policy",
                    "workload",
                    "phase",
                    "repetition",
                )
            )
        ):
            raise ValueError(f"{case.case_id} case row schema mismatch")
        if (
            row["source_tree_sha256"] != source_tree_sha256
            or row["model_manifest_sha256"]
            != contract.MODEL_MANIFEST_SHA256
            or row["workload_manifest_sha256"]
            != workload_manifest_sha256
            or row["correctness_prerequisites_sha256"]
            != prerequisites_sha256
        ):
            raise ValueError(f"{case.case_id} case row provenance mismatch")
        if (
            contract.canonical_json_sha256(row["output_token_ids"])
            != row["output_token_ids_sha256"]
        ):
            raise ValueError(f"{case.case_id} output token hash mismatch")


def _validate_process_rows(rows, case):
    if (
        len(rows) != 1
        or not isinstance(rows[0], dict)
        or set(rows[0]) != set(contract.PROCESS_ROW_FIELDS)
        or any(
            rows[0][name] != getattr(case, name)
            for name in (
                "case_id",
                "policy",
                "workload",
                "phase",
                "repetition",
            )
        )
    ):
        raise ValueError(f"{case.case_id} process row mismatch")


def _copy_case_logits(case_dir, output_root, rows, logits):
    for row in rows:
        relative = row["final_logits_path"]
        digest = row["final_logits_sha256"]
        if row["phase"] != "correctness":
            if relative is not None or digest is not None:
                raise ValueError("non-correctness logits are invalid")
            continue
        source = _safe_relative_file(
            case_dir,
            relative,
            f"{row['row_id']} logits",
        )
        if _sha256(source) != digest:
            raise ValueError(f"{row['row_id']} logits SHA mismatch")
        destination = output_root / relative
        if destination.exists():
            raise ValueError("duplicate logits path")
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        logits.append({"path": relative, "sha256": digest})


def _artifact_manifest(root):
    files = {}
    for relative in contract.ARTIFACT_MANIFEST_HASH_DOMAIN:
        path = _regular_file(root / relative, relative)
        files[relative] = {
            "sha256": _sha256(path),
            "size": path.stat().st_size,
        }
    for directory in contract.NESTED_ARTIFACT_DIRECTORIES:
        for path in sorted((root / directory).rglob("*")):
            if path.is_symlink() or not path.is_file():
                if path.is_dir() and not path.is_symlink():
                    continue
                raise ValueError("nested artifact inventory is invalid")
            relative = path.relative_to(root).as_posix()
            files[relative] = {
                "sha256": _sha256(path),
                "size": path.stat().st_size,
            }
    return {
        "schema_version": contract.SCHEMA_VERSION,
        "files": files,
    }


def assemble_benchmark_run(
    *,
    output_dir,
    cases_root,
    correctness_prerequisites_path,
    workload_manifest_path,
    source_manifest,
    environment,
    gpu_assignments,
    commands,
    worker_logs,
):
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise ValueError("benchmark output already exists")
    cases_root = Path(cases_root)
    if not cases_root.is_dir() or cases_root.is_symlink():
        raise ValueError("case inventory is invalid")
    matrix = contract.build_case_matrix()
    actual_cases = {
        path.name
        for path in cases_root.iterdir()
        if path.is_dir() and not path.is_symlink()
    }
    expected_cases = {case.case_id for case in matrix}
    if actual_cases != expected_cases:
        raise ValueError("case inventory is not canonical")
    source_manifest = _validate_document(
        source_manifest,
        {
            "schema_version",
            "source_tree_sha256",
            "model_manifest_sha256",
        },
        "source manifest",
    )
    source_tree_sha256 = source_manifest["source_tree_sha256"]
    if (
        not isinstance(source_tree_sha256, str)
        or len(source_tree_sha256) != 64
        or source_manifest["model_manifest_sha256"]
        != contract.MODEL_MANIFEST_SHA256
    ):
        raise ValueError("source manifest identity mismatch")
    _validate_document(
        environment,
        {"schema_version", "world_size", "python"},
        "environment",
    )
    if (
        environment["world_size"] != contract.WORLD_SIZE
        or not isinstance(environment["python"], str)
        or not environment["python"]
    ):
        raise ValueError("environment identity mismatch")
    if not isinstance(gpu_assignments, dict):
        raise ValueError("GPU assignments schema is invalid")
    gpu_assignment_fields = set(gpu_assignments)
    if gpu_assignment_fields == {
        "schema_version",
        "assignments",
    }:
        _validate_document(
            gpu_assignments,
            gpu_assignment_fields,
            "GPU assignments",
        )
    elif gpu_assignment_fields == {
        "schema_version",
        "resource_policy",
        "maximum_gpu_utilization_percent",
        "assignments",
    }:
        _validate_document(
            gpu_assignments,
            gpu_assignment_fields,
            "GPU assignments",
        )
        if (
            gpu_assignments["resource_policy"]
            != "shared-low-utilization"
            or isinstance(
                gpu_assignments["maximum_gpu_utilization_percent"],
                bool,
            )
            or not isinstance(
                gpu_assignments["maximum_gpu_utilization_percent"],
                int,
            )
            or not 0
            <= gpu_assignments["maximum_gpu_utilization_percent"]
            <= 100
        ):
            raise ValueError("GPU assignments policy is invalid")
    else:
        raise ValueError("GPU assignments schema is invalid")
    _validate_commands(commands, matrix)
    validated_logs = _validate_worker_logs(worker_logs, matrix)
    workload_manifest_path = _regular_file(
        workload_manifest_path,
        "workload manifest",
    )
    workload_manifest = _load_json(
        workload_manifest_path,
        "workload manifest",
    )
    if workload_manifest != contract.workload_manifest_payload():
        raise ValueError("workload manifest content mismatch")
    workload_manifest_sha256 = _sha256(workload_manifest_path)

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(
        dir=output_dir.parent,
        prefix=f".{output_dir.name}.",
    ))
    try:
        for directory in contract.NESTED_ARTIFACT_DIRECTORIES:
            (temporary / directory).mkdir()
        prerequisites_sha256 = _copy_prerequisites(
            Path(correctness_prerequisites_path),
            temporary,
        )
        shutil.copyfile(
            workload_manifest_path,
            temporary / "workload_manifest.json",
        )
        _write_json(temporary / "source_manifest.json", source_manifest)
        _write_json(temporary / "environment.json", environment)
        _write_json(
            temporary / "gpu_assignments.json",
            gpu_assignments,
        )
        _write_json(
            temporary / "commands.json",
            {
                "schema_version": contract.SCHEMA_VERSION,
                "commands": commands,
            },
        )

        case_rows = []
        process_rows = []
        logits = []
        log_manifest = []
        for case in matrix:
            case_dir = cases_root / case.case_id
            summary = _load_json(
                case_dir / "summary.json",
                f"{case.case_id} summary",
            )
            rows = _load_jsonl(
                case_dir / "case_rows.jsonl",
                f"{case.case_id} case rows",
            )
            processes = _load_jsonl(
                case_dir / "process_rows.jsonl",
                f"{case.case_id} process rows",
            )
            if (
                not isinstance(summary, dict)
                or set(summary)
                != {
                    "schema_version",
                    "complete",
                    "case_id",
                    "case_rows",
                    "process_rows",
                }
                or summary["schema_version"] != contract.SCHEMA_VERSION
                or summary["complete"] is not True
                or summary["case_id"] != case.case_id
                or summary["case_rows"] != len(rows)
                or summary["process_rows"] != len(processes)
                or (case_dir / "failure.json").exists()
            ):
                raise ValueError(f"{case.case_id} summary is invalid")
            _validate_case_rows(
                rows,
                case=case,
                source_tree_sha256=source_tree_sha256,
                workload_manifest_sha256=workload_manifest_sha256,
                prerequisites_sha256=prerequisites_sha256,
            )
            _validate_process_rows(processes, case)
            _copy_case_logits(case_dir, temporary, rows, logits)
            case_rows.extend(rows)
            process_rows.extend(processes)

            log_source = validated_logs[case.case_id]
            log_relative = f"logs/{case.case_id}.log"
            log_destination = temporary / log_relative
            shutil.copyfile(log_source, log_destination)
            log_manifest.append({
                "path": log_relative,
                "sha256": _sha256(log_destination),
            })

        _write_jsonl(temporary / "case_rows.jsonl", case_rows)
        _write_jsonl(temporary / "process_rows.jsonl", process_rows)
        _write_json(
            temporary / "logits_manifest.json",
            {
                "schema_version": contract.SCHEMA_VERSION,
                "files": logits,
            },
        )
        _write_json(
            temporary / "worker_logs_manifest.json",
            {
                "schema_version": contract.SCHEMA_VERSION,
                "files": log_manifest,
            },
        )
        summary = {
            "schema_version": contract.SCHEMA_VERSION,
            "classification": "UNTRUSTED_PRODUCER_COMPLETE",
            "case_rows": len(case_rows),
            "process_rows": len(process_rows),
        }
        _write_json(temporary / "summary.json", summary)
        _write_json(
            temporary / "artifact_manifest.json",
            _artifact_manifest(temporary),
        )
        os.replace(temporary, output_dir)
        return {
            "classification": "ASSEMBLED",
            "case_rows": len(case_rows),
            "process_rows": len(process_rows),
        }
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cases-root", type=Path, required=True)
    parser.add_argument(
        "--correctness-prerequisites",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--workload-manifest",
        type=Path,
        required=True,
    )
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--environment", type=Path, required=True)
    parser.add_argument("--gpu-assignments", type=Path, required=True)
    parser.add_argument("--commands", type=Path, required=True)
    parser.add_argument("--worker-logs", type=Path, required=True)
    args = parser.parse_args(argv)
    result = assemble_benchmark_run(
        output_dir=args.output_dir,
        cases_root=args.cases_root,
        correctness_prerequisites_path=(
            args.correctness_prerequisites
        ),
        workload_manifest_path=args.workload_manifest,
        source_manifest=_load_json(
            args.source_manifest,
            "source manifest",
        ),
        environment=_load_json(args.environment, "environment"),
        gpu_assignments=_load_json(
            args.gpu_assignments,
            "GPU assignments",
        ),
        commands=_load_json(args.commands, "commands")["commands"],
        worker_logs=_load_json(
            args.worker_logs,
            "worker logs",
        )["worker_logs"],
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
