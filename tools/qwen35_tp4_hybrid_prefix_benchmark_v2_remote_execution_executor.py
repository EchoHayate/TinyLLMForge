from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import sys
import tempfile


def _load_contract():
    name = "qwen35_tp4_hybrid_prefix_benchmark_v2_contract_for_executor"
    module = sys.modules.get(name)
    if module is not None:
        return module
    path = Path(__file__).resolve().parent / (
        "qwen35_tp4_hybrid_prefix_benchmark_v2_contract.py"
    )
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


contract = _load_contract()
REQUIRED_EXECUTION_ENV = dict(contract.EXECUTION_ENV)


def _load_receipt_module():
    name = (
        "qwen35_tp4_hybrid_prefix_benchmark_v2_remote_execution_receipt"
        "_for_executor"
    )
    module = sys.modules.get(name)
    if module is not None:
        return module
    path = Path(__file__).resolve().parent / (
        "qwen35_tp4_hybrid_prefix_benchmark_v2_remote_execution_receipt.py"
    )
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


receipt_module = _load_receipt_module()


def _bounded(value):
    if not isinstance(value, str):
        raise ValueError("command output must be text")
    encoded = value.encode("utf-8")
    limit = contract.MAX_BOUNDED_OUTPUT_BYTES
    if len(encoded) <= limit:
        return value, False
    return encoded[:limit].decode("utf-8", errors="ignore"), True


def _atomic_publish(path, payload, label):
    destination = Path(path)
    if destination.exists() or destination.is_symlink():
        raise ValueError(f"{label} output already exists")
    destination.parent.mkdir(parents=True, exist_ok=True)
    data = contract.canonical_json_bytes(payload) + b"\n"
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, destination)
        except FileExistsError as error:
            raise ValueError(f"{label} output already exists") from error
        directory_fd = os.open(destination.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _validate_bound_output(plan, artifact_root, run_dir, output_path):
    if (artifact_root is None) != (run_dir is None):
        raise ValueError("artifact root and run directory must be provided")
    if artifact_root is None:
        return
    receipt_module._validate_detached_output(
        bundle={"execution_plan": plan},
        artifact_root=artifact_root,
        run_dir=run_dir,
        output_path=output_path,
    )


def _write_failure(
    path,
    payload,
    *,
    plan,
    artifact_root,
    run_dir,
):
    if path is None:
        return
    _validate_bound_output(
        plan,
        artifact_root,
        run_dir,
        path,
    )
    _atomic_publish(path, payload, "failure evidence")


def execute_plan(
    *,
    plan,
    authorization_record,
    detached_receipt_path,
    artifact_root,
    run_dir,
    command_runner,
    execution_env=None,
    receipt_builder,
    failure_path=None,
):
    if execution_env != REQUIRED_EXECUTION_ENV:
        raise ValueError("execution environment is invalid")
    try:
        contract.validate_evidence_document("execution_plan", plan)
        contract.validate_evidence_document(
            "consumed_authorization",
            authorization_record,
        )
        contract._validate_gpu_assignments(plan["gpu_assignments"])
        contract._validate_case_port_pairs(plan["case_port_pairs"])
        contract._validate_artifact_paths(plan["artifact_paths"])
        if authorization_record["execution_plan_sha256"] != (
            contract.canonical_json_sha256(plan)
        ):
            raise ValueError("authorization execution plan mismatch")
        for field in (
            "run_tag",
            "nonce",
            *contract.EXECUTION_PROVENANCE_FIELDS,
            "required_gpu_indices",
            "world_size",
            "gpu_assignments",
            "case_port_pairs",
            "artifact_paths",
        ):
            if authorization_record[field] != plan[field]:
                raise ValueError(f"authorization binding mismatch: {field}")
        canonical_commands = contract.canonical_execution_commands(plan)
        contract.validate_execution_command_semantics(
            canonical_commands,
            expected_order=contract.EXECUTION_COMMAND_ORDER,
            execution_plan=plan,
        )
        command_manifest = [
            {
                "name": name,
                "command_sha256": contract.execution_command_sha256(
                    canonical_commands[name]
                ),
            }
            for name in contract.EXECUTION_COMMAND_ORDER
        ]
        if plan["command_manifest_sha256"] != (
            contract.canonical_json_sha256(command_manifest)
        ):
            raise ValueError("canonical command manifest mismatch")
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("canonical plan or authorization is invalid") from error
    _validate_bound_output(
        plan,
        artifact_root,
        run_dir,
        detached_receipt_path,
    )
    order = list(contract.EXECUTION_COMMAND_ORDER)

    results = []
    failed_name = None
    for index, name in enumerate(order):
        command = canonical_commands[name]
        raw = command_runner(
            name=name,
            argv=command,
            timeout_seconds=contract.EXECUTION_COMMAND_TIMEOUT_SECONDS[
                name
            ],
            env=REQUIRED_EXECUTION_ENV,
        )
        stdout, stdout_truncated = _bounded(raw.get("stdout"))
        stderr, stderr_truncated = _bounded(raw.get("stderr"))
        returncode = raw.get("returncode")
        row = {
            "name": name,
            "command_sha256": contract.execution_command_sha256(command),
            "outcome": "attempted",
            "returncode": returncode,
            "stdout": stdout,
            "stderr": stderr,
            "stdout_truncated": stdout_truncated,
            "stderr_truncated": stderr_truncated,
        }
        results.append(row)
        if returncode != 0:
            failed_name = name
            for skipped in order[index + 1:]:
                results.append(
                    {
                        "name": skipped,
                        "command_sha256": contract.execution_command_sha256(
                            canonical_commands[skipped]
                        ),
                        "outcome": "skipped",
                        "returncode": None,
                        "stdout": "",
                        "stderr": "",
                        "stdout_truncated": False,
                        "stderr_truncated": False,
                    }
                )
            break

    payload = receipt_builder(
        plan=plan,
        authorization_record=authorization_record,
        command_results=results,
        detached_receipt_path=detached_receipt_path,
    )
    try:
        contract.validate_execution_evidence_bundle(payload)
    except (TypeError, ValueError) as error:
        raise ValueError(
            "execution evidence bundle is invalid"
        ) from error
    if failed_name is not None:
        _write_failure(
            failure_path,
            payload,
            plan=plan,
            artifact_root=artifact_root,
            run_dir=run_dir,
        )
        raise ValueError(f"{failed_name} command failed")
    published = receipt_module.publish_execution_evidence_bundle(
        bundle=payload,
        artifact_root=artifact_root,
        run_dir=run_dir,
        output_path=detached_receipt_path,
    )
    output = Path(detached_receipt_path)
    if not output.is_file() or output.is_symlink():
        raise ValueError(
            "published receipt is missing or is not a regular file"
        )
    expected = contract.canonical_json_bytes(payload) + b"\n"
    try:
        actual = output.read_bytes()
    except OSError as error:
        raise ValueError("published receipt is unreadable") from error
    if actual != expected:
        try:
            output.unlink()
        except OSError:
            pass
        raise ValueError(
            "published receipt bytes do not match validated bundle"
        )
    return published
