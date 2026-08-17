from __future__ import annotations

from dataclasses import dataclass
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile


AUTHORITY_NAMES = (
    "tp4_root_logit",
    "cached_continuation",
    "engine_correctness",
)


@dataclass(frozen=True)
class RealAuthorityRun:
    name: str
    run_tag: str
    authority_dir: Path
    plan_path: Path | None = None
    consumed_authorization_path: Path | None = None
    receipt_path: Path | None = None


@dataclass(frozen=True)
class VerifierDependencies:
    root_verify: object
    root_plan_verify: object
    root_receipt_verify: object
    cached_verify: object
    engine_authority_verify: object
    engine_gate_verify: object
    cached_plan_verify: object
    cached_receipt_verify: object
    engine_plan_verify: object
    engine_receipt_verify: object


def _load_module(module_name, filename):
    module = sys.modules.get(module_name)
    if module is not None:
        return module
    path = Path(__file__).resolve().parent / filename
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _default_verifiers():
    root = _load_module(
        "verify_qwen35_tp4_real_root_logit_correctness_gate",
        "verify_qwen35_tp4_real_root_logit_correctness_gate.py",
    )
    root_plan = _load_module(
        "qwen35_tp4_root_logit_remote_execution_plan",
        "qwen35_tp4_root_logit_remote_execution_plan.py",
    )
    root_receipt = _load_module(
        "qwen35_tp4_root_logit_remote_execution_receipt",
        "qwen35_tp4_root_logit_remote_execution_receipt.py",
    )
    cached = _load_module(
        "verify_qwen35_tp4_cached_continuation_correctness_gate",
        "verify_qwen35_tp4_cached_continuation_correctness_gate.py",
    )
    engine_authority = _load_module(
        "verify_qwen35_tp4_engine_correctness_authority",
        "verify_qwen35_tp4_engine_correctness_authority.py",
    )
    engine_gate = _load_module(
        "verify_qwen35_tp4_engine_correctness_gate",
        "verify_qwen35_tp4_engine_correctness_gate.py",
    )
    cached_plan = _load_module(
        "qwen35_tp4_cached_continuation_remote_execution_plan",
        "qwen35_tp4_cached_continuation_remote_execution_plan.py",
    )
    cached_receipt = _load_module(
        "qwen35_tp4_cached_continuation_remote_execution_receipt",
        "qwen35_tp4_cached_continuation_remote_execution_receipt.py",
    )
    engine_plan = _load_module(
        "qwen35_tp4_engine_remote_execution_plan",
        "qwen35_tp4_engine_remote_execution_plan.py",
    )
    engine_receipt = _load_module(
        "qwen35_tp4_engine_remote_execution_receipt",
        "qwen35_tp4_engine_remote_execution_receipt.py",
    )
    return VerifierDependencies(
        root_verify=root.verify_run,
        root_plan_verify=root_plan.verify_remote_execution_plan,
        root_receipt_verify=lambda **paths: (
            root_receipt.verify_receipt_files(
                **paths,
                plan_verifier=root_plan.verify_remote_execution_plan,
            )
        ),
        cached_verify=cached.verify_run,
        engine_authority_verify=engine_authority.verify_run,
        engine_gate_verify=engine_gate.verify_run,
        cached_plan_verify=cached_plan.verify_remote_execution_plan,
        cached_receipt_verify=cached_receipt.verify_receipt_files,
        engine_plan_verify=engine_plan.verify_remote_execution_plan,
        engine_receipt_verify=engine_receipt.verify_receipt_files,
    )


_VERIFIERS = None


def _builder_module():
    _load_module(
        "qwen35_tp4_hybrid_prefix_benchmark_contract",
        "qwen35_tp4_hybrid_prefix_benchmark_contract.py",
    )
    return _load_module(
        "build_qwen35_tp4_performance_prerequisites",
        "build_qwen35_tp4_performance_prerequisites.py",
    )


def _contract_module():
    return _load_module(
        "qwen35_tp4_hybrid_prefix_benchmark_contract",
        "qwen35_tp4_hybrid_prefix_benchmark_contract.py",
    )


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _regular_file(path, label):
    if path is None:
        raise ValueError(f"{label} is required")
    path = Path(path)
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"{label} must be a regular file")
    return path.resolve()


def _authority_directory(path, label):
    path = Path(path)
    if not path.is_dir() or path.is_symlink():
        raise ValueError(f"{label} must be a regular directory")
    return path.resolve()


def _load_json(path, label):
    path = _regular_file(path, label)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is invalid") from error
    if not isinstance(payload, dict):
        raise ValueError(f"{label} is invalid")
    return payload


def _write_json(path, payload):
    path = Path(path)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        json.dump(
            payload,
            handle,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _require_pass(payload, label):
    if (
        not isinstance(payload, dict)
        or payload.get("classification") != "PASS"
    ):
        raise ValueError(f"{label} did not prove PASS")
    return payload


def _require_download_binding(plan, authority_dir, label):
    candidates = set()
    try:
        local_artifact_dir = plan["stage_inputs"]["verify"][
            "local_artifact_dir"
        ]
    except (KeyError, TypeError):
        local_artifact_dir = None
    if isinstance(local_artifact_dir, (str, os.PathLike)):
        artifact_path = Path(local_artifact_dir)
        if artifact_path.is_absolute():
            candidates.add(str(artifact_path.resolve()))
    try:
        argv = plan["commands"]["local_verify"]["argv"]
    except (KeyError, TypeError):
        argv = None
    if argv is not None:
        if not isinstance(argv, list):
            raise ValueError(f"{label} plan local verifier is invalid")
        for value in argv:
            if isinstance(value, (str, os.PathLike)):
                text = str(value)
                if Path(text).is_absolute():
                    candidates.add(str(Path(text).resolve()))
    if not candidates:
        raise ValueError(f"{label} plan local verifier is invalid")
    if str(authority_dir.resolve()) not in candidates:
        raise ValueError(
            f"{label} authority directory is not plan-bound"
        )


def _receipt_bound(run, authority_dir, *, plan_verify, receipt_verify):
    plan_path = _regular_file(run.plan_path, f"{run.name} plan")
    authorization_path = _regular_file(
        run.consumed_authorization_path,
        f"{run.name} consumed authorization",
    )
    receipt_path = _regular_file(
        run.receipt_path,
        f"{run.name} execution receipt",
    )
    plan = plan_verify(plan_path)
    summary = _require_pass(
        receipt_verify(
            plan_path=plan_path,
            receipt_path=receipt_path,
            authorization_path=authorization_path,
        ),
        f"{run.name} execution receipt",
    )
    _require_download_binding(
        plan,
        authority_dir,
        run.name,
    )
    if summary.get("run_tag") != run.run_tag:
        raise ValueError(f"{run.name} run tag mismatch")
    return plan, summary, receipt_path


def _provenance_payload(
    *,
    name,
    run_tag,
    source_tree_sha256,
    plan_path=None,
    plan_sha256=None,
    authorization_path=None,
    authorization_sha256=None,
    receipt_path=None,
    receipt_sha256=None,
):
    contract = _contract_module()
    if not all(
        isinstance(value, str) and len(value) == 64
        for value in (
            plan_sha256,
            authorization_sha256,
            receipt_sha256,
        )
    ):
        raise ValueError(f"{name} receipt provenance is incomplete")
    return {
        "schema_version": (
            contract.PREREQUISITE_PROVENANCE_SCHEMA_VERSION
        ),
        "authority_name": name,
        "run_tag": run_tag,
        "binding_kind": "remote_execution_receipt",
        "source_tree_sha256": source_tree_sha256,
        "model_manifest_sha256": contract.MODEL_MANIFEST_SHA256,
        "root_logit_receipt_gap": False,
        "plan_path": plan_path,
        "plan_sha256": plan_sha256,
        "authorization_path": authorization_path,
        "authorization_sha256": authorization_sha256,
        "receipt_path": receipt_path,
        "receipt_sha256": receipt_sha256,
    }


def adapt_real_authorities(*, runs, verification_output_dir):
    if not isinstance(runs, (tuple, list)):
        raise ValueError("real authority inventory is invalid")
    by_name = {}
    for run in runs:
        if (
            not isinstance(run, RealAuthorityRun)
            or run.name not in AUTHORITY_NAMES
            or run.name in by_name
        ):
            raise ValueError("real authority inventory is invalid")
        _authority_directory(
            run.authority_dir,
            f"{run.name} authority directory",
        )
        by_name[run.name] = run
    if set(by_name) != set(AUTHORITY_NAMES):
        raise ValueError("real authority inventory is invalid")
    verifiers = _VERIFIERS or _default_verifiers()
    verification_output_dir = Path(verification_output_dir)
    if verification_output_dir.exists():
        raise ValueError("verification output already exists")
    verification_output_dir.mkdir(parents=True)
    builder = _builder_module()
    rows = []
    try:
        root_run = by_name["tp4_root_logit"]
        root_dir = _authority_directory(
            root_run.authority_dir,
            "tp4_root_logit authority directory",
        )
        _, root_receipt, root_receipt_path = _receipt_bound(
            root_run,
            root_dir,
            plan_verify=verifiers.root_plan_verify,
            receipt_verify=verifiers.root_receipt_verify,
        )
        root_verification = _require_pass(
            verifiers.root_verify(root_dir),
            "tp4_root_logit independent verification",
        )
        root_manifest = _load_json(
            root_dir / "source_manifest.json",
            "tp4_root_logit source manifest",
        )
        root_source = root_manifest.get("source_tree_sha256")
        if root_receipt.get("source_tree_sha256") != root_source:
            raise ValueError(
                "tp4_root_logit source_tree_sha256 mismatch"
            )
        if (
            root_receipt.get("model_manifest_sha256")
            != _contract_module().MODEL_MANIFEST_SHA256
        ):
            raise ValueError(
                "tp4_root_logit model_manifest_sha256 mismatch"
            )
        root_artifact = _regular_file(
            root_dir / "tp4_real_root_logit_correctness.json",
            "tp4_root_logit artifact",
        )
        root_verification_path = (
            verification_output_dir / "tp4_root_logit.json"
        )
        root_provenance_path = (
            verification_output_dir / "tp4_root_logit.provenance.json"
        )
        root_plan_copy = (
            verification_output_dir
            / "tp4_root_logit.execution_plan.json"
        )
        root_authorization_copy = (
            verification_output_dir
            / "tp4_root_logit.consumed_authorization.json"
        )
        root_receipt_copy = (
            verification_output_dir
            / "tp4_root_logit.execution_receipt.json"
        )
        shutil.copyfile(root_run.plan_path, root_plan_copy)
        shutil.copyfile(
            root_run.consumed_authorization_path,
            root_authorization_copy,
        )
        shutil.copyfile(root_receipt_path, root_receipt_copy)
        _write_json(root_verification_path, root_verification)
        _write_json(
            root_provenance_path,
            _provenance_payload(
                name="tp4_root_logit",
                run_tag=root_run.run_tag,
                source_tree_sha256=root_source,
                plan_path=root_plan_copy.name,
                plan_sha256=_sha256(root_plan_copy),
                authorization_path=root_authorization_copy.name,
                authorization_sha256=_sha256(
                    root_authorization_copy
                ),
                receipt_path=root_receipt_copy.name,
                receipt_sha256=_sha256(root_receipt_copy),
            ),
        )
        rows.append(builder.AuthorityInput(
            name="tp4_root_logit",
            run_tag=root_run.run_tag,
            source_tree_sha256=root_source,
            artifact_path=root_artifact,
            artifact_sha256=_sha256(root_artifact),
            independent_verification_path=root_verification_path,
            independent_verification_sha256=_sha256(
                root_verification_path
            ),
            provenance_path=root_provenance_path,
            provenance_sha256=_sha256(root_provenance_path),
        ))

        cached_run = by_name["cached_continuation"]
        cached_dir = _authority_directory(
            cached_run.authority_dir,
            "cached_continuation authority directory",
        )
        _, cached_receipt, cached_receipt_path = _receipt_bound(
            cached_run,
            cached_dir,
            plan_verify=verifiers.cached_plan_verify,
            receipt_verify=verifiers.cached_receipt_verify,
        )
        cached_verification = _require_pass(
            verifiers.cached_verify(cached_dir),
            "cached_continuation independent verification",
        )
        for name in (
            "source_tree_sha256",
            "model_manifest_sha256",
            "workload_manifest_sha256",
        ):
            if cached_receipt.get(name) != cached_verification.get(name):
                raise ValueError(
                    f"cached_continuation {name} mismatch"
                )
        cached_artifact = _regular_file(
            cached_dir / "cached_continuation_correctness.json",
            "cached_continuation artifact",
        )
        cached_verification_path = (
            verification_output_dir / "cached_continuation.json"
        )
        cached_provenance_path = (
            verification_output_dir
            / "cached_continuation.provenance.json"
        )
        cached_plan_copy = (
            verification_output_dir
            / "cached_continuation.execution_plan.json"
        )
        cached_authorization_copy = (
            verification_output_dir
            / "cached_continuation.consumed_authorization.json"
        )
        cached_receipt_copy = (
            verification_output_dir
            / "cached_continuation.execution_receipt.json"
        )
        shutil.copyfile(cached_run.plan_path, cached_plan_copy)
        shutil.copyfile(
            cached_run.consumed_authorization_path,
            cached_authorization_copy,
        )
        shutil.copyfile(cached_receipt_path, cached_receipt_copy)
        _write_json(cached_verification_path, cached_verification)
        _write_json(
            cached_provenance_path,
            _provenance_payload(
                name="cached_continuation",
                run_tag=cached_run.run_tag,
                source_tree_sha256=cached_verification[
                    "source_tree_sha256"
                ],
                plan_path=cached_plan_copy.name,
                plan_sha256=_sha256(cached_plan_copy),
                authorization_path=cached_authorization_copy.name,
                authorization_sha256=_sha256(
                    cached_authorization_copy
                ),
                receipt_path=cached_receipt_copy.name,
                receipt_sha256=_sha256(cached_receipt_copy),
            ),
        )
        rows.append(builder.AuthorityInput(
            name="cached_continuation",
            run_tag=cached_run.run_tag,
            source_tree_sha256=cached_verification[
                "source_tree_sha256"
            ],
            artifact_path=cached_artifact,
            artifact_sha256=_sha256(cached_artifact),
            independent_verification_path=cached_verification_path,
            independent_verification_sha256=_sha256(
                cached_verification_path
            ),
            provenance_path=cached_provenance_path,
            provenance_sha256=_sha256(cached_provenance_path),
        ))

        engine_run = by_name["engine_correctness"]
        engine_dir = _authority_directory(
            engine_run.authority_dir,
            "engine_correctness authority directory",
        )
        _, engine_receipt, engine_receipt_path = _receipt_bound(
            engine_run,
            engine_dir,
            plan_verify=verifiers.engine_plan_verify,
            receipt_verify=verifiers.engine_receipt_verify,
        )
        engine_authority = _require_pass(
            verifiers.engine_authority_verify(engine_dir),
            "engine_correctness authority verification",
        )
        engine_verification = _require_pass(
            verifiers.engine_gate_verify(
                engine_dir / "engine_authority"
            ),
            "engine_correctness independent verification",
        )
        for name in (
            "source_tree_sha256",
            "model_manifest_sha256",
        ):
            values = {
                engine_receipt.get(name),
                engine_authority.get(name),
                engine_verification.get(name),
            }
            if len(values) != 1:
                raise ValueError(f"engine_correctness {name} mismatch")
        if (
            engine_receipt.get("workload_manifest_sha256")
            != engine_authority.get("workload_manifest_sha256")
        ):
            raise ValueError(
                "engine_correctness workload_manifest_sha256 mismatch"
            )
        engine_artifact = _regular_file(
            engine_dir
            / "engine_authority"
            / "engine_correctness.json",
            "engine_correctness artifact",
        )
        engine_verification_path = (
            verification_output_dir / "engine_correctness.json"
        )
        engine_provenance_path = (
            verification_output_dir / "engine_correctness.provenance.json"
        )
        engine_plan_copy = (
            verification_output_dir
            / "engine_correctness.execution_plan.json"
        )
        engine_authorization_copy = (
            verification_output_dir
            / "engine_correctness.consumed_authorization.json"
        )
        engine_receipt_copy = (
            verification_output_dir
            / "engine_correctness.execution_receipt.json"
        )
        shutil.copyfile(engine_run.plan_path, engine_plan_copy)
        shutil.copyfile(
            engine_run.consumed_authorization_path,
            engine_authorization_copy,
        )
        shutil.copyfile(engine_receipt_path, engine_receipt_copy)
        _write_json(engine_verification_path, engine_verification)
        _write_json(
            engine_provenance_path,
            _provenance_payload(
                name="engine_correctness",
                run_tag=engine_run.run_tag,
                source_tree_sha256=engine_verification[
                    "source_tree_sha256"
                ],
                plan_path=engine_plan_copy.name,
                plan_sha256=_sha256(engine_plan_copy),
                authorization_path=engine_authorization_copy.name,
                authorization_sha256=_sha256(
                    engine_authorization_copy
                ),
                receipt_path=engine_receipt_copy.name,
                receipt_sha256=_sha256(engine_receipt_copy),
            ),
        )
        rows.append(builder.AuthorityInput(
            name="engine_correctness",
            run_tag=engine_run.run_tag,
            source_tree_sha256=engine_verification[
                "source_tree_sha256"
            ],
            artifact_path=engine_artifact,
            artifact_sha256=_sha256(engine_artifact),
            independent_verification_path=engine_verification_path,
            independent_verification_sha256=_sha256(
                engine_verification_path
            ),
            provenance_path=engine_provenance_path,
            provenance_sha256=_sha256(engine_provenance_path),
        ))
        return tuple(rows)
    except BaseException:
        for path in verification_output_dir.iterdir():
            path.unlink()
        verification_output_dir.rmdir()
        raise
