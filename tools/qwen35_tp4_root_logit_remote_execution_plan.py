from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
import tarfile
import tempfile
from pathlib import PurePosixPath


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


runner = _load_module(
    "run_qwen35_tp4_real_root_logit_gate_remote_for_execution_plan",
    "run_qwen35_tp4_real_root_logit_gate_remote.py",
)
contract = _load_module(
    "qwen35_tp4_hybrid_prefix_benchmark_contract_for_root_execution_plan",
    "qwen35_tp4_hybrid_prefix_benchmark_contract.py",
)
resource_policy_module = _load_module(
    "qwen35_tp4_correctness_resource_policy_for_root_execution_plan",
    "qwen35_tp4_correctness_resource_policy.py",
)


SCHEMA_VERSION = (
    "qwen35.tp4-root-logit-remote-execution-plan.v1"
)
PLAN_NAME = "remote_execution_plan.json"
STAGE_ORDER = ["preflight", "run", "download", "verify"]
MIN_GPU_FREE_BYTES = 24 * 1024**3
RESOURCE_BASELINE_NAME = "resource_baseline.json"


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


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


def _load_json(path):
    path = Path(path)
    if not path.is_file() or path.is_symlink():
        raise ValueError("root execution plan is missing")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("root execution plan is invalid") from error


def _frozen_source_identity(repo_root, source_tag):
    source_tag = runner.validate_run_tag(source_tag)
    source_dir = (
        Path(repo_root).resolve()
        / runner.LOCAL_RUN_ROOT
        / source_tag
    )
    source_tar = source_dir / "source.tar.gz"
    if not source_tar.is_file() or source_tar.is_symlink():
        raise ValueError("root frozen source bundle is missing")
    sidecar_path = source_dir / "source_preparation.json"
    if not sidecar_path.is_file():
        sidecar_path = source_dir / "source_prep.json"
    sidecar = _load_json(sidecar_path)
    sidecar_tag = sidecar.get("source_tag", sidecar.get("tag"))
    sidecar_tree = sidecar.get(
        "root_source_tree_sha256",
        sidecar.get("source_tree_sha256"),
    )
    if sidecar_tag != source_tag:
        raise ValueError("root frozen source tag mismatch")
    tar_sha256 = sidecar.get("source_tar_sha256")
    if tar_sha256 is None:
        tar_sha_path = source_dir / "bundle_tar_sha256.txt"
        if (
            not tar_sha_path.is_file()
            or tar_sha_path.is_symlink()
        ):
            raise ValueError("root frozen source tar identity is missing")
        tar_sha256 = tar_sha_path.read_text(
            encoding="utf-8"
        ).strip()
    if _sha256(source_tar) != tar_sha256:
        raise ValueError("root frozen source tar identity mismatch")

    try:
        with tarfile.open(source_tar, "r:gz") as archive:
            members = archive.getmembers()
            names = [member.name for member in members]
            if len(names) != len(set(names)):
                raise ValueError(
                    "root frozen source inventory is duplicated"
                )
            manifest_name = "source/source_manifest.input.json"
            if manifest_name not in names:
                raise ValueError(
                    "root frozen source manifest is missing"
                )
            hashes = {}
            manifest = None
            for member in members:
                path = PurePosixPath(member.name)
                if (
                    path.is_absolute()
                    or ".." in path.parts
                    or not member.isfile()
                    or member.issym()
                    or member.islnk()
                ):
                    raise ValueError(
                        "root frozen source inventory is unsafe"
                    )
                stream = archive.extractfile(member)
                if stream is None:
                    raise ValueError(
                        "root frozen source member is unreadable"
                    )
                payload = stream.read()
                if member.name == manifest_name:
                    try:
                        manifest = json.loads(payload)
                    except (
                        UnicodeDecodeError,
                        json.JSONDecodeError,
                    ) as error:
                        raise ValueError(
                            "root frozen source manifest is invalid"
                        ) from error
                    continue
                if not member.name.startswith("source/"):
                    raise ValueError(
                        "root frozen source inventory is unsafe"
                    )
                hashes[member.name.removeprefix("source/")] = (
                    hashlib.sha256(payload).hexdigest()
                )
    except (OSError, tarfile.TarError) as error:
        raise ValueError("root frozen source bundle is invalid") from error
    if (
        not isinstance(manifest, dict)
        or manifest.get("source_file_sha256")
        != dict(sorted(hashes.items()))
    ):
        raise ValueError("root frozen source closure mismatch")
    tree_sha256 = hashlib.sha256(
        json.dumps(
            dict(sorted(hashes.items())),
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    if (
        manifest.get("source_tree_sha256") != tree_sha256
        or sidecar_tree != tree_sha256
    ):
        raise ValueError("root frozen source identity mismatch")
    return source_tag, tree_sha256


def _stage_inputs(
    run_tag,
    repo_root,
    local_run_dir,
    remote_run_dir,
    *,
    resource_binding=None,
    frozen_source_tree_sha256,
):
    common = {
        "run_tag": run_tag,
        "repo_root": str(repo_root),
    }
    result = {
        "preflight": dict(common),
        "run": {
            **common,
            "remote_run_dir": remote_run_dir,
            "frozen_source_tree_sha256": frozen_source_tree_sha256,
        },
        "download": {
            **common,
            "remote_run_dir": remote_run_dir,
            "local_artifact_dir": str(local_run_dir / "artifacts"),
            "exact_artifact_names": sorted(
                runner.EXACT_ARTIFACT_NAMES
            ),
        },
        "verify": {
            **common,
            "local_artifact_dir": str(local_run_dir / "artifacts"),
            "independent_verification_path": str(
                local_run_dir / "independent_verification.json"
            ),
            "frozen_source_tree_sha256": frozen_source_tree_sha256,
            "model_manifest_sha256": (
                contract.MODEL_MANIFEST_SHA256
            ),
        },
    }
    if resource_binding is not None:
        for name in ("preflight", "run"):
            result[name].update(resource_binding)
    return result


def _resource_binding(
    *,
    output_dir,
    resource_policy,
    resource_baseline_path,
):
    if resource_policy == resource_policy_module.STRICT_EXCLUSIVE:
        if resource_baseline_path is not None:
            raise ValueError(
                "strict resource policy does not accept a baseline"
            )
        return None
    if resource_policy != resource_policy_module.CONTROLLED_SHARED:
        raise ValueError("resource policy is unsupported")
    if resource_baseline_path is None:
        raise ValueError("controlled resource baseline is required")
    baseline_path = Path(resource_baseline_path).resolve()
    baseline = resource_policy_module.validate_baseline_manifest(
        baseline_path,
        ssh_target=runner.REMOTE_TARGET,
        gpu_indices=[2, 4, 5, 6],
    )
    return {
        "resource_policy": resource_policy,
        "resource_baseline_path": str(
            Path(output_dir).resolve() / RESOURCE_BASELINE_NAME
        ),
        "resource_baseline_sha256": (
            resource_policy_module.sha256(baseline_path)
        ),
        "gpu_indices": list(baseline["gpu_indices"]),
        "gpu_uuids": [
            row["gpu_uuid"] for row in baseline["selected"]
        ],
        "benchmark_execution_authorized": False,
        "_source_path": baseline_path,
    }


def _payload(
    *,
    repo_root,
    output_dir,
    run_tag,
    resource_policy=resource_policy_module.STRICT_EXCLUSIVE,
    resource_baseline_path=None,
    frozen_source_tag=None,
    frozen_source_tree_sha256=None,
):
    repo_root = Path(repo_root).resolve()
    output_dir = Path(output_dir).resolve()
    run_tag = runner.validate_run_tag(run_tag)
    if frozen_source_tag is None:
        frozen_source_tag = runner.FROZEN_SOURCE_TAG
    if frozen_source_tree_sha256 is None:
        frozen_source_tree_sha256 = (
            runner.FROZEN_SOURCE_TREE_SHA256
        )
    local_run_dir = (
        repo_root / runner.LOCAL_RUN_ROOT / run_tag
    )
    remote_run_dir = runner.remote_run_dir(run_tag)
    binding = _resource_binding(
        output_dir=output_dir,
        resource_policy=resource_policy,
        resource_baseline_path=resource_baseline_path,
    )
    stage_binding = None
    if binding is not None:
        stage_binding = {
            key: value
            for key, value in binding.items()
            if key != "_source_path"
        }
    payload = {
        "schema_version": SCHEMA_VERSION,
        "run_tag": run_tag,
        "repo_root": str(repo_root),
        "local_run_dir": str(local_run_dir),
        "ssh_target": runner.REMOTE_TARGET,
        "remote_run_dir": remote_run_dir,
        "frozen_source_tag": frozen_source_tag,
        "frozen_source_tree_sha256": frozen_source_tree_sha256,
        "model_manifest_sha256": contract.MODEL_MANIFEST_SHA256,
        "exact_artifact_names": sorted(
            runner.EXACT_ARTIFACT_NAMES
        ),
        "minimum_free_bytes_per_gpu": MIN_GPU_FREE_BYTES,
        "requires_no_active_compute_processes": (
            binding is None
        ),
        "stage_order": list(STAGE_ORDER),
        "stage_inputs": _stage_inputs(
            run_tag,
            repo_root,
            local_run_dir,
            remote_run_dir,
            resource_binding=stage_binding,
            frozen_source_tree_sha256=frozen_source_tree_sha256,
        ),
        "execution_performed": False,
        "claim_boundary": (
            "execution authorization only; no SSH, GPU, correctness, "
            "performance, cache, memory, compression, or quality claim"
        ),
        "plan_output_dir": str(output_dir),
    }
    if binding is not None:
        payload.update(stage_binding)
    return payload


def build_remote_execution_plan(
    *,
    repo_root,
    output_dir,
    run_tag,
    resource_policy=resource_policy_module.STRICT_EXCLUSIVE,
    resource_baseline_path=None,
):
    output_dir = Path(output_dir).resolve()
    if output_dir.exists():
        raise ValueError("root execution plan output already exists")
    payload = _payload(
        repo_root=repo_root,
        output_dir=output_dir,
        run_tag=run_tag,
        resource_policy=resource_policy,
        resource_baseline_path=resource_baseline_path,
    )
    if Path(payload["local_run_dir"]).exists():
        raise ValueError("root local run directory already exists")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir()
    try:
        if resource_policy == resource_policy_module.CONTROLLED_SHARED:
            shutil.copyfile(
                Path(resource_baseline_path).resolve(),
                output_dir / RESOURCE_BASELINE_NAME,
            )
        _write_json(output_dir / PLAN_NAME, payload)
        return verify_remote_execution_plan(output_dir / PLAN_NAME)
    except BaseException:
        if output_dir.exists():
            for path in output_dir.iterdir():
                path.unlink()
            output_dir.rmdir()
        raise


def verify_remote_execution_plan(path):
    path = Path(path).resolve()
    payload = _load_json(path)
    resource_policy = payload.get(
        "resource_policy",
        resource_policy_module.STRICT_EXCLUSIVE,
    ) if isinstance(payload, dict) else None
    resource_baseline_path = None
    if resource_policy == resource_policy_module.CONTROLLED_SHARED:
        resource_baseline_path = payload.get(
            "resource_baseline_path"
        )
    frozen_source_tag, frozen_source_tree_sha256 = (
        _frozen_source_identity(
            payload.get("repo_root", ""),
            payload.get("frozen_source_tag", ""),
        )
    )
    expected_keys = set(_payload(
        repo_root=payload.get("repo_root", ""),
        output_dir=path.parent,
        run_tag=payload.get("run_tag", ""),
        resource_policy=resource_policy,
        resource_baseline_path=resource_baseline_path,
        frozen_source_tag=frozen_source_tag,
        frozen_source_tree_sha256=frozen_source_tree_sha256,
    ))
    if (
        not isinstance(payload, dict)
        or set(payload) != expected_keys
        or payload.get("schema_version") != SCHEMA_VERSION
        or payload.get("execution_performed") is not False
    ):
        raise ValueError("root execution plan schema mismatch")
    expected = _payload(
        repo_root=payload["repo_root"],
        output_dir=path.parent,
        run_tag=payload["run_tag"],
        resource_policy=resource_policy,
        resource_baseline_path=resource_baseline_path,
        frozen_source_tag=frozen_source_tag,
        frozen_source_tree_sha256=frozen_source_tree_sha256,
    )
    if payload["frozen_source_tree_sha256"] != expected[
        "frozen_source_tree_sha256"
    ]:
        raise ValueError("root execution plan source identity mismatch")
    if payload["model_manifest_sha256"] != expected[
        "model_manifest_sha256"
    ]:
        raise ValueError("root execution plan model identity mismatch")
    if (
        payload["stage_order"] != STAGE_ORDER
        or payload["stage_inputs"] != expected["stage_inputs"]
    ):
        raise ValueError("root execution plan stage mismatch")
    if payload != expected:
        raise ValueError("root execution plan binding mismatch")
    return payload
