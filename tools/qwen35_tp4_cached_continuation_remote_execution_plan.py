from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import os
from pathlib import Path
import shlex
import shutil
import sys
import tempfile


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


engine_plan = _load_module(
    "qwen35_tp4_engine_remote_execution_plan",
    "qwen35_tp4_engine_remote_execution_plan.py",
)
source_runner = engine_plan.source_runner


SCHEMA_VERSION = (
    "qwen35.tp4-cached-continuation-remote-execution-plan.v1"
)
SSH_TARGET = engine_plan.SSH_TARGET
REMOTE_PYTHON = engine_plan.REMOTE_PYTHON
REMOTE_ROOT = (
    "/data00/home/sitian/sitian-workspace01/tllm/"
    "qwen35-tp4-cached-continuation-authority-runs"
)
MIN_GPU_FREE_BYTES = engine_plan.MIN_GPU_FREE_BYTES
PLAN_NAME = "remote_execution_plan.json"
REMOTE_CONFIGURATION_NAME = engine_plan.REMOTE_CONFIGURATION_NAME
SOURCE_TAR_NAME = engine_plan.SOURCE_TAR_NAME
DOWNLOADED_AUTHORITY_NAME = "downloaded_cached_authority"
LOCAL_VERIFIER_SOURCE_NAME = engine_plan.LOCAL_VERIFIER_SOURCE_NAME
RESOURCE_BASELINE_NAME = engine_plan.RESOURCE_BASELINE_NAME
resource_policy_module = engine_plan.resource_policy_module
EXACT_PACKAGE_ENTRIES = (
    "cached_continuation_authority",
    "cached_continuation_independent_verification.json",
)
REQUIRED_AUTHORITY_SOURCES = {
    "tools/run_qwen35_tp4_cached_continuation_authority.py",
    "tools/verify_qwen35_tp4_cached_continuation_correctness_gate.py",
    "tools/qwen35_tp4_cached_continuation_remote_execution_plan.py",
    "tools/qwen35_tp4_cached_continuation_remote_execution_receipt.py",
    "tools/qwen35_tp4_cached_continuation_remote_execution_executor.py",
    "tools/qwen35_tp4_engine_remote_execution_source_contract.py",
    "tools/qwen35_tp4_engine_remote_execution_authorization.py",
    "tools/qwen35_tp4_engine_remote_subprocess_adapter.py",
}
COMMAND_ORDER = [
    "reserve_remote",
    "upload",
    "stage",
    "resource_guard",
    "guarded_authority",
    "package_download",
    "safe_extract",
    "prepare_local_verifier",
    "local_verify",
]


_sha256 = engine_plan._sha256
_write_json = engine_plan._write_json
_load_json = engine_plan._load_json
_safe_run_tag = engine_plan._safe_run_tag
_require_remote_absolute = engine_plan._require_remote_absolute
_ssh = engine_plan._ssh
_scp = engine_plan._scp
_configuration_from_payload = engine_plan._configuration_from_payload
_load_source_inventory = engine_plan._load_source_inventory
_resource_guard_command = engine_plan._resource_guard_command
_guarded_authority_command = engine_plan._guarded_authority_command
_stage_script = engine_plan._stage_script
_prepare_local_verifier_command = (
    engine_plan._prepare_local_verifier_command
)


def _package_script(remote_authority_root):
    entries = " ".join(EXACT_PACKAGE_ENTRIES)
    return " && ".join([
        "set -eu",
        f"cd {shlex.quote(remote_authority_root)}",
        "test \"$(find . -mindepth 1 -maxdepth 1 | wc -l)\" -eq 2",
        "test -d cached_continuation_authority",
        (
            "test -f "
            "cached_continuation_independent_verification.json"
        ),
        f"tar -cf - {entries}",
    ])


def _extract_command(local_tar, destination):
    script = "\n".join([
        "import sys,tarfile",
        "from pathlib import Path,PurePosixPath",
        "archive=Path(sys.argv[1])",
        "destination=Path(sys.argv[2])",
        f"expected={list(EXACT_PACKAGE_ENTRIES)!r}",
        "if destination.exists(): raise SystemExit('download destination exists')",
        "with tarfile.open(archive,'r:') as handle:",
        " members=handle.getmembers()",
        " roots=sorted({PurePosixPath(member.name).parts[0] for member in members})",
        " if roots!=sorted(expected): raise SystemExit('cached authority inventory mismatch')",
        " for member in members:",
        "  path=PurePosixPath(member.name)",
        "  if path.is_absolute() or '..' in path.parts or member.issym() or member.islnk():",
        "   raise SystemExit('unsafe cached authority tar member')",
        " destination.mkdir()",
        " handle.extractall(destination,members=members)",
    ])
    return [sys.executable, "-c", script, str(local_tar), str(destination)]


def _local_verify_command(verifier_source, downloaded_authority):
    script = "\n".join([
        "import importlib.util,json,sys",
        "from pathlib import Path",
        "source=Path(sys.argv[1])",
        "authority=Path(sys.argv[2])",
        "remote_path=Path(sys.argv[3])",
        "spec=importlib.util.spec_from_file_location('cached_verifier',source)",
        "module=importlib.util.module_from_spec(spec)",
        "sys.modules[spec.name]=module",
        "spec.loader.exec_module(module)",
        "local=module.verify_run(authority)",
        "remote=json.loads(remote_path.read_text(encoding='utf-8'))",
        "if local!=remote: raise SystemExit('cached verification payload mismatch')",
        "print(json.dumps(local,sort_keys=True,separators=(',',':')))",
    ])
    return [
        sys.executable,
        "-c",
        script,
        str(
            verifier_source
            / "tools"
            / "verify_qwen35_tp4_cached_continuation_correctness_gate.py"
        ),
        str(
            downloaded_authority
            / "cached_continuation_authority"
        ),
        str(
            downloaded_authority
            / "cached_continuation_independent_verification.json"
        ),
    ]


def _paths(output_dir, run_tag):
    remote_run = f"{REMOTE_ROOT}/{run_tag}"
    remote_inputs_root = f"{remote_run}/inputs"
    remote_source = f"{remote_run}/source"
    remote_authority_root = f"{remote_run}/authority"
    return {
        "remote_run": remote_run,
        "remote_inputs_root": remote_inputs_root,
        "remote_source": remote_source,
        "remote_authority_root": remote_authority_root,
        "remote_cached_authority_dir": (
            f"{remote_authority_root}/cached_continuation_authority"
        ),
        "remote_cached_verification_path": (
            f"{remote_authority_root}/"
            "cached_continuation_independent_verification.json"
        ),
        "authority_tar": output_dir / "cached_authority.tar",
        "downloaded": output_dir / DOWNLOADED_AUTHORITY_NAME,
        "verifier_source": output_dir / LOCAL_VERIFIER_SOURCE_NAME,
    }


def _remote_inputs(remote_inputs_root):
    return {
        "configuration": (
            f"{remote_inputs_root}/{REMOTE_CONFIGURATION_NAME}"
        ),
        "source_inventory": (
            f"{remote_inputs_root}/source_inventory.json"
        ),
        "source_tar": f"{remote_inputs_root}/{SOURCE_TAR_NAME}",
        "workload_manifest": (
            f"{remote_inputs_root}/workload_manifest.json"
        ),
    }


def _resource_inputs(
    *,
    resource_policy,
    resource_baseline_path,
    configuration,
):
    if resource_policy == resource_policy_module.STRICT_EXCLUSIVE:
        if resource_baseline_path is not None:
            raise ValueError(
                "strict resource policy does not accept a baseline"
            )
        return None, None
    if resource_policy != resource_policy_module.CONTROLLED_SHARED:
        raise ValueError("resource policy is unsupported")
    if resource_baseline_path is None:
        raise ValueError("controlled resource baseline is required")
    path = Path(resource_baseline_path).resolve()
    resource_policy_module.validate_baseline_manifest(
        path,
        ssh_target=SSH_TARGET,
        gpu_indices=configuration.gpu_indices,
    )
    return path, resource_policy_module.sha256(path)


def _authority_argv(configuration, paths, remote_inputs):
    return [
        "env",
        f"PYTHONPATH={paths['remote_source']}",
        "PYTHONDONTWRITEBYTECODE=1",
        "TORCH_COMPILE_DISABLE=1",
        (
            "CUDA_VISIBLE_DEVICES="
            + ",".join(str(value) for value in configuration.gpu_indices)
        ),
        f"TINYVLLM_DIST_PORT={configuration.dist_port}",
        f"MASTER_PORT={configuration.master_port}",
        REMOTE_PYTHON,
        (
            f"{paths['remote_source']}/tools/"
            "run_qwen35_tp4_cached_continuation_authority.py"
        ),
        "--configuration",
        remote_inputs["configuration"],
        "--source-inventory",
        remote_inputs["source_inventory"],
        "--output-dir",
        paths["remote_cached_authority_dir"],
        "--verification-path",
        paths["remote_cached_verification_path"],
    ]


def _without_control_path_none(value):
    if not isinstance(value, list):
        return value
    result = []
    index = 0
    while index < len(value):
        if value[index:index + 2] == ["-o", "ControlPath=none"]:
            index += 2
            continue
        item = value[index]
        result.append(
            _without_control_path_none(item)
            if isinstance(item, list)
            else item
        )
        index += 1
    return result


def _legacy_commands_without_control_path_none(commands):
    return {
        name: {
            field: _without_control_path_none(value)
            for field, value in command.items()
        }
        for name, command in commands.items()
    }


def _commands_match(recorded, expected):
    if not isinstance(recorded, dict):
        return False
    normalized = copy.deepcopy(recorded)
    local_names = (
        "safe_extract",
        "prepare_local_verifier",
        "local_verify",
    )
    interpreters = []
    for name in local_names:
        command = normalized.get(name)
        argv = command.get("argv") if isinstance(command, dict) else None
        if not isinstance(argv, list) or not argv:
            return False
        interpreter = argv[0]
        if (
            not isinstance(interpreter, str)
            or not Path(interpreter).is_absolute()
            or not Path(interpreter).name.startswith("python")
        ):
            return False
        interpreters.append(interpreter)
    if len(set(interpreters)) != 1:
        return False
    for name in local_names:
        normalized[name]["argv"][0] = expected[name]["argv"][0]
    return normalized in (
        expected,
        _legacy_commands_without_control_path_none(expected),
    )


def _commands(
    *,
    configuration,
    paths,
    remote_inputs,
    identities,
    local_inputs,
    resource_policy,
    resource_baseline_sha256,
):
    authority_argv = _authority_argv(
        configuration,
        paths,
        remote_inputs,
    )
    if resource_baseline_sha256 is None:
        resource_argv = _resource_guard_command(
            configuration.gpu_indices
        )
    else:
        resource_argv = resource_policy_module.guard_command(
            resource_policy,
            configuration.gpu_indices,
            baseline_path=remote_inputs["resource_baseline"],
            baseline_sha256=resource_baseline_sha256,
            ssh_target=SSH_TARGET,
        )
    upload_argv = [
        _scp(
            Path(local_inputs["configuration"]),
            remote_inputs["configuration"],
        ),
        _scp(
            Path(local_inputs["source_inventory"]),
            remote_inputs["source_inventory"],
        ),
        _scp(
            Path(local_inputs["source_tar"]),
            remote_inputs["source_tar"],
        ),
        _scp(
            Path(local_inputs["workload_manifest"]),
            remote_inputs["workload_manifest"],
        ),
    ]
    if resource_baseline_sha256 is not None:
        upload_argv.append(_scp(
            Path(local_inputs["resource_baseline"]),
            remote_inputs["resource_baseline"],
        ))
    resource_guard = {
        "argv": _ssh(resource_argv),
        "gpu_indices": list(configuration.gpu_indices),
        "minimum_free_bytes_per_gpu": MIN_GPU_FREE_BYTES,
        "requires_no_active_compute_processes": (
            resource_policy
            == resource_policy_module.STRICT_EXCLUSIVE
        ),
    }
    if resource_baseline_sha256 is not None:
        resource_guard.update({
            "resource_policy": resource_policy,
            "resource_baseline_sha256": resource_baseline_sha256,
        })
    return {
        "reserve_remote": {
            "argv": _ssh([
                "bash",
                "-lc",
                " && ".join([
                    "set -eu",
                    f"test ! -e {shlex.quote(paths['remote_run'])}",
                    f"mkdir -p {shlex.quote(paths['remote_run'])}",
                    (
                        "mkdir "
                        f"{shlex.quote(paths['remote_inputs_root'])}"
                    ),
                    (
                        "mkdir "
                        f"{shlex.quote(paths['remote_authority_root'])}"
                    ),
                ]),
            ]),
        },
        "upload": {
            "argv": upload_argv,
        },
        "stage": {
            "argv": _ssh([
                "bash",
                "-lc",
                _stage_script(
                    paths["remote_source"],
                    paths["remote_inputs_root"],
                    identities,
                    resource_baseline_name=(
                        RESOURCE_BASELINE_NAME
                        if resource_baseline_sha256 is not None
                        else None
                    ),
                ),
            ]),
        },
        "resource_guard": resource_guard,
        "guarded_authority": {
            "authority_argv": authority_argv,
            "ssh_argv": _ssh(
                _guarded_authority_command(
                    configuration.gpu_indices,
                    authority_argv,
                    resource_policy=resource_policy,
                    resource_baseline_path=remote_inputs.get(
                        "resource_baseline"
                    ),
                    resource_baseline_sha256=(
                        resource_baseline_sha256
                    ),
                )
            ),
            "final_resource_recheck": True,
        },
        "package_download": {
            "remote_argv": _ssh([
                "bash",
                "-lc",
                _package_script(paths["remote_authority_root"]),
            ]),
            "local_output": str(paths["authority_tar"]),
        },
        "safe_extract": {
            "argv": _extract_command(
                paths["authority_tar"],
                paths["downloaded"],
            ),
        },
        "prepare_local_verifier": {
            "argv": _prepare_local_verifier_command(
                Path(local_inputs["source_tar"]),
                Path(local_inputs["source_inventory"]),
                configuration.source_tree_sha256,
                paths["verifier_source"],
            ),
            "source_tar": local_inputs["source_tar"],
            "source_inventory": local_inputs["source_inventory"],
            "source_tree_sha256": configuration.source_tree_sha256,
        },
        "local_verify": {
            "argv": _local_verify_command(
                paths["verifier_source"],
                paths["downloaded"],
            ),
        },
    }


def build_remote_execution_plan(
    *,
    repo_root,
    configuration_path,
    source_inventory_path,
    output_dir,
    run_tag,
    remote_model_dir,
    remote_model_manifest,
    resource_policy=resource_policy_module.STRICT_EXCLUSIVE,
    resource_baseline_path=None,
):
    repo_root = Path(repo_root).resolve()
    configuration_path = Path(configuration_path).resolve()
    source_inventory_path = Path(source_inventory_path).resolve()
    output_dir = Path(output_dir).resolve()
    run_tag = _safe_run_tag(run_tag)
    remote_model_dir = _require_remote_absolute(
        remote_model_dir,
        "remote model directory",
    )
    remote_model_manifest = _require_remote_absolute(
        remote_model_manifest,
        "remote model manifest",
    )
    if output_dir.exists():
        raise ValueError("remote plan output directory already exists")
    if not repo_root.is_dir():
        raise ValueError("repository root is missing")
    configuration = _configuration_from_payload(
        _load_json(configuration_path, "executor configuration")
    )
    (
        resource_baseline_path,
        resource_baseline_sha256,
    ) = _resource_inputs(
        resource_policy=resource_policy,
        resource_baseline_path=resource_baseline_path,
        configuration=configuration,
    )
    inventory = _load_source_inventory(source_inventory_path)
    if not REQUIRED_AUTHORITY_SOURCES.issubset(
        set(inventory["owned_files"])
    ):
        raise ValueError(
            "cached authority source inventory is incomplete"
        )
    if inventory["source_tree_sha256"] != configuration.source_tree_sha256:
        raise ValueError("source inventory and configuration mismatch")
    model_manifest = Path(configuration.model_manifest_path)
    workload_manifest = Path(configuration.workload_manifest_path)
    if _sha256(model_manifest) != configuration.model_manifest_sha256:
        raise ValueError("model manifest SHA mismatch")
    if _sha256(workload_manifest) != configuration.workload_manifest_sha256:
        raise ValueError("workload manifest SHA mismatch")
    files = source_runner._owned_source_files(
        repo_root,
        inventory["owned_files"],
    )
    if (
        [name for name, _ in files] != inventory["owned_files"]
        or source_runner._source_tree_sha256(files)
        != configuration.source_tree_sha256
    ):
        raise ValueError("source tree identity mismatch")

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(
        prefix=f".{output_dir.name}.",
        dir=output_dir.parent,
    ))
    try:
        source_tar = temporary / SOURCE_TAR_NAME
        bundle = source_runner.build_deterministic_source_bundle(
            repo_root=repo_root,
            owned_paths=inventory["owned_files"],
            output_tar=source_tar,
        )
        if (
            bundle["owned_files"] != inventory["owned_files"]
            or bundle["source_tree_sha256"]
            != configuration.source_tree_sha256
        ):
            raise ValueError("source bundle identity mismatch")
        paths = _paths(output_dir, run_tag)
        remote_inputs = _remote_inputs(paths["remote_inputs_root"])
        if resource_baseline_sha256 is not None:
            remote_inputs["resource_baseline"] = (
                f"{paths['remote_inputs_root']}/"
                f"{RESOURCE_BASELINE_NAME}"
            )
        remote_configuration = temporary / REMOTE_CONFIGURATION_NAME
        _write_json(remote_configuration, {
            **configuration.to_payload(),
            "model_dir": remote_model_dir,
            "model_manifest_path": remote_model_manifest,
            "workload_manifest_path": remote_inputs[
                "workload_manifest"
            ],
        })
        final_configuration = output_dir / REMOTE_CONFIGURATION_NAME
        final_source_tar = output_dir / SOURCE_TAR_NAME
        identities = {
            "configuration_sha256": _sha256(remote_configuration),
            "source_inventory_sha256": _sha256(
                source_inventory_path
            ),
            "source_tar_sha256": bundle["tar_sha256"],
            "source_tree_sha256": configuration.source_tree_sha256,
            "model_manifest_sha256": configuration.model_manifest_sha256,
            "workload_manifest_sha256": (
                configuration.workload_manifest_sha256
            ),
        }
        if resource_baseline_sha256 is not None:
            identities["resource_baseline_sha256"] = (
                resource_baseline_sha256
            )
        local_inputs = {
            "configuration": str(final_configuration),
            "configuration_sha256": identities[
                "configuration_sha256"
            ],
            "source_inventory": str(source_inventory_path),
            "source_inventory_sha256": identities[
                "source_inventory_sha256"
            ],
            "source_tar": str(final_source_tar),
            "source_tar_sha256": identities["source_tar_sha256"],
            "workload_manifest": str(workload_manifest),
            "workload_manifest_sha256": identities[
                "workload_manifest_sha256"
            ],
        }
        if resource_baseline_sha256 is not None:
            local_inputs["resource_baseline"] = str(
                output_dir / RESOURCE_BASELINE_NAME
            )
            local_inputs["resource_baseline_sha256"] = (
                resource_baseline_sha256
            )
        payload = {
            "schema_version": SCHEMA_VERSION,
            "run_tag": run_tag,
            "ssh_target": SSH_TARGET,
            "remote_run_root": paths["remote_run"],
            "remote_source_root": paths["remote_source"],
            "remote_authority_root": paths["remote_authority_root"],
            "remote_cached_authority_dir": paths[
                "remote_cached_authority_dir"
            ],
            "remote_cached_verification_path": paths[
                "remote_cached_verification_path"
            ],
            "gpu_indices": list(configuration.gpu_indices),
            "ports": {
                "dist_port": configuration.dist_port,
                "master_port": configuration.master_port,
            },
            "source_tree_sha256": configuration.source_tree_sha256,
            "model_manifest_sha256": configuration.model_manifest_sha256,
            "local_inputs": local_inputs,
            "remote_inputs": remote_inputs,
            "command_order": COMMAND_ORDER,
            "commands": _commands(
                configuration=configuration,
                paths=paths,
                remote_inputs=remote_inputs,
                identities=identities,
                local_inputs=local_inputs,
                resource_policy=resource_policy,
                resource_baseline_sha256=resource_baseline_sha256,
            ),
            "execution_performed": False,
            "claim_boundary": (
                "command authorization only; no SSH, GPU, correctness, "
                "performance, cache, memory, compression, or quality claim"
            ),
        }
        if resource_baseline_sha256 is not None:
            payload["resource_policy"] = resource_policy
            payload["resource_baseline_sha256"] = (
                resource_baseline_sha256
            )
            shutil.copyfile(
                resource_baseline_path,
                temporary / RESOURCE_BASELINE_NAME,
            )
        _write_json(temporary / PLAN_NAME, payload)
        os.replace(temporary, output_dir)
        return verify_remote_execution_plan(output_dir / PLAN_NAME)
    finally:
        if temporary.exists():
            for path in sorted(
                temporary.rglob("*"),
                key=lambda value: len(value.parts),
                reverse=True,
            ):
                if path.is_file():
                    path.unlink()
                elif path.is_dir():
                    path.rmdir()
            temporary.rmdir()


def verify_remote_execution_plan(path):
    path = Path(path)
    payload = _load_json(path, "remote execution plan")
    required = {
        "schema_version",
        "run_tag",
        "ssh_target",
        "remote_run_root",
        "remote_source_root",
        "remote_authority_root",
        "remote_cached_authority_dir",
        "remote_cached_verification_path",
        "gpu_indices",
        "ports",
        "source_tree_sha256",
        "model_manifest_sha256",
        "local_inputs",
        "remote_inputs",
        "command_order",
        "commands",
        "execution_performed",
        "claim_boundary",
    }
    resource_policy = payload.get(
        "resource_policy",
        resource_policy_module.STRICT_EXCLUSIVE,
    ) if isinstance(payload, dict) else None
    if resource_policy == resource_policy_module.CONTROLLED_SHARED:
        required.update({
            "resource_policy",
            "resource_baseline_sha256",
        })
    if (
        not isinstance(payload, dict)
        or set(payload) != required
        or payload["schema_version"] != SCHEMA_VERSION
        or payload["ssh_target"] != SSH_TARGET
        or payload["execution_performed"] is not False
        or payload["command_order"] != COMMAND_ORDER
    ):
        raise ValueError("remote execution plan schema mismatch")
    run_tag = _safe_run_tag(payload["run_tag"])
    local_inputs = payload["local_inputs"]
    if not isinstance(local_inputs, dict):
        raise ValueError("remote execution plan local inputs mismatch")
    configuration = _configuration_from_payload(_load_json(
        local_inputs.get("configuration", ""),
        "remote executor configuration",
    ))
    checks = [
        ("configuration", "configuration_sha256"),
        ("source_inventory", "source_inventory_sha256"),
        ("source_tar", "source_tar_sha256"),
        ("workload_manifest", "workload_manifest_sha256"),
    ]
    resource_baseline_sha256 = None
    if resource_policy == resource_policy_module.CONTROLLED_SHARED:
        checks.append(
            ("resource_baseline", "resource_baseline_sha256")
        )
        resource_baseline_sha256 = payload[
            "resource_baseline_sha256"
        ]
    elif resource_policy != resource_policy_module.STRICT_EXCLUSIVE:
        raise ValueError("resource policy is unsupported")
    for path_name, sha_name in checks:
        local_path = Path(local_inputs.get(path_name, ""))
        if (
            not local_path.is_file()
            or _sha256(local_path) != local_inputs.get(sha_name)
        ):
            raise ValueError(
                "remote execution plan input identity mismatch"
            )
    if resource_baseline_sha256 is not None:
        if (
            local_inputs["resource_baseline_sha256"]
            != resource_baseline_sha256
        ):
            raise ValueError(
                "remote execution plan resource baseline mismatch"
            )
        resource_policy_module.validate_baseline_manifest(
            local_inputs["resource_baseline"],
            ssh_target=SSH_TARGET,
            gpu_indices=configuration.gpu_indices,
        )
    paths = _paths(path.parent.resolve(), run_tag)
    remote_inputs = _remote_inputs(paths["remote_inputs_root"])
    if resource_baseline_sha256 is not None:
        remote_inputs["resource_baseline"] = (
            f"{paths['remote_inputs_root']}/{RESOURCE_BASELINE_NAME}"
        )
    identities = {
        "configuration_sha256": local_inputs[
            "configuration_sha256"
        ],
        "source_inventory_sha256": local_inputs[
            "source_inventory_sha256"
        ],
        "source_tar_sha256": local_inputs["source_tar_sha256"],
        "source_tree_sha256": configuration.source_tree_sha256,
        "model_manifest_sha256": configuration.model_manifest_sha256,
        "workload_manifest_sha256": local_inputs[
            "workload_manifest_sha256"
        ],
    }
    if resource_baseline_sha256 is not None:
        identities["resource_baseline_sha256"] = (
            resource_baseline_sha256
        )
    expected_commands = _commands(
        configuration=configuration,
        paths=paths,
        remote_inputs=remote_inputs,
        identities=identities,
        local_inputs=local_inputs,
        resource_policy=resource_policy,
        resource_baseline_sha256=resource_baseline_sha256,
    )
    if (
        payload["remote_run_root"] != paths["remote_run"]
        or payload["remote_source_root"] != paths["remote_source"]
        or payload["remote_authority_root"]
        != paths["remote_authority_root"]
        or payload["remote_cached_authority_dir"]
        != paths["remote_cached_authority_dir"]
        or payload["remote_cached_verification_path"]
        != paths["remote_cached_verification_path"]
        or payload["remote_inputs"] != remote_inputs
        or payload["gpu_indices"] != list(configuration.gpu_indices)
        or payload["ports"] != {
            "dist_port": configuration.dist_port,
            "master_port": configuration.master_port,
        }
        or payload["source_tree_sha256"]
        != configuration.source_tree_sha256
        or payload["model_manifest_sha256"]
        != configuration.model_manifest_sha256
        or (
            resource_baseline_sha256 is not None
            and payload["resource_baseline_sha256"]
            != resource_baseline_sha256
        )
    ):
        raise ValueError("remote execution plan binding mismatch")
    if not _commands_match(payload["commands"], expected_commands):
        raise ValueError("remote execution plan command mismatch")
    return payload


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--configuration", required=True)
    parser.add_argument("--source-inventory", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--run-tag", required=True)
    parser.add_argument("--remote-model-dir", required=True)
    parser.add_argument("--remote-model-manifest", required=True)
    args = parser.parse_args(argv)
    payload = build_remote_execution_plan(
        repo_root=args.repo_root,
        configuration_path=args.configuration,
        source_inventory_path=args.source_inventory,
        output_dir=args.output_dir,
        run_tag=args.run_tag,
        remote_model_dir=args.remote_model_dir,
        remote_model_manifest=args.remote_model_manifest,
    )
    print(json.dumps(payload, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
