from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path, PurePosixPath
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


source_runner = _load_module(
    "run_qwen35_tp4_hybrid_prefix_benchmark_remote",
    "run_qwen35_tp4_hybrid_prefix_benchmark_remote.py",
)
authority_driver = _load_module(
    "run_qwen35_tp4_engine_correctness_authority",
    "run_qwen35_tp4_engine_correctness_authority.py",
)
executor_module = _load_module(
    "qwen35_tp4_engine_correctness_executor",
    "qwen35_tp4_engine_correctness_executor.py",
)
resource_policy_module = _load_module(
    "qwen35_tp4_correctness_resource_policy",
    "qwen35_tp4_correctness_resource_policy.py",
)


SCHEMA_VERSION = "qwen35.tp4-engine-remote-execution-plan.v1"
SSH_TARGET = "sitian@10.232.195.203"
REMOTE_PYTHON = (
    "/data00/home/sitian/sitian-workspace01/tllm/env/bin/python"
)
REMOTE_ROOT = (
    "/data00/home/sitian/sitian-workspace01/tllm/"
    "qwen35-tp4-engine-authority-runs"
)
MIN_GPU_FREE_BYTES = 24 * 1024**3
PLAN_NAME = "remote_execution_plan.json"
REMOTE_CONFIGURATION_NAME = "remote_executor_configuration.json"
SOURCE_TAR_NAME = "authority_source.tar"
RESOURCE_BASELINE_NAME = "resource_baseline.json"
DOWNLOADED_AUTHORITY_NAME = "downloaded_authority"
LOCAL_VERIFIER_SOURCE_NAME = "local_verifier_source"
EXACT_AUTHORITY_ENTRIES = (
    "reference_authority",
    "reference_independent_verification.json",
    "engine_authority",
    "authority_summary.json",
)
REQUIRED_AUTHORITY_SOURCES = {
    "tools/run_qwen35_tp4_engine_correctness_authority.py",
    "tools/verify_qwen35_tp4_engine_correctness_authority.py",
    "tools/qwen35_tp4_engine_remote_execution_plan.py",
    "tools/qwen35_tp4_engine_remote_execution_receipt.py",
    "tools/qwen35_tp4_engine_remote_execution_executor.py",
    "tools/qwen35_tp4_engine_remote_execution_source_contract.py",
    "tools/qwen35_tp4_engine_remote_execution_authorization.py",
    "tools/qwen35_tp4_engine_remote_subprocess_adapter.py",
}


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path, payload):
    Path(path).write_text(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


def _load_json(path, label):
    path = Path(path)
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"{label} is missing")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is invalid") from error


def _safe_run_tag(value):
    return source_runner.safe_run_tag(value)


def _require_remote_absolute(value, label):
    if (
        not isinstance(value, str)
        or not value
        or not PurePosixPath(value).is_absolute()
        or ".." in PurePosixPath(value).parts
    ):
        raise ValueError(f"{label} must be a safe absolute remote path")
    return value


def _ssh(remote_argv):
    return source_runner._ssh_argv(remote_argv)


def _scp(local_path, remote_path):
    return [
        "scp",
        *source_runner.SSH_OPTIONS,
        os.fspath(local_path),
        f"{SSH_TARGET}:{remote_path}",
    ]


def _configuration_from_payload(payload):
    if (
        not isinstance(payload, dict)
        or set(payload) != set(executor_module.CONFIGURATION_FIELDS)
        or payload.get("world_size") != 4
    ):
        raise ValueError("executor configuration schema mismatch")
    values = dict(payload)
    values.pop("world_size")
    values["gpu_indices"] = tuple(values["gpu_indices"])
    return executor_module.ExecutorConfiguration(**values)


def _load_source_inventory(path):
    payload = _load_json(path, "source inventory")
    if (
        not isinstance(payload, dict)
        or set(payload) != {"owned_files", "source_tree_sha256"}
        or not isinstance(payload["owned_files"], list)
        or not payload["owned_files"]
        or len(payload["owned_files"])
        != len(set(payload["owned_files"]))
        or any(
            not isinstance(name, str)
            or not name
            or PurePosixPath(name).is_absolute()
            or ".." in PurePosixPath(name).parts
            or "\\" in name
            for name in payload["owned_files"]
        )
    ):
        raise ValueError("source inventory schema mismatch")
    source_runner._validate_sha256(
        payload["source_tree_sha256"],
        "source tree",
    )
    return payload


def _resource_guard_command(gpu_indices):
    parse_script = "\n".join([
        "import json,sys",
        "indices=[int(value) for value in sys.argv[1].split(',')]",
        "minimum=int(sys.argv[2])",
        "gpu_text,process_text=sys.stdin.read().split('\\n---PROCESSES---\\n',1)",
        "rows=[]",
        "for line in gpu_text.splitlines():",
        " parts=[value.strip() for value in line.split(',')]",
        " if len(parts)!=3: raise SystemExit('invalid GPU inventory')",
        " rows.append({'gpu_index':int(parts[0]),'gpu_uuid':parts[1],'free_bytes':int(parts[2])*1024*1024,'compute_processes':[]})",
        "by_uuid={row['gpu_uuid']:row for row in rows}",
        "for line in process_text.splitlines():",
        " if not line.strip() or line.strip()=='No running processes found': continue",
        " parts=[value.strip() for value in line.split(',',3)]",
        " if len(parts)!=4 or parts[0] not in by_uuid: raise SystemExit('invalid compute process inventory')",
        " by_uuid[parts[0]]['compute_processes'].append({'pid':int(parts[1]),'process_name':parts[2],'used_memory_mib':int(parts[3])})",
        "selected=[row for row in rows if row['gpu_index'] in indices]",
        "selected.sort(key=lambda row:indices.index(row['gpu_index']))",
        "if len(selected)!=4 or len({row['gpu_uuid'] for row in selected})!=4:",
        " raise SystemExit('four unique configured GPUs are required')",
        "if any(row['free_bytes']<minimum for row in selected):",
        " raise SystemExit('configured GPU free memory is insufficient')",
        "if any(row['compute_processes']!=[] for row in selected):",
        " raise SystemExit('configured GPU has active compute processes')",
        "print(json.dumps({'classification':'READY','selected':selected},sort_keys=True,separators=(',',':')))",
    ])
    gpu_query = (
        "nvidia-smi "
        "--query-gpu=index,uuid,memory.free "
        "--format=csv,noheader,nounits"
    )
    process_query = (
        "nvidia-smi "
        "--query-compute-apps=gpu_uuid,pid,process_name,used_memory "
        "--format=csv,noheader,nounits"
    )
    gpu_csv = ",".join(str(value) for value in gpu_indices)
    shell = " && ".join([
        "set -eu",
        f"gpu_rows=\"$({gpu_query})\"",
        f"process_rows=\"$({process_query})\"",
        (
            "printf '%s\\n---PROCESSES---\\n%s\\n' "
            "\"$gpu_rows\" \"$process_rows\" | "
            f"{shlex.quote(REMOTE_PYTHON)} -c "
            f"{shlex.quote(parse_script)} "
            f"{shlex.quote(gpu_csv)} "
            f"{MIN_GPU_FREE_BYTES}"
        ),
    ])
    return ["bash", "-lc", shell]


def _shared_low_utilization_resource_guard_command(
    gpu_indices,
    maximum_gpu_utilization_percent,
):
    if (
        isinstance(maximum_gpu_utilization_percent, bool)
        or not isinstance(maximum_gpu_utilization_percent, int)
        or not 0 <= maximum_gpu_utilization_percent <= 100
    ):
        raise ValueError("maximum GPU utilization is invalid")
    parse_script = "\n".join([
        "import json,sys",
        "indices=[int(value) for value in sys.argv[1].split(',')]",
        "minimum=int(sys.argv[2])",
        "maximum_utilization=int(sys.argv[3])",
        (
            "gpu_text,process_text=sys.stdin.read().split("
            "'\\n---PROCESSES---\\n',1)"
        ),
        "rows=[]",
        "for line in gpu_text.splitlines():",
        " parts=[value.strip() for value in line.split(',')]",
        " if len(parts)!=4: raise SystemExit('invalid GPU inventory')",
        (
            " rows.append({'gpu_index':int(parts[0]),"
            "'gpu_uuid':parts[1],"
            "'free_bytes':int(parts[2])*1024*1024,"
            "'utilization_percent':int(parts[3]),"
            "'compute_processes':[]})"
        ),
        "by_uuid={row['gpu_uuid']:row for row in rows}",
        "for line in process_text.splitlines():",
        (
            " if not line.strip() or "
            "line.strip()=='No running processes found': continue"
        ),
        " parts=[value.strip() for value in line.split(',',3)]",
        (
            " if len(parts)!=4 or parts[0] not in by_uuid: "
            "raise SystemExit('invalid compute process inventory')"
        ),
        (
            " by_uuid[parts[0]]['compute_processes'].append({"
            "'pid':int(parts[1]),'process_name':parts[2],"
            "'used_memory_mib':int(parts[3])})"
        ),
        "selected=[row for row in rows if row['gpu_index'] in indices]",
        (
            "selected.sort("
            "key=lambda row:indices.index(row['gpu_index']))"
        ),
        (
            "if len(selected)!=4 or "
            "len({row['gpu_uuid'] for row in selected})!=4:"
        ),
        " raise SystemExit('four unique configured GPUs are required')",
        "if any(row['free_bytes']<minimum for row in selected):",
        " raise SystemExit('configured GPU free memory is insufficient')",
        (
            "if any(row['utilization_percent']>maximum_utilization "
            "for row in selected):"
        ),
        " raise SystemExit('configured GPU utilization is too high')",
        (
            "print(json.dumps({'classification':'READY',"
            "'resource_policy':'shared-low-utilization',"
            "'maximum_gpu_utilization_percent':maximum_utilization,"
            "'selected':selected},"
            "sort_keys=True,separators=(',',':')))"
        ),
    ])
    gpu_query = (
        "nvidia-smi "
        "--query-gpu=index,uuid,memory.free,utilization.gpu "
        "--format=csv,noheader,nounits"
    )
    process_query = (
        "nvidia-smi "
        "--query-compute-apps=gpu_uuid,pid,process_name,used_memory "
        "--format=csv,noheader,nounits"
    )
    gpu_csv = ",".join(str(value) for value in gpu_indices)
    shell = " && ".join([
        "set -eu",
        f"gpu_rows=\"$({gpu_query})\"",
        f"process_rows=\"$({process_query})\"",
        (
            "printf '%s\\n---PROCESSES---\\n%s\\n' "
            "\"$gpu_rows\" \"$process_rows\" | "
            f"{shlex.quote(REMOTE_PYTHON)} -c "
            f"{shlex.quote(parse_script)} "
            f"{shlex.quote(gpu_csv)} "
            f"{MIN_GPU_FREE_BYTES} "
            f"{maximum_gpu_utilization_percent}"
        ),
    ])
    return ["bash", "-lc", shell]


def _guarded_authority_command(
    gpu_indices,
    authority_argv,
    *,
    resource_policy=resource_policy_module.STRICT_EXCLUSIVE,
    resource_baseline_path=None,
    resource_baseline_sha256=None,
):
    if resource_policy == resource_policy_module.STRICT_EXCLUSIVE:
        guard = _resource_guard_command(gpu_indices)
    else:
        guard = resource_policy_module.guard_command(
            resource_policy,
            gpu_indices,
            baseline_path=resource_baseline_path,
            baseline_sha256=resource_baseline_sha256,
            ssh_target=SSH_TARGET,
        )
    if guard[:2] != ["bash", "-lc"] or len(guard) != 3:
        raise ValueError("resource guard command is invalid")
    shell = (
        f"final_resource=\"$({guard[2]})\" && "
        "printf 'QWEN35_FINAL_RESOURCE_JSON=%s\\n' "
        "\"$final_resource\" && exec "
        f"{shlex.join([str(value) for value in authority_argv])}"
    )
    return ["bash", "-lc", shell]


def _stage_script(
    remote_source,
    remote_inputs,
    identities,
    *,
    resource_baseline_name=None,
):
    verify_script = "\n".join([
        "import hashlib,json,sys,tarfile",
        "from pathlib import Path,PurePosixPath",
        "archive=Path(sys.argv[1])",
        "destination=Path(sys.argv[2])",
        "inventory_path=Path(sys.argv[3])",
        "expected_tree=sys.argv[4]",
        "expected_tar=sys.argv[5]",
        "if hashlib.sha256(archive.read_bytes()).hexdigest()!=expected_tar:",
        " raise SystemExit('source tar SHA mismatch')",
        "inventory=json.loads(inventory_path.read_text())",
        "with tarfile.open(archive,'r:') as handle:",
        " members=handle.getmembers()",
        " names=[member.name for member in members]",
        " if names!=inventory['owned_files']:",
        "  raise SystemExit('source tar inventory mismatch')",
        " for member in members:",
        "  path=PurePosixPath(member.name)",
        "  if path.is_absolute() or '..' in path.parts or not member.isfile() or member.issym() or member.islnk():",
        "   raise SystemExit('unsafe source tar member')",
        " handle.extractall(destination,members=members)",
        "digest=hashlib.sha256()",
        "for name in inventory['owned_files']:",
        " path=destination.joinpath(*PurePosixPath(name).parts)",
        " encoded=name.encode('utf-8')",
        " digest.update(len(encoded).to_bytes(8,'big'))",
        " digest.update(encoded)",
        " with path.open('rb') as source:",
        "  for chunk in iter(lambda:source.read(1024*1024),b''):",
        "   digest.update(chunk)",
        "if digest.hexdigest()!=expected_tree:",
        " raise SystemExit('source tree SHA mismatch')",
    ])
    commands = [
        "set -eu",
        *[
            (
                f"test \"$(sha256sum {shlex.quote(remote_inputs + '/' + name)} "
                "| awk '{print $1}')\" = "
                f"{shlex.quote(identities[key])}"
            )
            for name, key in (
                (
                    REMOTE_CONFIGURATION_NAME,
                    "configuration_sha256",
                ),
                (
                    "source_inventory.json",
                    "source_inventory_sha256",
                ),
                (SOURCE_TAR_NAME, "source_tar_sha256"),
                (
                    "workload_manifest.json",
                    "workload_manifest_sha256",
                ),
                *(
                    ((
                        resource_baseline_name,
                        "resource_baseline_sha256",
                    ),)
                    if resource_baseline_name is not None
                    else ()
                ),
            )
        ],
        f"mkdir {shlex.quote(remote_source)}",
        (
            f"{shlex.quote(REMOTE_PYTHON)} -c "
            f"{shlex.quote(verify_script)} "
            f"{shlex.quote(remote_inputs + '/' + SOURCE_TAR_NAME)} "
            f"{shlex.quote(remote_source)} "
            f"{shlex.quote(remote_inputs + '/source_inventory.json')} "
            f"{shlex.quote(identities['source_tree_sha256'])} "
            f"{shlex.quote(identities['source_tar_sha256'])}"
        ),
    ]
    return " && ".join(commands)


def _package_script(remote_authority_root):
    expected = " ".join(EXACT_AUTHORITY_ENTRIES)
    return " && ".join([
        "set -eu",
        f"cd {shlex.quote(remote_authority_root)}",
        "test \"$(find . -mindepth 1 -maxdepth 1 | wc -l)\" -eq 4",
        *[
            f"test {'-d' if name.endswith('/') else '-e'} "
            f"{shlex.quote(name.rstrip('/'))}"
            for name in EXACT_AUTHORITY_ENTRIES
        ],
        f"tar -cf - {expected}",
    ])


def _extract_command(local_tar, destination):
    script = "\n".join([
        "import sys,tarfile",
        "from pathlib import Path,PurePosixPath",
        "archive=Path(sys.argv[1])",
        "destination=Path(sys.argv[2])",
        f"expected={list(EXACT_AUTHORITY_ENTRIES)!r}",
        "if destination.exists(): raise SystemExit('download destination exists')",
        "with tarfile.open(archive,'r:') as handle:",
        " members=handle.getmembers()",
        " roots=sorted({PurePosixPath(member.name).parts[0] for member in members})",
        " if roots!=sorted(expected): raise SystemExit('authority inventory mismatch')",
        " for member in members:",
        "  path=PurePosixPath(member.name)",
        "  if path.is_absolute() or '..' in path.parts or member.issym() or member.islnk():",
        "   raise SystemExit('unsafe authority tar member')",
        " destination.mkdir()",
        " handle.extractall(destination,members=members)",
    ])
    return [sys.executable, "-c", script, str(local_tar), str(destination)]


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


def _prepare_local_verifier_command(
    source_tar,
    source_inventory,
    source_tree_sha256,
    destination,
):
    script = "\n".join([
        "import hashlib,json,sys,tarfile",
        "from pathlib import Path,PurePosixPath",
        "archive=Path(sys.argv[1])",
        "inventory=json.loads(Path(sys.argv[2]).read_text())",
        "expected=sys.argv[3]",
        "destination=Path(sys.argv[4])",
        "if destination.exists(): raise SystemExit('local verifier source exists')",
        "with tarfile.open(archive,'r:') as handle:",
        " members=handle.getmembers()",
        " if [member.name for member in members]!=inventory['owned_files']:",
        "  raise SystemExit('local verifier source inventory mismatch')",
        " for member in members:",
        "  path=PurePosixPath(member.name)",
        "  if path.is_absolute() or '..' in path.parts or not member.isfile() or member.issym() or member.islnk():",
        "   raise SystemExit('unsafe local verifier source member')",
        " destination.mkdir()",
        " handle.extractall(destination,members=members)",
        "digest=hashlib.sha256()",
        "for name in inventory['owned_files']:",
        " path=destination.joinpath(*PurePosixPath(name).parts)",
        " encoded=name.encode('utf-8')",
        " digest.update(len(encoded).to_bytes(8,'big'))",
        " digest.update(encoded)",
        " with path.open('rb') as source:",
        "  for chunk in iter(lambda:source.read(1024*1024),b''):",
        "   digest.update(chunk)",
        "if digest.hexdigest()!=expected:",
        " raise SystemExit('local verifier source tree mismatch')",
    ])
    return [
        sys.executable,
        "-c",
        script,
        str(source_tar),
        str(source_inventory),
        source_tree_sha256,
        str(destination),
    ]


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
    configuration_payload = _load_json(
        configuration_path,
        "executor configuration",
    )
    configuration = _configuration_from_payload(configuration_payload)
    resource_baseline = None
    resource_baseline_sha256 = None
    if resource_policy == resource_policy_module.STRICT_EXCLUSIVE:
        if resource_baseline_path is not None:
            raise ValueError(
                "strict resource policy does not accept a baseline"
            )
    elif resource_policy == resource_policy_module.CONTROLLED_SHARED:
        if resource_baseline_path is None:
            raise ValueError("controlled resource baseline is required")
        resource_baseline_path = Path(resource_baseline_path).resolve()
        resource_baseline = (
            resource_policy_module.validate_baseline_manifest(
                resource_baseline_path,
                ssh_target=SSH_TARGET,
                gpu_indices=configuration.gpu_indices,
            )
        )
        resource_baseline_sha256 = resource_policy_module.sha256(
            resource_baseline_path
        )
    else:
        raise ValueError("resource policy is unsupported")
    inventory = _load_source_inventory(source_inventory_path)
    if not REQUIRED_AUTHORITY_SOURCES.issubset(
        set(inventory["owned_files"])
    ):
        raise ValueError("authority source inventory is incomplete")
    if inventory["source_tree_sha256"] != configuration.source_tree_sha256:
        raise ValueError("source inventory and configuration mismatch")
    model_manifest_path = Path(configuration.model_manifest_path)
    workload_manifest_path = Path(configuration.workload_manifest_path)
    if _sha256(model_manifest_path) != configuration.model_manifest_sha256:
        raise ValueError("model manifest SHA mismatch")
    if (
        _sha256(workload_manifest_path)
        != configuration.workload_manifest_sha256
    ):
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
        remote_run = f"{REMOTE_ROOT}/{run_tag}"
        remote_inputs = f"{remote_run}/inputs"
        remote_source = f"{remote_run}/source"
        remote_authority = f"{remote_run}/authority"
        remote_workload = f"{remote_inputs}/workload_manifest.json"
        remote_configuration_payload = {
            **configuration.to_payload(),
            "model_dir": remote_model_dir,
            "model_manifest_path": remote_model_manifest,
            "workload_manifest_path": remote_workload,
        }
        remote_configuration = (
            temporary / REMOTE_CONFIGURATION_NAME
        )
        _write_json(remote_configuration, remote_configuration_payload)
        final_configuration = (
            output_dir / REMOTE_CONFIGURATION_NAME
        )
        final_source_tar = output_dir / SOURCE_TAR_NAME
        local_authority_tar = output_dir / "authority.tar"
        downloaded_authority = (
            output_dir / DOWNLOADED_AUTHORITY_NAME
        )
        local_verifier_source = (
            output_dir / LOCAL_VERIFIER_SOURCE_NAME
        )
        remote_paths = {
            "configuration": (
                f"{remote_inputs}/{REMOTE_CONFIGURATION_NAME}"
            ),
            "source_inventory": (
                f"{remote_inputs}/source_inventory.json"
            ),
            "source_tar": f"{remote_inputs}/{SOURCE_TAR_NAME}",
            "workload_manifest": remote_workload,
        }
        final_resource_baseline = None
        if resource_baseline is not None:
            final_resource_baseline = (
                output_dir / RESOURCE_BASELINE_NAME
            )
            remote_paths["resource_baseline"] = (
                f"{remote_inputs}/{RESOURCE_BASELINE_NAME}"
            )
        identities = {
            "configuration_sha256": _sha256(remote_configuration),
            "source_inventory_sha256": _sha256(
                source_inventory_path
            ),
            "source_tar_sha256": bundle["tar_sha256"],
            "source_tree_sha256": configuration.source_tree_sha256,
            "model_manifest_sha256": (
                configuration.model_manifest_sha256
            ),
            "model_manifest_sha256": (
                configuration.model_manifest_sha256
            ),
            "workload_manifest_sha256": (
                configuration.workload_manifest_sha256
            ),
        }
        if resource_baseline_sha256 is not None:
            identities["resource_baseline_sha256"] = (
                resource_baseline_sha256
            )
            resource_argv = resource_policy_module.guard_command(
                resource_policy,
                configuration.gpu_indices,
                baseline_path=remote_paths["resource_baseline"],
                baseline_sha256=resource_baseline_sha256,
                ssh_target=SSH_TARGET,
            )
        else:
            resource_argv = _resource_guard_command(
                configuration.gpu_indices
            )
        authority_argv = [
            "env",
            f"PYTHONPATH={remote_source}",
            "PYTHONDONTWRITEBYTECODE=1",
            "TORCH_COMPILE_DISABLE=1",
            f"CUDA_VISIBLE_DEVICES={','.join(str(value) for value in configuration.gpu_indices)}",
            f"TINYVLLM_DIST_PORT={configuration.dist_port}",
            f"MASTER_PORT={configuration.master_port}",
            REMOTE_PYTHON,
            f"{remote_source}/tools/run_qwen35_tp4_engine_correctness_authority.py",
            "--configuration",
            remote_paths["configuration"],
            "--source-inventory",
            remote_paths["source_inventory"],
            "--output-root",
            remote_authority,
        ]
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
            "workload_manifest": str(workload_manifest_path),
            "workload_manifest_sha256": identities[
                "workload_manifest_sha256"
            ],
        }
        if final_resource_baseline is not None:
            local_inputs["resource_baseline"] = str(
                final_resource_baseline
            )
            local_inputs["resource_baseline_sha256"] = (
                resource_baseline_sha256
            )
        upload_argv = [
            _scp(
                final_configuration,
                remote_paths["configuration"],
            ),
            _scp(
                source_inventory_path,
                remote_paths["source_inventory"],
            ),
            _scp(final_source_tar, remote_paths["source_tar"]),
            _scp(
                workload_manifest_path,
                remote_paths["workload_manifest"],
            ),
        ]
        if final_resource_baseline is not None:
            upload_argv.append(_scp(
                final_resource_baseline,
                remote_paths["resource_baseline"],
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
                "resource_baseline_sha256": (
                    resource_baseline_sha256
                ),
            })
        payload = {
            "schema_version": SCHEMA_VERSION,
            "run_tag": run_tag,
            "ssh_target": SSH_TARGET,
            "remote_run_root": remote_run,
            "remote_source_root": remote_source,
            "remote_authority_root": remote_authority,
            "gpu_indices": list(configuration.gpu_indices),
            "ports": {
                "dist_port": configuration.dist_port,
                "master_port": configuration.master_port,
            },
            "source_tree_sha256": configuration.source_tree_sha256,
            "model_manifest_sha256": (
                configuration.model_manifest_sha256
            ),
            "local_inputs": local_inputs,
            "remote_inputs": remote_paths,
            "command_order": [
                "reserve_remote",
                "upload",
                "stage",
                "resource_guard",
                "guarded_authority",
                "package_download",
                "safe_extract",
                "prepare_local_verifier",
                "local_verify",
            ],
            "commands": {
                "reserve_remote": {
                    "argv": _ssh([
                        "bash",
                        "-lc",
                        " && ".join([
                            "set -eu",
                            f"test ! -e {shlex.quote(remote_run)}",
                            f"mkdir -p {shlex.quote(remote_run)}",
                            f"mkdir {shlex.quote(remote_inputs)}",
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
                            remote_source,
                            remote_inputs,
                            identities,
                            resource_baseline_name=(
                                RESOURCE_BASELINE_NAME
                                if resource_baseline is not None
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
                            resource_baseline_path=remote_paths.get(
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
                        _package_script(remote_authority),
                    ]),
                    "local_output": str(local_authority_tar),
                },
                "safe_extract": {
                    "argv": _extract_command(
                        local_authority_tar,
                        downloaded_authority,
                    ),
                },
                "prepare_local_verifier": {
                    "argv": _prepare_local_verifier_command(
                        final_source_tar,
                        source_inventory_path,
                        configuration.source_tree_sha256,
                        local_verifier_source,
                    ),
                    "source_tar": str(final_source_tar),
                    "source_inventory": str(source_inventory_path),
                    "source_tree_sha256": (
                        configuration.source_tree_sha256
                    ),
                },
                "local_verify": {
                    "argv": [
                        sys.executable,
                        str(
                            local_verifier_source
                            / "tools"
                            / "verify_qwen35_tp4_engine_correctness_authority.py"
                        ),
                        str(downloaded_authority),
                    ],
                },
            },
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
        or payload["command_order"] != [
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
    ):
        raise ValueError("remote execution plan schema mismatch")
    _safe_run_tag(payload["run_tag"])
    local_inputs = payload["local_inputs"]
    if not isinstance(local_inputs, dict):
        raise ValueError("remote execution plan local inputs mismatch")
    configuration_path = Path(local_inputs.get("configuration", ""))
    configuration_payload = _load_json(
        configuration_path,
        "remote executor configuration",
    )
    configuration = _configuration_from_payload(configuration_payload)
    checks = [
        ("configuration", "configuration_sha256"),
        ("source_inventory", "source_inventory_sha256"),
        ("source_tar", "source_tar_sha256"),
        ("workload_manifest", "workload_manifest_sha256"),
    ]
    resource_baseline = None
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
        expected = local_inputs.get(sha_name)
        if not local_path.is_file() or _sha256(local_path) != expected:
            raise ValueError("remote execution plan input identity mismatch")
    if resource_policy == resource_policy_module.CONTROLLED_SHARED:
        if (
            local_inputs["resource_baseline_sha256"]
            != resource_baseline_sha256
        ):
            raise ValueError(
                "remote execution plan resource baseline mismatch"
            )
        resource_baseline = (
            resource_policy_module.validate_baseline_manifest(
                local_inputs["resource_baseline"],
                ssh_target=SSH_TARGET,
                gpu_indices=configuration.gpu_indices,
            )
        )
    remote_run = f"{REMOTE_ROOT}/{payload['run_tag']}"
    remote_inputs_root = f"{remote_run}/inputs"
    remote_source = f"{remote_run}/source"
    remote_authority = f"{remote_run}/authority"
    remote_inputs = {
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
    if resource_baseline is not None:
        remote_inputs["resource_baseline"] = (
            f"{remote_inputs_root}/{RESOURCE_BASELINE_NAME}"
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
    authority_argv = [
        "env",
        f"PYTHONPATH={remote_source}",
        "PYTHONDONTWRITEBYTECODE=1",
        "TORCH_COMPILE_DISABLE=1",
        f"CUDA_VISIBLE_DEVICES={','.join(str(value) for value in configuration.gpu_indices)}",
        f"TINYVLLM_DIST_PORT={configuration.dist_port}",
        f"MASTER_PORT={configuration.master_port}",
        REMOTE_PYTHON,
        f"{remote_source}/tools/run_qwen35_tp4_engine_correctness_authority.py",
        "--configuration",
        remote_inputs["configuration"],
        "--source-inventory",
        remote_inputs["source_inventory"],
        "--output-root",
        remote_authority,
    ]
    output_dir = path.parent.resolve()
    authority_tar = output_dir / "authority.tar"
    downloaded_authority = output_dir / DOWNLOADED_AUTHORITY_NAME
    local_verifier_source = (
        output_dir / LOCAL_VERIFIER_SOURCE_NAME
    )
    local_verifier = (
        local_verifier_source
        / "tools"
        / "verify_qwen35_tp4_engine_correctness_authority.py"
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
    if resource_baseline is not None:
        upload_argv.append(_scp(
            Path(local_inputs["resource_baseline"]),
            remote_inputs["resource_baseline"],
        ))
        resource_argv = resource_policy_module.guard_command(
            resource_policy,
            configuration.gpu_indices,
            baseline_path=remote_inputs["resource_baseline"],
            baseline_sha256=resource_baseline_sha256,
            ssh_target=SSH_TARGET,
        )
    else:
        resource_argv = _resource_guard_command(
            configuration.gpu_indices
        )
    resource_guard = {
        "argv": _ssh(resource_argv),
        "gpu_indices": list(configuration.gpu_indices),
        "minimum_free_bytes_per_gpu": MIN_GPU_FREE_BYTES,
        "requires_no_active_compute_processes": (
            resource_policy
            == resource_policy_module.STRICT_EXCLUSIVE
        ),
    }
    if resource_baseline is not None:
        resource_guard.update({
            "resource_policy": resource_policy,
            "resource_baseline_sha256": resource_baseline_sha256,
        })
    expected_commands = {
        "reserve_remote": {
            "argv": _ssh([
                "bash",
                "-lc",
                " && ".join([
                    "set -eu",
                    f"test ! -e {shlex.quote(remote_run)}",
                    f"mkdir -p {shlex.quote(remote_run)}",
                    f"mkdir {shlex.quote(remote_inputs_root)}",
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
                    remote_source,
                    remote_inputs_root,
                    identities,
                    resource_baseline_name=(
                        RESOURCE_BASELINE_NAME
                        if resource_baseline is not None
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
                _package_script(remote_authority),
            ]),
            "local_output": str(authority_tar),
        },
        "safe_extract": {
            "argv": _extract_command(
                authority_tar,
                downloaded_authority,
            ),
        },
        "prepare_local_verifier": {
            "argv": _prepare_local_verifier_command(
                Path(local_inputs["source_tar"]),
                Path(local_inputs["source_inventory"]),
                configuration.source_tree_sha256,
                local_verifier_source,
            ),
            "source_tar": str(Path(local_inputs["source_tar"])),
            "source_inventory": str(
                Path(local_inputs["source_inventory"])
            ),
            "source_tree_sha256": configuration.source_tree_sha256,
        },
        "local_verify": {
            "argv": [
                sys.executable,
                str(local_verifier),
                str(downloaded_authority),
            ],
        },
    }
    if (
        payload["gpu_indices"] != list(configuration.gpu_indices)
        or payload["ports"] != {
            "dist_port": configuration.dist_port,
            "master_port": configuration.master_port,
        }
        or payload["source_tree_sha256"]
        != configuration.source_tree_sha256
        or payload["model_manifest_sha256"]
        != configuration.model_manifest_sha256
        or payload["remote_inputs"]["configuration"]
        not in payload["commands"]["guarded_authority"][
            "authority_argv"
        ]
        or payload["remote_inputs"]["source_inventory"]
        not in payload["commands"]["guarded_authority"][
            "authority_argv"
        ]
        or payload["remote_authority_root"]
        not in payload["commands"]["guarded_authority"][
            "authority_argv"
        ]
        or payload["commands"]["resource_guard"].get(
            "requires_no_active_compute_processes"
        )
        is not (
            resource_policy
            == resource_policy_module.STRICT_EXCLUSIVE
        )
        or payload["commands"]["resource_guard"].get(
            "minimum_free_bytes_per_gpu"
        )
        != MIN_GPU_FREE_BYTES
        or payload["commands"]["resource_guard"].get("gpu_indices")
        != payload["gpu_indices"]
        or (
            resource_baseline is not None
            and (
                payload["resource_baseline_sha256"]
                != resource_baseline_sha256
                or payload["commands"]["resource_guard"].get(
                    "resource_policy"
                )
                != resource_policy
            )
        )
    ):
        raise ValueError("remote execution plan binding mismatch")
    if (
        payload["remote_run_root"] != remote_run
        or payload["remote_source_root"] != remote_source
        or payload["remote_authority_root"] != remote_authority
        or payload["remote_inputs"] != remote_inputs
        or not _commands_match(payload["commands"], expected_commands)
    ):
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
