from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shlex


BASELINE_SCHEMA_VERSION = (
    "qwen35.tp4-controlled-shared-resource-baseline.v1"
)
STRICT_EXCLUSIVE = "strict_exclusive"
CONTROLLED_SHARED = "controlled_shared"
MIN_GPU_FREE_BYTES = 24 * 1024**3


def _valid_sha256(value):
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _load_regular_json(path, label):
    path = Path(path)
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"{label} must be a regular file")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is invalid") from error
    if not isinstance(payload, dict):
        raise ValueError(f"{label} is invalid")
    return payload


def _validate_gpu_indices(gpu_indices):
    if (
        not isinstance(gpu_indices, (list, tuple))
        or len(gpu_indices) != 4
        or len(set(gpu_indices)) != 4
        or any(
            isinstance(value, bool)
            or not isinstance(value, int)
            or value < 0
            for value in gpu_indices
        )
    ):
        raise ValueError("four unique GPU indices are required")
    return list(gpu_indices)


def _validate_process(process):
    required = {
        "pid",
        "process_name",
        "used_memory_mib",
        "start_time_ticks",
    }
    if (
        not isinstance(process, dict)
        or set(process) != required
        or isinstance(process.get("pid"), bool)
        or not isinstance(process.get("pid"), int)
        or process["pid"] <= 0
        or not isinstance(process.get("process_name"), str)
        or not process["process_name"]
        or isinstance(process.get("used_memory_mib"), bool)
        or not isinstance(process.get("used_memory_mib"), int)
        or process["used_memory_mib"] < 0
        or isinstance(process.get("start_time_ticks"), bool)
        or not isinstance(process.get("start_time_ticks"), int)
        or process["start_time_ticks"] <= 0
    ):
        raise ValueError("resource process schema mismatch")
    return process


def _validate_selected(
    selected,
    gpu_indices,
    *,
    require_start_time,
    require_no_processes=False,
):
    if (
        not isinstance(selected, list)
        or len(selected) != 4
        or [row.get("gpu_index") for row in selected] != gpu_indices
        or len({row.get("gpu_uuid") for row in selected}) != 4
    ):
        raise ValueError("resource GPU drift")
    validated = []
    for row in selected:
        required = {
            "gpu_index",
            "gpu_uuid",
            "free_bytes",
            "compute_processes",
        }
        if (
            not isinstance(row, dict)
            or set(row) != required
            or not isinstance(row.get("gpu_uuid"), str)
            or not row["gpu_uuid"]
            or isinstance(row.get("free_bytes"), bool)
            or not isinstance(row.get("free_bytes"), int)
            or row["free_bytes"] < MIN_GPU_FREE_BYTES
            or not isinstance(row.get("compute_processes"), list)
        ):
            if (
                isinstance(row, dict)
                and isinstance(row.get("free_bytes"), int)
                and row["free_bytes"] < MIN_GPU_FREE_BYTES
            ):
                raise ValueError("configured GPU free memory is insufficient")
            raise ValueError("resource GPU schema mismatch")
        processes = row["compute_processes"]
        if require_no_processes and processes:
            raise ValueError(
                "strict resource guard has active compute processes"
            )
        if require_start_time:
            for process in processes:
                _validate_process(process)
            identities = [
                (
                    process["pid"],
                    process["process_name"],
                    process["start_time_ticks"],
                )
                for process in processes
            ]
            if len(identities) != len(set(identities)):
                raise ValueError("resource process inventory is duplicated")
        validated.append(row)
    return validated


def validate_baseline_manifest(path, *, ssh_target, gpu_indices):
    gpu_indices = _validate_gpu_indices(gpu_indices)
    payload = _load_regular_json(path, "resource baseline")
    required = {
        "schema_version",
        "classification",
        "ssh_target",
        "captured_at",
        "gpu_indices",
        "selected",
        "minimum_free_bytes_per_gpu",
        "benchmark_execution_authorized",
    }
    if (
        set(payload) != required
        or payload.get("schema_version") != BASELINE_SCHEMA_VERSION
        or payload.get("classification") != "READY"
        or payload.get("ssh_target") != ssh_target
        or not isinstance(payload.get("captured_at"), str)
        or not payload["captured_at"]
        or payload.get("gpu_indices") != gpu_indices
        or payload.get("minimum_free_bytes_per_gpu")
        != MIN_GPU_FREE_BYTES
        or payload.get("benchmark_execution_authorized") is not False
    ):
        raise ValueError("resource baseline schema mismatch")
    _validate_selected(
        payload["selected"],
        gpu_indices,
        require_start_time=True,
    )
    return payload


def validate_guard_payload(
    resource_policy,
    payload,
    *,
    gpu_indices,
    baseline=None,
    baseline_sha256=None,
):
    gpu_indices = _validate_gpu_indices(gpu_indices)
    if not isinstance(payload, dict):
        raise ValueError("resource guard payload is invalid")
    if resource_policy == STRICT_EXCLUSIVE:
        if set(payload) != {"classification", "selected"}:
            raise ValueError("strict resource guard schema mismatch")
        if payload.get("classification") != "READY":
            raise ValueError("strict resource guard did not pass")
        _validate_selected(
            payload.get("selected"),
            gpu_indices,
            require_start_time=False,
            require_no_processes=True,
        )
        return gpu_indices
    if resource_policy != CONTROLLED_SHARED:
        raise ValueError("resource policy is unsupported")
    required = {
        "classification",
        "resource_policy",
        "baseline_sha256",
        "selected",
        "benchmark_execution_authorized",
    }
    if (
        set(payload) != required
        or payload.get("classification") != "READY"
        or payload.get("resource_policy") != CONTROLLED_SHARED
        or payload.get("benchmark_execution_authorized") is not False
        or not _valid_sha256(baseline_sha256)
        or payload.get("baseline_sha256") != baseline_sha256
        or not isinstance(baseline, dict)
    ):
        raise ValueError("controlled resource baseline mismatch")
    observed = _validate_selected(
        payload["selected"],
        gpu_indices,
        require_start_time=True,
    )
    frozen = _validate_selected(
        baseline.get("selected"),
        gpu_indices,
        require_start_time=True,
    )
    for observed_row, frozen_row in zip(observed, frozen):
        if observed_row["gpu_uuid"] != frozen_row["gpu_uuid"]:
            raise ValueError("controlled resource GPU drift")
        frozen_identities = {
            (
                process["pid"],
                process["process_name"],
                process["start_time_ticks"],
            )
            for process in frozen_row["compute_processes"]
        }
        observed_identities = {
            (
                process["pid"],
                process["process_name"],
                process["start_time_ticks"],
            )
            for process in observed_row["compute_processes"]
        }
        if not observed_identities.issubset(frozen_identities):
            raise ValueError("controlled resource process drift")
    return gpu_indices


def _inventory_script(mode):
    lines = [
        "import hashlib,json,sys",
        "from datetime import datetime,timezone",
        "from pathlib import Path",
        "indices=[int(value) for value in sys.argv[1].split(',')]",
        "minimum=int(sys.argv[2])",
        "target=sys.argv[3]",
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
        " pid=int(parts[1])",
        " stat=Path(f'/proc/{pid}/stat').read_text(encoding='utf-8')",
        " close=stat.rfind(')')",
        " if close<0: raise SystemExit('invalid process stat')",
        " fields=stat[close+2:].split()",
        " if len(fields)<=19: raise SystemExit('invalid process stat')",
        " by_uuid[parts[0]]['compute_processes'].append({'pid':pid,'process_name':parts[2],'used_memory_mib':int(parts[3]),'start_time_ticks':int(fields[19])})",
        "selected=[row for row in rows if row['gpu_index'] in indices]",
        "selected.sort(key=lambda row:indices.index(row['gpu_index']))",
        "if len(selected)!=4 or len({row['gpu_uuid'] for row in selected})!=4: raise SystemExit('four unique configured GPUs are required')",
        "if any(row['free_bytes']<minimum for row in selected): raise SystemExit('configured GPU free memory is insufficient')",
        "for row in selected: row['compute_processes'].sort(key=lambda process:(process['pid'],process['process_name'],process['start_time_ticks']))",
    ]
    if mode == "capture":
        lines.extend([
            "payload={'schema_version':'"
            + BASELINE_SCHEMA_VERSION
            + "','classification':'READY','ssh_target':target,'captured_at':datetime.now(timezone.utc).astimezone().isoformat(timespec='seconds'),'gpu_indices':indices,'selected':selected,'minimum_free_bytes_per_gpu':minimum,'benchmark_execution_authorized':False}",
            "print(json.dumps(payload,sort_keys=True,separators=(',',':')))",
        ])
    else:
        lines.extend([
            "baseline_path=Path(sys.argv[4])",
            "expected_sha=sys.argv[5]",
            "raw=baseline_path.read_bytes()",
            "if hashlib.sha256(raw).hexdigest()!=expected_sha: raise SystemExit('resource baseline SHA mismatch')",
            "baseline=json.loads(raw)",
            "if baseline.get('ssh_target')!=target or baseline.get('gpu_indices')!=indices: raise SystemExit('resource baseline binding mismatch')",
            "frozen=baseline.get('selected')",
            "if not isinstance(frozen,list) or len(frozen)!=4: raise SystemExit('resource baseline inventory mismatch')",
            "for current,original in zip(selected,frozen):",
            " if current['gpu_index']!=original.get('gpu_index') or current['gpu_uuid']!=original.get('gpu_uuid'): raise SystemExit('resource GPU drift')",
            " allowed={(process.get('pid'),process.get('process_name'),process.get('start_time_ticks')) for process in original.get('compute_processes',[])}",
            " observed={(process['pid'],process['process_name'],process['start_time_ticks']) for process in current['compute_processes']}",
            " if not observed.issubset(allowed): raise SystemExit('resource process drift')",
            "payload={'classification':'READY','resource_policy':'"
            + CONTROLLED_SHARED
            + "','baseline_sha256':expected_sha,'selected':selected,'benchmark_execution_authorized':False}",
            "print(json.dumps(payload,sort_keys=True,separators=(',',':')))",
        ])
    return "\n".join(lines)


def _query_shell(script, arguments):
    gpu_query = (
        "nvidia-smi --query-gpu=index,uuid,memory.free "
        "--format=csv,noheader,nounits"
    )
    process_query = (
        "nvidia-smi "
        "--query-compute-apps=gpu_uuid,pid,process_name,used_memory "
        "--format=csv,noheader,nounits"
    )
    shell = " && ".join([
        "set -eu",
        f"gpu_rows=\"$({gpu_query})\"",
        f"process_rows=\"$({process_query})\"",
        (
            "printf '%s\\n---PROCESSES---\\n%s\\n' "
            "\"$gpu_rows\" \"$process_rows\" | "
            f"python3 -c {shlex.quote(script)} "
            + " ".join(shlex.quote(str(value)) for value in arguments)
        ),
    ])
    return ["bash", "-lc", shell]


def capture_command(gpu_indices, *, ssh_target="sitian@10.232.195.203"):
    gpu_indices = _validate_gpu_indices(gpu_indices)
    return _query_shell(
        _inventory_script("capture"),
        [
            ",".join(str(value) for value in gpu_indices),
            MIN_GPU_FREE_BYTES,
            ssh_target,
        ],
    )


def guard_command(
    resource_policy,
    gpu_indices,
    *,
    baseline_path=None,
    baseline_sha256=None,
    ssh_target="sitian@10.232.195.203",
):
    gpu_indices = _validate_gpu_indices(gpu_indices)
    if resource_policy != CONTROLLED_SHARED:
        raise ValueError("guard command supports controlled_shared only")
    if (
        not isinstance(baseline_path, str)
        or not baseline_path
        or not _valid_sha256(baseline_sha256)
    ):
        raise ValueError("controlled resource baseline is invalid")
    return _query_shell(
        _inventory_script("guard"),
        [
            ",".join(str(value) for value in gpu_indices),
            MIN_GPU_FREE_BYTES,
            ssh_target,
            baseline_path,
            baseline_sha256,
        ],
    )


def sha256(path):
    path = Path(path)
    if not path.is_file() or path.is_symlink():
        raise ValueError("resource baseline must be a regular file")
    return hashlib.sha256(path.read_bytes()).hexdigest()
