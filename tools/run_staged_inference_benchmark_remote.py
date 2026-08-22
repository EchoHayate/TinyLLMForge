"""Safe remote orchestration for staged inference benchmark gates."""

from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import io
import json
import os
import platform
from pathlib import Path, PurePosixPath
import re
import shlex
import shutil
import signal
import subprocess
import sys
import tarfile
import tempfile
import time
import traceback


APPROVED_ROOT = (
    "/data00/home/sitian/tinyllmforge-workspaces/"
    "command-timeline-20260818"
)
REMOTE_HOST = "sitian@10.232.195.203"
REMOTE_PYTHON = "/data00/home/sitian/tllm/env/bin/python"
KRB5CCNAME = "FILE:/Users/bytedance/krb5cc_sitian"
TRACKING_REF = "origin/feat/kv-sparse-attention"
CONTROL_PATH = "/tmp/ssh-sitian-10.232.195.203"
MODEL_PATHS = {
    "qwen3-0.6b": "/data00/home/sitian/.ms_cache/Qwen/Qwen3-0___6B",
    "qwen3-8b": "/data00/home/sitian/.ms_cache/Qwen/Qwen3-8B",
}
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
LOCAL_ARTIFACT_ROOT = REPO_ROOT / "artifacts" / "staged_inference_benchmark"
RUN_TAG_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")
DOWNLOAD_CHUNK_BYTES = 4 * 1024 * 1024
MINIMUM_REMOTE_FREE_BYTES = 20 * 1024 * 1024 * 1024
MINIMUM_KERBEROS_LIFETIME_SECONDS = 5400


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _require_success(
    result: subprocess.CompletedProcess,
    context: str,
) -> subprocess.CompletedProcess:
    if result.returncode != 0:
        stderr = result.stderr
        if isinstance(stderr, bytes):
            detail = stderr.decode("utf-8", "replace").strip()
        else:
            detail = str(stderr or "").strip()
        raise RuntimeError(
            f"{context} failed"
            + (f": {detail}" if detail else "")
        )
    return result


def validate_run_tag(value: str) -> str:
    if (
        not isinstance(value, str)
        or RUN_TAG_PATTERN.fullmatch(value) is None
    ):
        raise ValueError("run tag is invalid")
    return value


def remote_paths(run_tag: str) -> dict[str, str]:
    tag = validate_run_tag(run_tag)
    root = APPROVED_ROOT + "/staged-benchmark"
    return {
        "staging": f"{root}/staging/{tag}",
        "primary": f"{root}/runs/{tag}",
        "controller": f"{root}/controller-verification/{tag}",
    }


def _validate_gpu_row(row: object) -> dict:
    if not isinstance(row, dict):
        raise ValueError("GPU row must be an object")
    index = row.get("index")
    uuid = row.get("uuid")
    name = row.get("name")
    memory_used_mib = row.get("memory_used_mib")
    utilization_percent = row.get("utilization_percent")
    compute_processes = row.get("compute_processes")
    if (
        isinstance(index, bool)
        or not isinstance(index, int)
        or index < 0
        or not isinstance(uuid, str)
        or not uuid
        or not isinstance(name, str)
        or not name
        or isinstance(memory_used_mib, bool)
        or not isinstance(memory_used_mib, int)
        or memory_used_mib < 0
        or isinstance(utilization_percent, bool)
        or not isinstance(utilization_percent, int)
        or utilization_percent < 0
        or not isinstance(compute_processes, list)
    ):
        raise ValueError("GPU row is invalid")
    normalized_processes = []
    for process in compute_processes:
        if not isinstance(process, dict):
            raise ValueError("GPU process row is invalid")
        pid = process.get("pid")
        process_name = process.get("process_name")
        if (
            isinstance(pid, bool)
            or not isinstance(pid, int)
            or pid <= 0
            or not isinstance(process_name, str)
            or not process_name
        ):
            raise ValueError("GPU process row is invalid")
        normalized_processes.append({
            "pid": pid,
            "process_name": process_name,
        })
    return {
        "index": index,
        "uuid": uuid,
        "name": name,
        "memory_used_mib": memory_used_mib,
        "utilization_percent": utilization_percent,
        "compute_processes": normalized_processes,
    }


def strict_clean_gpus(rows: list[dict]) -> list[dict]:
    normalized = [_validate_gpu_row(row) for row in rows]
    indices = [row["index"] for row in normalized]
    uuids = [row["uuid"] for row in normalized]
    if (
        len(indices) != len(set(indices))
        or len(uuids) != len(set(uuids))
    ):
        raise ValueError("GPU inventory contains duplicate identities")
    return [
        row
        for row in normalized
        if (
            row["memory_used_mib"] <= 1024
            and row["utilization_percent"] <= 5
            and not row["compute_processes"]
        )
    ]


def _validate_capacity_receipt(receipt: object) -> int:
    if (
        not isinstance(receipt, dict)
        or receipt.get("schema_version") != 1
        or receipt.get("model_tier") != "qwen3-8b"
        or receipt.get("status") != "PASS"
    ):
        raise ValueError("qwen3-8b requires a passing capacity preflight")
    count = receipt.get("required_gpu_count")
    if (
        isinstance(count, bool)
        or not isinstance(count, int)
        or count not in {1, 4}
    ):
        raise ValueError("capacity preflight GPU count is invalid")
    return count


def select_admitted_gpus(
    rows: list[dict],
    *,
    model_tier: str,
    capacity_receipt: dict | None,
) -> list[dict]:
    clean = strict_clean_gpus(rows)
    if model_tier == "qwen3-0.6b":
        if not clean:
            raise ValueError("Stage 1 requires one strict-clean GPU")
        return clean[:1]
    if model_tier != "qwen3-8b":
        raise ValueError("unsupported model tier")
    required = _validate_capacity_receipt(capacity_receipt)
    if len(clean) < required:
        raise ValueError(
            f"Stage 2 requires {required} strict-clean GPUs"
        )
    return clean[:required]


def gpu_ownership_conflict(
    *,
    baseline_rows: list[dict],
    observed_rows: list[dict],
    selected_gpu_indices: list[int],
    owned_pids: set[int],
    phase: str,
) -> dict | None:
    baseline = {
        row["index"]: _validate_gpu_row(row)
        for row in baseline_rows
    }
    observed = {
        row["index"]: _validate_gpu_row(row)
        for row in observed_rows
    }
    if (
        not isinstance(selected_gpu_indices, list)
        or not selected_gpu_indices
        or any(
            isinstance(index, bool)
            or not isinstance(index, int)
            or index < 0
            for index in selected_gpu_indices
        )
        or len(selected_gpu_indices) != len(set(selected_gpu_indices))
        or not isinstance(owned_pids, set)
        or any(
            isinstance(pid, bool)
            or not isinstance(pid, int)
            or pid <= 0
            for pid in owned_pids
        )
        or not isinstance(phase, str)
        or not phase
    ):
        raise ValueError("GPU ownership check arguments are invalid")
    conflicts = []
    for index in selected_gpu_indices:
        before = baseline.get(index)
        after = observed.get(index)
        if before is None or after is None:
            raise ValueError("selected GPU inventory is incomplete")
        if before["uuid"] != after["uuid"]:
            raise ValueError("selected GPU UUID changed")
        for process in after["compute_processes"]:
            if process["pid"] in owned_pids:
                continue
            conflicts.append({
                "gpu_index": index,
                "gpu_uuid": after["uuid"],
                "pid": process["pid"],
                "process_name": process["process_name"],
            })
    if not conflicts:
        return None
    return {
        "schema_version": 1,
        "phase": phase,
        "selected_gpu_indices": list(selected_gpu_indices),
        "conflicts": conflicts,
        "status": "CONFLICT",
    }


def _local_environment(
    environ: dict[str, str] | None = None,
) -> dict[str, str]:
    environment = dict(os.environ if environ is None else environ)
    environment["KRB5CCNAME"] = KRB5CCNAME
    return environment


def validate_kerberos(
    *,
    environ: dict[str, str] | None = None,
    command_runner=subprocess.run,
    now=lambda: datetime.now().astimezone(),
    minimum_lifetime_seconds: int = MINIMUM_KERBEROS_LIFETIME_SECONDS,
) -> dict:
    if (
        isinstance(minimum_lifetime_seconds, bool)
        or not isinstance(minimum_lifetime_seconds, int)
        or minimum_lifetime_seconds <= 0
        or not callable(now)
    ):
        raise ValueError("Kerberos lifetime policy is invalid")
    environment = _local_environment(environ)
    if environment.get("KRB5CCNAME") != KRB5CCNAME:
        raise ValueError("Kerberos cache path is invalid")
    result = command_runner(
        ["klist"],
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    _require_success(result, "Kerberos ticket validation")
    output = result.stdout
    if not isinstance(output, str):
        raise ValueError("Kerberos ticket inventory is invalid")
    cache = None
    principal = None
    tgt_principal = None
    expires_at = None
    ticket_pattern = re.compile(
        r"^([A-Z][a-z]{2}\s+\d{1,2}\s+\d{2}:\d{2}:\d{2}\s+"
        r"\d{4})\s+"
        r"([A-Z][a-z]{2}\s+\d{1,2}\s+\d{2}:\d{2}:\d{2}\s+"
        r"\d{4})\s+(\S+)$"
    )
    current = now()
    if not isinstance(current, datetime) or current.tzinfo is None:
        raise ValueError("current time must be timezone-aware")
    for raw_line in output.splitlines():
        line = raw_line.strip()
        if line.startswith("Credentials cache:"):
            cache = line.split(":", 1)[1].strip()
        elif (
            line.startswith("Principal:")
            or line.startswith("Default principal:")
        ):
            principal = line.split(":", 1)[1].strip()
        else:
            match = ticket_pattern.fullmatch(line)
            if match is None:
                continue
            ticket_principal = match.group(3)
            if not ticket_principal.startswith("krbtgt/"):
                continue
            tgt_principal = ticket_principal
            naive_expiry = datetime.strptime(
                match.group(2),
                "%b %d %H:%M:%S %Y",
            )
            expires_at = naive_expiry.replace(tzinfo=current.tzinfo)
    if (
        cache != KRB5CCNAME
        or not isinstance(principal, str)
        or not principal
        or not isinstance(tgt_principal, str)
        or expires_at is None
    ):
        raise ValueError("Kerberos TGT inventory is invalid")
    remaining = int((expires_at - current).total_seconds())
    if remaining < minimum_lifetime_seconds:
        raise ValueError(
            "Kerberos remaining lifetime is below the required minimum"
        )
    return {
        "schema_version": 1,
        "status": "PASS",
        "cache": cache,
        "principal": principal,
        "tgt_principal": tgt_principal,
        "expires_at": expires_at.isoformat(),
        "minimum_required_lifetime_seconds": (
            minimum_lifetime_seconds
        ),
        "remaining_lifetime_seconds": remaining,
    }


def require_pushed_head(
    repo_root: Path,
    *,
    command_runner=subprocess.run,
) -> str:
    heads = []
    for revision in ("HEAD", TRACKING_REF):
        result = command_runner(
            ["git", "rev-parse", revision],
            cwd=Path(repo_root),
            text=True,
            capture_output=True,
            check=False,
        )
        _require_success(result, f"resolve {revision}")
        value = result.stdout.strip()
        if re.fullmatch(r"[0-9a-f]{40}", value) is None:
            raise ValueError(f"{revision} is not a commit SHA")
        heads.append(value)
    if heads[0] != heads[1]:
        raise ValueError(
            "local HEAD must equal origin/feat/kv-sparse-attention"
        )
    return heads[0]


def _ssh_command(remote_command: str) -> list[str]:
    if not isinstance(remote_command, str) or not remote_command:
        raise ValueError("remote command is invalid")
    return [
        "ssh",
        "-o",
        "ControlMaster=no",
        "-o",
        "ControlPath=none",
        "-o",
        "BatchMode=yes",
        "-o",
        "GSSAPIAuthentication=yes",
        REMOTE_HOST,
        remote_command,
    ]


def _run_remote(
    remote_command: str,
    *,
    command_runner=subprocess.run,
    text: bool = True,
) -> subprocess.CompletedProcess:
    return command_runner(
        _ssh_command(remote_command),
        env=_local_environment(),
        text=text,
        capture_output=True,
        check=False,
    )


def _run_remote_with_input(
    remote_command: str,
    payload: bytes,
    *,
    command_runner=subprocess.run,
) -> subprocess.CompletedProcess:
    if not isinstance(payload, bytes):
        raise ValueError("remote input payload is invalid")
    return command_runner(
        _ssh_command(remote_command),
        env=_local_environment(),
        input=payload,
        text=False,
        capture_output=True,
        check=False,
    )


def require_remote_destinations_absent(
    paths: dict[str, str],
    *,
    command_runner=subprocess.run,
) -> None:
    expected = {"staging", "primary", "controller"}
    if set(paths) != expected:
        raise ValueError("remote path inventory is invalid")
    for path in paths.values():
        if not path.startswith(APPROVED_ROOT + "/"):
            raise ValueError("remote path is outside approved root")
    script = (
        "import json,os;"
        f"paths={_canonical_json(list(paths.values()))};"
        "print(json.dumps({p:os.path.exists(p) for p in paths},"
        "sort_keys=True))"
    )
    result = _run_remote(
        "python3 -c " + shlex.quote(script),
        command_runner=command_runner,
    )
    _require_success(result, "remote immutability preflight")
    try:
        existence = json.loads(result.stdout)
    except (TypeError, json.JSONDecodeError) as error:
        raise ValueError("remote path receipt is invalid") from error
    if (
        not isinstance(existence, dict)
        or set(existence) != set(paths.values())
        or any(not isinstance(value, bool) for value in existence.values())
    ):
        raise ValueError("remote path receipt is invalid")
    existing = sorted(
        path for path, present in existence.items() if present
    )
    if existing:
        raise ValueError(
            "immutable remote destination already exists: "
            + ", ".join(existing)
        )


def query_remote_gpu_rows(
    *,
    command_runner=subprocess.run,
) -> list[dict]:
    script = "\n".join((
        "import json,subprocess",
        "gpu=subprocess.run([",
        " 'nvidia-smi',",
        " '--query-gpu=index,uuid,name,memory.used,utilization.gpu',",
        " '--format=csv,noheader,nounits',",
        "],text=True,capture_output=True,check=False)",
        "apps=subprocess.run([",
        " 'nvidia-smi',",
        " '--query-compute-apps=pid,gpu_uuid,process_name',",
        " '--format=csv,noheader,nounits',",
        "],text=True,capture_output=True,check=False)",
        "if gpu.returncode or apps.returncode:",
        " raise RuntimeError('GPU inventory query failed')",
        "processes={}",
        "for line in apps.stdout.splitlines():",
        " fields=[part.strip() for part in line.split(',',2)]",
        " if len(fields)!=3:",
        "  raise ValueError('GPU process row is invalid')",
        " processes.setdefault(fields[1],[]).append({",
        "  'pid':int(fields[0]),'process_name':fields[2]})",
        "rows=[]",
        "for line in gpu.stdout.splitlines():",
        " fields=[part.strip() for part in line.split(',',4)]",
        " if len(fields)!=5:",
        "  raise ValueError('GPU inventory row is invalid')",
        " rows.append({",
        "  'index':int(fields[0]),'uuid':fields[1],",
        "  'name':fields[2],'memory_used_mib':int(fields[3]),",
        "  'utilization_percent':int(fields[4]),",
        "  'compute_processes':processes.get(fields[1],[])})",
        "print(json.dumps(rows,sort_keys=True,separators=(',',':')))",
    ))
    result = _run_remote(
        "python3 -c " + shlex.quote(script),
        command_runner=command_runner,
    )
    _require_success(result, "remote GPU inventory")
    try:
        payload = json.loads(result.stdout)
    except (TypeError, json.JSONDecodeError) as error:
        raise ValueError("remote GPU inventory is invalid") from error
    if not isinstance(payload, list):
        raise ValueError("remote GPU inventory is invalid")
    rows = [_validate_gpu_row(row) for row in payload]
    if rows != sorted(rows, key=lambda row: row["index"]):
        raise ValueError("remote GPU inventory is not sorted")
    return rows


def _query_gpu_rows_local() -> list[dict]:
    gpu = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,name,memory.used,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    apps = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,gpu_uuid,process_name",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    _require_success(gpu, "local GPU inventory")
    _require_success(apps, "local GPU process inventory")
    processes = {}
    for line in apps.stdout.splitlines():
        fields = [part.strip() for part in line.split(",", 2)]
        if len(fields) != 3:
            raise ValueError("GPU process row is invalid")
        processes.setdefault(fields[1], []).append({
            "pid": int(fields[0]),
            "process_name": fields[2],
        })
    rows = []
    for line in gpu.stdout.splitlines():
        fields = [part.strip() for part in line.split(",", 4)]
        if len(fields) != 5:
            raise ValueError("GPU inventory row is invalid")
        rows.append(_validate_gpu_row({
            "index": int(fields[0]),
            "uuid": fields[1],
            "name": fields[2],
            "memory_used_mib": int(fields[3]),
            "utilization_percent": int(fields[4]),
            "compute_processes": processes.get(fields[1], []),
        }))
    return sorted(rows, key=lambda row: row["index"])


def _owned_process_group_pids(process_group_id: int) -> set[int]:
    owned = set()
    proc = Path("/proc")
    if not proc.is_dir():
        raise RuntimeError("process ownership requires procfs")
    for entry in proc.iterdir():
        if not entry.name.isdigit():
            continue
        try:
            fields = (entry / "stat").read_text().split()
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        if len(fields) > 4 and int(fields[4]) == process_group_id:
            owned.add(int(entry.name))
    return owned


def _terminate_owned_process_group(
    process: subprocess.Popen,
    process_group_id: int,
) -> None:
    started = time.monotonic()
    term_deadline = started + 5
    hard_deadline = started + 30
    try:
        os.killpg(process_group_id, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        pass
    killed = False
    while _owned_process_group_pids(process_group_id):
        now = time.monotonic()
        if not killed and now >= term_deadline:
            try:
                os.killpg(process_group_id, signal.SIGKILL)
            except ProcessLookupError:
                return
            killed = True
        if now >= hard_deadline:
            raise RuntimeError("owned process group did not terminate")
        time.sleep(0.2)
    if process.poll() is None:
        try:
            process.wait(timeout=1)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process_group_id, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait(timeout=5)


def _selected_rows(
    rows: list[dict],
    indices: list[int],
) -> list[dict]:
    selected = {
        row["index"]: row
        for row in rows
        if row["index"] in indices
    }
    if set(selected) != set(indices):
        raise ValueError("selected GPU inventory is incomplete")
    return [selected[index] for index in indices]


def _sample_owned_gpu_rows(
    *,
    process_group_id: int,
    owned_pids: set[int],
) -> tuple[list[dict], set[int]]:
    observed_owned = set(owned_pids)
    observed_owned.update(
        _owned_process_group_pids(process_group_id)
    )
    rows = _query_gpu_rows_local()
    observed_owned.update(
        _owned_process_group_pids(process_group_id)
    )
    return rows, observed_owned


def _persist_conflict(
    primary: Path,
    case_id: str,
    conflict: dict,
) -> None:
    root = primary / "ownership_conflicts"
    root.mkdir(parents=True, exist_ok=True)
    _write_json_exclusive(root / f"{case_id}.json", conflict)


def append_ownership_sample(
    path: Path,
    *,
    phase: str,
    rows: list[dict],
    owned_pids: set[int],
) -> None:
    if (
        not isinstance(phase, str)
        or not phase
        or not isinstance(rows, list)
        or not isinstance(owned_pids, set)
        or any(
            isinstance(pid, bool)
            or not isinstance(pid, int)
            or pid <= 0
            for pid in owned_pids
        )
    ):
        raise ValueError("GPU ownership sample is invalid")
    sample = {
        "schema_version": 1,
        "sampled_at_unix_ns": time.time_ns(),
        "phase": phase,
        "gpu_rows": [_validate_gpu_row(row) for row in rows],
        "owned_pids": sorted(owned_pids),
    }
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("ab") as handle:
        handle.write((_canonical_json(sample) + "\n").encode("utf-8"))
        handle.flush()
        os.fsync(handle.fileno())


def _run_monitored_case(
    *,
    gate_module,
    source_root: Path,
    primary: Path,
    case_id: str,
    selected_gpu_indices: list[int],
    selected_gpu_uuids: list[str],
    environment: dict[str, str],
) -> dict:
    before_all = _query_gpu_rows_local()
    before = _selected_rows(before_all, selected_gpu_indices)
    ownership_samples = primary / "gpu_ownership_samples.jsonl"
    append_ownership_sample(
        ownership_samples,
        phase=f"{case_id}:before",
        rows=before_all,
        owned_pids=set(),
    )
    if strict_clean_gpus(before) != before:
        conflict = gpu_ownership_conflict(
            baseline_rows=before,
            observed_rows=before,
            selected_gpu_indices=selected_gpu_indices,
            owned_pids=set(),
            phase=f"{case_id}:before",
        )
        if conflict is not None:
            _persist_conflict(primary, case_id, conflict)
        raise ValueError("selected GPU is not clean before case")
    child_code = "\n".join((
        "import json,sys",
        f"sys.path.insert(0,{str(source_root)!r})",
        f"sys.path.insert(0,{str(source_root / 'tools')!r})",
        "import staged_inference_benchmark_gate as gate",
        (
            "receipt=gate.launch_case("
            f"{str(primary)!r},{case_id!r},"
            f"python_bin={REMOTE_PYTHON!r})"
        ),
        "print(json.dumps(receipt,sort_keys=True,separators=(',',':')))",
    ))
    process = subprocess.Popen(
        [REMOTE_PYTHON, "-c", child_code],
        cwd=source_root,
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )
    process_group_id = process.pid
    observed_owned = {process.pid}
    try:
        while process.poll() is None:
            observed, observed_owned = _sample_owned_gpu_rows(
                process_group_id=process_group_id,
                owned_pids=observed_owned,
            )
            append_ownership_sample(
                ownership_samples,
                phase=f"{case_id}:running",
                rows=observed,
                owned_pids=observed_owned,
            )
            conflict = gpu_ownership_conflict(
                baseline_rows=before_all,
                observed_rows=observed,
                selected_gpu_indices=selected_gpu_indices,
                owned_pids=observed_owned,
                phase=f"{case_id}:running",
            )
            if conflict is not None:
                _persist_conflict(primary, case_id, conflict)
                _terminate_owned_process_group(
                    process,
                    process_group_id,
                )
                raise ValueError(
                    "selected GPU acquired an unowned process"
                )
            time.sleep(0.5)
        stdout, stderr = process.communicate(timeout=30)
    finally:
        if process.poll() is None:
            _terminate_owned_process_group(process, process_group_id)
    case_dir = primary / "cases" / case_id
    case_dir.mkdir(parents=True, exist_ok=True)
    (case_dir / "controller.stdout.log").write_text(
        stdout,
        encoding="utf-8",
    )
    (case_dir / "controller.stderr.log").write_text(
        stderr,
        encoding="utf-8",
    )
    if process.returncode != 0:
        raise RuntimeError(
            f"case controller failed with {process.returncode}"
        )
    try:
        receipt = json.loads(stdout)
    except json.JSONDecodeError as error:
        raise ValueError("case controller receipt is invalid") from error
    if (
        not isinstance(receipt, dict)
        or receipt.get("case_id") != case_id
        or receipt.get("returncode") != 0
    ):
        raise RuntimeError("benchmark worker case failed")
    deadline = time.monotonic() + 30
    while True:
        observed = _query_gpu_rows_local()
        selected = _selected_rows(observed, selected_gpu_indices)
        append_ownership_sample(
            ownership_samples,
            phase=f"{case_id}:after",
            rows=observed,
            owned_pids=observed_owned,
        )
        conflict = gpu_ownership_conflict(
            baseline_rows=before_all,
            observed_rows=observed,
            selected_gpu_indices=selected_gpu_indices,
            owned_pids=observed_owned,
            phase=f"{case_id}:after",
        )
        if conflict is not None:
            _persist_conflict(primary, case_id, conflict)
            raise ValueError(
                "selected GPU acquired an unowned process"
            )
        if strict_clean_gpus(selected) == selected:
            break
        if time.monotonic() >= deadline:
            raise TimeoutError("selected GPU did not return to clean state")
        time.sleep(0.5)
    if [row["uuid"] for row in selected] != selected_gpu_uuids:
        raise ValueError("selected GPU UUID changed after case")
    return receipt


def probe_remote_requirements(
    model_tier: str,
    *,
    command_runner=subprocess.run,
) -> dict:
    model_path = MODEL_PATHS.get(model_tier)
    if model_path is None:
        raise ValueError("unsupported model tier")
    script = "\n".join((
        "import json,os",
        f"python_path={REMOTE_PYTHON!r}",
        f"model_path={model_path!r}",
        f"approved_root={APPROVED_ROOT!r}",
        "stat=os.statvfs(approved_root)",
        "payload={",
        " 'python':{",
        "  'path':python_path,",
        "  'is_file':os.path.isfile(python_path),",
        "  'is_executable':os.access(python_path,os.X_OK),",
        " },",
        " 'model':{",
        "  'path':model_path,",
        "  'is_dir':os.path.isdir(model_path),",
        "  'config_path':os.path.join(model_path,'config.json'),",
        "  'config_is_file':os.path.isfile(",
        "   os.path.join(model_path,'config.json')),",
        " },",
        " 'approved_root':{",
        "  'path':approved_root,",
        "  'is_dir':os.path.isdir(approved_root),",
        "  'free_bytes':stat.f_bavail*stat.f_frsize,",
        " },",
        "}",
        "print(json.dumps(payload,sort_keys=True,separators=(',',':')))",
    ))
    result = _run_remote(
        "python3 -c " + shlex.quote(script),
        command_runner=command_runner,
    )
    _require_success(result, "remote requirement probe")
    try:
        payload = json.loads(result.stdout)
    except (TypeError, json.JSONDecodeError) as error:
        raise ValueError("remote requirement receipt is invalid") from error
    expected_paths = {
        "python": REMOTE_PYTHON,
        "model": model_path,
        "approved_root": APPROVED_ROOT,
    }
    if not isinstance(payload, dict):
        raise ValueError("remote requirement receipt is invalid")
    python = payload.get("python")
    model = payload.get("model")
    root = payload.get("approved_root")
    if (
        not isinstance(python, dict)
        or python.get("path") != expected_paths["python"]
        or python.get("is_file") is not True
        or python.get("is_executable") is not True
        or not isinstance(model, dict)
        or model.get("path") != expected_paths["model"]
        or model.get("is_dir") is not True
        or model.get("config_path") != model_path + "/config.json"
        or model.get("config_is_file") is not True
        or not isinstance(root, dict)
        or root.get("path") != expected_paths["approved_root"]
        or root.get("is_dir") is not True
        or isinstance(root.get("free_bytes"), bool)
        or not isinstance(root.get("free_bytes"), int)
        or root["free_bytes"] < MINIMUM_REMOTE_FREE_BYTES
    ):
        raise ValueError("remote requirements are not satisfied")
    return payload


def build_preflight_receipt(
    *,
    gate_name: str,
    model_tier: str,
    run_tag: str,
    source_commit: str,
    kerberos_receipt: dict,
    remote_requirements: dict,
    gpu_rows: list[dict],
    selected_rows: list[dict],
    paths: dict[str, str],
    capacity_receipt: dict | None,
) -> dict:
    tag = validate_run_tag(run_tag)
    if gate_name not in {"prefix", "chunked"}:
        raise ValueError("unsupported gate")
    if model_tier not in MODEL_PATHS:
        raise ValueError("unsupported model tier")
    if re.fullmatch(r"[0-9a-f]{40}", source_commit) is None:
        raise ValueError("source commit is invalid")
    if (
        not isinstance(kerberos_receipt, dict)
        or kerberos_receipt.get("status") != "PASS"
        or not isinstance(remote_requirements, dict)
        or paths != remote_paths(tag)
    ):
        raise ValueError("preflight evidence is invalid")
    normalized_rows = [_validate_gpu_row(row) for row in gpu_rows]
    normalized_selected = [
        _validate_gpu_row(row) for row in selected_rows
    ]
    if any(row not in normalized_rows for row in normalized_selected):
        raise ValueError("selected GPU is absent from inventory")
    if model_tier == "qwen3-8b":
        _validate_capacity_receipt(capacity_receipt)
    elif capacity_receipt is not None:
        raise ValueError("Stage 1 must not use a capacity receipt")
    return {
        "schema_version": 1,
        "status": "READY",
        "gate": gate_name,
        "model_tier": model_tier,
        "run_tag": tag,
        "source_commit": source_commit,
        "kerberos": json.loads(json.dumps(kerberos_receipt)),
        "remote_requirements": json.loads(
            json.dumps(remote_requirements)
        ),
        "gpu_rows": normalized_rows,
        "selected_gpu_indices": [
            row["index"] for row in normalized_selected
        ],
        "selected_gpu_uuids": [
            row["uuid"] for row in normalized_selected
        ],
        "remote_paths": dict(paths),
        "capacity_receipt": (
            None
            if capacity_receipt is None
            else json.loads(json.dumps(capacity_receipt))
        ),
    }


def _write_json_exclusive(path: Path, payload: object) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(
        f".{destination.name}.tmp-{os.getpid()}-{time.time_ns()}"
    )
    try:
        with temporary.open("xb") as handle:
            handle.write(
                (
                    json.dumps(
                        payload,
                        sort_keys=True,
                        indent=2,
                        ensure_ascii=False,
                        allow_nan=False,
                    )
                    + "\n"
                ).encode("utf-8")
            )
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, destination)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def run_preflight(
    *,
    repo_root: Path,
    gate_name: str,
    model_tier: str,
    run_tag: str,
    output_dir: Path,
    capacity_receipt: dict | None,
    command_runner=subprocess.run,
    kerberos_validator=validate_kerberos,
    source_validator=require_pushed_head,
    destination_validator=require_remote_destinations_absent,
    requirement_probe=probe_remote_requirements,
    gpu_query=query_remote_gpu_rows,
) -> dict:
    destination = Path(output_dir)
    if destination.exists():
        raise ValueError(
            f"local preflight destination already exists: {destination}"
        )
    paths = remote_paths(run_tag)
    kerberos = kerberos_validator(
        command_runner=command_runner,
    )
    source_commit = source_validator(
        Path(repo_root),
        command_runner=command_runner,
    )
    destination_validator(
        paths,
        command_runner=command_runner,
    )
    requirements = requirement_probe(
        model_tier,
        command_runner=command_runner,
    )
    before = gpu_query(command_runner=command_runner)
    selected_before = select_admitted_gpus(
        before,
        model_tier=model_tier,
        capacity_receipt=capacity_receipt,
    )
    after = gpu_query(command_runner=command_runner)
    selected_after = select_admitted_gpus(
        after,
        model_tier=model_tier,
        capacity_receipt=capacity_receipt,
    )
    before_identity = [
        (row["index"], row["uuid"]) for row in selected_before
    ]
    after_identity = [
        (row["index"], row["uuid"]) for row in selected_after
    ]
    if before_identity != after_identity:
        raise ValueError("selected GPU ownership changed during preflight")
    conflict = gpu_ownership_conflict(
        baseline_rows=before,
        observed_rows=after,
        selected_gpu_indices=[
            row["index"] for row in selected_before
        ],
        owned_pids=set(),
        phase="preflight-after-probes",
    )
    if conflict is not None:
        raise ValueError(
            "selected GPU ownership changed during preflight: "
            + _canonical_json(conflict)
        )
    receipt = build_preflight_receipt(
        gate_name=gate_name,
        model_tier=model_tier,
        run_tag=run_tag,
        source_commit=source_commit,
        kerberos_receipt=kerberos,
        remote_requirements=requirements,
        gpu_rows=before,
        selected_rows=selected_before,
        paths=paths,
        capacity_receipt=capacity_receipt,
    )
    receipt["gpu_rows_before"] = before
    receipt["gpu_rows_after"] = after
    destination.mkdir(parents=True)
    _write_json_exclusive(destination / "preflight.json", receipt)
    return receipt


def build_execution_spec(
    *,
    preflight: dict,
    source_evidence: dict,
    promotion: dict | None,
) -> dict:
    if (
        not isinstance(preflight, dict)
        or preflight.get("schema_version") != 1
        or preflight.get("status") != "READY"
    ):
        raise ValueError("preflight receipt is invalid")
    gate_name = preflight.get("gate")
    model_tier = preflight.get("model_tier")
    run_tag = validate_run_tag(preflight.get("run_tag"))
    paths = remote_paths(run_tag)
    if (
        gate_name not in {"prefix", "chunked"}
        or model_tier not in MODEL_PATHS
        or preflight.get("remote_paths") != paths
        or not isinstance(source_evidence, dict)
        or source_evidence.get("base_commit")
        != preflight.get("source_commit")
        or source_evidence.get("local_head")
        != preflight.get("source_commit")
        or source_evidence.get("tracking_head")
        != preflight.get("source_commit")
        or source_evidence.get("dirty") is not False
    ):
        raise ValueError("execution source identity is invalid")
    selected_indices = preflight.get("selected_gpu_indices")
    selected_uuids = preflight.get("selected_gpu_uuids")
    if (
        not isinstance(selected_indices, list)
        or not selected_indices
        or not isinstance(selected_uuids, list)
        or len(selected_indices) != len(selected_uuids)
    ):
        raise ValueError("execution GPU identity is invalid")
    if model_tier == "qwen3-0.6b":
        if len(selected_indices) != 1 or promotion is not None:
            raise ValueError("Stage 1 execution identity is invalid")
    elif not isinstance(promotion, dict):
        raise ValueError("Stage 2 execution requires promotion evidence")
    return {
        "schema_version": 1,
        "gate": gate_name,
        "model_tier": model_tier,
        "run_tag": run_tag,
        "model_path": MODEL_PATHS[model_tier],
        "remote_python": REMOTE_PYTHON,
        "remote_paths": paths,
        "runtime_environment": remote_runtime_environment(
            paths["primary"]
        ),
        "selected_gpu_indices": list(selected_indices),
        "selected_gpu_uuids": list(selected_uuids),
        "source_evidence": json.loads(json.dumps(source_evidence)),
        "promotion": (
            None
            if promotion is None
            else json.loads(json.dumps(promotion))
        ),
    }


def prepare_execution_payload(
    *,
    repo_root: Path,
    preflight_path: Path,
    build_root: Path,
    promotion: dict | None,
    source_collector=None,
) -> tuple[bytes, dict]:
    destination = Path(build_root)
    if destination.exists():
        raise ValueError("local payload build destination already exists")
    preflight = _load_json_file(preflight_path)
    destination.mkdir(parents=True)
    source_evidence_root = destination / "source-evidence"
    if source_collector is None:
        from tools import staged_inference_benchmark_gate as gate

        source_collector = gate.collect_source_evidence
    source_evidence = source_collector(
        repo_root=Path(repo_root),
        output_dir=source_evidence_root,
    )
    spec = build_execution_spec(
        preflight=preflight,
        source_evidence=source_evidence,
        promotion=promotion,
    )
    payload_root = destination / "payload"
    payload_root.mkdir()
    shutil.copytree(
        source_evidence_root / "source",
        payload_root / "source",
    )
    evidence_root = payload_root / "evidence"
    evidence_root.mkdir()
    (evidence_root / "source_snapshot.tar").write_bytes(
        build_deterministic_tar(
            source_evidence_root / "source",
            prefix="source",
        )
    )
    for filename in ("source.patch", "source_evidence.json"):
        shutil.copyfile(
            source_evidence_root / filename,
            evidence_root / filename,
        )
    shutil.copyfile(preflight_path, evidence_root / "preflight.json")
    control_root = payload_root / "control"
    control_root.mkdir()
    for source in (
        Path(__file__).resolve(),
        REPO_ROOT / "tools" / "staged_inference_benchmark_verify.py",
    ):
        shutil.copyfile(source, control_root / source.name)
    _write_json_exclusive(
        payload_root / "execution_spec.json",
        spec,
    )
    payload = build_deterministic_tar(
        payload_root,
        prefix="payload",
    )
    return payload, spec


def upload_execution_payload(
    *,
    payload: bytes,
    spec: dict,
    command_runner=subprocess.run,
) -> dict:
    if not isinstance(spec, dict):
        raise ValueError("execution spec is invalid")
    paths = spec.get("remote_paths")
    if paths != remote_paths(spec.get("run_tag")):
        raise ValueError("execution remote paths are invalid")
    staging = paths["staging"]
    script = "\n".join((
        "import hashlib,io,json,os,pathlib,sys,tarfile",
        f"staging=pathlib.Path({staging!r})",
        "staging.parent.mkdir(parents=True,exist_ok=True)",
        "staging.mkdir(parents=False,exist_ok=False)",
        "payload=sys.stdin.buffer.read()",
        "archive_path=staging/'payload.tar'",
        "with archive_path.open('xb') as handle:",
        " handle.write(payload)",
        " handle.flush()",
        " os.fsync(handle.fileno())",
        "with tarfile.open(fileobj=io.BytesIO(payload),mode='r:') as tar:",
        " members=tar.getmembers()",
        " for member in members:",
        "  path=pathlib.PurePosixPath(member.name)",
        "  if (not member.isfile() or member.issym() or member.islnk()",
        "      or path.is_absolute() or '..' in path.parts",
        "      or not path.parts or path.parts[0]!='payload'):",
        "   raise ValueError('unsafe staging archive member')",
        " for member in members:",
        "  destination=staging/member.name",
        "  destination.parent.mkdir(parents=True,exist_ok=True)",
        "  source=tar.extractfile(member)",
        "  if source is None:",
        "   raise ValueError('staging archive member is unreadable')",
        "  with destination.open('xb') as handle:",
        "   handle.write(source.read())",
        "receipt={'schema_version':1,'status':'PASS',",
        " 'size_bytes':len(payload),",
        " 'sha256':hashlib.sha256(payload).hexdigest(),",
        " 'staging':str(staging)}",
        "print(json.dumps(receipt,sort_keys=True,separators=(',',':')))",
    ))
    result = _run_remote_with_input(
        f"{REMOTE_PYTHON} -c {shlex.quote(script)}",
        payload,
        command_runner=command_runner,
    )
    _require_success(result, "remote payload upload")
    try:
        receipt = json.loads(
            result.stdout.decode("utf-8")
            if isinstance(result.stdout, bytes)
            else result.stdout
        )
    except (TypeError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("remote upload receipt is invalid") from error
    expected = {
        "schema_version": 1,
        "status": "PASS",
        "size_bytes": len(payload),
        "sha256": _sha256_bytes(payload),
        "staging": staging,
    }
    if receipt != expected:
        raise ValueError("remote upload receipt mismatch")
    return receipt


def _collect_environment_evidence(
    *,
    spec: dict,
    gate_module,
) -> dict:
    import torch

    rows = _query_gpu_rows_local()
    selected = _selected_rows(
        rows,
        spec["selected_gpu_indices"],
    )
    if [row["uuid"] for row in selected] != spec["selected_gpu_uuids"]:
        raise ValueError("selected GPU UUID differs from preflight")
    if strict_clean_gpus(selected) != selected:
        raise ValueError("selected GPU is not clean at execution start")
    config_path = Path(spec["model_path"]) / "config.json"
    if not config_path.is_file():
        raise ValueError("model config is missing")
    engine_limits = (
        dict(gate_module.PREFIX_ENGINE_LIMITS)
        if spec["gate"] == "prefix"
        else dict(gate_module.contract.CHUNKED_ENGINE_CONFIG)
    )
    return {
        "model_tier": spec["model_tier"],
        "python_version": platform.python_version(),
        "torch_version": str(torch.__version__),
        "cuda_version": str(torch.version.cuda or "none"),
        "gpu_inventory": [
            {
                "index": row["index"],
                "uuid": row["uuid"],
                "name": row["name"],
            }
            for row in selected
        ],
        "selected_gpu_indices": [
            row["index"] for row in selected
        ],
        "model_config_sha256": hashlib.sha256(
            config_path.read_bytes()
        ).hexdigest(),
        "checkpoint_identifier": spec["model_tier"],
        "model_path": spec["model_path"],
        "engine_limits": engine_limits,
    }


def _remote_execute_from_spec(spec_path: Path) -> dict:
    spec_file = Path(spec_path).resolve()
    payload_root = spec_file.parent
    if payload_root.name != "payload":
        raise ValueError("remote execution payload root is invalid")
    spec = _load_json_file(spec_file)
    paths = spec.get("remote_paths")
    if paths != remote_paths(spec.get("run_tag")):
        raise ValueError("remote execution paths are invalid")
    staging = Path(paths["staging"])
    primary = Path(paths["primary"])
    controller = Path(paths["controller"])
    if spec_file != (staging / "payload" / "execution_spec.json"):
        raise ValueError("remote execution spec location is invalid")
    source_root = payload_root / "source"
    control_root = payload_root / "control"
    evidence_root = payload_root / "evidence"
    sys.path.insert(0, str(control_root))
    sys.path.insert(0, str(source_root / "tools"))
    sys.path.insert(0, str(source_root))
    import staged_inference_benchmark_gate as gate
    import staged_inference_benchmark_verify as verifier

    result_path = staging / "remote_execution_result.json"
    try:
        environment = os.environ.copy()
        environment.update(spec["runtime_environment"])
        environment["CUDA_VISIBLE_DEVICES"] = ",".join(
            str(index) for index in spec["selected_gpu_indices"]
        )
        environment["PYTHONPATH"] = str(source_root)
        os.environ.update(environment)
        environment_evidence = _collect_environment_evidence(
            spec=spec,
            gate_module=gate,
        )
        manifest = gate.initialize_run(
            run_dir=primary,
            run_tag=spec["run_tag"],
            gate_name=spec["gate"],
            model_tier=spec["model_tier"],
            source_evidence=spec["source_evidence"],
            environment_evidence=environment_evidence,
            promotion=spec["promotion"],
        )
        for path in spec["runtime_environment"].values():
            Path(path).mkdir(parents=True, exist_ok=True)
        for filename in (
            "source_snapshot.tar",
            "source.patch",
        ):
            shutil.copyfile(
                evidence_root / filename,
                primary / filename,
            )
        _write_json_exclusive(
            primary / "remote_preflight.json",
            _load_json_file(evidence_root / "preflight.json"),
        )
        case_receipts = []
        for case_id in manifest["case_order"]:
            case_receipts.append(_run_monitored_case(
                gate_module=gate,
                source_root=source_root,
                primary=primary,
                case_id=case_id,
                selected_gpu_indices=spec[
                    "selected_gpu_indices"
                ],
                selected_gpu_uuids=spec[
                    "selected_gpu_uuids"
                ],
                environment=environment,
            ))
        summary = gate.finalize_run(primary)
        verifier.verify_run(primary, controller)
        remote_receipt = _load_json_file(
            controller / "verification_receipt.json"
        )
        result = {
            "schema_version": 1,
            "status": "PASS",
            "run_tag": spec["run_tag"],
            "primary_run": str(primary),
            "controller_run": str(controller),
            "case_count": len(case_receipts),
            "classification": summary["classification"],
            "verification_receipt": remote_receipt,
        }
        _write_json_exclusive(
            controller / "remote_execution_receipt.json",
            result,
        )
        _write_json_exclusive(result_path, result)
        return result
    except BaseException as error:
        result = {
            "schema_version": 1,
            "status": "FAILED",
            "run_tag": spec.get("run_tag"),
            "primary_run": str(primary),
            "controller_run": str(controller),
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": traceback.format_exc(),
        }
        if primary.is_dir():
            failure_path = primary / "runtime_failure.json"
            if not failure_path.exists():
                _write_json_exclusive(failure_path, result)
        if not result_path.exists():
            _write_json_exclusive(result_path, result)
        return result


def launch_remote_execution(
    *,
    spec: dict,
    command_runner=subprocess.run,
    sleep=time.sleep,
    monotonic=time.monotonic,
    timeout_seconds: int = 24 * 60 * 60,
) -> dict:
    if (
        not callable(sleep)
        or not callable(monotonic)
        or isinstance(timeout_seconds, bool)
        or not isinstance(timeout_seconds, int)
        or timeout_seconds <= 0
    ):
        raise ValueError("remote execution polling policy is invalid")
    paths = spec.get("remote_paths")
    if paths != remote_paths(spec.get("run_tag")):
        raise ValueError("execution spec paths are invalid")
    spec_path = (
        Path(paths["staging"])
        / "payload"
        / "execution_spec.json"
    )
    control_path = (
        Path(paths["staging"])
        / "payload"
        / "control"
        / Path(__file__).name
    )
    result_path = (
        Path(paths["staging"]) / "remote_execution_result.json"
    )
    pid_path = Path(paths["staging"]) / "remote_execution.pid"
    stdout_path = (
        Path(paths["staging"]) / "remote_execution.stdout.log"
    )
    stderr_path = (
        Path(paths["staging"]) / "remote_execution.stderr.log"
    )
    child_script = "\n".join((
        "import json,sys",
        f"sys.path.insert(0,{str(control_path.parent)!r})",
        (
            "import run_staged_inference_benchmark_remote "
            "as remote"
        ),
        (
            "result=remote._remote_execute_from_spec("
            f"{str(spec_path)!r})"
        ),
        "print(json.dumps(result,sort_keys=True,separators=(',',':')))",
        "raise SystemExit(0 if result.get('status')=='PASS' else 1)",
    ))
    launch_script = "\n".join((
        "import json,os,pathlib,subprocess",
        f"pid_path=pathlib.Path({str(pid_path)!r})",
        f"stdout_path=pathlib.Path({str(stdout_path)!r})",
        f"stderr_path=pathlib.Path({str(stderr_path)!r})",
        "with (stdout_path.open('xb') as stdout_handle,",
        "      stderr_path.open('xb') as stderr_handle):",
        " process=subprocess.Popen(",
        f"  [{REMOTE_PYTHON!r},'-c',{child_script!r}],",
        f"  cwd={str(spec_path.parent)!r},",
        "  stdout=stdout_handle,stderr=stderr_handle,",
        "  start_new_session=True)",
        "with pid_path.open('x',encoding='utf-8') as handle:",
        " handle.write(str(process.pid)+'\\n')",
        " handle.flush()",
        " os.fsync(handle.fileno())",
        "print(json.dumps({'schema_version':1,'status':'STARTED',",
        " 'pid':process.pid},sort_keys=True,separators=(',',':')))",
    ))
    launch = _run_remote(
        f"{REMOTE_PYTHON} -c {shlex.quote(launch_script)}",
        command_runner=command_runner,
    )
    _require_success(launch, "detached remote execution launch")
    try:
        started = json.loads(launch.stdout)
    except (TypeError, json.JSONDecodeError) as error:
        raise ValueError("detached launch receipt is invalid") from error
    pid = started.get("pid") if isinstance(started, dict) else None
    if (
        started.get("status") != "STARTED"
        or isinstance(pid, bool)
        or not isinstance(pid, int)
        or pid <= 0
    ):
        raise ValueError("detached launch receipt is invalid")
    deadline = monotonic() + timeout_seconds
    poll_script = "\n".join((
        "import json,os,pathlib",
        f"result_path=pathlib.Path({str(result_path)!r})",
        f"pid={pid}",
        "alive=True",
        "try:",
        " os.kill(pid,0)",
        "except ProcessLookupError:",
        " alive=False",
        "result=(json.loads(result_path.read_text())",
        " if result_path.is_file() else None)",
        "print(json.dumps({'schema_version':1,",
        " 'done':result is not None,'alive':alive,'result':result},",
        " sort_keys=True,separators=(',',':')))",
    ))
    while True:
        if monotonic() >= deadline:
            raise TimeoutError(
                "detached remote execution is still running"
            )
        poll = _run_remote(
            f"{REMOTE_PYTHON} -c {shlex.quote(poll_script)}",
            command_runner=command_runner,
        )
        _require_success(poll, "detached remote execution poll")
        try:
            status = json.loads(poll.stdout)
        except (TypeError, json.JSONDecodeError) as error:
            raise ValueError(
                "detached execution poll receipt is invalid"
            ) from error
        if not isinstance(status, dict):
            raise ValueError(
                "detached execution poll receipt is invalid"
            )
        if status.get("done") is True:
            result = status.get("result")
            if (
                not isinstance(result, dict)
                or result.get("status") not in {"PASS", "FAILED"}
            ):
                raise ValueError(
                    "remote execution result is invalid"
                )
            return result
        if status.get("alive") is not True:
            raise RuntimeError(
                "detached remote execution exited without a result"
            )
        sleep(5)


def remote_runtime_environment(primary_run: str) -> dict[str, str]:
    if (
        not isinstance(primary_run, str)
        or not primary_run.startswith(
            APPROVED_ROOT + "/staged-benchmark/runs/"
        )
    ):
        raise ValueError("primary run path is invalid")
    return {
        "TMPDIR": primary_run + "/tmp",
        "TEMP": primary_run + "/tmp",
        "TMP": primary_run + "/tmp",
        "PYTHONPYCACHEPREFIX": primary_run + "/pycache",
        "HF_HOME": primary_run + "/hf-home",
        "TORCH_EXTENSIONS_DIR": primary_run + "/torch-extensions",
    }


def build_deterministic_tar(
    source_root: Path,
    *,
    prefix: str,
) -> bytes:
    root = Path(source_root)
    if not root.is_dir():
        raise ValueError("archive source root is invalid")
    prefix_path = PurePosixPath(prefix)
    if (
        not isinstance(prefix, str)
        or not prefix
        or prefix_path.is_absolute()
        or any(part in ("", ".", "..") for part in prefix_path.parts)
        or prefix_path.as_posix() != prefix
    ):
        raise ValueError("archive prefix is invalid")
    files = []
    for path in root.rglob("*"):
        if path.is_symlink():
            raise ValueError("archive source contains a symlink")
        if path.is_file():
            files.append(path)
        elif path.exists() and not path.is_dir():
            raise ValueError("archive source contains a non-regular path")
    files.sort(key=lambda path: path.relative_to(root).as_posix())
    output = io.BytesIO()
    with tarfile.open(
        fileobj=output,
        mode="w",
        format=tarfile.USTAR_FORMAT,
    ) as archive:
        for path in files:
            relative = path.relative_to(root).as_posix()
            payload = path.read_bytes()
            info = tarfile.TarInfo(
                name=f"{prefix_path.as_posix()}/{relative}"
            )
            info.size = len(payload)
            info.mode = 0o644
            info.mtime = 0
            info.uid = 0
            info.gid = 0
            info.uname = ""
            info.gname = ""
            archive.addfile(info, io.BytesIO(payload))
    return output.getvalue()


def iter_download_ranges(
    size: int,
    *,
    chunk_size: int = DOWNLOAD_CHUNK_BYTES,
):
    if (
        isinstance(size, bool)
        or not isinstance(size, int)
        or size < 0
        or isinstance(chunk_size, bool)
        or not isinstance(chunk_size, int)
        or chunk_size <= 0
    ):
        raise ValueError("download sizes must be valid")
    for offset in range(0, size, chunk_size):
        yield offset, min(chunk_size, size - offset)


def validate_download_member(
    name: str,
    *,
    is_file: bool,
    is_link: bool,
) -> str:
    if (
        not isinstance(name, str)
        or not name
        or "\\" in name
        or not isinstance(is_file, bool)
        or not isinstance(is_link, bool)
        or not is_file
        or is_link
    ):
        raise ValueError("download member is unsafe")
    path = PurePosixPath(name)
    if (
        path.is_absolute()
        or any(part in ("", ".", "..") for part in path.parts)
        or path.as_posix() != name
    ):
        raise ValueError("download member path is invalid")
    return name


def validate_download_inventory(
    inventory: object,
    *,
    expected_root: str,
    chunk_size: int = DOWNLOAD_CHUNK_BYTES,
) -> list[dict]:
    if (
        not isinstance(inventory, dict)
        or inventory.get("schema_version") != 1
        or inventory.get("root") != expected_root
        or not isinstance(expected_root, str)
        or not expected_root.startswith(APPROVED_ROOT + "/")
    ):
        raise ValueError("download inventory identity is invalid")
    files = inventory.get("files")
    if not isinstance(files, list):
        raise ValueError("download file inventory is invalid")
    normalized = []
    prior_path = None
    for record in files:
        if not isinstance(record, dict) or set(record) != {
            "path",
            "size_bytes",
            "sha256",
            "chunks",
        }:
            raise ValueError("download file record is invalid")
        relative = validate_download_member(
            record["path"],
            is_file=True,
            is_link=False,
        )
        if prior_path is not None and relative <= prior_path:
            raise ValueError("download inventory is not canonical")
        prior_path = relative
        size = record["size_bytes"]
        digest = record["sha256"]
        chunks = record["chunks"]
        if (
            isinstance(size, bool)
            or not isinstance(size, int)
            or size < 0
            or not isinstance(digest, str)
            or re.fullmatch(r"[0-9a-f]{64}", digest) is None
            or not isinstance(chunks, list)
        ):
            raise ValueError("download file record is invalid")
        expected_ranges = list(
            iter_download_ranges(size, chunk_size=chunk_size)
        )
        if len(chunks) != len(expected_ranges):
            raise ValueError("download chunk inventory is incomplete")
        normalized_chunks = []
        for chunk, (offset, length) in zip(chunks, expected_ranges):
            if (
                not isinstance(chunk, dict)
                or set(chunk) != {"offset", "length", "sha256"}
                or chunk.get("offset") != offset
                or chunk.get("length") != length
                or not isinstance(chunk.get("sha256"), str)
                or re.fullmatch(
                    r"[0-9a-f]{64}",
                    chunk["sha256"],
                )
                is None
            ):
                raise ValueError("download chunk inventory is invalid")
            normalized_chunks.append(dict(chunk))
        normalized.append({
            "path": relative,
            "size_bytes": size,
            "sha256": digest,
            "chunks": normalized_chunks,
        })
    return normalized


def download_chunk(
    remote_path: str,
    *,
    offset: int,
    length: int,
    expected_sha256: str,
    command_runner=subprocess.run,
) -> bytes:
    if (
        not isinstance(remote_path, str)
        or not remote_path.startswith(APPROVED_ROOT + "/")
        or isinstance(offset, bool)
        or not isinstance(offset, int)
        or offset < 0
        or isinstance(length, bool)
        or not isinstance(length, int)
        or length <= 0
        or re.fullmatch(r"[0-9a-f]{64}", expected_sha256) is None
    ):
        raise ValueError("download chunk request is invalid")
    script = (
        "import os,sys;"
        f"p={remote_path!r};o={offset};n={length};"
        "f=open(p,'rb');f.seek(o);b=f.read(n);"
        "sys.stdout.buffer.write(b);"
        "sys.exit(0 if len(b)==n else 3)"
    )
    result = _run_remote(
        "python3 -c " + shlex.quote(script),
        command_runner=command_runner,
        text=False,
    )
    _require_success(result, "remote artifact chunk download")
    payload = result.stdout
    if not isinstance(payload, bytes) or len(payload) != length:
        raise ValueError("download chunk size mismatch")
    if _sha256_bytes(payload) != expected_sha256:
        raise ValueError("download chunk sha256 mismatch")
    return payload


def fetch_remote_inventory(
    remote_root: str,
    *,
    command_runner=subprocess.run,
    chunk_size: int = DOWNLOAD_CHUNK_BYTES,
) -> list[dict]:
    if (
        not isinstance(remote_root, str)
        or not remote_root.startswith(APPROVED_ROOT + "/")
    ):
        raise ValueError("remote inventory root is invalid")
    script = "\n".join((
        "import hashlib,json,pathlib",
        f"root=pathlib.Path({remote_root!r})",
        f"chunk_size={chunk_size}",
        "if not root.is_dir():",
        " raise ValueError('remote artifact root is missing')",
        "files=[]",
        "for path in sorted(root.rglob('*')):",
        " if path.is_symlink():",
        "  raise ValueError('remote artifact contains a link')",
        " if path.is_dir():",
        "  continue",
        " if not path.is_file():",
        "  raise ValueError('remote artifact is non-regular')",
        " payload=path.read_bytes()",
        " chunks=[]",
        " for offset in range(0,len(payload),chunk_size):",
        "  chunk=payload[offset:offset+chunk_size]",
        "  chunks.append({'offset':offset,'length':len(chunk),",
        "   'sha256':hashlib.sha256(chunk).hexdigest()})",
        " files.append({'path':path.relative_to(root).as_posix(),",
        "  'size_bytes':len(payload),",
        "  'sha256':hashlib.sha256(payload).hexdigest(),",
        "  'chunks':chunks})",
        "print(json.dumps({'schema_version':1,'root':str(root),",
        " 'files':files},sort_keys=True,separators=(',',':')))",
    ))
    result = _run_remote(
        f"{REMOTE_PYTHON} -c {shlex.quote(script)}",
        command_runner=command_runner,
    )
    _require_success(result, "remote artifact inventory")
    try:
        inventory = json.loads(result.stdout)
    except (TypeError, json.JSONDecodeError) as error:
        raise ValueError("remote artifact inventory is invalid") from error
    return validate_download_inventory(
        inventory,
        expected_root=remote_root,
        chunk_size=chunk_size,
    )


def download_remote_tree(
    remote_root: str,
    destination: Path,
    *,
    command_runner=subprocess.run,
    chunk_size: int = DOWNLOAD_CHUNK_BYTES,
    retries: int = 3,
) -> list[dict]:
    target = Path(destination)
    if target.exists():
        raise ValueError("download destination already exists")
    partial = target.with_name(target.name + ".partial")
    if partial.is_symlink():
        raise ValueError("download partial destination is unsafe")
    if partial.exists():
        if not partial.is_dir():
            raise ValueError("download partial destination is invalid")
        shutil.rmtree(partial)
    if (
        isinstance(retries, bool)
        or not isinstance(retries, int)
        or retries <= 0
    ):
        raise ValueError("download retry policy is invalid")
    inventory = fetch_remote_inventory(
        remote_root,
        command_runner=command_runner,
        chunk_size=chunk_size,
    )
    partial.mkdir(parents=True)
    try:
        for record in inventory:
            path = partial / record["path"]
            path.parent.mkdir(parents=True, exist_ok=True)
            digest = hashlib.sha256()
            with path.open("xb") as handle:
                for chunk in record["chunks"]:
                    last_error = None
                    for _ in range(retries):
                        try:
                            payload = download_chunk(
                                remote_root + "/" + record["path"],
                                offset=chunk["offset"],
                                length=chunk["length"],
                                expected_sha256=chunk["sha256"],
                                command_runner=command_runner,
                            )
                            break
                        except (RuntimeError, ValueError) as error:
                            last_error = error
                    else:
                        raise RuntimeError(
                            f"artifact chunk download failed: "
                            f"{record['path']}"
                        ) from last_error
                    handle.write(payload)
                    digest.update(payload)
                handle.flush()
                os.fsync(handle.fileno())
            if (
                path.stat().st_size != record["size_bytes"]
                or digest.hexdigest() != record["sha256"]
            ):
                raise ValueError(
                    f"downloaded artifact mismatch: {record['path']}"
                )
        partial.rename(target)
    except BaseException:
        if partial.is_dir() and not partial.is_symlink():
            shutil.rmtree(partial)
        raise
    return inventory


def verify_downloaded_tree(
    destination: Path,
    inventory: list[dict],
) -> list[dict]:
    target = Path(destination)
    if target.is_symlink() or not target.is_dir():
        raise ValueError("download destination is invalid")
    if not isinstance(inventory, list):
        raise ValueError("download inventory is invalid")
    expected = {}
    for record in inventory:
        if not isinstance(record, dict):
            raise ValueError("download file record is invalid")
        relative = validate_download_member(
            record.get("path"),
            is_file=True,
            is_link=False,
        )
        if relative in expected:
            raise ValueError("download inventory contains duplicates")
        expected[relative] = record
    actual = {}
    for path in target.rglob("*"):
        if path.is_symlink():
            raise ValueError("download destination contains a link")
        if path.is_dir():
            continue
        if not path.is_file():
            raise ValueError("download destination is non-regular")
        relative = path.relative_to(target).as_posix()
        actual[relative] = path
    if set(actual) != set(expected):
        raise ValueError("downloaded artifact path set mismatch")
    for relative, record in expected.items():
        path = actual[relative]
        if (
            path.stat().st_size != record.get("size_bytes")
            or _sha256_bytes(path.read_bytes()) != record.get("sha256")
        ):
            raise ValueError(
                f"downloaded artifact mismatch: {relative}"
            )
    return inventory


def verify_local_download(
    *,
    primary_dir: Path,
    remote_controller_dir: Path,
    local_controller_dir: Path,
) -> dict:
    from tools import staged_inference_benchmark_verify as verifier

    local_controller = Path(local_controller_dir)
    verifier.verify_run(Path(primary_dir), local_controller)
    remote_receipt = _load_json_file(
        Path(remote_controller_dir) / "verification_receipt.json"
    )
    local_receipt = _load_json_file(
        local_controller / "verification_receipt.json"
    )
    comparison = compare_verification_receipts(
        remote_receipt,
        local_receipt,
    )
    _write_json_exclusive(
        local_controller / "receipt_comparison.json",
        comparison,
    )
    return comparison


def remote_path_exists(
    remote_path: str,
    *,
    command_runner=subprocess.run,
) -> bool:
    if (
        not isinstance(remote_path, str)
        or not remote_path.startswith(APPROVED_ROOT + "/")
    ):
        raise ValueError("remote path is invalid")
    script = (
        "import os;"
        f"print('1' if os.path.isdir({remote_path!r}) else '0')"
    )
    result = _run_remote(
        "python3 -c " + shlex.quote(script),
        command_runner=command_runner,
    )
    _require_success(result, "remote path probe")
    value = result.stdout.strip()
    if value not in {"0", "1"}:
        raise ValueError("remote path probe receipt is invalid")
    return value == "1"


def download_available_artifacts(
    *,
    run_tag: str,
    output_dir: Path,
    command_runner=subprocess.run,
) -> dict:
    paths = remote_paths(run_tag)
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    downloaded = {}
    for name in ("primary", "controller"):
        remote_root = paths[name]
        if not remote_path_exists(
            remote_root,
            command_runner=command_runner,
        ):
            downloaded[name] = None
            continue
        local_root = root / name
        if local_root.exists():
            inventory = fetch_remote_inventory(
                remote_root,
                command_runner=command_runner,
            )
            verify_downloaded_tree(local_root, inventory)
        else:
            inventory = download_remote_tree(
                remote_root,
                local_root,
                command_runner=command_runner,
            )
        downloaded[name] = {
            "path": str(local_root),
            "file_count": len(inventory),
        }
    return downloaded


def _error_record(error: BaseException) -> dict:
    return {
        "error_type": type(error).__name__,
        "error": str(error),
    }


def execute_and_collect(
    *,
    run_tag: str,
    output_dir: Path,
    payload: bytes,
    spec: dict,
    uploader=upload_execution_payload,
    launcher=launch_remote_execution,
    downloader=download_available_artifacts,
    kerberos_validator=validate_kerberos,
    local_verifier=verify_local_download,
) -> dict:
    kerberos_revalidation = None
    upload = None
    execution = None
    downloaded = None
    comparison = None
    orchestration_error = None
    download_error = None
    verification_error = None
    try:
        kerberos_revalidation = kerberos_validator()
        if (
            not isinstance(kerberos_revalidation, dict)
            or kerberos_revalidation.get("status") != "PASS"
        ):
            raise ValueError("Kerberos revalidation receipt is invalid")
        upload = uploader(payload=payload, spec=spec)
        execution = launcher(spec=spec)
    except Exception as error:
        orchestration_error = _error_record(error)
    try:
        downloaded = downloader(
            run_tag=run_tag,
            output_dir=output_dir,
        )
    except Exception as error:
        download_error = _error_record(error)
    if (
        execution is not None
        and execution.get("status") == "PASS"
        and isinstance(downloaded, dict)
        and downloaded.get("primary") is not None
        and downloaded.get("controller") is not None
    ):
        try:
            comparison = local_verifier(
                primary_dir=Path(output_dir) / "primary",
                remote_controller_dir=(
                    Path(output_dir) / "controller"
                ),
                local_controller_dir=(
                    Path(output_dir) / "local-verification"
                ),
            )
        except Exception as error:
            verification_error = _error_record(error)
    passed = (
        orchestration_error is None
        and download_error is None
        and verification_error is None
        and execution is not None
        and execution.get("status") == "PASS"
        and comparison is not None
        and comparison.get("status") == "PASS"
    )
    return {
        "schema_version": 1,
        "status": "PASS" if passed else "FAILED",
        "kerberos_revalidation": kerberos_revalidation,
        "upload": upload,
        "execution": execution,
        "downloaded": downloaded,
        "local_verification": comparison,
        "orchestration_error": orchestration_error,
        "download_error": download_error,
        "verification_error": verification_error,
    }


def compare_verification_receipts(
    remote_receipt: object,
    local_receipt: object,
) -> dict:
    if (
        not isinstance(remote_receipt, dict)
        or not isinstance(local_receipt, dict)
        or remote_receipt.get("status") != "PASS"
        or local_receipt.get("status") != "PASS"
    ):
        raise ValueError("verifier receipt is incomplete")
    required = {
        "status",
        "run_manifest_sha256",
        "primary_summary_sha256",
        "controller_summary_sha256",
        "classification",
    }
    if (
        set(remote_receipt) != required
        or set(local_receipt) != required
    ):
        raise ValueError("verifier receipt shape is invalid")
    for field in (
        "run_manifest_sha256",
        "primary_summary_sha256",
        "controller_summary_sha256",
    ):
        if re.fullmatch(
            r"[0-9a-f]{64}",
            str(remote_receipt.get(field, "")),
        ) is None:
            raise ValueError("verifier receipt hash is invalid")
    if remote_receipt != local_receipt:
        raise ValueError("remote and local verifier receipts disagree")
    return {
        "schema_version": 1,
        "status": "PASS",
        "receipt_sha256": _sha256_bytes(
            _canonical_json(remote_receipt).encode("utf-8")
        ),
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Safely run staged inference benchmarks remotely",
    )
    parser.add_argument(
        "command",
        choices=("preflight", "execute", "download-only", "verify-local"),
    )
    parser.add_argument("--gate", required=True, choices=("prefix", "chunked"))
    parser.add_argument(
        "--model-tier",
        required=True,
        choices=("qwen3-0.6b", "qwen3-8b"),
    )
    parser.add_argument("--run-tag", required=True)
    parser.add_argument("--promotion-prefix-run", type=Path)
    parser.add_argument("--promotion-chunked-run", type=Path)
    parser.add_argument("--capacity-receipt", type=Path)
    parser.add_argument("--local-run-dir", type=Path)
    args = parser.parse_args(argv)
    try:
        validate_run_tag(args.run_tag)
    except ValueError as error:
        parser.error(str(error))
    if args.model_tier == "qwen3-8b":
        if args.command == "preflight" and (
            args.promotion_prefix_run is None
            or args.promotion_chunked_run is None
            or args.capacity_receipt is None
        ):
            parser.error(
                "qwen3-8b preflight requires both Stage-1 "
                "promotion runs and a capacity receipt"
            )
        if args.command == "execute" and (
            args.promotion_prefix_run is None
            or args.promotion_chunked_run is None
        ):
            parser.error(
                "qwen3-8b execute requires both Stage-1 "
                "promotion runs"
            )
    return args


def _load_json_file(path: Path) -> dict:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid JSON file: {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"JSON file must contain an object: {path}")
    return value


def _local_controller_dir(args) -> Path:
    if args.local_run_dir is not None:
        return Path(args.local_run_dir)
    return LOCAL_ARTIFACT_ROOT / f"{args.run_tag}-controller"


def _load_stage1_promotion_bundle(path: Path) -> tuple[dict, dict]:
    root = Path(path)
    if (root / "primary").is_dir() and (root / "controller").is_dir():
        primary = root / "primary"
        controller = root / "controller"
        local_controller = root / "local-verification"
    elif (root / "summary.json").is_file():
        primary = root
        controller = root.parent / "controller"
        local_controller = root.parent / "local-verification"
    else:
        raise ValueError(f"invalid Stage-1 promotion bundle: {root}")
    summary = _load_json_file(primary / "summary.json")
    remote_receipt = _load_json_file(
        controller / "verification_receipt.json"
    )
    if not local_controller.is_dir():
        raise ValueError(
            "Stage-1 local verification bundle is missing"
        )
    local_receipt = _load_json_file(
        local_controller / "verification_receipt.json"
    )
    expected_comparison = compare_verification_receipts(
        remote_receipt,
        local_receipt,
    )
    if _load_json_file(
        local_controller / "receipt_comparison.json"
    ) != expected_comparison:
        raise ValueError(
            "Stage-1 local verification comparison is invalid"
        )
    if (
        not (primary / "run_manifest.json").is_file()
        or remote_receipt.get("run_manifest_sha256")
        != hashlib.sha256(
            (primary / "run_manifest.json").read_bytes()
        ).hexdigest()
    ):
        raise ValueError("Stage-1 promotion manifest is invalid")
    if (
        remote_receipt.get("classification")
        != summary.get("classification")
        or remote_receipt.get("primary_summary_sha256")
        != hashlib.sha256(
            (primary / "summary.json").read_bytes()
        ).hexdigest()
        or remote_receipt.get("controller_summary_sha256")
        != hashlib.sha256(
            (controller / "summary.json").read_bytes()
        ).hexdigest()
    ):
        raise ValueError("Stage-1 promotion receipt is invalid")
    canonical = _sha256_bytes(
        _canonical_json(summary).encode("utf-8")
    )
    return summary, {
        "status": "PASS",
        "primary_summary_sha256": canonical,
        "controller_summary_sha256": canonical,
    }


def _load_promotion(args) -> dict | None:
    if args.model_tier != "qwen3-8b":
        return None
    prefix_summary, prefix_receipt = (
        _load_stage1_promotion_bundle(args.promotion_prefix_run)
    )
    chunked_summary, chunked_receipt = (
        _load_stage1_promotion_bundle(args.promotion_chunked_run)
    )
    from tools import staged_inference_benchmark_contract as contract

    selection = contract.select_stage2_winner(
        prefix_summary,
        chunked_summary,
    )
    return {
        "winner": selection["winner"],
        "prefix_summary": prefix_summary,
        "chunked_summary": chunked_summary,
        "prefix_verification_receipt": prefix_receipt,
        "chunked_verification_receipt": chunked_receipt,
    }


def main(argv=None) -> int:
    args = parse_args(argv)
    capacity = (
        None
        if args.capacity_receipt is None
        else _load_json_file(args.capacity_receipt)
    )
    if args.command == "preflight":
        _load_promotion(args)
        receipt = run_preflight(
            repo_root=REPO_ROOT,
            gate_name=args.gate,
            model_tier=args.model_tier,
            run_tag=args.run_tag,
            output_dir=_local_controller_dir(args),
            capacity_receipt=capacity,
        )
        print(_canonical_json(receipt))
        return 0
    output_dir = _local_controller_dir(args)
    if args.command == "download-only":
        downloaded = download_available_artifacts(
            run_tag=args.run_tag,
            output_dir=output_dir,
        )
        path = output_dir / "download_result.json"
        _write_json_exclusive(path, downloaded)
        print(_canonical_json(downloaded))
        return 0
    if args.command == "verify-local":
        comparison = verify_local_download(
            primary_dir=output_dir / "primary",
            remote_controller_dir=output_dir / "controller",
            local_controller_dir=output_dir / "local-verification",
        )
        print(_canonical_json(comparison))
        return 0
    if args.command == "execute":
        preflight_path = output_dir / "preflight.json"
        if not preflight_path.is_file():
            raise ValueError(
                "execute requires a prior local preflight receipt"
            )
        promotion = _load_promotion(args)
        with tempfile.TemporaryDirectory(
            prefix="tinyllmforge-staged-payload-",
        ) as temporary:
            payload, spec = prepare_execution_payload(
                repo_root=REPO_ROOT,
                preflight_path=preflight_path,
                build_root=Path(temporary) / "build",
                promotion=promotion,
            )
            final = execute_and_collect(
                run_tag=args.run_tag,
                output_dir=output_dir,
                payload=payload,
                spec=spec,
            )
        _write_json_exclusive(output_dir / "final_result.json", final)
        print(_canonical_json(final))
        return 0 if final["status"] == "PASS" else 1
    raise AssertionError("unreachable command")


if __name__ == "__main__":
    raise SystemExit(main())
