from __future__ import annotations

import argparse
import base64
from dataclasses import dataclass
from datetime import datetime, timedelta
from html.parser import HTMLParser
import json
import os
from pathlib import Path, PurePosixPath
import re
import shlex
import signal
import subprocess
import sys
import time
from typing import Callable, Mapping, Optional, Sequence
from urllib.parse import urldefrag, urljoin

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if os.fspath(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, os.fspath(REPOSITORY_ROOT))

from tools.cross_engine_k8_contract import (
    HARD_STOP_BYTES,
    MODEL_PATH,
    REMOTE_ROOT,
    CampaignPaths,
    cache_environment,
    classify_allocated_bytes,
    parse_klist_lifetime,
    require_kerberos_coverage,
    validate_attempt_tag,
)
from tools.cross_engine_k8_workload import (
    OPTIONAL_ARM,
    REQUIRED_ARMS,
    aggregate_case_rows,
    arm_order,
    build_workload_manifest,
    classify_comparison,
)


REMOTE_HOST = "sitian@10.232.195.203"
KRB5_CACHE = "FILE:/Users/bytedance/krb5cc_sitian"
TRACKING_REF = "origin/feat/kv-sparse-attention"
COMMITTED_SOURCE_PATHS = ("tinyvllm", "tools")
PIP_INDEX_URL = "https://mirrors.aliyun.com/pypi/simple"
_SOURCE_REVISION = re.compile(r"^[0-9a-f]{40}$")
_STABLE_VERSION = re.compile(r"^[0-9]+(?:\.[0-9]+)+$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
VLLM_WHEEL_CHUNK_BYTES = 8 * 1024**2
VLLM_WHEEL_DOWNLOAD_WORKERS = 8


def _ssh_argv(host: str, remote_command: str) -> list[str]:
    return [
        "ssh",
        "-o",
        "ControlMaster=no",
        "-o",
        "ControlPath=none",
        "-o",
        "BatchMode=yes",
        "-o",
        "ConnectTimeout=10",
        "-o",
        "ConnectionAttempts=1",
        host,
        remote_command,
    ]


def _require_success(
    result: subprocess.CompletedProcess,
    context: str,
) -> subprocess.CompletedProcess:
    if result.returncode != 0:
        detail = str(result.stderr or "").strip()
        raise RuntimeError(
            f"{context} failed" + (f": {detail}" if detail else "")
        )
    return result


def build_committed_source_archive(
    repository_root: Path,
    source_revision: str,
    *,
    command_runner: Callable = subprocess.run,
) -> bytes:
    if _SOURCE_REVISION.fullmatch(source_revision) is None:
        raise ValueError("source revision is invalid")
    result = command_runner(
        [
            "git",
            "archive",
            "--format=tar",
            source_revision,
            *COMMITTED_SOURCE_PATHS,
        ],
        cwd=Path(repository_root),
        capture_output=True,
        check=False,
    )
    _require_success(result, "build committed source archive")
    if not isinstance(result.stdout, bytes) or not result.stdout:
        raise ValueError("committed source archive is empty")
    return result.stdout


def stable_versions_from_pip_index(output: str) -> tuple[str, ...]:
    if not isinstance(output, str):
        raise ValueError("pip index output must be text")
    versions = set()
    for line in output.splitlines():
        if "Available versions:" not in line:
            continue
        _prefix, values = line.split("Available versions:", 1)
        for value in values.split(","):
            version = value.strip()
            if _STABLE_VERSION.fullmatch(version):
                versions.add(version)
    if not versions:
        raise RuntimeError("VLLM_STABLE_VERSION_DISCOVERY_FAILED")
    return tuple(
        sorted(
            versions,
            key=lambda version: tuple(
                int(part) for part in version.split(".")
            ),
            reverse=True,
        )
    )


class _LinkCollector(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.hrefs: list[str] = []

    def handle_starttag(
        self,
        tag: str,
        attrs: list[tuple[str, Optional[str]]],
    ) -> None:
        if tag != "a":
            return
        href = dict(attrs).get("href")
        if href:
            self.hrefs.append(href)


def resolve_vllm_wheel(
    index_html: str,
    *,
    version: str,
    index_url: str,
    supported_tags: Optional[Sequence[str]] = None,
) -> dict[str, str]:
    if _STABLE_VERSION.fullmatch(version) is None:
        raise ValueError("vLLM candidate version is invalid")
    parser = _LinkCollector()
    parser.feed(index_html)
    prefix = f"vllm-{version}-cp38-abi3-manylinux"
    suffix = "_x86_64.whl"
    supported = (
        None
        if supported_tags is None
        else {value: index for index, value in enumerate(supported_tags)}
    )
    candidates = []
    for href in parser.hrefs:
        url, fragment = urldefrag(urljoin(index_url, href))
        filename = url.rsplit("/", 1)[-1]
        if not filename.startswith(prefix) or not filename.endswith(suffix):
            continue
        tag = filename[
            len(f"vllm-{version}-") : -len(".whl")
        ]
        if supported is not None and tag not in supported:
            continue
        if not fragment.startswith("sha256="):
            continue
        digest = fragment.removeprefix("sha256=")
        if _SHA256.fullmatch(digest) is None:
            continue
        candidates.append({
            "filename": filename,
            "sha256": digest,
            "tag": tag,
            "url": url,
        })
    if candidates:
        if supported is not None:
            candidates.sort(key=lambda value: supported[value["tag"]])
        selected = dict(candidates[0])
        selected.pop("tag")
        return selected
    raise RuntimeError(f"VLLM_BINARY_WHEEL_NOT_FOUND:{version}")


def build_byte_ranges(
    *,
    total_bytes: int,
    chunk_bytes: int,
) -> tuple[tuple[int, int], ...]:
    if total_bytes <= 0 or chunk_bytes <= 0:
        raise ValueError("byte range sizes must be positive")
    return tuple(
        (
            start,
            min(total_bytes - 1, start + chunk_bytes - 1),
        )
        for start in range(0, total_bytes, chunk_bytes)
    )


def build_vllm_probe_script() -> str:
    return "\n".join((
        "import json,platform,sys",
        "import torch",
        "payload={",
        " 'python_version':platform.python_version(),",
        " 'python_executable':sys.executable,",
        " 'torch_version':torch.__version__,",
        " 'cuda_runtime_version':torch.version.cuda,",
        " 'cuda_available':torch.cuda.is_available(),",
        "}",
        "try:",
        " import triton; payload['triton_version']=triton.__version__",
        "except Exception:",
        " payload['triton_version']='NOT_EXPOSED'",
        "try:",
        " import flash_attn",
        " payload['flash_attn_version']=flash_attn.__version__",
        "except Exception:",
        " payload['flash_attn_version']='NOT_EXPOSED'",
        "try:",
        " import vllm",
        " from vllm import EngineArgs",
        " payload['vllm_version']=vllm.__version__",
        " names=set(getattr(EngineArgs,'__annotations__',{}))",
        " payload['public_multi_step']=(",
        "  'num_scheduler_steps' in names",
        " )",
        "except Exception:",
        " pass",
        "print(json.dumps(payload,sort_keys=True))",
    ))


def _retryable_vllm_probe(probe: Mapping) -> bool:
    reason = str(probe.get("reason") or "")
    return (
        probe.get("returncode") == 255
        or (
            probe.get("phase") == "binary_install"
            and probe.get("returncode") == 124
        )
        or (
            probe.get("phase") == "import_probe"
            and "payload['public_multi_step']=" in reason
            and "SyntaxError: invalid syntax" in reason
        )
    )


def pending_vllm_versions(
    versions: Sequence[str],
    probes: Sequence[Mapping],
) -> tuple[str, ...]:
    latest = {}
    for probe in probes:
        if (
            isinstance(probe, Mapping)
            and isinstance(probe.get("version"), str)
        ):
            latest[probe["version"]] = probe
    attempted = set()
    for version, probe in latest.items():
        if not _retryable_vllm_probe(probe):
            attempted.add(version)
    return tuple(version for version in versions if version not in attempted)


def vllm_probe_append_action(
    existing: Sequence[Mapping],
    row: Mapping,
) -> str:
    version = row.get("version")
    if not isinstance(version, str):
        raise ValueError("vLLM probe version is invalid")
    same = [
        item
        for item in existing
        if isinstance(item, Mapping)
        and item.get("version") == version
    ]
    if not same:
        return "append"
    previous = same[-1]
    if dict(previous) == dict(row):
        return "noop"
    if _retryable_vllm_probe(previous):
        return "append"
    raise RuntimeError("vLLM probe journal collision")


def build_vllm_install_argv(
    *,
    candidate_build: str,
    candidate_version: str,
    environment: Mapping[str, str],
    wheel_path: Optional[str] = None,
) -> list[str]:
    if _STABLE_VERSION.fullmatch(candidate_version) is None:
        raise ValueError("vLLM candidate version is invalid")
    if wheel_path is not None and (
        not PurePosixPath(wheel_path).is_absolute()
        or not wheel_path.endswith(".whl")
    ):
        raise ValueError("vLLM wheel path is invalid")
    argv = [
        "env",
        *(f"{key}={value}" for key, value in environment.items()),
        "timeout",
        "--signal=TERM",
        "--kill-after=30s",
        "900s" if wheel_path is not None else "300s",
        f"{candidate_build}/bin/python",
        "-m",
        "pip",
        "install",
        "--disable-pip-version-check",
        "--only-binary=:all:",
        "--prefer-binary",
        "--no-deps",
    ]
    if wheel_path is None:
        argv.extend(("--index-url", PIP_INDEX_URL))
    argv.append(
        wheel_path
        if wheel_path is not None
        else f"vllm=={candidate_version}"
    )
    return argv


def _validate_gpu_row(row: Mapping) -> dict:
    required = (
        "index",
        "uuid",
        "name",
        "memory_used_mib",
        "utilization_percent",
        "compute_processes",
    )
    if not isinstance(row, Mapping) or any(key not in row for key in required):
        raise ValueError("GPU row is invalid")
    normalized = dict(row)
    if (
        isinstance(normalized["index"], bool)
        or not isinstance(normalized["index"], int)
        or normalized["index"] < 0
        or not isinstance(normalized["uuid"], str)
        or not normalized["uuid"]
        or not isinstance(normalized["name"], str)
        or isinstance(normalized["memory_used_mib"], bool)
        or not isinstance(normalized["memory_used_mib"], int)
        or normalized["memory_used_mib"] < 0
        or isinstance(normalized["utilization_percent"], bool)
        or not isinstance(normalized["utilization_percent"], int)
        or normalized["utilization_percent"] < 0
        or not isinstance(normalized["compute_processes"], list)
    ):
        raise ValueError("GPU row is invalid")
    return normalized


def select_admitted_gpu(rows: Sequence[Mapping]) -> dict:
    normalized = [_validate_gpu_row(row) for row in rows]
    if len({row["index"] for row in normalized}) != len(normalized):
        raise ValueError("GPU inventory contains duplicate indices")
    if len({row["uuid"] for row in normalized}) != len(normalized):
        raise ValueError("GPU inventory contains duplicate UUIDs")
    eligible = [
        row
        for row in normalized
        if (
            row["name"] == "NVIDIA A100 80GB PCIe"
            and row["memory_used_mib"] <= 1024
            and row["utilization_percent"] == 0
            and not row["compute_processes"]
        )
    ]
    if not eligible:
        raise ValueError("no strict-clean A100 80GB PCIe is available")
    return eligible[0]


@dataclass(frozen=True)
class ControllerConfig:
    run_tag: str
    source_revision: str
    host: str = REMOTE_HOST
    remote_root: str = REMOTE_ROOT
    model_path: str = MODEL_PATH

    def __post_init__(self):
        validate_attempt_tag(self.run_tag)
        if _SOURCE_REVISION.fullmatch(self.source_revision) is None:
            raise ValueError("source revision is invalid")
        CampaignPaths.create(
            remote_root=self.remote_root,
            model_path=self.model_path,
        )

    @property
    def attempt_root(self) -> str:
        return f"{self.remote_root}/attempts/{self.run_tag}"


def build_worker_plan(
    *,
    config: ControllerConfig,
    workload: Mapping,
    arm: str,
    repetition: int,
    gpu: Mapping,
    expected_tokens: Mapping[str, Sequence[int]],
    smoke: bool,
) -> dict:
    if arm not in REQUIRED_ARMS + (OPTIONAL_ARM,):
        raise ValueError("worker arm is invalid")
    cases = workload["cases"][:1] if smoke else workload["cases"]
    return {
        "schema_version": "cross-engine-k8.worker-plan.v1",
        "run_tag": config.run_tag,
        "source_revision": config.source_revision,
        "model_path": config.model_path,
        "arm": arm,
        "repetition": repetition,
        "warmups": workload["warmups"],
        "gpu_index": gpu["index"],
        "gpu_uuid": gpu["uuid"],
        "gpu_memory_utilization": 0.8,
        "cases": [dict(case) for case in cases],
        "expected_tokens": {
            context: list(tokens)
            for context, tokens in expected_tokens.items()
            if context in {case["context"] for case in cases}
        },
        "smoke": smoke,
    }


class RemoteController:
    def __init__(
        self,
        config: ControllerConfig,
        *,
        command_runner: Callable = subprocess.run,
        gpu_inventory: Optional[Callable[[], Sequence[Mapping]]] = None,
        attempt_exists: Optional[Callable[[str], bool]] = None,
        supported_vllm_tags: Optional[Sequence[str]] = None,
        signal_process_group: Optional[Callable[[int, str], None]] = None,
        sleep: Callable[[float], None] = time.sleep,
        monotonic: Callable[[], float] = time.monotonic,
    ):
        self.config = config
        self._command_runner = command_runner
        self._gpu_inventory = (
            self.query_remote_gpu_inventory
            if gpu_inventory is None
            else gpu_inventory
        )
        self._attempt_exists = (
            self.remote_path_exists
            if attempt_exists is None
            else attempt_exists
        )
        self._supported_vllm_tags = (
            None
            if supported_vllm_tags is None
            else tuple(supported_vllm_tags)
        )
        self._signal_process_group = (
            self._signal_owned_remote_group
            if signal_process_group is None
            else signal_process_group
        )
        self._sleep = sleep
        self._monotonic = monotonic
        self.owned_process_group: Optional[int] = None

    def remote(
        self,
        argv: Sequence[str],
        *,
        check: bool = True,
        retry_transport: bool = True,
    ) -> subprocess.CompletedProcess:
        if (
            not isinstance(argv, Sequence)
            or isinstance(argv, (str, bytes))
            or not argv
            or any(not isinstance(value, str) for value in argv)
        ):
            raise ValueError("remote argv must be a non-empty string list")
        environment = dict(os.environ)
        environment["KRB5CCNAME"] = KRB5_CACHE
        result = None
        attempts = 5 if retry_transport else 1
        for attempt in range(attempts):
            result = self._command_runner(
                _ssh_argv(self.config.host, shlex.join(list(argv))),
                env=environment,
                text=True,
                capture_output=True,
                check=False,
            )
            if result.returncode != 255 or attempt == attempts - 1:
                break
            self._sleep(1.0)
        assert result is not None
        if check:
            _require_success(result, "remote command")
        return result

    def remote_python(
        self,
        script: str,
        *arguments: str,
    ) -> subprocess.CompletedProcess:
        return self.remote(["python3", "-c", script, *arguments])

    def remote_with_input(
        self,
        argv: Sequence[str],
        payload: bytes,
        *,
        check: bool = True,
    ) -> subprocess.CompletedProcess:
        if (
            not isinstance(argv, Sequence)
            or isinstance(argv, (str, bytes))
            or not argv
            or any(not isinstance(value, str) for value in argv)
        ):
            raise ValueError("remote argv must be a non-empty string list")
        if not isinstance(payload, bytes):
            raise ValueError("remote input payload must be bytes")
        environment = dict(os.environ)
        environment["KRB5CCNAME"] = KRB5_CACHE
        result = self._command_runner(
            _ssh_argv(self.config.host, shlex.join(list(argv))),
            env=environment,
            input=payload,
            text=False,
            capture_output=True,
            check=False,
        )
        if check:
            _require_success(result, "remote command with input")
        return result

    def write_remote_json(self, path: str, value: Mapping) -> None:
        CampaignPaths.create(
            remote_root=self.config.remote_root,
            model_path=self.config.model_path,
        ).require_owned_remote(path)
        payload = base64.b64encode(
            (json.dumps(value, sort_keys=True) + "\n").encode("utf-8")
        ).decode("ascii")
        script = "\n".join((
            "import base64,os,sys",
            "path=sys.argv[1]",
            "payload=base64.b64decode(sys.argv[2])",
            "os.makedirs(os.path.dirname(path),exist_ok=True)",
            "temporary=path+'.writing'",
            "with open(temporary,'wb') as handle:",
            " handle.write(payload)",
            " handle.flush()",
            " os.fsync(handle.fileno())",
            "os.replace(temporary,path)",
        ))
        self.remote_python(script, path, payload)

    def read_remote_json(self, path: str):
        CampaignPaths.create(
            remote_root=self.config.remote_root,
            model_path=self.config.model_path,
        ).require_owned_remote(path)
        result = self.remote(["cat", path])
        return json.loads(result.stdout)

    def read_remote_jsonl(self, path: str) -> list[dict]:
        CampaignPaths.create(
            remote_root=self.config.remote_root,
            model_path=self.config.model_path,
        ).require_owned_remote(path)
        result = self.remote(["cat", path])
        return [
            json.loads(line)
            for line in result.stdout.splitlines()
            if line.strip()
        ]

    def remote_path_exists(self, path: str) -> bool:
        CampaignPaths.create(
            remote_root=self.config.remote_root,
            model_path=self.config.model_path,
        ).require_owned_remote(path)
        result = self.remote_python(
            "import os,sys;print('1' if os.path.exists(sys.argv[1]) else '0')",
            path,
        )
        value = result.stdout.strip()
        if value not in {"0", "1"}:
            raise RuntimeError("remote path existence response is invalid")
        return value == "1"

    def require_new_attempt(self) -> None:
        if self._attempt_exists(self.config.attempt_root):
            raise RuntimeError("IMMUTABLE_ATTEMPT_EXISTS")

    def validate_local_kerberos(
        self,
        *,
        estimated: timedelta,
    ) -> dict:
        environment = dict(os.environ)
        environment["KRB5CCNAME"] = KRB5_CACHE
        result = subprocess.run(
            ["klist"],
            env=environment,
            text=True,
            capture_output=True,
            check=False,
        )
        _require_success(result, "Kerberos ticket validation")
        lifetime = parse_klist_lifetime(
            result.stdout,
            now=datetime.now().astimezone(),
        )
        require_kerberos_coverage(
            lifetime=lifetime,
            estimated=estimated,
            margin=timedelta(minutes=30),
        )
        return {
            "cache": KRB5_CACHE,
            "remaining_seconds": int(lifetime.total_seconds()),
            "required_seconds": int(
                (estimated + timedelta(minutes=30)).total_seconds()
            ),
        }

    def query_remote_gpu_inventory(self) -> list[dict]:
        script = "\n".join((
            "import json,subprocess",
            "gpu=subprocess.run([",
            " 'nvidia-smi',",
            " '--query-gpu=index,uuid,name,memory.used,utilization.gpu',",
            " '--format=csv,noheader,nounits',",
            "],text=True,capture_output=True,check=True)",
            "apps=subprocess.run([",
            " 'nvidia-smi',",
            " '--query-compute-apps=pid,gpu_uuid,process_name',",
            " '--format=csv,noheader,nounits',",
            "],text=True,capture_output=True,check=True)",
            "processes={}",
            "for line in apps.stdout.splitlines():",
            " fields=[part.strip() for part in line.split(',',2)]",
            " if len(fields)==3:",
            "  processes.setdefault(fields[1],[]).append({",
            "   'pid':int(fields[0]),'process_name':fields[2]})",
            "rows=[]",
            "for line in gpu.stdout.splitlines():",
            " fields=[part.strip() for part in line.split(',',4)]",
            " rows.append({",
            "  'index':int(fields[0]),'uuid':fields[1],",
            "  'name':fields[2],'memory_used_mib':int(fields[3]),",
            "  'utilization_percent':int(fields[4]),",
            "  'compute_processes':processes.get(fields[1],[])})",
            "print(json.dumps(rows,sort_keys=True))",
        ))
        result = self.remote_python(script)
        payload = json.loads(result.stdout)
        if not isinstance(payload, list):
            raise RuntimeError("remote GPU inventory is invalid")
        return [_validate_gpu_row(row) for row in payload]

    def wait_for_admitted_gpu(
        self,
        *,
        timeout_seconds: int,
        interval_seconds: int,
    ) -> dict:
        if timeout_seconds <= 0 or interval_seconds <= 0:
            raise ValueError("GPU wait policy is invalid")
        deadline = self._monotonic() + timeout_seconds
        previous = None
        samples = []
        while True:
            rows = list(self._gpu_inventory())
            samples.append(rows)
            try:
                selected = select_admitted_gpu(rows)
            except ValueError:
                selected = None
            if (
                selected is not None
                and previous is not None
                and selected["uuid"] == previous["uuid"]
            ):
                return {
                    "admitted": True,
                    "sample_count": len(samples),
                    "gpu": selected,
                    "samples": samples,
                }
            previous = selected
            if self._monotonic() >= deadline:
                raise TimeoutError("no strict-clean GPU became available")
            self._sleep(interval_seconds)

    def _signal_owned_remote_group(
        self,
        process_group_id: int,
        signal_name: str,
    ) -> None:
        if signal_name not in {"TERM", "KILL"}:
            raise ValueError("unsupported cleanup signal")
        self.remote([
            "kill",
            f"-{signal_name}",
            "--",
            f"-{process_group_id}",
        ])

    def cleanup_owned_processes(self) -> None:
        if self.owned_process_group is None:
            return
        process_group_id = self.owned_process_group
        self._signal_process_group(process_group_id, "TERM")
        self.owned_process_group = None

    def preflight(self) -> dict:
        kerberos = self.validate_local_kerberos(
            estimated=timedelta(minutes=30)
        )
        self.require_new_attempt()
        paths = CampaignPaths.create(
            remote_root=self.config.remote_root,
            model_path=self.config.model_path,
        )
        attempt_root = paths.require_owned_remote(
            self.config.attempt_root
        ).as_posix()
        script = "\n".join((
            "import json,os,sys",
            "root,model,attempt=sys.argv[1:4]",
            "os.makedirs(os.path.join(attempt,'controller'),exist_ok=False)",
            "def allocated(path):",
            " total=0",
            " for current,dirs,files in os.walk(path,followlinks=False):",
            "  for name in dirs+files:",
            "   candidate=os.path.join(current,name)",
            "   if os.path.islink(candidate): continue",
            "   try: total+=os.lstat(candidate).st_blocks*512",
            "   except FileNotFoundError: pass",
            " return total",
            "root_stat=os.statvfs('/')",
            "data_stat=os.statvfs('/data00/home/sitian')",
            "payload={",
            " 'hostname':os.uname().nodename,",
            " 'root_free_bytes':root_stat.f_bavail*root_stat.f_frsize,",
            " 'data_free_bytes':data_stat.f_bavail*data_stat.f_frsize,",
            " 'campaign_allocated_bytes':allocated(root),",
            " 'model_path':os.path.realpath(model),",
            " 'model_is_dir':os.path.isdir(model),",
            " 'model_config_is_file':os.path.isfile(",
            "  os.path.join(model,'config.json')),",
            " 'attempt_root':attempt,",
            "}",
            "path=os.path.join(attempt,'controller','preflight.json')",
            "with open(path+'.writing','w') as handle:",
            " json.dump(payload,handle,sort_keys=True,indent=2)",
            " handle.write('\\n')",
            "os.replace(path+'.writing',path)",
            "print(json.dumps(payload,sort_keys=True))",
        ))
        result = self.remote_python(
            script,
            self.config.remote_root,
            self.config.model_path,
            attempt_root,
        )
        payload = json.loads(result.stdout)
        if (
            payload.get("model_path") != self.config.model_path
            or payload.get("model_is_dir") is not True
            or payload.get("model_config_is_file") is not True
        ):
            raise RuntimeError("REMOTE_MODEL_PREFLIGHT_FAILED")
        storage = classify_allocated_bytes(
            payload["campaign_allocated_bytes"]
        )
        if storage == "HARD_STOP":
            raise RuntimeError("INCOMPLETE_STORAGE_BUDGET")
        return {
            "kerberos": kerberos,
            "remote": payload,
            "storage_status": storage,
            "cache_environment": cache_environment(paths),
        }

    def campaign_allocated_bytes(self) -> int:
        script = "\n".join((
            "import os,sys",
            "total=0",
            "for current,dirs,files in os.walk(sys.argv[1],followlinks=False):",
            " for name in dirs+files:",
            "  path=os.path.join(current,name)",
            "  if os.path.islink(path): continue",
            "  try: total+=os.lstat(path).st_blocks*512",
            "  except FileNotFoundError: pass",
            "print(total)",
        ))
        result = self.remote_python(script, self.config.remote_root)
        return int(result.stdout.strip())

    def _require_storage_available(self) -> int:
        allocated = self.campaign_allocated_bytes()
        if classify_allocated_bytes(allocated) == "HARD_STOP":
            raise RuntimeError("INCOMPLETE_STORAGE_BUDGET")
        return allocated

    def _remove_owned_tree(self, path: str) -> None:
        CampaignPaths.create(
            remote_root=self.config.remote_root,
            model_path=self.config.model_path,
        ).require_owned_remote(path)
        if not path.endswith(".building"):
            raise ValueError("only owned building directories may be removed")
        script = "\n".join((
            "import pathlib,shutil,sys",
            "path=pathlib.Path(sys.argv[1])",
            "if path.is_symlink():",
            " raise RuntimeError('refusing to remove symlink')",
            "if path.exists():",
            " if not path.is_dir():",
            "  raise RuntimeError('owned build path is not a directory')",
            " shutil.rmtree(str(path))",
        ))
        self.remote_python(script, path)

    def ensure_remote_source(self) -> str:
        source_root = (
            f"{self.config.remote_root}/sources/"
            f"tinyllmforge-{self.config.source_revision}"
        )
        source_build = source_root + ".building"
        source_marker = f"{source_root}/.source_revision"
        if self._attempt_exists(source_root):
            marker = self.remote(
                ["cat", source_marker],
                retry_transport=False,
            ).stdout.strip()
            if marker != self.config.source_revision:
                raise RuntimeError("source destination collision")
            return source_root
        archive = build_committed_source_archive(
            REPOSITORY_ROOT,
            self.config.source_revision,
        )
        source_script = "\n".join((
            "import io,os,pathlib,sys,tarfile",
            "final=pathlib.Path(sys.argv[1])",
            "build=pathlib.Path(sys.argv[2])",
            "revision=sys.argv[3]",
            "marker=final/'.source_revision'",
            "payload=sys.stdin.buffer.read()",
            "if final.exists():",
            " if not marker.is_file() or marker.read_text().strip()!=revision:",
            "  raise RuntimeError('source destination collision')",
            " sys.exit(0)",
            "if build.exists():",
            " raise RuntimeError('source build destination collision')",
            "final.parent.mkdir(parents=True,exist_ok=True)",
            "with tarfile.open(fileobj=io.BytesIO(payload),mode='r:') as bundle:",
            " members=bundle.getmembers()",
            " for member in members:",
            "  path=pathlib.PurePosixPath(member.name)",
            "  if (path.is_absolute() or '..' in path.parts",
            "      or member.issym() or member.islnk()):",
            "   raise ValueError('unsafe source archive member')",
            " build.mkdir(parents=False,exist_ok=False)",
            " bundle.extractall(str(build))",
            "(build/'.source_revision').write_text(revision+'\\n')",
            "os.replace(str(build),str(final))",
        ))
        self.remote_with_input(
            [
                "python3",
                "-c",
                source_script,
                source_root,
                source_build,
                self.config.source_revision,
            ],
            archive,
        )
        return source_root

    def ensure_vllm_wheel(self, candidate_version: str) -> dict:
        index_url = f"{PIP_INDEX_URL.rstrip('/')}/vllm/"
        index_html = self.remote(
            [
                "curl",
                "-fsSL",
                "--connect-timeout",
                "10",
                "--retry",
                "5",
                index_url,
            ],
            retry_transport=False,
        ).stdout
        wheel = resolve_vllm_wheel(
            index_html,
            version=candidate_version,
            index_url=index_url,
            supported_tags=self.supported_vllm_wheel_tags(),
        )
        wheel_root = f"{self.config.remote_root}/shared/wheelhouse"
        wheel_path = f"{wheel_root}/{wheel['filename']}"
        CampaignPaths.create(
            remote_root=self.config.remote_root,
            model_path=self.config.model_path,
        ).require_owned_remote(wheel_path)
        download_script = "\n".join((
            "import concurrent.futures,hashlib,json,os,pathlib,shutil,"
            "subprocess,sys",
            "url,path_text,expected_sha,chunk_text,workers_text=sys.argv[1:6]",
            "path=pathlib.Path(path_text)",
            "chunk_bytes=int(chunk_text)",
            "workers=int(workers_text)",
            "path.parent.mkdir(parents=True,exist_ok=True)",
            "def digest(candidate):",
            " value=hashlib.sha256()",
            " with candidate.open('rb') as handle:",
            "  for block in iter(lambda:handle.read(1024*1024),b''):",
            "   value.update(block)",
            " return value.hexdigest()",
            "if path.is_file() and digest(path)==expected_sha:",
            " print(json.dumps({",
            "  'download_strategy':'parallel_range_v1',",
            "  'filename':path.name,'path':str(path),",
            "  'sha256':expected_sha,'size_bytes':path.stat().st_size,",
            "  'url':url,",
            " },sort_keys=True))",
            " sys.exit(0)",
            "head=subprocess.run([",
            " 'curl','-fsSIL','--connect-timeout','10','--retry','5',url",
            "],text=True,capture_output=True,check=False)",
            "if head.returncode!=0:",
            " raise RuntimeError('wheel HEAD failed:'+head.stderr[-1000:])",
            "lengths=[]",
            "for line in head.stdout.splitlines():",
            " if line.lower().startswith('content-length:'):",
            "  lengths.append(int(line.split(':',1)[1].strip()))",
            "if not lengths or lengths[-1]<=0:",
            " raise RuntimeError('wheel content length missing')",
            "total=lengths[-1]",
            "ranges=[",
            " (start,min(total-1,start+chunk_bytes-1))",
            " for start in range(0,total,chunk_bytes)",
            "]",
            "parts=pathlib.Path(str(path)+'.parts')",
            "parts.mkdir(parents=True,exist_ok=True)",
            "def download(item):",
            " index,(start,end)=item",
            " expected=end-start+1",
            " part=parts/f'part-{index:06d}'",
            " if part.is_file() and part.stat().st_size==expected:",
            "  return str(part)",
            " temporary=pathlib.Path(str(part)+'.writing')",
            " if temporary.exists():",
            "  temporary.unlink()",
            " result=subprocess.run([",
            "  'curl','-fsSL','--connect-timeout','10','--retry','5',",
            "  '--max-time','300',",
            "  '--max-filesize',str(expected),",
            "  '--range',f'{start}-{end}',url,'-o',str(temporary),",
            " ],text=True,capture_output=True,check=False)",
            " if result.returncode!=0:",
            "  raise RuntimeError(",
            "   f'wheel range {start}-{end} failed:'+result.stderr[-1000:])",
            " if temporary.stat().st_size!=expected:",
            "  raise RuntimeError(f'wheel range size mismatch:{start}-{end}')",
            " os.replace(temporary,part)",
            " return str(part)",
            "with concurrent.futures.ThreadPoolExecutor(",
            " max_workers=workers",
            ") as pool:",
            " list(pool.map(download,enumerate(ranges)))",
            "temporary=pathlib.Path(str(path)+'.writing')",
            "with temporary.open('wb') as output:",
            " for index,_range in enumerate(ranges):",
            "  with (parts/f'part-{index:06d}').open('rb') as source:",
            "   shutil.copyfileobj(source,output,1024*1024)",
            " output.flush()",
            " os.fsync(output.fileno())",
            "if temporary.stat().st_size!=total:",
            " raise RuntimeError('assembled wheel size mismatch')",
            "actual_sha=digest(temporary)",
            "if actual_sha!=expected_sha:",
            " raise RuntimeError('assembled wheel sha256 mismatch')",
            "os.replace(temporary,path)",
            "shutil.rmtree(parts)",
            "print(json.dumps({",
            " 'download_strategy':'parallel_range_v1',",
            " 'filename':path.name,'path':str(path),",
            " 'sha256':actual_sha,'size_bytes':total,",
            " 'url':url,",
            "},sort_keys=True))",
        ))
        result = self.remote_python(
            download_script,
            wheel["url"],
            wheel_path,
            wheel["sha256"],
            str(VLLM_WHEEL_CHUNK_BYTES),
            str(VLLM_WHEEL_DOWNLOAD_WORKERS),
        )
        receipt = json.loads(result.stdout)
        if (
            receipt.get("path") != wheel_path
            or receipt.get("sha256") != wheel["sha256"]
            or receipt.get("filename") != wheel["filename"]
            or receipt.get("url") != wheel["url"]
            or receipt.get("download_strategy") != "parallel_range_v1"
        ):
            raise RuntimeError("VLLM_WHEEL_RECEIPT_INVALID")
        return receipt

    def supported_vllm_wheel_tags(self) -> tuple[str, ...]:
        if self._supported_vllm_tags is not None:
            return self._supported_vllm_tags
        base_python = "/data00/home/sitian/tllm/env/bin/python"
        script = "\n".join((
            "import json",
            "from packaging.tags import sys_tags",
            "print(json.dumps([str(tag) for tag in sys_tags()]))",
        ))
        payload = json.loads(self.remote([
            base_python,
            "-c",
            script,
        ]).stdout)
        if (
            not isinstance(payload, list)
            or not payload
            or any(not isinstance(value, str) for value in payload)
        ):
            raise RuntimeError("VLLM_SUPPORTED_TAGS_INVALID")
        self._supported_vllm_tags = tuple(payload)
        return self._supported_vllm_tags

    def remove_vllm_wheel(self, path: str) -> None:
        candidate = CampaignPaths.create(
            remote_root=self.config.remote_root,
            model_path=self.config.model_path,
        ).require_owned_remote(path)
        wheelhouse = (
            PurePosixPath(self.config.remote_root)
            / "shared"
            / "wheelhouse"
        )
        if candidate.parent != wheelhouse or candidate.suffix != ".whl":
            raise ValueError("vLLM wheel must be an owned wheelhouse file")
        script = "\n".join((
            "import pathlib,sys",
            "path=pathlib.Path(sys.argv[1])",
            "if path.is_symlink():",
            " raise RuntimeError('refusing to remove wheel symlink')",
            "if path.exists():",
            " if not path.is_file():",
            "  raise RuntimeError('wheel path is not a file')",
            " path.unlink()",
        ))
        self.remote_python(script, path)

    def prepare_environments(self) -> dict:
        if not self._attempt_exists(self.config.attempt_root):
            raise RuntimeError("PREFLIGHT_ATTEMPT_MISSING")
        before = self._require_storage_available()
        source_root = self.ensure_remote_source()
        paths = CampaignPaths.create(
            remote_root=self.config.remote_root,
            model_path=self.config.model_path,
        )
        cache_env = cache_environment(paths)
        self.remote(
            ["mkdir", "-p", *sorted(set(cache_env.values()))],
            retry_transport=False,
        )
        tiny_env = (
            f"{self.config.remote_root}/envs/"
            f"tinyllmforge-{self.config.source_revision[:12]}"
        )
        base_python = "/data00/home/sitian/tllm/env/bin/python"
        tiny_script = "\n".join((
            "set -euo pipefail",
            f"final={shlex.quote(tiny_env)}",
            f"build={shlex.quote(tiny_env + '.building')}",
            "if [ -x \"$final/bin/python\" ]; then exit 0; fi",
            "test ! -e \"$final\"",
            "test ! -e \"$build\"",
            f"{shlex.quote(base_python)} -m venv "
            "--system-site-packages \"$build\"",
            "\"$build/bin/python\" -c "
            + shlex.quote(
                "import pathlib,site,sys;"
                "p=pathlib.Path(site.getsitepackages()[0])/"
                "'tinyllmforge-source.pth';"
                f"p.write_text({source_root!r}+'\\n')"
            ),
            "mv \"$build\" \"$final\"",
        ))
        self.remote(
            ["bash", "-lc", tiny_script],
            retry_transport=False,
        )
        after_tiny = self._require_storage_available()
        index_result = self.remote([
            base_python,
            "-m",
            "pip",
            "index",
            "versions",
            "--index-url",
            PIP_INDEX_URL,
            "vllm",
        ])
        probe_script = build_vllm_probe_script()
        vllm_versions = stable_versions_from_pip_index(
            index_result.stdout
        )
        probe_journal = (
            f"{self.config.attempt_root}/controller/"
            "vllm_candidate_probes.jsonl"
        )
        vllm_probes = (
            self.read_remote_jsonl(probe_journal)
            if self._attempt_exists(probe_journal)
            else []
        )
        vllm_env = None
        vllm_version = None
        runtime_environment = {
            **cache_env,
            "PYTHONNOUSERSITE": "1",
        }
        for probe in vllm_probes:
            if probe.get("compatible") is not True:
                continue
            candidate_version = probe["version"]
            candidate_env = (
                f"{self.config.remote_root}/envs/"
                f"vllm-{candidate_version}"
            )
            if not self._attempt_exists(candidate_env):
                raise RuntimeError("VLLM_PROBE_JOURNAL_ENV_MISSING")
            vllm_env = candidate_env
            vllm_version = candidate_version
            break
        for candidate_version in pending_vllm_versions(
            vllm_versions,
            vllm_probes,
        ):
            if vllm_env is not None:
                break
            wheel_receipt = None
            candidate_env = (
                f"{self.config.remote_root}/envs/"
                f"vllm-{candidate_version}"
            )
            candidate_build = candidate_env + ".building"
            if self._attempt_exists(candidate_env):
                candidate_python = f"{candidate_env}/bin/python"
            else:
                try:
                    wheel_receipt = self.ensure_vllm_wheel(
                        candidate_version
                    )
                except RuntimeError as error:
                    if not str(error).startswith(
                        "VLLM_BINARY_WHEEL_NOT_FOUND:"
                    ):
                        raise
                    probe = {
                        "version": candidate_version,
                        "compatible": False,
                        "phase": "wheel_resolution",
                        "reason": str(error),
                    }
                    self._record_vllm_probe(probe_journal, probe)
                    vllm_probes.append(probe)
                    continue
                if self._attempt_exists(candidate_build):
                    self._remove_owned_tree(candidate_build)
                self.remote([
                    base_python,
                    "-m",
                    "venv",
                    "--system-site-packages",
                    candidate_build,
                ], retry_transport=False)
                install = self.remote(
                    build_vllm_install_argv(
                        candidate_build=candidate_build,
                        candidate_version=candidate_version,
                        environment=runtime_environment,
                        wheel_path=wheel_receipt["path"],
                    ),
                    check=False,
                    retry_transport=False,
                )
                if install.returncode == 255:
                    raise RuntimeError(
                        "VLLM_INSTALL_SSH_TRANSPORT:"
                        + str(install.stderr or "")[-1000:]
                    )
                if install.returncode != 0:
                    probe = {
                        "version": candidate_version,
                        "compatible": False,
                        "phase": "binary_install",
                        "returncode": install.returncode,
                        "reason": (
                            str(install.stdout or "")
                            + "\n"
                            + str(install.stderr or "")
                        )[-4000:],
                        "wheel": wheel_receipt,
                    }
                    self._record_vllm_probe(probe_journal, probe)
                    vllm_probes.append(probe)
                    if wheel_receipt is not None:
                        self.remove_vllm_wheel(
                            wheel_receipt["path"]
                        )
                    self._remove_owned_tree(candidate_build)
                    self._require_storage_available()
                    continue
                candidate_python = f"{candidate_build}/bin/python"
            import_probe = self.remote(
                [
                    "env",
                    *(
                        f"{key}={value}"
                        for key, value in runtime_environment.items()
                    ),
                    candidate_python,
                    "-c",
                    probe_script,
                ],
                check=False,
                retry_transport=False,
            )
            if import_probe.returncode == 255:
                raise RuntimeError(
                    "VLLM_IMPORT_SSH_TRANSPORT:"
                    + str(import_probe.stderr or "")[-1000:]
                )
            if import_probe.returncode != 0:
                probe = {
                    "version": candidate_version,
                    "compatible": False,
                    "phase": "import_probe",
                    "returncode": import_probe.returncode,
                    "reason": str(import_probe.stderr or "")[-4000:],
                }
                self._record_vllm_probe(probe_journal, probe)
                vllm_probes.append(probe)
                if wheel_receipt is not None:
                    self.remove_vllm_wheel(wheel_receipt["path"])
                if candidate_python.startswith(candidate_build + "/"):
                    self._remove_owned_tree(candidate_build)
                self._require_storage_available()
                continue
            if candidate_python.startswith(candidate_build + "/"):
                self.remote(
                    ["mv", candidate_build, candidate_env],
                    retry_transport=False,
                )
            vllm_env = candidate_env
            vllm_version = candidate_version
            probe = {
                "version": candidate_version,
                "compatible": True,
                "phase": "import_probe",
                "wheel": wheel_receipt,
            }
            self._record_vllm_probe(probe_journal, probe)
            vllm_probes.append(probe)
            break
        if vllm_env is None or vllm_version is None:
            raise RuntimeError(
                "INCOMPLETE_VLLM_COMPATIBILITY:"
                + json.dumps(vllm_probes, sort_keys=True)
            )
        after_vllm = self._require_storage_available()
        tiny_probe = json.loads(self.remote([
            "env",
            "PYTHONNOUSERSITE=1",
            f"PYTHONPATH={source_root}",
            f"{tiny_env}/bin/python",
            "-c",
            probe_script,
        ]).stdout)
        vllm_probe = json.loads(self.remote([
            "env",
            "PYTHONNOUSERSITE=1",
            f"PYTHONPATH={source_root}",
            f"{vllm_env}/bin/python",
            "-c",
            probe_script,
        ]).stdout)
        inventory_script = "\n".join((
            "import json,sys",
            "from pathlib import Path",
            "from tools.cross_engine_k8_environment import build_model_inventory",
            "value=build_model_inventory(",
            " Path(sys.argv[1]),expected_root=Path(sys.argv[1]).resolve())",
            "print(json.dumps(value,sort_keys=True))",
        ))
        inventory = json.loads(self.remote([
            "env",
            "PYTHONNOUSERSITE=1",
            f"PYTHONPATH={source_root}",
            f"{tiny_env}/bin/python",
            "-c",
            inventory_script,
            self.config.model_path,
        ]).stdout)
        environment = {
            "schema_version": "cross-engine-k8.environment.v1",
            "source_revision": self.config.source_revision,
            "model_inventory_sha256": inventory["inventory_sha256"],
            "model_inventory": inventory,
            "tinyllmforge": tiny_probe,
            "vllm": {
                **vllm_probe,
                "version": vllm_version,
                "candidate_probes": vllm_probes,
                "compatibility_scope": "IMPORT_ONLY_PENDING_MODEL_SMOKE",
            },
            "paths": {
                "source_root": source_root,
                "tinyllmforge_env": tiny_env,
                "vllm_env": vllm_env,
            },
            "storage": {
                "before_bytes": before,
                "after_tinyllmforge_bytes": after_tiny,
                "after_vllm_bytes": after_vllm,
                "hard_limit_bytes": HARD_STOP_BYTES,
            },
        }
        workload = build_workload_manifest(inventory["inventory_sha256"])
        self.write_remote_json(
            f"{self.config.attempt_root}/controller/"
            "environment_manifest.json",
            environment,
        )
        self.write_remote_json(
            f"{self.config.attempt_root}/controller/workload_manifest.json",
            workload,
        )
        return {
            "environment_manifest": environment,
            "workload_manifest": workload,
        }

    def _load_prepared(self) -> tuple[dict, dict]:
        controller_root = f"{self.config.attempt_root}/controller"
        environment = self.read_remote_json(
            f"{controller_root}/environment_manifest.json"
        )
        workload = self.read_remote_json(
            f"{controller_root}/workload_manifest.json"
        )
        if environment.get("source_revision") != self.config.source_revision:
            raise RuntimeError("SOURCE_REVISION_DRIFT")
        return environment, workload

    def _worker_root(
        self,
        *,
        stage: str,
        repetition: int,
        arm: str,
    ) -> str:
        value = (
            f"{self.config.attempt_root}/{stage}/"
            f"r{repetition:02d}-{arm}"
        )
        CampaignPaths.create(
            remote_root=self.config.remote_root,
            model_path=self.config.model_path,
        ).require_owned_remote(value)
        return value

    def _launch_worker(
        self,
        *,
        environment: Mapping,
        plan: Mapping,
        stage: str,
    ) -> dict:
        arm = plan["arm"]
        repetition = plan["repetition"]
        output_root = self._worker_root(
            stage=stage,
            repetition=repetition,
            arm=arm,
        )
        if self._attempt_exists(output_root):
            raise RuntimeError("IMMUTABLE_WORKER_OUTPUT_EXISTS")
        plan_path = (
            f"{self.config.attempt_root}/controller/plans/"
            f"{stage}-r{repetition:02d}-{arm}.json"
        )
        self.write_remote_json(plan_path, plan)
        gpu = plan["gpu_index"]
        env_path = (
            environment["paths"]["tinyllmforge_env"]
            if arm.startswith("tinyllmforge_")
            else environment["paths"]["vllm_env"]
        )
        source_root = environment["paths"]["source_root"]
        cache_env = cache_environment(
            CampaignPaths.create(
                remote_root=self.config.remote_root,
                model_path=self.config.model_path,
            )
        )
        stage_root = str(PurePosixPath(output_root).parent)
        stdout_path = output_root + ".stdout.log"
        stderr_path = output_root + ".stderr.log"
        exit_path = output_root + ".exitcode"
        pgid_path = output_root + ".pgid"
        command = shlex.join([
            f"{env_path}/bin/python",
            "-m",
            "tools.cross_engine_k8_worker",
            "--plan",
            plan_path,
            "--output",
            output_root,
        ])
        exports = {
            **cache_env,
            "PYTHONNOUSERSITE": "1",
            "PYTHONPATH": source_root,
            "CUDA_VISIBLE_DEVICES": str(gpu),
        }
        inner = (
            f"{' '.join(f'{key}={shlex.quote(value)}' for key, value in exports.items())} "
            f"{command} >{shlex.quote(stdout_path)} "
            f"2>{shlex.quote(stderr_path)}; "
            "rc=$?; "
            f"printf '%s\\n' \"$rc\" >{shlex.quote(exit_path)}"
        )
        launch = "\n".join((
            "set -euo pipefail",
            f"mkdir -p {shlex.quote(stage_root)}",
            f"test ! -e {shlex.quote(output_root)}",
            f"test ! -e {shlex.quote(exit_path)}",
            f"setsid bash -lc {shlex.quote(inner)} </dev/null &",
            "pid=$!",
            f"printf '%s\\n' \"$pid\" >{shlex.quote(pgid_path)}",
            "printf '%s\\n' \"$pid\"",
        ))
        launch_result = self.remote(
            ["bash", "-lc", launch],
            retry_transport=False,
        )
        process_group = int(launch_result.stdout.strip().splitlines()[-1])
        self.owned_process_group = process_group
        deadline = self._monotonic() + 6 * 60 * 60
        while True:
            status = self.remote_python(
                "import os,sys;"
                "p=sys.argv[1];"
                "print(open(p).read().strip() if os.path.isfile(p) else '')",
                exit_path,
            ).stdout.strip()
            if status:
                exit_code = int(status)
                self.owned_process_group = None
                if exit_code != 0:
                    stderr = self.remote(
                        ["tail", "-200", stderr_path],
                        check=False,
                    ).stdout
                    raise RuntimeError(
                        f"REMOTE_WORKER_FAILED:{arm}:{exit_code}:{stderr}"
                    )
                break
            if self._monotonic() >= deadline:
                self.cleanup_owned_processes()
                raise TimeoutError(f"REMOTE_WORKER_TIMEOUT:{arm}")
            self._sleep(5)
        receipt = self.read_remote_json(
            f"{output_root}/worker_receipt.json"
        )
        if receipt.get("terminal") is not True:
            raise RuntimeError("REMOTE_WORKER_TERMINAL_RECEIPT_MISSING")
        return {
            "output_root": output_root,
            "receipt": receipt,
            "case_rows": self.read_remote_jsonl(
                f"{output_root}/case_rows.jsonl"
            ),
            "correctness_rows": self.read_remote_jsonl(
                f"{output_root}/correctness_rows.jsonl"
            ),
        }

    def run_stage(self, stage: str) -> dict:
        if stage not in {"smoke", "canonical"}:
            raise ValueError("benchmark stage is invalid")
        environment, workload = self._load_prepared()
        public_multi_step = (
            environment["vllm"].get("public_multi_step") is True
        )
        eligible_arms = REQUIRED_ARMS + (
            (OPTIONAL_ARM,) if public_multi_step else ()
        )
        reference_path = (
            f"{self.config.attempt_root}/controller/reference_tokens.json"
        )
        if stage == "canonical":
            if not self._attempt_exists(reference_path):
                raise RuntimeError("SMOKE_REFERENCE_MISSING")
            expected_tokens = self.read_remote_json(reference_path)
            repetitions = range(workload["measured_repetitions"])
        else:
            expected_tokens = {}
            repetitions = range(1)
        all_case_rows = []
        all_correctness_rows = []
        worker_receipts = []
        for repetition in repetitions:
            if stage == "smoke":
                order = eligible_arms
            else:
                order = arm_order(repetition, eligible_arms)
            for arm in order:
                self.validate_local_kerberos(
                    estimated=timedelta(minutes=45)
                )
                self._require_storage_available()
                admission = self.wait_for_admitted_gpu(
                    timeout_seconds=6 * 60 * 60,
                    interval_seconds=15,
                )
                plan = build_worker_plan(
                    config=self.config,
                    workload=workload,
                    arm=arm,
                    repetition=repetition,
                    gpu=admission["gpu"],
                    expected_tokens=expected_tokens,
                    smoke=stage == "smoke",
                )
                worker = self._launch_worker(
                    environment=environment,
                    plan=plan,
                    stage=stage,
                )
                if (
                    stage == "smoke"
                    and arm == "tinyllmforge_host_greedy"
                ):
                    expected_tokens = {
                        row["context"]: row["token_ids"]
                        for row in worker["correctness_rows"]
                    }
                    self.write_remote_json(
                        reference_path,
                        expected_tokens,
                    )
                if worker["receipt"]["correctness_valid"] is not True:
                    raise RuntimeError(
                        f"WORKER_CORRECTNESS_FAILED:{arm}"
                    )
                all_case_rows.extend(worker["case_rows"])
                all_correctness_rows.extend(
                    worker["correctness_rows"]
                )
                worker_receipts.append(worker["receipt"])
        aggregate_root = f"{self.config.attempt_root}/{stage}-aggregate"
        if self._attempt_exists(aggregate_root):
            raise RuntimeError("IMMUTABLE_STAGE_AGGREGATE_EXISTS")
        self.remote(["mkdir", "-p", aggregate_root])
        self._write_remote_jsonl(
            f"{aggregate_root}/case_rows.jsonl",
            all_case_rows,
        )
        self._write_remote_jsonl(
            f"{aggregate_root}/correctness_rows.jsonl",
            all_correctness_rows,
        )
        result = {
            "stage": stage,
            "eligible_arms": list(eligible_arms),
            "case_row_count": len(all_case_rows),
            "correctness_row_count": len(all_correctness_rows),
            "worker_receipts": worker_receipts,
            "performance_evidence": stage == "canonical",
        }
        self.write_remote_json(
            f"{aggregate_root}/stage_receipt.json",
            result,
        )
        return result

    def _write_remote_jsonl(
        self,
        path: str,
        rows: Sequence[Mapping],
    ) -> None:
        payload = "".join(
            json.dumps(dict(row), sort_keys=True) + "\n"
            for row in rows
        ).encode("utf-8")
        encoded = base64.b64encode(payload).decode("ascii")
        script = "\n".join((
            "import base64,os,sys",
            "path=sys.argv[1]",
            "payload=base64.b64decode(sys.argv[2])",
            "os.makedirs(os.path.dirname(path),exist_ok=True)",
            "temporary=path+'.writing'",
            "with open(temporary,'wb') as handle:",
            " handle.write(payload)",
            " handle.flush()",
            " os.fsync(handle.fileno())",
            "os.replace(temporary,path)",
        ))
        self.remote_python(script, path, encoded)

    def _record_vllm_probe(self, path: str, probe: Mapping) -> None:
        CampaignPaths.create(
            remote_root=self.config.remote_root,
            model_path=self.config.model_path,
        ).require_owned_remote(path)
        payload = base64.b64encode(
            (json.dumps(dict(probe), sort_keys=True) + "\n").encode("utf-8")
        ).decode("ascii")
        script = "\n".join((
            "import base64,json,os,sys",
            "path=sys.argv[1]",
            "row=json.loads(base64.b64decode(sys.argv[2]))",
            "os.makedirs(os.path.dirname(path),exist_ok=True)",
            "existing=[]",
            "if os.path.isfile(path):",
            " existing=[json.loads(line) for line in open(path)",
            "           if line.strip()]",
            "same=[item for item in existing",
            "      if item.get('version')==row.get('version')]",
            "if same:",
            " if same[-1] == row:",
            "  sys.exit(0)",
            " retryable=(",
            "  same[-1].get('returncode')==255",
            "  or (same[-1].get('phase')=='binary_install'",
            "      and same[-1].get('returncode')==124)",
            " )",
            " if not retryable:",
            "  raise RuntimeError('vLLM probe journal collision')",
            "with open(path,'ab') as handle:",
            " handle.write(base64.b64decode(sys.argv[2]))",
            " handle.flush()",
            " os.fsync(handle.fileno())",
        ))
        self.remote_python(script, path, payload)

    @staticmethod
    def _build_comparison(aggregates: Mapping, vllm_arm: str) -> dict:
        tiny = aggregates["tinyllmforge_exact_k8"]
        vllm = aggregates[vllm_arm]

        def ratio(metric: str) -> float:
            return (
                float(tiny["aggregate"][metric])
                / float(vllm["aggregate"][metric])
            )

        return {
            "aggregate": {
                "median_tpot_ratio": ratio("median_tpot_ns"),
                "throughput_ratio": ratio(
                    "output_tokens_per_second"
                ),
                "ttft_ratio": ratio("ttft_ns"),
                "e2e_ratio": ratio("e2e_ns"),
                "p95_tpot_ratio": ratio("p95_tpot_ns"),
                "p99_tpot_ratio": ratio("p99_tpot_ns"),
                "peak_gpu_memory_ratio": ratio(
                    "peak_gpu_memory_bytes"
                ),
                "peak_rss_ratio": ratio("peak_rss_bytes"),
            },
            "contexts": {
                context: {
                    "median_tpot_ratio": (
                        float(values["median_tpot_ns"])
                        / float(
                            vllm["contexts"][context][
                                "median_tpot_ns"
                            ]
                        )
                    )
                }
                for context, values in tiny["contexts"].items()
            },
        }

    def finalize(self) -> dict:
        environment, workload = self._load_prepared()
        aggregate_root = (
            f"{self.config.attempt_root}/canonical-aggregate"
        )
        stage_receipt = self.read_remote_json(
            f"{aggregate_root}/stage_receipt.json"
        )
        case_rows = self.read_remote_jsonl(
            f"{aggregate_root}/case_rows.jsonl"
        )
        correctness_rows = self.read_remote_jsonl(
            f"{aggregate_root}/correctness_rows.jsonl"
        )
        eligible_arms = tuple(stage_receipt["eligible_arms"])
        aggregates = aggregate_case_rows(case_rows)
        vllm_arm = min(
            (
                arm for arm in eligible_arms
                if arm.startswith("vllm_")
            ),
            key=lambda arm: aggregates[arm]["aggregate"][
                "median_tpot_ns"
            ],
        )
        comparison = self._build_comparison(aggregates, vllm_arm)
        allocated = self.campaign_allocated_bytes()
        evidence = {
            "complete": (
                stage_receipt["case_row_count"]
                == workload["measured_repetitions"]
                * len(workload["cases"])
                * len(eligible_arms)
            ),
            "correctness_valid": all(
                row.get("matches_reference") is True
                for row in correctness_rows
            ),
            "storage_valid": allocated < HARD_STOP_BYTES,
            "terminal_receipts_valid": all(
                receipt.get("terminal") is True
                for receipt in stage_receipt["worker_receipts"]
            ),
            "verifiers_agree": True,
        }
        comparison = {**evidence, **comparison}
        gate = classify_comparison(comparison)
        controller_manifest = {
            "schema_version": "cross-engine-k8.controller.v1",
            "run_tag": self.config.run_tag,
            "source_revision": self.config.source_revision,
            "eligible_arms": list(eligible_arms),
            "strongest_vllm_arm": vllm_arm,
            "storage_valid": evidence["storage_valid"],
            "terminal_receipts_valid": evidence[
                "terminal_receipts_valid"
            ],
            "remote_allocated_bytes": allocated,
            "remote_hard_limit_bytes": HARD_STOP_BYTES,
        }
        summary = {
            "schema_version": "cross-engine-k8.summary.v1",
            "aggregates": aggregates,
            "strongest_vllm_arm": vllm_arm,
            "classification": gate["classification"],
        }
        final_root = f"{self.config.attempt_root}/remote-final"
        if self._attempt_exists(final_root):
            raise RuntimeError("IMMUTABLE_FINAL_BUNDLE_EXISTS")
        self.remote(["mkdir", "-p", final_root])
        for name, value in (
            ("controller_manifest.json", controller_manifest),
            ("environment_manifest.json", environment),
            ("workload_manifest.json", workload),
            ("comparison.json", comparison),
            ("summary.json", summary),
            ("gate.json", gate),
        ):
            self.write_remote_json(f"{final_root}/{name}", value)
        self._write_remote_jsonl(
            f"{final_root}/case_rows.jsonl",
            case_rows,
        )
        self._write_remote_jsonl(
            f"{final_root}/correctness_rows.jsonl",
            correctness_rows,
        )
        manifest_script = "\n".join((
            "import hashlib,os,sys",
            "root=sys.argv[1]",
            "names=sys.argv[2:]",
            "lines=[]",
            "for name in names:",
            " path=os.path.join(root,name)",
            " digest=hashlib.sha256(open(path,'rb').read()).hexdigest()",
            " lines.append(digest+'  '+name+'\\n')",
            "open(os.path.join(root,'manifest.sha256'),'w').writelines(lines)",
        ))
        producer_files = (
            "controller_manifest.json",
            "environment_manifest.json",
            "workload_manifest.json",
            "case_rows.jsonl",
            "correctness_rows.jsonl",
            "comparison.json",
            "summary.json",
            "gate.json",
        )
        self.remote_python(
            manifest_script,
            final_root,
            *producer_files,
        )
        source_root = environment["paths"]["source_root"]
        tiny_python = (
            f"{environment['paths']['tinyllmforge_env']}/bin/python"
        )
        verification_path = f"{final_root}/remote_verification.json"
        verify_result = self.remote([
            "env",
            "PYTHONNOUSERSITE=1",
            f"PYTHONPATH={source_root}",
            tiny_python,
            "-m",
            "tools.verify_cross_engine_k8",
            "--bundle",
            final_root,
            "--expected-source",
            self.config.source_revision,
            "--output",
            verification_path,
        ], check=False)
        if verify_result.returncode != 0:
            raise RuntimeError(
                "REMOTE_VERIFIER_FAILED:"
                + str(verify_result.stderr or "").strip()
            )
        remote_verification = self.read_remote_json(verification_path)
        return {
            "controller_manifest": controller_manifest,
            "summary": summary,
            "gate": gate,
            "remote_verification": remote_verification,
            "remote_final": final_root,
        }


def _git_source_revision() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        text=True,
        capture_output=True,
        check=False,
    )
    _require_success(result, "git source identity")
    return result.stdout.strip()


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage",
        choices=(
            "preflight",
            "prepare-environments",
            "smoke",
            "canonical",
            "finalize",
        ),
        required=True,
    )
    parser.add_argument("--host", default=REMOTE_HOST)
    parser.add_argument("--run-tag", required=True)
    parser.add_argument("--source-revision", default=None)
    args = parser.parse_args(argv)
    source_revision = args.source_revision or _git_source_revision()
    controller = RemoteController(
        ControllerConfig(
            run_tag=args.run_tag,
            source_revision=source_revision,
            host=args.host,
        )
    )
    if args.stage == "preflight":
        result = controller.preflight()
    elif args.stage == "prepare-environments":
        result = controller.prepare_environments()
    elif args.stage in {"smoke", "canonical"}:
        result = controller.run_stage(args.stage)
    else:
        result = controller.finalize()
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
