#!/usr/bin/env python3
"""Production adapters for the Qwen3.8 TP4 collective-reduction gate."""

from __future__ import annotations

import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import subprocess
import tarfile
import tempfile
import time

from tools.assemble_qwen38_tp4_collective_reduction import (
    MANIFEST_SCHEMA,
    PRODUCER_ARTIFACTS,
)
from tools.run_qwen38_tp4_communication_profile import (
    MAX_GPU_MEMORY_USED_MIB,
    MAX_GPU_UTILIZATION_PERCENT,
    build_ssh_argv,
    query_local_kerberos,
    query_remote_gpu_inventory,
    query_remote_gpu_topology,
    validate_selected_gpu_processes,
)
from tools.verify_qwen38_tp4_collective_reduction import verify_bundle


REMOTE_PYTHON = "/data00/home/sitian/tllm/env/bin/python"
SOURCE_ARCHIVE_PATHS = ("tinyvllm", "tools")
MIN_POSTPROCESS_COMMAND_TIMEOUT_S = 600


def _sha256(payload):
    return hashlib.sha256(payload).hexdigest()


def run_remote_bytes(
    *,
    ssh_target,
    remote_argv,
    input_bytes,
    timeout_s,
    retry_count,
    control_path=None,
    command_runner=subprocess.run,
    sleep=time.sleep,
):
    if (
        input_bytes is not None
        and not isinstance(input_bytes, bytes)
    ):
        raise ValueError("remote input must be bytes")
    if (
        not isinstance(timeout_s, int)
        or isinstance(timeout_s, bool)
        or timeout_s <= 0
        or not isinstance(retry_count, int)
        or isinstance(retry_count, bool)
        or retry_count <= 0
        or not callable(command_runner)
        or not callable(sleep)
    ):
        raise ValueError("remote execution policy is invalid")
    argv = build_ssh_argv(
        ssh_target=ssh_target,
        remote_argv=remote_argv,
        control_path=control_path,
    )
    result = None
    for attempt in range(retry_count):
        result = command_runner(
            argv,
            input=input_bytes,
            text=False,
            capture_output=True,
            check=False,
            timeout=timeout_s,
        )
        returncode = getattr(result, "returncode", None)
        if (
            not isinstance(returncode, int)
            or isinstance(returncode, bool)
        ):
            raise ValueError("remote command result is invalid")
        if returncode != 255 or attempt + 1 == retry_count:
            return result
        sleep(1)
    raise AssertionError("remote retry loop is unreachable")


def _require_remote_success(result, context):
    if getattr(result, "returncode", None) != 0:
        stderr = getattr(result, "stderr", b"")
        if isinstance(stderr, bytes):
            stderr = stderr.decode("utf-8", errors="replace")
        raise RuntimeError(stderr or f"{context} failed")
    return result


def _parse_remote_json(result, context):
    _require_remote_success(result, context)
    stdout = getattr(result, "stdout", None)
    if isinstance(stdout, bytes):
        stdout = stdout.decode("utf-8")
    if not isinstance(stdout, str):
        raise ValueError(f"{context} JSON is invalid")
    lines = [line for line in stdout.splitlines() if line.strip()]
    if not lines:
        raise ValueError(f"{context} JSON is invalid")
    try:
        payload = json.loads(lines[-1])
    except json.JSONDecodeError as error:
        raise ValueError(f"{context} JSON is invalid") from error
    if not isinstance(payload, dict):
        raise ValueError(f"{context} JSON is invalid")
    return payload


def build_stage_payload(
    *,
    plan,
    archive,
    source_identity,
    model_manifest,
    gpu_topology,
):
    if (
        not isinstance(archive, bytes)
        or not archive
        or not all(
            isinstance(value, dict)
            for value in (
                plan,
                source_identity,
                model_manifest,
                gpu_topology,
            )
        )
    ):
        raise ValueError("stage payload is invalid")
    metadata = _canonical_json_bytes({
        "plan": plan,
        "source_identity": source_identity,
        "model_manifest": model_manifest,
        "gpu_topology": gpu_topology,
    })
    return len(metadata).to_bytes(8, "big") + metadata + archive


def build_committed_source_archive(
    *,
    repo_root,
    source_revision,
    source_identity,
    command_runner=subprocess.run,
):
    if (
        not isinstance(source_revision, str)
        or len(source_revision) != 40
        or not isinstance(source_identity, dict)
        or source_identity.get("source_revision") != source_revision
        or not isinstance(source_identity.get("source_files"), dict)
        or not callable(command_runner)
    ):
        raise ValueError("source archive identity is invalid")
    result = command_runner(
        [
            "git",
            "archive",
            "--format=tar",
            source_revision,
            *SOURCE_ARCHIVE_PATHS,
        ],
        cwd=Path(repo_root),
        capture_output=True,
        check=False,
    )
    if getattr(result, "returncode", None) != 0:
        raise RuntimeError(
            getattr(result, "stderr", b"")
            or b"committed source archive failed"
        )
    archive = getattr(result, "stdout", None)
    if not isinstance(archive, bytes) or not archive:
        raise ValueError("committed source archive is empty")
    observed = {}
    names = set()
    try:
        with tarfile.open(fileobj=io.BytesIO(archive), mode="r:") as bundle:
            for member in bundle.getmembers():
                path = PurePosixPath(member.name)
                if (
                    path.is_absolute()
                    or ".." in path.parts
                    or member.name in names
                ):
                    raise ValueError("source archive inventory is invalid")
                names.add(member.name)
                if member.isdir():
                    continue
                if member.isreg():
                    handle = bundle.extractfile(member)
                    if handle is None:
                        raise ValueError(
                            "source archive inventory is invalid"
                        )
                    payload = handle.read()
                elif member.issym():
                    payload = member.linkname.encode("utf-8")
                else:
                    raise ValueError(
                        "source archive inventory is invalid"
                    )
                observed[member.name] = _sha256(payload)
    except tarfile.TarError as error:
        raise ValueError("source archive inventory is invalid") from error
    if observed != source_identity["source_files"]:
        raise ValueError("source archive inventory does not match identity")
    return archive


def _identity_tuple(identity):
    if not isinstance(identity, dict):
        raise ValueError("attempt source identity is invalid")
    values = (
        identity.get("attempt"),
        identity.get("source_revision"),
        identity.get("source_tree_sha256"),
    )
    if (
        not all(isinstance(value, str) and value for value in values)
        or len(values[1]) != 40
        or len(values[2]) != 64
    ):
        raise ValueError("attempt source identity is invalid")
    return values


def classify_attempt_state(state, *, source_identity):
    expected_identity = _identity_tuple(source_identity)
    required = {
        "attempt_exists",
        "source_identity",
        "launch",
        "supervisor_receipt",
        "live_exact_tag_pids",
    }
    if (
        not isinstance(state, dict)
        or set(state) != required
        or type(state["attempt_exists"]) is not bool
        or not isinstance(state["live_exact_tag_pids"], list)
        or any(
            not isinstance(pid, int) or isinstance(pid, bool) or pid <= 0
            for pid in state["live_exact_tag_pids"]
        )
    ):
        raise ValueError("attempt state is invalid")
    if not state["attempt_exists"]:
        if any(
            state[name] is not None
            for name in (
                "source_identity",
                "launch",
                "supervisor_receipt",
            )
        ) or state["live_exact_tag_pids"]:
            raise ValueError("absent attempt has residual state")
        return "CREATE"
    if _identity_tuple(state["source_identity"]) != expected_identity:
        raise ValueError("attempt source identity mismatch")
    launch = state["launch"]
    receipt = state["supervisor_receipt"]
    live = state["live_exact_tag_pids"]
    if launch is None:
        if receipt is not None or live:
            raise RuntimeError("attempt execution is orphaned")
        return "LAUNCH"
    if (
        not isinstance(launch, dict)
        or not isinstance(launch.get("pid"), int)
        or isinstance(launch["pid"], bool)
        or launch["pid"] <= 0
        or launch.get("source_revision") != expected_identity[1]
    ):
        raise ValueError("attempt launch identity is invalid")
    if receipt is None:
        if launch["pid"] not in live:
            raise RuntimeError("attempt execution is orphaned")
        return "MONITOR"
    if (
        not isinstance(receipt, dict)
        or receipt.get("source_revision") != expected_identity[1]
        or not isinstance(receipt.get("classification"), str)
    ):
        raise ValueError("attempt supervisor receipt is invalid")
    if live:
        raise RuntimeError("terminal attempt still has live processes")
    return "POSTPROCESS"


def build_supervisor_argv(plan, *, remote_python=REMOTE_PYTHON):
    if (
        not isinstance(plan, dict)
        or not isinstance(remote_python, str)
        or not remote_python.startswith("/")
    ):
        raise ValueError("supervisor launch plan is invalid")
    for key in ("attempt_root", "source_root", "model_root"):
        value = plan.get(key)
        if (
            not isinstance(value, str)
            or not PurePosixPath(value).is_absolute()
        ):
            raise ValueError("supervisor launch path is invalid")
    selected = plan.get("selected_gpus")
    if not isinstance(selected, list) or len(selected) != 4:
        raise ValueError("supervisor GPU plan is invalid")
    return [
        remote_python,
        (
            f"{plan['source_root']}/tools/"
            "qwen38_tp4_collective_reduction_supervisor.py"
        ),
        "--attempt",
        plan["attempt_tag"],
        "--source-revision",
        plan["source_revision"],
        "--attempt-root",
        plan["attempt_root"],
        "--source-root",
        plan["source_root"],
        "--model-root",
        plan["model_root"],
        "--python",
        remote_python,
        "--selected-gpus-json",
        json.dumps(selected, sort_keys=True, separators=(",", ":")),
        "--dist-port",
        "29671",
        "--poll-interval-s",
        "1",
        "--worker-timeout-s",
        "21600",
    ]


def stage_remote_attempt(
    *,
    plan,
    source_identity,
    model_manifest,
    gpu_topology,
    archive,
    ssh_target,
    timeout_s,
    retry_count,
    control_path=None,
    remote_runner=run_remote_bytes,
):
    payload = build_stage_payload(
        plan=plan,
        archive=archive,
        source_identity=source_identity,
        model_manifest=model_manifest,
        gpu_topology=gpu_topology,
    )
    script = "\n".join((
        "import hashlib,io,json,os,pathlib,sys,tarfile",
        "root=pathlib.Path(sys.argv[1])",
        "attempt=pathlib.Path(sys.argv[2])",
        "source_expected=pathlib.Path(sys.argv[3])",
        "tag=sys.argv[4]",
        "if attempt.parent != root/'attempts':",
        " raise ValueError('attempt path is invalid')",
        "if source_expected != attempt/'source':",
        " raise ValueError('source path is invalid')",
        "raw=sys.stdin.buffer.read()",
        "if len(raw) < 8:",
        " raise ValueError('stage payload is truncated')",
        "size=int.from_bytes(raw[:8],'big')",
        "metadata=json.loads(raw[8:8+size])",
        "archive=raw[8+size:]",
        (
            "if set(metadata)!={'plan','source_identity','model_manifest',"
            "'gpu_topology'}:"
        ),
        " raise ValueError('stage metadata is invalid')",
        "identity=metadata['source_identity']",
        (
            "if identity.get('attempt')!=tag or "
            "identity.get('source_revision')!=sys.argv[5]:"
        ),
        " raise ValueError('stage source identity mismatch')",
        "(root/'attempts').mkdir(parents=True,exist_ok=True)",
        "(root/'.staging').mkdir(parents=True,exist_ok=True)",
        "staging=root/'.staging'/(tag+'.partial')",
        "if attempt.exists() or staging.exists():",
        " raise FileExistsError('attempt or staging path exists')",
        "staging.mkdir(parents=False,exist_ok=False)",
        "source=staging/'source'",
        "source.mkdir()",
        "with tarfile.open(fileobj=io.BytesIO(archive),mode='r:') as bundle:",
        " members=bundle.getmembers()",
        " names=set()",
        " for member in members:",
        "  relative=pathlib.PurePosixPath(member.name)",
        (
            "  if (relative.is_absolute() or '..' in relative.parts "
            "or member.name in names):"
        ),
        "   raise ValueError('unsafe source archive member')",
        "  names.add(member.name)",
        "  target=source.joinpath(*relative.parts)",
        "  if member.isdir():",
        "   target.mkdir(parents=True,exist_ok=True)",
        "   continue",
        "  if not member.isreg():",
        "   raise ValueError('source archive member is not regular')",
        "  target.parent.mkdir(parents=True,exist_ok=True)",
        "  handle=bundle.extractfile(member)",
        "  if handle is None:",
        "   raise ValueError('source archive member is unreadable')",
        "  with target.open('xb') as output:",
        "   output.write(handle.read())",
        "  os.chmod(target,member.mode & 0o777)",
        "observed={}",
        "for path in sorted(source.rglob('*')):",
        " if path.is_file() and not path.is_symlink():",
        "  name=path.relative_to(source).as_posix()",
        "  observed[name]=hashlib.sha256(path.read_bytes()).hexdigest()",
        "if observed != identity.get('source_files'):",
        " raise ValueError('staged source inventory mismatch')",
        (
            "canonical=json.dumps(observed,sort_keys=True,"
            "separators=(',',':')).encode()"
        ),
        (
            "if hashlib.sha256(canonical).hexdigest()!="
            "identity.get('source_tree_sha256'):"
        ),
        " raise ValueError('staged source tree hash mismatch')",
        "controller=staging/'controller'",
        "controller.mkdir()",
        "(staging/'cases').mkdir()",
        "(staging/'final_bundle').mkdir()",
        "(staging/'runtime').mkdir()",
        "def write(name,value):",
        " path=controller/name",
        " partial=controller/('.'+name+'.partial')",
        (
            " partial.write_text(json.dumps(value,sort_keys=True,"
            "separators=(',',':'))+'\\n',encoding='utf-8')"
        ),
        " os.replace(partial,path)",
        "write('source_identity.json',identity)",
        "write('plan.json',metadata['plan'])",
        "write('model_manifest.json',metadata['model_manifest'])",
        "write('gpu_topology.json',metadata['gpu_topology'])",
        "receipt={",
        " 'classification':'STAGED',",
        " 'attempt':tag,",
        " 'source_revision':identity['source_revision'],",
        " 'source_tree_sha256':identity['source_tree_sha256'],",
        "}",
        "write('stage_receipt.json',receipt)",
        "os.rename(staging,attempt)",
        "print(json.dumps(receipt,sort_keys=True))",
    ))
    result = remote_runner(
        ssh_target=ssh_target,
        remote_argv=[
            REMOTE_PYTHON,
            "-c",
            script,
            plan["remote_root"],
            plan["attempt_root"],
            plan["source_root"],
            plan["attempt_tag"],
            plan["source_revision"],
        ],
        input_bytes=payload,
        timeout_s=timeout_s,
        retry_count=retry_count,
        control_path=control_path,
    )
    return _parse_remote_json(result, "remote source staging")


def query_remote_attempt_state(
    *,
    plan,
    ssh_target,
    timeout_s,
    retry_count,
    control_path=None,
    remote_runner=run_remote_bytes,
):
    script = "\n".join((
        "import json,os,pathlib,sys",
        "attempt=pathlib.Path(sys.argv[1])",
        "tag=sys.argv[2]",
        "def load(path):",
        " if not path.is_file() or path.is_symlink(): return None",
        " try:",
        "  return json.loads(path.read_text(encoding='utf-8'))",
        " except (OSError,json.JSONDecodeError):",
        "  return None",
        "identity=load(attempt/'controller'/'source_identity.json')",
        "if isinstance(identity,dict):",
        " identity={key:identity.get(key) for key in (",
        "  'attempt','source_revision','source_tree_sha256')}",
        "launch=load(attempt/'controller'/'supervisor_launch.json')",
        "receipt=load(attempt/'controller'/'supervisor_receipt.json')",
        "live=[]",
        "for entry in pathlib.Path('/proc').iterdir():",
        " if not entry.name.isdigit() or int(entry.name)==os.getpid():",
        "  continue",
        " try:",
        "  args=[part.decode(errors='replace') for part in",
        "        (entry/'cmdline').read_bytes().split(b'\\0') if part]",
        " except (FileNotFoundError,PermissionError):",
        "  continue",
        " basenames={pathlib.PurePosixPath(arg).name for arg in args}",
        " if tag in args and basenames.intersection({",
        "  'qwen38_tp4_collective_reduction_supervisor.py',",
        "  'qwen38_tp4_collective_reduction_worker.py'}):",
        "  live.append(int(entry.name))",
        "print(json.dumps({",
        " 'attempt_exists':attempt.exists(),",
        " 'source_identity':identity,",
        " 'launch':launch,",
        " 'supervisor_receipt':receipt,",
        " 'live_exact_tag_pids':sorted(live),",
        "},sort_keys=True))",
    ))
    result = remote_runner(
        ssh_target=ssh_target,
        remote_argv=[
            REMOTE_PYTHON,
            "-c",
            script,
            plan["attempt_root"],
            plan["attempt_tag"],
        ],
        input_bytes=None,
        timeout_s=timeout_s,
        retry_count=retry_count,
        control_path=control_path,
    )
    return _parse_remote_json(result, "remote attempt state")


def query_remote_postprocess_state(
    *,
    plan,
    ssh_target,
    timeout_s,
    retry_count,
    control_path=None,
    remote_runner=run_remote_bytes,
):
    bundle_root = plan.get("bundle_root") if isinstance(plan, dict) else None
    if (
        not isinstance(bundle_root, str)
        or not PurePosixPath(bundle_root).is_absolute()
    ):
        raise ValueError("remote postprocess plan is invalid")
    script = "\n".join((
        "import hashlib,json,pathlib,sys",
        "bundle=pathlib.Path(sys.argv[1])",
        "producer_files=set(json.loads(sys.argv[2]))",
        "manifest_schema=sys.argv[3]",
        "if not bundle.is_dir() or bundle.is_symlink():",
        " raise ValueError('remote postprocess bundle is invalid')",
        "entries=list(bundle.iterdir())",
        "if any(path.is_symlink() or not path.is_file() for path in entries):",
        " raise ValueError('remote postprocess inventory is invalid')",
        "actual={path.name for path in entries}",
        "verified_files=producer_files|{'independent_verification.json'}",
        "if actual and actual not in (producer_files,verified_files):",
        " raise ValueError('remote postprocess inventory is partial')",
        "def load(name):",
        " path=bundle/name",
        " if not path.exists(): return None",
        " if not path.is_file() or path.is_symlink():",
        "  raise ValueError('remote postprocess artifact is invalid')",
        " value=json.loads(path.read_text(encoding='utf-8'))",
        " if not isinstance(value,dict):",
        "  raise ValueError('remote postprocess artifact is invalid')",
        " return value",
        "def digest(path):",
        " value=hashlib.sha256()",
        " with path.open('rb') as handle:",
        "  for chunk in iter(lambda:handle.read(1024*1024),b''):",
        "   value.update(chunk)",
        " return value.hexdigest()",
        "if not actual:",
        (
            " print(json.dumps({'producer':None,'verification':None},"
            "sort_keys=True))"
        ),
        " raise SystemExit(0)",
        "manifest=load('manifest.sha256')",
        (
            "if manifest.get('schema_version')!=manifest_schema "
            "or not isinstance(manifest.get('artifacts'),dict):"
        ),
        " raise ValueError('remote postprocess manifest is invalid')",
        "expected_hashes=actual-{'manifest.sha256'}",
        "if set(manifest['artifacts'])!=expected_hashes:",
        " raise ValueError('remote postprocess manifest is stale')",
        "for name,expected in manifest['artifacts'].items():",
        (
            " if not isinstance(expected,str) or len(expected)!=64 "
            "or digest(bundle/name)!=expected:"
        ),
        "  raise ValueError('remote postprocess manifest hash mismatch')",
        "print(json.dumps({",
        " 'producer':load('classification.json'),",
        " 'verification':load('independent_verification.json'),",
        "},sort_keys=True))",
    ))
    result = remote_runner(
        ssh_target=ssh_target,
        remote_argv=[
            REMOTE_PYTHON,
            "-c",
            script,
            bundle_root,
            json.dumps(
                sorted(PRODUCER_ARTIFACTS),
                separators=(",", ":"),
            ),
            MANIFEST_SCHEMA,
        ],
        input_bytes=None,
        timeout_s=timeout_s,
        retry_count=retry_count,
        control_path=control_path,
    )
    return _parse_remote_json(result, "remote postprocess state")


def validate_postprocess_state(state):
    if (
        not isinstance(state, dict)
        or set(state) != {"producer", "verification"}
    ):
        raise RuntimeError("remote postprocess state is invalid")
    producer = state["producer"]
    verification = state["verification"]
    if producer is not None and (
        not isinstance(producer, dict)
        or not isinstance(producer.get("classification"), str)
    ):
        raise RuntimeError("remote producer state is invalid")
    if verification is not None:
        if producer is None:
            raise RuntimeError(
                "remote verification exists without producer"
            )
        if (
            not isinstance(verification, dict)
            or verification.get("status") != "PASS"
            or not isinstance(
                verification.get("producer_classification"),
                str,
            )
            or not isinstance(
                verification.get("reconstructed_classification"),
                str,
            )
        ):
            raise RuntimeError("remote verification state is invalid")
        if (
            verification["producer_classification"]
            != producer["classification"]
            or verification["reconstructed_classification"]
            != producer["classification"]
        ):
            raise RuntimeError(
                "remote postprocess classification mismatch"
            )
    return {
        "producer": producer,
        "verification": verification,
    }


def launch_remote_supervisor(
    *,
    plan,
    supervisor_argv,
    ssh_target,
    timeout_s,
    retry_count,
    control_path=None,
    remote_runner=run_remote_bytes,
):
    expected = build_supervisor_argv(plan)
    if supervisor_argv != expected:
        raise ValueError("supervisor argv does not match plan")
    script = "\n".join((
        "import json,os,pathlib,subprocess,sys",
        "attempt=pathlib.Path(sys.argv[1])",
        "source=pathlib.Path(sys.argv[2])",
        "revision=sys.argv[3]",
        "argv=json.loads(sys.argv[4])",
        "controller=attempt/'controller'",
        "launch=controller/'supervisor_launch.json'",
        "lock=controller/'.supervisor_launch.lock'",
        "if launch.is_file() and not launch.is_symlink():",
        " print(launch.read_text(encoding='utf-8'),end='')",
        " raise SystemExit(0)",
        "fd=os.open(lock,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)",
        "os.close(fd)",
        "published=False",
        "try:",
        " runtime=attempt/'runtime'",
        " for path in (runtime/'tmp',runtime/'cache'/'xdg',",
        "              runtime/'cache'/'huggingface',",
        "              runtime/'cache'/'torch',",
        "              runtime/'cache'/'torch-extensions',",
        "              runtime/'cache'/'cuda',",
        "              runtime/'cache'/'triton'):",
        "  path.mkdir(parents=True,exist_ok=True)",
        " env=os.environ.copy()",
        " env.update({",
        "  'PYTHONDONTWRITEBYTECODE':'1',",
        "  'TMPDIR':str(runtime/'tmp'),",
        "  'XDG_CACHE_HOME':str(runtime/'cache'/'xdg'),",
        "  'HF_HOME':str(runtime/'cache'/'huggingface'),",
        "  'TORCH_HOME':str(runtime/'cache'/'torch'),",
        (
            "  'TORCH_EXTENSIONS_DIR':"
            "str(runtime/'cache'/'torch-extensions'),"
        ),
        "  'CUDA_CACHE_PATH':str(runtime/'cache'/'cuda'),",
        "  'TRITON_CACHE_DIR':str(runtime/'cache'/'triton'),",
        " })",
        " stdout=(controller/'supervisor.stdout').open('ab')",
        " stderr=(controller/'supervisor.stderr').open('ab')",
        " process=subprocess.Popen(",
        "  argv,cwd=source,env=env,stdin=subprocess.DEVNULL,",
        "  stdout=stdout,stderr=stderr,start_new_session=True)",
        " receipt={",
        "  'classification':'LAUNCHED',",
        "  'pid':process.pid,",
        "  'source_revision':revision,",
        " }",
        " partial=controller/'.supervisor_launch.json.partial'",
        (
            " partial.write_text(json.dumps(receipt,sort_keys=True,"
            "separators=(',',':'))+'\\n',encoding='utf-8')"
        ),
        " os.replace(partial,launch)",
        " published=True",
        " print(json.dumps(receipt,sort_keys=True))",
        "finally:",
        " if published and lock.exists(): lock.unlink()",
    ))
    result = remote_runner(
        ssh_target=ssh_target,
        remote_argv=[
            REMOTE_PYTHON,
            "-c",
            script,
            plan["attempt_root"],
            plan["source_root"],
            plan["source_revision"],
            json.dumps(
                supervisor_argv,
                sort_keys=True,
                separators=(",", ":"),
            ),
        ],
        input_bytes=None,
        timeout_s=timeout_s,
        retry_count=retry_count,
        control_path=control_path,
    )
    return _parse_remote_json(result, "remote supervisor launch")


def load_remote_json(
    *,
    path,
    ssh_target,
    timeout_s,
    retry_count,
    control_path=None,
    remote_runner=run_remote_bytes,
):
    remote_path = PurePosixPath(path)
    if not remote_path.is_absolute() or ".." in remote_path.parts:
        raise ValueError("remote JSON path is invalid")
    script = "\n".join((
        "import pathlib,sys",
        "path=pathlib.Path(sys.argv[1])",
        "if not path.is_file() or path.is_symlink():",
        " raise ValueError('remote JSON file is invalid')",
        "sys.stdout.buffer.write(path.read_bytes())",
    ))
    result = remote_runner(
        ssh_target=ssh_target,
        remote_argv=[REMOTE_PYTHON, "-c", script, path],
        input_bytes=None,
        timeout_s=timeout_s,
        retry_count=retry_count,
        control_path=control_path,
    )
    return _parse_remote_json(result, "remote JSON load")


def run_remote_json_command(
    *,
    remote_argv,
    ssh_target,
    timeout_s,
    retry_count,
    control_path=None,
    remote_runner=run_remote_bytes,
):
    result = remote_runner(
        ssh_target=ssh_target,
        remote_argv=remote_argv,
        input_bytes=None,
        timeout_s=timeout_s,
        retry_count=retry_count,
        control_path=control_path,
    )
    return _parse_remote_json(result, "remote command")


def fetch_remote_bundle(
    *,
    plan,
    names,
    ssh_target,
    timeout_s,
    retry_count,
    control_path=None,
    remote_runner=run_remote_bytes,
):
    expected = _validate_relative_file_set(names)
    if any(
        not name.startswith("final_bundle/")
        or PurePosixPath(name).parent
        != PurePosixPath("final_bundle")
        for name in expected
    ):
        raise ValueError("remote bundle inventory is invalid")
    script = "\n".join((
        "import json,pathlib,sys,tarfile",
        "bundle=pathlib.Path(sys.argv[1])",
        "names=json.loads(sys.argv[2])",
        "expected={pathlib.PurePosixPath(name).name for name in names}",
        "actual={",
        " path.name for path in bundle.iterdir()",
        " if path.is_file() and not path.is_symlink()",
        "}",
        "if actual != expected:",
        " raise ValueError('remote bundle inventory mismatch')",
        "with tarfile.open(fileobj=sys.stdout.buffer,mode='w|') as archive:",
        " for name in sorted(expected):",
        "  path=bundle/name",
        "  if not path.is_file() or path.is_symlink():",
        "   raise ValueError('remote bundle file is invalid')",
        "  archive.add(path,arcname='final_bundle/'+name,recursive=False)",
    ))
    result = remote_runner(
        ssh_target=ssh_target,
        remote_argv=[
            REMOTE_PYTHON,
            "-c",
            script,
            plan["bundle_root"],
            json.dumps(sorted(expected), separators=(",", ":")),
        ],
        input_bytes=None,
        timeout_s=timeout_s,
        retry_count=retry_count,
        control_path=control_path,
    )
    _require_remote_success(result, "remote bundle fetch")
    archive = getattr(result, "stdout", None)
    if not isinstance(archive, bytes) or not archive:
        raise ValueError("remote bundle fetch is empty")
    return archive


def build_gpu_topology_evidence(plan, topology):
    if (
        not isinstance(plan, dict)
        or not isinstance(plan.get("selected_gpus"), list)
        or len(plan["selected_gpus"]) != 4
        or not isinstance(topology, dict)
        or set(topology) != {"gpu_rows", "interconnect_matrix"}
        or not isinstance(topology["gpu_rows"], list)
        or not isinstance(topology["interconnect_matrix"], str)
        or not topology["interconnect_matrix"].strip()
    ):
        raise ValueError("GPU topology evidence is invalid")
    rows = {}
    for row in topology["gpu_rows"]:
        if (
            not isinstance(row, dict)
            or set(row)
            != {"gpu_index", "gpu_uuid", "pci_bus_id"}
            or not isinstance(row["gpu_index"], int)
            or isinstance(row["gpu_index"], bool)
            or row["gpu_index"] < 0
            or not isinstance(row["gpu_uuid"], str)
            or not row["gpu_uuid"].startswith("GPU-")
            or not isinstance(row["pci_bus_id"], str)
            or not row["pci_bus_id"]
            or row["gpu_uuid"] in rows
        ):
            raise ValueError("GPU topology evidence is invalid")
        rows[row["gpu_uuid"]] = row
    rank_mapping = []
    for rank, selected in enumerate(plan["selected_gpus"]):
        current = rows.get(selected.get("gpu_uuid"))
        if (
            current is None
            or current["gpu_index"] != selected.get("gpu_index")
            or f"GPU{current['gpu_index']}"
            not in topology["interconnect_matrix"]
        ):
            raise ValueError("GPU topology identity mismatch")
        rank_mapping.append({
            "rank": rank,
            "gpu_index": current["gpu_index"],
            "gpu_uuid": current["gpu_uuid"],
            "pci_bus_id": current["pci_bus_id"],
        })
    return {
        "schema_version": (
            "qwen38.tp4-collective-reduction-topology.v1"
        ),
        "rank_mapping": rank_mapping,
        "interconnect_matrix": topology["interconnect_matrix"],
    }


def validate_launch_inventory(plan, observed_gpus):
    if (
        not isinstance(plan, dict)
        or not isinstance(plan.get("selected_gpus"), list)
    ):
        raise ValueError("launch inventory plan is invalid")
    try:
        selected = validate_selected_gpu_processes(
            selected=tuple(plan["selected_gpus"]),
            observed=observed_gpus,
            owned_pids=set(),
        )
    except ValueError as error:
        raise ValueError(
            "planned GPUs are not strict-clean"
        ) from error
    if any(
        row["memory_used_mib"] > MAX_GPU_MEMORY_USED_MIB
        or row["utilization_percent"]
        > MAX_GPU_UTILIZATION_PERCENT
        for row in selected
    ):
        raise ValueError("planned GPUs are not strict-clean")
    return [dict(row) for row in selected]


def _validate_relative_file_set(expected_files):
    if (
        not isinstance(expected_files, (set, frozenset))
        or not expected_files
    ):
        raise ValueError("download inventory is invalid")
    result = set()
    for name in expected_files:
        if not isinstance(name, str):
            raise ValueError("download inventory is invalid")
        path = PurePosixPath(name)
        if (
            path.is_absolute()
            or ".." in path.parts
            or path.as_posix() != name
            or name in result
        ):
            raise ValueError("download inventory is invalid")
        result.add(name)
    return result


def extract_bounded_archive(archive, *, destination, expected_files):
    expected = _validate_relative_file_set(expected_files)
    if not isinstance(archive, bytes) or not archive:
        raise ValueError("download archive is invalid")
    destination = Path(destination)
    if destination.is_symlink() or (
        destination.exists() and not destination.is_dir()
    ):
        raise ValueError("existing download destination is invalid")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        dir=destination.parent,
        prefix=f".{destination.name}.",
    ) as temporary_name:
        temporary = Path(temporary_name)
        observed = {}
        try:
            with tarfile.open(
                fileobj=io.BytesIO(archive),
                mode="r:",
            ) as bundle:
                for member in bundle.getmembers():
                    path = PurePosixPath(member.name)
                    if (
                        path.is_absolute()
                        or ".." in path.parts
                        or path.as_posix() != member.name
                    ):
                        raise ValueError(
                            "download archive contains unsafe path"
                        )
                    if member.isdir():
                        continue
                    if not member.isreg() or member.name in observed:
                        raise ValueError(
                            "download archive inventory is invalid"
                        )
                    handle = bundle.extractfile(member)
                    if handle is None:
                        raise ValueError(
                            "download archive inventory is invalid"
                        )
                    payload = handle.read()
                    target = temporary / member.name
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with target.open("xb") as output:
                        output.write(payload)
                        output.flush()
                        os.fsync(output.fileno())
                    observed[member.name] = _sha256(payload)
        except tarfile.TarError as error:
            raise ValueError("download archive is invalid") from error
        if set(observed) != expected:
            raise ValueError("download archive inventory mismatch")
        if destination.exists():
            existing = {}
            for path in destination.rglob("*"):
                if path.is_symlink():
                    raise ValueError(
                        "existing download contains a symlink"
                    )
                if path.is_file():
                    relative = path.relative_to(destination).as_posix()
                    existing[relative] = _sha256(path.read_bytes())
            if existing != observed:
                raise ValueError(
                    "existing download does not match remote bytes"
                )
            return dict(sorted(observed.items()))
        os.replace(temporary, destination)
    return dict(sorted(observed.items()))


def require_byte_identical_verification(remote_bytes, local_bytes):
    if (
        not isinstance(remote_bytes, bytes)
        or not isinstance(local_bytes, bytes)
        or remote_bytes != local_bytes
    ):
        raise RuntimeError(
            "remote/local verification is not byte-identical"
        )
    try:
        payload = json.loads(remote_bytes)
    except json.JSONDecodeError as error:
        raise ValueError("verification JSON is invalid") from error
    if (
        not isinstance(payload, dict)
        or payload.get("status") != "PASS"
    ):
        raise ValueError("verification JSON is invalid")
    return payload


def _write_bytes_atomic(path, payload):
    path = Path(path)
    if not isinstance(payload, bytes):
        raise ValueError("atomic payload must be bytes")
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="wb",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".partial",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def _canonical_json_bytes(payload):
    return (
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


class CollectiveReductionProductionAdapter:
    def __init__(
        self,
        *,
        plan,
        source_identity,
        model_manifest,
        repo_root,
        local_attempt_root,
        topology_query,
        launch_guard,
        archive_builder=build_committed_source_archive,
        attempt_state_query,
        attempt_stager,
        supervisor_launcher,
        remote_json_loader,
        postprocess_state_query,
        remote_command_runner,
        bundle_fetcher,
        local_verifier_runner,
        sleep,
        maximum_poll_count=21600,
    ):
        callbacks = (
            topology_query,
            launch_guard,
            archive_builder,
            attempt_state_query,
            attempt_stager,
            supervisor_launcher,
            remote_json_loader,
            postprocess_state_query,
            remote_command_runner,
            bundle_fetcher,
            local_verifier_runner,
            sleep,
        )
        if (
            not isinstance(plan, dict)
            or not isinstance(source_identity, dict)
            or not isinstance(model_manifest, dict)
            or not all(callable(callback) for callback in callbacks)
            or not isinstance(maximum_poll_count, int)
            or isinstance(maximum_poll_count, bool)
            or maximum_poll_count <= 0
        ):
            raise ValueError("production adapter configuration is invalid")
        self.plan = plan
        self.source_identity = source_identity
        self.model_manifest = model_manifest
        self.repo_root = Path(repo_root).resolve()
        self.local_attempt_root = Path(local_attempt_root).resolve()
        self.local_controller_root = (
            self.local_attempt_root / "controller"
        )
        self.topology_query = topology_query
        self.launch_guard = launch_guard
        self.archive_builder = archive_builder
        self.attempt_state_query = attempt_state_query
        self.attempt_stager = attempt_stager
        self.supervisor_launcher = supervisor_launcher
        self.remote_json_loader = remote_json_loader
        self.postprocess_state_query = postprocess_state_query
        self.remote_command_runner = remote_command_runner
        self.bundle_fetcher = bundle_fetcher
        self.local_verifier_runner = local_verifier_runner
        self.sleep = sleep
        self.maximum_poll_count = maximum_poll_count
        self.download_root = self.local_attempt_root / "downloaded"
        self.remote_verification_bytes = None
        self.postprocess_state = None

    def _require_plan(self, plan):
        if plan != self.plan:
            raise ValueError("production adapter plan identity drift")

    def worker_runner(self, plan):
        self._require_plan(plan)
        state = self.attempt_state_query()
        action = classify_attempt_state(
            state,
            source_identity=self.source_identity,
        )
        if action == "CREATE":
            topology = build_gpu_topology_evidence(
                plan,
                self.topology_query(),
            )
            archive = self.archive_builder(
                repo_root=self.repo_root,
                source_revision=plan["source_revision"],
                source_identity=self.source_identity,
            )
            staged = self.attempt_stager(
                plan=plan,
                source_identity=self.source_identity,
                model_manifest=self.model_manifest,
                gpu_topology=topology,
                archive=archive,
            )
            if (
                not isinstance(staged, dict)
                or staged.get("classification") != "STAGED"
            ):
                raise RuntimeError("remote source staging failed")
            state = self.attempt_state_query()
            action = classify_attempt_state(
                state,
                source_identity=self.source_identity,
            )
        if action == "LAUNCH":
            launch_admission = self.launch_guard()
            if (
                not isinstance(launch_admission, dict)
                or launch_admission.get("classification") != "READY"
            ):
                raise RuntimeError(
                    "launch-time resource admission failed"
                )
            _write_bytes_atomic(
                self.local_controller_root
                / "launch_admission.json",
                _canonical_json_bytes(launch_admission),
            )
            launch = self.supervisor_launcher(
                build_supervisor_argv(plan)
            )
            if (
                not isinstance(launch, dict)
                or launch.get("source_revision")
                != plan["source_revision"]
                or not isinstance(launch.get("pid"), int)
                or isinstance(launch["pid"], bool)
                or launch["pid"] <= 0
            ):
                raise RuntimeError("remote supervisor launch failed")
            state = self.attempt_state_query()
            action = classify_attempt_state(
                state,
                source_identity=self.source_identity,
            )
        polls = 0
        while action == "MONITOR":
            if polls >= self.maximum_poll_count:
                raise TimeoutError("remote supervisor monitoring timed out")
            polls += 1
            self.sleep(1)
            state = self.attempt_state_query()
            action = classify_attempt_state(
                state,
                source_identity=self.source_identity,
            )
        if action != "POSTPROCESS":
            raise RuntimeError("remote attempt did not reach postprocess")
        receipt = state["supervisor_receipt"]
        if receipt.get("classification") != "PASS":
            raise RuntimeError("remote supervisor failed")
        worker = self.remote_json_loader(
            f"{plan['attempt_root']}/worker.json"
        )
        if not isinstance(worker, dict):
            raise RuntimeError("remote worker result is invalid")
        self.postprocess_state = validate_postprocess_state(
            self.postprocess_state_query()
        )
        return worker

    def assembler(self, plan, worker):
        self._require_plan(plan)
        if not isinstance(worker, dict):
            raise ValueError("worker result is invalid")
        if self.postprocess_state is None:
            raise RuntimeError("remote postprocess state was not queried")
        result = self.postprocess_state["producer"]
        if result is None:
            result = self.remote_command_runner([
                REMOTE_PYTHON,
                (
                    f"{plan['source_root']}/tools/"
                    "assemble_qwen38_tp4_collective_reduction.py"
                ),
                "--attempt-root",
                plan["attempt_root"],
                "--bundle-root",
                plan["bundle_root"],
            ])
        if (
            not isinstance(result, dict)
            or not isinstance(result.get("classification"), str)
        ):
            raise RuntimeError("remote producer result is invalid")
        _write_bytes_atomic(
            self.local_controller_root / "producer_result.json",
            _canonical_json_bytes(result),
        )
        self.postprocess_state["producer"] = result
        return result

    def remote_verifier(self, plan):
        self._require_plan(plan)
        if self.postprocess_state is None:
            raise RuntimeError("remote postprocess state was not queried")
        result = self.postprocess_state["verification"]
        if result is None:
            result = self.remote_command_runner([
                REMOTE_PYTHON,
                (
                    f"{plan['source_root']}/tools/"
                    "verify_qwen38_tp4_collective_reduction.py"
                ),
                "--bundle-root",
                plan["bundle_root"],
            ])
        if (
            not isinstance(result, dict)
            or result.get("status") != "PASS"
            or not isinstance(
                result.get("reconstructed_classification"),
                str,
            )
        ):
            raise RuntimeError("remote independent verifier failed")
        self.postprocess_state = validate_postprocess_state({
            "producer": self.postprocess_state["producer"],
            "verification": result,
        })
        normalized = dict(result)
        normalized["classification"] = result[
            "reconstructed_classification"
        ]
        return normalized

    def downloader(self, plan):
        self._require_plan(plan)
        names = {
            f"final_bundle/{name}"
            for name in (
                *PRODUCER_ARTIFACTS,
                "independent_verification.json",
            )
        }
        archive = self.bundle_fetcher(plan, names)
        inventory = extract_bounded_archive(
            archive,
            destination=self.download_root,
            expected_files=names,
        )
        bundle_root = self.download_root / "final_bundle"
        self.remote_verification_bytes = (
            bundle_root / "independent_verification.json"
        ).read_bytes()
        _write_bytes_atomic(
            self.local_controller_root
            / "remote-independent-verification.json",
            self.remote_verification_bytes,
        )
        _write_bytes_atomic(
            self.local_controller_root
            / "remote-post-verification-manifest.json",
            (bundle_root / "manifest.sha256").read_bytes(),
        )
        return {
            "downloaded": True,
            "artifact_count": len(inventory),
            "inventory": inventory,
        }

    def local_verifier(self, plan):
        self._require_plan(plan)
        if self.remote_verification_bytes is None:
            raise RuntimeError("remote verification was not downloaded")
        bundle_root = self.download_root / "final_bundle"
        result = self.local_verifier_runner(bundle_root)
        local_bytes = (
            bundle_root / "independent_verification.json"
        ).read_bytes()
        verified = require_byte_identical_verification(
            self.remote_verification_bytes,
            local_bytes,
        )
        if result != verified:
            raise RuntimeError("local verifier result is inconsistent")
        _write_bytes_atomic(
            self.local_controller_root
            / "local-independent-verification.json",
            local_bytes,
        )
        normalized = dict(result)
        normalized["classification"] = result[
            "reconstructed_classification"
        ]
        return normalized

    def cleanup_validator(self, plan, worker):
        self._require_plan(plan)
        if not isinstance(worker, dict):
            raise ValueError("worker result is invalid")
        cleanup = json.loads(
            (
                self.download_root
                / "final_bundle"
                / "cleanup.json"
            ).read_text(encoding="utf-8")
        )
        if (
            not isinstance(cleanup, dict)
            or cleanup.get("complete") is not True
            or cleanup.get("process_group_destroyed") is not True
            or cleanup.get("owned_children_remaining") != []
            or cleanup.get("exact_tag_scans") != [[], [], []]
        ):
            raise RuntimeError("remote cleanup evidence is invalid")
        return cleanup


def create_production_adapter(
    *,
    plan,
    source_identity,
    model_manifest,
    repo_root,
    local_attempt_root,
    ssh_target,
    control_path,
    command_timeout_s,
    retry_count,
    command_runner=subprocess.run,
    sleep=time.sleep,
    topology_query=query_remote_gpu_topology,
):
    if not callable(topology_query):
        raise ValueError("GPU topology query is invalid")

    def remote_runner(**kwargs):
        return run_remote_bytes(
            **kwargs,
            command_runner=command_runner,
            sleep=sleep,
        )

    common = {
        "ssh_target": ssh_target,
        "timeout_s": command_timeout_s,
        "retry_count": retry_count,
        "control_path": control_path,
    }
    postprocess_common = {
        **common,
        "timeout_s": max(
            command_timeout_s,
            MIN_POSTPROCESS_COMMAND_TIMEOUT_S,
        ),
    }

    def launch_guard():
        kerberos = query_local_kerberos(
            command_runner=command_runner,
        )
        if kerberos.get("classification") != "READY":
            raise RuntimeError("launch-time Kerberos TTL is insufficient")
        observed = query_remote_gpu_inventory(
            **common,
            command_runner=command_runner,
        )
        selected = validate_launch_inventory(plan, observed)
        return {
            "schema_version": (
                "qwen38.tp4-collective-reduction-launch-admission.v1"
            ),
            "classification": "READY",
            "kerberos": kerberos,
            "selected_gpus": selected,
            "maximum_memory_used_mib": MAX_GPU_MEMORY_USED_MIB,
            "maximum_utilization_percent": (
                MAX_GPU_UTILIZATION_PERCENT
            ),
            "compute_processes_required_empty": True,
        }

    return CollectiveReductionProductionAdapter(
        plan=plan,
        source_identity=source_identity,
        model_manifest=model_manifest,
        repo_root=repo_root,
        local_attempt_root=local_attempt_root,
        topology_query=lambda: topology_query(
            **common,
            command_runner=command_runner,
        ),
        launch_guard=launch_guard,
        attempt_state_query=lambda: query_remote_attempt_state(
            plan=plan,
            remote_runner=remote_runner,
            **common,
        ),
        attempt_stager=lambda **kwargs: stage_remote_attempt(
            **kwargs,
            remote_runner=remote_runner,
            **common,
        ),
        supervisor_launcher=lambda argv: launch_remote_supervisor(
            plan=plan,
            supervisor_argv=argv,
            remote_runner=remote_runner,
            **common,
        ),
        remote_json_loader=lambda path: load_remote_json(
            path=path,
            remote_runner=remote_runner,
            **common,
        ),
        postprocess_state_query=lambda: query_remote_postprocess_state(
            plan=plan,
            remote_runner=remote_runner,
            **common,
        ),
        remote_command_runner=lambda argv: run_remote_json_command(
            remote_argv=argv,
            remote_runner=remote_runner,
            **postprocess_common,
        ),
        bundle_fetcher=lambda current_plan, names: fetch_remote_bundle(
            plan=current_plan,
            names=names,
            remote_runner=remote_runner,
            **postprocess_common,
        ),
        local_verifier_runner=verify_bundle,
        sleep=sleep,
    )
