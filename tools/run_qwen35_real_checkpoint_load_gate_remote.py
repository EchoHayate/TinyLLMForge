from __future__ import annotations

import argparse
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
from pathlib import PurePosixPath
import re
import shlex
import subprocess
import sys
import tarfile
import tempfile

import qwen35_real_checkpoint_load_contract as contract


SSH_CONTROL_PATH = "/tmp/ssh-sitian-10.232.195.203"
REMOTE_RUN_ROOT = (
    "/data00/home/sitian/sitian-workspace01/tllm/"
    "qwen35-real-checkpoint-load-runs"
)
LOCAL_RUN_ROOT = Path("experiments/qwen35_hybrid_state")
RUN_TAG_PATTERN = re.compile(r"^[A-Za-z0-9_-]+$")
OWNED_SOURCE_FILES = (
    "tools/qwen35_real_checkpoint_load_contract.py",
    "tools/qwen35_real_checkpoint_load_authorization.py",
    "tools/qwen35_real_checkpoint_load_worker.py",
    "tools/verify_qwen35_real_checkpoint_load_gate.py",
    "tools/run_qwen35_real_checkpoint_load_gate_remote.py",
    "tools/test_qwen35_real_checkpoint_load_authorization.py",
    "tools/test_qwen35_real_checkpoint_load_safety_gate.py",
)
MODES = (
    "preflight",
    "restore-model-manifest",
    "run",
    "download-only",
    "verify-only",
    "authorization-only",
    "dry-run",
)
PREFLIGHT_CHECK_NAMES = (
    "source_identity",
    "remote_identity",
    "runtime_dependencies",
    "model_identity",
    "model_files",
    "proc_telemetry",
    "run_root_space",
    "cuda_disabled",
    "gpu0_idle",
    "payload_zero",
)


def validate_run_tag(value):
    text = str(value)
    if not RUN_TAG_PATTERN.fullmatch(text):
        raise ValueError("run tag must match [A-Za-z0-9_-]+")
    return text


def build_ssh_command(remote_arguments):
    remote_command = shlex.join(
        [str(value) for value in remote_arguments]
    )
    return [
        "ssh",
        "-S",
        SSH_CONTROL_PATH,
        "-o",
        "BatchMode=yes",
        contract.REMOTE_TARGET,
        remote_command,
    ]


def reject_unimplemented_execution(mode):
    if mode in (
        "dry-run",
        "preflight",
        "restore-model-manifest",
        "verify-only",
        "download-only",
        "authorization-only",
    ):
        return
    if mode not in MODES:
        raise ValueError("remote runner mode is invalid")
    raise RuntimeError(
        f"{mode} execution is not implemented; dry-run, read-only "
        "preflight, zero-payload restore, local verify-only, and "
        "manifest-bound download-only are authorized"
    )


def _load_local_verifier():
    path = Path(__file__).with_name(
        "verify_qwen35_real_checkpoint_load_gate.py"
    )
    spec = importlib.util.spec_from_file_location(
        "qwen35_real_checkpoint_load_verifier_for_runner",
        os.fspath(path),
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.verify_run


def _load_local_authorization():
    path = Path(__file__).with_name(
        "qwen35_real_checkpoint_load_authorization.py"
    )
    spec = importlib.util.spec_from_file_location(
        "qwen35_real_checkpoint_load_authorization_for_runner",
        os.fspath(path),
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.authorize_run


def _safe_artifact_path(value):
    if not isinstance(value, str) or not value:
        raise ValueError("artifact path is invalid")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise ValueError(f"unsafe artifact path: {value}")
    return path


def _sha256_path(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _source_hashes(repo_root):
    root = Path(repo_root)
    result = {}
    for relative in OWNED_SOURCE_FILES:
        path = root / relative
        if not path.is_file():
            raise ValueError(f"missing owned source file: {relative}")
        result[relative] = _sha256_path(path)
    return result


def _source_tree_sha256(hashes):
    payload = json.dumps(
        hashes,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def build_source_tar(repo_root):
    root = Path(repo_root)
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w") as archive:
        for relative in OWNED_SOURCE_FILES:
            path = root / relative
            if not path.is_file():
                raise ValueError(f"missing owned source file: {relative}")
            info = archive.gettarinfo(os.fspath(path), arcname=relative)
            info.uid = 0
            info.gid = 0
            info.uname = ""
            info.gname = ""
            info.mtime = 0
            with path.open("rb") as source:
                archive.addfile(info, source)
    return stream.getvalue()


def _require_success(result, context):
    if result.returncode != 0:
        detail = result.stderr or result.stdout or ""
        if isinstance(detail, bytes):
            detail = detail.decode("utf-8", errors="replace")
        raise RuntimeError(f"{context} failed: {str(detail).strip()}")
    return result


def stage_owned_source(repo_root, run_tag, command_runner=subprocess.run):
    run_tag = validate_run_tag(run_tag)
    remote_run_dir = f"{REMOTE_RUN_ROOT}/{run_tag}"
    remote_source_dir = f"{remote_run_dir}/source"
    payload = build_source_tar(repo_root)
    stage = command_runner(
        build_ssh_command([
            "bash",
            "-c",
            (
                f"test ! -e {shlex.quote(remote_run_dir)} && "
                f"mkdir -p {shlex.quote(remote_source_dir)} && "
                f"tar -xf - -C {shlex.quote(remote_source_dir)}"
            ),
        ]),
        input=payload,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    _require_success(stage, "source staging")
    local_hashes = _source_hashes(repo_root)
    script = "\n".join([
        "import hashlib,json,pathlib",
        f"root=pathlib.Path({remote_source_dir!r})",
        f"names={list(OWNED_SOURCE_FILES)!r}",
        "result={}",
        "for name in names:",
        " path=root/name",
        " if not path.is_file(): raise SystemExit('missing staged source: '+name)",
        " digest=hashlib.sha256()",
        " with path.open('rb') as handle:",
        "  while True:",
        "   block=handle.read(1048576)",
        "   if not block: break",
        "   digest.update(block)",
        " result[name]=digest.hexdigest()",
        "print(json.dumps(result,sort_keys=True,separators=(',',':')))",
    ])
    verified = command_runner(
        build_ssh_command([contract.REMOTE_PYTHON, "-c", script]),
        text=True,
        capture_output=True,
    )
    _require_success(verified, "remote source hashing")
    remote_hashes = json.loads(verified.stdout)
    if remote_hashes != local_hashes:
        raise ValueError("remote staged source hashes do not match local")
    return {
        "remote_source_dir": remote_source_dir,
        "local_file_sha256": local_hashes,
        "remote_file_sha256": remote_hashes,
        "source_tree_sha256": _source_tree_sha256(local_hashes),
    }


def build_source_manifest(repo_root, staged):
    return {
        "schema_version": contract.SCHEMA_VERSION,
        "branch": _git_output(repo_root, "branch", "--show-current"),
        "commit": _git_output(repo_root, "rev-parse", "HEAD"),
        "remote_target": contract.REMOTE_TARGET,
        "remote_source_dir": staged["remote_source_dir"],
        "source_tree_sha256": staged["source_tree_sha256"],
        "local_file_sha256": dict(staged["local_file_sha256"]),
        "remote_file_sha256": dict(staged["remote_file_sha256"]),
    }


def _git_output(repo_root, *arguments):
    completed = subprocess.run(
        ["git", *arguments],
        cwd=repo_root,
        check=False,
        text=True,
        capture_output=True,
    )
    _require_success(completed, f"git {' '.join(arguments)}")
    value = completed.stdout.strip()
    if not value:
        raise ValueError(f"git {' '.join(arguments)} returned empty output")
    return value


def build_remote_preflight_script(run_tag, source_file_sha256):
    run_tag = validate_run_tag(run_tag)
    remote_run_dir = f"{REMOTE_RUN_ROOT}/{run_tag}"
    source_dir = f"{remote_run_dir}/source"
    return f"""
import getpass
import hashlib
import importlib.metadata
import json
import os
import pathlib
import platform
import shutil
import subprocess
import sys

REMOTE_TARGET={contract.REMOTE_TARGET!r}
REMOTE_PYTHON={contract.REMOTE_PYTHON!r}
MODEL_REPOSITORY={contract.MODEL_REPOSITORY!r}
MODEL_REVISION={contract.MODEL_REVISION!r}
MANIFEST_PATH=pathlib.Path({contract.APPROVED_MODEL_MANIFEST_PATH!r})
MODEL_DIR=pathlib.Path({contract.APPROVED_MODEL_DIR!r})
RUN_DIR=pathlib.Path({remote_run_dir!r})
SOURCE_DIR=pathlib.Path({source_dir!r})
EXPECTED_SOURCE_HASHES={dict(source_file_sha256)!r}
REQUIRED_BYTES={contract.PREFLIGHT_ARTIFACT_ALLOWANCE_BYTES}
payload_open_count=0
payload_bytes_read=0

def sha256_non_payload(path):
    if str(path).endswith('.safetensors'):
        raise RuntimeError('payload hash access is forbidden')
    digest=hashlib.sha256()
    with path.open('rb') as handle:
        while True:
            block=handle.read(1048576)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()

def package_version(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None

def read_key_values(path):
    result={{}}
    try:
        with path.open('r',encoding='utf-8') as handle:
            for line in handle:
                if ':' in line:
                    key,value=line.split(':',1)
                    result[key]=value.strip()
    except OSError:
        pass
    return result

def filesystem(path,fallback):
    query_path=path if path.exists() else fallback
    result=subprocess.run(
        ['findmnt','-J','-T',str(query_path),'-o','SOURCE,TARGET,FSTYPE'],
        text=True,capture_output=True,check=False,
    )
    row={{'requested_path':str(path),'path_exists':path.exists()}}
    if result.returncode == 0:
        parsed=json.loads(result.stdout)
        filesystems=parsed.get('filesystems') or []
        if filesystems:
            item=filesystems[0]
            row.update({{
                'source':item.get('source'),
                'mountpoint':item.get('target'),
                'fstype':item.get('fstype'),
            }})
    row['device']=query_path.stat().st_dev
    return row

def atomic_json(path,payload):
    temporary=path.with_name('.'+path.name+'.tmp')
    with temporary.open('w',encoding='utf-8') as handle:
        json.dump(payload,handle,sort_keys=True,separators=(',',':'))
        handle.write('\\n')
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary,path)

RUN_DIR.mkdir(parents=True,exist_ok=True)
manifest_exists=MANIFEST_PATH.is_file()
manifest=(
    json.loads(MANIFEST_PATH.read_text(encoding='utf-8'))
    if manifest_exists else {{}}
)
config_path=MODEL_DIR/'config.json'
index_path=MODEL_DIR/'model.safetensors.index.json'
config_sha256=(
    sha256_non_payload(config_path) if config_path.is_file() else None
)
index_sha256=(
    sha256_non_payload(index_path) if index_path.is_file() else None
)
manifest_sha256=(
    sha256_non_payload(MANIFEST_PATH) if manifest_exists else None
)
source_hashes={{}}
for name in EXPECTED_SOURCE_HASHES:
    source_hashes[name]=sha256_non_payload(SOURCE_DIR/name)

declared_shards={{
    name:entry for name,entry in manifest.get('files',{{}}).items()
    if name.endswith('.safetensors')
}}
shards=[]
for name,entry in sorted(declared_shards.items()):
    path=MODEL_DIR/name
    if not path.is_file():
        continue
    stat=path.stat()
    shards.append({{
        'name':name,
        'expected_size':entry.get('size'),
        'observed_size':stat.st_size,
        'expected_sha256':entry.get('sha256'),
        'resolved_path':str(path.resolve()),
        'inode':stat.st_ino,
        'device':stat.st_dev,
        'mode':stat.st_mode,
    }})
observed_shard_names=(
    sorted(
        path.name for path in MODEL_DIR.iterdir()
        if path.is_file() and path.name.endswith('.safetensors')
    )
    if MODEL_DIR.is_dir() else []
)
identity_payload={{
    'config_sha256':manifest.get('files',{{}}).get('config.json',{{}}).get('sha256'),
    'index_sha256':manifest.get('files',{{}}).get(
        'model.safetensors.index.json',{{}}
    ).get('sha256'),
    'shards':declared_shards,
}}
identity_sha256=(
    hashlib.sha256(
        json.dumps(
            identity_payload,sort_keys=True,separators=(',',':')
        ).encode()
    ).hexdigest()
    if manifest_exists else None
)
gpu=subprocess.run(
    ['nvidia-smi','--id=0','--query-compute-apps=pid,process_name,used_memory',
     '--format=csv,noheader,nounits'],
    text=True,capture_output=True,check=False,
)
gpu_identity=subprocess.run(
    ['nvidia-smi','--id=0','--query-gpu=name,uuid,driver_version',
     '--format=csv,noheader,nounits'],
    text=True,capture_output=True,check=False,
)
gpu_fields=[
    value.strip()
    for value in (gpu_identity.stdout.splitlines() or [''])[0].split(',')
]
proc_meminfo=read_key_values(pathlib.Path('/proc/meminfo'))
proc_status=read_key_values(pathlib.Path('/proc/self/status'))
usage=shutil.disk_usage(RUN_DIR)
record={{
    'schema_version':{contract.SCHEMA_VERSION!r},
    'status':'INCOMPLETE',
    'failure_reasons':['classification not run'],
    'checks':{{name:False for name in {PREFLIGHT_CHECK_NAMES!r}}},
    'remote_target':REMOTE_TARGET,
    'observed_user':getpass.getuser(),
    'observed_hostname':platform.node(),
    'remote_python':sys.executable,
    'python_version':sys.version.split()[0],
    'packages':{{
        name:package_version(name)
        for name in {contract.REQUIRED_PREFLIGHT_PACKAGES!r}
    }},
    'model_repository':(
        manifest.get('repository') if manifest_exists else MODEL_REPOSITORY
    ),
    'model_revision':(
        manifest.get('resolved_revision') if manifest_exists
        else MODEL_REVISION
    ),
    'approved_model_manifest_path':str(MANIFEST_PATH),
    'approved_model_dir':str(MODEL_DIR),
    'cuda_visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES'),
    'cuda_initialized':False,
    'gpu_processes':[
        line for line in gpu.stdout.splitlines() if line.strip()
    ],
    'gpu_name':gpu_fields[0] if len(gpu_fields)>0 else None,
    'gpu_uuid':gpu_fields[1] if len(gpu_fields)>1 else None,
    'driver_version':gpu_fields[2] if len(gpu_fields)>2 else None,
    'source_tree_sha256':hashlib.sha256(
        json.dumps(source_hashes,sort_keys=True,separators=(',',':')).encode()
    ).hexdigest(),
    'source_file_sha256':EXPECTED_SOURCE_HASHES,
    'remote_source_file_sha256':source_hashes,
    'model_manifest_sha256':manifest_sha256,
    'config_index_header_sha256':identity_sha256,
    'config_sha256':config_sha256,
    'index_sha256':index_sha256,
    'shards':shards,
    'observed_shard_names':observed_shard_names,
    'payload_open_count':payload_open_count,
    'payload_bytes_read':payload_bytes_read,
    'payload_hashes_recomputed':False,
    'payload_identity_source':'approved_model_manifest',
    'proc_telemetry_available':all(
        name in proc_status for name in ('VmRSS','VmHWM')
    ),
    'proc_meminfo':proc_meminfo,
    'proc_status_fields':{{
        'VmRSS':'VmRSS' in proc_status,
        'VmHWM':'VmHWM' in proc_status,
    }},
    'run_root_filesystem':filesystem(RUN_DIR,RUN_DIR),
    'model_root_filesystem':filesystem(MODEL_DIR,RUN_DIR),
    'free_run_root_bytes':usage.free,
    'required_run_root_bytes':REQUIRED_BYTES,
}}
print(json.dumps(record,sort_keys=True,separators=(',',':')))
""".strip()


def build_restore_model_manifest_script(run_tag):
    run_tag = validate_run_tag(run_tag)
    remote_run_dir = f"{REMOTE_RUN_ROOT}/{run_tag}"
    canonical = contract.canonical_approved_model_manifest_bytes()
    return f"""
import hashlib
import json
import os
import pathlib
import tempfile

REMOTE_TARGET={contract.REMOTE_TARGET!r}
MANIFEST_PATH=pathlib.Path({contract.APPROVED_MODEL_MANIFEST_PATH!r})
MODEL_DIR=pathlib.Path({contract.APPROVED_MODEL_DIR!r})
RUN_DIR=pathlib.Path({remote_run_dir!r})
ARTIFACT_NAME='restore_model_manifest.json'
APPROVED_BYTES={canonical!r}
APPROVED_SHA256={contract.APPROVED_MODEL_MANIFEST_SHA256!r}
APPROVED_FILES={contract.APPROVED_MODEL_FILES!r}
APPROVED_SHARD={contract.APPROVED_SHARD_NAME!r}
payload_open_count=0
payload_bytes_read=0

def sha256_non_payload(path):
    if str(path).endswith('.safetensors'):
        raise RuntimeError('payload hash access is forbidden')
    digest=hashlib.sha256()
    with path.open('rb') as handle:
        while True:
            block=handle.read(1048576)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()

RUN_DIR.mkdir(parents=True,exist_ok=True)
observed_files={{}}
non_payload_ok=True
for name,expected in APPROVED_FILES.items():
    if name.endswith('.safetensors'):
        continue
    path=MODEL_DIR/name
    if not path.is_file():
        observed_files[name]={{'size':None,'sha256':None}}
        non_payload_ok=False
        continue
    stat=path.stat()
    digest=sha256_non_payload(path)
    observed_files[name]={{'size':stat.st_size,'sha256':digest}}
    if observed_files[name] != expected:
        non_payload_ok=False

config={{}}
index={{}}
try:
    config=json.loads((MODEL_DIR/'config.json').read_text(encoding='utf-8'))
except Exception:
    config={{}}
try:
    index=json.loads(
        (MODEL_DIR/'model.safetensors.index.json').read_text(
            encoding='utf-8'
        )
    )
except Exception:
    index={{}}
model_type=config.get('model_type')
index_shard_names=sorted(set((index.get('weight_map') or {{}}).values()))
observed_shard_names=(
    sorted(
        path.name for path in MODEL_DIR.iterdir()
        if path.is_file() and path.name.endswith('.safetensors')
    )
    if MODEL_DIR.is_dir() else []
)
shard_path=MODEL_DIR/APPROVED_SHARD
shard_stat=shard_path.stat() if shard_path.is_file() else None
shard={{
    'name':APPROVED_SHARD,
    'expected_size':APPROVED_FILES[APPROVED_SHARD]['size'],
    'observed_size':shard_stat.st_size if shard_stat else None,
    'inode':shard_stat.st_ino if shard_stat else 0,
    'device':shard_stat.st_dev if shard_stat else 0,
    'mode':shard_stat.st_mode if shard_stat else 0,
}}
target_exists=MANIFEST_PATH.is_file()
observed_manifest_sha256=(
    sha256_non_payload(MANIFEST_PATH) if target_exists else None
)
checks={{
    'target_state':(
        not target_exists or observed_manifest_sha256 == APPROVED_SHA256
    ),
    'model_directory':MODEL_DIR.is_dir(),
    'non_payload_files':non_payload_ok,
    'config_identity':model_type == 'qwen3_5',
    'index_identity':index_shard_names == [APPROVED_SHARD],
    'shard_inventory':(
        observed_shard_names == [APPROVED_SHARD]
        and shard['observed_size'] == shard['expected_size']
    ),
    'payload_zero':True,
}}
reason_by_check={{
    'target_state':'target manifest conflicts with approved bytes',
    'model_directory':'approved model directory is missing',
    'non_payload_files':'non-payload file identity mismatch',
    'config_identity':'config model identity mismatch',
    'index_identity':'index shard mapping mismatch',
    'shard_inventory':'shard stat inventory mismatch',
    'payload_zero':'payload-zero invariant was violated',
}}
write_performed=False
if target_exists and observed_manifest_sha256 != APPROVED_SHA256:
    status='CONFLICT'
elif all(checks.values()) and target_exists:
    status='ALREADY_PRESENT'
elif all(checks.values()):
    MANIFEST_PATH.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode='wb',
        dir=MANIFEST_PATH.parent,
        prefix='.model_manifest.',
        suffix='.tmp',
        delete=True,
    ) as handle:
        handle.write(APPROVED_BYTES)
        handle.flush()
        os.fsync(handle.fileno())
        try:
            os.link(handle.name,MANIFEST_PATH)
            write_performed=True
            status='RESTORED'
        except FileExistsError:
            observed_manifest_sha256=sha256_non_payload(MANIFEST_PATH)
            if observed_manifest_sha256 == APPROVED_SHA256:
                status='ALREADY_PRESENT'
            else:
                status='CONFLICT'
                checks['target_state']=False
else:
    status='INCOMPLETE'
if status in ('RESTORED','ALREADY_PRESENT'):
    observed_manifest_sha256=sha256_non_payload(MANIFEST_PATH)
failure_reasons=[
    reason_by_check[name] for name,passed in checks.items() if not passed
]
record={{
    'schema_version':{contract.SCHEMA_VERSION!r},
    'status':status,
    'checks':checks,
    'failure_reasons':failure_reasons,
    'remote_target':REMOTE_TARGET,
    'approved_model_manifest_path':str(MANIFEST_PATH),
    'approved_model_dir':str(MODEL_DIR),
    'approved_manifest_sha256':APPROVED_SHA256,
    'observed_manifest_sha256':observed_manifest_sha256,
    'non_payload_files':observed_files,
    'config_model_type':model_type,
    'index_shard_names':index_shard_names,
    'observed_shard_names':observed_shard_names,
    'shard':shard,
    'payload_open_count':payload_open_count,
    'payload_bytes_read':payload_bytes_read,
    'payload_hashes_recomputed':False,
    'write_performed':write_performed,
}}
print(json.dumps(record,sort_keys=True,separators=(',',':')))
""".strip()


def classify_preflight_payload(payload):
    record = dict(payload)
    checks = {
        "source_identity": (
            record.get("source_file_sha256")
            == record.get("remote_source_file_sha256")
            and record.get("source_tree_sha256")
            == _source_tree_sha256(record.get("source_file_sha256", {}))
        ),
        "remote_identity": (
            record.get("remote_target") == contract.REMOTE_TARGET
            and record.get("observed_user") == "sitian"
            and record.get("remote_python") == contract.REMOTE_PYTHON
        ),
        "runtime_dependencies": all(
            record.get("packages", {}).get(name)
            for name in contract.REQUIRED_PREFLIGHT_PACKAGES
        ),
        "model_identity": (
            record.get("model_repository") == contract.MODEL_REPOSITORY
            and record.get("model_revision") == contract.MODEL_REVISION
            and record.get("approved_model_manifest_path")
            == contract.APPROVED_MODEL_MANIFEST_PATH
            and record.get("approved_model_dir")
            == contract.APPROVED_MODEL_DIR
            and record.get("model_manifest_sha256")
            == contract.APPROVED_MODEL_MANIFEST_SHA256
            and record.get("config_sha256")
            == contract.APPROVED_CONFIG_SHA256
            and record.get("index_sha256")
            == contract.APPROVED_INDEX_SHA256
        ),
        "model_files": (
            len(record.get("shards", [])) == 1
            and all(
                shard.get("name") == contract.APPROVED_SHARD_NAME
                and shard.get("expected_size")
                == contract.APPROVED_SHARD_SIZE
                and shard.get("observed_size")
                == contract.APPROVED_SHARD_SIZE
                and shard.get("expected_sha256")
                == contract.APPROVED_SHARD_SHA256
                and shard.get("resolved_path")
                == (
                    f"{contract.APPROVED_MODEL_DIR}/"
                    f"{contract.APPROVED_SHARD_NAME}"
                )
                for shard in record.get("shards", [])
            )
            and record.get(
                "observed_shard_names",
                [contract.APPROVED_SHARD_NAME],
            ) == [contract.APPROVED_SHARD_NAME]
        ),
        "proc_telemetry": record.get("proc_telemetry_available") is True,
        "run_root_space": (
            isinstance(record.get("free_run_root_bytes"), int)
            and isinstance(record.get("required_run_root_bytes"), int)
            and record["free_run_root_bytes"]
            >= record["required_run_root_bytes"]
        ),
        "cuda_disabled": (
            record.get("cuda_visible_devices") == ""
            and record.get("cuda_initialized") is False
        ),
        "gpu0_idle": record.get("gpu_processes") == [],
        "payload_zero": (
            record.get("payload_open_count") == 0
            and record.get("payload_bytes_read") == 0
            and record.get("payload_hashes_recomputed") is False
            and record.get("payload_identity_source")
            == "approved_model_manifest"
        ),
    }
    reason_by_check = {
        "source_identity": "staged source identity mismatch",
        "remote_identity": "approved remote identity mismatch",
        "runtime_dependencies": "required runtime package is missing",
        "model_identity": "approved model identity mismatch",
        "model_files": "approved model shard stat inventory mismatch",
        "proc_telemetry": "required /proc telemetry is unavailable",
        "run_root_space": "run root free space is insufficient",
        "cuda_disabled": "CUDA visibility or initialization is unsafe",
        "gpu0_idle": "GPU0 has active compute processes",
        "payload_zero": "payload-zero invariant was violated",
    }
    record["checks"] = checks
    record["failure_reasons"] = [
        reason_by_check[name] for name, passed in checks.items() if not passed
    ]
    record["status"] = (
        "READY" if not record["failure_reasons"] else "INCOMPLETE"
    )
    return record


def persist_and_download_preflight_artifacts(
    run_tag,
    *,
    preflight,
    source_manifest,
    command_runner=subprocess.run,
):
    run_tag = validate_run_tag(run_tag)
    remote_run_dir = f"{REMOTE_RUN_ROOT}/{run_tag}"
    script = "\n".join([
        "import json,os,pathlib,sys",
        f"root=pathlib.Path({remote_run_dir!r})",
        "payload=json.load(sys.stdin)",
        "def atomic_json(path,value):",
        " temporary=path.with_name('.'+path.name+'.tmp')",
        " with temporary.open('w',encoding='utf-8') as handle:",
        "  json.dump(value,handle,sort_keys=True,separators=(',',':'))",
        "  handle.write('\\n')",
        "  handle.flush()",
        "  os.fsync(handle.fileno())",
        " os.replace(temporary,path)",
        "atomic_json(root/'preflight.json',payload['preflight'])",
        "atomic_json(root/'source_manifest.json',payload['source_manifest'])",
        "out={",
        " 'preflight':json.loads((root/'preflight.json').read_text()),",
        " 'source_manifest':json.loads((root/'source_manifest.json').read_text()),",
        "}",
        "print(json.dumps(out,sort_keys=True,separators=(',',':')))",
    ])
    expected = {
        "preflight": preflight,
        "source_manifest": source_manifest,
    }
    completed = command_runner(
        build_ssh_command([
            contract.REMOTE_PYTHON,
            "-c",
            script,
        ]),
        input=json.dumps(
            expected,
            sort_keys=True,
            separators=(",", ":"),
        ),
        text=True,
        capture_output=True,
    )
    _require_success(completed, "remote preflight artifact persistence")
    downloaded = json.loads(completed.stdout)
    if downloaded != expected:
        raise ValueError("remote preflight artifact round-trip mismatch")
    return downloaded


def persist_and_download_restore_artifacts(
    run_tag,
    *,
    restore,
    source_manifest,
    command_runner=subprocess.run,
):
    run_tag = validate_run_tag(run_tag)
    remote_run_dir = f"{REMOTE_RUN_ROOT}/{run_tag}"
    script = "\n".join([
        "import json,os,pathlib,sys",
        f"root=pathlib.Path({remote_run_dir!r})",
        "payload=json.load(sys.stdin)",
        "def atomic_json(path,value):",
        " temporary=path.with_name('.'+path.name+'.tmp')",
        " with temporary.open('w',encoding='utf-8') as handle:",
        "  json.dump(value,handle,sort_keys=True,separators=(',',':'))",
        "  handle.write('\\n')",
        "  handle.flush()",
        "  os.fsync(handle.fileno())",
        " os.replace(temporary,path)",
        "atomic_json(root/'restore_model_manifest.json',payload['restore'])",
        "atomic_json(root/'source_manifest.json',payload['source_manifest'])",
        "out={",
        " 'restore':json.loads((root/'restore_model_manifest.json').read_text()),",
        " 'source_manifest':json.loads((root/'source_manifest.json').read_text()),",
        "}",
        "print(json.dumps(out,sort_keys=True,separators=(',',':')))",
    ])
    expected = {
        "restore": restore,
        "source_manifest": source_manifest,
    }
    completed = command_runner(
        build_ssh_command([
            contract.REMOTE_PYTHON,
            "-c",
            script,
        ]),
        input=json.dumps(
            expected,
            sort_keys=True,
            separators=(",", ":"),
        ),
        text=True,
        capture_output=True,
    )
    _require_success(completed, "remote restore artifact persistence")
    downloaded = json.loads(completed.stdout)
    if downloaded != expected:
        raise ValueError("remote restore artifact round-trip mismatch")
    return downloaded


def run_remote_model_manifest_restore(
    repo_root,
    run_tag,
    *,
    staged,
    destination,
    command_runner=subprocess.run,
):
    source_manifest = build_source_manifest(repo_root, staged)
    completed = command_runner(
        build_ssh_command([
            "env",
            "CUDA_VISIBLE_DEVICES=",
            contract.REMOTE_PYTHON,
            "-c",
            build_restore_model_manifest_script(run_tag),
        ]),
        text=True,
        capture_output=True,
    )
    _require_success(completed, "remote zero-payload manifest restore")
    record = json.loads(completed.stdout)
    contract.validate_restore_record(record)
    downloaded = persist_and_download_restore_artifacts(
        run_tag,
        restore=record,
        source_manifest=source_manifest,
        command_runner=command_runner,
    )
    destination = Path(destination)
    _atomic_write_json(
        destination / "restore_model_manifest.json",
        downloaded["restore"],
    )
    _atomic_write_json(
        destination / "source_manifest.json",
        downloaded["source_manifest"],
    )
    return downloaded["restore"]


def run_remote_preflight(
    repo_root,
    run_tag,
    *,
    staged,
    destination,
    command_runner=subprocess.run,
):
    source_manifest = build_source_manifest(repo_root, staged)
    script = build_remote_preflight_script(
        run_tag,
        staged["local_file_sha256"],
    )
    completed = command_runner(
        build_ssh_command([
            "env",
            "CUDA_VISIBLE_DEVICES=",
            contract.REMOTE_PYTHON,
            "-c",
            script,
        ]),
        text=True,
        capture_output=True,
    )
    _require_success(completed, "remote read-only preflight")
    record = classify_preflight_payload(json.loads(completed.stdout))
    contract.validate_preflight(record)
    downloaded = persist_and_download_preflight_artifacts(
        run_tag,
        preflight=record,
        source_manifest=source_manifest,
        command_runner=command_runner,
    )
    destination = Path(destination)
    _atomic_write_json(
        destination / "preflight.json",
        downloaded["preflight"],
    )
    _atomic_write_json(
        destination / "source_manifest.json",
        downloaded["source_manifest"],
    )
    return downloaded["preflight"]


def execute_preflight(
    repo_root,
    run_tag,
    *,
    command_runner=subprocess.run,
    stage_function=stage_owned_source,
    audit_function=run_remote_preflight,
):
    run_tag = validate_run_tag(run_tag)
    staged = stage_function(repo_root, run_tag, command_runner)
    destination = Path(repo_root) / LOCAL_RUN_ROOT / run_tag
    return audit_function(
        repo_root,
        run_tag,
        staged=staged,
        destination=destination,
        command_runner=command_runner,
    )


def execute_model_manifest_restore(
    repo_root,
    run_tag,
    *,
    command_runner=subprocess.run,
    stage_function=stage_owned_source,
    restore_function=run_remote_model_manifest_restore,
):
    run_tag = validate_run_tag(run_tag)
    staged = stage_function(repo_root, run_tag, command_runner)
    destination = Path(repo_root) / LOCAL_RUN_ROOT / run_tag
    return restore_function(
        repo_root,
        run_tag,
        staged=staged,
        destination=destination,
        command_runner=command_runner,
    )


def execute_verify_only(
    repo_root,
    run_tag,
    *,
    verifier_function=None,
):
    run_tag = validate_run_tag(run_tag)
    if verifier_function is None:
        verifier_function = _load_local_verifier()
    destination = Path(repo_root) / LOCAL_RUN_ROOT / run_tag
    return verifier_function(destination, write_report=True)


def execute_authorization_only(
    repo_root,
    run_tag,
    *,
    authorization_function=None,
):
    run_tag = validate_run_tag(run_tag)
    if authorization_function is None:
        authorization_function = _load_local_authorization()
    destination = Path(repo_root) / LOCAL_RUN_ROOT / run_tag
    return authorization_function(
        destination,
        owned_source_files=OWNED_SOURCE_FILES,
    )


def read_remote_artifact(
    run_tag,
    relative_path,
    command_runner=subprocess.run,
):
    run_tag = validate_run_tag(run_tag)
    relative = _safe_artifact_path(relative_path)
    remote_run_dir = f"{REMOTE_RUN_ROOT}/{run_tag}"
    remote_path = (
        PurePosixPath(remote_run_dir).joinpath(*relative.parts).as_posix()
    )
    completed = command_runner(
        build_ssh_command(["cat", "--", remote_path]),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    _require_success(completed, f"remote artifact read: {relative}")
    payload = completed.stdout
    if isinstance(payload, str):
        payload = payload.encode("utf-8")
    if not isinstance(payload, bytes):
        raise TypeError("remote artifact reader must return bytes")
    return payload


def _validate_download_manifest(payload):
    try:
        manifest = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid remote manifest JSON: {exc}") from exc
    if manifest.get("schema_version") != contract.SCHEMA_VERSION:
        raise ValueError("remote manifest schema version mismatch")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list):
        raise ValueError("remote manifest artifacts must be a list")
    expected = (
        set(contract.REQUIRED_ARTIFACTS)
        - {
            "manifest.json",
            "independent_verification.json",
            "report.md",
        }
    )
    listed = {}
    for entry in artifacts:
        if not isinstance(entry, dict):
            raise ValueError("remote manifest artifact entry is invalid")
        relative = _safe_artifact_path(entry.get("path"))
        relative_text = relative.as_posix()
        if relative_text in listed:
            raise ValueError(
                f"duplicate remote manifest artifact: {relative_text}"
            )
        size = entry.get("size")
        digest = entry.get("sha256")
        if (
            isinstance(size, bool)
            or not isinstance(size, int)
            or size < 0
        ):
            raise ValueError(
                f"remote manifest artifact size is invalid: {relative_text}"
            )
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(
                character not in "0123456789abcdef"
                for character in digest
            )
        ):
            raise ValueError(
                f"remote manifest artifact SHA256 is invalid: "
                f"{relative_text}"
            )
        listed[relative_text] = entry
    if set(listed) != expected:
        missing = sorted(expected - set(listed))
        extra = sorted(set(listed) - expected)
        if missing:
            raise ValueError(
                f"remote manifest is missing artifact: {missing[0]}"
            )
        raise ValueError(
            f"remote manifest has unexpected artifact: {extra[0]}"
        )
    return manifest, listed


def execute_download_only(
    repo_root,
    run_tag,
    *,
    command_runner=subprocess.run,
    artifact_reader=read_remote_artifact,
    verifier_function=None,
):
    run_tag = validate_run_tag(run_tag)
    destination = Path(repo_root) / LOCAL_RUN_ROOT / run_tag
    if destination.exists():
        raise ValueError("local run directory already exists")
    destination.parent.mkdir(parents=True, exist_ok=True)
    manifest_payload = artifact_reader(
        run_tag,
        "manifest.json",
        command_runner,
    )
    manifest, listed = _validate_download_manifest(manifest_payload)
    temporary = Path(tempfile.mkdtemp(
        dir=destination.parent,
        prefix=f".{run_tag}.",
        suffix=".tmp",
    ))
    try:
        _write_bytes_chunked(
            temporary / "manifest.json",
            manifest_payload,
        )
        for relative_text in sorted(listed):
            entry = listed[relative_text]
            payload = artifact_reader(
                run_tag,
                relative_text,
                command_runner,
            )
            if len(payload) != entry["size"]:
                raise ValueError(
                    f"artifact size mismatch: {relative_text}"
                )
            digest = hashlib.sha256(payload).hexdigest()
            if digest != entry["sha256"]:
                raise ValueError(
                    f"artifact sha256 mismatch: {relative_text}"
                )
            relative = PurePosixPath(relative_text)
            path = temporary.joinpath(*relative.parts)
            _write_bytes_chunked(path, payload)
        os.replace(temporary, destination)
    except BaseException:
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
        raise
    if verifier_function is None:
        verifier_function = _load_local_verifier()
    result = verifier_function(destination, write_report=True)
    if not isinstance(result, dict):
        raise TypeError("local verifier result must be a dictionary")
    if manifest.get("schema_version") != result.get("schema_version"):
        raise ValueError("downloaded run verifier schema mismatch")
    return result


def build_dry_run_plan(repo_root, run_tag):
    run_tag = validate_run_tag(run_tag)
    hashes = _source_hashes(repo_root)
    remote_run_dir = f"{REMOTE_RUN_ROOT}/{run_tag}"
    return {
        "schema_version": contract.SCHEMA_VERSION,
        "mode": "dry-run",
        "execution_authorized": False,
        "remote_target": contract.REMOTE_TARGET,
        "ssh_control_path": SSH_CONTROL_PATH,
        "remote_python": contract.REMOTE_PYTHON,
        "remote_run_dir": remote_run_dir,
        "local_run_dir": str(
            Path(repo_root) / LOCAL_RUN_ROOT / run_tag
        ),
        "model_repository": contract.MODEL_REPOSITORY,
        "model_revision": contract.MODEL_REVISION,
        "cuda_visible_devices": "",
        "case_order_mib": list(contract.CASE_ORDER_MIB),
        "measured_repeats_per_budget": (
            contract.MEASURED_REPEATS_PER_BUDGET
        ),
        "minimum_wall_time_improvement_fraction": (
            contract.MIN_WALL_TIME_IMPROVEMENT_FRACTION
        ),
        "maximum_vmhwm_regression_bytes": (
            contract.MAX_VMHWM_REGRESSION_BYTES
        ),
        "required_artifacts": list(contract.REQUIRED_ARTIFACTS),
        "owned_source_files": list(OWNED_SOURCE_FILES),
        "owned_source_sha256": hashes,
        "source_tree_sha256": _source_tree_sha256(hashes),
        "preflight_command": build_ssh_command([
            contract.REMOTE_PYTHON,
            f"{remote_run_dir}/source/"
            "tools/run_qwen35_real_checkpoint_load_gate_remote.py",
            "preflight",
            "--run-tag",
            run_tag,
        ]),
        "run_command": build_ssh_command([
            "env",
            "CUDA_VISIBLE_DEVICES=",
            contract.REMOTE_PYTHON,
            f"{remote_run_dir}/source/"
            "tools/qwen35_real_checkpoint_load_worker.py",
            "--run-dir",
            remote_run_dir,
        ]),
        "subprocess_count": 0,
        "ssh_count": 0,
        "payload_open_count": 0,
        "notes": [
            "No subprocess, SSH, or payload access occurs in dry-run mode.",
            (
                "Read-only preflight and local verify-only are authorized; "
                "manifest-bound download-only is authorized; run remains "
                "fail-closed until worker execution is authorized."
            ),
            (
                "Local authorization-only checks READY, exact current "
                "source hashes, and Git cleanliness without SSH or payload "
                "access."
            ),
        ],
    }


def _atomic_write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(
                payload,
                handle,
                indent=2,
                ensure_ascii=False,
                allow_nan=False,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


def _write_bytes_chunked(path, payload, chunk_bytes=1 << 20):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        view = memoryview(payload)
        for offset in range(0, len(view), chunk_bytes):
            handle.write(view[offset:offset + chunk_bytes])
        handle.flush()
        os.fsync(handle.fileno())


def _parse_arguments():
    parser = argparse.ArgumentParser(
        description=(
            "Fail-closed remote safety harness for a future Qwen3.5 "
            "real checkpoint load comparison."
        )
    )
    parser.add_argument("mode", choices=MODES)
    parser.add_argument("--run-tag", required=True)
    parser.add_argument("--output-json", type=Path)
    return parser.parse_args()


def main():
    arguments = _parse_arguments()
    reject_unimplemented_execution(arguments.mode)
    if arguments.mode == "preflight":
        plan = execute_preflight(ROOT, arguments.run_tag)
    elif arguments.mode == "restore-model-manifest":
        plan = execute_model_manifest_restore(ROOT, arguments.run_tag)
    elif arguments.mode == "verify-only":
        plan = execute_verify_only(ROOT, arguments.run_tag)
    elif arguments.mode == "authorization-only":
        plan = execute_authorization_only(ROOT, arguments.run_tag)
    elif arguments.mode == "download-only":
        plan = execute_download_only(ROOT, arguments.run_tag)
    else:
        plan = build_dry_run_plan(ROOT, arguments.run_tag)
    if arguments.output_json is not None:
        _atomic_write_json(arguments.output_json, plan)
    print(
        json.dumps(
            plan,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
    )


ROOT = Path(__file__).resolve().parents[1]


if __name__ == "__main__":
    main()
