from __future__ import annotations

import argparse
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import shlex
import subprocess
import sys
import tarfile


REMOTE_TARGET = "sitian@10.232.195.203"
REMOTE_PYTHON = (
    "/data00/home/sitian/sitian-workspace01/tllm/env/bin/python"
)
REMOTE_GATE_ROOT = (
    "/data00/home/sitian/sitian-workspace01/tllm/"
    "qwen35-tp4-root-logit-tests"
)
FROZEN_SOURCE_TAG = "qwen35-tp4-source-prep-20260729-170818"
FROZEN_SOURCE_TREE_SHA256 = (
    "ec19a8fa68abfba72e9594bdd1e05428b0add9169d3dbdde24190686c013411f"
)
FROZEN_SOURCE_ROOT = (
    f"{REMOTE_GATE_ROOT}/{FROZEN_SOURCE_TAG}/source"
)
FROZEN_PREFLIGHT = (
    f"{FROZEN_SOURCE_ROOT}/tools/"
    "qwen35_tp4_real_root_logit_correctness_preflight.py"
)
FROZEN_VERIFIER = (
    f"{FROZEN_SOURCE_ROOT}/tools/"
    "verify_qwen35_tp4_real_root_logit_correctness_gate.py"
)
FROZEN_MANIFEST = f"{FROZEN_SOURCE_ROOT}/source_manifest.input.json"
LOCAL_RUN_ROOT = Path("experiments/qwen35_hybrid_state")
RUN_TAG_PATTERN = re.compile(r"^[A-Za-z0-9_-]+$")
EXACT_ARTIFACT_NAMES = {
    "tp4_real_root_logit_correctness.json",
    "reference_logits.pt",
    "native_rank0_logits.pt",
    "rank_evidence.json",
    "source_manifest.json",
}
NATIVE_SMOKE_ARTIFACT_NAMES = {
    "native_smoke.json",
    "native_rank0_logits.pt",
    "rank_evidence.json",
}
MODES = (
    "preflight",
    "native-smoke",
    "run",
    "download-only",
    "verify-only",
    "authority",
)


def validate_run_tag(value) -> str:
    text = str(value)
    if not RUN_TAG_PATTERN.fullmatch(text):
        raise ValueError("run tag must match [A-Za-z0-9_-]+")
    return text


def remote_run_dir(run_tag) -> str:
    return f"{REMOTE_GATE_ROOT}/{validate_run_tag(run_tag)}"


def require_new_local_run_dir(repo_root, run_tag) -> Path:
    destination = (
        Path(repo_root)
        / LOCAL_RUN_ROOT
        / validate_run_tag(run_tag)
    )
    if destination.exists():
        raise ValueError("local run directory already exists")
    return destination


def build_ssh_command(remote_arguments) -> list[str]:
    remote_command = shlex.join([str(value) for value in remote_arguments])
    return [
        "ssh",
        "-o",
        "BatchMode=yes",
        "-o",
        "ControlMaster=no",
        "-o",
        "ConnectTimeout=20",
        "-o",
        "ServerAliveInterval=30",
        "-o",
        "ServerAliveCountMax=3",
        REMOTE_TARGET,
        remote_command,
    ]


def _require_success(result, context):
    if result.returncode != 0:
        detail = result.stderr or result.stdout or ""
        if isinstance(detail, bytes):
            detail = detail.decode("utf-8", errors="replace")
        raise RuntimeError(f"{context} failed: {str(detail).strip()}")
    return result


def _run(command, **kwargs):
    environment = dict(os.environ)
    environment["KRB5CCNAME"] = (
        "FILE:/Users/bytedance/krb5cc_sitian"
    )
    return subprocess.run(
        command,
        check=False,
        env=environment,
        **kwargs,
    )


def classify_preflight_payload(payload) -> dict[str, object]:
    if not isinstance(payload, dict):
        raise ValueError("preflight payload must be a dictionary")
    if payload.get("source_tree_sha256") != FROZEN_SOURCE_TREE_SHA256:
        raise ValueError("frozen source tree identity mismatch")
    rows = payload.get("rows")
    if not isinstance(rows, list):
        raise ValueError("preflight rows are invalid")
    if payload.get("eligible") is False:
        error = payload.get("error")
        if error != "four eligible GPUs are required":
            raise ValueError("preflight blocked reason is invalid")
        return {
            "status": "BLOCKED",
            "error": error,
            "source_tree_sha256": payload["source_tree_sha256"],
            "selected": [],
            "rows": rows,
        }
    if payload.get("eligible") is not True:
        raise ValueError("preflight eligibility is invalid")
    selected = payload.get("selected")
    if not isinstance(selected, list) or len(selected) != 4:
        raise ValueError("preflight must contain four selected GPUs")
    if [row.get("rank") for row in selected] != [0, 1, 2, 3]:
        raise ValueError("preflight selected rank inventory is invalid")
    if len({row.get("gpu_index") for row in selected}) != 4:
        raise ValueError("preflight selected GPU indices are invalid")
    if len({row.get("gpu_uuid") for row in selected}) != 4:
        raise ValueError("preflight selected GPU UUIDs are invalid")
    if any(row.get("compute_processes") != [] for row in selected):
        raise ValueError("preflight selected GPU is not idle")
    return {
        "status": "READY",
        "source_tree_sha256": payload["source_tree_sha256"],
        "selected": selected,
        "rows": rows,
    }


def _preflight_script() -> str:
    return "\n".join([
        "import importlib.util,json,sys",
        "from pathlib import Path",
        f"path=Path({FROZEN_PREFLIGHT!r})",
        "spec=importlib.util.spec_from_file_location('tp4_runner_preflight',path)",
        "module=importlib.util.module_from_spec(spec)",
        "sys.modules[spec.name]=module",
        "spec.loader.exec_module(module)",
        f"manifest=json.loads(Path({FROZEN_MANIFEST!r}).read_text())",
        "source_tree=manifest.get('source_tree_sha256')",
        "rows=module._query_tp4_gpu_resources()",
        "try:",
        " selected=module.select_tp4_gpu_resources(rows)",
        " payload={'eligible':True,'selected':list(selected),'rows':list(rows),'source_tree_sha256':source_tree}",
        "except ValueError as error:",
        " payload={'eligible':False,'error':str(error),'rows':list(rows),'source_tree_sha256':source_tree}",
        "print(json.dumps(payload,sort_keys=True,separators=(',',':')))",
    ])


def execute_preflight(
    *,
    run_tag: str,
    repo_root,
    command_runner=_run,
) -> dict[str, object]:
    destination = require_new_local_run_dir(repo_root, run_tag)
    result = command_runner(
        build_ssh_command([
            REMOTE_PYTHON,
            "-c",
            _preflight_script(),
        ]),
        text=True,
        capture_output=True,
    )
    _require_success(result, "remote TP4 resource preflight")
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError as error:
        raise ValueError("remote preflight JSON is invalid") from error
    classified = classify_preflight_payload(payload)
    destination.mkdir(parents=True)
    evidence = {
        "run_tag": validate_run_tag(run_tag),
        "frozen_source_tag": FROZEN_SOURCE_TAG,
        "frozen_source_tree_sha256": FROZEN_SOURCE_TREE_SHA256,
        **classified,
    }
    (destination / "remote_resource_preflight.json").write_text(
        json.dumps(evidence, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    return evidence


def _remote_inventory_script(run_tag) -> str:
    directory = remote_run_dir(run_tag)
    return "\n".join([
        "import json,pathlib",
        f"root=pathlib.Path({directory!r})",
        "if not root.is_dir(): raise SystemExit('remote run directory missing')",
        "names=sorted(path.name for path in root.iterdir())",
        "print(json.dumps(names,separators=(',',':')))",
    ])


def execute_run(
    *,
    run_tag: str,
    repo_root,
    command_runner=_run,
) -> dict[str, object]:
    destination = (
        Path(repo_root) / LOCAL_RUN_ROOT / validate_run_tag(run_tag)
    )
    preflight_path = destination / "remote_resource_preflight.json"
    if not preflight_path.is_file():
        raise ValueError("preflight evidence is missing")
    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    if preflight.get("status") != "READY":
        raise ValueError("preflight is not ready")
    remote = remote_run_dir(run_tag)
    existence = command_runner(
        build_ssh_command([
            "bash",
            "-c",
            f"test ! -e {shlex.quote(remote)}",
        ]),
        text=True,
        capture_output=True,
    )
    _require_success(existence, "remote run uniqueness check")
    result = command_runner(
        build_ssh_command([
            "env",
            f"PYTHONPATH={FROZEN_SOURCE_ROOT}",
            REMOTE_PYTHON,
            FROZEN_PREFLIGHT,
            "run",
            "--run-dir",
            remote,
            "--run-tag",
            validate_run_tag(run_tag),
            "--source-manifest",
            FROZEN_MANIFEST,
        ]),
        text=True,
        capture_output=True,
    )
    _require_success(result, "remote TP4 authority run")
    inventory_result = command_runner(
        build_ssh_command([
            REMOTE_PYTHON,
            "-c",
            _remote_inventory_script(run_tag),
        ]),
        text=True,
        capture_output=True,
    )
    _require_success(inventory_result, "remote artifact inventory")
    inventory = set(json.loads(inventory_result.stdout))
    if inventory != EXACT_ARTIFACT_NAMES:
        raise ValueError("remote exact-five artifact inventory is invalid")
    evidence = {
        "status": "REMOTE_PASS",
        "run_tag": validate_run_tag(run_tag),
        "remote_run_dir": remote,
        "artifact_names": sorted(inventory),
    }
    (destination / "remote_run.json").write_text(
        json.dumps(evidence, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    return evidence


def _native_smoke_script(run_tag) -> str:
    tag = validate_run_tag(run_tag)
    remote = remote_run_dir(tag)
    return "\n".join([
        "import hashlib,importlib.util,json,os,shutil,sys",
        "from pathlib import Path",
        f"gate_path=Path({FROZEN_PREFLIGHT!r})",
        f"manifest_path=Path({FROZEN_MANIFEST!r})",
        f"source_root=Path({FROZEN_SOURCE_ROOT!r})",
        f"run_dir=Path({remote!r})",
        f"run_tag={tag!r}",
        f"expected_source_tree={FROZEN_SOURCE_TREE_SHA256!r}",
        "spec=importlib.util.spec_from_file_location('tp4_native_smoke_gate',gate_path)",
        "module=importlib.util.module_from_spec(spec)",
        "sys.modules[spec.name]=module",
        "spec.loader.exec_module(module)",
        "if 'internal-native-rank' not in gate_path.read_text(): raise SystemExit('frozen native worker mode missing')",
        "manifest=json.loads(manifest_path.read_text())",
        "if manifest.get('source_tree_sha256')!=expected_source_tree: raise SystemExit('frozen source tree identity mismatch')",
        "work_dir=run_dir.parent/f'.{run_tag}.native-smoke.work'",
        "publish_dir=run_dir.parent/f'.{run_tag}.native-smoke.publish'",
        "if run_dir.exists() or work_dir.exists() or publish_dir.exists(): raise SystemExit('native smoke path already exists')",
        "work_dir.mkdir(parents=True)",
        "selected=module.select_tp4_gpu_resources(module._query_tp4_gpu_resources())",
        "dist_port,master_port=module.fresh_port_pair()",
        "rendezvous=f'tcp://127.0.0.1:{dist_port}'",
        "nonce=hashlib.sha256(f'{run_tag}:{dist_port}:{master_port}:native-smoke'.encode()).hexdigest()",
        "base_environment=dict(os.environ)",
        "base_environment.update({'PYTHONPATH':os.fspath(source_root),'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','HF_DATASETS_OFFLINE':'1','TOKENIZERS_PARALLELISM':'false'})",
        "def process_factory(**kwargs):",
        " rank=kwargs['rank']",
        " return module.make_native_rank_subprocess(**kwargs,script_path=gate_path,python_executable=sys.executable,work_dir=work_dir,rank_output=work_dir/f'rank-{rank}.json.partial',logits_output=(work_dir/'native_rank0_logits.pt.partial' if rank==0 else None))",
        "launched=module.launch_native_rank_group(selected_gpus=selected,rendezvous=rendezvous,process_group_nonce=nonce,tinyvllm_dist_port=dist_port,master_port=master_port,process_factory=process_factory,timeout_seconds=1800,pid_alive=module._pid_alive,base_environment=base_environment)",
        "persisted=tuple(module._load_json(work_dir/f'rank-{rank}.json.partial',label=f'rank {rank}') for rank in range(4))",
        "ranks=module.validate_rank_evidence(module.bind_launched_rank_evidence(launched,persisted))",
        "cases=module._TP4_CONTRACT.prompt_cases()",
        "case_ids=tuple(case.case_id for case in cases)",
        "logits=module._validate_tp4_tensor_map(module._load_tensor_map(work_dir/'native_rank0_logits.pt.partial',label='native rank zero'),case_ids=case_ids,label='native rank zero')",
        "publish_dir.mkdir()",
        "smoke={'schema_version':'qwen35.tp4-native-only-smoke.v1','run_tag':run_tag,'classification':'NATIVE_SMOKE_PASS','source_tree_sha256':expected_source_tree,'case_ids':list(case_ids),'rank_count':4,'gpu_indices':[row['gpu_index'] for row in ranks],'gpu_uuids':[row['gpu_uuid'] for row in ranks],'reference_worker_started':False,'forbidden_counters':{'engine':0,'model_runner':0,'scheduler':0,'sampler':0,'generation':0},'claim_boundary':'native-only TP4 distributed execution and cleanup smoke; no reference comparison, correctness authority, performance, cache, memory, compression, or quality claim'}",
        "(publish_dir/'native_smoke.json').write_text(json.dumps(smoke,sort_keys=True,indent=2)+'\\n')",
        "module.torch.save(logits,publish_dir/'native_rank0_logits.pt')",
        "(publish_dir/'rank_evidence.json').write_text(json.dumps(list(ranks),sort_keys=True,indent=2)+'\\n')",
        "os.replace(publish_dir,run_dir)",
        "shutil.rmtree(work_dir)",
        "result={'classification':'NATIVE_SMOKE_PASS','run_tag':run_tag,'remote_run_dir':os.fspath(run_dir),'source_tree_sha256':expected_source_tree,'artifact_names':sorted(path.name for path in run_dir.iterdir())}",
        "print(json.dumps(result,sort_keys=True,separators=(',',':')))",
    ])


def execute_native_smoke_run(
    *,
    run_tag: str,
    repo_root,
    command_runner=_run,
) -> dict[str, object]:
    destination = (
        Path(repo_root) / LOCAL_RUN_ROOT / validate_run_tag(run_tag)
    )
    preflight_path = destination / "remote_resource_preflight.json"
    if not preflight_path.is_file():
        raise ValueError("preflight evidence is missing")
    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    if preflight.get("status") != "READY":
        raise ValueError("preflight is not ready")
    result = command_runner(
        build_ssh_command([
            "env",
            f"PYTHONPATH={FROZEN_SOURCE_ROOT}",
            REMOTE_PYTHON,
            "-c",
            _native_smoke_script(run_tag),
        ]),
        text=True,
        capture_output=True,
    )
    _require_success(result, "remote TP4 native-only smoke")
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError as error:
        raise ValueError("remote native smoke JSON is invalid") from error
    if (
        payload.get("classification") != "NATIVE_SMOKE_PASS"
        or payload.get("source_tree_sha256")
        != FROZEN_SOURCE_TREE_SHA256
        or set(payload.get("artifact_names", ()))
        != NATIVE_SMOKE_ARTIFACT_NAMES
    ):
        raise ValueError("remote native smoke evidence is invalid")
    (destination / "remote_native_smoke.json").write_text(
        json.dumps(payload, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    return payload


def execute_native_smoke(
    *,
    run_tag: str,
    repo_root,
    preflight=execute_preflight,
    run_smoke=execute_native_smoke_run,
) -> dict[str, object]:
    resource = preflight(run_tag=run_tag, repo_root=repo_root)
    if resource.get("status") != "READY":
        return resource
    return run_smoke(run_tag=run_tag, repo_root=repo_root)


def extract_exact_five_tar(payload: bytes, destination) -> tuple[Path, ...]:
    directory = Path(destination)
    if directory.exists():
        raise ValueError("download destination already exists")
    try:
        archive = tarfile.open(fileobj=io.BytesIO(payload), mode="r:")
    except tarfile.TarError as error:
        raise ValueError("artifact tar is invalid") from error
    with archive:
        members = archive.getmembers()
        names = []
        for member in members:
            path = PurePosixPath(member.name)
            if (
                not member.isfile()
                or path.is_absolute()
                or len(path.parts) != 1
                or ".." in path.parts
            ):
                raise ValueError("artifact tar path is invalid")
            names.append(path.name)
        if set(names) != EXACT_ARTIFACT_NAMES or len(names) != len(
            EXACT_ARTIFACT_NAMES
        ):
            raise ValueError("artifact tar inventory is invalid")
        directory.mkdir(parents=True)
        try:
            for member in members:
                source = archive.extractfile(member)
                if source is None:
                    raise ValueError("artifact tar member is invalid")
                (directory / member.name).write_bytes(source.read())
        except BaseException:
            for path in directory.iterdir():
                path.unlink()
            directory.rmdir()
            raise
    return tuple(
        directory / name for name in sorted(EXACT_ARTIFACT_NAMES)
    )


def _remote_tar_command(run_tag) -> list[str]:
    remote = remote_run_dir(run_tag)
    names = sorted(EXACT_ARTIFACT_NAMES)
    script = (
        "set -euo pipefail; "
        f"cd {shlex.quote(remote)}; "
        f"test \"$(find . -mindepth 1 -maxdepth 1 | wc -l)\" -eq 5; "
        f"test \"$(find . -mindepth 1 -maxdepth 1 -type f | wc -l)\" -eq 5; "
        + " ".join(
            f"test -f {shlex.quote(name)};" for name in names
        )
        + " tar -cf - "
        + " ".join(shlex.quote(name) for name in names)
    )
    return build_ssh_command(["bash", "-c", script])


def execute_download(
    *,
    run_tag: str,
    repo_root,
    command_runner=_run,
) -> dict[str, object]:
    run_root = (
        Path(repo_root) / LOCAL_RUN_ROOT / validate_run_tag(run_tag)
    )
    if not (run_root / "remote_run.json").is_file():
        raise ValueError("remote run evidence is missing")
    artifact_dir = run_root / "artifacts"
    result = command_runner(
        _remote_tar_command(run_tag),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    _require_success(result, "remote exact-five download")
    paths = extract_exact_five_tar(result.stdout, artifact_dir)
    evidence = {
        "status": "DOWNLOADED",
        "artifact_names": [path.name for path in paths],
    }
    (run_root / "download.json").write_text(
        json.dumps(evidence, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    return evidence


def _build_local_artifact_tar(artifact_dir: Path) -> bytes:
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w") as archive:
        for name in sorted(EXACT_ARTIFACT_NAMES):
            path = artifact_dir / name
            if not path.is_file():
                raise ValueError("downloaded exact-five inventory is invalid")
            info = archive.gettarinfo(os.fspath(path), arcname=name)
            info.uid = 0
            info.gid = 0
            info.uname = ""
            info.gname = ""
            info.mtime = 0
            with path.open("rb") as source:
                archive.addfile(info, source)
    return stream.getvalue()


def _stream_verify_script() -> str:
    return "\n".join([
        "import importlib.util,json,sys,tarfile,tempfile",
        "from pathlib import Path",
        f"verifier_path=Path({FROZEN_VERIFIER!r})",
        f"source_root=Path({FROZEN_SOURCE_ROOT!r})",
        "with tempfile.TemporaryDirectory(prefix='qwen35-tp4-local-download-verify-') as tmp:",
        " root=Path(tmp)",
        " with tarfile.open(fileobj=sys.stdin.buffer,mode='r|') as archive:",
        "  for member in archive:",
        "   if not member.isfile() or '/' in member.name or member.name.startswith('.'):",
        "    raise SystemExit('unsafe streamed artifact')",
        "   source=archive.extractfile(member)",
        "   (root/member.name).write_bytes(source.read())",
        " spec=importlib.util.spec_from_file_location('tp4_stream_verifier',verifier_path)",
        " module=importlib.util.module_from_spec(spec)",
        " spec.loader.exec_module(module)",
        " result=module.verify_run(root,source_root=source_root)",
        " print(json.dumps(result,sort_keys=True,separators=(',',':')))",
    ])


def verify_downloaded_artifacts(
    artifact_dir,
    *,
    command_runner=_run,
) -> dict[str, object]:
    artifact_dir = Path(artifact_dir)
    if (
        not artifact_dir.is_dir()
        or {path.name for path in artifact_dir.iterdir()}
        != EXACT_ARTIFACT_NAMES
    ):
        raise ValueError("downloaded exact-five inventory is invalid")
    command = build_ssh_command([
        REMOTE_PYTHON,
        "-c",
        _stream_verify_script(),
    ])
    artifact_tar = _build_local_artifact_tar(artifact_dir)
    for _ in range(3):
        result = command_runner(
            command,
            input=artifact_tar,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        if result.returncode == 0:
            break
    _require_success(result, "independent downloaded-artifact verification")
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError as error:
        raise ValueError("independent verifier JSON is invalid") from error
    if payload.get("classification") != "PASS":
        raise ValueError("independent verifier did not return PASS")
    return payload


def execute_verify(
    *,
    run_tag: str,
    repo_root,
    command_runner=_run,
) -> dict[str, object]:
    run_root = (
        Path(repo_root) / LOCAL_RUN_ROOT / validate_run_tag(run_tag)
    )
    payload = verify_downloaded_artifacts(
        run_root / "artifacts",
        command_runner=command_runner,
    )
    (run_root / "independent_verification.json").write_text(
        json.dumps(payload, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    return payload


def execute_authority(
    *,
    run_tag: str,
    repo_root,
    preflight=execute_preflight,
    run=execute_run,
    download=execute_download,
    verify=execute_verify,
) -> dict[str, object]:
    resource = preflight(run_tag=run_tag, repo_root=repo_root)
    if resource.get("status") != "READY":
        return resource
    run(run_tag=run_tag, repo_root=repo_root)
    download(run_tag=run_tag, repo_root=repo_root)
    result = verify(run_tag=run_tag, repo_root=repo_root)
    if result.get("classification") != "PASS":
        raise ValueError("TP4 authority verification did not pass")
    return result


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=MODES)
    parser.add_argument("--run-tag", required=True)
    parser.add_argument(
        "--repo-root",
        default=os.fspath(Path(__file__).resolve().parents[1]),
    )
    return parser


def result_exit_code(mode: str, result) -> int:
    if mode == "preflight":
        return 0 if result.get("status") == "READY" else 2
    if mode in ("native-smoke", "authority"):
        if (
            mode == "native-smoke"
            and result.get("classification") == "NATIVE_SMOKE_PASS"
        ):
            return 0
        if result.get("classification") == "PASS":
            return 0
        if result.get("status") == "BLOCKED":
            return 2
        return 1
    return 0


def main(argv=None) -> int:
    arguments = _build_parser().parse_args(argv)
    kwargs = {
        "run_tag": arguments.run_tag,
        "repo_root": Path(arguments.repo_root),
    }
    functions = {
        "preflight": execute_preflight,
        "native-smoke": execute_native_smoke,
        "run": execute_run,
        "download-only": execute_download,
        "verify-only": execute_verify,
        "authority": execute_authority,
    }
    result = functions[arguments.mode](**kwargs)
    print(json.dumps(result, sort_keys=True))
    return result_exit_code(arguments.mode, result)


if __name__ == "__main__":
    raise SystemExit(main())
