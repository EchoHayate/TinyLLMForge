#!/usr/bin/env python3
"""Strict-clean controller for the TP4 segmented-capture census."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import sys
import tempfile
import time

if __package__:
    from tools.run_qwen38_tp4_communication_profile import (
        DEFAULT_SSH_TARGET,
        MAX_GPU_MEMORY_USED_MIB,
        MAX_GPU_UTILIZATION_PERCENT,
        build_ssh_argv,
        query_local_kerberos,
        query_remote_gpu_inventory,
        validate_selected_gpu_processes,
        wait_for_strict_clean_gpus,
    )
    from tools.run_tp4_decode_replay import (
        ProductionAdapter as _DecodeReplayAdapter,
    )
    from tools.verify_tp4_segmented_capture_census import verify_bundle
else:
    from run_qwen38_tp4_communication_profile import (
        DEFAULT_SSH_TARGET,
        MAX_GPU_MEMORY_USED_MIB,
        MAX_GPU_UTILIZATION_PERCENT,
        build_ssh_argv,
        query_local_kerberos,
        query_remote_gpu_inventory,
        validate_selected_gpu_processes,
        wait_for_strict_clean_gpus,
    )
    from run_tp4_decode_replay import (
        ProductionAdapter as _DecodeReplayAdapter,
    )
    from verify_tp4_segmented_capture_census import verify_bundle


REMOTE_BASE = (
    "/data00/home/sitian/tinyllmforge-workspaces/"
    "command-timeline-20260818"
)
REMOTE_ROOT = f"{REMOTE_BASE}/tp4-segmented-capture-census"
MODEL_REPOSITORY = "Qwen/Qwen3.8-27B"
MODEL_REVISION = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
MODEL_ROOT = (
    f"{REMOTE_BASE}/models/Qwen3.8-27B/snapshots/{MODEL_REVISION}"
)
SOURCE_SCHEMA = "tinyllmforge.tp4-segmented-capture-source.v1"
PLAN_SCHEMA = "tinyllmforge.tp4-segmented-capture-plan.v1"
RUN_TAG_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
WORLD_SIZE = 4
DEFAULT_REMOTE_PYTHON = "/data00/home/sitian/tllm/env/bin/python"
DEFAULT_COMMAND_TIMEOUT_S = 21_600
DEFAULT_GPU_WAIT_TIMEOUT_S = 21_600
DEFAULT_GPU_POLL_INTERVAL_S = 15
DEFAULT_RETRY_COUNT = 3
KERBEROS_GUARD_MARGIN_S = 900


def _validate_run_tag(run_tag: object) -> str:
    if (
        not isinstance(run_tag, str)
        or not RUN_TAG_PATTERN.fullmatch(run_tag)
        or ".." in run_tag
    ):
        raise ValueError("run tag is invalid")
    return run_tag


def _below(path: str, root: str) -> bool:
    candidate = PurePosixPath(path)
    approved = PurePosixPath(root)
    return candidate.is_absolute() and candidate.is_relative_to(approved)


def _matches_exact_tag_identity(
    command: bytes,
    environment: bytes,
    *,
    run_tag: str,
    attempt_root: str,
) -> bool:
    tag_entry = f"TINYLLMFORGE_RUN_TAG={run_tag}".encode("utf-8")
    attempt_bytes = attempt_root.encode("utf-8")
    matched_command = any(
        argument == attempt_bytes
        or argument.startswith(attempt_bytes + b"/")
        for argument in command.split(b"\0")
    )
    return (
        matched_command
        or tag_entry in environment.split(b"\0")
    )


def _validate_source_identity(
    source_identity: object,
    *,
    run_tag: str,
) -> dict:
    if (
        not isinstance(source_identity, dict)
        or source_identity.get("schema_version") != SOURCE_SCHEMA
        or source_identity.get("run_tag") != run_tag
        or source_identity.get("model_repository") != MODEL_REPOSITORY
        or source_identity.get("model_revision") != MODEL_REVISION
        or not re.fullmatch(
            r"[0-9a-f]{40}",
            str(source_identity.get("source_revision", "")),
        )
        or not re.fullmatch(
            r"[0-9a-f]{64}",
            str(source_identity.get("source_tree_sha256", "")),
        )
    ):
        raise ValueError("source identity is invalid")
    return dict(source_identity)


def _normalize_gpus(selected_gpus: object) -> list[dict]:
    if not isinstance(selected_gpus, list) or len(selected_gpus) != WORLD_SIZE:
        raise ValueError("strict_clean requires exactly four GPUs")
    normalized = []
    indices = set()
    uuids = set()
    for rank, gpu in enumerate(selected_gpus):
        if not isinstance(gpu, dict):
            raise ValueError("strict_clean GPU inventory is invalid")
        index = gpu.get("gpu_index", gpu.get("index"))
        uuid = gpu.get("gpu_uuid", gpu.get("uuid"))
        if (
            isinstance(index, bool)
            or not isinstance(index, int)
            or index < 0
            or not isinstance(uuid, str)
            or not uuid
            or index in indices
            or uuid in uuids
            or gpu.get("memory_used_mib", 0)
            > MAX_GPU_MEMORY_USED_MIB
            or gpu.get("utilization_percent", 0)
            > MAX_GPU_UTILIZATION_PERCENT
            or gpu.get("compute_processes", []) != []
        ):
            raise ValueError("strict_clean GPU inventory is invalid")
        indices.add(index)
        uuids.add(uuid)
        normalized.append(dict(gpu))
    return normalized


def build_plan(
    *,
    run_tag: str,
    source_identity: dict,
    selected_gpus: list[dict],
    admission_mode: str,
) -> dict:
    run_tag = _validate_run_tag(run_tag)
    if admission_mode != "strict_clean":
        raise ValueError("strict_clean admission is required")
    source = _validate_source_identity(
        source_identity,
        run_tag=run_tag,
    )
    selected = _normalize_gpus(selected_gpus)
    attempt_root = f"{REMOTE_ROOT}/{run_tag}"
    runtime_root = f"{attempt_root}/runtime"
    paths = {
        "attempt_root": attempt_root,
        "source_root": f"{attempt_root}/source",
        "raw_root": f"{attempt_root}/raw",
        "bundle_root": f"{attempt_root}/final_bundle",
        "controller_root": f"{attempt_root}/controller",
        "worker_stdout_path": (
            f"{attempt_root}/controller/worker.stdout"
        ),
        "worker_stderr_path": (
            f"{attempt_root}/controller/worker.stderr"
        ),
        "remote_verification_path": (
            f"{attempt_root}/controller/"
            "remote_independent_verification.json"
        ),
    }
    environment = {
        "TMPDIR": f"{runtime_root}/tmp",
        "XDG_CACHE_HOME": f"{runtime_root}/cache/xdg",
        "HF_HOME": f"{runtime_root}/cache/huggingface",
        "TRANSFORMERS_CACHE": (
            f"{runtime_root}/cache/huggingface/transformers"
        ),
        "TORCH_EXTENSIONS_DIR": (
            f"{runtime_root}/cache/torch-extensions"
        ),
        "CUDA_CACHE_PATH": f"{runtime_root}/cache/cuda",
    }
    if (
        not all(_below(path, REMOTE_ROOT) for path in paths.values())
        or not all(
            _below(path, attempt_root) for path in paths.values()
        )
        or not all(
            _below(path, attempt_root)
            for path in environment.values()
        )
    ):
        raise ValueError("remote path escapes the large mounted root")
    return {
        "schema_version": PLAN_SCHEMA,
        "run_tag": run_tag,
        "admission_mode": admission_mode,
        "strict_clean": True,
        "source_identity": source,
        "source_revision": source["source_revision"],
        "source_tree_sha256": source["source_tree_sha256"],
        "model_root": MODEL_ROOT,
        "selected_gpus": selected,
        "selected_gpu_indices": [
            gpu["gpu_index"] for gpu in selected
        ],
        "paths": paths,
        "environment": environment,
        "process_environment": {
            "PYTHONNOUSERSITE": "1",
            "PYTHONPATH": (
                f"{paths['source_root']}:"
                f"{paths['source_root']}/tools"
            ),
            "TINYLLMFORGE_RUN_TAG": run_tag,
        },
    }


def _execute_attempt(
    *,
    plan: dict,
    adapter: object,
    preflight: dict,
) -> dict:
    launch = None
    operation_error = None
    payload = {}
    try:
        admission = adapter.gpu_admission(plan, preflight)
        if (
            not isinstance(admission, dict)
            or admission.get("classification") != "READY"
        ):
            raise RuntimeError("strict-clean admission was lost")
        launch = adapter.launch(plan, admission)
        waited = adapter.wait(plan, launch)
        downloaded = adapter.download(plan, waited)
        remote = adapter.remote_verify(plan, downloaded)
        local = adapter.local_verify(plan, downloaded)
        if (
            not isinstance(remote, dict)
            or not isinstance(local, dict)
            or remote != local
        ):
            raise RuntimeError("independent verifiers disagree")
        payload = {
            "classification": local["classification"],
            "plan": plan,
            "preflight": preflight,
            "admission": admission,
            "launch": launch,
            "wait": waited,
            "download": downloaded,
            "remote_verification": remote,
            "local_verification": local,
        }
    except BaseException as error:
        operation_error = error
    cleanup_error = None
    cleanup = None
    try:
        cleanup = adapter.validate_cleanup(plan, launch)
        final_scans = cleanup.get(
            "final_exact_tag_scans",
            cleanup.get("exact_tag_scans", []),
        )
        if (
            not isinstance(cleanup, dict)
            or cleanup.get("classification") != "CLEAN"
            or any(final_scans)
        ):
            raise RuntimeError("exact-tag cleanup is not clean")
    except BaseException as error:
        cleanup_error = error
    if operation_error is not None:
        if cleanup_error is not None:
            raise operation_error from cleanup_error
        raise operation_error
    if cleanup_error is not None:
        raise cleanup_error
    payload["cleanup"] = cleanup
    return payload


def monitor_and_run(
    *,
    run_tag: str,
    admission_mode: str,
    gpu_monitor: object,
    adapter: object,
) -> dict:
    run_tag = _validate_run_tag(run_tag)
    if admission_mode != "strict_clean":
        raise ValueError("strict_clean admission is required")
    if not callable(gpu_monitor):
        raise ValueError("GPU monitor is invalid")
    seed = {"run_tag": run_tag, "remote_root": REMOTE_ROOT}
    source = adapter.freeze_source(seed)
    source = _validate_source_identity(source, run_tag=run_tag)
    preflight = adapter.ssh_storage_preflight(seed, source)
    if (
        not isinstance(preflight, dict)
        or preflight.get("classification") != "PASS"
        or preflight.get("attempt_exists") is not False
        or preflight.get("remote_root") != REMOTE_ROOT
    ):
        raise RuntimeError("SSH/storage preflight rejected the run")
    monitor = gpu_monitor()
    if (
        not isinstance(monitor, dict)
        or monitor.get("classification") != "READY"
        or not isinstance(monitor.get("selected_gpus"), list)
    ):
        raise RuntimeError("four strict-clean GPUs were not admitted")
    plan = build_plan(
        run_tag=run_tag,
        source_identity=source,
        selected_gpus=monitor["selected_gpus"],
        admission_mode=admission_mode,
    )
    return _execute_attempt(
        plan=plan,
        adapter=adapter,
        preflight=preflight,
    )


def _atomic_write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
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
    temporary.replace(path)


def _capture_source_identity(run_tag: str) -> dict:
    root = Path(__file__).resolve().parents[1]

    def git(*arguments):
        result = subprocess.run(
            ["git", "-C", str(root), *arguments],
            text=True,
            capture_output=True,
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError(result.stderr or "git command failed")
        return result.stdout

    revision = git("rev-parse", "HEAD").strip()
    dirty = git(
        "status",
        "--porcelain=v1",
        "--untracked-files=no",
        "--",
        "tinyvllm",
        "tools",
    )
    if dirty:
        raise ValueError("source archive scope has tracked changes")
    tree = git(
        "ls-tree",
        "-r",
        "--full-tree",
        revision,
        "tinyvllm",
        "tools",
    ).encode("utf-8")
    return {
        "schema_version": SOURCE_SCHEMA,
        "run_tag": _validate_run_tag(run_tag),
        "source_revision": revision,
        "source_tree_sha256": hashlib.sha256(tree).hexdigest(),
        "model_repository": MODEL_REPOSITORY,
        "model_revision": MODEL_REVISION,
    }


def _write_bundle_manifest(root: Path) -> dict:
    artifacts = {}
    for path in sorted(root.iterdir()):
        if path.is_file() and path.name != "manifest.json":
            artifacts[path.name] = hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
    payload = {
        "schema_version": (
            "tinyllmforge.tp4-segmented-capture-manifest.v1"
        ),
        "artifacts": artifacts,
    }
    _atomic_write_json(root / "manifest.json", payload)
    return payload


class ProductionAdapter(_DecodeReplayAdapter):
    """Census-specific adapter reusing the hardened SSH transport."""

    def __init__(
        self,
        *,
        run_tag: str,
        local_attempt_root: Path,
        ssh_target: str = DEFAULT_SSH_TARGET,
        remote_python: str = DEFAULT_REMOTE_PYTHON,
        control_path: str | None = None,
        command_timeout_s: int = DEFAULT_COMMAND_TIMEOUT_S,
        retry_count: int = DEFAULT_RETRY_COUNT,
    ):
        super().__init__(
            run_tag=run_tag,
            local_attempt_root=local_attempt_root,
            ssh_target=ssh_target,
            remote_python=remote_python,
            control_path=control_path,
            command_timeout_s=command_timeout_s,
            retry_count=retry_count,
            kerberos_query=query_local_kerberos,
        )

    def freeze_source(self, seed: dict) -> dict:
        first_freeze = self._source is None
        if (
            first_freeze
            and os.path.lexists(self.local_attempt_root)
        ):
            raise ValueError("local attempt root already exists")
        source = _capture_source_identity(seed["run_tag"])
        if self._source is not None and source != self._source:
            raise ValueError("frozen source identity drift")
        self._source = source
        self.local_controller_root.mkdir(
            parents=True,
            exist_ok=not first_freeze,
        )
        _atomic_write_json(
            self.local_controller_root / "source_identity.json",
            source,
        )
        return dict(source)

    def ssh_storage_preflight(
        self,
        seed: dict,
        source: dict,
    ) -> dict:
        del source
        kerberos = self._query_kerberos_window()
        if kerberos.get("classification") not in {"READY", "PASS"}:
            return {
                "classification": "INCOMPLETE",
                "reason": "Kerberos TTL preflight failed",
                "attempt_exists": False,
                "remote_root": REMOTE_ROOT,
                "kerberos": kerberos,
            }
        attempt_root = f"{REMOTE_ROOT}/{seed['run_tag']}"
        script = "\n".join([
            "import json,os,sys",
            "base,root,attempt,model,revision=sys.argv[1:]",
            "config_path=os.path.join(model,'config.json')",
            "config={}",
            "if os.path.isfile(config_path):",
            "  with open(config_path,encoding='utf-8') as handle:",
            "    config=json.load(handle)",
            "text=config.get('text_config',{}) if isinstance(config,dict) else {}",
            "print(json.dumps({",
            "'base_ready':os.path.isdir(base) and os.access(base,os.R_OK|os.W_OK|os.X_OK),",
            "'remote_root_safe':not os.path.islink(root) and (not os.path.exists(root) or os.path.isdir(root)),",
            "'attempt_exists':os.path.lexists(attempt),",
            "'model_ready':os.path.isdir(model) and os.access(model,os.R_OK|os.X_OK),",
            "'model_revision_matches':os.path.basename(os.path.realpath(model))==revision,",
            "'num_hidden_layers':text.get('num_hidden_layers'),",
            "},sort_keys=True))",
        ])
        state = json.loads(self._remote([
            "python3",
            "-c",
            script,
            REMOTE_BASE,
            REMOTE_ROOT,
            attempt_root,
            MODEL_ROOT,
            MODEL_REVISION,
        ]).stdout)
        if (
            state.get("base_ready") is not True
            or state.get("remote_root_safe") is not True
            or state.get("attempt_exists") is not False
            or state.get("model_ready") is not True
            or state.get("model_revision_matches") is not True
            or state.get("num_hidden_layers") != 64
        ):
            raise ValueError("remote storage or model preflight failed")
        receipt = {
            "classification": "PASS",
            "attempt_exists": False,
            "remote_root": REMOTE_ROOT,
            "kerberos": kerberos,
            "model_root": MODEL_ROOT,
            "model_revision": MODEL_REVISION,
        }
        _atomic_write_json(
            self.local_controller_root
            / "ssh_storage_preflight.json",
            receipt,
        )
        return receipt

    def gpu_admission(self, plan: dict, preflight: dict) -> dict:
        del preflight
        self._require_kerberos_window()
        observed = query_remote_gpu_inventory(
            ssh_target=self.ssh_target,
            control_path=self.control_path,
            timeout_s=self.command_timeout_s,
            retry_count=self.retry_count,
        )
        selected = validate_selected_gpu_processes(
            selected=tuple(plan["selected_gpus"]),
            observed=list(observed),
            owned_pids=frozenset(),
        )
        if any(
            row["memory_used_mib"] > MAX_GPU_MEMORY_USED_MIB
            or row["utilization_percent"]
            > MAX_GPU_UTILIZATION_PERCENT
            for row in selected
        ):
            raise RuntimeError("planned GPU inventory is not strict-clean")
        rows = [
            {
                "rank": rank,
                "index": row["gpu_index"],
                "uuid": row["gpu_uuid"],
                "memory_used_mib": row["memory_used_mib"],
                "utilization_percent": row[
                    "utilization_percent"
                ],
                "compute_processes": list(row["compute_processes"]),
            }
            for rank, row in enumerate(selected)
        ]
        self._admission = {
            "schema_version": (
                "tinyllmforge.tp4-segmented-capture-admission.v1"
            ),
            "run_tag": plan["run_tag"],
            "admission_mode": "strict_clean",
            "strict_clean": True,
            "world_size": WORLD_SIZE,
            "selected_gpus": rows,
        }
        _atomic_write_json(
            self.local_controller_root
            / "strict_clean_admission.json",
            self._admission,
        )
        return {
            "classification": "READY",
            "selected_gpus": [dict(row) for row in selected],
        }

    def launch(self, plan: dict, admission: dict) -> dict:
        del admission
        if self._source is None or self._admission is None:
            raise RuntimeError("launch prerequisites are incomplete")
        self._plan = dict(plan)
        directories = [
            plan["paths"]["attempt_root"],
            plan["paths"]["source_root"],
            plan["paths"]["raw_root"],
            plan["paths"]["bundle_root"],
            plan["paths"]["controller_root"],
            *plan["environment"].values(),
        ]
        script = "\n".join([
            "import os,sys",
            "paths=sys.argv[1:]",
            "for path in paths:",
            "  if os.path.lexists(path):",
            "    raise SystemExit('path already exists: '+path)",
            "for path in paths:",
            "  os.makedirs(path,exist_ok=False)",
        ])
        self._remote(["python3", "-c", script, *directories])
        self._stage_source(plan)
        for name, payload in (
            ("source_manifest.json", self._source),
            ("gpu_inventory.json", self._admission),
        ):
            self._upload_bytes(
                f"{plan['paths']['raw_root']}/{name}",
                (
                    json.dumps(
                        payload,
                        sort_keys=True,
                        separators=(",", ":"),
                        allow_nan=False,
                    )
                    + "\n"
                ).encode("utf-8"),
            )
        patch = subprocess.run(
            [
                "git",
                "-C",
                str(Path(__file__).resolve().parents[1]),
                "diff",
                "--binary",
                plan["source_revision"],
                "--",
                "tinyvllm",
                "tools",
            ],
            capture_output=True,
            check=True,
        ).stdout
        self._upload_bytes(
            f"{plan['paths']['raw_root']}/source.patch",
            patch or b"# no tracked diff\n",
        )
        environment = {
            **plan["environment"],
            **plan["process_environment"],
            "CUDA_VISIBLE_DEVICES": ",".join(
                str(index) for index in plan["selected_gpu_indices"]
            ),
        }
        wrapper = "\n".join([
            "import json,os,subprocess,sys",
            "python,stdout_path,stderr_path,env_json,*argv=sys.argv[1:]",
            "environment=os.environ.copy()",
            "environment.update(json.loads(env_json))",
            "with open(stdout_path,'w',encoding='utf-8') as out, open(stderr_path,'w',encoding='utf-8') as err:",
            "  result=subprocess.run([python,*argv],env=environment,stdout=out,stderr=err,text=True,check=False)",
            "raise SystemExit(result.returncode)",
        ])
        remote_argv = [
            "python3",
            "-c",
            wrapper,
            self.remote_python,
            plan["paths"]["worker_stdout_path"],
            plan["paths"]["worker_stderr_path"],
            json.dumps(environment, sort_keys=True),
            (
                f"{plan['paths']['source_root']}/tools/"
                "tp4_segmented_capture_census_worker.py"
            ),
            "--run-tag",
            plan["run_tag"],
            "--model-root",
            MODEL_ROOT,
            "--output-dir",
            plan["paths"]["raw_root"],
            "--timeout-s",
            "900",
        ]
        self._process = subprocess.Popen(
            build_ssh_argv(
                ssh_target=self.ssh_target,
                remote_argv=remote_argv,
                control_path=self.control_path,
            ),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        return {
            "local_ssh_pid": self._process.pid,
            "owned_pids": [self._process.pid],
        }

    def wait(self, plan: dict, launch: dict) -> dict:
        if self._process is None:
            raise RuntimeError("worker process was not launched")
        stdout, stderr = self._process.communicate(
            timeout=self.command_timeout_s
        )
        returncode = self._process.returncode
        self._cleanup = {
            "schema_version": (
                "tinyllmforge.tp4-segmented-capture-cleanup.v1"
            ),
            "run_tag": plan["run_tag"],
            "classification": (
                "CLEAN" if returncode == 0 else "DIRTY"
            ),
            "owned_children_remaining": [],
            "exact_tag_scans": [],
            "rank_rows": [
                {
                    "rank": rank,
                    "exit_code": 0 if returncode == 0 else returncode,
                    "process_group_destroyed": returncode == 0,
                }
                for rank in range(WORLD_SIZE)
            ],
        }
        _atomic_write_json(
            self.local_controller_root / "worker_wait.json",
            {
                "returncode": returncode,
                "stdout": stdout,
                "stderr": stderr,
            },
        )
        if returncode != 0:
            tail = self._remote([
                "python3",
                "-c",
                (
                    "import pathlib,sys;"
                    "p=pathlib.Path(sys.argv[1]);"
                    "print(p.read_text(errors='replace')[-12000:] "
                    "if p.is_file() else '')"
                ),
                plan["paths"]["worker_stderr_path"],
            ]).stdout
            raise RuntimeError(
                f"remote census worker failed with {returncode}: {tail}"
            )
        cleanup = self.validate_cleanup(plan, launch)
        if (
            cleanup.get("classification") != "CLEAN"
            or cleanup.get("final_exact_tag_scans") != [[], [], []]
        ):
            raise RuntimeError(
                "remote census worker cleanup is not clean"
            )
        return {"exit_code": 0, "stdout": stdout}

    def download(self, plan: dict, waited: dict) -> dict:
        if waited.get("exit_code") != 0 or self._cleanup is None:
            raise RuntimeError("worker completion evidence is missing")
        raw_names = (
            "source_manifest.json",
            "source.patch",
            "gpu_inventory.json",
            "segment_rows.jsonl",
            "process_receipts.json",
            "worker_result.json",
        )
        self._download_archive(
            remote_root=plan["paths"]["raw_root"],
            names=raw_names,
            local_root=self.local_raw_root,
        )
        self.local_bundle_root.mkdir(parents=True, exist_ok=False)
        _atomic_write_json(
            self.local_bundle_root / "source_identity.json",
            self._source,
        )
        shutil.copy2(
            self.local_raw_root / "source_manifest.json",
            self.local_bundle_root / "source_manifest.json",
        )
        _atomic_write_json(
            self.local_bundle_root / "launch_admission.json",
            self._admission,
        )
        _atomic_write_json(
            self.local_bundle_root / "cleanup.json",
            self._cleanup,
        )
        shutil.copy2(
            self.local_raw_root / "process_receipts.json",
            self.local_bundle_root / "process_receipts.json",
        )
        shutil.copy2(
            self.local_raw_root / "segment_rows.jsonl",
            self.local_bundle_root / "segment_rows.jsonl",
        )
        _write_bundle_manifest(self.local_bundle_root)
        self._upload_bundle(plan)
        return {
            "downloaded": True,
            "bundle_root": str(self.local_bundle_root),
        }

    def remote_verify(self, plan: dict, downloaded: dict) -> dict:
        if downloaded.get("downloaded") is not True:
            raise RuntimeError("census bundle was not downloaded")
        result = self._remote([
            self.remote_python,
            (
                f"{plan['paths']['source_root']}/tools/"
                "verify_tp4_segmented_capture_census.py"
            ),
            "--bundle-root",
            plan["paths"]["bundle_root"],
        ])
        verification = json.loads(result.stdout)
        _atomic_write_json(
            self.local_controller_root
            / "remote_independent_verification.json",
            verification,
        )
        return verification

    def local_verify(self, plan: dict, downloaded: dict) -> dict:
        del plan
        if downloaded.get("downloaded") is not True:
            raise RuntimeError("census bundle was not downloaded")
        result = subprocess.run(
            [
                sys.executable,
                str(
                    Path(__file__).resolve().parent
                    / "verify_tp4_segmented_capture_census.py"
                ),
                "--bundle-root",
                str(self.local_bundle_root),
            ],
            text=True,
            capture_output=True,
            check=False,
            timeout=self.command_timeout_s,
        )
        if result.returncode != 0:
            raise RuntimeError(
                result.stderr or "local census verifier failed"
            )
        verification = json.loads(result.stdout)
        _atomic_write_json(
            self.local_controller_root
            / "local_independent_verification.json",
            verification,
        )
        return verification

    def _scan_exact_tag(self, plan: dict) -> list[dict]:
        script = "\n".join([
            "import json,os,sys",
            "tag,attempt_root=sys.argv[1:]",
            "tag_entry=('TINYLLMFORGE_RUN_TAG='+tag).encode()",
            "attempt_bytes=attempt_root.encode()",
            "excluded={os.getpid(),os.getppid()}",
            "rows=[]",
            "for name in os.listdir('/proc'):",
            "  if not name.isdigit(): continue",
            "  pid=int(name)",
            "  if pid in excluded: continue",
            "  try:",
            "    data=open('/proc/'+name+'/cmdline','rb').read()",
            "  except OSError:",
            "    data=b''",
            "  try:",
            "    environment=open('/proc/'+name+'/environ','rb').read()",
            "  except OSError:",
            "    environment=b''",
            "  command=data.replace(b'\\0',b' ').decode(errors='replace')",
            (
                "  matched_cmdline=any("
                "value==attempt_bytes or "
                "value.startswith(attempt_bytes+b'/') "
                "for value in data.split(b'\\0'))"
            ),
            "  matched_environment=tag_entry in environment.split(b'\\0')",
            "  if matched_cmdline or matched_environment:",
            "    rows.append({",
            "      'pid':pid,",
            "      'command':command,",
            "      'matched_cmdline':matched_cmdline,",
            "      'matched_environment':matched_environment,",
            "    })",
            "print(json.dumps(sorted(rows,key=lambda row:row['pid'])))",
        ])
        result = self._remote([
            "python3",
            "-c",
            script,
            plan["run_tag"],
            plan["paths"]["attempt_root"],
        ])
        rows = json.loads(result.stdout)
        if not isinstance(rows, list):
            raise ValueError("remote exact-tag scan is invalid")
        return rows

    def validate_cleanup(
        self,
        plan: dict,
        launch: dict | None,
    ) -> dict:
        del launch
        initial_scan = self._scan_exact_tag(plan)
        reap_receipt = None
        if initial_scan:
            reap_receipt = self._reap_exact_tag(plan, initial_scan)
        self._finalize_local_process()
        final_scans = [self._scan_exact_tag(plan) for _ in range(3)]
        scans = [initial_scan, *final_scans]
        existing_rank_rows = (
            self._cleanup.get("rank_rows")
            if isinstance(self._cleanup, dict)
            else None
        )
        rank_lifecycle_clean = (
            isinstance(existing_rank_rows, list)
            and len(existing_rank_rows) == WORLD_SIZE
            and {
                row.get("rank") for row in existing_rank_rows
            }
            == set(range(WORLD_SIZE))
            and all(
                row.get("exit_code") == 0
                and row.get("process_group_destroyed") is True
                for row in existing_rank_rows
            )
        )
        if self._cleanup is None:
            self._cleanup = {
                "schema_version": (
                    "tinyllmforge.tp4-segmented-capture-cleanup.v1"
                ),
                "run_tag": plan["run_tag"],
                "classification": (
                    "CLEAN"
                    if final_scans == [[], [], []]
                    else "DIRTY"
                ),
                "owned_children_remaining": [
                    row for scan in final_scans for row in scan
                ],
                "exact_tag_scans": scans,
                "final_exact_tag_scans": final_scans,
                "rank_rows": [],
            }
        elif (
            final_scans != [[], [], []]
            or not rank_lifecycle_clean
        ):
            self._cleanup["classification"] = "DIRTY"
            self._cleanup["owned_children_remaining"] = [
                row for scan in final_scans for row in scan
            ]
            self._cleanup["exact_tag_scans"] = scans
            self._cleanup["final_exact_tag_scans"] = final_scans
        else:
            self._cleanup["classification"] = "CLEAN"
            self._cleanup["owned_children_remaining"] = []
            self._cleanup["exact_tag_scans"] = scans
            self._cleanup["final_exact_tag_scans"] = final_scans
        if reap_receipt is not None:
            self._cleanup["reap_receipt"] = reap_receipt
        _atomic_write_json(
            self.local_controller_root / "cleanup.json",
            self._cleanup,
        )
        return dict(self._cleanup)

    def _finalize_local_process(self) -> None:
        process = getattr(self, "_process", None)
        if process is None or process.poll() is not None:
            return
        process.terminate()
        try:
            process.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.communicate(timeout=5)

    def _reap_exact_tag(
        self,
        plan: dict,
        observed_rows: list[dict],
    ) -> dict:
        requested_pids = sorted({
            int(row["pid"])
            for row in observed_rows
            if (
                isinstance(row, dict)
                and type(row.get("pid")) is int
                and row["pid"] > 0
                and (
                    row.get("matched_cmdline") is True
                    or row.get("matched_environment") is True
                )
            )
        })
        if len(requested_pids) != len({
            row.get("pid")
            for row in observed_rows
            if isinstance(row, dict)
        }):
            raise RuntimeError("exact-tag cleanup inventory is invalid")
        script = "\n".join([
            "import json,os,signal,sys,time",
            "tag,attempt_root,*raw_pids=sys.argv[1:]",
            "tag_entry=('TINYLLMFORGE_RUN_TAG='+tag).encode()",
            "attempt_bytes=attempt_root.encode()",
            "excluded={os.getpid(),os.getppid()}",
            "requested=sorted({int(value) for value in raw_pids})",
            "def matches(pid):",
            "  if pid in excluded: return False",
            "  try:",
            "    command=open(f'/proc/{pid}/cmdline','rb').read()",
            "    environment=open(f'/proc/{pid}/environ','rb').read()",
            "  except OSError:",
            "    return False",
            "  arguments=command.split(b'\\0')",
            (
                "  return any(value==attempt_bytes or "
                "value.startswith(attempt_bytes+b'/') "
                "for value in arguments) "
                "or tag_entry in environment.split(b'\\0')"
            ),
            "eligible=[pid for pid in requested if matches(pid)]",
            "for pid in eligible:",
            "  try: os.kill(pid,signal.SIGTERM)",
            "  except ProcessLookupError: pass",
            "deadline=time.monotonic()+10.0",
            "while time.monotonic()<deadline:",
            "  remaining=[pid for pid in eligible if matches(pid)]",
            "  if not remaining: break",
            "  time.sleep(0.1)",
            "remaining=[pid for pid in eligible if matches(pid)]",
            "for pid in remaining:",
            "  try: os.kill(pid,signal.SIGKILL)",
            "  except ProcessLookupError: pass",
            "time.sleep(0.2)",
            "survivors=[pid for pid in eligible if matches(pid)]",
            "print(json.dumps({",
            "'requested_pids':requested,",
            "'terminated_pids':eligible,",
            "'killed_pids':remaining,",
            "'remaining_pids':survivors,",
            "},sort_keys=True))",
        ])
        result = self._remote([
            "python3",
            "-c",
            script,
            plan["run_tag"],
            plan["paths"]["attempt_root"],
            *(str(pid) for pid in requested_pids),
        ])
        receipt = json.loads(result.stdout)
        if (
            not isinstance(receipt, dict)
            or receipt.get("requested_pids") != requested_pids
            or receipt.get("remaining_pids") != []
        ):
            raise RuntimeError("exact-tag owned cleanup failed")
        return receipt


def _parse_args(argv=None):
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("monitor-and-run")
    run.add_argument("--run-tag", required=True)
    run.add_argument(
        "--admission-mode",
        choices=("strict_clean",),
        default="strict_clean",
    )
    run.add_argument(
        "--ssh-target",
        default="sitian@10.232.195.203",
    )
    run.add_argument(
        "--remote-python",
        default=DEFAULT_REMOTE_PYTHON,
    )
    run.add_argument("--control-path")
    run.add_argument(
        "--command-timeout-s",
        type=int,
        default=DEFAULT_COMMAND_TIMEOUT_S,
    )
    run.add_argument(
        "--gpu-wait-timeout-s",
        type=int,
        default=DEFAULT_GPU_WAIT_TIMEOUT_S,
    )
    run.add_argument(
        "--gpu-poll-interval-s",
        type=int,
        default=DEFAULT_GPU_POLL_INTERVAL_S,
    )
    run.add_argument(
        "--retry-count",
        type=int,
        default=DEFAULT_RETRY_COUNT,
    )
    run.add_argument("--local-attempt-root", type=Path)
    return parser.parse_args(argv)


def main(
    argv=None,
    *,
    gpu_monitor=None,
    adapter_factory=None,
) -> int:
    args = _parse_args(argv)
    local_attempt_root = (
        args.local_attempt_root
        if args.local_attempt_root is not None
        else (
            Path(__file__).resolve().parents[1]
            / "artifacts"
            / "tp4_segmented_capture_census"
            / args.run_tag
        )
    )
    if adapter_factory is None:
        adapter_factory = ProductionAdapter
    adapter = adapter_factory(
        run_tag=args.run_tag,
        local_attempt_root=local_attempt_root,
        ssh_target=args.ssh_target,
        remote_python=args.remote_python,
        control_path=args.control_path,
        command_timeout_s=args.command_timeout_s,
        retry_count=args.retry_count,
    )
    if gpu_monitor is None:
        def query_inventory():
            adapter._require_kerberos_window()
            return query_remote_gpu_inventory(
                ssh_target=args.ssh_target,
                control_path=args.control_path,
                timeout_s=args.command_timeout_s,
                retry_count=args.retry_count,
            )

        gpu_monitor = lambda: wait_for_strict_clean_gpus(
            query_inventory=query_inventory,
            timeout_s=args.gpu_wait_timeout_s,
            poll_interval_s=args.gpu_poll_interval_s,
        )
    result = monitor_and_run(
        run_tag=args.run_tag,
        admission_mode=args.admission_mode,
        gpu_monitor=gpu_monitor,
        adapter=adapter,
    )
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
