from __future__ import annotations

import argparse
import ast
from collections import Counter
from dataclasses import dataclass, field, make_dataclass
import gc
import getpass
import hashlib
import io
import importlib
from itertools import count
import json
import os
from pathlib import Path
import pickle
import shlex
import socket
import subprocess
import sys
import tarfile
import tempfile
from types import ModuleType, SimpleNamespace
import weakref


EXPECTED_FILE_SHA256 = {
    "tinyvllm/config.py": (
        "9b860eafe88c1734e5135ab0f65188f025762f5d0d0a702eb4994157aabec076"
    ),
    "tinyvllm/engine/llm_engine.py": (
        "6cf68dc76641bf772c01d31fd60ee42cbab82e3c62a0ee8aa154dbe802c727ae"
    ),
    "tinyvllm/engine/model_runner.py": (
        "0cba7e97b2d3425186722f53a0c14e7b01e2c90e190e6cdf2c9472517bcda849"
    ),
}
CONSTRUCTOR_RUNTIME_FILE_SHA256 = {
    "tinyvllm/engine/exact_cuda_graph_cache.py": (
        "e3e7486c54dea9e2c10ac84756080098faa121a87918fbb52bf1260795a1b524"
    ),
}
DIRECT_GATE_FILE_SHA256 = {
    "tools/qwen35_tp4_live_concurrent_candidate_ownership_preflight.py": (
        "f49d2d2724c288afa36a54c742fb1aceb276f576eb5726014b63c39452997391"
    ),
}
EXPECTED_METHOD_SHA256 = {
    "LLMEngine.__init__": (
        "f770308d40248be4515838a720b288fd69f718d25746398bc145b4b43478fd9c"
    ),
    "LLMEngine.bind_qwen35_loaded_checkpoint_candidates": (
        "82c0528d6b06ae8d67812d1a8802e8163aadb4886afc3894bf28a0cf35c3c84c"
    ),
    "LLMEngine.call_model_runner_acknowledged": (
        "6eed126b80c9c823ceff37cc51273735d656c2b1be963bbea2bbd4ad9da9f14d"
    ),
    "ModelRunner.__init__": (
        "8aa2747cff30e8398737cb024d375f9f04763efdd53cb23084c32c3d872f4edc"
    ),
    "ModelRunner.bind_published_qwen35_loaded_checkpoint_candidate": (
        "aa178f886d314893593039c5e890239fb954740f059b2d12fc697bd25790fbcd"
    ),
    "ModelRunner.bind_qwen35_loaded_checkpoint_candidate": (
        "a14e6856ad74eb935116075ee6fe81516c8e212f89914e0fcd55bb39e86d63e0"
    ),
    "ModelRunner.dispatch_command": (
        "9a63e40ef7d16b6300e70d41c2f05575a50adbf6dc04942677034d6bee363342"
    ),
    "ModelRunner.publish_qwen35_loaded_checkpoint_candidate": (
        "37f95954c287d5dd0e8883f299d7049e66dcb3c79806624eb6da3ca7d51a6d4f"
    ),
}
EXPECTED_CONSTRUCTOR_SIGNATURES = {
    "LLMEngine.__init__": "(self, model, **kwargs)",
    "ModelRunner.__init__": (
        "(self, config: Config, rank: int, "
        "event: Event | list[Event], ack_sender=None)"
    ),
}
CONSTRUCTOR_REPLACEMENT_ALLOWLIST = {
    "llm_engine.Config",
    "llm_engine.ModelRunnerCommandAckCollector",
    "llm_engine.AutoTokenizer",
    "llm_engine.Scheduler",
    "llm_engine.mp.get_context",
    "llm_engine.atexit.register",
    "model_runner.dist.init_process_group",
    "model_runner.dist.barrier",
    "model_runner.torch.cuda.set_device",
    "model_runner.torch.get_default_dtype",
    "model_runner.torch.set_default_dtype",
    "model_runner.torch.set_default_device",
    "model_runner.set_quant_config",
    "model_runner.Qwen3ForCausalLM",
    "model_runner.load_model",
    "model_runner.apply_cpu_offload",
    "model_runner.Sampler",
    "model_runner.SharedMemory",
    "ModelRunner.warmup_model",
    "ModelRunner.allocate_kv_cache",
    "ModelRunner.capture_cudagraph",
    "ModelRunner.loop",
}
EXPECTED_CONSTRUCTOR_CALL_COUNTS = {
    "LLMEngine.__init__": 1,
    "Config": 1,
    "mp.get_context": 1,
    "context.Pipe": 3,
    "context.Event": 3,
    "context.Process": 3,
    "process.start": 3,
    "sender.close": 3,
    "dist.init_process_group": 4,
    "torch.cuda.set_device": 4,
    "torch.get_default_dtype": 4,
    "torch.set_default_dtype": 8,
    "torch.set_default_device": 8,
    "set_quant_config": 4,
    "Qwen3ForCausalLM": 4,
    "load_model": 4,
    "Sampler": 4,
    "ModelRunner.warmup_model": 4,
    "ModelRunner.allocate_kv_cache": 4,
    "SharedMemory.create": 1,
    "SharedMemory.attach": 3,
    "dist.barrier": 4,
    "ModelRunner.loop": 3,
    "runner.deferred_construct": 3,
    "ack_collector.construct": 1,
    "AutoTokenizer.from_pretrained": 1,
    "Scheduler": 1,
    "atexit.register": 1,
}
IMPORT_TIME_STUB_MODULES = {
    "tinyvllm.engine.flash_attn_split_policy",
    "tinyvllm.engine.kv_cartridge",
    "tinyvllm.engine.qwen35_hybrid_prefix_engine_publication",
    "tinyvllm.engine.qwen35_hybrid_prefix_engine_restore",
    "tinyvllm.engine.qwen35_hybrid_prefix_owner",
    "tinyvllm.engine.qwen35_hybrid_prefix_publication_ticket",
    "tinyvllm.engine.qwen35_hybrid_prefix_restore_ticket",
    "tinyvllm.engine.qwen35_hybrid_prefix_source_publication",
    "tinyvllm.engine.scheduler",
    "tinyvllm.engine.sequence",
    "tinyvllm.layers.linear",
    "tinyvllm.layers.sampler",
    "tinyvllm.models.qwen3",
    "tinyvllm.sampling_params",
    "tinyvllm.utils.cpu_offload",
    "tinyvllm.utils.loader",
}
PROVENANCE = (
    "real-checkpoint-derived-constructed-engine-model-runner-ownership"
)
CLAIM_BOUNDARY = "no-scheduler-step-forward-or-inference"
RESULT_SCHEMA_VERSION = (
    "qwen35.constructed-engine-model-runner-ownership.v1"
)
RESULT_NAME = "constructed_engine_model_runner_ownership.json"
MANIFEST_NAME = "source_manifest.json"
FORBIDDEN_COUNTER_NAMES = (
    "production_scheduler_constructor",
    "scheduler_step",
    "engine_step",
    "model_runner_run",
    "model_forward",
    "attention_forward",
    "sampler_call",
    "tokenization",
    "cuda_operation",
    "nccl_operation",
    "generation",
    "inference",
    "os_process_creation",
    "os_shared_memory_creation",
    "real_atexit_registration",
)
MEMORY_CEILINGS_KIB = {
    "process_total_vmhwm_increment": 12582912,
    "ready_vmrss": 8388608,
    "host_mem_available_decrease": 12582912,
    "minimum_preflight_mem_available": 16777216,
}
PREREQUISITE_ORACLE_SHA256 = (
    "d750d664219378c234a2127b708ec191feb9b2c9f1f2902c47d0ad5dc152d3ef"
)
PREREQUISITE_ORACLE_NAME = (
    "tp4_real_candidate_provenance_oracle.json"
)
APPROVED_MODEL_DIR = (
    "/data00/home/sitian/sitian-workspace01/tllm/"
    "qwen35-hybrid-state-runs/"
    "qwen35-2b-hybrid-acquire-20260723-222004/model"
)
REMOTE_TARGET = "sitian@10.232.195.203"
REMOTE_PYTHON = (
    "/data00/home/sitian/sitian-workspace01/tllm/env/bin/python"
)
REMOTE_RUN_ROOT = (
    "/data00/home/sitian/sitian-workspace01/tllm/"
    "qwen35-constructed-engine-model-runner-ownership-runs"
)
LOCAL_RUN_ROOT = Path("experiments/qwen35_hybrid_state")
CONTROL_PATH = "/tmp/ssh-sitian-10.232.195.203"


def _sha256(payload):
    return hashlib.sha256(payload).hexdigest()


def _canonical(value):
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _method_source_sha256(path, class_name, method_name):
    source = path.read_text()
    tree = ast.parse(source, filename=str(path))
    class_node = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == class_name
    )
    method = next(
        node
        for node in class_node.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == method_name
    )
    method_source = ast.get_source_segment(source, method)
    return _sha256(method_source.encode("utf-8")), method


def _format_argument(argument):
    value = argument.arg
    if argument.annotation is not None:
        value += ": " + ast.unparse(argument.annotation)
    return value


def _format_signature(method):
    arguments = method.args
    positional = [
        *arguments.posonlyargs,
        *arguments.args,
    ]
    defaults = [None] * (
        len(positional) - len(arguments.defaults)
    ) + list(arguments.defaults)
    parts = []
    for argument, default in zip(positional, defaults):
        value = _format_argument(argument)
        if default is not None:
            value += "=" + ast.unparse(default)
        parts.append(value)
    if arguments.vararg is not None:
        parts.append("*" + _format_argument(arguments.vararg))
    elif arguments.kwonlyargs:
        parts.append("*")
    for argument, default in zip(
        arguments.kwonlyargs,
        arguments.kw_defaults,
    ):
        value = _format_argument(argument)
        if default is not None:
            value += "=" + ast.unparse(default)
        parts.append(value)
    if arguments.kwarg is not None:
        parts.append("**" + _format_argument(arguments.kwarg))
    return "(" + ", ".join(parts) + ")"


def inspect_constructed_runtime_source_contract(source_root):
    root = Path(source_root)
    actual_files = {}
    for relative, expected in EXPECTED_FILE_SHA256.items():
        path = root / relative
        if not path.is_file():
            raise ValueError(f"missing production source: {relative}")
        actual = _sha256(path.read_bytes())
        if actual != expected:
            raise ValueError(
                "source hash mismatch: "
                f"{relative}: expected={expected}, actual={actual}"
            )
        actual_files[relative] = actual

    method_locations = {
        "LLMEngine": (
            root / "tinyvllm/engine/llm_engine.py",
            (
                "__init__",
                "bind_qwen35_loaded_checkpoint_candidates",
                "call_model_runner_acknowledged",
            ),
        ),
        "ModelRunner": (
            root / "tinyvllm/engine/model_runner.py",
            (
                "__init__",
                "bind_published_qwen35_loaded_checkpoint_candidate",
                "bind_qwen35_loaded_checkpoint_candidate",
                "dispatch_command",
                "publish_qwen35_loaded_checkpoint_candidate",
            ),
        ),
    }
    actual_methods = {}
    constructor_signatures = {}
    for class_name, (path, names) in method_locations.items():
        for method_name in names:
            key = f"{class_name}.{method_name}"
            digest, method = _method_source_sha256(
                path,
                class_name,
                method_name,
            )
            expected = EXPECTED_METHOD_SHA256[key]
            if digest != expected:
                raise ValueError(
                    "method source hash mismatch: "
                    f"{key}: expected={expected}, actual={digest}"
                )
            actual_methods[key] = digest
            if method_name == "__init__":
                signature = _format_signature(method)
                expected_signature = EXPECTED_CONSTRUCTOR_SIGNATURES[key]
                if signature != expected_signature:
                    raise ValueError(
                        "constructor signature mismatch: "
                        f"{key}: expected={expected_signature}, "
                        f"actual={signature}"
                    )
                constructor_signatures[key] = signature
    return {
        "files": dict(sorted(actual_files.items())),
        "methods": dict(sorted(actual_methods.items())),
        "constructor_signatures": dict(
            sorted(constructor_signatures.items())
        ),
        "forbidden_execution_forms": {
            "object_new": False,
            "constructor_ast_compile": False,
            "subclass_construction": False,
            "class_replacement": False,
        },
    }


def validate_constructor_replacement_names(names):
    names = set(names)
    if names != CONSTRUCTOR_REPLACEMENT_ALLOWLIST:
        missing = sorted(CONSTRUCTOR_REPLACEMENT_ALLOWLIST - names)
        extra = sorted(names - CONSTRUCTOR_REPLACEMENT_ALLOWLIST)
        raise ValueError(
            "replacement allowlist mismatch: "
            f"missing={missing!r}, extra={extra!r}"
        )
    return frozenset(names)


def validate_constructor_evidence(evidence):
    if not isinstance(evidence, dict):
        raise ValueError("constructor evidence must be an object")
    if (
        evidence.get("engine_constructor_count") != 1
        or evidence.get("runner_constructor_count") != 4
        or evidence.get("runner_constructor_ranks") != [0, 1, 2, 3]
        or evidence.get("restoration_complete") is not True
    ):
        raise ValueError("constructor identity evidence is invalid")
    counts = evidence.get("dependency_call_counts")
    if not isinstance(counts, dict):
        raise ValueError("constructor call counts are invalid")
    actual = {
        name: counts.get(name, 0)
        for name in EXPECTED_CONSTRUCTOR_CALL_COUNTS
    }
    if actual != EXPECTED_CONSTRUCTOR_CALL_COUNTS:
        raise ValueError(
            "constructor call counts are invalid: "
            f"expected={EXPECTED_CONSTRUCTOR_CALL_COUNTS!r}, "
            f"actual={actual!r}"
        )
    for forbidden in (
        "apply_cpu_offload",
        "ModelRunner.capture_cudagraph",
        "LLMEngine.step",
        "ModelRunner.run",
        "model.forward",
        "attention.forward",
        "tokenizer.call",
        "sampler.call",
        "inference",
    ):
        if counts.get(forbidden, 0) != 0:
            raise ValueError(
                f"forbidden constructor call is non-zero: {forbidden}"
            )
    return evidence


def _validate_constructed_memory(memory):
    if not isinstance(memory, dict):
        raise ValueError("constructed memory evidence is invalid")
    required = {
        "process_before",
        "process_ready",
        "process_after_cleanup",
        "host_before",
        "host_ready",
    }
    if set(memory) != required:
        raise ValueError("constructed memory points are incomplete")
    for name in (
        "process_before",
        "process_ready",
        "process_after_cleanup",
    ):
        point = memory[name]
        if (
            not isinstance(point, dict)
            or set(point) != {"vmrss_kib", "vmhwm_kib"}
            or any(
                isinstance(value, bool)
                or not isinstance(value, int)
                or value <= 0
                for value in point.values()
            )
        ):
            raise ValueError(
                f"constructed process memory point is invalid: {name}"
            )
    for name in ("host_before", "host_ready"):
        point = memory[name]
        if (
            not isinstance(point, dict)
            or set(point) != {"mem_available_kib"}
            or isinstance(point["mem_available_kib"], bool)
            or not isinstance(point["mem_available_kib"], int)
            or point["mem_available_kib"] <= 0
        ):
            raise ValueError(
                f"constructed host memory point is invalid: {name}"
            )
    total_increment = (
        memory["process_ready"]["vmhwm_kib"]
        - memory["process_before"]["vmhwm_kib"]
    )
    host_decrease = (
        memory["host_before"]["mem_available_kib"]
        - memory["host_ready"]["mem_available_kib"]
    )
    if (
        memory["host_before"]["mem_available_kib"]
        < MEMORY_CEILINGS_KIB["minimum_preflight_mem_available"]
        or total_increment
        > MEMORY_CEILINGS_KIB["process_total_vmhwm_increment"]
        or memory["process_ready"]["vmrss_kib"]
        > MEMORY_CEILINGS_KIB["ready_vmrss"]
        or host_decrease
        > MEMORY_CEILINGS_KIB["host_mem_available_decrease"]
    ):
        raise ValueError("constructed memory ceiling exceeded")
    return {
        **memory,
        "process_total_vmhwm_increment_kib": total_increment,
        "host_mem_available_decrease_kib": host_decrease,
        "memory_contract_passed": True,
    }


def build_constructed_runtime_artifact(
    *,
    run_tag,
    smoke,
    cleanup,
    memory,
    source_file_sha256,
    prerequisite_oracle_sha256,
    observed_user,
    observed_hostname,
):
    if (
        not isinstance(run_tag, str)
        or not run_tag
        or not isinstance(observed_user, str)
        or not observed_user
        or not isinstance(observed_hostname, str)
        or not observed_hostname
    ):
        raise ValueError("constructed artifact identity is invalid")
    if not isinstance(smoke, dict) or smoke.get("status") != "PASS":
        raise ValueError("constructed smoke is not complete")
    validate_constructor_evidence(smoke.get("constructor_evidence"))
    class_identity = smoke.get("class_identity")
    if (
        not isinstance(class_identity, dict)
        or class_identity.get("engine_exact_class") is not True
        or class_identity.get("runner_exact_class_by_rank")
        != [True, True, True, True]
    ):
        raise ValueError("constructed class identity is invalid")
    ledger = smoke.get("constructor_ledger")
    if (
        not isinstance(ledger, list)
        or not ledger
        or [row.get("sequence") for row in ledger]
        != list(range(len(ledger)))
    ):
        raise ValueError("constructed constructor ledger is invalid")
    payloads = smoke.get("rank_payloads")
    if (
        not isinstance(payloads, list)
        or [row.get("rank") for row in payloads] != [0, 1, 2, 3]
        or any(
            row.get("binding_hash_count") != 320
            or len(row.get("binding_destination_sha256", ())) != 320
            or row.get("phase_hash_count") != 26
            or len(row.get("phase_destination_sha256", {})) != 26
            or len(row.get("alias_groups", ())) != 24
            or row.get("transfer_evidence", {}).get(
                "candidate_published"
            )
            is not True
            or row.get("transfer_evidence", {}).get(
                "candidate_bound_before_engine_dispatch"
            )
            is not False
            for row in payloads
        )
    ):
        raise ValueError("constructed rank payloads are invalid")
    first = smoke.get("first_binding")
    repeat = smoke.get("repeat_binding")
    envelope = first.get("command_envelope", {}) if isinstance(
        first,
        dict,
    ) else {}
    acknowledgements = first.get(
        "worker_acknowledgements",
        (),
    ) if isinstance(first, dict) else ()
    if (
        not isinstance(first, dict)
        or not isinstance(repeat, dict)
        or first.get("zero_payload_command") is not True
        or first.get("exact_repeat_zero_dispatch") is not None
        or repeat.get("zero_payload_command") is not True
        or repeat.get("exact_repeat_zero_dispatch") is not True
        or first.get("rows") != repeat.get("rows")
        or first.get("configuration") != repeat.get("configuration")
        or envelope
        != {
            "command_id": envelope.get("command_id"),
            "method_name": (
                "bind_published_qwen35_loaded_checkpoint_candidate"
            ),
            "args": [],
            "requires_ack": True,
        }
        or not isinstance(envelope.get("command_id"), int)
        or isinstance(envelope.get("command_id"), bool)
        or envelope["command_id"] < 0
        or not isinstance(acknowledgements, list)
        or [row.get("rank") for row in acknowledgements] != [1, 2, 3]
        or any(
            row.get("command_id") != envelope["command_id"]
            or row.get("status") != "ok"
            or row.get("result") != first["rows"][row["rank"]]
            or row.get("error_type") != ""
            or row.get("error_detail") != ""
            for row in acknowledgements
        )
        or repeat.get("command_envelope") != envelope
        or repeat.get("worker_acknowledgements")
        != acknowledgements
    ):
        raise ValueError("constructed binding evidence is invalid")
    forbidden = smoke.get("forbidden_counters")
    if (
        not isinstance(forbidden, dict)
        or set(forbidden) != set(FORBIDDEN_COUNTER_NAMES)
        or any(value != 0 for value in forbidden.values())
    ):
        raise ValueError("forbidden counters are non-zero")
    if (
        smoke.get("cuda_initialized_after") is not False
        or smoke.get("transport_restoration", {}).get("restored")
        is not True
        or smoke.get("transport_restoration", {}).get(
            "envelope_class_identity"
        )
        is not True
    ):
        raise ValueError("constructed transport safety is invalid")
    required_cleanup = {
        "release_rank_order": [3, 2, 1, 0],
        "all_selected_destinations_zero_after_clear": True,
        "non_selected_tensors_unchanged": True,
        "tensor_identity_preserved": True,
        "pool_unchanged": True,
        "all_inert_resources_closed": True,
        "production_exit_call_count": 0,
        "all_private_objects_collected": True,
    }
    if (
        not isinstance(cleanup, dict)
        or any(
            cleanup.get(name) != value
            for name, value in required_cleanup.items()
        )
        or not isinstance(cleanup.get("collected_private_objects"), dict)
        or not cleanup["collected_private_objects"]
        or not all(cleanup["collected_private_objects"].values())
    ):
        raise ValueError("constructed cleanup evidence is invalid")
    if (
        not isinstance(source_file_sha256, dict)
        or not source_file_sha256
        or any(
            not isinstance(name, str)
            or not name
            or not isinstance(digest, str)
            or len(digest) != 64
            for name, digest in source_file_sha256.items()
        )
        or not isinstance(prerequisite_oracle_sha256, str)
        or len(prerequisite_oracle_sha256) != 64
    ):
        raise ValueError("constructed source evidence is invalid")
    memory_evidence = _validate_constructed_memory(memory)
    source_hashes = dict(sorted(source_file_sha256.items()))
    return {
        "schema_version": RESULT_SCHEMA_VERSION,
        "status": "PASS",
        "run_tag": run_tag,
        "provenance": PROVENANCE,
        "claim_boundary": CLAIM_BOUNDARY,
        "observed_user": observed_user,
        "observed_hostname": observed_hostname,
        "prerequisite_oracle_sha256": prerequisite_oracle_sha256,
        "source_file_sha256": source_hashes,
        "source_tree_sha256": _sha256(_canonical(source_hashes)),
        "source_contract": smoke["source_contract"],
        "constructor_replacement_allowlist": sorted(
            CONSTRUCTOR_REPLACEMENT_ALLOWLIST
        ),
        "class_identity": class_identity,
        "constructor_evidence": smoke["constructor_evidence"],
        "constructor_ledger": ledger,
        "constructor_ledger_sha256": _sha256(_canonical(ledger)),
        "rank_payloads": payloads,
        "rank_payloads_sha256": _sha256(_canonical(payloads)),
        "first_binding": first,
        "repeat_binding": repeat,
        "transport_restoration": smoke["transport_restoration"],
        "forbidden_counters": forbidden,
        "cleanup": cleanup,
        "memory": memory_evidence,
        "cuda_initialized_after": False,
    }


def _atomic_write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    payload = _canonical(value) + b"\n"
    if path.exists() or temporary.exists():
        raise ValueError(f"artifact path already exists: {path}")
    try:
        temporary.write_bytes(payload)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return path


def finalize_constructed_runtime_artifact(
    *,
    run_dir,
    artifact,
    remote_target,
    remote_python,
):
    run_dir = Path(run_dir)
    if run_dir.exists():
        if any(run_dir.iterdir()):
            raise ValueError(
                "constructed run directory is not empty"
            )
    else:
        run_dir.mkdir(parents=True)
    if (
        not isinstance(artifact, dict)
        or artifact.get("schema_version") != RESULT_SCHEMA_VERSION
        or artifact.get("status") != "PASS"
        or not isinstance(remote_target, str)
        or not remote_target
        or not isinstance(remote_python, str)
        or not remote_python
    ):
        raise ValueError("constructed finalization input is invalid")
    result_path = run_dir / RESULT_NAME
    manifest_path = run_dir / MANIFEST_NAME
    try:
        _atomic_write_json(result_path, artifact)
        manifest = {
            "schema_version": RESULT_SCHEMA_VERSION,
            "run_tag": artifact["run_tag"],
            "remote_target": remote_target,
            "remote_python": remote_python,
            "source_file_sha256": artifact["source_file_sha256"],
            "source_tree_sha256": artifact["source_tree_sha256"],
            "prerequisite_oracle_sha256": artifact[
                "prerequisite_oracle_sha256"
            ],
            "result_sha256": _sha256(result_path.read_bytes()),
        }
        _atomic_write_json(manifest_path, manifest)
    except Exception:
        if manifest_path.exists():
            manifest_path.unlink()
        if result_path.exists():
            result_path.unlink()
        raise
    if {path.name for path in run_dir.iterdir()} != {
        RESULT_NAME,
        MANIFEST_NAME,
    }:
        raise RuntimeError(
            "constructed artifact inventory is invalid"
        )
    return result_path, manifest_path


def validate_run_tag(run_tag):
    if (
        not isinstance(run_tag, str)
        or not run_tag
        or any(
            character not in (
                "abcdefghijklmnopqrstuvwxyz"
                "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
                "0123456789-_"
            )
            for character in run_tag
        )
    ):
        raise ValueError("run tag is invalid")
    return run_tag


def _read_process_memory():
    values = {}
    for line in Path("/proc/self/status").read_text(
        encoding="utf-8"
    ).splitlines():
        if ":" not in line:
            continue
        name, raw = line.split(":", 1)
        parts = raw.strip().split()
        if name in ("VmRSS", "VmHWM") and parts:
            values[name.lower() + "_kib"] = int(parts[0])
    if set(values) != {"vmrss_kib", "vmhwm_kib"}:
        raise ValueError("process memory observation is incomplete")
    return values


def _read_host_memory():
    values = {}
    for line in Path("/proc/meminfo").read_text(
        encoding="utf-8"
    ).splitlines():
        if ":" not in line:
            continue
        name, raw = line.split(":", 1)
        parts = raw.strip().split()
        if name == "MemAvailable" and parts:
            values["mem_available_kib"] = int(parts[0])
    if set(values) != {"mem_available_kib"}:
        raise ValueError("host memory observation is incomplete")
    return values


def _source_closure_from_oracle(prerequisite_oracle):
    payload = Path(prerequisite_oracle).read_bytes()
    if _sha256(payload) != PREREQUISITE_ORACLE_SHA256:
        raise ValueError("prerequisite oracle hash is invalid")
    oracle = json.loads(payload)
    inherited = oracle.get("source_file_sha256")
    if not isinstance(inherited, dict) or not inherited:
        raise ValueError("prerequisite source closure is invalid")
    return tuple(sorted({
        *inherited,
        *EXPECTED_FILE_SHA256,
        *CONSTRUCTOR_RUNTIME_FILE_SHA256,
        *DIRECT_GATE_FILE_SHA256,
        (
            "tools/"
            "qwen35_constructed_engine_model_runner_ownership_preflight.py"
        ),
        (
            "tools/"
            "verify_qwen35_constructed_engine_model_runner_ownership_gate.py"
        ),
    }))


def _source_hashes(source_root, prerequisite_oracle):
    root = Path(source_root)
    hashes = {}
    for name in _source_closure_from_oracle(prerequisite_oracle):
        path = root / name
        if not path.is_file():
            raise ValueError(f"missing constructed source: {name}")
        hashes[name] = _sha256(path.read_bytes())
    return dict(sorted(hashes.items()))


def build_source_tar(source_root, prerequisite_oracle):
    root = Path(source_root)
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w") as archive:
        for name in _source_closure_from_oracle(prerequisite_oracle):
            path = root / name
            if not path.is_file():
                raise ValueError(
                    f"missing constructed source: {name}"
                )
            info = archive.gettarinfo(os.fspath(path), arcname=name)
            info.uid = info.gid = 0
            info.uname = info.gname = ""
            info.mtime = 0
            with path.open("rb") as source:
                archive.addfile(info, source)
    return stream.getvalue()


def build_ssh_command(arguments):
    return [
        "ssh",
        "-o",
        f"ControlPath={CONTROL_PATH}",
        REMOTE_TARGET,
        shlex.join([str(argument) for argument in arguments]),
    ]


def _require_success(completed, label):
    if completed.returncode != 0:
        stdout = getattr(completed, "stdout", "") or ""
        stderr = getattr(completed, "stderr", "") or ""
        raise RuntimeError(
            f"{label} failed: stdout={stdout!r}, stderr={stderr!r}"
        )
    return completed


def run_source_bound_constructed_gate(
    *,
    source_root,
    checkpoint_dir,
    prerequisite_oracle,
    run_dir,
    run_tag,
):
    validate_run_tag(run_tag)
    checkpoint = os.fspath(Path(checkpoint_dir).resolve())
    if checkpoint != APPROVED_MODEL_DIR:
        raise ValueError("constructed checkpoint is invalid")
    run_dir = Path(run_dir)
    if run_dir.name != run_tag:
        raise ValueError("constructed run directory tag mismatch")
    source_hashes = _source_hashes(
        source_root,
        prerequisite_oracle,
    )
    process_before = _read_process_memory()
    host_before = _read_host_memory()
    if (
        host_before["mem_available_kib"]
        < MEMORY_CEILINGS_KIB["minimum_preflight_mem_available"]
    ):
        raise RuntimeError(
            "host MemAvailable preflight is below "
            f"{MEMORY_CEILINGS_KIB['minimum_preflight_mem_available']} "
            "KiB"
        )
    smoke = run_constructed_binding_smoke(
        source_root=source_root,
        checkpoint_dir=checkpoint,
        prerequisite_oracle=prerequisite_oracle,
    )
    process_ready = _read_process_memory()
    host_ready = _read_host_memory()
    live_scope = smoke.pop("_live_scope")
    cleanup = release_constructed_runtime(live_scope)
    process_after_cleanup = _read_process_memory()
    artifact = build_constructed_runtime_artifact(
        run_tag=run_tag,
        smoke=smoke,
        cleanup=cleanup,
        memory={
            "process_before": process_before,
            "process_ready": process_ready,
            "process_after_cleanup": process_after_cleanup,
            "host_before": host_before,
            "host_ready": host_ready,
        },
        source_file_sha256=source_hashes,
        prerequisite_oracle_sha256=PREREQUISITE_ORACLE_SHA256,
        observed_user=getpass.getuser(),
        observed_hostname=socket.gethostname(),
    )
    result_path, manifest_path = (
        finalize_constructed_runtime_artifact(
            run_dir=run_dir,
            artifact=artifact,
            remote_target=REMOTE_TARGET,
            remote_python=REMOTE_PYTHON,
        )
    )
    return {
        "result": artifact,
        "manifest": json.loads(manifest_path.read_bytes()),
        "result_path": os.fspath(result_path),
        "manifest_path": os.fspath(manifest_path),
    }


def execute_remote_constructed_gate(
    *,
    source_root,
    prerequisite_oracle,
    run_tag,
    local_run_root=LOCAL_RUN_ROOT,
    command_runner=subprocess.run,
):
    validate_run_tag(run_tag)
    source_root = Path(source_root).resolve()
    local_hashes = _source_hashes(
        source_root,
        prerequisite_oracle,
    )
    remote_base = f"{REMOTE_RUN_ROOT}/{run_tag}"
    remote_source = f"{remote_base}/source"
    remote_evidence = f"{remote_base}/evidence/{run_tag}"
    remote_oracle = f"{remote_base}/{PREREQUISITE_ORACLE_NAME}"
    staged = command_runner(
        build_ssh_command([
            "bash",
            "-c",
            (
                f"test ! -e {shlex.quote(remote_base)} && "
                f"mkdir -p {shlex.quote(remote_source)} "
                f"{shlex.quote(remote_base + '/evidence')} && "
                f"tar -xf - -C {shlex.quote(remote_source)}"
            ),
        ]),
        input=build_source_tar(source_root, prerequisite_oracle),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    _require_success(staged, "constructed source staging")
    uploaded = command_runner(
        build_ssh_command([
            "bash",
            "-c",
            f"cat > {shlex.quote(remote_oracle)}",
        ]),
        input=Path(prerequisite_oracle).read_bytes(),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    _require_success(uploaded, "constructed oracle staging")
    remote_worker = (
        f"{remote_source}/tools/"
        "qwen35_constructed_engine_model_runner_ownership_preflight.py"
    )
    completed = command_runner(
        build_ssh_command([
            "env",
            "CUDA_VISIBLE_DEVICES=",
            "PYTHONDONTWRITEBYTECODE=1",
            "OMP_NUM_THREADS=8",
            "MKL_NUM_THREADS=8",
            REMOTE_PYTHON,
            "-B",
            remote_worker,
            "internal-run",
            "--source-root",
            remote_source,
            "--checkpoint-dir",
            APPROVED_MODEL_DIR,
            "--prerequisite-oracle",
            remote_oracle,
            "--run-dir",
            remote_evidence,
            "--run-tag",
            run_tag,
        ]),
        text=True,
        capture_output=True,
    )
    _require_success(completed, "constructed remote gate")
    remote_payload = json.loads(completed.stdout)
    if (
        remote_payload.get("result", {}).get("source_file_sha256")
        != local_hashes
    ):
        raise ValueError("constructed remote source binding mismatch")
    destination = Path(local_run_root) / run_tag
    if destination.exists():
        raise ValueError(
            f"local constructed run directory exists: {destination}"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(
        prefix=f".{run_tag}.",
        suffix=".tmp",
        dir=destination.parent,
    ))
    try:
        for name in (RESULT_NAME, MANIFEST_NAME):
            fetched = command_runner(
                build_ssh_command([
                    "cat",
                    f"{remote_evidence}/{name}",
                ]),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            _require_success(fetched, f"constructed fetch {name}")
            (temporary / name).write_bytes(fetched.stdout)
        temporary.replace(destination)
    finally:
        if temporary.exists():
            for child in temporary.iterdir():
                child.unlink()
            temporary.rmdir()
    verifier = _load_sibling_module(
        "_qwen35_constructed_runtime_verifier",
        source_root
        / "tools"
        / "verify_qwen35_constructed_engine_model_runner_ownership_gate.py",
    )
    verification = verifier.verify_run(
        destination,
        source_root=source_root,
        prerequisite_oracle=prerequisite_oracle,
    )
    return {
        "run_dir": os.fspath(destination),
        "verification": verification,
        "remote_evidence_dir": remote_evidence,
    }


class MutableDependencyNamespace:
    def __init__(self, values):
        if not isinstance(values, dict) or any(
            not isinstance(name, str) or not name
            for name in values
        ):
            raise ValueError("dependency namespace is invalid")
        self._values = dict(values)

    def get(self, name):
        if name not in self._values:
            raise ValueError(f"unknown dependency: {name}")
        return self._values[name]

    def set(self, name, value):
        if name not in self._values:
            raise ValueError(f"unknown dependency: {name}")
        self._values[name] = value

    def identity_snapshot(self):
        return {
            name: id(value)
            for name, value in sorted(self._values.items())
        }


class ModuleDependencyNamespace:
    def __init__(self, roots):
        if not isinstance(roots, dict) or any(
            not isinstance(name, str) or not name
            for name in roots
        ):
            raise ValueError("module dependency roots are invalid")
        self._roots = dict(roots)

    def _resolve_parent(self, name):
        parts = name.split(".")
        if len(parts) < 2 or parts[0] not in self._roots:
            raise ValueError(f"unknown dependency: {name}")
        parent = self._roots[parts[0]]
        for part in parts[1:-1]:
            if not hasattr(parent, part):
                raise ValueError(f"unknown dependency: {name}")
            parent = getattr(parent, part)
        attribute = parts[-1]
        if not hasattr(parent, attribute):
            raise ValueError(f"unknown dependency: {name}")
        return parent, attribute

    def get(self, name):
        parent, attribute = self._resolve_parent(name)
        return getattr(parent, attribute)

    def set(self, name, value):
        parent, attribute = self._resolve_parent(name)
        setattr(parent, attribute, value)

    def identity_snapshot(self):
        raise RuntimeError(
            "module dependency namespace requires explicit names"
        )


class TemporaryModuleRegistry:
    def __init__(self, replacements):
        replacements = dict(replacements)
        if any(
            not isinstance(name, str)
            or not name
            or not isinstance(module, ModuleType)
            for name, module in replacements.items()
        ):
            raise ValueError("temporary module replacements are invalid")
        self.replacements = replacements
        self.originals = {}
        self.missing = set()
        self.installed = False
        self.restoration_complete = False

    def __enter__(self):
        if self.installed:
            raise RuntimeError(
                "temporary module registry is already installed"
            )
        for name, replacement in self.replacements.items():
            if name in sys.modules:
                self.originals[name] = sys.modules[name]
            else:
                self.missing.add(name)
            sys.modules[name] = replacement
        self.installed = True
        return self

    def __exit__(self, exc_type, exc, traceback):
        try:
            for name, original in self.originals.items():
                sys.modules[name] = original
            for name in self.missing:
                sys.modules.pop(name, None)
            self.restoration_complete = all(
                sys.modules.get(name) is original
                for name, original in self.originals.items()
            ) and all(
                name not in sys.modules for name in self.missing
            )
            if not self.restoration_complete:
                raise RuntimeError(
                    "temporary module registry restoration failed"
                )
        finally:
            self.installed = False
        return False


def remove_new_tinyvllm_modules(
    before,
    *,
    preserve=(),
):
    before = set(before)
    preserve = set(preserve)
    removed = []
    for name in sorted(set(sys.modules) - before):
        if (
            name == "tinyvllm"
            or name.startswith("tinyvllm.")
        ) and name not in preserve:
            sys.modules.pop(name, None)
            removed.append(name)
    return tuple(removed)


def constructed_transport_module_preserve_names(
    *,
    model_runner_module,
):
    envelope_type = getattr(
        model_runner_module,
        "ModelRunnerCommandEnvelope",
        None,
    )
    if not isinstance(envelope_type, type):
        raise ValueError(
            "constructed transport envelope type is invalid"
        )
    module_name = getattr(envelope_type, "__module__", "")
    if module_name != "tinyvllm.engine.model_runner_command_ack":
        raise ValueError(
            "constructed transport module name is invalid"
        )
    transport_module = sys.modules.get(module_name)
    if (
        transport_module is None
        or getattr(
            transport_module,
            "ModelRunnerCommandEnvelope",
            None,
        )
        is not envelope_type
    ):
        raise RuntimeError(
            "constructed transport module identity is invalid"
        )
    return (module_name,)


def restore_constructed_transport_module_identity(
    *,
    model_runner_module,
    transport_module,
):
    envelope_type = getattr(
        model_runner_module,
        "ModelRunnerCommandEnvelope",
        None,
    )
    if not isinstance(envelope_type, type):
        raise ValueError(
            "constructed transport envelope type is invalid"
        )
    module_name = getattr(envelope_type, "__module__", "")
    if module_name != "tinyvllm.engine.model_runner_command_ack":
        raise ValueError(
            "constructed transport module name is invalid"
        )
    if (
        not isinstance(transport_module, ModuleType)
        or transport_module.__name__ != module_name
        or getattr(
            transport_module,
            "ModelRunnerCommandEnvelope",
            None,
        )
        is not envelope_type
    ):
        raise ValueError(
            "constructed transport module is invalid"
        )
    sys.modules[module_name] = transport_module
    restored = (
        sys.modules.get(module_name) is transport_module
        and getattr(
            sys.modules[module_name],
            "ModelRunnerCommandEnvelope",
            None,
        )
        is envelope_type
    )
    if not restored:
        raise RuntimeError(
            "constructed transport module restoration failed"
        )
    return {
        "module_name": module_name,
        "restored": True,
        "envelope_class_identity": True,
    }


def build_inert_tp4_config(*, model, torch_dtype):
    if not isinstance(model, str) or not model:
        raise ValueError("model must be a non-empty path")
    return SimpleNamespace(
        model=model,
        hf_config=SimpleNamespace(torch_dtype=torch_dtype),
        tensor_parallel_size=4,
        enforce_eager=True,
        cpu_offload=False,
        cpu_offload_num_layers=-1,
        kv_quant_bits=0,
        am_compact_blocks=0,
        kv_offload_mvp0=False,
        multi_sequence_cuda_graphs=False,
        multi_sequence_cuda_graph_batch_allowlist=(2, 4, 8),
        multi_sequence_cuda_graph_min_observations=3,
        multi_sequence_cuda_graph_max_entries=8,
        multi_sequence_cuda_graph_max_static_bytes=64 * 1024 * 1024,
        multi_sequence_cuda_graph_max_reserved_bytes=512 * 1024 * 1024,
        multi_sequence_cuda_graph_max_total_capture_ns=5_000_000_000,
        multi_sequence_cuda_graph_max_single_capture_ns=2_000_000_000,
        kvcache_block_size=256,
        quantization=None,
        quant_group_size=128,
        act_quant_bits=0,
        act_quant_skip_first=0,
        act_quant_skip_last=0,
        act_quant_skip_layers=None,
        smoothquant_scale_path=None,
        eos=-1,
    )


def build_inert_tp4_config_class(
    *,
    expected_model,
    torch_dtype,
    ledger,
):
    if not isinstance(ledger, DependencyLedger):
        raise ValueError("ledger is invalid")
    template = build_inert_tp4_config(
        model=expected_model,
        torch_dtype=torch_dtype,
    )
    definitions = []
    for name, value in vars(template).items():
        if name == "model":
            definitions.append((name, str))
            continue
        definitions.append((
            name,
            object,
            field(default_factory=lambda value=value: value),
        ))

    def post_init(self):
        if self.model != expected_model:
            raise ValueError("constructed Engine model path changed")
        ledger.record(
            "Config",
            arguments={"model": self.model},
            result=self,
        )

    return make_dataclass(
        "_InertTP4Config",
        definitions,
        namespace={"__post_init__": post_init},
    )


@dataclass(frozen=True)
class DependencyCall:
    sequence: int
    dependency: str
    rank: int | None
    arguments: dict
    result_identity: str


class InertConstructorDependencyCapsule:
    def __init__(
        self,
        *,
        namespace,
        replacements,
        allowed_names,
    ):
        if not all(
            callable(getattr(namespace, name, None))
            for name in ("get", "set")
        ):
            raise ValueError("namespace is invalid")
        replacements = dict(replacements)
        allowed_names = set(allowed_names)
        if set(replacements) != allowed_names:
            raise ValueError(
                "replacement allowlist mismatch: "
                f"replacement={sorted(replacements)!r}, "
                f"allowed={sorted(allowed_names)!r}"
            )
        for name in replacements:
            namespace.get(name)
        self.namespace = namespace
        self.replacements = replacements
        self.allowed_names = frozenset(allowed_names)
        self.installed = False
        self.restoration_complete = False
        self._originals = {}

    def __enter__(self):
        if self.installed:
            raise RuntimeError(
                "constructor dependency capsule is already installed"
            )
        self._originals = {
            name: self.namespace.get(name)
            for name in self.replacements
        }
        for name, replacement in self.replacements.items():
            self.namespace.set(name, replacement)
        self.installed = True
        self.restoration_complete = False
        return self

    def __exit__(self, exc_type, exc, traceback):
        try:
            for name, original in self._originals.items():
                self.namespace.set(name, original)
            self.restoration_complete = all(
                self.namespace.get(name) is original
                for name, original in self._originals.items()
            )
            if not self.restoration_complete:
                raise RuntimeError(
                    "constructor dependency restoration failed"
                )
        finally:
            self.installed = False
        return False


class DependencyLedger:
    def __init__(self):
        self._sequence = count()
        self.calls = []

    def record(
        self,
        dependency,
        *,
        rank=None,
        arguments=None,
        result=None,
    ):
        if not isinstance(dependency, str) or not dependency:
            raise ValueError("dependency name is invalid")
        call = DependencyCall(
            sequence=next(self._sequence),
            dependency=dependency,
            rank=rank,
            arguments=dict(arguments or {}),
            result_identity=(
                "none"
                if result is None
                else f"{type(result).__module__}.{type(result).__qualname__}:{id(result)}"
            ),
        )
        self.calls.append(call)
        return result

    def counts(self):
        return dict(sorted(Counter(
            call.dependency for call in self.calls
        ).items()))


class _InertReceiver:
    def __init__(self, *, ledger):
        self.ledger = ledger
        self.rank = None
        self.closed = False

    def close(self):
        if not self.closed:
            self.closed = True
            self.ledger.record(
                "receiver.close",
                rank=self.rank,
                result=self,
            )


class _InertSender:
    def __init__(self, *, ledger):
        self.ledger = ledger
        self.rank = None
        self.closed = False

    def close(self):
        if not self.closed:
            self.closed = True
            self.ledger.record(
                "sender.close",
                rank=self.rank,
                result=self,
            )


class _InertEvent:
    def __init__(self):
        self.set_calls = 0
        self.clear_calls = 0
        self.wait_calls = 0

    def set(self):
        self.set_calls += 1

    def clear(self):
        self.clear_calls += 1

    def wait(self):
        self.wait_calls += 1


class _DeferredModelRunnerProcess:
    def __init__(self, *, target, args, ledger):
        if not callable(target):
            raise ValueError("process target must be callable")
        if not isinstance(args, tuple):
            raise ValueError("process args must be a tuple")
        self.target = target
        self.args = args
        self.ledger = ledger
        self.started = False
        self.runner = None
        self.exitcode = None

    def start(self):
        if self.started:
            raise RuntimeError("inert process is already started")
        self.started = True
        self.ledger.record(
            "process.start",
            rank=self.args[1] if len(self.args) > 1 else None,
            result=self,
        )

    def construct_deferred(self):
        if not self.started:
            raise RuntimeError("inert process is not started")
        if self.runner is None:
            self.runner = self.target(*self.args)
            self.ledger.record(
                "runner.deferred_construct",
                rank=self.args[1] if len(self.args) > 1 else None,
                result=self.runner,
            )
        return self.runner

    def is_alive(self):
        return self.started and self.exitcode is None

    def join(self, timeout=None):
        self.exitcode = 0


class InertSpawnContext:
    def __init__(self, *, ledger, expected_process_target=None):
        if not isinstance(ledger, DependencyLedger):
            raise ValueError("ledger is invalid")
        if (
            expected_process_target is not None
            and not callable(expected_process_target)
        ):
            raise ValueError("expected process target is invalid")
        self.ledger = ledger
        self.expected_process_target = expected_process_target
        self.processes = []

    def Pipe(self, duplex=False):
        if duplex is not False:
            raise ValueError("inert pipe must be one-way")
        receiver = _InertReceiver(ledger=self.ledger)
        sender = _InertSender(ledger=self.ledger)
        self.ledger.record(
            "context.Pipe",
            arguments={"duplex": False},
            result=receiver,
        )
        return receiver, sender

    def Event(self):
        event = _InertEvent()
        return self.ledger.record(
            "context.Event",
            result=event,
        )

    def Process(self, *, target, args):
        if (
            self.expected_process_target is not None
            and target is not self.expected_process_target
        ):
            raise RuntimeError(
                "inert process target must be the exact production "
                "ModelRunner class"
            )
        process = _DeferredModelRunnerProcess(
            target=target,
            args=args,
            ledger=self.ledger,
        )
        self.processes.append(process)
        return self.ledger.record(
            "context.Process",
            rank=args[1] if len(args) > 1 else None,
            result=process,
        )


class _InertSharedMemoryHandle:
    def __init__(self, *, registry, name, buffer):
        self._registry = registry
        self.name = name
        self.buf = buffer
        self.closed = False

    def close(self):
        if not self.closed:
            self.closed = True
            self._registry.ledger.record(
                "SharedMemory.close",
                arguments={"name": self.name},
                result=self,
            )

    def unlink(self):
        self._registry.unlink(self.name)


class InertSharedMemoryRegistry:
    def __init__(self, *, ledger):
        if not isinstance(ledger, DependencyLedger):
            raise ValueError("ledger is invalid")
        self.ledger = ledger
        self._buffers = {}

    def open(self, *, name, create=False, size=0):
        if not isinstance(name, str) or not name:
            raise ValueError("shared-memory name is invalid")
        if create:
            if name in self._buffers:
                raise RuntimeError("inert shared memory already exists")
            if (
                isinstance(size, bool)
                or not isinstance(size, int)
                or size <= 0
            ):
                raise ValueError("shared-memory size is invalid")
            self._buffers[name] = bytearray(size)
            dependency = "SharedMemory.create"
        else:
            if name not in self._buffers:
                raise RuntimeError("inert shared memory does not exist")
            dependency = "SharedMemory.attach"
        handle = _InertSharedMemoryHandle(
            registry=self,
            name=name,
            buffer=self._buffers[name],
        )
        return self.ledger.record(
            dependency,
            arguments={
                "name": name,
                "create": bool(create),
                "size": int(size),
            },
            result=handle,
        )

    def unlink(self, name):
        if name not in self._buffers:
            raise RuntimeError("inert shared memory does not exist")
        del self._buffers[name]
        self.ledger.record(
            "SharedMemory.unlink",
            arguments={"name": name},
        )

    def resource_names(self):
        return tuple(sorted(self._buffers))


class InProcessAckCollector:
    def __init__(
        self,
        receivers,
        *,
        processes,
        envelope_reader,
        ack_factory,
        ledger,
    ):
        if not isinstance(receivers, tuple) or not receivers:
            raise ValueError("receivers must be a non-empty tuple")
        if not isinstance(processes, tuple) or not processes:
            raise ValueError("processes must be a non-empty tuple")
        if not callable(envelope_reader):
            raise ValueError("envelope_reader must be callable")
        if not callable(ack_factory):
            raise ValueError("ack_factory must be callable")
        if not isinstance(ledger, DependencyLedger):
            raise ValueError("ledger is invalid")
        receiver_ranks = tuple(rank for rank, _ in receivers)
        process_ranks = tuple(process.args[1] for process in processes)
        if receiver_ranks != process_ranks:
            raise ValueError("receiver and process ranks do not match")
        self.receivers = receivers
        self.processes = processes
        self.envelope_reader = envelope_reader
        self.ack_factory = ack_factory
        self.ledger = ledger
        self.collect_calls = 0
        self.last_envelope = None
        self.last_acknowledgements = ()
        for process in self.processes:
            process.construct_deferred()
        self.ledger.record(
            "ack_collector.construct",
            arguments={"ranks": list(receiver_ranks)},
            result=self,
        )

    def collect(
        self,
        command_id,
        *,
        expected_ranks,
        timeout_s,
        is_rank_alive,
    ):
        if tuple(expected_ranks) != tuple(
            process.args[1] for process in self.processes
        ):
            raise ValueError("acknowledgement ranks are invalid")
        if (
            isinstance(timeout_s, bool)
            or not isinstance(timeout_s, (int, float))
            or timeout_s <= 0
        ):
            raise ValueError("timeout_s must be positive")
        if not callable(is_rank_alive):
            raise ValueError("is_rank_alive must be callable")
        envelope = self.envelope_reader()
        self.last_envelope = envelope
        if envelope.command_id != command_id:
            raise RuntimeError("command id mismatch")
        if (
            envelope.method_name
            != "bind_published_qwen35_loaded_checkpoint_candidate"
            or envelope.args != ()
            or envelope.requires_ack is not True
        ):
            raise RuntimeError(
                "constructed runtime command must be zero-payload"
            )
        acknowledgements = []
        for process in self.processes:
            rank = process.args[1]
            if not is_rank_alive(rank):
                raise RuntimeError(f"worker rank is not alive: {rank}")
            runner = process.construct_deferred()
            method = getattr(runner, envelope.method_name, None)
            if not callable(method):
                raise RuntimeError(
                    f"worker method is missing: rank={rank}"
                )
            result = method(*envelope.args)
            self.ledger.record(
                "worker.bind.invoke",
                rank=rank,
                result=result,
            )
            acknowledgements.append(self.ack_factory(
                command_id=envelope.command_id,
                rank=rank,
                status="ok",
                result=result,
            ))
        self.collect_calls += 1
        self.last_acknowledgements = tuple(acknowledgements)
        self.ledger.record(
            "ack_collector.collect",
            arguments={
                "command_id": command_id,
                "expected_ranks": list(expected_ranks),
                "timeout_s": float(timeout_s),
            },
            result=tuple(acknowledgements),
        )
        return tuple(acknowledgements)


class _InertTokenizer:
    def __init__(self, *, eos_token_id):
        self.eos_token_id = eos_token_id


class _InertAutoTokenizer:
    def __init__(self, *, ledger, eos_token_id):
        self.ledger = ledger
        self.eos_token_id = eos_token_id

    def from_pretrained(self, model, use_fast=True):
        if use_fast is not True:
            raise ValueError("inert tokenizer requires use_fast=True")
        tokenizer = _InertTokenizer(
            eos_token_id=self.eos_token_id
        )
        return self.ledger.record(
            "AutoTokenizer.from_pretrained",
            arguments={"model": model, "use_fast": True},
            result=tokenizer,
        )


class _InertSchedulerSentinel:
    def __init__(self, config):
        self.config = config


@dataclass
class ConstructedRuntimeScope:
    engine: object
    runners_by_rank: dict
    ledger: DependencyLedger
    context: InertSpawnContext
    shared_memory_registry: InertSharedMemoryRegistry
    original_dependency_identities: dict
    restored_dependency_identities: dict
    restoration_complete: bool

    def constructor_evidence(self):
        counts = self.ledger.counts()
        runner_ranks = [
            call.rank
            for call in self.ledger.calls
            if call.dependency == "dist.init_process_group"
        ]
        return {
            "engine_constructor_count": counts.get(
                "LLMEngine.__init__",
                0,
            ),
            "runner_constructor_count": len(runner_ranks),
            "runner_constructor_ranks": runner_ranks,
            "dependency_call_counts": counts,
            "original_dependency_identities": dict(
                self.original_dependency_identities
            ),
            "restored_dependency_identities": dict(
                self.restored_dependency_identities
            ),
            "restoration_complete": self.restoration_complete,
        }

    def close_inert_resources(self):
        for process in self.context.processes:
            if process.is_alive():
                process.join()
        for _, receiver in getattr(
            self.engine,
            "model_runner_ack_receivers",
            (),
        ):
            receiver.close()
        for _, sender in getattr(
            self.engine,
            "model_runner_ack_parent_senders",
            (),
        ):
            sender.close()
        for runner in self.runners_by_rank.values():
            shared = getattr(runner, "shm", None)
            if shared is not None:
                shared.close()
        for name in self.shared_memory_registry.resource_names():
            self.shared_memory_registry.unlink(name)


@dataclass
class ConstructedRankCandidateState:
    rank: int
    runner: object
    target: object
    request: object
    candidate: object
    payload: dict
    transfer_evidence: dict


_BIND_OBSERVATIONS = {}


def _weak_reference(value, name):
    try:
        return weakref.ref(value)
    except TypeError as error:
        raise RuntimeError(
            f"constructed runtime object is not weak-referenceable: {name}"
        ) from error


def release_constructed_runtime(live_scope):
    if (
        not isinstance(live_scope, dict)
        or set(live_scope)
        != {"constructed_scope", "rank_states"}
    ):
        raise ValueError("constructed live scope is invalid")
    scope = live_scope["constructed_scope"]
    states = live_scope["rank_states"]
    if (
        getattr(scope, "released", False)
        or not isinstance(states, list)
        or [state.rank for state in states] != [0, 1, 2, 3]
    ):
        raise ValueError("constructed live scope state is invalid")

    def execute_cleanup():
        references = {}

        def retain(name, value):
            if value is not None:
                references[name] = _weak_reference(value, name)

        retain("engine", scope.engine)
        retain(
            "ack_collector",
            getattr(scope.engine, "model_runner_ack_collector", None),
        )
        retain("scheduler", getattr(scope.engine, "scheduler", None))
        retain("tokenizer", getattr(scope.engine, "tokenizer", None))
        for index, process in enumerate(scope.context.processes):
            retain(f"process_{index}", process)
        for rank, receiver in getattr(
            scope.engine,
            "model_runner_ack_receivers",
            (),
        ):
            retain(f"receiver_{rank}", receiver)
        for rank, sender in getattr(
            scope.engine,
            "model_runner_ack_parent_senders",
            (),
        ):
            retain(f"sender_{rank}", sender)
        for state in states:
            rank = state.rank
            runner = state.runner
            candidate = state.candidate
            owner = candidate.owner
            retain(f"runner_{rank}", runner)
            retain(
                f"slot_{rank}",
                getattr(
                    runner,
                    "qwen35_loaded_checkpoint_candidate_slot",
                    None,
                ),
            )
            retain(f"request_{rank}", state.request)
            retain(f"candidate_{rank}", candidate)
            retain(f"owner_{rank}", owner)
            retain(
                f"runtime_bridge_{rank}",
                getattr(owner, "runtime_bridge", None),
            )
            retain(
                f"runtime_identity_{rank}",
                getattr(
                    runner,
                    "qwen35_hybrid_prefix_runtime_identity",
                    None,
                ),
            )
            retain(f"model_{rank}", getattr(owner, "model", None))
            retain(f"pool_{rank}", state.pool)
            retain(f"target_{rank}", state.target)

        release_rank_order = []
        for state in reversed(states):
            release_rank_order.append(state.rank)
            clear_error = None
            for tensor in reversed(tuple(state.selected.values())):
                try:
                    with state.no_grad():
                        tensor.zero_()
                except Exception as error:
                    if clear_error is None:
                        clear_error = error
            try:
                state.require_identity_unchanged(
                    state.registered,
                    state.identity_snapshot,
                )
                if any(
                    int(tensor.count_nonzero().item())
                    for tensor in state.selected.values()
                ):
                    raise RuntimeError(
                        "constructed selected destination clear failed"
                    )
                if any(
                    not tensor.equal(
                        state.non_selected_values[id(tensor)]
                    )
                    for tensor in state.registered
                    if id(tensor) not in state.selected_ids
                ):
                    raise RuntimeError(
                        "constructed non-selected tensor changed"
                    )
                if not state.pool_unchanged(
                    state.pool,
                    state.pool_snapshot,
                ):
                    raise RuntimeError(
                        "constructed pool state changed"
                    )
            except Exception as error:
                if clear_error is None:
                    clear_error = error
            if clear_error is not None:
                raise RuntimeError(
                    "constructed rank cleanup failed: "
                    f"rank={state.rank}"
                ) from clear_error

        scope.close_inert_resources()
        process_exitcodes = [
            process.exitcode for process in scope.context.processes
        ]
        receiver_closed = [
            receiver.closed
            for _, receiver in getattr(
                scope.engine,
                "model_runner_ack_receivers",
                (),
            )
        ]
        sender_closed = [
            sender.closed
            for _, sender in getattr(
                scope.engine,
                "model_runner_ack_parent_senders",
                (),
            )
        ]
        shared_handles_closed = [
            bool(getattr(getattr(runner, "shm", None), "closed", False))
            for runner in scope.runners_by_rank.values()
        ]
        expected_process_exitcodes = (
            [0, 0, 0]
            if scope.context.processes
            else []
        )
        expected_channel_closed = (
            [True, True, True]
            if receiver_closed or sender_closed
            else []
        )
        expected_shared_closed = (
            [True, True, True, True]
            if shared_handles_closed
            and all(
                getattr(runner, "shm", None) is not None
                for runner in scope.runners_by_rank.values()
            )
            else shared_handles_closed
        )
        if (
            process_exitcodes != expected_process_exitcodes
            or receiver_closed != expected_channel_closed
            or sender_closed != expected_channel_closed
            or shared_handles_closed != expected_shared_closed
            or scope.shared_memory_registry.resource_names()
        ):
            raise RuntimeError(
                "constructed inert resource cleanup is incomplete"
            )

        for state in reversed(states):
            runner = state.runner
            slot = getattr(
                runner,
                "qwen35_loaded_checkpoint_candidate_slot",
                None,
            )
            if slot is not None and hasattr(slot, "_publication"):
                slot._publication = None
            for name in (
                "model",
                "qwen35_hybrid_model_owner",
                "hybrid_state_runtime_bridge",
                "qwen35_hybrid_prefix_runtime_identity",
                "qwen35_hybrid_prefix_runtime_identity_owner",
                "qwen35_loaded_checkpoint_candidate_slot",
                "scheduler",
                "sampler",
                "shm",
                "event",
                "ack_sender",
            ):
                if hasattr(runner, name):
                    setattr(runner, name, None)
            state.runner = None
            state.target = None
            state.request = None
            state.candidate = None
            state.registered = ()
            state.identity_snapshot = None
            state.pool_snapshot = None
            state.selected.clear()
            state.selected_ids.clear()
            state.non_selected_values.clear()
            state.pool = None

        collector = getattr(
            scope.engine,
            "model_runner_ack_collector",
            None,
        )
        if collector is not None:
            collector.last_envelope = None
            collector.receivers = ()
            collector.processes = ()
            collector.envelope_reader = None
            collector.ack_factory = None
        for process in scope.context.processes:
            process.runner = None
            process.target = None
            process.args = ()
        for name in (
            "model_runner",
            "model_runner_ack_collector",
            "model_runner_ack_receivers",
            "model_runner_ack_parent_senders",
            "ps",
            "scheduler",
            "tokenizer",
        ):
            if hasattr(scope.engine, name):
                setattr(
                    scope.engine,
                    name,
                    () if name in {
                        "model_runner_ack_receivers",
                        "model_runner_ack_parent_senders",
                        "ps",
                    } else None,
                )
        scope.runners_by_rank.clear()
        scope.context.processes.clear()
        scope.engine = None
        scope.released = True
        states.clear()
        live_scope.clear()
        return {
            "release_rank_order": release_rank_order,
            "all_selected_destinations_zero_after_clear": True,
            "non_selected_tensors_unchanged": True,
            "tensor_identity_preserved": True,
            "pool_unchanged": True,
            "all_inert_resources_closed": True,
            "production_exit_call_count": 0,
        }, references

    evidence, references = execute_cleanup()
    gc.collect()
    collected = {
        name: reference() is None
        for name, reference in sorted(references.items())
    }
    if not all(collected.values()):
        escaped = sorted(
            name for name, value in collected.items() if not value
        )
        raise RuntimeError(
            "constructed runtime objects escaped cleanup: "
            + ", ".join(escaped)
        )
    evidence["collected_private_objects"] = collected
    evidence["all_private_objects_collected"] = True
    return evidence


def _dependency_identity_snapshot(namespace, names):
    return {
        name: id(namespace.get(name))
        for name in sorted(names)
    }


def _read_envelope_from_runner(runner):
    shared = getattr(runner, "shm", None)
    if shared is None:
        raise RuntimeError("rank zero inert shared memory is missing")
    size = int.from_bytes(shared.buf[0:4], "little")
    if size <= 0 or size + 4 > len(shared.buf):
        raise RuntimeError("inert shared-memory envelope is invalid")
    return pickle.loads(bytes(shared.buf[4:size + 4]))


def _resolve_ack_factory():
    try:
        module = importlib.import_module(
            "tinyvllm.engine.model_runner_command_ack"
        )
    except ModuleNotFoundError:
        return lambda **kwargs: SimpleNamespace(**kwargs)
    return module.ModelRunnerCommandAck


def _ensure_namespace_package(name, path):
    current = sys.modules.get(name)
    if current is not None:
        return current
    module = ModuleType(name)
    module.__path__ = [str(path)]
    sys.modules[name] = module
    return module


def _load_source_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load source module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _load_sibling_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load sibling module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def build_import_time_stubs():
    def inert_module(name, *, types=(), functions=()):
        module = ModuleType(name)
        for type_name in types:
            setattr(
                module,
                type_name,
                type(f"_ImportInert{type_name}", (), {}),
            )
        for function_name in functions:
            setattr(
                module,
                function_name,
                lambda *args, **kwargs: None,
            )
        return module

    flash_policy = inert_module(
        "tinyvllm.engine.flash_attn_split_policy",
        types=("FlashAttentionSplitInputs",),
        functions=("build_flash_attn_263_graph_identity",),
    )
    kv_cartridge = inert_module(
        "tinyvllm.engine.kv_cartridge",
        functions=(
            "compress_decode_block_table_rows",
            "should_use_kv_cartridge",
        ),
    )
    prefix_engine_publication = inert_module(
        "tinyvllm.engine.qwen35_hybrid_prefix_engine_publication",
        types=("Qwen35HybridPrefixEnginePublicationCoordinator",),
    )
    prefix_engine_restore = inert_module(
        "tinyvllm.engine.qwen35_hybrid_prefix_engine_restore",
        types=("Qwen35HybridPrefixEngineRestoreCoordinator",),
    )
    prefix_owner = inert_module(
        "tinyvllm.engine.qwen35_hybrid_prefix_owner",
        types=("Qwen35HybridPrefixRestoreOwner",),
        functions=("build_qwen35_hybrid_prefix_restore_owner",),
    )
    prefix_publication_ticket = inert_module(
        "tinyvllm.engine.qwen35_hybrid_prefix_publication_ticket",
        types=(
            "Qwen35HybridPrefixPublicationParticipant",
            "Qwen35HybridPrefixPublicationPayload",
        ),
    )
    prefix_restore_ticket = inert_module(
        "tinyvllm.engine.qwen35_hybrid_prefix_restore_ticket",
        types=("Qwen35HybridPrefixRestoreParticipant",),
    )
    prefix_source_publication = inert_module(
        "tinyvllm.engine.qwen35_hybrid_prefix_source_publication",
        types=("Qwen35HybridPrefixSourcePublisher",),
    )
    scheduler = inert_module(
        "tinyvllm.engine.scheduler",
        types=("Scheduler",),
    )
    sequence = inert_module(
        "tinyvllm.engine.sequence",
        types=("Sequence",),
    )
    linear = ModuleType("tinyvllm.layers.linear")
    for name in (
        "LinearBase",
        "ColumnParallelLinear",
        "HeadPairedColumnParallelLinear",
        "KVHeadParallelLinear",
        "MergedColumnParallelLinear",
        "QKVParallelLinear",
        "ReplicatedHeadPairedColumnParallelLinear",
        "ReplicatedKVHeadParallelLinear",
        "ReplicatedLinear",
        "ReplicatedLocalOutputLinear",
        "ReplicatedMergedColumnParallelLinear",
        "ReplicatedWeightRowParallelLinear",
        "RowParallelLinear",
        "SegmentedColumnParallelLinear",
    ):
        setattr(
            linear,
            name,
            type(f"_ImportInert{name}", (), {}),
        )
    linear.get_quant_method = lambda *args, **kwargs: None
    linear.set_quant_config = lambda *args, **kwargs: None

    sampler = ModuleType("tinyvllm.layers.sampler")
    sampler.Sampler = type("_ImportInertSampler", (), {})

    qwen3 = ModuleType("tinyvllm.models.qwen3")
    qwen3.Qwen3ForCausalLM = type(
        "_ImportInertQwen3ForCausalLM",
        (),
        {},
    )

    sampling_params = inert_module(
        "tinyvllm.sampling_params",
        types=("SamplingParams",),
    )
    cpu_offload = ModuleType("tinyvllm.utils.cpu_offload")
    cpu_offload.apply_cpu_offload = lambda *args, **kwargs: None

    loader = ModuleType("tinyvllm.utils.loader")
    loader.load_model = lambda *args, **kwargs: None

    return {
        flash_policy.__name__: flash_policy,
        kv_cartridge.__name__: kv_cartridge,
        prefix_engine_publication.__name__: prefix_engine_publication,
        prefix_engine_restore.__name__: prefix_engine_restore,
        prefix_owner.__name__: prefix_owner,
        prefix_publication_ticket.__name__: prefix_publication_ticket,
        prefix_restore_ticket.__name__: prefix_restore_ticket,
        prefix_source_publication.__name__: prefix_source_publication,
        scheduler.__name__: scheduler,
        sequence.__name__: sequence,
        linear.__name__: linear,
        sampler.__name__: sampler,
        qwen3.__name__: qwen3,
        sampling_params.__name__: sampling_params,
        cpu_offload.__name__: cpu_offload,
        loader.__name__: loader,
    }


def load_production_engine_modules_without_cuda_import(
    *,
    source_root,
    torch_module,
):
    root = Path(source_root)
    if bool(torch_module.cuda.is_initialized()):
        raise RuntimeError(
            "CUDA must be uninitialized before production source import"
        )
    for name, path in (
        ("tinyvllm", root / "tinyvllm"),
        ("tinyvllm.engine", root / "tinyvllm/engine"),
        ("tinyvllm.layers", root / "tinyvllm/layers"),
        ("tinyvllm.models", root / "tinyvllm/models"),
        ("tinyvllm.utils", root / "tinyvllm/utils"),
    ):
        _ensure_namespace_package(name, path)
    stubs = build_import_time_stubs()
    if set(stubs) != IMPORT_TIME_STUB_MODULES:
        raise RuntimeError("import-time stub module set is invalid")
    model_runner_name = (
        "_qwen35_constructed_runtime_production_model_runner"
    )
    llm_engine_name = (
        "_qwen35_constructed_runtime_production_llm_engine"
    )
    with TemporaryModuleRegistry(stubs) as registry:
        model_runner = _load_source_module(
            model_runner_name,
            root / "tinyvllm/engine/model_runner.py",
        )
        sys.modules["tinyvllm.engine.model_runner"] = model_runner
        try:
            llm_engine = _load_source_module(
                llm_engine_name,
                root / "tinyvllm/engine/llm_engine.py",
            )
        finally:
            sys.modules.pop("tinyvllm.engine.model_runner", None)
    if not registry.restoration_complete:
        raise RuntimeError(
            "import-time dependency restoration is incomplete"
        )
    if bool(torch_module.cuda.is_initialized()):
        raise RuntimeError(
            "production source import initialized CUDA"
        )
    return llm_engine, model_runner


def construct_engine_runtime_under_inert_capsule(
    *,
    llm_engine_module,
    model_runner_module,
    torch_module,
    model,
):
    engine_class = getattr(llm_engine_module, "LLMEngine", None)
    runner_class = getattr(llm_engine_module, "ModelRunner", None)
    if not isinstance(engine_class, type):
        raise ValueError("LLMEngine class identity is invalid")
    if not isinstance(runner_class, type):
        raise ValueError("ModelRunner class identity is invalid")
    if getattr(model_runner_module, "ModelRunner", None) is not runner_class:
        raise ValueError("ModelRunner module identity is inconsistent")

    ledger = DependencyLedger()
    config_class = build_inert_tp4_config_class(
        expected_model=model,
        torch_dtype=getattr(torch_module, "bfloat16"),
        ledger=ledger,
    )
    context = InertSpawnContext(
        ledger=ledger,
        expected_process_target=runner_class,
    )
    shared_registry = InertSharedMemoryRegistry(ledger=ledger)
    registered_exit_callbacks = []
    model_placeholders = {}
    default_dtype = getattr(torch_module, "float32")

    def get_context(name):
        if name != "spawn":
            raise ValueError("constructed Engine context must be spawn")
        return ledger.record(
            "mp.get_context",
            arguments={"name": name},
            result=context,
        )

    def init_process_group(**kwargs):
        rank = kwargs.get("rank")
        expected = {
            "backend": "nccl",
            "init_method": kwargs.get("init_method"),
            "world_size": 4,
            "rank": rank,
        }
        if (
            kwargs != expected
            or not isinstance(rank, int)
            or isinstance(rank, bool)
            or rank not in range(4)
            or not isinstance(kwargs["init_method"], str)
            or not kwargs["init_method"].startswith(
                "tcp://localhost:"
            )
        ):
            raise ValueError("init_process_group call is invalid")
        ledger.record(
            "dist.init_process_group",
            rank=rank,
            arguments=kwargs,
        )

    def barrier():
        ledger.record("dist.barrier")

    def set_device(rank):
        if rank not in range(4):
            raise ValueError("CUDA device rank is invalid")
        ledger.record(
            "torch.cuda.set_device",
            rank=rank,
            arguments={"rank": rank},
        )

    def get_default_dtype():
        return ledger.record(
            "torch.get_default_dtype",
            result=default_dtype,
        )

    def set_default_dtype(value):
        ledger.record(
            "torch.set_default_dtype",
            arguments={"value": repr(value)},
        )

    def set_default_device(value):
        if value not in {"cuda", "cpu"}:
            raise ValueError("default device is invalid")
        ledger.record(
            "torch.set_default_device",
            arguments={"value": value},
        )

    def set_quant_config(*args):
        ledger.record(
            "set_quant_config",
            arguments={"values": [repr(value) for value in args]},
        )

    def model_factory(hf_config):
        placeholder = SimpleNamespace(
            hf_config=hf_config,
            constructor_placeholder=True,
        )
        index = len(model_placeholders)
        model_placeholders[index] = placeholder
        return ledger.record(
            "Qwen3ForCausalLM",
            rank=index,
            result=placeholder,
        )

    def load_model(model_object, model_path, **kwargs):
        if model_path != model:
            raise ValueError("load_model path changed")
        ledger.record(
            "load_model",
            arguments={
                "model_path": model_path,
                "kwargs": {
                    name: repr(value)
                    for name, value in sorted(kwargs.items())
                },
            },
        )

    def forbidden_cpu_offload(*args, **kwargs):
        ledger.record("apply_cpu_offload")
        raise RuntimeError("CPU offload is forbidden in constructor gate")

    def sampler_factory():
        return ledger.record(
            "Sampler",
            result=SimpleNamespace(inert_sampler=True),
        )

    def shared_memory_factory(**kwargs):
        return shared_registry.open(**kwargs)

    def warmup_model(runner):
        ledger.record(
            "ModelRunner.warmup_model",
            rank=runner.rank,
        )

    def allocate_kv_cache(runner):
        ledger.record(
            "ModelRunner.allocate_kv_cache",
            rank=runner.rank,
        )

    def forbidden_capture(runner):
        ledger.record(
            "ModelRunner.capture_cudagraph",
            rank=runner.rank,
        )
        raise RuntimeError(
            "CUDA graph capture is forbidden in constructor gate"
        )

    def worker_loop(runner):
        ledger.record(
            "ModelRunner.loop",
            rank=runner.rank,
        )

    def scheduler_factory(config_value):
        sentinel = _InertSchedulerSentinel(config_value)
        return ledger.record(
            "Scheduler",
            result=sentinel,
        )

    auto_tokenizer = _InertAutoTokenizer(
        ledger=ledger,
        eos_token_id=0,
    )

    def atexit_register(callback):
        registered_exit_callbacks.append(callback)
        return ledger.record(
            "atexit.register",
            arguments={
                "callback": getattr(callback, "__qualname__", repr(callback))
            },
            result=callback,
        )

    collector_holder = {}

    def collector_factory(receivers):
        collector = InProcessAckCollector(
            receivers,
            processes=tuple(context.processes),
            envelope_reader=lambda: _read_envelope_from_runner(
                collector_holder["engine"].model_runner
            ),
            ack_factory=_resolve_ack_factory(),
            ledger=ledger,
        )
        collector_holder["collector"] = collector
        return collector

    roots = {
        "llm_engine": llm_engine_module,
        "model_runner": model_runner_module,
        "ModelRunner": runner_class,
    }
    namespace = ModuleDependencyNamespace(roots)
    replacements = {
        "llm_engine.Config": config_class,
        "llm_engine.ModelRunnerCommandAckCollector": collector_factory,
        "llm_engine.AutoTokenizer": auto_tokenizer,
        "llm_engine.Scheduler": scheduler_factory,
        "llm_engine.mp.get_context": get_context,
        "llm_engine.atexit.register": atexit_register,
        "model_runner.dist.init_process_group": init_process_group,
        "model_runner.dist.barrier": barrier,
        "model_runner.torch.cuda.set_device": set_device,
        "model_runner.torch.get_default_dtype": get_default_dtype,
        "model_runner.torch.set_default_dtype": set_default_dtype,
        "model_runner.torch.set_default_device": set_default_device,
        "model_runner.set_quant_config": set_quant_config,
        "model_runner.Qwen3ForCausalLM": model_factory,
        "model_runner.load_model": load_model,
        "model_runner.apply_cpu_offload": forbidden_cpu_offload,
        "model_runner.Sampler": sampler_factory,
        "model_runner.SharedMemory": shared_memory_factory,
        "ModelRunner.warmup_model": warmup_model,
        "ModelRunner.allocate_kv_cache": allocate_kv_cache,
        "ModelRunner.capture_cudagraph": forbidden_capture,
        "ModelRunner.loop": worker_loop,
    }
    validate_constructor_replacement_names(replacements)
    original = _dependency_identity_snapshot(
        namespace,
        replacements,
    )
    capsule = InertConstructorDependencyCapsule(
        namespace=namespace,
        replacements=replacements,
        allowed_names=CONSTRUCTOR_REPLACEMENT_ALLOWLIST,
    )
    with capsule:
        ledger.record(
            "LLMEngine.__init__",
            result=engine_class,
        )
        engine = engine_class(model)
        collector_holder["engine"] = engine
    restored = _dependency_identity_snapshot(
        namespace,
        replacements,
    )
    if original != restored or not capsule.restoration_complete:
        raise RuntimeError(
            "constructor dependency restoration is incomplete"
        )
    if type(engine) is not engine_class:
        raise RuntimeError("constructed Engine class identity changed")
    runners = {
        0: engine.model_runner,
        **{
            process.args[1]: process.runner
            for process in context.processes
        },
    }
    if (
        set(runners) != set(range(4))
        or any(type(runner) is not runner_class for runner in runners.values())
        or len({id(runner) for runner in runners.values()}) != 4
    ):
        raise RuntimeError(
            "constructed ModelRunner class identities are invalid"
        )
    ranks = [
        call.rank
        for call in ledger.calls
        if call.dependency == "dist.init_process_group"
    ]
    if ranks != [0, 1, 2, 3]:
        raise RuntimeError(
            f"constructed ModelRunner rank order is invalid: {ranks!r}"
        )
    if len(registered_exit_callbacks) != 1:
        raise RuntimeError("atexit registration count is invalid")
    return ConstructedRuntimeScope(
        engine=engine,
        runners_by_rank=runners,
        ledger=ledger,
        context=context,
        shared_memory_registry=shared_registry,
        original_dependency_identities=original,
        restored_dependency_identities=restored,
        restoration_complete=True,
    )


def transfer_candidate_to_constructed_runner(
    *,
    runner,
    expected_runner_type,
    candidate,
    expected_rank,
):
    if type(runner) is not expected_runner_type:
        raise ValueError("constructed runner class is invalid")
    if (
        isinstance(expected_rank, bool)
        or not isinstance(expected_rank, int)
        or expected_rank not in range(4)
        or runner.rank != expected_rank
    ):
        raise ValueError("constructed runner rank is invalid")
    if runner.world_size != 4:
        raise ValueError("constructed runner world size is invalid")
    owner = getattr(candidate, "owner", None)
    model = getattr(owner, "model", None)
    if owner is None or model is None:
        raise ValueError("candidate owner model is invalid")
    placeholder = getattr(runner, "model", None)
    if placeholder is None or placeholder is model:
        raise ValueError(
            "constructed runner model placeholder is invalid"
        )
    slot = getattr(
        runner,
        "qwen35_loaded_checkpoint_candidate_slot",
        None,
    )
    if slot is None or getattr(slot, "candidate", None) is not None:
        raise ValueError("constructed runner publication slot is invalid")
    publish = getattr(
        runner,
        "publish_qwen35_loaded_checkpoint_candidate",
        None,
    )
    if not callable(publish):
        raise ValueError(
            "constructed runner publication method is invalid"
        )
    runner.model = model
    published = publish(candidate)
    if (
        published is not candidate
        or getattr(slot, "candidate", None) is not candidate
    ):
        raise RuntimeError(
            "constructed runner candidate publication failed"
        )
    return {
        "rank": expected_rank,
        "exact_runner_class": True,
        "world_size": 4,
        "constructor_placeholder_replaced": True,
        "candidate_published": True,
        "candidate_bound_before_engine_dispatch": False,
    }


def rebind_constructed_runner_candidate_types(
    *,
    model_runner_module,
    candidate_type,
    owner_type,
    publication_slot_type,
    identity_binder,
):
    if not isinstance(candidate_type, type):
        raise ValueError("candidate type is invalid")
    if not isinstance(owner_type, type):
        raise ValueError("owner type is invalid")
    if not isinstance(publication_slot_type, type):
        raise ValueError("publication slot type is invalid")
    if not callable(identity_binder):
        raise ValueError("identity binder is invalid")
    bindings = {
        "Qwen35LoadedCheckpointCandidate": candidate_type,
        "Qwen35HybridModelOwner": owner_type,
        "Qwen35HybridModelOwnerPublicationSlot": (
            publication_slot_type
        ),
        "_bind_qwen35_hybrid_prefix_runtime_identity": (
            identity_binder
        ),
    }
    original = {
        name: id(getattr(model_runner_module, name))
        for name in bindings
    }
    for name, value in bindings.items():
        setattr(model_runner_module, name, value)
    rebound = {
        name: id(getattr(model_runner_module, name))
        for name in bindings
    }
    if any(
        getattr(model_runner_module, name) is not value
        for name, value in bindings.items()
    ):
        raise RuntimeError(
            "constructed runner candidate type rebinding failed"
        )
    return {
        "candidate_type_rebound": True,
        "owner_type_rebound": True,
        "publication_slot_type_rebound": True,
        "identity_binder_rebound": True,
        "original_identities": original,
        "rebound_identities": rebound,
    }


def validate_prebind_payload_against_pristine_rank(
    payload,
    pristine_row,
    *,
    rank,
):
    if (
        not isinstance(payload, dict)
        or not isinstance(pristine_row, dict)
        or pristine_row.get("tp_rank") != rank
    ):
        raise ValueError("pristine rank identity is invalid")
    for name, detail in (
        ("binding_destination_sha256", "binding payload"),
        ("phase_destination_sha256", "phase payload"),
        ("aggregate_destination_sha256", "aggregate payload"),
        ("alias_groups", "alias payload"),
        ("loader_stats", "loader statistics"),
    ):
        if payload.get(name) != pristine_row.get(name):
            raise ValueError(
                f"pristine rank {detail} mismatch: rank={rank}"
            )
    anticipated_identity = payload.get("anticipated_identity")
    if (
        not isinstance(anticipated_identity, dict)
        or anticipated_identity.get("model_fingerprint")
        != pristine_row.get("model_manifest_sha256")
        or anticipated_identity.get("layout_fingerprint")
        != pristine_row.get("layout_fingerprint")
        or anticipated_identity.get("dtype")
        != pristine_row.get("dtype")
    ):
        raise ValueError(
            f"pristine rank anticipated identity mismatch: rank={rank}"
        )
    return payload


def prepare_candidate_for_constructed_runner(
    *,
    runner,
    expected_runner_type,
    rank,
    model_fingerprint,
    scope_kwargs,
    pristine_row,
    pristine_validator,
):
    if not isinstance(scope_kwargs, dict):
        raise ValueError("candidate scope kwargs are invalid")
    required = {
        "private_graph_factory",
        "candidate_validator",
        "payload_recorder",
    }
    if not required.issubset(scope_kwargs):
        raise ValueError("candidate scope components are incomplete")
    private_graph_factory = scope_kwargs["private_graph_factory"]
    candidate_validator = scope_kwargs["candidate_validator"]
    payload_recorder = scope_kwargs["payload_recorder"]
    if not all(
        callable(value)
        for value in (
            private_graph_factory,
            candidate_validator,
            payload_recorder,
        )
    ):
        raise ValueError("candidate scope component is invalid")
    target, request, installed_loader = private_graph_factory()
    if not callable(installed_loader):
        raise ValueError("candidate loader is invalid")
    candidate = installed_loader(request)
    validation = candidate_validator(
        candidate=candidate,
        target=target,
        model_fingerprint=model_fingerprint,
    )
    if not isinstance(validation, dict):
        raise ValueError("candidate validation result is invalid")
    payload = dict(validation)
    recorded = payload_recorder(
        candidate=candidate,
        target=target,
        model_fingerprint=model_fingerprint,
    )
    if not isinstance(recorded, dict):
        raise ValueError("candidate payload record is invalid")
    payload.update(recorded)
    payload["loader_stats"] = {
        name: getattr(candidate.stats, name)
        for name in (
            "assigned_bindings",
            "source_tensors",
            "shard_count",
            "loaded_bytes",
            "peak_source_bytes",
        )
    }
    transfer = transfer_candidate_to_constructed_runner(
        runner=runner,
        expected_runner_type=expected_runner_type,
        candidate=candidate,
        expected_rank=rank,
    )
    if pristine_validator is not None:
        if not callable(pristine_validator):
            raise ValueError("pristine validator is invalid")
        pristine_validator(
            payload,
            pristine_row,
            rank=rank,
        )
    return ConstructedRankCandidateState(
        rank=rank,
        runner=runner,
        target=target,
        request=request,
        candidate=candidate,
        payload=payload,
        transfer_evidence=transfer,
    )


def prepare_real_candidate_for_constructed_runner(
    *,
    runner,
    expected_runner_type,
    rank,
    scope_kwargs,
    model_fingerprint,
    pristine_row,
    pristine_validator,
    helpers,
    pool_helpers,
    torch_runtime,
):
    private_graph_factory = scope_kwargs["private_graph_factory"]
    candidate_validator = scope_kwargs["candidate_validator"]
    payload_recorder = scope_kwargs["payload_recorder"]
    target, request, installed_loader = private_graph_factory()
    model = target.assembly.packed.model
    pool = target.pool
    registered = helpers._registered_tensors(model)
    identity_snapshot = helpers._snapshot_identity(registered)
    pool_snapshot = pool_helpers._snapshot_pool(pool)
    selected = {}
    for binding in target.binding_plan.bindings:
        selected.setdefault(id(binding.destination), binding.destination)
    selected_ids = set(selected)
    non_selected_values = {
        id(tensor): tensor.detach().clone()
        for tensor in registered
        if id(tensor) not in selected_ids
    }
    with torch_runtime.no_grad():
        for tensor in selected.values():
            tensor.zero_()
    candidate = installed_loader(request)
    payload = dict(candidate_validator(
        candidate=candidate,
        target=target,
        model_fingerprint=model_fingerprint,
    ))
    payload.update(payload_recorder(
        candidate=candidate,
        target=target,
        model_fingerprint=model_fingerprint,
    ))
    payload["alias_groups"] = helpers._alias_groups(
        target.binding_plan
    )
    payload["loader_stats"] = {
        name: getattr(candidate.stats, name)
        for name in (
            "assigned_bindings",
            "source_tensors",
            "shard_count",
            "loaded_bytes",
            "peak_source_bytes",
        )
    }
    payload["anticipated_identity"] = {
        "model_fingerprint": candidate.model_fingerprint,
        "layout_fingerprint": candidate.owner.pool.layout.fingerprint,
        "dtype": "bfloat16",
    }
    pristine_validator(
        payload,
        pristine_row,
        rank=rank,
    )
    transfer = transfer_candidate_to_constructed_runner(
        runner=runner,
        expected_runner_type=expected_runner_type,
        candidate=candidate,
        expected_rank=rank,
    )
    state = ConstructedRankCandidateState(
        rank=rank,
        runner=runner,
        target=target,
        request=request,
        candidate=candidate,
        payload=payload,
        transfer_evidence=transfer,
    )
    state.registered = registered
    state.identity_snapshot = identity_snapshot
    state.pool_snapshot = pool_snapshot
    state.selected = selected
    state.selected_ids = selected_ids
    state.non_selected_values = non_selected_values
    state.pool = pool
    state.require_identity_unchanged = (
        helpers._require_identity_unchanged
    )
    state.pool_unchanged = pool_helpers._pool_unchanged
    state.no_grad = torch_runtime.no_grad
    return state


def bind_constructed_runtime_candidates(
    *,
    engine,
    expected_engine_type,
    timeout_s,
):
    if type(engine) is not expected_engine_type:
        raise ValueError("constructed Engine class is invalid")
    method = getattr(
        engine,
        "bind_qwen35_loaded_checkpoint_candidates",
        None,
    )
    if not callable(method):
        raise ValueError("constructed Engine binding method is invalid")
    collector = getattr(
        engine,
        "model_runner_ack_collector",
        None,
    )
    collect_before = int(getattr(collector, "collect_calls", 0))
    rows_before = getattr(
        engine,
        "qwen35_loaded_checkpoint_candidate_binding_rows",
        None,
    )
    rows = method(timeout_s=timeout_s)
    if (
        not isinstance(rows, tuple)
        or len(rows) != 4
        or tuple(row.get("participant_id") for row in rows)
        != (0, 1, 2, 3)
        or any(
            not isinstance(row, dict)
            or row.get("operation")
            != "bind_loaded_checkpoint_candidate"
            or row.get("status") != "bound"
            or row.get("detail") != ""
            for row in rows
        )
    ):
        raise RuntimeError(
            "constructed Engine all-rank binding rows are invalid"
        )
    identities = {
        (
            row.get("model_fingerprint"),
            row.get("layout_fingerprint"),
            row.get("dtype"),
        )
        for row in rows
    }
    if len(identities) != 1:
        raise RuntimeError(
            "constructed Engine all-rank identity is heterogeneous"
        )
    identity = next(iter(identities))
    configuration = getattr(
        engine,
        "qwen35_loaded_checkpoint_candidate_binding_configuration",
        None,
    )
    if configuration != (*identity, float(timeout_s)):
        raise RuntimeError(
            "constructed Engine completion configuration is invalid"
        )
    stored_rows = getattr(
        engine,
        "qwen35_loaded_checkpoint_candidate_binding_rows",
        None,
    )
    if stored_rows is not rows:
        raise RuntimeError(
            "constructed Engine completion rows are not canonical"
        )
    collect_after = int(getattr(collector, "collect_calls", 0))
    envelope = getattr(collector, "last_envelope", None)
    acknowledgements = tuple(
        getattr(collector, "last_acknowledgements", ())
    )
    first_observation = rows_before is None
    if first_observation:
        if (
            envelope is None
            or getattr(envelope, "method_name", None)
            != "bind_published_qwen35_loaded_checkpoint_candidate"
            or getattr(envelope, "args", None) != ()
            or getattr(envelope, "requires_ack", None) is not True
        ):
            raise RuntimeError(
                "constructed Engine command is not zero-payload"
            )
        if (
            len(acknowledgements) != 3
            or tuple(ack.rank for ack in acknowledgements)
            != (1, 2, 3)
            or any(
                ack.command_id != envelope.command_id
                or ack.status != "ok"
                or ack.error_type != ""
                or ack.error_detail != ""
                or ack.result != rows[ack.rank]
                for ack in acknowledgements
            )
        ):
            raise RuntimeError(
                "constructed Engine acknowledgements are invalid"
            )
        exact_repeat_zero_dispatch = None
        _BIND_OBSERVATIONS[id(engine)] = {
            "rows": rows,
            "collect_calls": collect_after,
        }
    else:
        prior = _BIND_OBSERVATIONS.get(id(engine))
        if (
            prior is None
            or prior["rows"] is not rows
            or collect_after != collect_before
            or collect_after != prior["collect_calls"]
        ):
            raise RuntimeError(
                "constructed Engine exact repeat dispatched again"
            )
        exact_repeat_zero_dispatch = True
    return {
        "rows": rows,
        "configuration": configuration,
        "command_envelope": {
            "command_id": envelope.command_id,
            "method_name": envelope.method_name,
            "args": list(envelope.args),
            "requires_ack": envelope.requires_ack,
        },
        "worker_acknowledgements": [
            {
                "command_id": ack.command_id,
                "rank": ack.rank,
                "status": ack.status,
                "result": ack.result,
                "error_type": ack.error_type,
                "error_detail": ack.error_detail,
            }
            for ack in acknowledgements
        ],
        "zero_payload_command": True,
        "exact_repeat_zero_dispatch": exact_repeat_zero_dispatch,
    }


def run_constructor_smoke(*, source_root, model):
    source_contract = inspect_constructed_runtime_source_contract(
        source_root
    )
    import torch
    cuda_initialized_before = bool(torch.cuda.is_initialized())
    llm_engine_module, model_runner_module = (
        load_production_engine_modules_without_cuda_import(
            source_root=source_root,
            torch_module=torch,
        )
    )
    scope = construct_engine_runtime_under_inert_capsule(
        llm_engine_module=llm_engine_module,
        model_runner_module=model_runner_module,
        torch_module=torch,
        model=model,
    )
    evidence = scope.constructor_evidence()
    validate_constructor_evidence(evidence)
    cuda_initialized_after = bool(torch.cuda.is_initialized())
    if cuda_initialized_before or cuda_initialized_after:
        raise RuntimeError(
            "CUDA initialized during constructed runtime smoke"
        )
    class_identity = {
        "engine": {
            "module": type(scope.engine).__module__,
            "qualname": type(scope.engine).__qualname__,
            "exact_class": (
                type(scope.engine) is llm_engine_module.LLMEngine
            ),
        },
        "runners": [
            {
                "rank": rank,
                "module": type(scope.runners_by_rank[rank]).__module__,
                "qualname": type(
                    scope.runners_by_rank[rank]
                ).__qualname__,
                "exact_class": (
                    type(scope.runners_by_rank[rank])
                    is model_runner_module.ModelRunner
                ),
            }
            for rank in range(4)
        ],
    }
    if (
        class_identity["engine"]["exact_class"] is not True
        or any(
            row["exact_class"] is not True
            for row in class_identity["runners"]
        )
    ):
        raise RuntimeError("constructed class identity is invalid")
    constructor_ledger = [
        {
            "sequence": call.sequence,
            "dependency": call.dependency,
            "rank": call.rank,
            "arguments": call.arguments,
            "result_identity": call.result_identity,
        }
        for call in scope.ledger.calls
    ]
    result = {
        "status": "PASS",
        "source_contract": source_contract,
        "class_identity": class_identity,
        "constructor_evidence": evidence,
        "constructor_ledger": constructor_ledger,
        "cuda_initialized_before": cuda_initialized_before,
        "cuda_initialized_after": cuda_initialized_after,
        "os_process_count": 0,
        "os_shared_memory_count": 0,
        "production_scheduler_constructor_count": 0,
        "production_tokenizer_io_count": 0,
        "atexit_registration_count": 0,
    }
    scope.close_inert_resources()
    return result


def run_constructed_binding_smoke(
    *,
    source_root,
    checkpoint_dir,
    prerequisite_oracle,
):
    root = Path(source_root)
    source_contract = inspect_constructed_runtime_source_contract(root)
    import torch

    modules_before = set(sys.modules)
    llm_engine_module, model_runner_module = (
        load_production_engine_modules_without_cuda_import(
            source_root=root,
            torch_module=torch,
        )
    )
    scope = construct_engine_runtime_under_inert_capsule(
        llm_engine_module=llm_engine_module,
        model_runner_module=model_runner_module,
        torch_module=torch,
        model=checkpoint_dir,
    )
    constructor_evidence = scope.constructor_evidence()
    validate_constructor_evidence(constructor_evidence)
    constructor_ledger = [
        {
            "sequence": call.sequence,
            "dependency": call.dependency,
            "rank": call.rank,
            "arguments": call.arguments,
            "result_identity": call.result_identity,
        }
        for call in scope.ledger.calls
    ]
    transport_preserve_names = (
        constructed_transport_module_preserve_names(
            model_runner_module=model_runner_module,
        )
    )
    transport_module = sys.modules[transport_preserve_names[0]]
    removed_modules = remove_new_tinyvllm_modules(
        modules_before,
        preserve=transport_preserve_names,
    )
    live_gate = _load_sibling_module(
        "_qwen35_constructed_runtime_live_base",
        root
        / "tools"
        / "qwen35_tp4_live_concurrent_candidate_ownership_preflight.py",
    )
    serial_gate = live_gate.serial_gate
    oracle = live_gate._load_prerequisite_oracle(
        prerequisite_oracle
    )
    pristine_rows = oracle["producer_rows"]
    serial_gate._install_namespace_packages(root)
    torch_runtime = serial_gate._load_runtime_module("torch")
    torch_runtime.set_num_threads(8)

    def backend_factory(diagnostics):
        class Backend(torch_runtime.nn.Module):
            def forward(self, *_args, **_kwargs):
                diagnostics["attention_forward_count"] += 1
                raise AssertionError(
                    "attention backend must not execute"
                )

        return Backend()

    components = (
        serial_gate.build_tp4_real_candidate_producer_components(
            source_root=root,
            module_loader=serial_gate._load_runtime_module,
            torch_runtime=torch_runtime,
            backend_factory=backend_factory,
        )
    )
    streaming_module = serial_gate._load_runtime_module(
        "tinyvllm.models.qwen35_checkpoint_streaming"
    )
    owner_module = serial_gate._load_runtime_module(
        "tinyvllm.engine.qwen35_hybrid_model_owner"
    )
    identity_module = serial_gate._load_runtime_module(
        "tinyvllm.engine.qwen35_hybrid_prefix_runtime_identity"
    )
    publication_module = serial_gate._load_runtime_module(
        "tinyvllm.engine.qwen35_hybrid_model_publication"
    )
    rebind_evidence = rebind_constructed_runner_candidate_types(
        model_runner_module=model_runner_module,
        candidate_type=(
            streaming_module.Qwen35LoadedCheckpointCandidate
        ),
        owner_type=owner_module.Qwen35HybridModelOwner,
        publication_slot_type=(
            publication_module.Qwen35HybridModelOwnerPublicationSlot
        ),
        identity_binder=(
            identity_module
            .bind_qwen35_hybrid_prefix_runtime_identity
        ),
    )
    for runner in scope.runners_by_rank.values():
        runner.qwen35_loaded_checkpoint_candidate_slot = (
            publication_module.Qwen35HybridModelOwnerPublicationSlot()
        )
    helpers = (
        serial_gate.real_binding_gate.load_publish_gate.publication_gate
        .publication.ownership.loader_core
    )
    pool_helpers = (
        serial_gate.real_binding_gate.load_publish_gate.publication_gate
        .publication
    )
    states = []
    for rank in range(4):
        runtime = (
            serial_gate.assemble_tp4_real_candidate_producer_runtime(
                checkpoint_dir=checkpoint_dir,
                tensor_parallel_size=4,
                tensor_parallel_rank=rank,
                status_reader=live_gate._read_self_memory,
                components=components,
            )
        )
        state = prepare_real_candidate_for_constructed_runner(
            runner=scope.runners_by_rank[rank],
            expected_runner_type=model_runner_module.ModelRunner,
            rank=rank,
            scope_kwargs=runtime["scope_kwargs"],
            model_fingerprint=(
                serial_gate.APPROVED_MODEL_MANIFEST_SHA256
            ),
            pristine_row=pristine_rows[rank],
            pristine_validator=(
                validate_prebind_payload_against_pristine_rank
            ),
            helpers=helpers,
            pool_helpers=pool_helpers,
            torch_runtime=torch_runtime,
        )
        states.append(state)
    transport_restoration = (
        restore_constructed_transport_module_identity(
            model_runner_module=model_runner_module,
            transport_module=transport_module,
        )
    )
    first = bind_constructed_runtime_candidates(
        engine=scope.engine,
        expected_engine_type=llm_engine_module.LLMEngine,
        timeout_s=0.25,
    )
    repeat = bind_constructed_runtime_candidates(
        engine=scope.engine,
        expected_engine_type=llm_engine_module.LLMEngine,
        timeout_s=0.25,
    )
    return {
        "status": "PASS",
        "source_contract": source_contract,
        "constructor_evidence": constructor_evidence,
        "class_identity": {
            "engine_module": type(scope.engine).__module__,
            "engine_qualname": type(scope.engine).__qualname__,
            "engine_exact_class": (
                type(scope.engine) is llm_engine_module.LLMEngine
            ),
            "runner_module": (
                model_runner_module.ModelRunner.__module__
            ),
            "runner_qualname": (
                model_runner_module.ModelRunner.__qualname__
            ),
            "runner_exact_class_by_rank": [
                type(scope.runners_by_rank[rank])
                is model_runner_module.ModelRunner
                for rank in range(4)
            ],
        },
        "constructor_ledger": constructor_ledger,
        "removed_import_modules": list(removed_modules),
        "preserved_transport_modules": list(
            transport_preserve_names
        ),
        "transport_restoration": transport_restoration,
        "type_rebinding": rebind_evidence,
        "forbidden_counters": {
            name: 0 for name in FORBIDDEN_COUNTER_NAMES
        },
        "rank_payloads": [
            {
                "rank": state.rank,
                **state.payload,
                "transfer_evidence": state.transfer_evidence,
            }
            for state in states
        ],
        "first_binding": first,
        "repeat_binding": repeat,
        "cuda_initialized_after": bool(
            torch_runtime.cuda.is_initialized()
        ),
        "_live_scope": {
            "constructed_scope": scope,
            "rank_states": states,
        },
    }


def _parser():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run")
    run.add_argument("--run-tag", required=True)
    run.add_argument(
        "--source-root",
        default=str(Path(__file__).resolve().parents[1]),
    )
    run.add_argument("--prerequisite-oracle", required=True)
    internal = subparsers.add_parser("internal-run")
    internal.add_argument("--source-root", required=True)
    internal.add_argument("--checkpoint-dir", required=True)
    internal.add_argument("--prerequisite-oracle", required=True)
    internal.add_argument("--run-dir", required=True)
    internal.add_argument("--run-tag", required=True)
    validate = subparsers.add_parser("validate")
    validate.add_argument("--run-dir", required=True)
    validate.add_argument("--source-root", required=True)
    validate.add_argument("--prerequisite-oracle", required=True)
    smoke = subparsers.add_parser("internal-constructor-smoke")
    smoke.add_argument("--source-root", required=True)
    smoke.add_argument("--model", required=True)
    binding = subparsers.add_parser("internal-binding-smoke")
    binding.add_argument("--source-root", required=True)
    binding.add_argument("--checkpoint-dir", required=True)
    binding.add_argument("--prerequisite-oracle", required=True)
    return parser


def main(argv=None):
    arguments = _parser().parse_args(argv)
    if arguments.command == "run":
        result = execute_remote_constructed_gate(
            source_root=arguments.source_root,
            prerequisite_oracle=arguments.prerequisite_oracle,
            run_tag=arguments.run_tag,
        )
        print(json.dumps(result, sort_keys=True))
        return 0
    if arguments.command == "internal-run":
        result = run_source_bound_constructed_gate(
            source_root=arguments.source_root,
            checkpoint_dir=arguments.checkpoint_dir,
            prerequisite_oracle=arguments.prerequisite_oracle,
            run_dir=arguments.run_dir,
            run_tag=arguments.run_tag,
        )
        print(json.dumps(result, sort_keys=True))
        return 0
    if arguments.command == "validate":
        verifier = _load_sibling_module(
            "_qwen35_constructed_runtime_validate",
            Path(arguments.source_root)
            / "tools"
            / (
                "verify_qwen35_constructed_engine_"
                "model_runner_ownership_gate.py"
            ),
        )
        result = verifier.verify_run(
            arguments.run_dir,
            source_root=arguments.source_root,
            prerequisite_oracle=arguments.prerequisite_oracle,
        )
        print(f"PASS, {result['checks']} checks")
        return 0
    if arguments.command == "internal-constructor-smoke":
        result = run_constructor_smoke(
            source_root=arguments.source_root,
            model=arguments.model,
        )
        print(json.dumps(result, sort_keys=True))
        return 0
    if arguments.command == "internal-binding-smoke":
        result = run_constructed_binding_smoke(
            source_root=arguments.source_root,
            checkpoint_dir=arguments.checkpoint_dir,
            prerequisite_oracle=arguments.prerequisite_oracle,
        )
        result.pop("_live_scope")
        print(json.dumps(result, sort_keys=True))
        return 0
    raise AssertionError("unreachable")


if __name__ == "__main__":
    raise SystemExit(main())
