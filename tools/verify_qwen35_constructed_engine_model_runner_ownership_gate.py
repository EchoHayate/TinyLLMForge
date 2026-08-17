from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path


RESULT_NAME = "constructed_engine_model_runner_ownership.json"
MANIFEST_NAME = "source_manifest.json"
RESULT_SCHEMA = "qwen35.constructed-engine-model-runner-ownership.v1"
PROVENANCE = (
    "real-checkpoint-derived-constructed-engine-model-runner-ownership"
)
CLAIM_BOUNDARY = "no-scheduler-step-forward-or-inference"
PREREQUISITE_ORACLE_SHA256 = (
    "d750d664219378c234a2127b708ec191feb9b2c9f1f2902c47d0ad5dc152d3ef"
)
APPROVED_MODEL_MANIFEST_SHA256 = (
    "3e650a908234771c3cf1ac4e20c4d38fe69982efedaf4a3e631ad0b14aad7dd0"
)
REMOTE_TARGET = "sitian@10.232.195.203"
REMOTE_PYTHON = (
    "/data00/home/sitian/sitian-workspace01/tllm/env/bin/python"
)
GATE_SOURCE = (
    "tools/qwen35_constructed_engine_model_runner_ownership_preflight.py"
)
VERIFIER_SOURCE = (
    "tools/verify_qwen35_constructed_engine_model_runner_ownership_gate.py"
)
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
FORBIDDEN_COUNTERS = {
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
}
MEMORY_CEILINGS_KIB = {
    "process_total_vmhwm_increment": 12582912,
    "ready_vmrss": 8388608,
    "host_mem_available_decrease": 12582912,
    "minimum_preflight_mem_available": 16777216,
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


class VerificationError(ValueError):
    pass


class Checker:
    def __init__(self):
        self.count = 0

    def require(self, condition, detail):
        self.count += 1
        if not condition:
            raise VerificationError(detail)


def _canonical(value):
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _sha256(payload):
    return hashlib.sha256(payload).hexdigest()


def _read(path):
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise VerificationError(
            f"invalid JSON: {Path(path).name}"
        ) from error
    if not isinstance(value, dict):
        raise VerificationError(f"{Path(path).name} must be an object")
    return value


def _is_sha(value):
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _read_oracle(path, checker):
    payload = Path(path).read_bytes()
    checker.require(
        _sha256(payload) == PREREQUISITE_ORACLE_SHA256,
        "pristine oracle SHA mismatch",
    )
    oracle = json.loads(payload)
    checker.require(
        isinstance(oracle, dict)
        and oracle.get("status") == "PASS"
        and oracle.get("model_manifest_sha256")
        == APPROVED_MODEL_MANIFEST_SHA256,
        "pristine oracle identity mismatch",
    )
    rows = oracle.get("producer_rows")
    checker.require(
        isinstance(rows, list)
        and [row.get("tp_rank") for row in rows] == [0, 1, 2, 3],
        "pristine oracle rows are invalid",
    )
    return oracle


def _verify_sources(record, manifest, oracle, source_root, checker):
    hashes = record.get("source_file_sha256")
    expected_names = (
        set(oracle.get("source_file_sha256", {}))
        | set(EXPECTED_FILE_SHA256)
        | set(CONSTRUCTOR_RUNTIME_FILE_SHA256)
        | set(DIRECT_GATE_FILE_SHA256)
        | {GATE_SOURCE, VERIFIER_SOURCE}
    )
    checker.require(
        isinstance(hashes, dict) and set(hashes) == expected_names,
        "source closure mismatch",
    )
    checker.require(
        manifest.get("source_file_sha256") == hashes,
        "manifest source hashes mismatch",
    )
    checker.require(
        tuple(hashes) == tuple(sorted(hashes)),
        "source ordering mismatch",
    )
    for name, expected in hashes.items():
        checker.require(_is_sha(expected), f"source SHA invalid: {name}")
        path = Path(source_root) / name
        checker.require(path.is_file(), f"missing source: {name}")
        checker.require(
            _sha256(path.read_bytes()) == expected,
            f"source hash mismatch: {name}",
        )
    tree = _sha256(_canonical(hashes))
    checker.require(
        record.get("source_tree_sha256") == tree,
        "result source tree mismatch",
    )
    checker.require(
        manifest.get("source_tree_sha256") == tree,
        "manifest source tree mismatch",
    )


def _verify_payload(row, pristine, rank, checker):
    checker.require(row.get("rank") == rank, f"rank mismatch: {rank}")
    for name in (
        "binding_destination_sha256",
        "phase_destination_sha256",
        "aggregate_destination_sha256",
        "alias_groups",
        "loader_stats",
    ):
        checker.require(
            row.get(name) == pristine.get(name),
            f"pristine payload mismatch: rank={rank}, field={name}",
        )
    checker.require(
        row.get("binding_hash_count") == 320
        and len(row["binding_destination_sha256"]) == 320,
        f"binding payload count mismatch: rank={rank}",
    )
    checker.require(
        row.get("phase_hash_count") == 26
        and len(row["phase_destination_sha256"]) == 26,
        f"phase payload count mismatch: rank={rank}",
    )
    checker.require(
        len(row.get("alias_groups", ())) == 24,
        f"alias group count mismatch: rank={rank}",
    )
    identity = row.get("anticipated_identity")
    checker.require(
        isinstance(identity, dict)
        and identity.get("model_fingerprint")
        == pristine.get("model_manifest_sha256")
        and identity.get("layout_fingerprint")
        == pristine.get("layout_fingerprint")
        and identity.get("dtype") == pristine.get("dtype"),
        f"anticipated identity mismatch: rank={rank}",
    )
    transfer = row.get("transfer_evidence")
    checker.require(
        isinstance(transfer, dict)
        and transfer.get("candidate_published") is True
        and transfer.get("candidate_bound_before_engine_dispatch")
        is False,
        f"candidate transfer mismatch: rank={rank}",
    )


def _verify_binding(record, checker):
    first = record.get("first_binding")
    repeat = record.get("repeat_binding")
    checker.require(
        isinstance(first, dict) and isinstance(repeat, dict),
        "binding evidence is invalid",
    )
    rows = first.get("rows")
    checker.require(
        isinstance(rows, list)
        and [row.get("participant_id") for row in rows]
        == [0, 1, 2, 3],
        "binding participants are invalid",
    )
    for rank, row in enumerate(rows):
        checker.require(
            row == {
                "participant_id": rank,
                "operation": "bind_loaded_checkpoint_candidate",
                "status": "bound",
                "model_fingerprint": APPROVED_MODEL_MANIFEST_SHA256,
                "layout_fingerprint": rows[0]["layout_fingerprint"],
                "dtype": "bfloat16",
                "detail": "",
            },
            f"binding row mismatch: rank={rank}",
        )
    checker.require(
        first.get("zero_payload_command") is True
        and first.get("exact_repeat_zero_dispatch") is None,
        "first binding dispatch mismatch",
    )
    envelope = first.get("command_envelope")
    checker.require(
        isinstance(envelope, dict)
        and isinstance(envelope.get("command_id"), int)
        and not isinstance(envelope.get("command_id"), bool)
        and envelope["command_id"] >= 0
        and envelope.get("method_name")
        == "bind_published_qwen35_loaded_checkpoint_candidate"
        and envelope.get("args") == []
        and envelope.get("requires_ack") is True,
        "zero-payload envelope mismatch",
    )
    acknowledgements = first.get("worker_acknowledgements")
    checker.require(
        isinstance(acknowledgements, list)
        and [row.get("rank") for row in acknowledgements] == [1, 2, 3],
        "worker acknowledgement inventory mismatch",
    )
    for row in acknowledgements:
        checker.require(
            row == {
                "command_id": envelope["command_id"],
                "rank": row["rank"],
                "status": "ok",
                "result": rows[row["rank"]],
                "error_type": "",
                "error_detail": "",
            },
            f"worker acknowledgement mismatch: rank={row['rank']}",
        )
    checker.require(
        repeat.get("zero_payload_command") is True
        and repeat.get("exact_repeat_zero_dispatch") is True
        and repeat.get("rows") == rows
        and repeat.get("configuration") == first.get("configuration"),
        "repeat binding mismatch",
    )
    checker.require(
        repeat.get("command_envelope") == envelope
        and repeat.get("worker_acknowledgements") == acknowledgements,
        "repeat binding mismatch",
    )
    checker.require(
        first.get("configuration")
        == [
            APPROVED_MODEL_MANIFEST_SHA256,
            rows[0]["layout_fingerprint"],
            "bfloat16",
            0.25,
        ],
        "binding configuration mismatch",
    )


def _verify_memory(memory, checker):
    checker.require(
        isinstance(memory, dict)
        and memory.get("memory_contract_passed") is True,
        "memory contract is invalid",
    )
    before = memory.get("process_before", {})
    ready = memory.get("process_ready", {})
    host_before = memory.get("host_before", {})
    host_ready = memory.get("host_ready", {})
    increment = ready.get("vmhwm_kib", 0) - before.get("vmhwm_kib", 0)
    decrease = (
        host_before.get("mem_available_kib", 0)
        - host_ready.get("mem_available_kib", 0)
    )
    checker.require(
        memory.get("process_total_vmhwm_increment_kib") == increment
        and increment
        <= MEMORY_CEILINGS_KIB["process_total_vmhwm_increment"],
        "process memory mismatch",
    )
    checker.require(
        ready.get("vmrss_kib", 0)
        <= MEMORY_CEILINGS_KIB["ready_vmrss"],
        "ready VmRSS ceiling exceeded",
    )
    checker.require(
        host_before.get("mem_available_kib", 0)
        >= MEMORY_CEILINGS_KIB["minimum_preflight_mem_available"],
        "host preflight memory is insufficient",
    )
    checker.require(
        memory.get("host_mem_available_decrease_kib") == decrease
        and decrease
        <= MEMORY_CEILINGS_KIB["host_mem_available_decrease"],
        "host memory decrease mismatch",
    )


def verify_run(run_dir, *, source_root, prerequisite_oracle):
    run_dir = Path(run_dir)
    inventory = {path.name for path in run_dir.iterdir() if path.is_file()}
    checker = Checker()
    checker.require(
        inventory == {RESULT_NAME, MANIFEST_NAME},
        "run inventory is not exact",
    )
    result_path = run_dir / RESULT_NAME
    manifest_path = run_dir / MANIFEST_NAME
    record = _read(result_path)
    manifest = _read(manifest_path)
    oracle = _read_oracle(prerequisite_oracle, checker)
    checker.require(
        record.get("schema_version") == RESULT_SCHEMA
        and record.get("status") == "PASS"
        and record.get("provenance") == PROVENANCE
        and record.get("claim_boundary") == CLAIM_BOUNDARY,
        "result identity mismatch",
    )
    checker.require(
        record.get("prerequisite_oracle_sha256")
        == PREREQUISITE_ORACLE_SHA256
        and manifest.get("prerequisite_oracle_sha256")
        == PREREQUISITE_ORACLE_SHA256,
        "prerequisite identity mismatch",
    )
    source_contract = record.get("source_contract", {})
    checker.require(
        source_contract.get("files") == EXPECTED_FILE_SHA256,
        "production file contract mismatch",
    )
    checker.require(
        source_contract.get("methods") == EXPECTED_METHOD_SHA256,
        "production method contract mismatch",
    )
    checker.require(
        source_contract.get("forbidden_execution_forms")
        == {
            "object_new": False,
            "constructor_ast_compile": False,
            "subclass_construction": False,
            "class_replacement": False,
        },
        "forbidden execution form mismatch",
    )
    checker.require(
        record.get("constructor_replacement_allowlist")
        == sorted(CONSTRUCTOR_REPLACEMENT_ALLOWLIST),
        "constructor replacement allowlist mismatch",
    )
    _verify_sources(record, manifest, oracle, source_root, checker)
    class_identity = record.get("class_identity", {})
    checker.require(
        class_identity.get("engine_module")
        == "_qwen35_constructed_runtime_production_llm_engine"
        and class_identity.get("engine_qualname") == "LLMEngine"
        and class_identity.get("engine_exact_class") is True
        and class_identity.get("runner_module")
        == "_qwen35_constructed_runtime_production_model_runner"
        and class_identity.get("runner_qualname") == "ModelRunner"
        and class_identity.get("runner_exact_class_by_rank")
        == [True, True, True, True],
        "exact class identity mismatch",
    )
    constructor = record.get("constructor_evidence", {})
    checker.require(
        constructor.get("engine_constructor_count") == 1
        and constructor.get("runner_constructor_count") == 4
        and constructor.get("runner_constructor_ranks") == [0, 1, 2, 3]
        and constructor.get("restoration_complete") is True
        and constructor.get("original_dependency_identities")
        == constructor.get("restored_dependency_identities"),
        "constructor evidence mismatch",
    )
    ledger = record.get("constructor_ledger")
    checker.require(
        isinstance(ledger, list)
        and [row.get("sequence") for row in ledger]
        == list(range(len(ledger))),
        "constructor ledger ordering mismatch",
    )
    checker.require(
        record.get("constructor_ledger_sha256")
        == _sha256(_canonical(ledger)),
        "constructor ledger hash mismatch",
    )
    ledger_counts = dict(sorted(Counter(
        row.get("dependency") for row in ledger
    ).items()))
    checker.require(
        ledger_counts == constructor.get("dependency_call_counts"),
        "constructor ledger counts mismatch",
    )
    payloads = record.get("rank_payloads")
    checker.require(
        isinstance(payloads, list) and len(payloads) == 4,
        "rank payload inventory mismatch",
    )
    checker.require(
        record.get("rank_payloads_sha256")
        == _sha256(_canonical(payloads)),
        "rank payload aggregate hash mismatch",
    )
    for rank in range(4):
        _verify_payload(
            payloads[rank],
            oracle["producer_rows"][rank],
            rank,
            checker,
        )
    _verify_binding(record, checker)
    checker.require(
        record.get("transport_restoration")
        == {
            "module_name": "tinyvllm.engine.model_runner_command_ack",
            "restored": True,
            "envelope_class_identity": True,
        },
        "transport restoration mismatch",
    )
    forbidden = record.get("forbidden_counters")
    checker.require(
        isinstance(forbidden, dict)
        and set(forbidden) == FORBIDDEN_COUNTERS
        and all(value == 0 for value in forbidden.values()),
        "forbidden counters mismatch",
    )
    cleanup = record.get("cleanup", {})
    checker.require(
        cleanup.get("release_rank_order") == [3, 2, 1, 0]
        and cleanup.get("all_selected_destinations_zero_after_clear")
        is True
        and cleanup.get("non_selected_tensors_unchanged") is True
        and cleanup.get("tensor_identity_preserved") is True
        and cleanup.get("pool_unchanged") is True
        and cleanup.get("all_inert_resources_closed") is True
        and cleanup.get("production_exit_call_count") == 0
        and cleanup.get("all_private_objects_collected") is True
        and all(cleanup.get("collected_private_objects", {}).values()),
        "cleanup evidence mismatch",
    )
    checker.require(
        record.get("cuda_initialized_after") is False,
        "CUDA initialization mismatch",
    )
    _verify_memory(record.get("memory"), checker)
    checker.require(
        manifest.get("schema_version") == RESULT_SCHEMA
        and manifest.get("run_tag") == record.get("run_tag")
        and manifest.get("remote_target") == REMOTE_TARGET
        and manifest.get("remote_python") == REMOTE_PYTHON,
        "manifest identity mismatch",
    )
    checker.require(
        manifest.get("result_sha256")
        == _sha256(result_path.read_bytes()),
        "manifest result SHA mismatch",
    )
    return {
        "status": "PASS",
        "checks": checker.count,
        "result_sha256": _sha256(result_path.read_bytes()),
        "manifest_sha256": _sha256(manifest_path.read_bytes()),
        "source_tree_sha256": record["source_tree_sha256"],
    }


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--prerequisite-oracle", required=True)
    arguments = parser.parse_args(argv)
    result = verify_run(
        arguments.run_dir,
        source_root=arguments.source_root,
        prerequisite_oracle=arguments.prerequisite_oracle,
    )
    print(f"PASS, {result['checks']} checks")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
