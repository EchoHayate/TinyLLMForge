from __future__ import annotations

import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"


def _load(name, filename):
    path = TOOLS / filename
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


contract = _load(
    "qwen35_tp4_hybrid_prefix_contract_for_verifier_test",
    "qwen35_tp4_hybrid_prefix_benchmark_contract.py",
)
verifier = _load(
    "verify_qwen35_tp4_hybrid_prefix_benchmark",
    "verify_qwen35_tp4_hybrid_prefix_benchmark.py",
)
builder_fixture = _load(
    "qwen35_prerequisite_builder_fixture_for_verifier",
    "test_build_qwen35_tp4_performance_prerequisites.py",
)
BENCHMARK_SOURCE_TREE_SHA256 = "c" * 64


def _write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            value,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        ) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path, rows):
    path.write_text(
        "".join(
            json.dumps(
                row,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def test_gpu_assignments_accept_shared_low_utilization_policy():
    with tempfile.TemporaryDirectory() as temporary:
        run_dir = Path(temporary)
        _write_json(
            run_dir / "gpu_assignments.json",
            {
                "schema_version": contract.SCHEMA_VERSION,
                "resource_policy": "shared-low-utilization",
                "maximum_gpu_utilization_percent": 10,
                "assignments": [
                    {
                        "rank": rank,
                        "gpu_index": rank,
                        "gpu_uuid": f"GPU-{rank}",
                        "free_bytes": contract.MIN_GPU_FREE_BYTES,
                        "utilization_percent": 10,
                        "compute_processes": [{"pid": 1000 + rank}],
                    }
                    for rank in range(contract.WORLD_SIZE)
                ],
            },
        )

        verifier._verify_gpu_assignments(run_dir)


def test_gpu_assignments_reject_shared_utilization_above_limit():
    with tempfile.TemporaryDirectory() as temporary:
        run_dir = Path(temporary)
        _write_json(
            run_dir / "gpu_assignments.json",
            {
                "schema_version": contract.SCHEMA_VERSION,
                "resource_policy": "shared-low-utilization",
                "maximum_gpu_utilization_percent": 10,
                "assignments": [
                    {
                        "rank": rank,
                        "gpu_index": rank,
                        "gpu_uuid": f"GPU-{rank}",
                        "free_bytes": contract.MIN_GPU_FREE_BYTES,
                        "utilization_percent": 11 if rank == 0 else 0,
                        "compute_processes": [],
                    }
                    for rank in range(contract.WORLD_SIZE)
                ],
            },
        )

        try:
            verifier._verify_gpu_assignments(run_dir)
        except verifier.VerificationError as error:
            assert "not eligible" in str(error)
        else:
            raise AssertionError(
                "shared GPU utilization above the limit was accepted"
            )


def _authority(run_dir, name):
    artifact = run_dir / "prerequisites" / f"{name}.json"
    independent = (
        run_dir
        / "prerequisites"
        / f"{name}.independent.json"
    )
    provenance = (
        run_dir
        / "prerequisites"
        / f"{name}.provenance.json"
    )
    authority_name = {
        "root-logit": "tp4_root_logit",
        "cached-continuation": "cached_continuation",
        "engine-correctness": "engine_correctness",
    }.get(name, name)
    source_tree_sha256 = (
        contract.TP4_ROOT_SOURCE_TREE_SHA256
        if authority_name == "tp4_root_logit"
        else "d" * 64
    )
    if authority_name == "tp4_root_logit":
        artifact_payload, verification_payload = (
            builder_fixture._root_payloads()
        )
    elif authority_name == "cached_continuation":
        artifact_payload, verification_payload = (
            builder_fixture._cached_payloads(source_tree_sha256)
        )
    else:
        artifact_payload, verification_payload = (
            builder_fixture._engine_payloads(source_tree_sha256)
        )
    _write_json(artifact, artifact_payload)
    _write_json(independent, verification_payload)
    evidence = {}
    for filename, kind in (
        (f"{name}.execution_plan.json", "plan"),
        (
            f"{name}.consumed_authorization.json",
            "authorization",
        ),
        (f"{name}.execution_receipt.json", "receipt"),
    ):
        path = run_dir / "prerequisites" / filename
        _write_json(path, {"kind": kind, "authority": authority_name})
        evidence[filename] = _sha256(path)
    _write_json(provenance, {
        "schema_version": (
            contract.PREREQUISITE_PROVENANCE_SCHEMA_VERSION
        ),
        "authority_name": authority_name,
        "run_tag": name,
        "binding_kind": "remote_execution_receipt",
        "source_tree_sha256": source_tree_sha256,
        "model_manifest_sha256": contract.MODEL_MANIFEST_SHA256,
        "root_logit_receipt_gap": False,
        "plan_path": f"{name}.execution_plan.json",
        "plan_sha256": evidence[f"{name}.execution_plan.json"],
        "authorization_path": f"{name}.consumed_authorization.json",
        "authorization_sha256": evidence[
            f"{name}.consumed_authorization.json"
        ],
        "receipt_path": f"{name}.execution_receipt.json",
        "receipt_sha256": evidence[f"{name}.execution_receipt.json"],
    })
    return {
        "run_tag": name,
        "source_tree_sha256": source_tree_sha256,
        "artifact_path": artifact.relative_to(run_dir).as_posix(),
        "artifact_sha256": _sha256(artifact),
        "independent_verification_path": (
            independent.relative_to(run_dir).as_posix()
        ),
        "independent_verification_sha256": _sha256(independent),
        "provenance_path": provenance.relative_to(run_dir).as_posix(),
        "provenance_sha256": _sha256(provenance),
        "classification": "PASS",
    }


def _policy_values(policy, workload):
    baseline = policy == "recompute"
    ttft = 1000
    e2e = 2000
    if not baseline:
        if workload == "w1_medium_reuse":
            ttft = 800
        elif workload == "w2_long_reuse":
            ttft = 700
        elif workload == "w3_batched_fanout":
            ttft = 850
            e2e = 1600
    return ttft, e2e


def _cache_values(policy):
    if policy == "recompute":
        return {
            "hybrid_cache_current_entries": 0,
            "hybrid_cache_current_bytes": 0,
            "hybrid_cache_current_logical_bytes": 0,
            "hybrid_cache_deduplicated_bytes": 0,
            "hybrid_cache_peak_entries": 0,
            "hybrid_cache_peak_bytes": 0,
            "hybrid_cache_hits": 0,
            "hybrid_cache_misses": 0,
            "hybrid_cache_evictions": 0,
            "hybrid_cache_validation_failures": 0,
            "hybrid_cache_failed_restores": 0,
        }
    return {
        "hybrid_cache_current_entries": 2,
        "hybrid_cache_current_bytes": 4096,
        "hybrid_cache_current_logical_bytes": 6144,
        "hybrid_cache_deduplicated_bytes": 2048,
        "hybrid_cache_peak_entries": 2,
        "hybrid_cache_peak_bytes": 4096,
        "hybrid_cache_hits": 8,
        "hybrid_cache_misses": 1,
        "hybrid_cache_evictions": 0,
        "hybrid_cache_validation_failures": 0,
        "hybrid_cache_failed_restores": 0,
    }


def _complete_run_dir(root):
    run_dir = Path(root)
    for name in contract.NESTED_ARTIFACT_DIRECTORIES:
        (run_dir / name).mkdir(parents=True)

    prerequisites = {
        "schema_version": contract.PREREQUISITE_SCHEMA_VERSION,
        "model_manifest_sha256": contract.MODEL_MANIFEST_SHA256,
        "tp4_root_logit": _authority(run_dir, "root-logit"),
        "cached_continuation": _authority(
            run_dir,
            "cached-continuation",
        ),
        "engine_correctness": _authority(
            run_dir,
            "engine-correctness",
        ),
    }
    _write_json(
        run_dir / "correctness_prerequisites.json",
        prerequisites,
    )
    prerequisite_sha = _sha256(
        run_dir / "correctness_prerequisites.json"
    )

    workload_manifest = contract.workload_manifest_payload()
    _write_json(run_dir / "workload_manifest.json", workload_manifest)
    workload_sha = _sha256(run_dir / "workload_manifest.json")
    _write_json(
        run_dir / "source_manifest.json",
        {
            "schema_version": contract.SCHEMA_VERSION,
            "source_tree_sha256": BENCHMARK_SOURCE_TREE_SHA256,
            "model_manifest_sha256": (
                contract.MODEL_MANIFEST_SHA256
            ),
        },
    )
    _write_json(
        run_dir / "environment.json",
        {
            "schema_version": contract.SCHEMA_VERSION,
            "world_size": 4,
            "python": "/remote/env/bin/python",
        },
    )
    _write_json(
        run_dir / "gpu_assignments.json",
        {
            "schema_version": contract.SCHEMA_VERSION,
            "assignments": [
                {
                    "rank": rank,
                    "gpu_index": rank,
                    "gpu_uuid": (
                        "GPU-00000000-0000-0000-0000-"
                        f"00000000000{rank}"
                    ),
                    "free_bytes": contract.MIN_GPU_FREE_BYTES,
                    "compute_processes": [],
                }
                for rank in range(4)
            ],
        },
    )

    case_rows = []
    process_rows = []
    logits = []
    commands = []
    for case_index, case in enumerate(contract.build_case_matrix()):
        spec = contract.WORKLOAD_SPECS[case.workload]
        command = {
            "case_id": case.case_id,
            "policy": case.policy,
            "workload": case.workload,
            "phase": case.phase,
            "repetition": case.repetition,
            "dist_port": 22000 + case_index * 2,
            "master_port": 22001 + case_index * 2,
        }
        commands.append(command)
        ttft_ns, e2e_ns = _policy_values(
            case.policy,
            case.workload,
        )
        restored = (
            case.policy == "exact_restore"
            and case.workload
            in {
                "w1_medium_reuse",
                "w2_long_reuse",
                "w3_batched_fanout",
            }
        )
        for request_index in range(spec["continuations"]):
            output_ids = [
                (
                    len(case.workload)
                    + request_index
                    + token_index
                )
                % 32000
                for token_index in range(spec["generated_tokens"])
            ]
            logits_path = None
            logits_sha = None
            if case.phase == "correctness":
                relative = (
                    Path("logits")
                    / f"{case.case_id}__{request_index}.json"
                )
                _write_json(run_dir / relative, [0.25, 0.75])
                logits_path = relative.as_posix()
                logits_sha = _sha256(run_dir / relative)
                logits.append({
                    "path": logits_path,
                    "sha256": logits_sha,
                })
            row = {
                "row_id": (
                    f"{case.case_id}__request-{request_index}"
                ),
                "case_id": case.case_id,
                "policy": case.policy,
                "workload": case.workload,
                "phase": case.phase,
                "repetition": case.repetition,
                "request_id": f"request-{request_index}",
                "source_tree_sha256": (
                    BENCHMARK_SOURCE_TREE_SHA256
                ),
                "model_manifest_sha256": (
                    contract.MODEL_MANIFEST_SHA256
                ),
                "workload_manifest_sha256": workload_sha,
                "correctness_prerequisites_sha256": prerequisite_sha,
                "prompt_tokens": (
                    spec["shared_prefix_tokens"]
                    + spec["suffix_tokens"]
                ),
                "reused_kv_tokens": (
                    spec["shared_prefix_tokens"] if restored else 0
                ),
                "restored_hybrid_state": restored,
                "executed_prefill_tokens": (
                    spec["suffix_tokens"]
                    if restored
                    else (
                        spec["shared_prefix_tokens"]
                        + spec["suffix_tokens"]
                    )
                ),
                "generated_tokens": spec["generated_tokens"],
                "ttft_ns": ttft_ns,
                "e2e_ns": e2e_ns,
                "decode_step_ns": [
                    100
                    for _ in range(spec["generated_tokens"] - 1)
                ],
                "output_token_ids": output_ids,
                "output_token_ids_sha256": (
                    contract.canonical_json_sha256(output_ids)
                ),
                "final_logits_path": logits_path,
                "final_logits_sha256": logits_sha,
            }
            case_rows.append(row)
        process_rows.append({
            "case_id": case.case_id,
            "policy": case.policy,
            "workload": case.workload,
            "phase": case.phase,
            "repetition": case.repetition,
            "initialization_ns": 1_000_000,
            "cuda_allocated_bytes": 1_000_000,
            "cuda_reserved_bytes": 2_000_000,
            "cuda_peak_allocated_bytes": 3_000_000,
            "cuda_peak_reserved_bytes": (
                10_000_000
                if case.policy == "recompute"
                else 10_500_000
            ),
            "kv_capacity_bytes": 5_000_000,
            "scheduler_visible_kv_blocks": 64,
            **_cache_values(case.policy),
        })

    _write_json(
        run_dir / "commands.json",
        {
            "schema_version": contract.SCHEMA_VERSION,
            "commands": commands,
        },
    )
    _write_jsonl(run_dir / "case_rows.jsonl", case_rows)
    _write_jsonl(run_dir / "process_rows.jsonl", process_rows)
    _write_json(
        run_dir / "logits_manifest.json",
        {
            "schema_version": contract.SCHEMA_VERSION,
            "files": logits,
        },
    )
    worker_log = run_dir / "logs" / "workers.log"
    worker_log.write_text(
        "QWEN35_TP4_BENCHMARK_WORKER_COMPLETE\n",
        encoding="utf-8",
    )
    _write_json(
        run_dir / "worker_logs_manifest.json",
        {
            "schema_version": contract.SCHEMA_VERSION,
            "files": [{
                "path": "logs/workers.log",
                "sha256": _sha256(worker_log),
            }],
        },
    )
    _write_json(
        run_dir / "summary.json",
        {
            "schema_version": contract.SCHEMA_VERSION,
            "classification": "UNTRUSTED_PRODUCER_GO",
            "case_rows": len(case_rows),
            "process_rows": len(process_rows),
        },
    )

    files = {}
    for name in contract.ARTIFACT_MANIFEST_HASH_DOMAIN:
        path = run_dir / name
        files[name] = {
            "sha256": _sha256(path),
            "size": path.stat().st_size,
        }
    for directory in contract.NESTED_ARTIFACT_DIRECTORIES:
        for path in sorted((run_dir / directory).rglob("*")):
            if path.is_file():
                relative = path.relative_to(run_dir).as_posix()
                files[relative] = {
                    "sha256": _sha256(path),
                    "size": path.stat().st_size,
                }
    _write_json(
        run_dir / "artifact_manifest.json",
        {
            "schema_version": contract.SCHEMA_VERSION,
            "files": files,
        },
    )
    return run_dir


def _read_jsonl(path):
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _rewrite_artifact_manifest(run_dir):
    payload = json.loads(
        (run_dir / "artifact_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    for relative in list(payload["files"]):
        path = run_dir / relative
        if not path.is_file():
            continue
        payload["files"][relative] = {
            "sha256": _sha256(path),
            "size": path.stat().st_size,
        }
    _write_json(run_dir / "artifact_manifest.json", payload)


def _expect_invalid(mutator, fragment, *, resign=False):
    with tempfile.TemporaryDirectory() as temporary:
        run_dir = _complete_run_dir(temporary)
        mutator(run_dir)
        if resign:
            _rewrite_artifact_manifest(run_dir)
        try:
            verifier.verify_run(run_dir)
        except verifier.VerificationError as error:
            assert fragment in str(error), str(error)
        else:
            raise AssertionError(
                f"tampered fixture accepted: {fragment}"
            )


def test_complete_fixture_reconstructs_go_without_trusting_summary():
    with tempfile.TemporaryDirectory() as temporary:
        run_dir = _complete_run_dir(temporary)

        result = verifier.verify_run(run_dir)

        assert result["classification"] == "GO"
        assert result["case_rows"] == 280
        assert result["process_rows"] == 70
        assert result["workloads"]["w1_medium_reuse"][
            "median_ttft_ratio"
        ] == 0.8
        assert result["workloads"]["w2_long_reuse"][
            "median_ttft_ratio"
        ] == 0.7
        assert result["workloads"]["w3_batched_fanout"][
            "throughput_ratio"
        ] == 1.25
        cache_efficiency = result["cache_efficiency"]
        assert cache_efficiency[
            "logical_to_physical_snapshot_ratio"
        ] == 1.5
        reuse_workloads = (
            "w1_medium_reuse",
            "w2_long_reuse",
            "w3_batched_fanout",
        )
        reused_tokens = sum(
            contract.WORKLOAD_SPECS[workload][
                "shared_prefix_tokens"
            ]
            * contract.WORKLOAD_SPECS[workload]["continuations"]
            * contract.MEASURED_REPETITIONS
            for workload in reuse_workloads
        )
        physical_snapshot_bytes = (
            4096
            * len(reuse_workloads)
            * contract.MEASURED_REPETITIONS
        )
        added_cuda_bytes = (
            500_000
            * len(reuse_workloads)
            * contract.MEASURED_REPETITIONS
        )
        assert math.isclose(
            cache_efficiency[
                "physical_snapshot_bytes_per_reused_token"
            ],
            physical_snapshot_bytes / reused_tokens,
        )
        assert math.isclose(
            cache_efficiency[
                "added_cuda_bytes_per_reused_token"
            ],
            added_cuda_bytes / reused_tokens,
        )
        assert math.isclose(
            cache_efficiency[
                "saved_prefill_tokens_per_physical_snapshot_byte"
            ],
            reused_tokens / physical_snapshot_bytes,
        )
        assert (
            run_dir / "independent_verification.json"
        ).is_file()
        assert (run_dir / "report.md").is_file()


def test_artifact_hash_tamper_is_rejected():
    def mutate(run_dir):
        path = run_dir / "case_rows.jsonl"
        data = path.read_bytes()
        replacement = b"X" if data[:1] != b"X" else b"Y"
        path.write_bytes(replacement + data[1:])

    _expect_invalid(mutate, "artifact hash mismatch")


def test_extra_file_is_rejected_even_if_not_manifested():
    def mutate(run_dir):
        (run_dir / "extra.txt").write_text(
            "unexpected\n",
            encoding="utf-8",
        )

    _expect_invalid(mutate, "unexpected artifact")


def test_resigned_extra_nested_file_is_rejected():
    def mutate(run_dir):
        (run_dir / "logs" / "extra.log").write_text(
            "QWEN35_TP4_BENCHMARK_WORKER_COMPLETE\n",
            encoding="utf-8",
        )
        manifest = json.loads(
            (run_dir / "artifact_manifest.json").read_text(
                encoding="utf-8"
            )
        )
        path = run_dir / "logs" / "extra.log"
        manifest["files"]["logs/extra.log"] = {
            "sha256": _sha256(path),
            "size": path.stat().st_size,
        }
        _write_json(run_dir / "artifact_manifest.json", manifest)

    _expect_invalid(
        mutate,
        "nested artifact inventory mismatch",
    )


def test_resigned_output_mismatch_is_rejected():
    def mutate(run_dir):
        rows = _read_jsonl(run_dir / "case_rows.jsonl")
        row = next(
            value
            for value in rows
            if value["policy"] == "exact_restore"
            and value["phase"] == "correctness"
        )
        row["output_token_ids"] = [999]
        row["output_token_ids_sha256"] = (
            contract.canonical_json_sha256([999])
        )
        _write_jsonl(run_dir / "case_rows.jsonl", rows)

    _expect_invalid(
        mutate,
        "output token mismatch",
        resign=True,
    )


def test_resigned_workload_token_manifest_tamper_is_rejected():
    def mutate(run_dir):
        path = run_dir / "workload_manifest.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        token_ids = payload["workloads"]["w2_long_reuse"][
            "shared_prefix_token_ids"
        ]
        token_ids[17] = (token_ids[17] + 1) % (
            contract.TOKEN_ID_UPPER_BOUND
        )
        _write_json(path, payload)

    _expect_invalid(
        mutate,
        "workload manifest mismatch",
        resign=True,
    )


def test_resigned_request_accounting_tamper_is_rejected():
    def mutate(run_dir):
        rows = _read_jsonl(run_dir / "case_rows.jsonl")
        for row in rows:
            if (
                row["workload"] == "w2_long_reuse"
                and row["phase"] == "warmup"
                and row["repetition"] == 0
                and row["request_id"] == "request-0"
            ):
                row["prompt_tokens"] += 1
        _write_jsonl(run_dir / "case_rows.jsonl", rows)

    _expect_invalid(
        mutate,
        "prompt token accounting mismatch",
        resign=True,
    )


def test_resigned_prefill_and_generation_shape_tamper_is_rejected():
    def mutate_prefill(run_dir):
        rows = _read_jsonl(run_dir / "case_rows.jsonl")
        for row in rows:
            if (
                row["workload"] == "w1_medium_reuse"
                and row["policy"] == "exact_restore"
                and row["phase"] == "warmup"
            ):
                row["executed_prefill_tokens"] += 1
        _write_jsonl(run_dir / "case_rows.jsonl", rows)

    _expect_invalid(
        mutate_prefill,
        "executed prefill accounting mismatch",
        resign=True,
    )

    def mutate_generation(run_dir):
        rows = _read_jsonl(run_dir / "case_rows.jsonl")
        for row in rows:
            if (
                row["workload"] == "w0_short_control"
                and row["phase"] == "warmup"
            ):
                row["generated_tokens"] -= 1
                row["output_token_ids"].pop()
                row["decode_step_ns"].pop()
                row["output_token_ids_sha256"] = (
                    contract.canonical_json_sha256(
                        row["output_token_ids"]
                    )
                )
        _write_jsonl(run_dir / "case_rows.jsonl", rows)

    _expect_invalid(
        mutate_generation,
        "generated token accounting mismatch",
        resign=True,
    )


def test_resigned_missing_measured_process_row_is_rejected():
    def mutate(run_dir):
        rows = _read_jsonl(run_dir / "process_rows.jsonl")
        rows.pop()
        _write_jsonl(run_dir / "process_rows.jsonl", rows)

    _expect_invalid(
        mutate,
        "process matrix mismatch",
        resign=True,
    )


def test_resigned_cache_accounting_tamper_is_rejected():
    def mutate(run_dir):
        rows = _read_jsonl(run_dir / "process_rows.jsonl")
        row = next(
            value
            for value in rows
            if value["policy"] == "exact_restore"
        )
        row["hybrid_cache_deduplicated_bytes"] = 1
        _write_jsonl(run_dir / "process_rows.jsonl", rows)

    _expect_invalid(
        mutate,
        "cache accounting mismatch",
        resign=True,
    )


def test_resigned_worker_traceback_is_rejected():
    def mutate(run_dir):
        log = run_dir / "logs" / "workers.log"
        log.write_text(
            "Traceback (most recent call last):\nboom\n",
            encoding="utf-8",
        )
        manifest = json.loads(
            (run_dir / "worker_logs_manifest.json").read_text(
                encoding="utf-8"
            )
        )
        manifest["files"][0]["sha256"] = _sha256(log)
        _write_json(run_dir / "worker_logs_manifest.json", manifest)

    _expect_invalid(
        mutate,
        "worker log contains traceback",
        resign=True,
    )


def _run():
    tests = [
        value
        for name, value in sorted(globals().items())
        if name.startswith("test_") and callable(value)
    ]
    for test in tests:
        test()
    print(
        "qwen35 TP4 hybrid-prefix benchmark verifier tests passed "
        f"({len(tests)} tests)"
    )


if __name__ == "__main__":
    _run()
