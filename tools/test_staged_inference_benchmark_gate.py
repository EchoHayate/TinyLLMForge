"""Dependency-light tests for the staged benchmark primary orchestrator.

Run:
    python3 tools/test_staged_inference_benchmark_gate.py
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import types


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools import staged_inference_benchmark_contract as contract
from tools import staged_inference_benchmark_gate as gate


def _source_evidence() -> dict:
    return {
        "schema_version": 1,
        "base_commit": "1" * 40,
        "local_head": "1" * 40,
        "tracking_head": "1" * 40,
        "dirty": False,
        "tree_sha256": "2" * 64,
        "owned_roots": list(gate.OWNED_SOURCE_ROOTS),
    }


def _environment_evidence(
    model_tier: str = "qwen3-0.6b",
    *,
    gate_name: str = "prefix",
) -> dict:
    if gate_name == "prefix":
        engine_limits = {
            "max_model_len": gate.PREFIX_PROFILE_POLICY["max_model_len"],
            "max_num_batched_tokens": gate.PREFIX_PROFILE_POLICY[
                "max_num_batched_tokens"
            ],
            "max_num_seqs": gate.PREFIX_PROFILE_POLICY["max_num_seqs"],
        }
    elif gate_name == "chunked":
        engine_limits = dict(contract.CHUNKED_ENGINE_CONFIG)
    else:
        raise ValueError(f"unsupported test gate: {gate_name}")
    return {
        "model_tier": model_tier,
        "python_version": "3.11.13",
        "torch_version": "2.8.0",
        "cuda_version": "12.8",
        "gpu_inventory": [
            {
                "index": index,
                "uuid": f"GPU-{index}",
                "name": "NVIDIA H100 80GB HBM3",
            }
            for index in range(4)
        ],
        "selected_gpu_indices": [0, 1, 2, 3],
        "model_config_sha256": "3" * 64,
        "checkpoint_identifier": model_tier,
        "model_path": f"/models/{model_tier}",
        "engine_limits": engine_limits,
    }


def _prefix_shape(prefix_tokens: int, batch_size: int) -> dict:
    expected_cached = prefix_tokens * batch_size
    cold_query = (prefix_tokens + 64) * batch_size
    warm_query = 64 * batch_size

    def state(
        elapsed_ms: float,
        cached_tokens: int,
        query_tokens: int,
        reserved_bytes: int,
        model_batches: int,
    ) -> dict:
        return {
            "median_elapsed_ms": elapsed_ms,
            "p95_elapsed_ms": elapsed_ms,
            "median_cached_prompt_tokens": cached_tokens,
            "median_executed_query_tokens": query_tokens,
            "median_model_batches": model_batches,
            "peak_cuda_reserved_bytes": reserved_bytes,
            "exact_outputs": True,
            "logit_argmax_match": True,
            "logit_max_abs": 0.0,
            "logit_mean_abs": 0.0,
            "samples": 7,
        }

    return {
        "prefix_tokens": prefix_tokens,
        "suffix_tokens": 64,
        "batch_size": batch_size,
        "expected_reusable_tokens": expected_cached,
        "cold": state(
            100.0,
            0,
            cold_query,
            1_000,
            1 if batch_size == 1 else 2,
        ),
        "warm": state(75.0, expected_cached, warm_query, 1_040, 1),
        "cache_cleared": state(
            101.0,
            0,
            cold_query,
            1_000,
            1 if batch_size == 1 else 2,
        ),
        "retained_reusable_blocks": expected_cached // 256,
        "retained_logical_kv_bytes": expected_cached * 32,
        "median_cache_clear_host_ms": 0.2,
    }


def _complete_prefix_bundle() -> dict:
    return {
        "artifact_complete": True,
        "single": {
            str(tokens): _prefix_shape(tokens, 1)
            for tokens in (256, 1024, 2048)
        },
        "batch": {
            str(tokens): _prefix_shape(tokens, 8)
            for tokens in (1024, 2048)
        },
    }


def _prefix_raw_artifacts() -> dict[str, list[dict]]:
    performance = []
    for family, prefix_tokens, batch_size in (
        ("single", 256, 1),
        ("single", 1024, 1),
        ("single", 2048, 1),
        ("batch8", 1024, 8),
        ("batch8", 2048, 8),
    ):
        shape = f"{family}-{prefix_tokens}"
        contract_family = "single" if family == "single" else "batch"
        shape_summary = _complete_prefix_bundle()[contract_family][
            str(prefix_tokens)
        ]
        for state in ("cold", "warm", "cache_cleared"):
            state_summary = shape_summary[state]
            for repetition in range(7):
                performance.append({
                    "schema_version": 2,
                    "case_id": f"{shape}__{state}__r{repetition}",
                    "shape": shape,
                    "state": state,
                    "repetition": repetition,
                    "warmup": False,
                    "shared_prefix_tokens": prefix_tokens,
                    "suffix_tokens": 64,
                    "batch_size": batch_size,
                    "ttft_ns": int(
                        state_summary["median_elapsed_ms"] * 1_000_000
                    ),
                    "cached_prompt_tokens": state_summary[
                        "median_cached_prompt_tokens"
                    ],
                    "executed_query_tokens": state_summary[
                        "median_executed_query_tokens"
                    ],
                    "model_batches": state_summary[
                        "median_model_batches"
                    ],
                    "correct": True,
                    "logit": {
                        "argmax_match": True,
                        "max_abs": 0.0,
                        "mean_abs": 0.0,
                    },
                    "retained_reusable_blocks": shape_summary[
                        "retained_reusable_blocks"
                    ],
                    "retained_logical_kv_bytes": shape_summary[
                        "retained_logical_kv_bytes"
                    ],
                    "cache_clear_host_ns": int(
                        shape_summary["median_cache_clear_host_ms"]
                        * 1_000_000
                    ),
                    "cuda_peak_reserved_bytes": state_summary[
                        "peak_cuda_reserved_bytes"
                    ],
                })
    cache_rows = [
        {
            key: row[key]
            for key in (
                "schema_version",
                "case_id",
                "shape",
                "state",
                "repetition",
                "warmup",
                "cached_prompt_tokens",
                "executed_query_tokens",
                "retained_reusable_blocks",
                "retained_logical_kv_bytes",
                "cache_clear_host_ns",
            )
        }
        for row in performance
    ]
    memory_rows = [
        {
            key: row[key]
            for key in (
                "schema_version",
                "case_id",
                "shape",
                "state",
                "repetition",
                "warmup",
                "cuda_peak_reserved_bytes",
                "retained_logical_kv_bytes",
            )
        }
        for row in performance
    ]
    correctness_cases = (
        "cpu_collision_and_lifecycle_preflight",
        "repeat_255",
        "repeat_256",
        "repeat_257",
        "repeat_512",
        "repeat_513",
        "same_batch_p_q_p_first",
        "same_batch_p_q_p_middle",
        "same_batch_p_q_p",
        "shared_prefix_different_suffix",
        "cache_cleared",
    )
    return {
        "prefix_correctness_rows.jsonl": [
            {
                "case": case,
                "state": (
                    "preflight"
                    if case == "cpu_collision_and_lifecycle_preflight"
                    else "correctness"
                ),
                "correct": True,
            }
            for case in correctness_cases
        ],
        "prefix_performance_rows.jsonl": performance,
        "prefix_cache_rows.jsonl": cache_rows,
        "prefix_memory_rows.jsonl": memory_rows,
    }


def _populate_prefix_output(
    run_dir: Path,
    manifest: dict,
    *,
    empty_raw_filename: str | None = None,
    drop_correctness_case: str | None = None,
    raw_warm_ttft_matches_cold: bool = False,
) -> None:
    case_id = manifest["case_order"][0]
    case_dir = run_dir / "cases" / case_id
    output_dir = case_dir / "output"
    output_dir.mkdir(parents=True)
    prefix_bundle = _complete_prefix_bundle()
    _write_json(
        output_dir / "summary.json",
        {
            "staged_contract_bundle": prefix_bundle,
            "staged_decision": contract.classify_prefix_bundle(
                prefix_bundle
            ),
        },
    )
    for filename, rows in _prefix_raw_artifacts().items():
        if (
            filename == "prefix_performance_rows.jsonl"
            and raw_warm_ttft_matches_cold
        ):
            cold_ttft_by_shape = {
                row["shape"]: row["ttft_ns"]
                for row in rows
                if row["state"] == "cold"
            }
            rows = [
                {
                    **row,
                    "ttft_ns": (
                        cold_ttft_by_shape[row["shape"]]
                        if row["state"] == "warm"
                        else row["ttft_ns"]
                    ),
                }
                for row in rows
            ]
        selected_rows = [
            row
            for row in rows
            if not (
                filename == "prefix_correctness_rows.jsonl"
                and row.get("case") == drop_correctness_case
            )
        ]
        _write_jsonl(
            output_dir / filename,
            [] if filename == empty_raw_filename else selected_rows,
        )
    _write_json(
        case_dir / "process.json",
        {"case_id": case_id, "returncode": 0},
    )
    (case_dir / "stdout.log").write_text("", encoding="utf-8")
    (case_dir / "stderr.log").write_text("", encoding="utf-8")
    (case_dir / "exitcode").write_text("0\n", encoding="utf-8")


def _chunked_summary() -> dict:
    repetitions = []
    for repetition in range(5):
        base = {
            "short_p99_ttft_ns": 100.0,
            "short_p99_itl_ns": 100.0,
            "maximum_decode_gap_ns": 100.0,
            "service_class_p95_completion_ns": {
                bucket: 100.0
                for bucket in gate.SERVICE_CLASS_BUCKETS
            },
            "long_p95_completion_ns": 100.0,
            "request_throughput_rps": 100.0,
            "output_token_throughput_tps": 100.0,
            "peak_cuda_reserved_bytes": 1_000.0,
            "exact_outputs": True,
            "complete_lifecycle": True,
            "dropped_requests": 0,
            "rejected_requests": 0,
            "truncated_requests": 0,
            "unfinished_requests": 0,
            "starved_requests": 0,
        }
        candidate = json.loads(json.dumps(base))
        candidate.update({
            "short_p99_ttft_ns": 85.0,
            "short_p99_itl_ns": 103.0,
            "maximum_decode_gap_ns": 108.0,
            "long_p95_completion_ns": 109.0,
            "request_throughput_rps": 98.0,
            "output_token_throughput_tps": 98.0,
            "peak_cuda_reserved_bytes": 1_040.0,
        })
        candidate["service_class_p95_completion_ns"] = {
            bucket: 105.0
            for bucket in gate.SERVICE_CLASS_BUCKETS
        }
        repetitions.append({
            "repetition": repetition,
            "OFF": base,
            "FAIR_CHUNKED": candidate,
        })
    return contract.classify_chunked_bundle({
        "artifact_complete": True,
        "repetitions": repetitions,
    })


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(row, sort_keys=True) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def test_initialize_binds_source_environment_workload_and_policy():
    with TemporaryDirectory() as temporary:
        run_dir = Path(temporary) / "run"
        manifest = gate.initialize_run(
            run_dir=run_dir,
            run_tag="qwen3-06b-prefix-r1",
            gate_name="prefix",
            model_tier="qwen3-0.6b",
            source_evidence=_source_evidence(),
            environment_evidence=_environment_evidence(),
        )

        assert manifest == _read_json(run_dir / "run_manifest.json")
        assert manifest["source_tree_sha256"] == "2" * 64
        assert len(manifest["environment_sha256"]) == 64
        assert len(manifest["workload_sha256"]) == 64
        assert len(manifest["policy_sha256"]) == 64
        assert manifest["case_order"] == ["prefix_full__qwen3-0.6b"]
        assert (run_dir / "resolved_config.json").is_file()
        workload_rows = [
            json.loads(line)
            for line in (
                run_dir / "workload_manifest.jsonl"
            ).read_text(encoding="utf-8").splitlines()
        ]
        assert len(workload_rows) == 15


def test_existing_run_tag_is_never_overwritten():
    with TemporaryDirectory() as temporary:
        run_dir = Path(temporary) / "run"
        run_dir.mkdir()
        try:
            gate.initialize_run(
                run_dir=run_dir,
                run_tag="qwen3-06b-prefix-r1",
                gate_name="prefix",
                model_tier="qwen3-0.6b",
                source_evidence=_source_evidence(),
                environment_evidence=_environment_evidence(),
            )
        except ValueError as error:
            assert "already exists" in str(error)
        else:
            raise AssertionError("existing run directory must fail closed")


def test_initialize_rejects_unbound_source_or_model_tier():
    with TemporaryDirectory() as temporary:
        source = _source_evidence()
        source["tracking_head"] = "9" * 40
        try:
            gate.initialize_run(
                run_dir=Path(temporary) / "source-mismatch",
                run_tag="qwen3-06b-prefix-source-mismatch",
                gate_name="prefix",
                model_tier="qwen3-0.6b",
                source_evidence=source,
                environment_evidence=_environment_evidence(),
            )
        except ValueError as error:
            assert "tracking" in str(error)
        else:
            raise AssertionError("unbound source evidence must fail")

    with TemporaryDirectory() as temporary:
        environment = _environment_evidence()
        environment["model_tier"] = "qwen3-8b"
        try:
            gate.initialize_run(
                run_dir=Path(temporary) / "tier-mismatch",
                run_tag="qwen3-06b-prefix-tier-mismatch",
                gate_name="prefix",
                model_tier="qwen3-0.6b",
                source_evidence=_source_evidence(),
                environment_evidence=environment,
            )
        except ValueError as error:
            assert "model tier" in str(error)
        else:
            raise AssertionError("environment tier mismatch must fail")

    with TemporaryDirectory() as temporary:
        environment = _environment_evidence()
        environment["engine_limits"]["max_num_seqs"] = 512
        try:
            gate.initialize_run(
                run_dir=Path(temporary) / "limit-mismatch",
                run_tag="qwen3-06b-prefix-limit-mismatch",
                gate_name="prefix",
                model_tier="qwen3-0.6b",
                source_evidence=_source_evidence(),
                environment_evidence=environment,
            )
        except ValueError as error:
            assert "engine limits" in str(error)
        else:
            raise AssertionError("environment engine limit mismatch must fail")


def test_launch_case_uses_frozen_worker_cli_and_atomic_receipts():
    commands = []

    def process_runner(command, **kwargs):
        commands.append((list(command), kwargs))
        return types.SimpleNamespace(
            returncode=0,
            stdout='{"status":"PASS"}\n',
            stderr="",
        )

    with TemporaryDirectory() as temporary:
        run_dir = Path(temporary) / "run"
        manifest = gate.initialize_run(
            run_dir=run_dir,
            run_tag="qwen3-06b-chunked-r1",
            gate_name="chunked",
            model_tier="qwen3-0.6b",
            source_evidence=_source_evidence(),
            environment_evidence=_environment_evidence(
                gate_name="chunked"
            ),
        )
        case_id = manifest["case_order"][0]
        next_case_id = manifest["case_order"][1]
        ports = iter((
            31001,
            31002,
            31001,
            31002,
            31003,
            31004,
        ))
        receipt = gate.launch_case(
            run_dir,
            case_id,
            python_bin="/usr/bin/python3",
            process_runner=process_runner,
            port_allocator=ports.__next__,
        )
        case_dir = run_dir / "cases" / case_id

        command = commands[0][0]
        assert command[:2] == [
            "/usr/bin/python3",
            str(
                gate.REPO_ROOT
                / "tools"
                / "staged_inference_benchmark_worker.py"
            ),
        ]
        assert "--spec" in command
        assert "--workload-manifest" in command
        assert "--output-dir" in command
        assert "--case-spec" not in command
        assert "--workload" not in command
        assert receipt == _read_json(case_dir / "process.json")
        assert (case_dir / "stdout.log").read_text() == (
            '{"status":"PASS"}\n'
        )
        assert (case_dir / "stderr.log").read_text() == ""
        assert (case_dir / "exitcode").read_text() == "0\n"
        assert commands[0][1]["env"]["MASTER_PORT"] == "31001"
        assert commands[0][1]["env"]["TINYLLMFORGE_DIST_PORT"] == "31002"

        next_receipt = gate.launch_case(
            run_dir,
            next_case_id,
            python_bin="/usr/bin/python3",
            process_runner=process_runner,
            port_allocator=ports.__next__,
        )
        assert next_receipt["master_port"] == 31003
        assert next_receipt["distributed_port"] == 31004

        try:
            gate.launch_case(
                run_dir,
                case_id,
                python_bin="/usr/bin/python3",
                process_runner=process_runner,
                port_allocator=iter((32001, 32002)).__next__,
            )
        except ValueError as error:
            assert "already exists" in str(error)
        else:
            raise AssertionError("case directory reuse must fail closed")


def test_launch_case_rejects_case_spec_tamper():
    with TemporaryDirectory() as temporary:
        run_dir = Path(temporary) / "run"
        manifest = gate.initialize_run(
            run_dir=run_dir,
            run_tag="qwen3-06b-prefix-tampered-case",
            gate_name="prefix",
            model_tier="qwen3-0.6b",
            source_evidence=_source_evidence(),
            environment_evidence=_environment_evidence(),
        )
        case_id = manifest["case_order"][0]
        case_path = run_dir / manifest["case_specs"][case_id]
        case = _read_json(case_path)
        case["profile_args"]["model"] = "/models/not-the-bound-checkpoint"
        _write_json(case_path, case)

        try:
            gate.launch_case(run_dir, case_id)
        except ValueError as error:
            assert "case specification identity mismatch" in str(error)
        else:
            raise AssertionError("tampered case spec must fail before launch")


def test_finalize_prefix_writes_complete_hashed_primary_bundle():
    with TemporaryDirectory() as temporary:
        run_dir = Path(temporary) / "run"
        manifest = gate.initialize_run(
            run_dir=run_dir,
            run_tag="qwen3-06b-prefix-r1",
            gate_name="prefix",
            model_tier="qwen3-0.6b",
            source_evidence=_source_evidence(),
            environment_evidence=_environment_evidence(),
        )
        case_id = manifest["case_order"][0]
        _populate_prefix_output(run_dir, manifest)

        summary = gate.finalize_run(run_dir)

        assert summary["classification"] == "PREFIX_CACHE_GO"
        report = (run_dir / "report.md").read_text(encoding="utf-8")
        assert "| Benefit | Cost |" in report
        assert "| --- | --- |" in report
        hashes = _read_json(run_dir / "artifact_hashes.json")
        assert any(
            filename.startswith(f"cases/{case_id}/")
            for filename in hashes
        )
        assert f"case_specs/{case_id}.json" in hashes
        for filename, expected in hashes.items():
            assert hashlib.sha256(
                (run_dir / filename).read_bytes()
            ).hexdigest() == expected
        assert (run_dir / "manifest.sha256").read_text().strip() == (
            hashlib.sha256(
                (run_dir / "run_manifest.json").read_bytes()
            ).hexdigest()
        )
        receipt = _read_json(
            run_dir / "primary_verification_receipt.json"
        )
        assert receipt["classification"] == "PREFIX_CACHE_GO"
        assert receipt["case_count"] == 1
        before = hashlib.sha256(
            (run_dir / "artifact_hashes.json").read_bytes()
        ).hexdigest()
        try:
            gate.finalize_run(run_dir)
        except ValueError as error:
            assert "already finalized" in str(error)
        else:
            raise AssertionError("finalized run must be immutable")
        assert hashlib.sha256(
            (run_dir / "artifact_hashes.json").read_bytes()
        ).hexdigest() == before


def test_finalize_prefix_rejects_empty_raw_artifact():
    with TemporaryDirectory() as temporary:
        run_dir = Path(temporary) / "run"
        manifest = gate.initialize_run(
            run_dir=run_dir,
            run_tag="qwen3-06b-prefix-empty-raw",
            gate_name="prefix",
            model_tier="qwen3-0.6b",
            source_evidence=_source_evidence(),
            environment_evidence=_environment_evidence(),
        )
        _populate_prefix_output(
            run_dir,
            manifest,
            empty_raw_filename="prefix_memory_rows.jsonl",
        )

        try:
            gate.finalize_run(run_dir)
        except ValueError as error:
            assert "prefix raw artifacts" in str(error)
        else:
            raise AssertionError("empty Prefix raw evidence must fail")


def test_finalize_prefix_rejects_incomplete_correctness_matrix():
    with TemporaryDirectory() as temporary:
        run_dir = Path(temporary) / "run"
        manifest = gate.initialize_run(
            run_dir=run_dir,
            run_tag="qwen3-06b-prefix-missing-correctness",
            gate_name="prefix",
            model_tier="qwen3-0.6b",
            source_evidence=_source_evidence(),
            environment_evidence=_environment_evidence(),
        )
        _populate_prefix_output(
            run_dir,
            manifest,
            drop_correctness_case="cache_cleared",
        )

        try:
            gate.finalize_run(run_dir)
        except ValueError as error:
            assert "prefix raw artifacts" in str(error)
        else:
            raise AssertionError(
                "incomplete Prefix correctness matrix must fail"
            )


def test_finalize_prefix_rejects_summary_that_disagrees_with_raw_rows():
    with TemporaryDirectory() as temporary:
        run_dir = Path(temporary) / "run"
        manifest = gate.initialize_run(
            run_dir=run_dir,
            run_tag="qwen3-06b-prefix-summary-raw-mismatch",
            gate_name="prefix",
            model_tier="qwen3-0.6b",
            source_evidence=_source_evidence(),
            environment_evidence=_environment_evidence(),
        )
        _populate_prefix_output(
            run_dir,
            manifest,
            raw_warm_ttft_matches_cold=True,
        )

        try:
            gate.finalize_run(run_dir)
        except ValueError as error:
            assert "does not match raw Prefix evidence" in str(error)
        else:
            raise AssertionError(
                "Prefix summary/raw disagreement must fail closed"
            )


def test_finalize_prefix_rejects_workload_tamper():
    with TemporaryDirectory() as temporary:
        run_dir = Path(temporary) / "run"
        manifest = gate.initialize_run(
            run_dir=run_dir,
            run_tag="qwen3-06b-prefix-workload-tamper",
            gate_name="prefix",
            model_tier="qwen3-0.6b",
            source_evidence=_source_evidence(),
            environment_evidence=_environment_evidence(),
        )
        _populate_prefix_output(run_dir, manifest)
        workload_path = run_dir / "workload_manifest.jsonl"
        workload = [
            json.loads(line)
            for line in workload_path.read_text(encoding="utf-8").splitlines()
        ]
        workload[0]["case_id"] = "tampered-case"
        _write_jsonl(workload_path, workload)

        try:
            gate.finalize_run(run_dir)
        except ValueError as error:
            assert "workload identity mismatch" in str(error)
        else:
            raise AssertionError("tampered workload must fail finalization")


def _chunked_timeline(
    workload: list[dict],
    *,
    policy: str,
) -> list[dict]:
    rows = []
    ttft_ns = 800 if policy == "FAIR_CHUNKED" else 1_000
    for ordinal, request in enumerate(workload):
        scheduled_ns = 1_000_000_000 + request["arrival_offset_ns"]
        first_token_ns = scheduled_ns + ttft_ns
        output_token_ids = [
            ordinal * 1000 + index
            for index in range(request["requested_output_tokens"])
        ]
        timestamps = [
            first_token_ns + index * 100
            for index in range(len(output_token_ids))
        ]
        rows.append({
            "request_id": request["request_id"],
            "seq_id": ordinal + 1,
            "scheduled_arrival_ns": scheduled_ns,
            "actual_arrival_ns": scheduled_ns + 5,
            "first_scheduled_ns": scheduled_ns + 10,
            "first_token_ns": first_token_ns,
            "token_timestamps_ns": timestamps,
            "completion_ns": timestamps[-1],
            "output_token_ids": output_token_ids,
            "prompt_token_count": request["prompt_tokens"],
            "requested_output_tokens": request[
                "requested_output_tokens"
            ],
            "warmup": request["warmup"],
            "phase": request["phase"],
            "service_time_bucket": request[
                "service_time_bucket"
            ],
            "starvation_deadline_ns": request[
                "starvation_deadline_ns"
            ],
            "finish_reason": "length",
            "error": None,
        })
    return rows


def _populate_chunked_outputs(
    run_dir: Path,
    manifest: dict,
    *,
    duplicate_measured_id: bool = False,
) -> None:
    workload = contract.build_chunked_workload()
    for case_id in manifest["case_order"]:
        case = _read_json(
            run_dir / manifest["case_specs"][case_id]
        )
        case_dir = run_dir / "cases" / case_id
        output_dir = case_dir / "output"
        output_dir.mkdir(parents=True)
        timeline = _chunked_timeline(
            workload,
            policy=case["policy"],
        )
        if duplicate_measured_id:
            timeline[8]["request_id"] = timeline[9]["request_id"]
        _write_jsonl(
            output_dir / "request_timeline.jsonl",
            timeline,
        )
        _write_jsonl(
            output_dir / "scheduler_trace.jsonl",
            [{"step_index": 0, "policy": case["policy"]}],
        )
        _write_jsonl(
            output_dir / "memory_trace.jsonl",
            [{
                "step_index": 0,
                "cuda_peak_reserved_bytes": 1_000,
            }],
        )
        _write_json(
            output_dir / "case_result.json",
            {
                "case_id": case_id,
                "status": "PASS",
                "error_type": None,
                "completed_request_count": 104,
            },
        )
        _write_json(
            case_dir / "process.json",
            {"case_id": case_id, "returncode": 0},
        )
        (case_dir / "stdout.log").write_text("", encoding="utf-8")
        (case_dir / "stderr.log").write_text("", encoding="utf-8")
        (case_dir / "exitcode").write_text("0\n", encoding="utf-8")


def test_finalize_chunked_rebuilds_paired_metrics_and_raw_order():
    with TemporaryDirectory() as temporary:
        run_dir = Path(temporary) / "run"
        manifest = gate.initialize_run(
            run_dir=run_dir,
            run_tag="qwen3-06b-chunked-r1",
            gate_name="chunked",
            model_tier="qwen3-0.6b",
            source_evidence=_source_evidence(),
            environment_evidence=_environment_evidence(
                gate_name="chunked"
            )
        )
        _populate_chunked_outputs(run_dir, manifest)

        summary = gate.finalize_run(run_dir)

        assert summary["classification"] == "FAIR_CHUNKED_GO"
        assert summary["benefit"]["favorable_repetitions"] == 5
        timeline = [
            json.loads(line)
            for line in (
                run_dir / "request_timeline.jsonl"
            ).read_text(encoding="utf-8").splitlines()
        ]
        assert len(timeline) == 1040
        assert timeline[0]["case_id"] == manifest["case_order"][0]
        assert timeline[-1]["case_id"] == manifest["case_order"][-1]
        case_rows = [
            json.loads(line)
            for line in (
                run_dir / "case_rows.jsonl"
            ).read_text(encoding="utf-8").splitlines()
        ]
        assert len(case_rows) == 10
        assert {
            (row["repetition"], row["policy"])
            for row in case_rows
        } == {
            (repetition, policy)
            for repetition in range(5)
            for policy in ("OFF", "FAIR_CHUNKED")
        }


def test_finalize_chunked_rejects_duplicate_measured_request_id():
    with TemporaryDirectory() as temporary:
        run_dir = Path(temporary) / "run"
        manifest = gate.initialize_run(
            run_dir=run_dir,
            run_tag="qwen3-06b-chunked-duplicate-request",
            gate_name="chunked",
            model_tier="qwen3-0.6b",
            source_evidence=_source_evidence(),
            environment_evidence=_environment_evidence(
                gate_name="chunked"
            ),
        )
        _populate_chunked_outputs(
            run_dir,
            manifest,
            duplicate_measured_id=True,
        )

        try:
            gate.finalize_run(run_dir)
        except ValueError as error:
            assert "request identity" in str(error)
        else:
            raise AssertionError(
                "duplicate measured request identity must fail"
            )


def test_qwen8_promotion_binds_two_verified_stage1_summaries():
    prefix_summary = contract.classify_prefix_bundle(
        _complete_prefix_bundle()
    )
    chunked_summary = _chunked_summary()
    winner = contract.select_stage2_winner(
        prefix_summary,
        chunked_summary,
    )["winner"]
    assert winner == "prefix"

    promotion = {
        "winner": winner,
        "prefix_summary": prefix_summary,
        "chunked_summary": chunked_summary,
        "prefix_verification_receipt": {
            "status": "PASS",
            "primary_summary_sha256": contract.canonical_json_sha256(
                prefix_summary
            ),
            "controller_summary_sha256": contract.canonical_json_sha256(
                prefix_summary
            ),
        },
        "chunked_verification_receipt": {
            "status": "PASS",
            "primary_summary_sha256": contract.canonical_json_sha256(
                chunked_summary
            ),
            "controller_summary_sha256": contract.canonical_json_sha256(
                chunked_summary
            ),
        },
    }
    with TemporaryDirectory() as temporary:
        run_dir = Path(temporary) / "run"
        manifest = gate.initialize_run(
            run_dir=run_dir,
            run_tag="qwen3-8b-prefix-r1",
            gate_name="prefix",
            model_tier="qwen3-8b",
            source_evidence=_source_evidence(),
            environment_evidence=_environment_evidence("qwen3-8b"),
            promotion=promotion,
        )
        recorded = manifest["promotion"]
        assert recorded["winner"] == "prefix"
        assert len(recorded["prefix_summary_sha256"]) == 64
        assert len(recorded["chunked_summary_sha256"]) == 64
        assert recorded["selection_rule"]["exact_tie"] == "prefix"

    with TemporaryDirectory() as temporary:
        try:
            gate.initialize_run(
                run_dir=Path(temporary) / "run",
                run_tag="qwen3-8b-chunked-r1",
                gate_name="chunked",
                model_tier="qwen3-8b",
                source_evidence=_source_evidence(),
                environment_evidence=_environment_evidence(
                    "qwen3-8b",
                    gate_name="chunked",
                ),
                promotion={**promotion, "winner": "chunked"},
            )
        except ValueError as error:
            assert "selected winner" in str(error)
        else:
            raise AssertionError("promotion winner mismatch must fail")

    with TemporaryDirectory() as temporary:
        unverified = dict(promotion)
        del unverified["chunked_verification_receipt"]
        try:
            gate.initialize_run(
                run_dir=Path(temporary) / "run",
                run_tag="qwen3-8b-prefix-r2",
                gate_name="prefix",
                model_tier="qwen3-8b",
                source_evidence=_source_evidence(),
                environment_evidence=_environment_evidence("qwen3-8b"),
                promotion=unverified,
            )
        except ValueError as error:
            assert "independent verification" in str(error)
        else:
            raise AssertionError(
                "unverified Stage-1 promotion must fail"
            )


def main():
    test_initialize_binds_source_environment_workload_and_policy()
    test_existing_run_tag_is_never_overwritten()
    test_initialize_rejects_unbound_source_or_model_tier()
    test_launch_case_uses_frozen_worker_cli_and_atomic_receipts()
    test_launch_case_rejects_case_spec_tamper()
    test_finalize_prefix_writes_complete_hashed_primary_bundle()
    test_finalize_prefix_rejects_empty_raw_artifact()
    test_finalize_prefix_rejects_incomplete_correctness_matrix()
    test_finalize_prefix_rejects_summary_that_disagrees_with_raw_rows()
    test_finalize_prefix_rejects_workload_tamper()
    test_finalize_chunked_rebuilds_paired_metrics_and_raw_order()
    test_finalize_chunked_rejects_duplicate_measured_request_id()
    test_qwen8_promotion_binds_two_verified_stage1_summaries()
    print("staged inference benchmark gate tests passed")


if __name__ == "__main__":
    main()
