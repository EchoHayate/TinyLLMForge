from __future__ import annotations

from contextlib import redirect_stdout
from contextlib import redirect_stderr
import importlib.util
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = (
    ROOT
    / "tools/qwen35_tp4_hybrid_prefix_benchmark_contract.py"
)
WORKER_PATH = (
    ROOT
    / "tools/qwen35_tp4_hybrid_prefix_benchmark_worker.py"
)


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


contract = _load(
    "qwen35_tp4_hybrid_prefix_benchmark_contract_for_worker_test",
    CONTRACT_PATH,
)
worker = _load(
    "qwen35_tp4_hybrid_prefix_benchmark_worker",
    WORKER_PATH,
)
builder_fixture = _load(
    "qwen35_prerequisite_builder_fixture_for_worker",
    ROOT / "tools/test_build_qwen35_tp4_performance_prerequisites.py",
)
BENCHMARK_SOURCE_TREE_SHA256 = "c" * 64


def _write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_workload_manifest(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(
        contract.canonical_json_bytes(
            contract.workload_manifest_payload()
        )
        + b"\n"
    )


def _runtime_artifacts(root):
    model_dir = root / "model"
    model_dir.mkdir()
    manifest = root / "model_manifest.json"
    _write_json(manifest, {"local_path": str(model_dir)})
    model_manifest_sha256 = worker._sha256_file(manifest)
    prerequisites = root / "correctness_prerequisites.json"
    prerequisite_payload = {
        "schema_version": contract.PREREQUISITE_SCHEMA_VERSION,
        "model_manifest_sha256": model_manifest_sha256,
    }
    original_fixture_model = (
        builder_fixture.contract.MODEL_MANIFEST_SHA256
    )
    builder_fixture.contract.MODEL_MANIFEST_SHA256 = (
        model_manifest_sha256
    )
    for name in (
        "tp4_root_logit",
        "cached_continuation",
        "engine_correctness",
    ):
        authority_dir = root / name
        authority_dir.mkdir()
        artifact = authority_dir / "artifact.json"
        verification = authority_dir / "independent_verification.json"
        provenance = authority_dir / "provenance.json"
        source_tree_sha256 = (
            contract.TP4_ROOT_SOURCE_TREE_SHA256
            if name == "tp4_root_logit"
            else "d" * 64
        )
        if name == "tp4_root_logit":
            artifact_payload, verification_payload = (
                builder_fixture._root_payloads()
            )
        elif name == "cached_continuation":
            artifact_payload, verification_payload = (
                builder_fixture._cached_payloads(
                    source_tree_sha256
                )
            )
        else:
            artifact_payload, verification_payload = (
                builder_fixture._engine_payloads(
                    source_tree_sha256
                )
            )
        _write_json(artifact, artifact_payload)
        _write_json(verification, verification_payload)
        evidence = {}
        for filename, kind in (
            ("execution_plan.json", "plan"),
            ("consumed_authorization.json", "authorization"),
            ("execution_receipt.json", "receipt"),
        ):
            path = authority_dir / filename
            _write_json(path, {"kind": kind, "authority": name})
            evidence[filename] = worker._sha256_file(path)
        _write_json(provenance, {
            "schema_version": (
                contract.PREREQUISITE_PROVENANCE_SCHEMA_VERSION
            ),
            "authority_name": name,
            "run_tag": name,
            "binding_kind": "remote_execution_receipt",
            "source_tree_sha256": source_tree_sha256,
            "model_manifest_sha256": model_manifest_sha256,
            "root_logit_receipt_gap": False,
            "plan_path": "execution_plan.json",
            "plan_sha256": evidence["execution_plan.json"],
            "authorization_path": "consumed_authorization.json",
            "authorization_sha256": evidence[
                "consumed_authorization.json"
            ],
            "receipt_path": "execution_receipt.json",
            "receipt_sha256": evidence["execution_receipt.json"],
        })
        prerequisite_payload[name] = {
            "run_tag": name,
            "source_tree_sha256": source_tree_sha256,
            "artifact_path": artifact.relative_to(root).as_posix(),
            "artifact_sha256": worker._sha256_file(artifact),
            "independent_verification_path": (
                verification.relative_to(root).as_posix()
            ),
            "independent_verification_sha256": (
                worker._sha256_file(verification)
            ),
            "provenance_path": provenance.relative_to(root).as_posix(),
            "provenance_sha256": worker._sha256_file(provenance),
            "classification": "PASS",
        }
    builder_fixture.contract.MODEL_MANIFEST_SHA256 = (
        original_fixture_model
    )
    _write_json(prerequisites, prerequisite_payload)
    return model_dir, manifest, prerequisites, model_manifest_sha256


class FakeClock:

    def __init__(self):
        self.value = 1_000_000

    def __call__(self):
        self.value += 100
        return self.value


class FakeEngine:

    def __init__(self, configuration):
        self.configuration = configuration
        self.events = []
        self.workload_specs = []
        self.closed = False
        self.cache_snapshot = {
            "current_entries": 2,
            "current_bytes": 4096,
            "current_logical_bytes": 6144,
            "deduplicated_bytes": 2048,
            "peak_entries": 2,
            "peak_bytes": 4096,
            "hits": 5,
            "misses": 1,
            "evictions": 0,
            "validation_failures": 0,
            "failed_restores": 0,
        }

    def configure_qwen35_hybrid_prefix_publication_runtime(
        self,
        *,
        model_fingerprint,
        max_entries,
        max_bytes,
        timeout_s,
    ):
        self.events.append((
            "configure_exact_restore",
            model_fingerprint,
            max_entries,
            max_bytes,
            timeout_s,
        ))

    def run_benchmark_workload(
        self,
        *,
        workload,
        workload_spec,
        phase,
        repetition,
        policy,
    ):
        self.workload_specs.append(workload_spec)
        spec = workload_spec["spec"]
        self.events.append((
            "run",
            workload,
            phase,
            repetition,
            policy,
        ))
        continuation_count = spec["continuations"]
        restored = (
            policy == "exact_restore"
            and workload in {
                "w1_medium_reuse",
                "w2_long_reuse",
                "w3_batched_fanout",
            }
        )
        reused = (
            spec["shared_prefix_tokens"]
            if restored
            else 0
        )
        return {
            "requests": [
                {
                    "request_id": (
                        f"{phase}-{repetition}-{request_index}"
                    ),
                    "prompt_tokens": (
                        spec["shared_prefix_tokens"]
                        + spec["suffix_tokens"]
                    ),
                    "reused_kv_tokens": reused,
                    "restored_hybrid_state": restored,
                    "executed_prefill_tokens": (
                        spec["suffix_tokens"]
                        if restored
                        else (
                            spec["shared_prefix_tokens"]
                            + spec["suffix_tokens"]
                        )
                    ),
                    "generated_tokens": (
                        spec["generated_tokens"]
                    ),
                    "ttft_ns": 500,
                    "e2e_ns": 1500,
                    "decode_step_ns": [
                        100 + step
                        for step in range(
                            spec["generated_tokens"] - 1
                        )
                    ],
                    "output_token_ids": [
                        7 + ((request_index + step) % 11)
                        for step in range(spec["generated_tokens"])
                    ],
                    "final_logits": (
                        [0.25, 0.75]
                        if phase == "correctness"
                        else None
                    ),
                }
                for request_index in range(continuation_count)
            ]
        }

    def memory_snapshot(self):
        return {
            "cuda_allocated_bytes": 1000,
            "cuda_reserved_bytes": 2000,
            "cuda_peak_allocated_bytes": 3000,
            "cuda_peak_reserved_bytes": 4000,
            "kv_capacity_bytes": 5000,
        }

    def capacity_snapshot(self):
        return {
            "num_kvcache_blocks": 64,
            "block_size": 256,
        }

    def hybrid_prefix_cache_snapshot(self):
        return dict(self.cache_snapshot)

    def profile_snapshot(self):
        decode_internal = None
        if self.configuration.get("profiling", {}).get(
            "decode_internal",
            False,
        ):
            ranks = []
            for rank in range(4):
                steps = [{
                    "rank": rank,
                    "step_index": 0,
                    "batch_kind": "prefill",
                    "is_decode": False,
                    "decode_ordinal": None,
                    "active_sequence_count": 1,
                    "request_set_sha256": "a" * 64,
                    "wall_ns": 100,
                    "cuda_ns": 80,
                    "non_cuda_upper_bound_ns": 20,
                    "dispatch": "eager",
                }]
                for decode_ordinal in range(7):
                    steps.append({
                        "rank": rank,
                        "step_index": decode_ordinal + 1,
                        "batch_kind": "decode",
                        "is_decode": True,
                        "decode_ordinal": decode_ordinal,
                        "active_sequence_count": 1,
                        "request_set_sha256": "a" * 64,
                        "wall_ns": 100,
                        "cuda_ns": 80,
                        "non_cuda_upper_bound_ns": 20,
                        "dispatch": "eager",
                    })
                ranks.append({
                    "rank": rank,
                    "enabled": True,
                    "finalization_status": "complete",
                    "steps": steps,
                    "collectives": [],
                })
            decode_internal = {
                "enabled": True,
                "rank_inventory": [0, 1, 2, 3],
                "ranks": ranks,
            }
        return {
            "enabled": bool(
                self.configuration.get("profiling", {}).get(
                    "enabled",
                    False,
                )
            ),
            "events": [{
                "name": "restore_total",
                "request_id": 1,
                "duration_ns": 25,
                "status": "ok",
            }],
            "requests": [{
                "request_id": "request-0",
                "ttft_ns": 500,
                "decode_ns": 6300,
                "e2e_ns": 6800,
                "executed_prefill_tokens": 64,
                "reused_kv_tokens": 3840,
            }],
            "decode_internal": decode_internal,
        }

    def close(self):
        self.events.append(("close",))
        self.closed = True


class FailingEngine(FakeEngine):

    def run_benchmark_workload(self, **kwargs):
        raise RuntimeError("synthetic worker failure")


class FakeTP4CacheEngine(FakeEngine):

    hybrid_prefix_cache_snapshot = None

    def qwen35_hybrid_prefix_cache_snapshots(self, *, timeout_s):
        assert timeout_s == worker.HYBRID_PREFIX_TIMEOUT_S
        return tuple(
            {
                "rank": rank,
                "current_entries": 2,
                "current_bytes": 1024,
                "current_logical_bytes": 1536,
                "deduplicated_bytes": 512,
                "peak_entries": 2,
                "peak_bytes": 1024,
                "hits": 5,
                "misses": 1,
                "evictions": 0,
                "validation_failures": 0,
                "failed_restores": 0,
            }
            for rank in range(4)
        )


class FakeTP4ObservationEngine(FakeTP4CacheEngine):

    def memory_snapshots(self, *, timeout_s):
        assert timeout_s == worker.HYBRID_PREFIX_TIMEOUT_S
        return tuple(
            {
                "rank": rank,
                "cuda_allocated_bytes": 1000 + rank,
                "cuda_reserved_bytes": 2000 + rank,
                "cuda_peak_allocated_bytes": 3000 + rank,
                "cuda_peak_reserved_bytes": 4000 + rank,
                "kv_capacity_bytes": 5000 + rank,
            }
            for rank in range(4)
        )


def test_worker_passes_frozen_token_manifest_to_engine():
    engines = []
    hook_events, cuda_sync, reset_peak = _hooks()
    case = next(
        case
        for case in contract.build_case_matrix()
        if (
            case.policy == "recompute"
            and case.workload == "w3_batched_fanout"
            and case.phase == "warmup"
            and case.repetition == 0
        )
    )
    with tempfile.TemporaryDirectory() as temporary:
        worker.run_benchmark_case(
            case=case,
            output_dir=Path(temporary),
            engine_factory=_factory(engines),
            clock_ns=FakeClock(),
            cuda_sync=cuda_sync,
            reset_peak_memory=reset_peak,
            source_tree_sha256=BENCHMARK_SOURCE_TREE_SHA256,
            correctness_prerequisites_sha256="c" * 64,
        )

    payload = engines[0].workload_specs[0]
    assert payload == contract.workload_payload("w3_batched_fanout")
    assert len(payload["shared_prefix_token_ids"]) == 2048
    assert len(payload["continuations"]) == 8


def test_worker_default_workload_sha_matches_canonical_manifest_file():
    engines = []
    hook_events, cuda_sync, reset_peak = _hooks()
    case = next(
        case
        for case in contract.build_case_matrix()
        if (
            case.policy == "recompute"
            and case.workload == "w0_short_control"
            and case.phase == "warmup"
            and case.repetition == 0
        )
    )
    expected = hashlib.sha256(
        contract.canonical_json_bytes(
            contract.workload_manifest_payload()
        )
        + b"\n"
    ).hexdigest()
    with tempfile.TemporaryDirectory() as temporary:
        output_dir = Path(temporary)
        worker.run_benchmark_case(
            case=case,
            output_dir=output_dir,
            engine_factory=_factory(engines),
            clock_ns=FakeClock(),
            cuda_sync=cuda_sync,
            reset_peak_memory=reset_peak,
            source_tree_sha256=BENCHMARK_SOURCE_TREE_SHA256,
            correctness_prerequisites_sha256="c" * 64,
        )
        row = json.loads(
            (output_dir / "case_rows.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()[0]
        )

    assert row["workload_manifest_sha256"] == expected


def test_request_validation_rejects_non_boolean_restore_state():
    request = FakeEngine({}).run_benchmark_workload(
        workload="w1_medium_reuse",
        workload_spec=contract.workload_payload("w1_medium_reuse"),
        phase="warmup",
        repetition=0,
        policy="exact_restore",
    )["requests"][0]
    request["restored_hybrid_state"] = "false"

    try:
        worker.validate_benchmark_requests(
            workload="w1_medium_reuse",
            policy="exact_restore",
            requests=[request] * 4,
        )
    except ValueError as error:
        assert "restored_hybrid_state" in str(error)
    else:
        raise AssertionError("string restore state was accepted")


def test_request_validation_rejects_inconsistent_prefill_accounting():
    requests = FakeEngine({}).run_benchmark_workload(
        workload="w2_long_reuse",
        workload_spec=contract.workload_payload("w2_long_reuse"),
        phase="warmup",
        repetition=0,
        policy="exact_restore",
    )["requests"]
    requests[0]["executed_prefill_tokens"] += 1

    try:
        worker.validate_benchmark_requests(
            workload="w2_long_reuse",
            policy="exact_restore",
            requests=requests,
        )
    except ValueError as error:
        assert "executed_prefill_tokens" in str(error)
    else:
        raise AssertionError("invalid prefill accounting was accepted")


def test_request_validation_rejects_output_or_request_count_mismatch():
    requests = FakeEngine({}).run_benchmark_workload(
        workload="w3_batched_fanout",
        workload_spec=contract.workload_payload("w3_batched_fanout"),
        phase="warmup",
        repetition=0,
        policy="recompute",
    )["requests"]
    requests[0]["output_token_ids"].pop()

    try:
        worker.validate_benchmark_requests(
            workload="w3_batched_fanout",
            policy="recompute",
            requests=requests,
        )
    except ValueError as error:
        assert "output_token_ids" in str(error)
    else:
        raise AssertionError("short output token list was accepted")

    try:
        worker.validate_benchmark_requests(
            workload="w3_batched_fanout",
            policy="recompute",
            requests=requests[:-1],
        )
    except ValueError as error:
        assert "request count" in str(error)
    else:
        raise AssertionError("short request list was accepted")


def _factory(holder, engine_type=FakeEngine):
    def build(configuration):
        engine = engine_type(configuration)
        holder.append(engine)
        return engine
    return build


def _hooks():
    events = []
    return (
        events,
        lambda: events.append("sync"),
        lambda: events.append("reset_peak"),
    )


def _canonical_correctness_case():
    return next(
        case
        for case in contract.build_case_matrix()
        if (
            case.policy == "exact_restore"
            and case.workload == "w0_short_control"
            and case.phase == "correctness"
            and case.repetition == 0
        )
    )


def _capture_identity_fields():
    return {
        "model_manifest_sha256": contract.MODEL_MANIFEST_SHA256,
        "source_tree_sha256": BENCHMARK_SOURCE_TREE_SHA256,
        "workload_manifest_sha256": (
            contract.canonical_json_file_sha256(
                contract.workload_manifest_payload()
            )
        ),
        "world_size": contract.WORLD_SIZE,
        "workload_ids": list(contract.WORKLOADS),
    }


def test_capture_flag_adds_only_optional_capture_configuration():
    with tempfile.TemporaryDirectory() as temporary:
        capture_root = Path(temporary) / "capture"
        configuration = worker.build_engine_configuration(
            "exact_restore",
            _canonical_correctness_case(),
            recurrent_calibration_capture_dir=capture_root,
            capture_identity_fields=_capture_identity_fields(),
            expected_capture_identity_fields=(
                _capture_identity_fields()
            ),
        )

    assert configuration["recurrent_calibration_capture"] == {
        "capture_root": str(capture_root.resolve()),
        **_capture_identity_fields(),
    }


def test_capture_cli_rejects_empty_or_whitespace_path():
    for raw_value in ("", "   "):
        stderr = io.StringIO()
        try:
            with redirect_stderr(stderr):
                worker.main([
                    "--policy", "exact_restore",
                    "--workload", "w0_short_control",
                    "--phase", "correctness",
                    "--repetition", "0",
                    "--output-dir", "/tmp/output",
                    "--source-tree-sha256", "a" * 64,
                    "--model-manifest-sha256", "b" * 64,
                    "--prerequisites-sha256", "c" * 64,
                    "--model-dir", "/tmp/model",
                    "--model-manifest", "/tmp/model.json",
                    "--correctness-prerequisites", "/tmp/prereq.json",
                    "--workload-manifest", "/tmp/workload.json",
                    "--workload-manifest-sha256", "d" * 64,
                    "--recurrent-calibration-capture-dir",
                    raw_value,
                ])
        except SystemExit as error:
            assert error.code == 2
        else:
            raise AssertionError("empty capture CLI path accepted")
        assert "capture" in stderr.getvalue().lower()


def test_no_capture_flag_preserves_existing_configuration():
    case = _canonical_correctness_case()
    expected = {
        "schema_version": contract.SCHEMA_VERSION,
        "policy": "exact_restore",
        "tensor_parallel_size": 4,
        "num_kvcache_blocks": 64,
        "kvcache_block_size": 256,
        "engine": {
            "tensor_parallel_size": 4,
            "num_kvcache_blocks": 64,
            "kvcache_block_size": 256,
            "enforce_eager": True,
            "max_model_len": 4352,
            "max_num_batched_tokens": 17408,
            "max_num_seqs": 8,
        },
        "sampling": {
            "temperature": 0.0,
            "ignore_eos": True,
            "max_tokens": contract.WORKLOAD_SPECS[
                "w0_short_control"
            ]["generated_tokens"],
        },
        "workload": {
            "name": "w0_short_control",
            **contract.WORKLOAD_SPECS["w0_short_control"],
        },
        "hybrid_prefix": {
            "enabled": True,
            "representation": "exact_full_fidelity",
            "max_entries": 16,
            "max_bytes": 2 * 1024**3,
            "timeout_s": 120.0,
        },
    }

    actual = worker.build_engine_configuration(
        "exact_restore",
        case,
        recurrent_calibration_capture_dir=None,
        capture_identity_fields=None,
    )

    assert actual == expected
    assert "recurrent_calibration_capture" not in actual


def test_capture_flag_rejects_ineligible_case_and_file_root():
    base = _canonical_correctness_case()
    ineligible = (
        next(
            case
            for case in contract.build_case_matrix()
            if (
                case.policy == "recompute"
                and case.phase == "correctness"
                and case.repetition == 0
            )
        ),
        next(
            case
            for case in contract.build_case_matrix()
            if (
                case.policy == "exact_restore"
                and case.phase == "warmup"
                and case.repetition == 0
            )
        ),
        next(
            case
            for case in contract.build_case_matrix()
            if (
                case.policy == "exact_restore"
                and case.phase == "measured"
                and case.repetition == 0
            )
        ),
    )
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        for case in ineligible:
            try:
                worker.build_engine_configuration(
                    case.policy,
                    case,
                    recurrent_calibration_capture_dir=root / "capture",
                    capture_identity_fields=_capture_identity_fields(),
                        expected_capture_identity_fields=(
                            _capture_identity_fields()
                        ),
                )
            except ValueError as error:
                assert "capture" in str(error)
            else:
                raise AssertionError("ineligible capture case accepted")
        capture_file = root / "capture-file"
        capture_file.write_text("not a directory", encoding="utf-8")
        try:
            worker.build_engine_configuration(
                base.policy,
                base,
                recurrent_calibration_capture_dir=capture_file,
                capture_identity_fields=_capture_identity_fields(),
                expected_capture_identity_fields=(
                    _capture_identity_fields()
                ),
            )
        except ValueError as error:
            assert "directory" in str(error)
        else:
            raise AssertionError("capture file root accepted")


def test_capture_flag_rejects_identity_different_from_authority():
    mutations = (
        {"model_manifest_sha256": "0" * 64},
        {"source_tree_sha256": "0" * 64},
        {"workload_manifest_sha256": "0" * 64},
        {"world_size": 2},
        {"world_size": 4.0},
        {"workload_ids": list(reversed(contract.WORKLOADS))},
    )
    with tempfile.TemporaryDirectory() as temporary:
        for mutation in mutations:
            fields = {**_capture_identity_fields(), **mutation}
            try:
                worker.build_engine_configuration(
                    "exact_restore",
                    _canonical_correctness_case(),
                    recurrent_calibration_capture_dir=(
                        Path(temporary) / "capture"
                    ),
                    capture_identity_fields=fields,
                    expected_capture_identity_fields=(
                        _capture_identity_fields()
                    ),
                )
            except ValueError as error:
                assert "identity" in str(error)
            else:
                raise AssertionError(
                    "noncanonical capture identity accepted"
                )


def test_run_case_binds_capture_identity_to_validated_inputs():
    engines = []
    hook_events, cuda_sync, reset_peak = _hooks()
    model_sha = "e" * 64
    workload_sha = "f" * 64
    source_sha = BENCHMARK_SOURCE_TREE_SHA256
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        capture_root = root / "capture"
        worker.run_benchmark_case(
            case=_canonical_correctness_case(),
            output_dir=root / "output",
            engine_factory=_factory(engines),
            clock_ns=FakeClock(),
            cuda_sync=cuda_sync,
            reset_peak_memory=reset_peak,
            source_tree_sha256=source_sha,
            model_manifest_sha256=model_sha,
            workload_manifest_sha256=workload_sha,
            correctness_prerequisites_sha256="1" * 64,
            recurrent_calibration_capture_dir=capture_root,
        )

    assert engines[0].configuration[
        "recurrent_calibration_capture"
    ] == {
        "capture_root": str(capture_root.resolve()),
        "model_manifest_sha256": model_sha,
        "source_tree_sha256": source_sha,
        "workload_manifest_sha256": workload_sha,
        "world_size": contract.WORLD_SIZE,
        "workload_ids": list(contract.WORKLOADS),
    }


def test_engine_configuration_diff_is_limited_to_exact_restore_switch():
    case = contract.build_case_matrix()[0]

    recompute = worker.build_engine_configuration(
        "recompute",
        case,
    )
    exact = worker.build_engine_configuration(
        "exact_restore",
        case,
    )

    assert recompute["tensor_parallel_size"] == 4
    assert recompute["num_kvcache_blocks"] == 64
    assert recompute["kvcache_block_size"] == 256
    assert recompute["engine"]["max_model_len"] >= 4224
    assert recompute["engine"]["max_num_batched_tokens"] >= 16896
    assert recompute["sampling"] == exact["sampling"]
    assert recompute["workload"] == exact["workload"]
    assert recompute["engine"] == exact["engine"]
    assert recompute["hybrid_prefix"] == {
        "enabled": False,
        "representation": "none",
    }
    assert exact["hybrid_prefix"] == {
        "enabled": True,
        "representation": "exact_full_fidelity",
        "max_entries": 16,
        "max_bytes": 2 * 1024**3,
        "timeout_s": 120.0,
    }


def test_recompute_never_configures_hybrid_prefix_runtime():
    engines = []
    hook_events, cuda_sync, reset_peak = _hooks()
    with tempfile.TemporaryDirectory() as temporary:
        result = worker.run_policy_workload(
            policy="recompute",
            workload="w1_medium_reuse",
            output_dir=Path(temporary),
            engine_factory=_factory(engines),
            clock_ns=FakeClock(),
            cuda_sync=cuda_sync,
            reset_peak_memory=reset_peak,
            source_tree_sha256=BENCHMARK_SOURCE_TREE_SHA256,
        )

    assert not any(
        event[0] == "configure_exact_restore"
        for event in engines[0].events
    )
    assert result["complete"] is True
    assert hook_events.count("reset_peak") == 1
    assert engines[0].closed is True


def test_exact_restore_configures_full_fidelity_runtime_once():
    engines = []
    hook_events, cuda_sync, reset_peak = _hooks()
    with tempfile.TemporaryDirectory() as temporary:
        worker.run_policy_workload(
            policy="exact_restore",
            workload="w1_medium_reuse",
            output_dir=Path(temporary),
            engine_factory=_factory(engines),
            clock_ns=FakeClock(),
            cuda_sync=cuda_sync,
            reset_peak_memory=reset_peak,
            source_tree_sha256=BENCHMARK_SOURCE_TREE_SHA256,
        )

    configure_events = [
        event
        for event in engines[0].events
        if event[0] == "configure_exact_restore"
    ]
    assert configure_events == [(
        "configure_exact_restore",
        worker.MODEL_FINGERPRINT,
        16,
        2 * 1024**3,
        120.0,
    )]
    assert hook_events.count("reset_peak") == 1


def test_worker_runs_one_warmup_one_correctness_and_five_measured():
    engines = []
    hook_events, cuda_sync, reset_peak = _hooks()
    with tempfile.TemporaryDirectory() as temporary:
        result = worker.run_policy_workload(
            policy="exact_restore",
            workload="w2_long_reuse",
            output_dir=Path(temporary),
            engine_factory=_factory(engines),
            clock_ns=FakeClock(),
            cuda_sync=cuda_sync,
            reset_peak_memory=reset_peak,
            source_tree_sha256=BENCHMARK_SOURCE_TREE_SHA256,
        )

    run_events = [
        event
        for event in engines[0].events
        if event[0] == "run"
    ]
    assert [event[2:] for event in run_events] == [
        ("warmup", 0, "exact_restore"),
        ("correctness", 0, "exact_restore"),
        ("measured", 0, "exact_restore"),
        ("measured", 1, "exact_restore"),
        ("measured", 2, "exact_restore"),
        ("measured", 3, "exact_restore"),
        ("measured", 4, "exact_restore"),
    ]
    assert result["case_rows"] == 28
    assert result["process_rows"] == 7
    assert hook_events[0] == "sync"
    assert hook_events[-1] == "sync"


def test_single_case_worker_executes_only_authorized_case():
    engines = []
    hook_events, cuda_sync, reset_peak = _hooks()
    case = next(
        case
        for case in contract.build_case_matrix()
        if case.workload == "w2_long_reuse"
        and case.policy == "exact_restore"
        and case.phase == "measured"
        and case.repetition == 3
    )
    with tempfile.TemporaryDirectory() as temporary:
        result = worker.run_benchmark_case(
            case=case,
            output_dir=Path(temporary),
            engine_factory=_factory(engines),
            clock_ns=FakeClock(),
            cuda_sync=cuda_sync,
            reset_peak_memory=reset_peak,
            source_tree_sha256=BENCHMARK_SOURCE_TREE_SHA256,
            correctness_prerequisites_sha256="a" * 64,
        )

    run_events = [
        event
        for event in engines[0].events
        if event[0] == "run"
    ]
    assert run_events == [(
        "run",
        "w2_long_reuse",
        "measured",
        3,
        "exact_restore",
    )]
    assert hook_events.count("reset_peak") == 1
    assert result == {
        "schema_version": contract.SCHEMA_VERSION,
        "complete": True,
        "case_id": case.case_id,
        "case_rows": 4,
        "process_rows": 1,
    }
    assert engines[0].closed is True


def test_exact_restore_uses_authorized_model_manifest_sha_as_fingerprint():
    engines = []
    hook_events, cuda_sync, reset_peak = _hooks()
    model_manifest_sha256 = "e" * 64
    case = next(
        case
        for case in contract.build_case_matrix()
        if case.workload == "w0_short_control"
        and case.policy == "exact_restore"
        and case.phase == "warmup"
        and case.repetition == 0
    )

    with tempfile.TemporaryDirectory() as temporary:
        worker.run_benchmark_case(
            case=case,
            output_dir=Path(temporary),
            engine_factory=_factory(engines),
            clock_ns=FakeClock(),
            cuda_sync=cuda_sync,
            reset_peak_memory=reset_peak,
            source_tree_sha256=BENCHMARK_SOURCE_TREE_SHA256,
            model_manifest_sha256=model_manifest_sha256,
            correctness_prerequisites_sha256="a" * 64,
        )

    configure_event = next(
        event
        for event in engines[0].events
        if event[0] == "configure_exact_restore"
    )
    assert configure_event[1] == model_manifest_sha256


def test_single_case_worker_rejects_noncanonical_case_before_engine():
    engines = []
    hook_events, cuda_sync, reset_peak = _hooks()
    invalid = contract.BenchmarkCase(
        case_id="w2_long_reuse__measured__r99__exact_restore",
        workload="w2_long_reuse",
        policy="exact_restore",
        phase="measured",
        repetition=99,
    )
    with tempfile.TemporaryDirectory() as temporary:
        try:
            worker.run_benchmark_case(
                case=invalid,
                output_dir=Path(temporary),
                engine_factory=_factory(engines),
                clock_ns=FakeClock(),
                cuda_sync=cuda_sync,
                reset_peak_memory=reset_peak,
                source_tree_sha256=BENCHMARK_SOURCE_TREE_SHA256,
                correctness_prerequisites_sha256="a" * 64,
            )
        except ValueError as error:
            assert "canonical" in str(error)
        else:
            raise AssertionError("noncanonical case was accepted")

    assert engines == []


def test_runtime_artifacts_bind_model_and_prerequisite_files():
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        (
            model_dir,
            manifest,
            prerequisites,
            model_manifest_sha256,
        ) = _runtime_artifacts(root)
        workload_manifest = root / "workload_manifest.json"
        _write_workload_manifest(workload_manifest)
        original = worker.contract.MODEL_MANIFEST_SHA256
        worker.contract.MODEL_MANIFEST_SHA256 = model_manifest_sha256
        try:
            result = worker.validate_runtime_artifacts(
                model_dir=model_dir,
                model_manifest_path=manifest,
                expected_model_manifest_sha256=model_manifest_sha256,
                correctness_prerequisites_path=prerequisites,
                expected_correctness_prerequisites_sha256=(
                    worker._sha256_file(prerequisites)
                ),
                workload_manifest_path=workload_manifest,
                expected_workload_manifest_sha256=(
                    worker._sha256_file(workload_manifest)
                ),
            )
        finally:
            worker.contract.MODEL_MANIFEST_SHA256 = original

        assert result["model_dir"] == str(model_dir)
        assert result["model_manifest_sha256"] == model_manifest_sha256
        assert result["correctness_prerequisites_sha256"] == (
            worker._sha256_file(prerequisites)
        )
        assert result["workload_manifest_sha256"] == (
            worker._sha256_file(workload_manifest)
        )


def test_runtime_artifacts_reject_tampered_workload_manifest():
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        (
            model_dir,
            manifest,
            prerequisites,
            model_manifest_sha256,
        ) = _runtime_artifacts(root)
        workload_manifest = root / "workload_manifest.json"
        payload = contract.workload_manifest_payload()
        _write_workload_manifest(workload_manifest)
        expected_sha256 = worker._sha256_file(workload_manifest)
        payload["workloads"]["w1_medium_reuse"][
            "shared_prefix_token_ids"
        ][0] += 1
        _write_json(workload_manifest, payload)
        original = worker.contract.MODEL_MANIFEST_SHA256
        worker.contract.MODEL_MANIFEST_SHA256 = model_manifest_sha256
        try:
            try:
                worker.validate_runtime_artifacts(
                    model_dir=model_dir,
                    model_manifest_path=manifest,
                    expected_model_manifest_sha256=(
                        model_manifest_sha256
                    ),
                    correctness_prerequisites_path=prerequisites,
                    expected_correctness_prerequisites_sha256=(
                        worker._sha256_file(prerequisites)
                    ),
                    workload_manifest_path=workload_manifest,
                    expected_workload_manifest_sha256=expected_sha256,
                )
            except ValueError as error:
                assert "workload manifest SHA" in str(error)
            else:
                raise AssertionError(
                    "tampered workload manifest was accepted"
                )
        finally:
            worker.contract.MODEL_MANIFEST_SHA256 = original


def test_runtime_artifacts_reject_hash_mismatch_before_runtime_load():
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        model_dir, manifest, prerequisites, _ = _runtime_artifacts(root)
        try:
            worker.validate_runtime_artifacts(
                model_dir=model_dir,
                model_manifest_path=manifest,
                expected_model_manifest_sha256="0" * 64,
                correctness_prerequisites_path=prerequisites,
                expected_correctness_prerequisites_sha256=(
                    worker._sha256_file(prerequisites)
                ),
            )
        except ValueError as error:
            assert "model manifest SHA" in str(error)
        else:
            raise AssertionError("model manifest mismatch was accepted")


def test_runtime_artifacts_reject_noncanonical_model_manifest():
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        (
            model_dir,
            manifest,
            prerequisites,
            model_manifest_sha256,
        ) = _runtime_artifacts(root)
        try:
            worker.validate_runtime_artifacts(
                model_dir=model_dir,
                model_manifest_path=manifest,
                expected_model_manifest_sha256=model_manifest_sha256,
                correctness_prerequisites_path=prerequisites,
                expected_correctness_prerequisites_sha256=(
                    worker._sha256_file(prerequisites)
                ),
            )
        except ValueError as error:
            assert "canonical model manifest" in str(error)
        else:
            raise AssertionError("noncanonical model manifest was accepted")


def test_runtime_artifacts_reject_handwritten_top_level_pass():
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        model_dir = root / "model"
        model_dir.mkdir()
        manifest = root / "model_manifest.json"
        _write_json(manifest, {"local_path": str(model_dir)})
        prerequisites = root / "correctness_prerequisites.json"
        _write_json(prerequisites, {"classification": "PASS"})
        model_manifest_sha256 = worker._sha256_file(manifest)
        original = worker.contract.MODEL_MANIFEST_SHA256
        worker.contract.MODEL_MANIFEST_SHA256 = model_manifest_sha256
        try:
            try:
                worker.validate_runtime_artifacts(
                    model_dir=model_dir,
                    model_manifest_path=manifest,
                    expected_model_manifest_sha256=(
                        model_manifest_sha256
                    ),
                    correctness_prerequisites_path=prerequisites,
                    expected_correctness_prerequisites_sha256=(
                        worker._sha256_file(prerequisites)
                    ),
                )
            except ValueError as error:
                assert "correctness prerequisites" in str(error)
            else:
                raise AssertionError(
                    "handwritten prerequisite PASS was accepted"
                )
        finally:
            worker.contract.MODEL_MANIFEST_SHA256 = original


def test_worker_cli_authorizes_before_loading_runtime():
    events = []
    engines = []
    hook_events, cuda_sync, reset_peak = _hooks()
    case = contract.build_case_matrix()[0]
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        (
            model_dir,
            manifest,
            prerequisites,
            model_manifest_sha256,
        ) = _runtime_artifacts(root)
        workload_manifest = root / "workload_manifest.json"
        _write_workload_manifest(workload_manifest)

        def runtime_loader(configuration, authorized):
            events.append(("runtime", configuration, authorized))
            return _factory(engines)(configuration)

        original = worker.contract.MODEL_MANIFEST_SHA256
        worker.contract.MODEL_MANIFEST_SHA256 = model_manifest_sha256
        try:
            stdout = io.StringIO()
            with redirect_stdout(stdout):
                result = worker.main(
                    [
                        "--policy", case.policy,
                        "--workload", case.workload,
                        "--phase", case.phase,
                        "--repetition", str(case.repetition),
                        "--output-dir", str(root / "output"),
                        "--source-tree-sha256",
                        BENCHMARK_SOURCE_TREE_SHA256,
                        "--model-manifest-sha256",
                        model_manifest_sha256,
                        "--prerequisites-sha256",
                        worker._sha256_file(prerequisites),
                        "--model-dir", str(model_dir),
                        "--model-manifest", str(manifest),
                        "--correctness-prerequisites",
                        str(prerequisites),
                        "--workload-manifest",
                        str(workload_manifest),
                        "--workload-manifest-sha256",
                        worker._sha256_file(workload_manifest),
                    ],
                    runtime_loader=runtime_loader,
                    clock_ns=FakeClock(),
                    cuda_sync=cuda_sync,
                    reset_peak_memory=reset_peak,
                )
        finally:
            worker.contract.MODEL_MANIFEST_SHA256 = original

    assert result == 0
    assert stdout.getvalue() == (
        "QWEN35_TP4_BENCHMARK_WORKER_COMPLETE\n"
    )
    assert len(events) == 1
    assert events[0][2]["model_dir"] == str(model_dir)
    assert engines[0].closed is True


def test_default_runtime_loader_lazily_constructs_engine_adapter():
    events = []

    class Adapter:

        def __init__(self, configuration, authorized):
            events.append((configuration, authorized))

    module_name = (
        "qwen35_tp4_hybrid_prefix_benchmark_engine_adapter"
    )
    previous = sys.modules.get(module_name)
    module = type(sys)(module_name)
    module.BenchmarkEngineAdapter = Adapter
    sys.modules[module_name] = module
    try:
        result = worker._default_runtime_loader(
            {"policy": "recompute"},
            {"model_dir": "/model"},
        )
    finally:
        if previous is None:
            sys.modules.pop(module_name, None)
        else:
            sys.modules[module_name] = previous

    assert isinstance(result, Adapter)
    assert events == [
        ({"policy": "recompute"}, {"model_dir": "/model"})
    ]


def test_raw_rows_include_hashes_timing_outputs_logits_memory_and_cache():
    engines = []
    hook_events, cuda_sync, reset_peak = _hooks()
    with tempfile.TemporaryDirectory() as temporary:
        output_dir = Path(temporary)
        worker.run_policy_workload(
            policy="exact_restore",
            workload="w1_medium_reuse",
            output_dir=output_dir,
            engine_factory=_factory(engines),
            clock_ns=FakeClock(),
            cuda_sync=cuda_sync,
            reset_peak_memory=reset_peak,
            source_tree_sha256=BENCHMARK_SOURCE_TREE_SHA256,
        )
        case_rows = [
            json.loads(line)
            for line in (output_dir / "case_rows.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        ]
        process_rows = [
            json.loads(line)
            for line in (output_dir / "process_rows.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        ]

        correctness = next(
            row
            for row in case_rows
            if row["phase"] == "correctness"
        )
        measured_process = next(
            row
            for row in process_rows
            if row["phase"] == "measured"
        )

        assert set(correctness) == set(contract.CASE_ROW_FIELDS)
        assert correctness["ttft_ns"] == 500
        assert len(correctness["decode_step_ns"]) == 63
        assert all(
            value > 0 for value in correctness["decode_step_ns"]
        )
        assert correctness["output_token_ids"][:2] == [7, 8]
        assert len(correctness["output_token_ids"]) == 64
        assert len(correctness["output_token_ids_sha256"]) == 64
        assert correctness["final_logits_path"].startswith("logits/")
        assert len(correctness["final_logits_sha256"]) == 64
        assert (
            output_dir / correctness["final_logits_path"]
        ).is_file()

        assert set(measured_process) == set(
            contract.PROCESS_ROW_FIELDS
        )
        assert measured_process["cuda_peak_reserved_bytes"] == 4000
        assert measured_process["kv_capacity_bytes"] == 5000
        assert measured_process["scheduler_visible_kv_blocks"] == 64
        assert measured_process["hybrid_cache_current_bytes"] == 4096
        assert (
            measured_process["hybrid_cache_current_logical_bytes"]
            == 6144
        )
        assert measured_process["hybrid_cache_deduplicated_bytes"] == 2048


def test_recompute_process_rows_force_zero_hybrid_cache_observation():
    engines = []
    hook_events, cuda_sync, reset_peak = _hooks()
    with tempfile.TemporaryDirectory() as temporary:
        output_dir = Path(temporary)
        worker.run_policy_workload(
            policy="recompute",
            workload="w1_medium_reuse",
            output_dir=output_dir,
            engine_factory=_factory(engines),
            clock_ns=FakeClock(),
            cuda_sync=cuda_sync,
            reset_peak_memory=reset_peak,
            source_tree_sha256=BENCHMARK_SOURCE_TREE_SHA256,
        )
        process_rows = [
            json.loads(line)
            for line in (output_dir / "process_rows.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        ]

    cache_fields = [
        field
        for field in contract.PROCESS_ROW_FIELDS
        if field.startswith("hybrid_cache_")
    ]
    assert all(
        row[field] == 0
        for row in process_rows
        for field in cache_fields
    )


def test_worker_uses_all_rank_cache_transport_for_exact_restore():
    engines = []
    hook_events, cuda_sync, reset_peak = _hooks()
    with tempfile.TemporaryDirectory() as temporary:
        output_dir = Path(temporary)
        worker.run_policy_workload(
            policy="exact_restore",
            workload="w1_medium_reuse",
            output_dir=output_dir,
            engine_factory=_factory(engines, FakeTP4CacheEngine),
            clock_ns=FakeClock(),
            cuda_sync=cuda_sync,
            reset_peak_memory=reset_peak,
            source_tree_sha256=BENCHMARK_SOURCE_TREE_SHA256,
        )
        process_rows = [
            json.loads(line)
            for line in (output_dir / "process_rows.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        ]

    assert process_rows[0]["hybrid_cache_current_bytes"] == 4096
    assert (
        process_rows[0]["hybrid_cache_current_logical_bytes"]
        == 6144
    )
    assert process_rows[0]["hybrid_cache_deduplicated_bytes"] == 2048
    assert process_rows[0]["hybrid_cache_hits"] == 5


def test_worker_uses_all_rank_memory_transport():
    engines = []
    hook_events, cuda_sync, reset_peak = _hooks()
    with tempfile.TemporaryDirectory() as temporary:
        output_dir = Path(temporary)
        worker.run_policy_workload(
            policy="exact_restore",
            workload="w1_medium_reuse",
            output_dir=output_dir,
            engine_factory=_factory(
                engines,
                FakeTP4ObservationEngine,
            ),
            clock_ns=FakeClock(),
            cuda_sync=cuda_sync,
            reset_peak_memory=reset_peak,
            source_tree_sha256=BENCHMARK_SOURCE_TREE_SHA256,
        )
        row = json.loads(
            (output_dir / "process_rows.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()[0]
        )

    assert row["cuda_allocated_bytes"] == 4006
    assert row["cuda_reserved_bytes"] == 8006
    assert row["cuda_peak_allocated_bytes"] == 12006
    assert row["cuda_peak_reserved_bytes"] == 16006
    assert row["kv_capacity_bytes"] == 20006


def test_tp4_memory_snapshot_rejects_missing_rank():
    rows = [
        {
            "rank": rank,
            "cuda_allocated_bytes": 1000,
            "cuda_reserved_bytes": 2000,
            "cuda_peak_allocated_bytes": 3000,
            "cuda_peak_reserved_bytes": 4000,
            "kv_capacity_bytes": 5000,
        }
        for rank in range(3)
    ]
    try:
        worker.aggregate_tp4_memory_snapshots(rows)
    except ValueError as error:
        assert "rank inventory" in str(error)
    else:
        raise AssertionError("missing TP4 memory rank was accepted")


def test_tp4_cache_snapshot_sums_bytes_and_requires_counter_parity():
    rows = [
        {
            "rank": rank,
            "current_entries": 2,
            "current_bytes": 100 + rank,
            "current_logical_bytes": 150 + rank,
            "deduplicated_bytes": 50,
            "peak_entries": 3,
            "peak_bytes": 200 + rank,
            "hits": 5,
            "misses": 1,
            "evictions": 0,
            "validation_failures": 0,
            "failed_restores": 0,
        }
        for rank in range(4)
    ]
    snapshot = worker.aggregate_tp4_cache_snapshots(rows)

    assert snapshot == {
        "current_entries": 2,
        "current_bytes": 406,
        "current_logical_bytes": 606,
        "deduplicated_bytes": 200,
        "peak_entries": 3,
        "peak_bytes": 806,
        "hits": 5,
        "misses": 1,
        "evictions": 0,
        "validation_failures": 0,
        "failed_restores": 0,
    }


def test_tp4_cache_snapshot_accepts_transport_extension_fields():
    rows = [
        {
            "rank": rank,
            "representation": "int8",
            "representation_version": "v1",
            "codec": "symmetric-per-token",
            "current_entries": 2,
            "current_bytes": 100 + rank,
            "current_logical_bytes": 150 + rank,
            "deduplicated_bytes": 50,
            "peak_entries": 3,
            "peak_bytes": 200 + rank,
            "publishes": 4,
            "hits": 5,
            "misses": 1,
            "evictions": 0,
            "validation_failures": 0,
            "failed_restores": 0,
            "current_encoded_physical_bytes": 100 + rank,
            "current_encoded_logical_bytes": 150 + rank,
            "current_full_fidelity_logical_bytes": 300 + rank,
            "current_codec_metadata_bytes": 8,
            "current_reader_leases": 0,
            "current_temporary_encode_workspace_bytes": 0,
            "current_temporary_decode_workspace_bytes": 0,
            "current_temporary_decode_cuda_allocated_bytes": 0,
            "current_temporary_decode_cuda_reserved_bytes": 0,
            "peak_encoded_logical_bytes": 200 + rank,
            "peak_full_fidelity_logical_bytes": 400 + rank,
            "peak_codec_metadata_bytes": 8,
            "peak_reader_leases": 0,
            "peak_temporary_encode_workspace_bytes": 0,
            "peak_temporary_decode_workspace_bytes": 0,
            "peak_temporary_decode_cuda_allocated_bytes": 0,
            "peak_temporary_decode_cuda_reserved_bytes": 0,
            "deferred_snapshot_releases": 0,
            "quarantines": 0,
            "decode_failures": 0,
            "commit_failures": 0,
            "rollback_failures": 0,
            "fallbacks": 0,
            "partial_restore_attempts": 0,
            "mixed_representation_rejections": 0,
            "missing_layer_rejections": 0,
        }
        for rank in range(4)
    ]

    snapshot = worker.aggregate_tp4_cache_snapshots(rows)

    assert snapshot == {
        "current_entries": 2,
        "current_bytes": 406,
        "current_logical_bytes": 606,
        "deduplicated_bytes": 200,
        "peak_entries": 3,
        "peak_bytes": 806,
        "hits": 5,
        "misses": 1,
        "evictions": 0,
        "validation_failures": 0,
        "failed_restores": 0,
    }


def test_tp4_cache_snapshot_rejects_missing_rank_and_counter_divergence():
    base = {
        "current_entries": 2,
        "current_bytes": 100,
        "current_logical_bytes": 150,
        "deduplicated_bytes": 50,
        "peak_entries": 3,
        "peak_bytes": 200,
        "hits": 5,
        "misses": 1,
        "evictions": 0,
        "validation_failures": 0,
        "failed_restores": 0,
    }
    try:
        worker.aggregate_tp4_cache_snapshots([
            {"rank": rank, **base} for rank in range(3)
        ])
    except ValueError as error:
        assert "rank inventory" in str(error)
    else:
        raise AssertionError("missing TP4 cache rank was accepted")

    rows = [{"rank": rank, **base} for rank in range(4)]
    rows[3]["hits"] = 6
    try:
        worker.aggregate_tp4_cache_snapshots(rows)
    except ValueError as error:
        assert "counter parity" in str(error)
    else:
        raise AssertionError("divergent TP4 cache counter was accepted")


def test_worker_failure_is_atomic_and_closes_engine():
    engines = []
    hook_events, cuda_sync, reset_peak = _hooks()
    with tempfile.TemporaryDirectory() as temporary:
        output_dir = Path(temporary)
        try:
            worker.run_policy_workload(
                policy="exact_restore",
                workload="w1_medium_reuse",
                output_dir=output_dir,
                engine_factory=_factory(engines, FailingEngine),
                clock_ns=FakeClock(),
                cuda_sync=cuda_sync,
                reset_peak_memory=reset_peak,
                source_tree_sha256=BENCHMARK_SOURCE_TREE_SHA256,
            )
        except RuntimeError as error:
            assert "synthetic worker failure" in str(error)
        else:
            raise AssertionError("expected synthetic worker failure")

        failure = json.loads(
            (output_dir / "failure.json").read_text(encoding="utf-8")
        )
        assert failure["complete"] is False
        assert failure["error_type"] == "RuntimeError"
        assert "synthetic worker failure" in failure["error"]
        assert not list(output_dir.glob("*.partial"))

    assert engines[0].closed is True


def test_worker_requires_explicit_benchmark_source_identity():
    engines = []
    hook_events, cuda_sync, reset_peak = _hooks()
    with tempfile.TemporaryDirectory() as temporary:
        try:
            worker.run_policy_workload(
                policy="recompute",
                workload="w1_medium_reuse",
                output_dir=Path(temporary),
                engine_factory=_factory(engines),
                clock_ns=FakeClock(),
                cuda_sync=cuda_sync,
                reset_peak_memory=reset_peak,
                source_tree_sha256=None,
            )
        except ValueError as error:
            assert "benchmark source tree" in str(error)
        else:
            raise AssertionError(
                "missing benchmark source tree was accepted"
            )


def test_profile_mode_writes_separate_profile_artifact():
    engines = []
    _, cuda_sync, reset_peak = _hooks()
    case = next(
        case
        for case in contract.build_case_matrix()
        if (
            case.policy == "exact_restore"
            and case.workload == "w2_long_reuse"
            and case.phase == "measured"
            and case.repetition == 0
        )
    )
    with tempfile.TemporaryDirectory() as temporary:
        output_dir = Path(temporary)
        worker.run_benchmark_case(
            case=case,
            output_dir=output_dir,
            engine_factory=_factory(engines),
            clock_ns=FakeClock(),
            cuda_sync=cuda_sync,
            reset_peak_memory=reset_peak,
            source_tree_sha256=BENCHMARK_SOURCE_TREE_SHA256,
            correctness_prerequisites_sha256="d" * 64,
            profiling=True,
        )
        profile = json.loads(
            (output_dir / "profile.json").read_text(
                encoding="utf-8"
            )
        )

    assert engines[0].configuration["profiling"] == {
        "enabled": True
    }
    assert profile["schema_version"] == (
        "qwen35.tp4-w2-restore-profile-case.v1"
    )
    assert profile["case_id"] == case.case_id
    assert profile["variant"] == "canonical_output"
    assert profile["canonical_generated_tokens"] == 64
    assert profile["generated_tokens"] == 64
    assert profile["events"][0]["name"] == "restore_total"
    assert profile["requests"][0]["reused_kv_tokens"] == 3840


def test_profile_mode_can_override_w2_generated_tokens():
    engines = []
    _, cuda_sync, reset_peak = _hooks()
    case = next(
        case
        for case in contract.build_case_matrix()
        if (
            case.policy == "exact_restore"
            and case.workload == "w2_long_reuse"
            and case.phase == "measured"
            and case.repetition == 0
        )
    )
    with tempfile.TemporaryDirectory() as temporary:
        output_dir = Path(temporary)
        worker.run_benchmark_case(
            case=case,
            output_dir=output_dir,
            engine_factory=_factory(engines),
            clock_ns=FakeClock(),
            cuda_sync=cuda_sync,
            reset_peak_memory=reset_peak,
            source_tree_sha256=BENCHMARK_SOURCE_TREE_SHA256,
            correctness_prerequisites_sha256="d" * 64,
            profiling=True,
            generated_tokens_override=8,
        )
        rows = [
            json.loads(line)
            for line in (output_dir / "case_rows.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        ]
        profile = json.loads(
            (output_dir / "profile.json").read_text(
                encoding="utf-8"
            )
        )

    assert engines[0].configuration["sampling"]["max_tokens"] == 8
    assert engines[0].configuration["workload"][
        "generated_tokens"
    ] == 8
    assert engines[0].workload_specs[0]["spec"][
        "generated_tokens"
    ] == 8
    assert all(row["generated_tokens"] == 8 for row in rows)
    assert all(len(row["output_token_ids"]) == 8 for row in rows)
    assert all(len(row["decode_step_ns"]) == 7 for row in rows)
    assert profile["variant"] == "short_output"
    assert profile["canonical_generated_tokens"] == 64
    assert profile["generated_tokens"] == 8


def test_decode_internal_profile_writes_independent_artifact():
    engines = []
    _, cuda_sync, reset_peak = _hooks()
    case = next(
        case
        for case in contract.build_case_matrix()
        if (
            case.policy == "recompute"
            and case.workload == "w2_long_reuse"
            and case.phase == "measured"
            and case.repetition == 0
        )
    )
    with tempfile.TemporaryDirectory() as temporary:
        output_dir = Path(temporary)
        worker.run_benchmark_case(
            case=case,
            output_dir=output_dir,
            engine_factory=_factory(engines),
            clock_ns=FakeClock(),
            cuda_sync=cuda_sync,
            reset_peak_memory=reset_peak,
            source_tree_sha256=BENCHMARK_SOURCE_TREE_SHA256,
            correctness_prerequisites_sha256="d" * 64,
            profiling=True,
            generated_tokens_override=8,
            decode_internal_profile=True,
        )
        decode_profile = json.loads(
            (output_dir / "decode_profile.json").read_text(
                encoding="utf-8"
            )
        )
        restore_profile = json.loads(
            (output_dir / "profile.json").read_text(
                encoding="utf-8"
            )
        )

    assert engines[0].configuration["profiling"] == {
        "enabled": True,
        "decode_internal": True,
    }
    assert decode_profile["schema_version"] == (
        "qwen35.tp4-decode-internal-case.v1"
    )
    assert decode_profile["resource_policy"] == (
        "shared-low-utilization"
    )
    assert decode_profile["exclusive"] is False
    assert decode_profile["generated_tokens"] == 8
    assert decode_profile["rank_inventory"] == [0, 1, 2, 3]
    assert "decode_internal" not in restore_profile


def test_nsys_replay_is_decode_profile_only_diagnostic_case():
    class Arguments:
        policy = "recompute"
        workload = "w2_long_reuse"
        phase = "nsys_replay"
        repetition = 3
        profile = True
        generated_tokens_override = 8
        decode_internal_profile = True

    case = worker._case_from_arguments(Arguments())

    assert case.case_id == (
        "w2_long_reuse__nsys_replay__r3__recompute"
    )
    assert case.phase == "nsys_replay"
    configuration = worker.build_engine_configuration(
        case.policy,
        case,
    )
    assert configuration["workload"]["name"] == "w2_long_reuse"

    for name, value in (
        ("decode_internal_profile", False),
        ("profile", False),
        ("generated_tokens_override", 7),
        ("workload", "w0_short_control"),
    ):
        invalid = Arguments()
        setattr(invalid, name, value)
        try:
            worker._case_from_arguments(invalid)
        except ValueError as error:
            assert "nsys replay" in str(error)
        else:
            raise AssertionError(
                f"invalid nsys replay accepted for {name}"
            )


def test_generated_token_override_is_profile_only_and_w2_only():
    _, cuda_sync, reset_peak = _hooks()
    cases = {
        workload: next(
            case
            for case in contract.build_case_matrix()
            if (
                case.policy == "recompute"
                and case.workload == workload
                and case.phase == "measured"
                and case.repetition == 0
            )
        )
        for workload in ("w1_medium_reuse", "w2_long_reuse")
    }
    attempts = (
        (cases["w2_long_reuse"], False, 8, "requires profiling"),
        (cases["w1_medium_reuse"], True, 8, "requires w2_long_reuse"),
        (cases["w2_long_reuse"], True, 0, "positive integer"),
        (cases["w2_long_reuse"], True, 65, "canonical generated tokens"),
    )
    for case, profiling, override, message in attempts:
        with tempfile.TemporaryDirectory() as temporary:
            try:
                worker.run_benchmark_case(
                    case=case,
                    output_dir=Path(temporary),
                    engine_factory=_factory([]),
                    clock_ns=FakeClock(),
                    cuda_sync=cuda_sync,
                    reset_peak_memory=reset_peak,
                    source_tree_sha256=(
                        BENCHMARK_SOURCE_TREE_SHA256
                    ),
                    correctness_prerequisites_sha256="d" * 64,
                    profiling=profiling,
                    generated_tokens_override=override,
                )
            except ValueError as error:
                assert message in str(error)
            else:
                raise AssertionError(
                    "invalid generated-token override was accepted"
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
        "qwen35 TP4 hybrid-prefix benchmark worker tests passed "
        f"({len(tests)} tests)"
    )


if __name__ == "__main__":
    _run()
