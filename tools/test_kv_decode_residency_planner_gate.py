"""Dependency-light tests for the KV decode residency planner gate."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import math
import os
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import torch


THIS_DIR = Path(__file__).resolve().parent
CONTRACT_PATH = THIS_DIR / "kv_decode_residency_planner_contract.py"
SPEC = importlib.util.spec_from_file_location(
    "kv_decode_residency_planner_contract_under_test",
    os.fspath(CONTRACT_PATH),
)
contract = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = contract
SPEC.loader.exec_module(contract)

VERIFIER_PATH = THIS_DIR / "verify_kv_decode_residency_planner_gate.py"
VERIFIER_SPEC = importlib.util.spec_from_file_location(
    "verify_kv_decode_residency_planner_gate_under_test",
    os.fspath(VERIFIER_PATH),
)
verifier = importlib.util.module_from_spec(VERIFIER_SPEC)
sys.modules[VERIFIER_SPEC.name] = verifier
VERIFIER_SPEC.loader.exec_module(verifier)

PROFILE_PATH = THIS_DIR / "profile_ngram_commit.py"
PROFILE_SPEC = importlib.util.spec_from_file_location(
    "profile_ngram_commit_under_test",
    os.fspath(PROFILE_PATH),
)
profile = importlib.util.module_from_spec(PROFILE_SPEC)
sys.modules[PROFILE_SPEC.name] = profile
PROFILE_SPEC.loader.exec_module(profile)

RUNNER_PATH = THIS_DIR / "run_kv_decode_residency_planner_gate_remote.py"


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_runner():
    spec = importlib.util.spec_from_file_location(
        "run_kv_decode_residency_planner_gate_remote_under_test",
        os.fspath(RUNNER_PATH),
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_json(path, value):
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _load_rows(run_dir):
    return [
        json.loads(line)
        for line in (run_dir / "case_rows.jsonl").read_text(
            encoding="utf-8",
        ).splitlines()
        if line.strip()
    ]


def _write_rows(run_dir, rows):
    (run_dir / "case_rows.jsonl").write_text(
        "".join(
            json.dumps(row, sort_keys=True) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def _complete_run_dir(root):
    run_dir = Path(root)
    (run_dir / "logs").mkdir(parents=True)
    (run_dir / "logits").mkdir(parents=True)
    source_sha256_by_policy = {
        "baseline": "a" * 64,
        "candidate": "b" * 64,
    }
    prompt_sha256_by_workload = {
        workload: hashlib.sha256(workload.encode()).hexdigest()
        for workload in contract.WORKLOADS
    }
    environment = {
        "schema_version": contract.SCHEMA_VERSION,
        "cuda_visible_devices": "0",
        "model_path": "/models/Qwen3-0.6B",
        "python_path": "/env/bin/python",
    }
    matrix = contract.build_case_matrix()
    manifest = {
        "schema_version": contract.SCHEMA_VERSION,
        "complete": True,
        "expected_case_ids": [case.case_id for case in matrix],
        "source_sha256_by_policy": source_sha256_by_policy,
        "prompt_sha256_by_workload": prompt_sha256_by_workload,
        **environment,
    }
    source_manifest = {
        "schema_version": contract.SCHEMA_VERSION,
        "source_sha256_by_policy": source_sha256_by_policy,
    }
    rows = []
    next_port = 22000
    for row_index, case in enumerate(matrix):
        logits_path = None
        logits_sha256 = None
        logits_shape = None
        if case.phase == "correctness":
            logits = torch.tensor(
                [[1.0, 2.0]],
                dtype=torch.float32,
            )
            relative_path = Path("logits") / f"{case.case_id}.pt"
            torch.save(logits, run_dir / relative_path)
            logits_path = relative_path.as_posix()
            logits_sha256 = _sha256(run_dir / relative_path)
            logits_shape = list(logits.shape)
        baseline_movement = 100
        candidate_movement = 90
        movement = (
            baseline_movement
            if case.policy == "baseline"
            else candidate_movement
        )
        rows.append({
            "row_id": f"row-{row_index:03d}",
            "case_id": case.case_id,
            "policy": case.policy,
            "workload": case.workload,
            "gpu_blocks": case.gpu_blocks,
            "blockwise_blocks": case.blockwise_blocks,
            "repetition": case.repetition,
            "phase": case.phase,
            "warmup": case.warmup,
            "source_sha256": source_sha256_by_policy[case.policy],
            "worker_pid": 1000 + row_index,
            "tinyvllm_dist_port": next_port,
            "master_port": next_port + 1,
            "cuda_visible_devices": "0",
            "model_path": environment["model_path"],
            "python_path": environment["python_path"],
            "prompt_sha256": prompt_sha256_by_workload[case.workload],
            "decoded_token_ids": [11, 12, 13],
            "decode_logits_path": logits_path,
            "decode_logits_sha256": logits_sha256,
            "decode_logits_shape": logits_shape,
            "decode_step_ms": [1.0, 1.1, 0.9],
            "peak_cuda_allocated_bytes": 1000,
            "peak_cuda_reserved_bytes": 2000,
            "peak_resident_blocks": case.gpu_blocks,
            "kv_offload": {
                "h2d_copies": movement,
                "h2d_bytes": movement * 16,
                "d2h_copies": 0,
                "d2h_bytes": 0,
                "evictions": movement,
                "copy_waits": 5,
                "prefetch_plans": 7,
                "evict_dirty": 0,
            },
            "planner": {
                field: 0
                for field in contract.PLANNER_COUNTER_FIELDS
            },
            "complete": True,
        })
        next_port += 2
    log_path = run_dir / "logs" / "workers.log"
    log_path.write_text("synthetic worker log\n", encoding="utf-8")
    worker_logs_manifest = {
        "schema_version": contract.SCHEMA_VERSION,
        "logs": [{
            "path": "logs/workers.log",
            "sha256": _sha256(log_path),
        }],
    }
    _write_json(run_dir / "manifest.json", manifest)
    _write_json(run_dir / "environment.json", environment)
    _write_json(run_dir / "source_manifest.json", source_manifest)
    _write_json(
        run_dir / "worker_logs_manifest.json",
        worker_logs_manifest,
    )
    _write_rows(run_dir, rows)
    _write_json(
        run_dir / "summary.json",
        verifier.build_raw_summary(rows),
    )
    return run_dir


def _expect_invalid(mutator, message_fragment):
    with tempfile.TemporaryDirectory() as tmp:
        run_dir = _complete_run_dir(tmp)
        mutator(run_dir)
        if message_fragment != "summary/raw disagreement":
            _write_json(
                run_dir / "summary.json",
                verifier.build_raw_summary(_load_rows(run_dir)),
            )
        try:
            verifier.verify_run(run_dir, write_report=False)
        except verifier.VerificationError as exc:
            assert message_fragment in str(exc)
        else:
            raise AssertionError(
                f"tampered run accepted: {message_fragment}"
            )


def _passing_ratio_fixture():
    return {
        "valid": True,
        "h2d_improvement": 0.06,
        "eviction_improvement": 0.06,
        "h2d_regression": 0.0,
        "eviction_regression": 0.0,
        "low_capacity_movement_improvement": 0.06,
        "multi_prompt_movement_improvement": 0.06,
        "pair_regressions_pass": True,
        "copy_waits_pass": True,
        "prefetch_plans_pass": True,
        "d2h_copies_pass": True,
        "d2h_bytes_pass": True,
        "evict_dirty_pass": True,
        "peak_resident_blocks_pass": True,
        "peak_cuda_allocated_bytes_pass": True,
        "peak_cuda_reserved_bytes_pass": True,
        "decode_latency_pass": True,
    }


def test_canonical_matrix_is_closed_and_complete():
    matrix = contract.build_case_matrix()
    assert all(
        gpu_blocks >= blockwise_blocks + 1
        for gpu_blocks, blockwise_blocks in contract.STAGING_SHAPES
    )
    assert len(matrix) == (
        len(contract.STAGING_SHAPES)
        * len(contract.WORKLOADS)
        * len(contract.POLICIES)
        * (
            contract.WARMUP_REPETITIONS
            + contract.CORRECTNESS_REPETITIONS
            + contract.MEASURED_REPETITIONS
        )
    )
    assert len({case.case_id for case in matrix}) == len(matrix)
    assert {
        (case.gpu_blocks, case.blockwise_blocks)
        for case in matrix
    } == set(contract.STAGING_SHAPES)


def test_profiler_cli_keeps_decode_logits_recording_default_off():
    original_argv = sys.argv
    try:
        sys.argv = ["profile_ngram_commit.py"]
        args = profile.parse_args()
    finally:
        sys.argv = original_argv
    assert args.record_decode_logits is False
    assert args.decode_logits_out is None


def test_baseline_profile_emits_decode_residency_gate_evidence():
    planner = {
        field: index + 1
        for index, field in enumerate(contract.PLANNER_COUNTER_FIELDS)
    }

    class FakeTokenizer:
        def encode(self, prompt):
            return [1, 2, 3]

        def decode(self, token_ids):
            return ",".join(str(token_id) for token_id in token_ids)

    class FakeRunner:
        def __init__(self):
            self.recording_enabled = None
            self.logits = [
                torch.tensor([[1.0, 2.0]], dtype=torch.float32),
                torch.tensor([[3.0, 4.0]], dtype=torch.float32),
            ]
            self.logits_index = 0
            self.summary_index = 0

        def enable_step_logits_recording(self, enabled):
            self.recording_enabled = bool(enabled)

        def last_step_logits(self):
            tensor = self.logits[self.logits_index]
            self.logits_index += 1
            return tensor.clone()

        def kv_offload_summary(self):
            resident_blocks = [1, 3, 2][
                min(self.summary_index, 2)
            ]
            self.summary_index += 1
            return {
                "resident_blocks": resident_blocks,
                **planner,
            }

    class FakeLLM:
        def __init__(self):
            self.model_runner = FakeRunner()
            self.tokenizer = FakeTokenizer()
            self.scheduler = SimpleNamespace(waiting=[])
            self.steps = [
                ([], 3),
                ([(7, [11])], -1),
                ([(7, [11, 12])], -1),
            ]

        def add_request(self, prompt, sampling_params):
            del prompt, sampling_params
            self.scheduler.waiting.append(SimpleNamespace(seq_id=7))

        def is_finished(self):
            return not self.steps

        def step(self):
            return self.steps.pop(0)

    fake_llm = FakeLLM()
    args = SimpleNamespace(
        mode="baseline-only",
        prompt=["prompt"],
        max_output_len=2,
        simulate_kv_upload_mb=0.0,
        record_decode_logits=True,
        decode_logits_out=None,
    )
    original_create_llm = profile._create_llm
    original_run_warmup = profile._run_warmup
    original_simulate_upload = profile._simulate_kv_upload
    original_sync = profile.cuda_sync_if_available
    original_reset = torch.cuda.reset_peak_memory_stats
    original_peak_allocated = torch.cuda.max_memory_allocated
    original_peak_reserved = torch.cuda.max_memory_reserved
    with tempfile.TemporaryDirectory() as tmp:
        logits_path = Path(tmp) / "decode_logits.pt"
        args.decode_logits_out = os.fspath(logits_path)
        try:
            profile._create_llm = lambda unused: (
                fake_llm,
                args.prompt,
                object(),
            )
            profile._run_warmup = lambda *unused: None
            profile._simulate_kv_upload = lambda *unused: 0.0
            profile.cuda_sync_if_available = lambda: None
            torch.cuda.reset_peak_memory_stats = lambda: None
            torch.cuda.max_memory_allocated = lambda: 1234
            torch.cuda.max_memory_reserved = lambda: 5678
            result = profile.run_baseline_only_profile(args)
        finally:
            profile._create_llm = original_create_llm
            profile._run_warmup = original_run_warmup
            profile._simulate_kv_upload = original_simulate_upload
            profile.cuda_sync_if_available = original_sync
            torch.cuda.reset_peak_memory_stats = original_reset
            torch.cuda.max_memory_allocated = original_peak_allocated
            torch.cuda.max_memory_reserved = original_peak_reserved

        assert fake_llm.model_runner.recording_enabled is True
        assert result["per_prompt"][0]["token_ids"] == [11, 12]
        assert len(result["summary"]["decode_step_ms"]) == 2
        assert result["summary"]["peak_cuda_allocated_bytes"] == 1234
        assert result["summary"]["peak_cuda_reserved_bytes"] == 5678
        assert result["summary"]["peak_resident_blocks"] == 3
        assert result["planner"] == planner
        assert result["decode_logits_path"] == os.fspath(logits_path)
        assert len(result["decode_logits_sha256"]) == 64
        assert result["decode_logits_shape"] == [2, 2]
        assert logits_path.is_file()
        assert not Path(f"{logits_path}.partial").exists()
        assert torch.equal(
            torch.load(logits_path, weights_only=True),
            torch.tensor(
                [[1.0, 2.0], [3.0, 4.0]],
                dtype=torch.float32,
            ),
        )


def test_remote_runner_command_contract_is_closed_and_phase_exact():
    runner = _load_runner()
    assert runner.SSH_TARGET == "sitian@10.232.195.203"
    assert runner.SSH_CONTROL_PATH == (
        "/tmp/ssh-sitian-10.232.195.203"
    )
    assert runner.REMOTE_PYTHON == (
        "/data00/home/sitian/sitian-workspace01/tllm/env/bin/python"
    )
    assert runner.REMOTE_MODEL.endswith("Qwen/Qwen3-0___6B")
    assert runner.CUDA_VISIBLE_DEVICES == "0"

    correctness = contract.GateCase(
        workload="multi_prompt_thrash",
        policy="candidate",
        gpu_blocks=4,
        blockwise_blocks=2,
        repetition=0,
        phase="correctness",
        warmup=False,
    )
    command = runner.build_worker_command(
        case=correctness,
        remote_source="/remote/candidate",
        case_output_dir="/remote/output/case",
        source_sha256="b" * 64,
        dist_port=23001,
        master_port=23002,
    )
    assert command["cwd"] == "/remote/candidate"
    assert command["env"] == {
        "CUDA_VISIBLE_DEVICES": "0",
        "TINYVLLM_DIST_PORT": "23001",
        "MASTER_PORT": "23002",
        "PYTHONPATH": "/remote/candidate",
        "PYTHONDONTWRITEBYTECODE": "1",
        "TINYVLLM_SOURCE_SHA256": "b" * 64,
    }
    argv = command["argv"]
    assert argv[0] == runner.REMOTE_PYTHON
    assert argv[1:4] == [
        "tools/profile_ngram_commit.py",
        "--mode",
        "baseline-only",
    ]
    assert argv[argv.index("--model") + 1] == runner.REMOTE_MODEL
    assert argv[argv.index("--kv-offload-gpu-blocks") + 1] == "4"
    assert argv[argv.index("--kv-offload-blockwise-blocks") + 1] == "2"
    assert argv[argv.index("--max-num-seqs") + 1] == "2"
    assert argv[argv.index("--max-model-len") + 1] == "2048"
    assert argv.count("--prompt") == 2
    assert "--record-decode-logits" in argv
    assert argv[argv.index("--decode-logits-out") + 1] == (
        "/remote/output/case/decode_logits.pt"
    )
    assert argv[argv.index("--out-json") + 1] == (
        "/remote/output/case/profile.json"
    )

    measured = contract.GateCase(
        workload="single_long_context",
        policy="baseline",
        gpu_blocks=2,
        blockwise_blocks=1,
        repetition=3,
        phase="measured",
        warmup=False,
    )
    measured_command = runner.build_worker_command(
        case=measured,
        remote_source="/remote/baseline",
        case_output_dir="/remote/output/measured",
        source_sha256="a" * 64,
        dist_port=23003,
        master_port=23004,
    )
    measured_argv = measured_command["argv"]
    assert measured_argv[measured_argv.index("--max-num-seqs") + 1] == "1"
    assert measured_argv[
        measured_argv.index("--max-model-len") + 1
    ] == "4096"
    assert measured_argv.count("--prompt") == 1
    assert "--record-decode-logits" not in measured_argv
    assert "--decode-logits-out" not in measured_argv

    try:
        runner.build_worker_command(
            case=measured,
            remote_source="/remote/baseline",
            case_output_dir="/remote/output/measured",
            source_sha256="a" * 64,
            dist_port=23003,
            master_port=23003,
        )
    except ValueError as exc:
        assert "distinct" in str(exc)
    else:
        raise AssertionError("equal worker ports accepted")


def test_remote_runner_ports_retry_and_source_are_safe():
    runner = _load_runner()
    allocated = iter([
        (24000, 24001),
        (24002, 24003),
        (24004, 24005),
    ])
    assert runner.allocate_unique_port_pairs(
        3,
        allocator=lambda: next(allocated),
    ) == [
        (24000, 24001),
        (24002, 24003),
        (24004, 24005),
    ]
    assert runner.is_retryable_port_collision(
        1,
        "RuntimeError: EADDRINUSE",
    )
    assert not runner.is_retryable_port_collision(0, "EADDRINUSE")
    assert not runner.is_retryable_port_collision(
        1,
        "Address already in use",
    )
    source = RUNNER_PATH.read_text(encoding="utf-8")
    for forbidden in (
        "rsync",
        "pkill",
        "killall",
        "git checkout",
        "git reset",
        "CUDA_VISIBLE_DEVICES=1",
    ):
        assert forbidden not in source
    smoke_source = (
        THIS_DIR / "smoke_blockwise_prefill_remote.sh"
    ).read_text(encoding="utf-8")
    assert '"${PYTHON_BIN}" tools/test_blockwise_attention_planning.py' in (
        smoke_source
    )
    assert '"${PYTHON_BIN}" tools/test_kv_offload.py' in smoke_source


def test_remote_runner_manifest_keeps_frozen_case_order_and_paths():
    runner = _load_runner()
    execution = runner._execution_matrix("canonical")
    assert execution[0].policy == "baseline"
    assert execution[1].policy == "candidate"
    odd_pair = next(
        index
        for index in range(0, len(execution), 2)
        if execution[index].repetition == 1
    )
    assert execution[odd_pair].policy == "candidate"
    assert execution[odd_pair + 1].policy == "baseline"

    manifest = runner.build_run_manifest(
        mode="canonical",
        source_sha256_by_policy={
            "baseline": "a" * 64,
            "candidate": "b" * 64,
        },
    )
    assert manifest["expected_case_ids"] == [
        case.case_id for case in contract.build_case_matrix()
    ]
    assert manifest["python_path"] == runner.REMOTE_PYTHON
    assert manifest["model_path"] == runner.REMOTE_MODEL
    assert manifest["cuda_visible_devices"] == "0"


def test_classification_requires_real_movement_improvement():
    ratios = _passing_ratio_fixture()
    ratios["h2d_improvement"] = 0.0
    ratios["eviction_improvement"] = 0.0
    assert contract.classify_ratios(ratios) == "NO_GO"


def test_classification_rejects_other_metric_regression():
    ratios = _passing_ratio_fixture()
    ratios["h2d_improvement"] = 0.06
    ratios["eviction_improvement"] = 0.0
    ratios["eviction_regression"] = 0.02
    assert contract.classify_ratios(ratios) == "NO_GO"


def test_complete_fixture_verifies_as_go_and_writes_outputs():
    with tempfile.TemporaryDirectory() as tmp:
        run_dir = _complete_run_dir(tmp)
        report = verifier.verify_run(run_dir, write_report=True)
        assert report["classification"] == "GO"
        assert (run_dir / "independent_verification.json").is_file()
        assert (run_dir / "report.md").is_file()


def test_missing_case_id_is_rejected():
    def mutate(run_dir):
        rows = _load_rows(run_dir)
        rows.pop()
        _write_rows(run_dir, rows)

    _expect_invalid(mutate, "missing case IDs")


def test_duplicate_row_id_is_rejected():
    def mutate(run_dir):
        rows = _load_rows(run_dir)
        rows[1]["row_id"] = rows[0]["row_id"]
        _write_rows(run_dir, rows)

    _expect_invalid(mutate, "duplicate row_id")


def test_unexpected_extra_row_is_rejected():
    def mutate(run_dir):
        rows = _load_rows(run_dir)
        extra = copy.deepcopy(rows[0])
        extra["row_id"] = "extra-row"
        extra["case_id"] = "unexpected-case"
        rows.append(extra)
        _write_rows(run_dir, rows)

    _expect_invalid(mutate, "unexpected case IDs")


def test_source_sha_mismatch_is_rejected():
    def mutate(run_dir):
        rows = _load_rows(run_dir)
        rows[0]["source_sha256"] = "f" * 64
        _write_rows(run_dir, rows)

    _expect_invalid(mutate, "source SHA mismatch")


def test_equal_baseline_candidate_ports_are_rejected():
    def mutate(run_dir):
        rows = _load_rows(run_dir)
        baseline = rows[0]
        candidate = rows[1]
        candidate["tinyvllm_dist_port"] = baseline[
            "tinyvllm_dist_port"
        ]
        _write_rows(run_dir, rows)

    _expect_invalid(mutate, "ports must be globally unique")


def test_nonzero_gpu_is_rejected():
    def mutate(run_dir):
        rows = _load_rows(run_dir)
        rows[0]["cuda_visible_devices"] = "1"
        _write_rows(run_dir, rows)

    _expect_invalid(mutate, "CUDA_VISIBLE_DEVICES")


def test_missing_decoded_token_is_rejected():
    def mutate(run_dir):
        rows = _load_rows(run_dir)
        measured = next(
            row for row in rows
            if row["phase"] == "measured"
        )
        measured["decoded_token_ids"] = []
        _write_rows(run_dir, rows)

    _expect_invalid(mutate, "decoded tokens missing")


def test_token_mismatch_is_rejected():
    def mutate(run_dir):
        rows = _load_rows(run_dir)
        candidate = next(
            row for row in rows
            if row["phase"] == "measured"
            and row["policy"] == "candidate"
        )
        candidate["decoded_token_ids"][-1] += 1
        _write_rows(run_dir, rows)

    _expect_invalid(mutate, "decoded token mismatch")


def test_missing_correctness_logits_is_rejected():
    def mutate(run_dir):
        rows = _load_rows(run_dir)
        row = next(
            item for item in rows
            if item["phase"] == "correctness"
        )
        row["decode_logits_path"] = None
        row["decode_logits_sha256"] = None
        row["decode_logits_shape"] = None
        _write_rows(run_dir, rows)

    _expect_invalid(mutate, "correctness logits missing")


def test_logits_hash_mismatch_is_rejected():
    def mutate(run_dir):
        rows = _load_rows(run_dir)
        row = next(
            item for item in rows
            if item["phase"] == "correctness"
        )
        path = run_dir / row["decode_logits_path"]
        path.write_bytes(path.read_bytes() + b"tamper")

    _expect_invalid(mutate, "logits SHA mismatch")


def test_logits_outside_tolerance_is_rejected():
    def mutate(run_dir):
        rows = _load_rows(run_dir)
        row = next(
            item for item in rows
            if item["phase"] == "correctness"
            and item["policy"] == "candidate"
        )
        path = run_dir / row["decode_logits_path"]
        logits = torch.tensor([[1.0, 3.0]], dtype=torch.float32)
        torch.save(logits, path)
        row["decode_logits_sha256"] = _sha256(path)
        row["decode_logits_shape"] = list(logits.shape)
        _write_rows(run_dir, rows)

    _expect_invalid(mutate, "logits mismatch")


def test_missing_kv_counter_is_rejected():
    def mutate(run_dir):
        rows = _load_rows(run_dir)
        del rows[0]["kv_offload"]["h2d_copies"]
        _write_rows(run_dir, rows)

    _expect_invalid(mutate, "missing KV counter")


def test_nonfinite_decode_latency_is_rejected():
    def mutate(run_dir):
        rows = _load_rows(run_dir)
        measured = next(
            row for row in rows
            if row["phase"] == "measured"
        )
        measured["decode_step_ms"] = [math.nan]
        _write_rows(run_dir, rows)

    _expect_invalid(mutate, "non-finite decode latency")


def test_summary_raw_disagreement_is_rejected():
    def mutate(run_dir):
        summary = json.loads(
            (run_dir / "summary.json").read_text(encoding="utf-8")
        )
        summary["case_count"] -= 1
        _write_json(run_dir / "summary.json", summary)

    _expect_invalid(mutate, "summary/raw disagreement")


def test_worker_log_hash_mismatch_is_rejected():
    def mutate(run_dir):
        (run_dir / "logs" / "workers.log").write_text(
            "tampered worker log\n",
            encoding="utf-8",
        )

    _expect_invalid(mutate, "worker log SHA mismatch")


def test_pair_movement_regression_is_no_go():
    with tempfile.TemporaryDirectory() as tmp:
        run_dir = _complete_run_dir(tmp)
        rows = _load_rows(run_dir)
        for row in rows:
            if (
                row["phase"] == "measured"
                and row["workload"] == "single_long_context"
                and row["gpu_blocks"] == 4
                and row["blockwise_blocks"] == 2
                and row["policy"] == "candidate"
            ):
                row["kv_offload"]["evictions"] = 102
        _write_rows(run_dir, rows)
        _write_json(
            run_dir / "summary.json",
            verifier.build_raw_summary(rows),
        )
        report = verifier.verify_run(run_dir, write_report=False)
        assert report["classification"] == "NO_GO"
        assert report["ratios"]["pair_regressions_pass"] is False


def test_multi_prompt_without_movement_win_is_no_go():
    with tempfile.TemporaryDirectory() as tmp:
        run_dir = _complete_run_dir(tmp)
        rows = _load_rows(run_dir)
        for row in rows:
            if (
                row["workload"] == "multi_prompt_thrash"
                and row["policy"] == "candidate"
            ):
                row["kv_offload"]["h2d_copies"] = 100
                row["kv_offload"]["evictions"] = 100
        _write_rows(run_dir, rows)
        _write_json(
            run_dir / "summary.json",
            verifier.build_raw_summary(rows),
        )
        report = verifier.verify_run(run_dir, write_report=False)
        assert report["classification"] == "NO_GO"


def test_low_capacity_without_movement_win_is_no_go():
    with tempfile.TemporaryDirectory() as tmp:
        run_dir = _complete_run_dir(tmp)
        rows = _load_rows(run_dir)
        for row in rows:
            if (
                row["gpu_blocks"] == 2
                and row["policy"] == "candidate"
            ):
                row["kv_offload"]["h2d_copies"] = 100
                row["kv_offload"]["evictions"] = 100
        _write_rows(run_dir, rows)
        _write_json(
            run_dir / "summary.json",
            verifier.build_raw_summary(rows),
        )
        report = verifier.verify_run(run_dir, write_report=False)
        assert report["classification"] == "NO_GO"


def main():
    test_canonical_matrix_is_closed_and_complete()
    test_profiler_cli_keeps_decode_logits_recording_default_off()
    test_baseline_profile_emits_decode_residency_gate_evidence()
    test_remote_runner_command_contract_is_closed_and_phase_exact()
    test_remote_runner_ports_retry_and_source_are_safe()
    test_remote_runner_manifest_keeps_frozen_case_order_and_paths()
    test_classification_requires_real_movement_improvement()
    test_classification_rejects_other_metric_regression()
    test_complete_fixture_verifies_as_go_and_writes_outputs()
    test_missing_case_id_is_rejected()
    test_duplicate_row_id_is_rejected()
    test_unexpected_extra_row_is_rejected()
    test_source_sha_mismatch_is_rejected()
    test_equal_baseline_candidate_ports_are_rejected()
    test_nonzero_gpu_is_rejected()
    test_missing_decoded_token_is_rejected()
    test_token_mismatch_is_rejected()
    test_missing_correctness_logits_is_rejected()
    test_logits_hash_mismatch_is_rejected()
    test_logits_outside_tolerance_is_rejected()
    test_missing_kv_counter_is_rejected()
    test_nonfinite_decode_latency_is_rejected()
    test_summary_raw_disagreement_is_rejected()
    test_worker_log_hash_mismatch_is_rejected()
    test_pair_movement_regression_is_no_go()
    test_multi_prompt_without_movement_win_is_no_go()
    test_low_capacity_without_movement_win_is_no_go()
    print("KV decode residency planner gate tests passed")


if __name__ == "__main__":
    main()
