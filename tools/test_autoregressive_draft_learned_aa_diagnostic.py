from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
DIAGNOSTIC_PATH = (
    ROOT / "tools" / "autoregressive_draft_learned_aa_diagnostic.py"
)
PERFORMANCE_GATE_TEST_PATH = (
    ROOT / "tools" / "test_autoregressive_draft_performance_gate.py"
)
VERIFIER_PATH = (
    ROOT
    / "tools"
    / "verify_autoregressive_draft_learned_aa_diagnostic.py"
)
EPOCHS = ("learned_a", "learned_b")


def _load_module(name: str, path: Path):
    assert path.exists(), f"missing module: {path}"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _diagnostic():
    return _load_module(
        "autoregressive_draft_learned_aa_diagnostic_test_module",
        DIAGNOSTIC_PATH,
    )


def _performance_gate_test_helpers():
    return _load_module(
        "autoregressive_draft_performance_gate_test_helpers",
        PERFORMANCE_GATE_TEST_PATH,
    )


def _verifier():
    diagnostic = _diagnostic()
    sys.modules[
        "autoregressive_draft_learned_aa_diagnostic"
    ] = diagnostic
    return _load_module(
        "verify_autoregressive_draft_learned_aa_test_module",
        VERIFIER_PATH,
    )


def _measured_worker() -> dict:
    worker = _performance_gate_test_helpers()._diagnostic_worker("learned")
    cursor = 1_000_000_000
    for run in worker["warmup_runs"] + worker["measured_runs"]:
        run["campaign_interval"] = {
            "started_at_unix_ns": cursor,
            "finished_at_unix_ns": cursor + 1_000_000_000,
        }
        cursor += 2_000_000_000
    return worker


def _prime_worker() -> dict:
    worker = _measured_worker()
    worker["measured_runs"] = worker["measured_runs"][:1]
    return worker


def _shift_intervals(worker: dict, delta_ns: int) -> None:
    for run in worker["warmup_runs"] + worker["measured_runs"]:
        interval = run["campaign_interval"]
        interval["started_at_unix_ns"] += delta_ns
        interval["finished_at_unix_ns"] += delta_ns


@pytest.mark.parametrize("artifact_identity", EPOCHS)
def test_measured_worker_requires_learned_policy(artifact_identity):
    worker = _measured_worker()
    worker["policy"] = "target"

    with pytest.raises(ValueError, match="policy must be learned"):
        _diagnostic().validate_measured_worker(
            worker,
            artifact_identity=artifact_identity,
        )


def test_prime_worker_requires_two_warmups_and_one_repeat():
    worker = _prime_worker()
    worker["measured_runs"].append(
        copy.deepcopy(worker["measured_runs"][0])
    )

    with pytest.raises(ValueError, match="prime worker"):
        _diagnostic().validate_prime_worker(
            worker,
            artifact_identity="learned_a",
        )


def test_measured_worker_requires_batch_four():
    worker = _measured_worker()
    worker["batch_size"] = 1

    with pytest.raises(ValueError, match="batch size must be four"):
        _diagnostic().validate_measured_worker(
            worker,
            artifact_identity="learned_a",
        )


def test_measured_worker_requires_valid_artifact_identity():
    with pytest.raises(
        ValueError,
        match="invalid learned A/A artifact identity",
    ):
        _diagnostic().validate_measured_worker(
            _measured_worker(),
            artifact_identity="learned",
        )


def test_workload_identity_requires_exact_proposal_kv_capacity():
    learned_a = _measured_worker()
    learned_b = copy.deepcopy(learned_a)
    learned_b["proposal_slot_capacity"] += 1

    with pytest.raises(ValueError, match="Proposal-KV"):
        _diagnostic().validate_workload_identity(
            learned_a,
            learned_b,
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda worker: worker.__setitem__(
                "target_checkpoint_identifier",
                "different-target",
            ),
            "target checkpoint",
        ),
        (
            lambda worker: worker.__setitem__(
                "draft_checkpoint_identifier",
                "different-draft",
            ),
            "draft checkpoint",
        ),
        (
            lambda worker: worker["prompt_rows"][0]["token_ids"].append(99),
            "prompt rows",
        ),
        (
            lambda worker: worker.__setitem__("tensor_parallel_size", 1),
            "tensor parallel",
        ),
    ],
)
def test_workload_identity_rejects_worker_drift(mutation, message):
    learned_a = _measured_worker()
    learned_b = copy.deepcopy(learned_a)
    mutation(learned_b)

    with pytest.raises(ValueError, match=message):
        _diagnostic().validate_workload_identity(
            learned_a,
            learned_b,
        )


def test_workload_identity_records_fixed_greedy_contract():
    identity = _diagnostic().validate_workload_identity(
        _measured_worker(),
        _measured_worker(),
    )

    assert identity["batch_size"] == 4
    assert identity["temperature"] == 0.0
    assert identity["max_proposal_tokens"] == 4
    assert identity["tensor_parallel_size"] == 4
    assert identity["proposal_slot_capacity"] == 4 * (256 + 16 + 4)


def _set_metrics(
    worker: dict,
    *,
    e2e_values: list[float],
    tpot_values: list[float],
    proposal_forward_values: list[float],
) -> None:
    for repeat, run in enumerate(worker["measured_runs"]):
        for request in run["timing"]["per_request"]:
            request["completion_latency_s"] = e2e_values[repeat]
            request["tpot_s"] = tpot_values[repeat]
        executor_timing = run["runtime"]["draft_executor_timing"]
        for row in executor_timing["ranks"]:
            row["proposal_forward"] = (
                proposal_forward_values[repeat]
                - (3 - row["rank"]) * 0.1
            )
        executor_timing["max_rank_ms"]["proposal_forward"] = (
            proposal_forward_values[repeat]
        )
        detail = run["runtime"]["draft_executor_proposal_detail"]
        detail["residual_ms"] = (
            proposal_forward_values[repeat]
            - detail["detail_sum_ms"]
        )


def _gpu_sample(sampled_at_unix_ns: int, gpu_index: int) -> dict:
    return {
        "sampled_at_unix_ns": sampled_at_unix_ns,
        "nvidia_timestamp": "2026/08/15 23:47:03.101",
        "gpu_index": gpu_index,
        "uuid": f"GPU-{gpu_index:032d}",
        "pstate": "P0",
        "sm_clock_mhz": 1410,
        "memory_clock_mhz": 1512,
        "power_w": 70.0,
        "temperature_c": 41,
        "gpu_utilization_percent": 93,
        "memory_utilization_percent": 12,
        "memory_used_mib": 72_455,
        "throttle_reasons_active": 0,
    }


def _gpu_samples(worker: dict) -> list[dict]:
    samples = []
    for run in worker["measured_runs"]:
        interval = run["campaign_interval"]
        start = interval["started_at_unix_ns"]
        finish = interval["finished_at_unix_ns"]
        timestamps = (
            start,
            start + 250_000_000,
            start + 500_000_000,
            start + 750_000_000,
            finish,
        )
        for gpu_index in (3, 4, 6, 7):
            samples.extend(
                _gpu_sample(timestamp, gpu_index)
                for timestamp in timestamps
            )
    return samples


def _host_sample(unix_ns: int, monotonic_ns: int, offset: int) -> dict:
    return {
        "schema_version": 1,
        "sampled_at_unix_ns": unix_ns,
        "sampled_at_monotonic_ns": monotonic_ns,
        "cpu_user_ticks": 100 + offset,
        "cpu_nice_ticks": 2,
        "cpu_system_ticks": 20 + offset,
        "cpu_idle_ticks": 300 + offset,
        "cpu_iowait_ticks": 5 + offset,
        "cpu_irq_ticks": 1,
        "cpu_softirq_ticks": 2,
        "cpu_steal_ticks": 0,
        "procs_running": 2 + offset,
        "procs_blocked": 1,
        "context_switches_total": 1000 + 10 * offset,
        "processes_forked_total": 50 + offset,
        "loadavg_1m": 1.0 + offset,
        "loadavg_5m": 2.0,
        "loadavg_15m": 3.0,
        "major_faults_total": 10 + offset,
        "page_in_kib_total": 100 + 4 * offset,
        "page_out_kib_total": 200 + 8 * offset,
        "swap_in_kib_total": 0,
        "swap_out_kib_total": 0,
        "memory_available_kib": 100_000 - offset,
        "memory_cached_kib": 5_000,
        "memory_dirty_kib": 20 + offset,
        "memory_writeback_kib": 2 + offset,
        "cpu_psi_some_total_us": 100 + 10 * offset,
        "cpu_psi_full_total_us": None,
        "io_psi_some_total_us": 200 + 20 * offset,
        "io_psi_full_total_us": 20 + 2 * offset,
        "memory_psi_some_total_us": 300 + 30 * offset,
        "memory_psi_full_total_us": 30 + 3 * offset,
    }


def _host_samples(worker: dict) -> list[dict]:
    samples = []
    offset = 0
    for run in worker["measured_runs"]:
        interval = run["campaign_interval"]
        start = interval["started_at_unix_ns"]
        finish = interval["finished_at_unix_ns"]
        for timestamp in (
            start - 200_000_000,
            start,
            start + 200_000_000,
            start + 400_000_000,
            start + 600_000_000,
            start + 800_000_000,
            finish,
            finish + 200_000_000,
        ):
            samples.append(_host_sample(timestamp, timestamp, offset))
            offset += 1
    return samples


def _valid_inputs(
    *,
    learned_a_metrics=(10.0, 5.0, 100.0),
    learned_b_metrics=(10.0, 5.0, 100.0),
) -> dict:
    learned_a = _measured_worker()
    learned_b = _measured_worker()
    _shift_intervals(learned_b, 100_000_000_000)
    learned_a_prime = _prime_worker()
    learned_b_prime = _prime_worker()
    _shift_intervals(learned_b_prime, 100_000_000_000)
    _set_metrics(
        learned_a,
        e2e_values=[learned_a_metrics[0]] * 8,
        tpot_values=[learned_a_metrics[1]] * 8,
        proposal_forward_values=[learned_a_metrics[2]] * 8,
    )
    _set_metrics(
        learned_b,
        e2e_values=[learned_b_metrics[0]] * 8,
        tpot_values=[learned_b_metrics[1]] * 8,
        proposal_forward_values=[learned_b_metrics[2]] * 8,
    )
    return {
        "prime_workers": {
            "learned_a": learned_a_prime,
            "learned_b": learned_b_prime,
        },
        "workers": {
            "learned_a": learned_a,
            "learned_b": learned_b,
        },
        "gpu_samples": {
            "learned_a": _gpu_samples(learned_a),
            "learned_b": _gpu_samples(learned_b),
        },
        "host_samples": {
            "learned_a": _host_samples(learned_a),
            "learned_b": _host_samples(learned_b),
        },
        "epoch_order": ["learned_a", "learned_b"],
        "prime_each_epoch": True,
        "bundle_role": "discovery",
        "input_files": {
            "placeholder": {
                "path": "placeholder",
                "sha256": "a" * 64,
            },
        },
        "source_files": {
            "tools/source.py": "b" * 64,
        },
    }


def test_build_artifact_requires_exact_output_parity_per_repeat():
    inputs = _valid_inputs()
    inputs["workers"]["learned_b"]["measured_runs"][3][
        "outputs"
    ][0][0] += 1

    with pytest.raises(
        ValueError,
        match="exact parity failed at repeat 3",
    ):
        _diagnostic().build_learned_aa_artifact(**inputs)


def test_build_artifact_requires_five_samples_per_gpu_per_repeat():
    inputs = _valid_inputs()
    worker = inputs["workers"]["learned_b"]
    interval = worker["measured_runs"][5]["campaign_interval"]
    matching = [
        row
        for row in inputs["gpu_samples"]["learned_b"]
        if row["gpu_index"] == 7
        and interval["started_at_unix_ns"]
        <= row["sampled_at_unix_ns"]
        <= interval["finished_at_unix_ns"]
    ]
    for row in matching[4:]:
        inputs["gpu_samples"]["learned_b"].remove(row)

    with pytest.raises(ValueError, match="GPU telemetry coverage"):
        _diagnostic().build_learned_aa_artifact(**inputs)


def test_build_artifact_requires_host_repeat_local_gap_below_limit():
    inputs = _valid_inputs()
    worker = inputs["workers"]["learned_a"]
    interval = worker["measured_runs"][2]["campaign_interval"]
    start = interval["started_at_unix_ns"]
    rows = inputs["host_samples"]["learned_a"]
    rows[:] = [
        row
        for row in rows
        if not (
            start + 200_000_000
            <= row["sampled_at_unix_ns"]
            <= start + 600_000_000
        )
    ]

    with pytest.raises(ValueError, match="sample gap"):
        _diagnostic().build_learned_aa_artifact(**inputs)


@pytest.mark.parametrize(
    (
        "learned_a_metrics",
        "learned_b_metrics",
        "expected",
    ),
    [
        (
            (10.5, 5.2, 102.0),
            (10.0, 5.0, 100.0),
            "LEARNED_AA_STABLE",
        ),
        (
            (12.0, 6.0, 120.0),
            (10.0, 5.0, 100.0),
            "LEARNED_AA_PROCESS_BOUNDARY_EFFECT",
        ),
        (
            (12.0, 4.0, 120.0),
            (10.0, 5.0, 100.0),
            "LEARNED_AA_INCONCLUSIVE",
        ),
        (
            (12.0, 6.0, 100.0),
            (10.0, 5.0, 100.0),
            "LEARNED_AA_INCONCLUSIVE",
        ),
    ],
)
def test_classification_thresholds_and_direction(
    learned_a_metrics,
    learned_b_metrics,
    expected,
):
    artifact = _diagnostic().build_learned_aa_artifact(
        **_valid_inputs(
            learned_a_metrics=learned_a_metrics,
            learned_b_metrics=learned_b_metrics,
        )
    )

    assert artifact["classification"] == expected
    assert (
        artifact["claim_state"]["process_boundary_effect_established"]
        is False
    )


@pytest.mark.parametrize(
    ("learned_a_e2e", "expected"),
    [
        (10.99999, "LEARNED_AA_STABLE"),
        (11.0, "LEARNED_AA_PROCESS_BOUNDARY_EFFECT"),
    ],
)
def test_classification_uses_exact_ten_percent_boundary(
    learned_a_e2e,
    expected,
):
    artifact = _diagnostic().build_learned_aa_artifact(
        **_valid_inputs(
            learned_a_metrics=(
                learned_a_e2e,
                learned_a_e2e / 2,
                learned_a_e2e * 10,
            ),
            learned_b_metrics=(10.0, 5.0, 100.0),
        )
    )

    assert artifact["classification"] == expected


@pytest.mark.parametrize(
    ("maximum", "expected_stable"),
    [
        (11.25, True),
        (11.2501, False),
    ],
)
def test_stationarity_uses_exact_range_over_median_boundary(
    maximum,
    expected_stable,
):
    result = _diagnostic()._stationarity(
        "e2e_s",
        [8.75, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, maximum],
    )

    assert result["range_over_median"] == pytest.approx(
        (maximum - 8.75) / 10.0
    )
    assert result["half_drift_fraction"] == 0.0
    assert result["stable"] is expected_stable


@pytest.mark.parametrize(
    ("first_half", "second_half", "expected_stable"),
    [
        (9.0, 11.0, True),
        (8.999, 11.001, False),
    ],
)
def test_stationarity_uses_exact_half_drift_boundary(
    first_half,
    second_half,
    expected_stable,
):
    result = _diagnostic()._stationarity(
        "e2e_s",
        [first_half] * 4 + [second_half] * 4,
    )

    assert result["range_over_median"] == pytest.approx(
        (second_half - first_half) / 10.0
    )
    assert result["half_drift_fraction"] == pytest.approx(
        (second_half - first_half) / 10.0
    )
    assert result["stable"] is expected_stable


def test_nonstationary_primary_metric_is_inconclusive():
    inputs = _valid_inputs()
    worker = inputs["workers"]["learned_a"]
    _set_metrics(
        worker,
        e2e_values=[10.0] * 4 + [14.0] * 4,
        tpot_values=[5.0] * 8,
        proposal_forward_values=[100.0] * 8,
    )

    artifact = _diagnostic().build_learned_aa_artifact(**inputs)

    assert artifact["classification"] == "LEARNED_AA_INCONCLUSIVE"
    assert any(
        "stationarity" in reason
        for reason in artifact["classification_reasons"]
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _gpu_csv(samples: list[dict]) -> str:
    return "".join(
        (
            f'{row["sampled_at_unix_ns"]}, '
            f'{row["nvidia_timestamp"]}, '
            f'{row["gpu_index"]}, '
            f'{row["uuid"]}, '
            f'{row["pstate"]}, '
            f'{row["sm_clock_mhz"]}, '
            f'{row["memory_clock_mhz"]}, '
            f'{row["power_w"]}, '
            f'{row["temperature_c"]}, '
            f'{row["gpu_utilization_percent"]}, '
            f'{row["memory_utilization_percent"]}, '
            f'{row["memory_used_mib"]}, '
            f'{row["throttle_reasons_active"]:#x}\n'
        )
        for row in samples
    )


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_raw_bundle(tmp_path: Path):
    diagnostic = _diagnostic()
    bundle = tmp_path / "bundle"
    source_root = tmp_path / "source"
    inputs = _valid_inputs()
    paths = {
        "learned_a_prime_worker": (
            bundle / "prime-workers/learned-a-prime-b4.json"
        ),
        "learned_b_prime_worker": (
            bundle / "prime-workers/learned-b-prime-b4.json"
        ),
        "learned_a_worker": bundle / "workers/learned-a-b4.json",
        "learned_b_worker": bundle / "workers/learned-b-b4.json",
        "learned_a_gpu_csv": (
            bundle / "telemetry/learned-a-gpu.csv"
        ),
        "learned_b_gpu_csv": (
            bundle / "telemetry/learned-b-gpu.csv"
        ),
        "learned_a_host_jsonl": (
            bundle / "host-semantic/learned-a-host.jsonl"
        ),
        "learned_b_host_jsonl": (
            bundle / "host-semantic/learned-b-host.jsonl"
        ),
        "epoch_order": bundle / "epoch-order.txt",
        "prime_each_epoch": bundle / "prime-each-epoch.txt",
    }
    _write_json(
        paths["learned_a_prime_worker"],
        inputs["prime_workers"]["learned_a"],
    )
    _write_json(
        paths["learned_b_prime_worker"],
        inputs["prime_workers"]["learned_b"],
    )
    _write_json(
        paths["learned_a_worker"],
        inputs["workers"]["learned_a"],
    )
    _write_json(
        paths["learned_b_worker"],
        inputs["workers"]["learned_b"],
    )
    for epoch in EPOCHS:
        slug = epoch.replace("_", "-")
        paths[f"{epoch}_gpu_csv"].parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        paths[f"{epoch}_gpu_csv"].write_text(
            _gpu_csv(inputs["gpu_samples"][epoch]),
            encoding="utf-8",
        )
        paths[f"{epoch}_host_jsonl"].parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        paths[f"{epoch}_host_jsonl"].write_text(
            "".join(
                json.dumps(row, sort_keys=True) + "\n"
                for row in inputs["host_samples"][epoch]
            ),
            encoding="utf-8",
        )
        assert slug in paths[f"{epoch}_gpu_csv"].name
    paths["epoch_order"].write_text(
        "learned_a,learned_b\n",
        encoding="utf-8",
    )
    paths["prime_each_epoch"].write_text("1\n", encoding="utf-8")
    for relative_path in diagnostic.SOURCE_FILE_PATHS:
        source_path = source_root / relative_path
        source_path.parent.mkdir(parents=True, exist_ok=True)
        source_path.write_text(
            f"source fixture for {relative_path}\n",
            encoding="utf-8",
        )
    return bundle, source_root, paths


def _build_bound_campaign(tmp_path: Path):
    diagnostic = _diagnostic()
    bundle, source_root, paths = _write_raw_bundle(tmp_path)
    artifact_path = bundle / "learned-aa.json"
    status = diagnostic.main([
        "--learned-a-prime-worker",
        str(paths["learned_a_prime_worker"]),
        "--learned-b-prime-worker",
        str(paths["learned_b_prime_worker"]),
        "--learned-a-worker",
        str(paths["learned_a_worker"]),
        "--learned-b-worker",
        str(paths["learned_b_worker"]),
        "--learned-a-gpu-csv",
        str(paths["learned_a_gpu_csv"]),
        "--learned-b-gpu-csv",
        str(paths["learned_b_gpu_csv"]),
        "--learned-a-host-jsonl",
        str(paths["learned_a_host_jsonl"]),
        "--learned-b-host-jsonl",
        str(paths["learned_b_host_jsonl"]),
        "--epoch-order-file",
        str(paths["epoch_order"]),
        "--prime-each-epoch-file",
        str(paths["prime_each_epoch"]),
        "--bundle-role",
        "discovery",
        "--repo-root",
        str(source_root),
        "--out",
        str(artifact_path),
    ])
    assert status == 0
    return artifact_path, source_root, paths


def test_main_builds_ten_input_source_bound_artifact(tmp_path):
    artifact_path, _, _ = _build_bound_campaign(tmp_path)
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))

    assert artifact["status"] == "PASS"
    assert len(artifact["input_files"]) == 10
    assert set(artifact["input_files"]) == {
        "learned_a_prime_worker",
        "learned_b_prime_worker",
        "learned_a_worker",
        "learned_b_worker",
        "learned_a_gpu_csv",
        "learned_b_gpu_csv",
        "learned_a_host_jsonl",
        "learned_b_host_jsonl",
        "epoch_order",
        "prime_each_epoch",
    }
    assert (
        artifact["input_files"]["learned_a_worker"]["sha256"]
        != artifact["input_files"]["learned_b_worker"]["sha256"]
    )
    assert (
        artifact["input_files"]["learned_a_gpu_csv"]["sha256"]
        != artifact["input_files"]["learned_b_gpu_csv"]["sha256"]
    )
    assert (
        artifact["input_files"]["learned_a_host_jsonl"]["sha256"]
        != artifact["input_files"]["learned_b_host_jsonl"]["sha256"]
    )


def test_verifier_recomputes_from_hash_bound_raw_inputs(tmp_path):
    artifact_path, source_root, _ = _build_bound_campaign(tmp_path)

    receipt = _verifier().verify_learned_aa_diagnostic(
        artifact_path,
        source_root,
    )

    assert receipt["status"] == "PASS"
    assert receipt["input_files_verified"] == 10
    assert receipt["source_files_verified"] >= 6
    assert receipt["process_boundary_effect_established"] is False


@pytest.mark.parametrize("unsafe_path", ["/tmp/worker.json", "../worker.json"])
def test_verifier_rejects_unsafe_input_paths(tmp_path, unsafe_path):
    artifact_path, source_root, _ = _build_bound_campaign(tmp_path)
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    artifact["input_files"]["learned_a_worker"]["path"] = unsafe_path
    _write_json(artifact_path, artifact)

    with pytest.raises(ValueError, match="relative path"):
        _verifier().verify_learned_aa_diagnostic(
            artifact_path,
            source_root,
        )


def test_verifier_rejects_raw_input_tampering(tmp_path):
    artifact_path, source_root, paths = _build_bound_campaign(tmp_path)
    paths["learned_a_worker"].write_text(
        paths["learned_a_worker"].read_text(encoding="utf-8") + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="input hash mismatch"):
        _verifier().verify_learned_aa_diagnostic(
            artifact_path,
            source_root,
        )


def test_verifier_rejects_source_tampering(tmp_path):
    artifact_path, source_root, _ = _build_bound_campaign(tmp_path)
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    relative_path = next(iter(artifact["source_files"]))
    source_path = source_root / relative_path
    source_path.write_text(
        source_path.read_text(encoding="utf-8") + "tampered\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="source hash mismatch"):
        _verifier().verify_learned_aa_diagnostic(
            artifact_path,
            source_root,
        )


def test_verifier_rejects_canonical_artifact_tampering(tmp_path):
    artifact_path, source_root, _ = _build_bound_campaign(tmp_path)
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    artifact["classification"] = "LEARNED_AA_INCONCLUSIVE"
    _write_json(artifact_path, artifact)

    with pytest.raises(ValueError, match="recomputation mismatch"):
        _verifier().verify_learned_aa_diagnostic(
            artifact_path,
            source_root,
        )
