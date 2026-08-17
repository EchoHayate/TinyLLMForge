from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import statistics
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
DIAGNOSTIC_PATH = (
    ROOT / "tools" / "autoregressive_draft_paired_stability_diagnostic.py"
)
PERFORMANCE_GATE_TEST_PATH = (
    ROOT / "tools" / "test_autoregressive_draft_performance_gate.py"
)
LEARNED_AA_TEST_PATH = (
    ROOT / "tools" / "test_autoregressive_draft_learned_aa_diagnostic.py"
)
VERIFIER_PATH = (
    ROOT
    / "tools"
    / "verify_autoregressive_draft_paired_stability_diagnostic.py"
)
RUNNER_PATH = (
    ROOT / "tools" / "run_autoregressive_draft_paired_stability_remote.sh"
)


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _performance_gate_test_helpers():
    return load_module(
        PERFORMANCE_GATE_TEST_PATH,
        "paired_performance_gate_test_helpers",
    )


def _learned_aa_test_helpers():
    return load_module(
        LEARNED_AA_TEST_PATH,
        "paired_learned_aa_test_helpers",
    )


def _worker(*, measured_runs: int) -> dict:
    worker = _performance_gate_test_helpers()._diagnostic_worker("learned")
    worker["measured_runs"] = worker["measured_runs"][:measured_runs]
    cursor = 1_000_000_000
    for run in worker["warmup_runs"] + worker["measured_runs"]:
        run["campaign_interval"] = {
            "started_at_unix_ns": cursor,
            "finished_at_unix_ns": cursor + 1_000_000_000,
        }
        cursor += 2_000_000_000
    return worker


@pytest.fixture
def valid_prime() -> dict:
    return _worker(measured_runs=1)


@pytest.fixture
def valid_worker() -> dict:
    return _worker(measured_runs=5)


@pytest.fixture
def identity():
    diagnostic = load_module(DIAGNOSTIC_PATH, "paired_identity_fixture")
    return diagnostic.expected_epoch_identities()[0]


@pytest.fixture
def valid_raw_epoch(valid_prime, valid_worker, identity) -> dict:
    helpers = _learned_aa_test_helpers()
    proposals = [
        run["runtime"]["proposed_tokens"]
        for run in valid_worker["measured_runs"]
    ]
    accepted = [
        run["runtime"]["accepted_draft_tokens"]
        for run in valid_worker["measured_runs"]
    ]
    outputs = [
        copy.deepcopy(run["outputs"])
        for run in valid_worker["measured_runs"]
    ]
    return {
        "prime_worker": copy.deepcopy(valid_prime),
        "worker": copy.deepcopy(valid_worker),
        "gpu_rows": helpers._gpu_samples(valid_worker),
        "host_rows": helpers._host_samples(valid_worker),
        "gpu_invariants": {
            "telemetry_available": True,
            "uuid_by_index": {
                str(index): f"GPU-{index:032d}"
                for index in (3, 4, 6, 7)
            },
            "undeclared_gpu_indices": [],
            "xid_events": [],
            "reset_events": [],
            "throttle_valid": True,
            "clocks_pstate_valid": True,
        },
        "process_before": {
            "protected_gpu7_pid_present": True,
            "unrelated_process_inventory": [101, 202],
        },
        "process_after": {
            "protected_gpu7_pid_present": True,
            "runner_owned_pids_remaining": [],
            "unrelated_process_inventory": [101, 202],
        },
        "exact_parity": True,
        "accepted_prefix_semantics": True,
        "proposal_counts": proposals,
        "proposal_lengths": [4] * len(proposals),
        "accepted_token_counts": accepted,
        "total_verified_tokens": sum(proposals),
        "output_token_ids": outputs,
        "prime_excluded_from_measured_statistics": True,
        "source_paths": {
            "prime_worker": f"{identity.key}/prime-worker.json",
            "worker": f"{identity.key}/worker.json",
            "gpu_rows": f"{identity.key}/gpu.csv",
            "host_rows": f"{identity.key}/host.jsonl",
            "gpu_invariants": f"{identity.key}/gpu-invariants.json",
            "process_before": f"{identity.key}/process-before.json",
            "process_after": f"{identity.key}/process-after.json",
        },
    }


def _epoch_envelope(diagnostic, identity, worker: dict) -> dict:
    return {
        "identity": {
            "block_index": identity.block_index,
            "order": identity.order,
            "label": identity.label,
            "position": identity.position,
            "epoch_index": identity.epoch_index,
        },
        "temperature": 0.0,
        "max_proposal_tokens": 4,
        "gpu_indices": [3, 4, 6, 7],
        "request_order": [0, 1, 2, 3],
        "accepted_prefix_semantics": True,
        "proposal_kv_capacity": {
            "allocator": "direct",
            "slots": worker["proposal_slot_capacity"],
        },
        "worker": copy.deepcopy(worker),
    }


def _valid_epochs(diagnostic, worker: dict) -> dict[str, dict]:
    return {
        identity.key: _epoch_envelope(diagnostic, identity, worker)
        for identity in diagnostic.expected_epoch_identities()
    }


def _mutate_prompt_rows(epoch: dict) -> None:
    row = epoch["worker"]["prompt_rows"][0]
    row["token_ids"][0] += 1
    row["sha256"] = hashlib.sha256(
        json.dumps(
            row["token_ids"],
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _mutate_proposal_count(worker: dict) -> None:
    runtime = worker["measured_runs"][0]["runtime"]
    runtime["proposed_tokens"] += 4
    runtime["acceptance_rate"] = (
        runtime["accepted_draft_tokens"] / runtime["proposed_tokens"]
    )


def test_fixed_schedule_and_digest():
    diagnostic = load_module(DIAGNOSTIC_PATH, "paired_schedule")
    assert diagnostic.BLOCK_SCHEDULE == (
        ("A", "B"),
        ("B", "A"),
        ("B", "A"),
        ("A", "B"),
    )
    assert diagnostic.SCHEDULE_TEXT == "AB\nBA\nBA\nAB\n"
    assert diagnostic.SCHEDULE_SHA256 == hashlib.sha256(
        diagnostic.SCHEDULE_TEXT.encode("utf-8")
    ).hexdigest()


def test_expected_epoch_identities_are_unique_and_ordered():
    diagnostic = load_module(DIAGNOSTIC_PATH, "paired_identities")
    identities = diagnostic.expected_epoch_identities()
    assert len(identities) == 8
    assert len({identity.key for identity in identities}) == 8
    assert [
        (
            identity.block_index,
            identity.order,
            identity.label,
            identity.position,
            identity.epoch_index,
        )
        for identity in identities
    ] == [
        (0, "AB", "A", "first", 0),
        (0, "AB", "B", "second", 1),
        (1, "BA", "B", "first", 2),
        (1, "BA", "A", "second", 3),
        (2, "BA", "B", "first", 4),
        (2, "BA", "A", "second", 5),
        (3, "AB", "A", "first", 6),
        (3, "AB", "B", "second", 7),
    ]


def test_prime_is_two_warmups_and_one_measured_run(
    valid_prime,
    identity,
):
    diagnostic = load_module(DIAGNOSTIC_PATH, "paired_prime")
    invalid = copy.deepcopy(valid_prime)
    invalid["measured_runs"].append(
        copy.deepcopy(invalid["measured_runs"][0])
    )
    with pytest.raises(ValueError, match="prime worker"):
        diagnostic.validate_prime_worker(invalid, identity=identity)


def test_measured_worker_is_two_warmups_and_five_repeats(
    valid_worker,
    identity,
):
    diagnostic = load_module(DIAGNOSTIC_PATH, "paired_worker_count")
    invalid = copy.deepcopy(valid_worker)
    invalid["measured_runs"].pop()
    with pytest.raises(ValueError, match="five measured repeats"):
        diagnostic.validate_measured_worker(invalid, identity=identity)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("policy", "target", "policy must be learned"),
        ("batch_size", 1, "batch size must be four"),
    ],
)
def test_worker_fixed_runtime_fields(
    valid_worker,
    identity,
    field,
    value,
    message,
):
    diagnostic = load_module(DIAGNOSTIC_PATH, "paired_worker_fields")
    invalid = copy.deepcopy(valid_worker)
    invalid[field] = value
    with pytest.raises(ValueError, match=message):
        diagnostic.validate_measured_worker(invalid, identity=identity)


def test_epoch_identity_rejects_wrong_schedule_position(
    valid_worker,
    identity,
):
    diagnostic = load_module(DIAGNOSTIC_PATH, "paired_wrong_identity")
    wrong = diagnostic.EpochIdentity(
        block_index=identity.block_index,
        order=identity.order,
        label=identity.label,
        position="second",
        epoch_index=identity.epoch_index,
    )
    with pytest.raises(ValueError, match="epoch identity"):
        diagnostic.validate_measured_worker(valid_worker, identity=wrong)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda epoch: epoch["worker"].__setitem__(
                "target_checkpoint_identifier",
                "different-target",
            ),
            "target checkpoint",
        ),
        (
            lambda epoch: epoch["worker"].__setitem__(
                "draft_checkpoint_identifier",
                "different-draft",
            ),
            "draft checkpoint",
        ),
        (
            lambda epoch: epoch["worker"].__setitem__(
                "tokenizer_identifier",
                "different-tokenizer",
            ),
            "tokenizer",
        ),
        (
            _mutate_prompt_rows,
            "prompt rows",
        ),
        (
            lambda epoch: epoch["worker"].__setitem__(
                "tensor_parallel_size",
                1,
            ),
            "tensor parallel",
        ),
        (
            lambda epoch: epoch.__setitem__(
                "request_order",
                [1, 0, 2, 3],
            ),
            "request order",
        ),
        (
            lambda epoch: epoch.__setitem__("temperature", 0.5),
            "temperature",
        ),
        (
            lambda epoch: epoch.__setitem__(
                "accepted_prefix_semantics",
                False,
            ),
            "accepted-prefix",
        ),
    ],
)
def test_epoch_workload_identity_rejects_drift(
    valid_worker,
    mutation,
    message,
):
    diagnostic = load_module(DIAGNOSTIC_PATH, "paired_workload_drift")
    epochs = _valid_epochs(diagnostic, valid_worker)
    mutation(epochs[diagnostic.expected_epoch_identities()[-1].key])
    with pytest.raises(ValueError, match=message):
        diagnostic.validate_epoch_workload_identity(epochs)


def test_epoch_workload_identity_requires_exact_proposal_kv_capacity(
    valid_worker,
):
    diagnostic = load_module(DIAGNOSTIC_PATH, "paired_workload_capacity")
    epochs = _valid_epochs(diagnostic, valid_worker)
    epochs[diagnostic.expected_epoch_identities()[-1].key][
        "proposal_kv_capacity"
    ]["slots"] += 1
    with pytest.raises(ValueError, match="Proposal-KV capacity"):
        diagnostic.validate_epoch_workload_identity(epochs)


def test_epoch_workload_identity_records_fixed_contract(valid_worker):
    diagnostic = load_module(DIAGNOSTIC_PATH, "paired_workload")
    epochs = _valid_epochs(diagnostic, valid_worker)
    workload = diagnostic.validate_epoch_workload_identity(epochs)
    assert workload["batch_size"] == 4
    assert workload["temperature"] == 0.0
    assert workload["max_proposal_tokens"] == 4
    assert workload["tensor_parallel_size"] == 4
    assert workload["gpu_indices"] == [3, 4, 6, 7]
    assert workload["request_order"] == [0, 1, 2, 3]
    assert workload["accepted_prefix_semantics"] is True
    assert workload["proposal_slot_capacity"] == 4 * (256 + 16 + 4)
    assert workload["requested_output_tokens"] == 16
    assert workload["epoch_keys"] == [
        identity.key for identity in diagnostic.expected_epoch_identities()
    ]


def test_epoch_workload_identity_rejects_output_token_drift(valid_worker):
    diagnostic = load_module(DIAGNOSTIC_PATH, "paired_output_drift")
    epochs = _valid_epochs(diagnostic, valid_worker)
    last = epochs[diagnostic.expected_epoch_identities()[-1].key]["worker"]
    last["measured_runs"][0]["outputs"][0][0] += 1
    with pytest.raises(ValueError, match="output token IDs"):
        diagnostic.validate_epoch_workload_identity(epochs)


def test_epoch_workload_identity_rejects_proposal_counter_drift(
    valid_worker,
):
    diagnostic = load_module(DIAGNOSTIC_PATH, "paired_proposal_drift")
    epochs = _valid_epochs(diagnostic, valid_worker)
    last = epochs[diagnostic.expected_epoch_identities()[-1].key]["worker"]
    _mutate_proposal_count(last)
    with pytest.raises(ValueError, match="proposal counts"):
        diagnostic.validate_epoch_workload_identity(epochs)


def test_stationarity_uses_all_values_for_median_and_mad():
    diagnostic = load_module(DIAGNOSTIC_PATH, "paired_stationarity_all")
    row = diagnostic.stationarity_for_values(
        "e2e_s",
        [9.0, 10.0, 1000.0, 11.0, 12.0],
    )
    assert row["epoch_median"] == 11.0
    assert row["epoch_mad"] == 1.0


def test_half_drift_excludes_center_repeat():
    diagnostic = load_module(DIAGNOSTIC_PATH, "paired_half_drift")
    row = diagnostic.stationarity_for_values(
        "e2e_s",
        [10.0, 10.0, 500.0, 11.5, 11.5],
    )
    assert row["first_half_values"] == [10.0, 10.0]
    assert row["center_value"] == 500.0
    assert row["second_half_values"] == [11.5, 11.5]
    assert row["half_drift"] == pytest.approx(
        abs(11.5 - 10.0) / row["epoch_median"]
    )


@pytest.mark.parametrize(
    ("values", "stable"),
    [
        ([9.0, 9.0, 10.0, 11.0, 11.0], True),
        ([8.9, 8.9, 10.0, 11.1, 11.1], False),
    ],
)
def test_robust_dispersion_equality_passes(values, stable):
    diagnostic = load_module(DIAGNOSTIC_PATH, "paired_mad_boundary")
    assert (
        diagnostic.stationarity_for_values("e2e_s", values)[
            "robust_dispersion"
        ]
        <= diagnostic.ROBUST_DISPERSION_LIMIT
    ) is stable


@pytest.mark.parametrize(
    ("values", "stable"),
    [
        ([10.0, 10.0, 10.0, 11.5, 11.5], True),
        ([10.0, 10.0, 10.0, 11.501, 11.501], False),
    ],
)
def test_half_drift_equality_passes(values, stable):
    diagnostic = load_module(DIAGNOSTIC_PATH, "paired_drift_boundary")
    assert (
        diagnostic.stationarity_for_values("e2e_s", values)["stable"]
        is stable
    )


@pytest.mark.parametrize(
    "values",
    [
        None,
        [1.0] * 4,
        [1.0] * 6,
        [0.0] * 5,
        [-1.0] * 5,
        [float("inf")] * 5,
        [float("nan")] * 5,
    ],
)
def test_stationarity_rejects_invalid_values(values):
    diagnostic = load_module(DIAGNOSTIC_PATH, "paired_bad_stationarity")
    with pytest.raises(ValueError, match="stationarity"):
        diagnostic.stationarity_for_values("e2e_s", values)


def _failure_codes(admission: dict) -> set[str]:
    return {failure["code"] for failure in admission["failures"]}


def test_epoch_rejects_duplicate_repeat_index(valid_raw_epoch, identity):
    diagnostic = load_module(DIAGNOSTIC_PATH, "paired_duplicate_repeat")
    invalid = copy.deepcopy(valid_raw_epoch)
    invalid["worker"]["measured_runs"][4]["repeat"] = 3
    admission = diagnostic.build_epoch_admission(identity, invalid)
    assert admission["passed"] is False
    assert "DUPLICATE_REPEAT_INDEX" in _failure_codes(admission)


def test_epoch_rejects_missing_gpu_coverage(valid_raw_epoch, identity):
    diagnostic = load_module(DIAGNOSTIC_PATH, "paired_gpu_coverage")
    invalid = copy.deepcopy(valid_raw_epoch)
    invalid["gpu_rows"] = invalid["gpu_rows"][:1]
    admission = diagnostic.build_epoch_admission(identity, invalid)
    assert admission["passed"] is False
    assert "GPU_TELEMETRY_COVERAGE" in _failure_codes(admission)


def test_epoch_rejects_protected_process_disappearance(
    valid_raw_epoch,
    identity,
):
    diagnostic = load_module(DIAGNOSTIC_PATH, "paired_protected_pid")
    invalid = copy.deepcopy(valid_raw_epoch)
    invalid["process_after"]["protected_gpu7_pid_present"] = False
    admission = diagnostic.build_epoch_admission(identity, invalid)
    assert admission["passed"] is False
    assert "PROTECTED_PROCESS_MISSING" in _failure_codes(admission)


def test_epoch_rejects_script_owned_process_leak(
    valid_raw_epoch,
    identity,
):
    diagnostic = load_module(DIAGNOSTIC_PATH, "paired_process_leak")
    invalid = copy.deepcopy(valid_raw_epoch)
    invalid["process_after"]["runner_owned_pids_remaining"] = [999999]
    admission = diagnostic.build_epoch_admission(identity, invalid)
    assert admission["passed"] is False
    assert "RUNNER_PROCESS_LEAK" in _failure_codes(admission)


@pytest.mark.parametrize(
    ("mutation", "code"),
    [
        (
            lambda raw: raw["worker"]["measured_runs"].pop(),
            "MEASURED_REPEAT_COUNT",
        ),
        (
            lambda raw: raw["worker"]["measured_runs"].append(
                copy.deepcopy(raw["worker"]["measured_runs"][-1])
            ),
            "MEASURED_REPEAT_COUNT",
        ),
        (
            lambda raw: raw["worker"]["measured_runs"][1][
                "campaign_interval"
            ].__setitem__(
                "started_at_unix_ns",
                raw["worker"]["measured_runs"][0]["campaign_interval"][
                    "started_at_unix_ns"
                ],
            ),
            "NON_MONOTONIC_INTERVALS",
        ),
        (
            lambda raw: raw["host_rows"].__setitem__(
                1,
                {
                    **raw["host_rows"][1],
                    "sampled_at_unix_ns": raw["host_rows"][0][
                        "sampled_at_unix_ns"
                    ],
                },
            ),
            "HOST_TELEMETRY_COVERAGE",
        ),
        (
            lambda raw: raw["gpu_rows"][0].__setitem__(
                "uuid",
                "GPU-different",
            ),
            "GPU_UUID_CHANGED",
        ),
        (
            lambda raw: raw["gpu_invariants"][
                "undeclared_gpu_indices"
            ].append(0),
            "UNDECLARED_GPU_USAGE",
        ),
        (
            lambda raw: raw["gpu_invariants"]["xid_events"].append(31),
            "GPU_XID",
        ),
        (
            lambda raw: raw["gpu_invariants"]["reset_events"].append(
                "reset"
            ),
            "GPU_RESET",
        ),
        (
            lambda raw: raw["gpu_invariants"].__setitem__(
                "throttle_valid",
                False,
            ),
            "GPU_THROTTLE_INVALID",
        ),
        (
            lambda raw: raw["gpu_invariants"].__setitem__(
                "telemetry_available",
                False,
            ),
            "GPU_TELEMETRY_UNAVAILABLE",
        ),
        (
            lambda raw: raw["gpu_invariants"].__setitem__(
                "clocks_pstate_valid",
                False,
            ),
            "GPU_CLOCK_PSTATE_INVALID",
        ),
        (
            lambda raw: raw["process_after"].__setitem__(
                "unrelated_process_inventory",
                [101, 303],
            ),
            "UNRELATED_PROCESS_INVENTORY_CHANGED",
        ),
        (
            lambda raw: raw.__setitem__("exact_parity", False),
            "EXACT_PARITY_FAILED",
        ),
        (
            lambda raw: raw.__setitem__(
                "accepted_prefix_semantics",
                False,
            ),
            "ACCEPTED_PREFIX_MISMATCH",
        ),
        (
            lambda raw: raw["proposal_counts"].__setitem__(0, 1),
            "PROPOSAL_COUNT_MISMATCH",
        ),
        (
            lambda raw: raw["proposal_lengths"].__setitem__(0, 3),
            "PROPOSAL_LENGTH_MISMATCH",
        ),
        (
            lambda raw: raw.__setitem__(
                "total_verified_tokens",
                raw["total_verified_tokens"] + 1,
            ),
            "VERIFIED_TOKEN_COUNT_MISMATCH",
        ),
        (
            lambda raw: raw.__setitem__(
                "prime_excluded_from_measured_statistics",
                False,
            ),
            "PRIME_ENTERED_MEASURED_STATISTICS",
        ),
    ],
)
def test_epoch_invariant_failures_are_stable_codes(
    valid_raw_epoch,
    identity,
    mutation,
    code,
):
    diagnostic = load_module(DIAGNOSTIC_PATH, f"paired_invariant_{code}")
    invalid = copy.deepcopy(valid_raw_epoch)
    mutation(invalid)
    admission = diagnostic.build_epoch_admission(identity, invalid)
    assert admission["passed"] is False
    assert code in _failure_codes(admission)


def test_valid_epoch_admission_retains_prime_but_excludes_it(
    valid_raw_epoch,
    identity,
):
    diagnostic = load_module(DIAGNOSTIC_PATH, "paired_valid_admission")
    admission = diagnostic.build_epoch_admission(
        identity,
        valid_raw_epoch,
    )
    assert admission["passed"] is True
    assert admission["repeat_count"] == 5
    assert admission["prime"] == {
        "recorded": True,
        "excluded_from_measured_statistics": True,
    }
    assert set(admission["stationarity"]) == set(
        diagnostic.PRIMARY_METRICS
    )


def test_bundle_admission_is_all_or_nothing(valid_raw_epoch):
    diagnostic = load_module(DIAGNOSTIC_PATH, "paired_bundle_admission")
    epochs = {}
    for identity in diagnostic.expected_epoch_identities():
        raw = copy.deepcopy(valid_raw_epoch)
        epochs[identity.key] = diagnostic.build_epoch_admission(
            identity,
            raw,
        )
    epochs[next(iter(epochs))]["failures"].append(
        diagnostic.AdmissionFailure(
            code="FORCED",
            identity=diagnostic.expected_epoch_identities()[0],
            metric=None,
            observed=True,
            expected="false",
            source_path="forced",
        ).to_dict()
    )
    epochs[next(iter(epochs))]["passed"] = False
    bundle = diagnostic.build_bundle_admission(epochs)
    assert bundle["passed"] is False
    assert bundle["epoch_count"] == 8
    assert bundle["measured_repeat_count_total"] == 40
    assert bundle["failed_epoch_keys"] == [next(iter(epochs))]


def make_admitted_epoch_summaries(
    *,
    block_medians: dict[int, dict[str, float]],
) -> dict[str, dict]:
    diagnostic = load_module(DIAGNOSTIC_PATH, "paired_effect_fixture")
    epochs = {}
    for identity in diagnostic.expected_epoch_identities():
        e2e = block_medians[identity.block_index][identity.label]
        epochs[identity.key] = {
            "identity": {
                "key": identity.key,
                "block_index": identity.block_index,
                "order": identity.order,
                "label": identity.label,
                "position": identity.position,
                "epoch_index": identity.epoch_index,
            },
            "passed": True,
            "metric_medians": {
                "e2e_s": e2e,
                "tpot_s": e2e / 2.0,
                "executor_proposal_forward_ms": e2e * 10.0,
                "executor_backend_submit_ms": e2e,
            },
            "metrics": {
                "e2e_s": [e2e] * 5,
                "tpot_s": [e2e / 2.0] * 5,
                "executor_proposal_forward_ms": [e2e * 10.0] * 5,
                "executor_backend_submit_ms": [e2e] * 5,
            },
            "telemetry": {
                "gpu": {"epoch": identity.epoch_index},
                "host": [{"repeat": repeat} for repeat in range(5)],
            },
            "acceptance_rate": 0.5,
            "proposal_lengths": [4] * 5,
            "total_verified_tokens": 480,
        }
    return epochs


def make_effect_fixture(
    *,
    e2e_position_relatives: list[float],
    e2e_label_relatives: list[float] | None = None,
) -> dict[str, dict]:
    del e2e_label_relatives
    diagnostic = load_module(DIAGNOSTIC_PATH, "paired_relative_fixture")
    epochs = {}
    for identity in diagnostic.expected_epoch_identities():
        relative = e2e_position_relatives[identity.block_index]
        value = 10.0 if identity.position == "first" else 10.0 * (
            1.0 + relative
        )
        epochs[identity.key] = {
            "identity": {"key": identity.key},
            "passed": True,
            "metric_medians": {
                "e2e_s": value,
                "tpot_s": value,
                "executor_proposal_forward_ms": value,
                "executor_backend_submit_ms": value,
            },
            "metrics": {
                metric: [value * (1.0 + repeat / 1000.0)
                         for repeat in range(5)]
                for metric in (
                    "e2e_s",
                    "tpot_s",
                    "executor_proposal_forward_ms",
                    "executor_backend_submit_ms",
                )
            },
            "telemetry": {"gpu": {}, "host": []},
            "acceptance_rate": 0.5,
            "proposal_lengths": [4] * 5,
            "total_verified_tokens": 480,
        }
    return epochs


def make_candidate_effects(
    *,
    e2e_block_relatives: list[float] | None = None,
    e2e_aggregate_relative: float = 0.20,
) -> dict:
    if e2e_block_relatives is None:
        e2e_block_relatives = [0.20, 0.20, 0.20, -0.01]
    block_local = {}
    aggregate = {}
    for metric in (
        "e2e_s",
        "tpot_s",
        "executor_proposal_forward_ms",
        "executor_backend_submit_ms",
    ):
        relatives = (
            e2e_block_relatives
            if metric == "e2e_s"
            else [0.20, 0.20, 0.20, -0.01]
        )
        block_local[metric] = [
            {
                "block_index": index,
                "order": "AB" if index in (0, 3) else "BA",
                "position_effect": math.log1p(relative),
                "position_relative": relative,
                "label_effect": 0.0,
                "label_relative": 0.0,
            }
            for index, relative in enumerate(relatives)
        ]
        relative = (
            e2e_aggregate_relative if metric == "e2e_s" else 0.20
        )
        aggregate[metric] = {
            "log_effect": math.log1p(relative),
            "relative_effect": relative,
        }
    return {
        "block_local": block_local,
        "aggregate_position_effects": aggregate,
        "aggregate_label_effects": {
            metric: {"log_effect": 0.0, "relative_effect": 0.0}
            for metric in aggregate
        },
        "ab_position_effects": {
            metric: {
                "log_effect": math.log1p(0.20),
                "relative_effect": 0.20,
            }
            for metric in aggregate
        },
        "ba_position_effects": {
            metric: {
                "log_effect": math.log1p(0.20),
                "relative_effect": 0.20,
            }
            for metric in aggregate
        },
        "sequence_interactions": {
            metric: 0.0 for metric in aggregate
        },
        "diagnostic_effects": {},
    }


def test_ab_block_position_and_label_effects():
    diagnostic = load_module(DIAGNOSTIC_PATH, "paired_ab_effects")
    epochs = make_admitted_epoch_summaries(
        block_medians={
            0: {"A": 10.0, "B": 12.0},
            1: {"B": 10.0, "A": 12.0},
            2: {"B": 10.0, "A": 12.0},
            3: {"A": 10.0, "B": 12.0},
        }
    )
    effects = diagnostic.compute_paired_effects(epochs)
    block0 = effects["block_local"]["e2e_s"][0]
    assert block0["position_effect"] == pytest.approx(
        math.log(12.0 / 10.0)
    )
    assert block0["position_relative"] == pytest.approx(0.20)
    assert block0["label_effect"] == pytest.approx(
        math.log(10.0 / 12.0)
    )
    assert block0["label_relative"] == pytest.approx(-1.0 / 6.0)


def test_ba_block_position_and_label_effects():
    diagnostic = load_module(DIAGNOSTIC_PATH, "paired_ba_effects")
    epochs = make_admitted_epoch_summaries(
        block_medians={
            0: {"A": 10.0, "B": 12.0},
            1: {"B": 10.0, "A": 12.0},
            2: {"B": 10.0, "A": 12.0},
            3: {"A": 10.0, "B": 12.0},
        }
    )
    effects = diagnostic.compute_paired_effects(epochs)
    block1 = effects["block_local"]["e2e_s"][1]
    assert block1["position_effect"] == pytest.approx(
        math.log(12.0 / 10.0)
    )
    assert block1["label_effect"] == pytest.approx(
        math.log(12.0 / 10.0)
    )


def test_aggregate_and_sequence_interaction_are_log_medians():
    diagnostic = load_module(DIAGNOSTIC_PATH, "paired_aggregate_effects")
    effects = diagnostic.compute_paired_effects(
        make_effect_fixture(
            e2e_position_relatives=[0.10, 0.20, 0.30, 0.40],
            e2e_label_relatives=[0.01, -0.01, 0.02, -0.02],
        )
    )
    expected = statistics.median(
        [math.log1p(value) for value in [0.10, 0.20, 0.30, 0.40]]
    )
    assert effects["aggregate_position_effects"]["e2e_s"][
        "log_effect"
    ] == pytest.approx(expected)
    assert effects["sequence_interactions"]["e2e_s"] == pytest.approx(
        effects["ab_position_effects"]["e2e_s"]["log_effect"]
        - effects["ba_position_effects"]["e2e_s"]["log_effect"]
    )


def test_diagnostic_effects_retain_non_candidate_evidence():
    diagnostic = load_module(DIAGNOSTIC_PATH, "paired_diagnostic_effects")
    effects = diagnostic.compute_paired_effects(
        make_effect_fixture(
            e2e_position_relatives=[0.10, 0.20, 0.30, 0.40],
        )
    )
    evidence = effects["diagnostic_effects"]
    assert len(evidence["raw_repeat_ratios"]["e2e_s"]) == 4
    assert len(evidence["chronological_block_trend"]["e2e_s"]) == 4
    assert len(evidence["backend_submit_effects"]) == 4
    assert len(evidence["acceptance_summaries"]) == 8
    assert len(evidence["proposal_length_summaries"]) == 8
    assert len(evidence["verified_token_summaries"]) == 8


def test_any_admission_failure_has_highest_precedence():
    diagnostic = load_module(DIAGNOSTIC_PATH, "paired_unstable_precedence")
    admission = {"passed": False, "failures": [{"code": "HALF_DRIFT"}]}
    effects = make_candidate_effects()
    order_check = {"passed": True, "reasons": []}
    classification, candidate, reasons = (
        diagnostic.classify_paired_stability(
            admission,
            effects,
            order_check,
        )
    )
    assert classification == "PAIRED_PROTOCOL_UNSTABLE"
    assert candidate is False
    assert reasons == ["bundle admission failed"]


def test_candidate_accepts_exact_ten_percent_aggregate_boundary():
    diagnostic = load_module(DIAGNOSTIC_PATH, "paired_ten_percent")
    effects = make_candidate_effects(
        e2e_block_relatives=[0.10, 0.10, 0.10, -0.01],
        e2e_aggregate_relative=0.10,
    )
    classification, candidate, _ = diagnostic.classify_paired_stability(
        {"passed": True, "failures": []},
        effects,
        {"passed": True, "reasons": []},
    )
    assert classification == "CANDIDATE_PROCESS_BOUNDARY_EFFECT"
    assert candidate is True


def test_candidate_requires_three_of_four_common_direction():
    diagnostic = load_module(DIAGNOSTIC_PATH, "paired_three_of_four")
    effects = make_candidate_effects(
        e2e_block_relatives=[0.20, 0.20, -0.20, -0.20]
    )
    classification, candidate, reasons = (
        diagnostic.classify_paired_stability(
            {"passed": True, "failures": []},
            effects,
            {"passed": True, "reasons": []},
        )
    )
    assert classification == "NO_REPRODUCIBLE_PROCESS_EFFECT"
    assert candidate is False
    assert "fewer than three E2E blocks share a direction" in reasons


@pytest.mark.parametrize(
    "metric",
    ["tpot_s", "executor_proposal_forward_ms"],
)
def test_candidate_rejects_primary_direction_disagreement(metric):
    diagnostic = load_module(
        DIAGNOSTIC_PATH,
        "paired_direction_disagreement",
    )
    effects = make_candidate_effects()
    effects["aggregate_position_effects"][metric]["log_effect"] *= -1.0
    effects["aggregate_position_effects"][metric]["relative_effect"] = (
        math.exp(
            effects["aggregate_position_effects"][metric]["log_effect"]
        )
        - 1.0
    )
    classification, candidate, _ = diagnostic.classify_paired_stability(
        {"passed": True, "failures": []},
        effects,
        {"passed": True, "reasons": []},
    )
    assert classification == "NO_REPRODUCIBLE_PROCESS_EFFECT"
    assert candidate is False


@pytest.mark.parametrize(
    ("mutation", "failed_check"),
    [
        (
            lambda effects: effects["ab_position_effects"]["e2e_s"].update(
                {
                    "log_effect": -math.log1p(0.20),
                    "relative_effect": -1.0 / 6.0,
                }
            ),
            "ab_direction_matches",
        ),
        (
            lambda effects: [
                row.update(
                    {
                        "position_effect": math.log1p(0.01),
                        "position_relative": 0.01,
                    }
                )
                for row in effects["block_local"]["e2e_s"]
                if row["order"] == "AB"
            ],
            "ab_has_qualifying_block",
        ),
        (
            lambda effects: [
                row.update(
                    {
                        "position_effect": math.log1p(0.01),
                        "position_relative": 0.01,
                    }
                )
                for row in effects["block_local"]["e2e_s"]
                if row["order"] == "BA"
            ],
            "ba_has_qualifying_block",
        ),
        (
            lambda effects: effects["aggregate_label_effects"][
                "e2e_s"
            ].update(
                {
                    "log_effect": math.log1p(0.10),
                    "relative_effect": 0.10,
                }
            ),
            "aggregate_label_below_threshold",
        ),
    ],
)
def test_order_effect_check_rejects_confounds(mutation, failed_check):
    diagnostic = load_module(DIAGNOSTIC_PATH, "paired_order_check")
    effects = make_candidate_effects()
    mutation(effects)
    result = diagnostic.order_effect_check(effects)
    assert result["passed"] is False
    assert failed_check in result["reasons"]


def test_order_effect_rejects_three_large_common_label_effects():
    diagnostic = load_module(DIAGNOSTIC_PATH, "paired_label_candidate")
    effects = make_candidate_effects()
    for row in effects["block_local"]["e2e_s"][:3]:
        row["label_effect"] = math.log1p(0.10)
        row["label_relative"] = 0.10
    result = diagnostic.order_effect_check(effects)
    assert result["passed"] is False
    assert "label_does_not_form_candidate" in result["reasons"]


@pytest.fixture
def valid_bundle_inputs(valid_raw_epoch) -> dict:
    diagnostic = load_module(DIAGNOSTIC_PATH, "paired_bundle_fixture")
    epoch_raw_inputs = {
        identity.key: copy.deepcopy(valid_raw_epoch)
        for identity in diagnostic.expected_epoch_identities()
    }
    input_files = {}
    for identity in diagnostic.expected_epoch_identities():
        path = f"{identity.key}/raw.json"
        input_files[identity.key] = {
            "path": path,
            "sha256": hashlib.sha256(path.encode("utf-8")).hexdigest(),
        }
    source_files = {
        path: hashlib.sha256((ROOT / path).read_bytes()).hexdigest()
        for path in (
            "tools/autoregressive_draft_paired_stability_diagnostic.py",
            "tools/autoregressive_draft_performance_gate.py",
        )
    }
    return {
        "metadata": {
            "run_tag": (
                "tp4-qwen3-b4-paired-stability-20260815T230000Z"
            ),
            "bundle_start_utc": "2026-08-15T23:00:00Z",
            "bundle_finish_utc": "2026-08-15T23:30:00Z",
            "remote_host": "sitian@10.232.195.203",
            "remote_base": (
                "/dev/shm/sitian/"
                "tllm-qwen35-target-qwen3-draft-20260815"
            ),
            "configuration": {
                "batch_size": 4,
                "max_proposal_tokens": 4,
                "temperature": 0.0,
                "gpu_indices": [3, 4, 6, 7],
            },
            "model_identity": {
                "target": "Qwen3-1.7B",
                "draft": "Qwen3-0.6B",
            },
            "prompt_identity": {"batch_size": 4, "prompt_tokens": 256},
            "command_identity": {
                "python": (
                    "/data00/home/sitian/miniconda3/envs/py311/bin/python"
                )
            },
        },
        "epoch_raw_inputs": epoch_raw_inputs,
        "input_files": input_files,
        "source_files": source_files,
    }


@pytest.fixture
def valid_bundle_directory(
    valid_bundle_inputs,
    tmp_path,
) -> Path:
    root = tmp_path / "bundle"
    root.mkdir()
    (root / "schedule.txt").write_text(
        "AB\nBA\nBA\nAB\n",
        encoding="utf-8",
    )
    (root / "command.txt").write_text(
        "paired-stability\n",
        encoding="utf-8",
    )
    (root / "metadata.json").write_text(
        json.dumps(valid_bundle_inputs["metadata"], sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (root / "source-files.json").write_text(
        json.dumps(
            valid_bundle_inputs["source_files"],
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    diagnostic = load_module(DIAGNOSTIC_PATH, "paired_directory_fixture")
    for identity in diagnostic.expected_epoch_identities():
        epoch_dir = root / identity.key
        epoch_dir.mkdir(parents=True)
        (epoch_dir / "raw.json").write_text(
            json.dumps(
                valid_bundle_inputs["epoch_raw_inputs"][identity.key],
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
    return root


@pytest.fixture
def canonical_artifact_path(valid_bundle_directory) -> Path:
    diagnostic = load_module(DIAGNOSTIC_PATH, "paired_artifact_path")
    path = valid_bundle_directory / "paired-stability.json"
    diagnostic.build_from_paths(
        bundle_root=valid_bundle_directory,
        repo_root=ROOT,
        out=path,
    )
    return path


@pytest.fixture
def valid_manifest_path(
    valid_bundle_directory,
    canonical_artifact_path,
) -> Path:
    del canonical_artifact_path
    manifest = valid_bundle_directory / "manifest.sha256"
    rows = []
    for path in sorted(
        candidate
        for candidate in valid_bundle_directory.rglob("*")
        if candidate.is_file() and candidate != manifest
    ):
        relative = path.relative_to(valid_bundle_directory).as_posix()
        rows.append(
            f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {relative}"
        )
    manifest.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return manifest


def test_canonical_artifact_is_self_describing(valid_bundle_inputs):
    diagnostic = load_module(DIAGNOSTIC_PATH, "paired_artifact")
    artifact = diagnostic.build_paired_stability_artifact(
        **valid_bundle_inputs
    )
    required = {
        "schema_version",
        "classification",
        "candidate_process_boundary_effect",
        "process_boundary_effect_established",
        "claim_boundary",
        "run_tag",
        "bundle_start_utc",
        "bundle_finish_utc",
        "remote_host",
        "remote_base",
        "schedule",
        "schedule_sha256",
        "configuration",
        "source_files",
        "source_sha256",
        "model_identity",
        "prompt_identity",
        "command_identity",
        "blocks",
        "epochs",
        "measured_repeat_count_total",
        "epoch_admission",
        "bundle_admission",
        "primary_stationarity",
        "coverage",
        "gpu_invariants",
        "process_invariants",
        "exact_parity",
        "block_local_position_effects",
        "block_local_label_effects",
        "aggregate_position_effects",
        "aggregate_label_effects",
        "ab_position_effects",
        "ba_position_effects",
        "sequence_interactions",
        "diagnostic_effects",
        "raw_input_files",
        "raw_input_sha256",
    }
    assert required <= set(artifact)
    assert artifact["process_boundary_effect_established"] is False
    assert artifact["schedule"] == ["AB", "BA", "BA", "AB"]
    assert artifact["schedule_sha256"] == diagnostic.SCHEDULE_SHA256
    assert len(artifact["blocks"]) == 4
    assert len(artifact["epochs"]) == 8
    assert artifact["measured_repeat_count_total"] == 40


def test_unstable_artifact_retains_all_raw_inputs(valid_bundle_inputs):
    diagnostic = load_module(DIAGNOSTIC_PATH, "paired_partial_evidence")
    invalid = copy.deepcopy(valid_bundle_inputs)
    first_key = diagnostic.expected_epoch_identities()[0].key
    invalid["epoch_raw_inputs"][first_key]["worker"][
        "measured_runs"
    ].pop()
    artifact = diagnostic.build_paired_stability_artifact(**invalid)
    assert artifact["classification"] == "PAIRED_PROTOCOL_UNSTABLE"
    assert len(artifact["epochs"]) == 8
    assert first_key in artifact["raw_input_files"]


def test_artifact_forces_established_false(valid_bundle_inputs):
    diagnostic = load_module(DIAGNOSTIC_PATH, "paired_claim_boundary")
    artifact = diagnostic.build_paired_stability_artifact(
        **valid_bundle_inputs
    )
    artifact["process_boundary_effect_established"] = True
    with pytest.raises(ValueError, match="must remain false"):
        diagnostic.validate_paired_stability_artifact(artifact)


def test_artifact_binds_schedule_sources_and_raw_inputs(
    valid_bundle_inputs,
):
    diagnostic = load_module(DIAGNOSTIC_PATH, "paired_artifact_binding")
    artifact = diagnostic.build_paired_stability_artifact(
        **valid_bundle_inputs
    )
    assert artifact["schedule_text"] == diagnostic.SCHEDULE_TEXT
    assert artifact["source_sha256"] == diagnostic.digest_mapping(
        valid_bundle_inputs["source_files"]
    )
    assert artifact["raw_input_sha256"] == (
        diagnostic.digest_input_inventory(
            valid_bundle_inputs["input_files"]
        )
    )
    for row in artifact["raw_input_files"].values():
        assert not Path(row["path"]).is_absolute()
        assert ".." not in Path(row["path"]).parts
        assert len(row["sha256"]) == 64


@pytest.mark.parametrize(
    ("classification", "candidate"),
    [
        ("PAIRED_PROTOCOL_UNSTABLE", False),
        ("NO_REPRODUCIBLE_PROCESS_EFFECT", False),
        ("CANDIDATE_PROCESS_BOUNDARY_EFFECT", True),
    ],
)
def test_artifact_classification_boolean_pairs(
    valid_bundle_inputs,
    classification,
    candidate,
):
    diagnostic = load_module(DIAGNOSTIC_PATH, "paired_artifact_pair")
    artifact = diagnostic.build_paired_stability_artifact(
        **valid_bundle_inputs
    )
    artifact["classification"] = classification
    artifact["candidate_process_boundary_effect"] = candidate
    diagnostic.validate_paired_stability_artifact(artifact)


def test_artifact_rejects_unsafe_raw_input_path(valid_bundle_inputs):
    diagnostic = load_module(DIAGNOSTIC_PATH, "paired_unsafe_raw")
    invalid = copy.deepcopy(valid_bundle_inputs)
    first = next(iter(invalid["input_files"].values()))
    first["path"] = "../escape.json"
    with pytest.raises(ValueError, match="safe relative"):
        diagnostic.build_paired_stability_artifact(**invalid)


def test_cli_requires_bundle_root_repo_root_and_out():
    diagnostic = load_module(DIAGNOSTIC_PATH, "paired_cli")
    with pytest.raises(SystemExit):
        diagnostic.parse_args([])
    args = diagnostic.parse_args(
        [
            "--bundle-root",
            "/tmp/bundle",
            "--repo-root",
            str(ROOT),
            "--out",
            "/tmp/paired-stability.json",
        ]
    )
    assert args.bundle_root == "/tmp/bundle"
    assert args.repo_root == str(ROOT)
    assert args.out == "/tmp/paired-stability.json"


def test_cli_refuses_existing_output(valid_bundle_directory, tmp_path):
    diagnostic = load_module(DIAGNOSTIC_PATH, "paired_cli_overwrite")
    out = tmp_path / "paired-stability.json"
    out.write_text("{}\n", encoding="utf-8")
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        diagnostic.build_from_paths(
            bundle_root=valid_bundle_directory,
            repo_root=ROOT,
            out=out,
        )


def test_cli_builds_from_exact_epoch_layout(
    valid_bundle_directory,
    tmp_path,
):
    diagnostic = load_module(DIAGNOSTIC_PATH, "paired_cli_build")
    out = tmp_path / "paired-stability.json"
    artifact = diagnostic.build_from_paths(
        bundle_root=valid_bundle_directory,
        repo_root=ROOT,
        out=out,
    )
    assert out.exists()
    assert json.loads(out.read_text(encoding="utf-8")) == artifact
    assert len(artifact["epochs"]) == 8


def test_verifier_recomputes_artifact_from_raw_inputs(
    valid_bundle_directory,
    canonical_artifact_path,
):
    del valid_bundle_directory
    verifier = load_module(VERIFIER_PATH, "paired_verifier")
    receipt = verifier.verify_paired_stability_diagnostic(
        artifact_path=canonical_artifact_path,
        repo_root=ROOT,
    )
    assert receipt["verified"] is True
    assert receipt["classification"] in {
        "PAIRED_PROTOCOL_UNSTABLE",
        "NO_REPRODUCIBLE_PROCESS_EFFECT",
        "CANDIDATE_PROCESS_BOUNDARY_EFFECT",
    }
    assert receipt["process_boundary_effect_established"] is False


def test_verifier_rejects_schedule_tamper(
    valid_bundle_directory,
    canonical_artifact_path,
):
    verifier = load_module(VERIFIER_PATH, "paired_schedule_tamper")
    schedule = valid_bundle_directory / "schedule.txt"
    schedule.write_text("AB\nBA\nAB\nBA\n", encoding="utf-8")
    with pytest.raises(ValueError, match="raw input hash mismatch|schedule"):
        verifier.verify_paired_stability_diagnostic(
            artifact_path=canonical_artifact_path,
            repo_root=ROOT,
        )


def test_verifier_rejects_raw_input_tamper(
    valid_bundle_directory,
    canonical_artifact_path,
):
    verifier = load_module(VERIFIER_PATH, "paired_raw_tamper")
    worker = (
        valid_bundle_directory
        / "block-0-ab"
        / "a-first"
        / "raw.json"
    )
    worker.write_text(
        worker.read_text(encoding="utf-8") + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="raw input hash mismatch"):
        verifier.verify_paired_stability_diagnostic(
            artifact_path=canonical_artifact_path,
            repo_root=ROOT,
        )


def test_verifier_rejects_derived_classification_tamper(
    canonical_artifact_path,
):
    verifier = load_module(VERIFIER_PATH, "paired_derived_tamper")
    artifact = json.loads(canonical_artifact_path.read_text())
    artifact["classification"] = "CANDIDATE_PROCESS_BOUNDARY_EFFECT"
    artifact["candidate_process_boundary_effect"] = True
    canonical_artifact_path.write_text(
        json.dumps(artifact, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="canonical artifact mismatch"):
        verifier.verify_paired_stability_diagnostic(
            artifact_path=canonical_artifact_path,
            repo_root=ROOT,
        )


@pytest.mark.parametrize(
    "mutate",
    [
        lambda artifact: artifact["primary_stationarity"][
            next(iter(artifact["primary_stationarity"]))
        ]["e2e_s"].__setitem__("epoch_mad", 999.0),
        lambda artifact: artifact["aggregate_position_effects"][
            "e2e_s"
        ].__setitem__("log_effect", 999.0),
        lambda artifact: artifact["sequence_interactions"].__setitem__(
            "e2e_s",
            999.0,
        ),
        lambda artifact: artifact["order_effect_check"].__setitem__(
            "passed",
            not artifact["order_effect_check"]["passed"],
        ),
    ],
)
def test_verifier_rejects_derived_numeric_tamper(
    canonical_artifact_path,
    mutate,
):
    verifier = load_module(VERIFIER_PATH, "paired_numeric_tamper")
    artifact = json.loads(canonical_artifact_path.read_text())
    mutate(artifact)
    canonical_artifact_path.write_text(
        json.dumps(artifact, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="canonical artifact mismatch"):
        verifier.verify_paired_stability_diagnostic(
            artifact_path=canonical_artifact_path,
            repo_root=ROOT,
        )


def test_verifier_rejects_source_hash_tamper(canonical_artifact_path):
    diagnostic = load_module(DIAGNOSTIC_PATH, "paired_source_tamper_diag")
    verifier = load_module(VERIFIER_PATH, "paired_source_tamper")
    artifact = json.loads(canonical_artifact_path.read_text())
    first = next(iter(artifact["source_files"]))
    artifact["source_files"][first] = "0" * 64
    artifact["source_sha256"] = diagnostic.digest_mapping(
        artifact["source_files"]
    )
    canonical_artifact_path.write_text(
        json.dumps(artifact, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="source hash mismatch"):
        verifier.verify_paired_stability_diagnostic(
            artifact_path=canonical_artifact_path,
            repo_root=ROOT,
        )


@pytest.mark.parametrize("path", ["../escape.json", "/tmp/escape.json"])
def test_verifier_rejects_unsafe_bound_paths(
    canonical_artifact_path,
    path,
):
    verifier = load_module(VERIFIER_PATH, "paired_unsafe_bound")
    artifact = json.loads(canonical_artifact_path.read_text())
    first = next(iter(artifact["raw_input_files"].values()))
    first["path"] = path
    canonical_artifact_path.write_text(
        json.dumps(artifact, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="safe relative"):
        verifier.verify_paired_stability_diagnostic(
            artifact_path=canonical_artifact_path,
            repo_root=ROOT,
        )


def test_verifier_rejects_missing_bound_file(
    valid_bundle_directory,
    canonical_artifact_path,
):
    verifier = load_module(VERIFIER_PATH, "paired_missing_raw")
    raw = valid_bundle_directory / "block-0-ab" / "a-first" / "raw.json"
    raw.unlink()
    with pytest.raises(ValueError, match="bound file is missing"):
        verifier.verify_paired_stability_diagnostic(
            artifact_path=canonical_artifact_path,
            repo_root=ROOT,
        )


def test_verifier_rejects_established_true(canonical_artifact_path):
    verifier = load_module(VERIFIER_PATH, "paired_established_true")
    artifact = json.loads(canonical_artifact_path.read_text())
    artifact["process_boundary_effect_established"] = True
    canonical_artifact_path.write_text(
        json.dumps(artifact, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="must remain false"):
        verifier.verify_paired_stability_diagnostic(
            artifact_path=canonical_artifact_path,
            repo_root=ROOT,
        )


def test_remote_and_local_receipts_differ_only_in_permitted_metadata(
    canonical_artifact_path,
):
    verifier = load_module(VERIFIER_PATH, "paired_receipt_equivalence")
    remote = verifier.verify_paired_stability_diagnostic(
        artifact_path=canonical_artifact_path,
        repo_root=ROOT,
    )
    local = verifier.verify_paired_stability_diagnostic(
        artifact_path=canonical_artifact_path,
        repo_root=ROOT,
    )
    for receipt in (remote, local):
        receipt.pop("verified_at_utc", None)
        receipt.pop("verification_location", None)
    assert remote == local


def test_manifest_must_cover_every_bundle_file(
    canonical_artifact_path,
    valid_manifest_path,
):
    verifier = load_module(VERIFIER_PATH, "paired_manifest")
    receipt = verifier.verify_paired_stability_diagnostic(
        artifact_path=canonical_artifact_path,
        repo_root=ROOT,
        manifest_path=valid_manifest_path,
    )
    assert receipt["manifest_verified"] is True
    assert receipt["manifest_sha256"]


def test_manifest_rejects_unlisted_authoritative_file(
    valid_bundle_directory,
    canonical_artifact_path,
    valid_manifest_path,
):
    verifier = load_module(VERIFIER_PATH, "paired_manifest_extra")
    (valid_bundle_directory / "unlisted.json").write_text(
        "{}\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="manifest inventory"):
        verifier.verify_paired_stability_diagnostic(
            artifact_path=canonical_artifact_path,
            repo_root=ROOT,
            manifest_path=valid_manifest_path,
        )
