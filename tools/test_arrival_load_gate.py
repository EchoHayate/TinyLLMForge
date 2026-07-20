"""Dependency-light tests for the production arrival-load gate."""

from __future__ import annotations

from collections import Counter
import hashlib
import importlib.util
import json
import math
import shutil
import subprocess
import tarfile
import tempfile
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
GATE_PATH = REPO_ROOT / "tools" / "arrival_load_gate.py"


def _load_gate():
    spec = importlib.util.spec_from_file_location(
        "arrival_load_gate",
        GATE_PATH,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load arrival_load_gate")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


gate = _load_gate()

ADAPTIVE_DEFAULTS = {
    "chunked_prefill_decode_first": True,
    "chunked_prefill_max_consecutive_chunks": 0,
    "chunked_prefill_mixed_batch": False,
    "chunked_prefill_mixed_min_prompt_tokens": 0,
    "chunked_prefill_adaptive_mixed": False,
    "chunked_prefill_adaptive_enter_waiting": 8,
    "chunked_prefill_adaptive_exit_waiting": 2,
    "chunked_prefill_adaptive_transition_steps": 2,
    "chunked_prefill_adaptive_max_mixed_steps": 2,
    "chunked_prefill_slo_mixed": False,
    "chunked_prefill_slo_target_gap_ns": 0,
    "chunked_prefill_slo_reserve_ns": 0,
    "chunked_prefill_slo_cost_intercept_ns": 0,
    "chunked_prefill_slo_cost_per_prefill_token_ns": 0,
    "chunked_prefill_slo_min_chunk_tokens": 16,
}
P5_COST_ENVELOPE = {
    "artifact_sha256": "a" * 64,
    "cost_intercept_ns": 4_000_000,
    "cost_per_prefill_token_ns": 100_000,
}


class FakeTokenizer:
    def encode(self, prompt: str) -> list[int]:
        return list(range(len(prompt.split())))


def _prompt_bank() -> dict:
    prompts = []
    for prompt_class, token_count in (
        ("short", 64),
        ("medium", 512),
        ("long", 1536),
    ):
        token_ids = list(range(token_count))
        prompt = " ".join(f"t{token_id}" for token_id in token_ids)
        prompts.append({
            "prompt_id": f"{prompt_class}-0",
            "prompt": prompt,
            "prompt_token_ids": token_ids,
            "prompt_token_count": token_count,
            "prompt_class": prompt_class,
            "prompt_sha256": gate.canonical_json_sha256({
                "prompt": prompt,
                "prompt_token_ids": token_ids,
            }),
        })
    bank = {
        "schema_version": gate.SCHEMA_VERSION,
        "model_id": "fake-model",
        "prompts": sorted(
            prompts,
            key=lambda record: record["prompt_id"],
        ),
    }
    bank["prompt_bank_sha256"] = gate.canonical_json_sha256(bank)
    return bank


def test_seeded_steady_and_burst_workloads_are_byte_stable():
    first = gate.build_canonical_workload(
        lambda_ref=4.0,
        prompt_bank=_prompt_bank(),
    )
    second = gate.build_canonical_workload(
        lambda_ref=4.0,
        prompt_bank=_prompt_bank(),
    )
    assert first == second
    assert gate.canonical_json_sha256(first) == (
        gate.canonical_json_sha256(second)
    )
    scenario_order = {
        name: index
        for index, name in enumerate(gate.CANONICAL_SCENARIOS)
    }
    assert first == sorted(
        first,
        key=lambda row: (
            scenario_order[row["scenario"]],
            row["arrival_offset_ns"],
            row["request_id"],
        ),
    )
    burst = [
        row for row in first if row["scenario"] == "burst"
    ]
    assert len(burst) == 64 + gate.CANONICAL_WARMUP_REQUESTS
    assert max(row["arrival_offset_ns"] for row in burst) > 6_000_000_000


def test_built_prompt_bank_is_sorted_hashed_and_valid():
    bank = gate.build_prompt_bank(
        FakeTokenizer(),
        model_id="fake-model",
    )
    gate.validate_prompt_bank(bank)
    assert [row["prompt_id"] for row in bank["prompts"]] == sorted(
        row["prompt_id"] for row in bank["prompts"]
    )
    assert {
        row["prompt_token_count"]
        for row in bank["prompts"]
    } == {64, 512, 1536}


def test_service_buckets_are_fixed_before_execution():
    rows = gate.build_canonical_workload(
        lambda_ref=4.0,
        prompt_bank=_prompt_bank(),
    )
    fairness = [
        row for row in rows
        if row["scenario"] == "mixed_service_fairness"
        and not row["warmup"]
    ]
    counts = Counter(row["service_time_bucket"] for row in fairness)
    assert set(counts) == {
        "short__short",
        "short__long",
        "medium__short",
        "medium__long",
        "long__short",
        "long__long",
    }
    assert set(counts.values()) == {
        gate.FAIRNESS_REQUESTS_PER_BUCKET
    }


def test_p5_policy_contract_is_frozen_and_source_bound():
    contract = gate._resolved_policy_contract(
        cost_envelope=P5_COST_ENVELOPE,
    )
    assert tuple(contract["canonical_policy_by_name"]) == (
        "P0",
        "P4",
        "P5",
    )
    resolved = contract["resolved_policy_config_by_name"]
    assert resolved["P4"] == {
        **gate.COMMON_ENGINE_CONFIG,
        **ADAPTIVE_DEFAULTS,
        "chunked_prefill_decode_first": True,
        "chunked_prefill_max_consecutive_chunks": 0,
        "chunked_prefill_mixed_batch": False,
        "chunked_prefill_mixed_min_prompt_tokens": 0,
        "chunked_prefill_adaptive_mixed": True,
        "chunked_prefill_adaptive_enter_waiting": 8,
        "chunked_prefill_adaptive_exit_waiting": 2,
        "chunked_prefill_adaptive_transition_steps": 2,
        "chunked_prefill_adaptive_max_mixed_steps": 2,
    }
    p5 = resolved["P5"]
    assert p5["chunked_prefill_slo_mixed"] is True
    assert p5["chunked_prefill_slo_target_gap_ns"] == 64_000_000
    assert p5["chunked_prefill_slo_reserve_ns"] == 8_000_000
    assert p5["chunked_prefill_slo_min_chunk_tokens"] == 16
    assert p5["max_num_prefill_tokens_per_step"] == 128
    assert p5["chunked_prefill_slo_cost_intercept_ns"] == 4_000_000
    assert p5[
        "chunked_prefill_slo_cost_per_prefill_token_ns"
    ] == 100_000
    assert p5["chunked_prefill_slo_token_ladder"] == [
        128, 112, 96, 80, 64, 48, 32, 16,
    ]
    assert p5["cost_calibration_artifact_sha256"] == "a" * 64
    assert len({
        gate.policy_identity(resolved[name])
        for name in ("P0", "P4", "P5")
    }) == 3


def test_nearest_rank_boundaries():
    assert gate.nearest_rank([7.0], 0.50) == 7.0
    assert gate.nearest_rank([1.0, 9.0], 0.50) == 1.0
    assert gate.nearest_rank([1.0, 9.0], 0.95) == 9.0
    values = list(range(1, 21))
    assert gate.nearest_rank(values, 0.50) == 10
    assert gate.nearest_rank(values, 0.95) == 19
    assert gate.nearest_rank(values, 0.99) == 20


def test_canonical_rates_and_sampling_contract_are_frozen():
    rows = gate.build_canonical_workload(
        lambda_ref=10.0,
        prompt_bank=_prompt_bank(),
    )
    rates = {}
    for row in rows:
        rates.setdefault(row["scenario"], row["requested_rate_rps"])
        assert row["sampling"] == {
            "temperature": 0.0,
            "ignore_eos": True,
            "max_tokens": row["requested_output_tokens"],
        }
    assert rates["steady_moderate"] == 6.0
    assert rates["near_saturation"] == 9.0
    assert rates["overload"] == 12.0


def test_invalid_lambda_and_policy_fail_closed():
    for invalid in (-1.0, 0.0, float("inf"), float("nan")):
        try:
            gate.build_canonical_workload(
                lambda_ref=invalid,
                prompt_bank=_prompt_bank(),
            )
        except ValueError as exc:
            assert "lambda_ref" in str(exc)
        else:
            raise AssertionError(f"invalid lambda_ref accepted: {invalid}")

    try:
        gate.resolve_policy_config("P9", {})
    except ValueError as exc:
        assert "unknown policy" in str(exc)
    else:
        raise AssertionError("unknown policy accepted")


def test_unexpected_candidate_policy_collision_is_rejected():
    resolved = gate._resolved_policy_contract(
        cost_envelope=P5_COST_ENVELOPE,
    )["resolved_policy_config_by_name"]
    resolved["P5"] = dict(resolved["P4"])
    try:
        gate.deduplicate_policies(resolved)
    except ValueError as exc:
        assert "unexpected policy identity collision" in str(exc)
    else:
        raise AssertionError("candidate collision must fail")


def test_prompt_bank_hash_detects_drift():
    bank = _prompt_bank()
    gate.validate_prompt_bank(bank)
    bank["prompts"][0]["prompt"] += " changed"
    try:
        gate.validate_prompt_bank(bank)
    except ValueError as exc:
        assert "prompt hash mismatch" in str(exc)
    else:
        raise AssertionError("prompt drift must fail")


def test_calibration_manifest_is_deterministic_and_p0_only():
    first = gate.build_calibration_manifest(_prompt_bank())
    second = gate.build_calibration_manifest(_prompt_bank())
    assert first == second
    assert all(row["policy"] == "P0" for row in first)
    rates = [row["requested_rate_rps"] for row in first]
    assert rates == sorted(rates)
    assert len(rates) == gate.CALIBRATION_MAX_DOUBLINGS + 1
    assert math.isclose(
        rates[-1],
        gate.CALIBRATION_INITIAL_RATE_RPS
        * (2 ** gate.CALIBRATION_MAX_DOUBLINGS),
    )


def _calibration_row(
    rate: float,
    throughput: float,
    *,
    slope: float,
    complete: bool = True,
    exact: bool = True,
    finite: bool = True,
) -> dict:
    samples = []
    for index in range(9):
        relative_time_s = float(index)
        unfinished_count = 4.0 + slope * relative_time_s
        samples.append({
            "relative_time_s": relative_time_s,
            "unfinished_count": unfinished_count,
        })
    return {
        "calibration_id": f"rate-{rate}",
        "offered_rate_rps": rate,
        "completed_request_throughput_rps": throughput,
        "complete_requests": complete,
        "exact_outputs": exact,
        "finite_metrics": finite,
        "backlog_samples": samples,
        "offered_window_duration_s": 8.0,
    }


def test_select_lambda_ref_excludes_post_arrival_drain_from_slope():
    overloaded = _calibration_row(4.0, 1.0, slope=0.20)
    overloaded["backlog_samples"].append({
        "relative_time_s": 100.0,
        "unfinished_count": 0.0,
    })

    selected = gate.select_lambda_ref([
        _calibration_row(1.0, 1.0, slope=0.0),
        overloaded,
    ])

    assert selected["status"] == "PASS"
    assert selected["lambda_ref"] == 1.0
    assert selected["ceiling_rate_rps"] == 4.0
    assert selected["evaluated_rows"][1]["stable"] is False


def test_reconstruct_calibration_backlog_samples_uses_offered_window():
    samples = gate.reconstruct_calibration_backlog_samples([
        {
            "scheduled_arrival_ns": 0,
            "completion_ns": 10_000_000_000,
        },
        {
            "scheduled_arrival_ns": 1_000_000_000,
            "completion_ns": 10_000_000_000,
        },
        {
            "scheduled_arrival_ns": 2_000_000_000,
            "completion_ns": 10_000_000_000,
        },
    ], sample_count=5)

    assert samples == [
        {"relative_time_s": 0.0, "unfinished_count": 1},
        {"relative_time_s": 0.5, "unfinished_count": 1},
        {"relative_time_s": 1.0, "unfinished_count": 2},
        {"relative_time_s": 1.5, "unfinished_count": 2},
        {"relative_time_s": 2.0, "unfinished_count": 3},
    ]


def test_select_lambda_ref_recomputes_tail_slope_and_uses_95_percent_ceiling():
    rows = [
        _calibration_row(1.0, 1.0, slope=0.02),
        _calibration_row(2.0, 1.94, slope=0.02),
        _calibration_row(3.0, 2.0, slope=0.05),
        _calibration_row(4.0, 2.0, slope=0.20),
    ]

    selected = gate.select_lambda_ref(rows)

    assert selected["status"] == "PASS"
    assert selected["lambda_ref"] == 3.0
    assert selected["maximum_stable_throughput_rps"] == 2.0
    assert selected["ceiling_rate_rps"] == 4.0
    by_rate = {
        row["offered_rate_rps"]: row
        for row in selected["evaluated_rows"]
    }
    assert math.isclose(
        by_rate[2.0]["backlog_slope_rps"],
        0.02,
    )
    assert by_rate[1.0]["stable"] is True
    assert by_rate[3.0]["backlog_slope_threshold_rps"] == 0.06
    assert by_rate[3.0]["stable"] is True
    assert by_rate[4.0]["stable"] is False


def test_select_lambda_ref_requires_stable_point_and_higher_ceiling():
    no_stable = gate.select_lambda_ref([
        _calibration_row(1.0, 1.0, slope=0.5),
        _calibration_row(2.0, 1.5, slope=0.5),
    ])
    assert no_stable["status"] == "INCOMPLETE"
    assert no_stable["error_type"] == "no_stable_point"

    no_ceiling = gate.select_lambda_ref([
        _calibration_row(1.0, 1.0, slope=0.0),
        _calibration_row(2.0, 1.9, slope=0.0),
        _calibration_row(3.0, 2.0, slope=0.0),
    ])
    assert no_ceiling["status"] == "INCOMPLETE"
    assert no_ceiling["error_type"] == "no_clear_ceiling"


def test_select_lambda_ref_rejects_structural_or_nonfinite_rows():
    selected = gate.select_lambda_ref([
        _calibration_row(1.0, 1.0, slope=0.0),
        _calibration_row(
            2.0,
            2.0,
            slope=0.0,
            exact=False,
        ),
    ])
    assert selected["status"] == "PASS"
    assert selected["lambda_ref"] == 1.0
    assert selected["evaluated_rows"][1]["stable"] is False

    malformed = gate.select_lambda_ref([
        {
            **_calibration_row(1.0, 1.0, slope=0.0),
            "completed_request_throughput_rps": float("nan"),
        },
        _calibration_row(2.0, 2.0, slope=0.5),
    ])
    assert malformed["status"] == "INCOMPLETE"
    assert malformed["error_type"] == "no_stable_point"


def _case_matrix_manifest() -> dict:
    contract = gate._resolved_policy_contract(
        cost_envelope=P5_COST_ENVELOPE,
    )
    return {
        "run_tag": "arrival-test",
        "required_scenarios": list(gate.CANONICAL_SCENARIOS),
        "measured_repetitions": 3,
        **contract,
        "workload_sha256": "workload-hash",
        "source_tree_sha256": "source-hash",
        "environment_sha256": "environment-hash",
        "drain_timeout_ns": 123,
    }


def test_build_case_matrix_has_exact_interleaved_non_alias_cases():
    matrix = gate.build_case_matrix(_case_matrix_manifest())

    assert len(matrix) == 54
    keys = [
        (
            row["policy"],
            row["scenario"],
            row["repetition"],
        )
        for row in matrix
    ]
    assert len(keys) == len(set(keys))
    for repetition, expected_policy_order in (
        gate.POLICY_ORDER_BY_REPETITION.items()
    ):
        rows = [
            row for row in matrix
            if row["repetition"] == repetition
        ]
        for scenario_index, scenario in enumerate(
            gate.CANONICAL_SCENARIOS
        ):
            start = scenario_index * len(expected_policy_order)
            scenario_rows = rows[
                start:start + len(expected_policy_order)
            ]
            assert [
                row["scenario"] for row in scenario_rows
            ] == [scenario] * 3
            assert [
                row["policy"] for row in scenario_rows
            ] == list(expected_policy_order)
    assert {
        repetition: tuple(
            row["policy"]
            for row in matrix
            if row["repetition"] == repetition
            and row["scenario"] == gate.CANONICAL_SCENARIOS[0]
        )
        for repetition in range(3)
    } == {
        0: ("P0", "P4", "P5"),
        1: ("P4", "P5", "P0"),
        2: ("P5", "P0", "P4"),
    }
    assert {row["policy"] for row in matrix} == {"P0", "P4", "P5"}
    assert all(row["policy"] != "P3" for row in matrix)
    assert len({row["case_id"] for row in matrix}) == 54
    assert all(
        row["workload_sha256"] == "workload-hash"
        and row["source_tree_sha256"] == "source-hash"
        and row["environment_sha256"] == "environment-hash"
        for row in matrix
    )


def test_build_case_matrix_rejects_bad_alias_or_repetition_contract():
    bad_alias = _case_matrix_manifest()
    bad_alias["canonical_policy_by_name"]["P4"] = "P0"
    try:
        gate.build_case_matrix(bad_alias)
    except ValueError as exc:
        assert "canonical policy" in str(exc)
    else:
        raise AssertionError("unexpected candidate alias accepted")

    bad_repetitions = _case_matrix_manifest()
    bad_repetitions["measured_repetitions"] = 2
    try:
        gate.build_case_matrix(bad_repetitions)
    except ValueError as exc:
        assert "repetitions" in str(exc)
    else:
        raise AssertionError("too few repetitions accepted")


def _write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, sort_keys=True) + "\n"
    )


def _write_jsonl(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(row, sort_keys=True) + "\n"
            for row in rows
        )
    )


def _canonical_run_fixture(root: Path):
    manifest = _case_matrix_manifest()
    manifest["smoke_verification"] = {
        "status": "PASS",
        "source_tree_sha256": manifest["source_tree_sha256"],
        "environment_sha256": manifest["environment_sha256"],
    }
    _write_json(root / "run_manifest.json", manifest)
    workload = []
    for scenario in gate.CANONICAL_SCENARIOS:
        workload.append({
            "request_id": f"{scenario}-request",
            "scenario": scenario,
            "arrival_offset_ns": 0,
            "prompt_token_ids": [1, 2, 3],
            "prompt_token_count": 3,
            "requested_output_tokens": 1,
            "sampling": {
                "temperature": 0.0,
                "ignore_eos": True,
                "max_tokens": 1,
            },
        })
    _write_jsonl(root / "workload_manifest.jsonl", workload)
    return manifest


def test_allocate_port_pair_returns_distinct_ephemeral_ports():
    first = gate.allocate_port_pair()
    second = gate.allocate_port_pair()
    assert first[0] != first[1]
    assert second[0] != second[1]
    assert all(0 < port < 65536 for port in (*first, *second))


def test_run_canonical_uses_unique_ports_and_resume_is_immutable():
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        manifest = _canonical_run_fixture(root)
        launched = []
        next_ports = iter(
            (20_000 + index * 2, 20_001 + index * 2)
            for index in range(54)
        )
        original_ports = gate.allocate_port_pair
        original_run = gate.subprocess.run

        def fake_run(command, **kwargs):
            launched.append({
                "command": list(command),
                "env": dict(kwargs["env"]),
            })
            output_dir = Path(
                command[command.index("--output-dir") + 1]
            )
            case_spec = json.loads(
                Path(
                    command[command.index("--case-spec") + 1]
                ).read_text()
            )
            _write_json(output_dir / "case_result.json", {
                "case_id": case_spec["case_id"],
                "status": "PASS",
            })
            (output_dir / "exitcode").write_text("0\n")
            for filename in (
                "request_timeline.jsonl",
                "scheduler_trace.jsonl",
                "memory_trace.jsonl",
            ):
                (output_dir / filename).write_text("{}\n")
            return subprocess.CompletedProcess(
                command,
                0,
                stdout="driver stdout\n",
                stderr="",
            )

        gate.allocate_port_pair = lambda: next(next_ports)
        gate.subprocess.run = fake_run
        try:
            first = gate.run_canonical(
                run_dir=root,
                python_bin="/fake/python",
                model_path="/fake/model",
                run_manifest=manifest,
                resume=False,
            )
            assert first["status"] == "PASS"
            assert len(launched) == 54
            pairs = {
                (
                    int(call["env"]["TINYVLLM_DIST_PORT"]),
                    int(call["env"]["MASTER_PORT"]),
                )
                for call in launched
            }
            assert len(pairs) == 54
            before = {
                path: path.read_bytes()
                for path in (root / "processes").glob(
                    "*/process.json"
                )
            }
            second = gate.run_canonical(
                run_dir=root,
                python_bin="/fake/python",
                model_path="/fake/model",
                run_manifest=manifest,
                resume=True,
            )
            assert second["status"] == "PASS"
            assert len(launched) == 54
            assert before == {
                path: path.read_bytes()
                for path in (root / "processes").glob(
                    "*/process.json"
                )
            }
        finally:
            gate.allocate_port_pair = original_ports
            gate.subprocess.run = original_run


def test_run_canonical_replaces_incomplete_case_and_rejects_identity_drift():
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        manifest = _canonical_run_fixture(root)
        matrix = gate.build_case_matrix(manifest)
        failed_dir = root / "processes" / matrix[0]["case_id"]
        failed_dir.mkdir(parents=True)
        _write_json(failed_dir / "case_result.json", {
            "status": "INCOMPLETE",
        })
        (failed_dir / "sentinel").write_text("preserve me")

        original_time_ns = gate.time.time_ns
        original_ports = gate.allocate_port_pair
        original_run = gate.subprocess.run
        ports = iter(
            (30_000 + index * 2, 30_001 + index * 2)
            for index in range(54)
        )

        def fake_run(command, **kwargs):
            output_dir = Path(
                command[command.index("--output-dir") + 1]
            )
            case_spec = json.loads(
                Path(
                    command[command.index("--case-spec") + 1]
                ).read_text()
            )
            _write_json(output_dir / "case_result.json", {
                "case_id": case_spec["case_id"],
                "status": "PASS",
            })
            (output_dir / "exitcode").write_text("0\n")
            return subprocess.CompletedProcess(
                command, 0, stdout="", stderr=""
            )

        gate.time.time_ns = lambda: 123456789
        gate.allocate_port_pair = lambda: next(ports)
        gate.subprocess.run = fake_run
        try:
            result = gate.run_canonical(
                run_dir=root,
                python_bin="/fake/python",
                model_path="/fake/model",
                run_manifest=manifest,
                resume=True,
            )
            assert result["status"] == "PASS"
            replaced = (
                root
                / "processes"
                / f"{matrix[0]['case_id']}.replaced.123456789"
            )
            assert (replaced / "sentinel").read_text() == "preserve me"

            drifted = dict(manifest)
            drifted["source_tree_sha256"] = "changed-source"
            try:
                gate.run_canonical(
                    run_dir=root,
                    python_bin="/fake/python",
                    model_path="/fake/model",
                    run_manifest=drifted,
                    resume=True,
                )
            except ValueError as exc:
                assert "resume identity" in str(exc)
            else:
                raise AssertionError("identity drift accepted")

            threshold_drifted = {
                **manifest,
                "resolved_policy_config_by_name": {
                    name: dict(config)
                    for name, config in manifest[
                        "resolved_policy_config_by_name"
                    ].items()
                },
            }
            threshold_drifted[
                "resolved_policy_config_by_name"
            ]["P4"]["chunked_prefill_adaptive_enter_waiting"] = 9
            try:
                gate.run_canonical(
                    run_dir=root,
                    python_bin="/fake/python",
                    model_path="/fake/model",
                    run_manifest=threshold_drifted,
                    resume=True,
                )
            except ValueError as exc:
                assert "resume identity" in str(exc)
            else:
                raise AssertionError("P4 threshold drift accepted")
        finally:
            gate.time.time_ns = original_time_ns
            gate.allocate_port_pair = original_ports
            gate.subprocess.run = original_run


def test_run_canonical_requires_matching_smoke_and_frozen_calibration():
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        manifest = _canonical_run_fixture(root)
        missing_smoke = dict(manifest)
        missing_smoke.pop("smoke_verification")
        try:
            gate.run_canonical(
                run_dir=root,
                python_bin="/fake/python",
                model_path="/fake/model",
                run_manifest=missing_smoke,
                resume=False,
            )
        except ValueError as exc:
            assert "smoke" in str(exc)
        else:
            raise AssertionError("canonical started without smoke")

        (root / "case_rows.jsonl").write_text("{}\n")
        try:
            gate.run_calibration(
                run_dir=root,
                python_bin="/fake/python",
                model_path="/fake/model",
                run_manifest=manifest,
                resume=True,
            )
        except ValueError as exc:
            assert "canonical rows" in str(exc)
        else:
            raise AssertionError("calibration changed after canonical")


def test_predecessor_identity_binds_source_environment_and_p4():
    current = _case_matrix_manifest()
    predecessor = json.loads(json.dumps(current))
    gate._validate_predecessor_identity(
        current,
        predecessor,
        "smoke",
    )

    for field, expected_error in (
        ("source_tree_sha256", "smoke source_tree_sha256"),
        ("environment_sha256", "smoke environment_sha256"),
    ):
        drifted = json.loads(json.dumps(predecessor))
        drifted[field] = f"changed-{field}"
        try:
            gate._validate_predecessor_identity(
                current,
                drifted,
                "smoke",
            )
        except ValueError as exc:
            assert expected_error in str(exc)
        else:
            raise AssertionError(f"{field} drift accepted")

    drifted = json.loads(json.dumps(predecessor))
    drifted["policy_identity_by_name"]["P4"] = "changed-p4"
    try:
        gate._validate_predecessor_identity(
            current,
            drifted,
            "calibration",
        )
    except ValueError as exc:
        assert "calibration P4 policy identity" in str(exc)
    else:
        raise AssertionError("P4 predecessor drift accepted")


def test_resolved_policy_contract_is_recomputed_from_current_source():
    contract = gate._resolved_policy_contract(
        cost_envelope=P5_COST_ENVELOPE,
    )
    assert set(contract) == {
        "resolved_policy_config_by_name",
        "policy_identity_by_name",
        "canonical_policy_by_name",
    }
    assert contract["resolved_policy_config_by_name"]["P4"] == (
        gate.resolve_policy_config("P4", ADAPTIVE_DEFAULTS)
    )
    assert contract["policy_identity_by_name"]["P4"] == (
        gate.policy_identity(
            contract["resolved_policy_config_by_name"]["P4"]
        )
    )


def test_run_canonical_remote_freezes_predecessor_before_launch():
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        run_dir = root / "canonical"
        smoke_dir = root / "smoke"
        cost_dir = root / "cost"
        workload_dir = root / "workload"
        smoke_dir.mkdir()
        cost_dir.mkdir()
        workload_dir.mkdir()

        workload_manifest = _case_matrix_manifest()
        workload_manifest["run_tag"] = "workload-tag"
        workload_manifest["run_type"] = "workload_calibration"
        workload_manifest["smoke_verification"] = {
            "status": "PASS",
            "run_tag": "smoke-tag",
            "source_tree_sha256": workload_manifest[
                "source_tree_sha256"
            ],
            "environment_sha256": workload_manifest[
                "environment_sha256"
            ],
        }
        workload = [{
            "request_id": "frozen-request",
            "scenario": "steady_moderate",
        }]
        workload_manifest["workload_sha256"] = (
            gate.canonical_json_sha256(workload)
        )
        workload_manifest["calibration"] = {
            "status": "PASS",
            "lambda_ref_rps": 1.0,
        }
        _write_jsonl(
            workload_dir / "workload_manifest.jsonl",
            workload,
        )
        _write_jsonl(
            workload_dir / "calibration_manifest.jsonl",
            [{"offered_rate_rps": 1.0}],
        )
        _write_jsonl(
            workload_dir / "calibration_rows.jsonl",
            [{"offered_rate_rps": 1.0, "stable": True}],
        )
        _write_json(
            workload_dir / "prompt_bank.json",
            _prompt_bank(),
        )
        cost_summary = {
            "status": "PASS",
            "purpose": "authoritative",
            "source_tree_sha256": workload_manifest[
                "source_tree_sha256"
            ],
            "environment_sha256": workload_manifest[
                "environment_sha256"
            ],
            "engine_config_sha256": "a" * 64,
            "required_shape_sha256": "b" * 64,
            "raw_rows_sha256": "c" * 64,
            "cost_intercept_ns": P5_COST_ENVELOPE[
                "cost_intercept_ns"
            ],
            "cost_per_prefill_token_ns": P5_COST_ENVELOPE[
                "cost_per_prefill_token_ns"
            ],
            "envelope_sha256": "d" * 64,
        }
        _write_json(
            cost_dir / "cost_calibration_summary.json",
            cost_summary,
        )
        cost_artifact_sha256 = gate.sha256_file(
            cost_dir / "cost_calibration_summary.json"
        )
        cost_envelope = {
            **P5_COST_ENVELOPE,
            "artifact_sha256": cost_artifact_sha256,
        }
        cost_contract = gate._resolved_policy_contract(
            cost_envelope=cost_envelope,
        )
        cost_manifest = {
            "run_tag": "cost-tag",
            "run_type": "cost_calibration",
            "purpose": "authoritative",
            "source_tree_sha256": workload_manifest[
                "source_tree_sha256"
            ],
            "environment_sha256": workload_manifest[
                "environment_sha256"
            ],
            "cost_calibration_artifact_sha256":
                cost_artifact_sha256,
            **cost_contract,
        }
        _write_json(cost_dir / "run_manifest.json", cost_manifest)
        _write_json(
            cost_dir / "cost_calibration_capacity.json",
            {"num_kvcache_blocks": 448, "block_size": 256},
        )
        _write_jsonl(
            cost_dir / "cost_calibration_manifest.jsonl",
            [{"shape_id": "shape"}],
        )
        _write_jsonl(
            cost_dir / "cost_calibration_rows.jsonl",
            [{"shape_id": "shape", "duration_ns": 1}],
        )
        workload_manifest.update(cost_contract)
        workload_manifest["cost_calibration_verification"] = {
            "status": "PASS",
            "run_tag": "cost-tag",
            "artifact_sha256": cost_artifact_sha256,
            "source_tree_sha256": workload_manifest[
                "source_tree_sha256"
            ],
            "environment_sha256": workload_manifest[
                "environment_sha256"
            ],
        }
        _write_json(
            workload_dir / "run_manifest.json",
            workload_manifest,
        )
        _write_json(
            smoke_dir / "run_manifest.json",
            {
                **workload_manifest,
                "run_tag": "smoke-tag",
            },
        )
        _write_json(
            smoke_dir / "summary.json",
            {
                "classification": "SMOKE_ONLY",
                "lifecycle_complete": True,
                "exact_outputs": True,
            },
        )
        _write_json(
            root / "source_evidence.json",
            {
                "tree_sha256": workload_manifest[
                    "source_tree_sha256"
                ],
            },
        )
        environment = {
            "gpu": "fake-gpu",
            "torch": "fake-torch",
        }
        workload_manifest["environment_sha256"] = (
            gate.environment_identity_sha256(environment)
        )
        workload_manifest["smoke_verification"][
            "environment_sha256"
        ] = workload_manifest["environment_sha256"]
        workload_manifest["cost_calibration_verification"][
            "environment_sha256"
        ] = workload_manifest["environment_sha256"]
        cost_manifest["environment_sha256"] = workload_manifest[
            "environment_sha256"
        ]
        cost_summary["environment_sha256"] = workload_manifest[
            "environment_sha256"
        ]
        _write_json(
            cost_dir / "cost_calibration_summary.json",
            cost_summary,
        )
        cost_artifact_sha256 = gate.sha256_file(
            cost_dir / "cost_calibration_summary.json"
        )
        cost_manifest["cost_calibration_artifact_sha256"] = (
            cost_artifact_sha256
        )
        workload_manifest["cost_calibration_verification"][
            "artifact_sha256"
        ] = cost_artifact_sha256
        final_envelope = {
            **P5_COST_ENVELOPE,
            "artifact_sha256": cost_artifact_sha256,
        }
        final_contract = gate._resolved_policy_contract(
            cost_envelope=final_envelope,
        )
        cost_manifest.update(final_contract)
        workload_manifest.update(final_contract)
        smoke_manifest = json.loads(
            (smoke_dir / "run_manifest.json").read_text()
        )
        smoke_manifest["environment_sha256"] = workload_manifest[
            "environment_sha256"
        ]
        smoke_manifest["smoke_verification"][
            "environment_sha256"
        ] = workload_manifest["environment_sha256"]
        _write_json(smoke_dir / "run_manifest.json", smoke_manifest)
        _write_json(
            cost_dir / "run_manifest.json",
            cost_manifest,
        )
        _write_json(
            workload_dir / "run_manifest.json",
            workload_manifest,
        )
        _write_json(root / "capability.json", environment)

        original_run_canonical = gate.run_canonical
        captured = {}

        def fake_run_canonical(**kwargs):
            captured.update(kwargs)
            return {"status": "PASS", "case_count": 54}

        gate.run_canonical = fake_run_canonical
        try:
            result = gate.run_canonical_remote(
                run_dir=run_dir,
                python_bin="/fake/python",
                model_path="/fake/model",
                run_tag="canonical-tag",
                source_evidence_path=root / "source_evidence.json",
                environment_evidence_path=root / "capability.json",
                smoke_run_dir=smoke_dir,
                cost_calibration_run_dir=cost_dir,
                workload_calibration_run_dir=workload_dir,
            )
        finally:
            gate.run_canonical = original_run_canonical

        assert result == {"status": "PASS", "case_count": 54}
        current = captured["run_manifest"]
        assert current["run_tag"] == "canonical-tag"
        assert current["workload_sha256"] == (
            gate.canonical_json_sha256(workload)
        )
        assert json.loads(
            (run_dir / "run_manifest.json").read_text()
        ) == current
        assert (run_dir / "workload_manifest.jsonl").read_bytes() == (
            workload_dir / "workload_manifest.jsonl"
        ).read_bytes()
        for filename in gate.COST_CALIBRATION_ARTIFACT_FILES:
            assert (run_dir / filename).read_bytes() == (
                cost_dir / filename
            ).read_bytes()
        assert captured["resume"] is False


def test_run_canonical_remote_rejects_tampered_frozen_workload():
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        smoke_dir = root / "smoke"
        cost_dir = root / "cost"
        workload_dir = root / "workload"
        smoke_dir.mkdir()
        cost_dir.mkdir()
        workload_dir.mkdir()
        contract = gate._resolved_policy_contract(
            cost_envelope=P5_COST_ENVELOPE,
        )
        environment = {"gpu": "fake-gpu", "torch": "fake-torch"}
        environment_sha256 = gate.environment_identity_sha256(
            environment
        )
        source_sha256 = "source-hash"
        smoke_manifest = {
            "run_tag": "smoke-tag",
            "source_tree_sha256": source_sha256,
            "environment_sha256": environment_sha256,
            **contract,
        }
        workload_manifest = {
            **smoke_manifest,
            "run_tag": "workload-tag",
            "run_type": "workload_calibration",
            "workload_sha256": "tampered-hash",
            "calibration": {"status": "PASS"},
            "smoke_verification": {
                "status": "PASS",
                "run_tag": "smoke-tag",
                "source_tree_sha256": source_sha256,
                "environment_sha256": environment_sha256,
            },
        }
        cost_summary = {
            "status": "PASS",
            "purpose": "authoritative",
            "source_tree_sha256": source_sha256,
            "environment_sha256": environment_sha256,
            "engine_config_sha256": "a" * 64,
            "required_shape_sha256": "b" * 64,
            "raw_rows_sha256": "c" * 64,
            "cost_intercept_ns": P5_COST_ENVELOPE[
                "cost_intercept_ns"
            ],
            "cost_per_prefill_token_ns": P5_COST_ENVELOPE[
                "cost_per_prefill_token_ns"
            ],
            "envelope_sha256": "d" * 64,
        }
        _write_json(
            cost_dir / "cost_calibration_summary.json",
            cost_summary,
        )
        cost_artifact_sha256 = gate.sha256_file(
            cost_dir / "cost_calibration_summary.json"
        )
        cost_manifest = {
            **smoke_manifest,
            "run_tag": "cost-tag",
            "run_type": "cost_calibration",
            "purpose": "authoritative",
            "cost_calibration_artifact_sha256":
                cost_artifact_sha256,
            **gate._resolved_policy_contract(cost_envelope={
                **P5_COST_ENVELOPE,
                "artifact_sha256": cost_artifact_sha256,
            }),
        }
        workload_manifest.update(
            gate._resolved_policy_contract(cost_envelope={
                **P5_COST_ENVELOPE,
                "artifact_sha256": cost_artifact_sha256,
            })
        )
        workload_manifest["cost_calibration_verification"] = {
            "status": "PASS",
            "run_tag": "cost-tag",
            "artifact_sha256": cost_artifact_sha256,
            "source_tree_sha256": source_sha256,
            "environment_sha256": environment_sha256,
        }
        _write_json(smoke_dir / "run_manifest.json", smoke_manifest)
        _write_json(smoke_dir / "summary.json", {
            "classification": "SMOKE_ONLY",
            "lifecycle_complete": True,
            "exact_outputs": True,
        })
        _write_json(
            cost_dir / "run_manifest.json",
            cost_manifest,
        )
        _write_json(
            cost_dir / "cost_calibration_capacity.json",
            {"num_kvcache_blocks": 448, "block_size": 256},
        )
        _write_jsonl(
            cost_dir / "cost_calibration_manifest.jsonl",
            [{"shape_id": "shape"}],
        )
        _write_jsonl(
            cost_dir / "cost_calibration_rows.jsonl",
            [{"shape_id": "shape", "duration_ns": 1}],
        )
        _write_json(
            workload_dir / "run_manifest.json",
            workload_manifest,
        )
        _write_jsonl(
            workload_dir / "workload_manifest.jsonl",
            [{"request_id": "actual-workload"}],
        )
        _write_json(root / "source_evidence.json", {
            "tree_sha256": source_sha256,
        })
        _write_json(root / "capability.json", environment)

        try:
            gate.run_canonical_remote(
                run_dir=root / "canonical",
                python_bin="/fake/python",
                model_path="/fake/model",
                run_tag="canonical-tag",
                source_evidence_path=root / "source_evidence.json",
                environment_evidence_path=root / "capability.json",
                smoke_run_dir=smoke_dir,
                cost_calibration_run_dir=cost_dir,
                workload_calibration_run_dir=workload_dir,
            )
        except ValueError as exc:
            assert "workload calibration identity" in str(exc)
        else:
            raise AssertionError("tampered calibration workload accepted")
        assert not (root / "canonical" / "processes").exists()


def test_run_calibration_doubles_bisects_and_freezes_workload():
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        manifest = _canonical_run_fixture(root)
        manifest["workload_sha256"] = None
        _write_json(root / "run_manifest.json", manifest)
        _write_json(root / "prompt_bank.json", _prompt_bank())
        launched_rates = []
        original_ports = gate.allocate_port_pair
        original_run = gate.subprocess.run
        ports = iter(
            (40_000 + index * 2, 40_001 + index * 2)
            for index in range(16)
        )

        def fake_run(command, **kwargs):
            del kwargs
            output_dir = Path(
                command[command.index("--output-dir") + 1]
            )
            case_spec = json.loads(
                Path(
                    command[command.index("--case-spec") + 1]
                ).read_text()
            )
            rate = case_spec["requested_rate_rps"]
            launched_rates.append(rate)
            workload = [
                json.loads(line)
                for line in Path(
                    command[
                        command.index("--workload-manifest") + 1
                    ]
                ).read_text().splitlines()
            ]
            stable = rate <= 2.5
            timeline = []
            for index, request in enumerate(workload):
                scheduled = 1_000_000_000 + request[
                    "arrival_offset_ns"
                ]
                completion = scheduled + 100_000_000
                timeline.append({
                    "request_id": request["request_id"],
                    "seq_id": index,
                    "scheduled_arrival_ns": scheduled,
                    "actual_arrival_ns": scheduled,
                    "first_scheduled_ns": scheduled,
                    "first_token_ns": scheduled + 50_000_000,
                    "token_timestamps_ns": [
                        completion
                    ] * request["requested_output_tokens"],
                    "completion_ns": completion,
                    "output_token_ids": list(range(
                        request["requested_output_tokens"]
                    )),
                    "finish_reason": "length",
                    "error": None,
                })
            _write_jsonl(
                output_dir / "request_timeline.jsonl",
                timeline,
            )
            samples = []
            for index in range(9):
                samples.append({
                    "step_index": index,
                    "step_end_ns": 1_000_000_000
                    + index * 1_000_000_000,
                    "queue_after": {
                        "waiting_seq_ids": list(range(
                            0 if stable else index
                        )),
                        "prefilling_seq_ids": [],
                        "running_seq_ids": [],
                    },
                })
            _write_jsonl(
                output_dir / "scheduler_trace.jsonl",
                samples,
            )
            _write_jsonl(
                output_dir / "memory_trace.jsonl",
                [{"step_index": 0}],
            )
            _write_json(output_dir / "case_result.json", {
                "case_id": case_spec["case_id"],
                "status": "PASS",
            })
            (output_dir / "exitcode").write_text("0\n")
            return subprocess.CompletedProcess(
                command, 0, stdout="", stderr=""
            )

        gate.allocate_port_pair = lambda: next(ports)
        gate.subprocess.run = fake_run
        try:
            result = gate.run_calibration(
                run_dir=root,
                python_bin="/fake/python",
                model_path="/fake/model",
                run_manifest=manifest,
                resume=False,
            )
        finally:
            gate.allocate_port_pair = original_ports
            gate.subprocess.run = original_run

        assert result["status"] == "PASS"
        assert result["lambda_ref"] > 0
        assert 4.0 in launched_rates
        assert len(launched_rates) >= 6
        assert any(
            rate not in {
                gate.CALIBRATION_INITIAL_RATE_RPS
                * (2 ** index)
                for index in range(
                    gate.CALIBRATION_MAX_DOUBLINGS + 1
                )
            }
            for rate in launched_rates
        )
        calibration_rows = [
            json.loads(line)
            for line in (
                root / "calibration_rows.jsonl"
            ).read_text().splitlines()
        ]
        assert len(calibration_rows) == len(launched_rates)
        frozen = [
            json.loads(line)
            for line in (
                root / "workload_manifest.jsonl"
            ).read_text().splitlines()
        ]
        assert frozen
        stored = json.loads(
            (root / "run_manifest.json").read_text()
        )
        assert stored["calibration"]["status"] == "PASS"
        assert stored["workload_sha256"] == (
            gate.canonical_json_sha256(frozen)
        )


def _run_git(repo_root: Path, *args: str) -> None:
    completed = subprocess.run(
        ["git", *args],
        cwd=repo_root,
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        raise AssertionError(completed.stderr)


def _remove_tree_with_retries(path: Path) -> None:
    for attempt in range(10):
        try:
            shutil.rmtree(path)
            return
        except FileNotFoundError:
            return
        except OSError:
            if attempt == 9:
                raise
            time.sleep(0.05)


def test_snapshot_source_stages_matching_bytes_and_archive():
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary) / "repo"
        output = Path(temporary) / "snapshot"
        root.mkdir()
        _run_git(root, "init")
        _run_git(root, "config", "user.email", "test@example.com")
        _run_git(root, "config", "user.name", "Arrival Test")
        _run_git(root, "config", "gc.auto", "0")
        _run_git(root, "config", "maintenance.auto", "false")
        for index, owned_root in enumerate(
            gate.OWNED_SOURCE_ROOTS
        ):
            path = root / owned_root
            if Path(owned_root).suffix:
                path.parent.mkdir(parents=True, exist_ok=True)
                if owned_root == "tools/source_audit.py":
                    path.write_bytes(
                        (REPO_ROOT / owned_root).read_bytes()
                    )
                else:
                    path.write_bytes(f"owned-{index}\n".encode())
            else:
                path.mkdir(parents=True, exist_ok=True)
                (path / "module.py").write_bytes(
                    f"owned-dir-{index}\n".encode()
                )
        _run_git(root, "add", *gate.OWNED_SOURCE_ROOTS)
        _run_git(root, "commit", "-m", "fixture")
        changed = root / "tools" / "arrival_load_gate.py"
        changed.write_bytes(changed.read_bytes() + b"dirty-owned\n")

        evidence = gate.snapshot_source(root, output)

        assert json.loads(
            (output / "source_evidence.json").read_text()
        ) == evidence
        assert (output / "source.patch").is_file()
        for record in evidence["files"]:
            assert (
                output / "source" / record["path"]
            ).read_bytes() == (root / record["path"]).read_bytes()
        with tarfile.open(
            output / "source_snapshot.tar.gz",
            "r:gz",
        ) as archive:
            names = sorted(
                member.name
                for member in archive.getmembers()
                if member.isfile()
            )
            assert names == [
                f"source/{record['path']}"
                for record in evidence["files"]
            ]
            for record in evidence["files"]:
                extracted = archive.extractfile(
                    f"source/{record['path']}"
                )
                assert extracted is not None
                assert extracted.read() == (
                    root / record["path"]
                ).read_bytes()
        subprocess.run(
            ["git", "maintenance", "stop"],
            cwd=root,
            text=True,
            capture_output=True,
            check=False,
        )
        _remove_tree_with_retries(root / ".git")


def test_arrival_load_artifacts_are_the_only_new_ignored_experiment_root():
    assert "experiments/arrival_load" in (
        gate.IGNORED_UNTRACKED_PREFIXES
    )
    assert "experiments" not in gate.IGNORED_UNTRACKED_PREFIXES


def _finalization_fixture(root: Path) -> dict:
    manifest = _case_matrix_manifest()
    workload_rows = []
    for scenario in gate.CANONICAL_SCENARIOS:
        workload_rows.append({
            **_workload_row(
                f"{scenario}-request",
                output_tokens=2,
            ),
            "scenario": scenario,
        })
    manifest["workload_sha256"] = gate.canonical_json_sha256(
        workload_rows
    )
    source_root = root / "source"
    source_root.mkdir()
    (source_root / "marker.txt").write_text("arrival source\n")
    source_files = [{
        "path": "marker.txt",
        "size_bytes": (source_root / "marker.txt").stat().st_size,
        "sha256": gate.sha256_file(source_root / "marker.txt"),
    }]
    manifest["source_tree_sha256"] = gate.canonical_json_sha256(
        source_files
    )
    _write_json(root / "run_manifest.json", manifest)
    _write_jsonl(root / "calibration_manifest.jsonl", [{
        "offered_rate_rps": 1.0,
    }])
    _write_jsonl(root / "calibration_rows.jsonl", [{
        "offered_rate_rps": 1.0,
        "stable": True,
    }])
    _write_jsonl(root / "cost_calibration_manifest.jsonl", [{
        "shape_id": "decode-0__prefill-0x0",
    }])
    _write_jsonl(root / "cost_calibration_rows.jsonl", [{
        "shape_id": "decode-0__prefill-0x0",
        "duration_ns": 1,
    }])
    _write_json(root / "cost_calibration_summary.json", {
        "status": "PASS",
        "purpose": "authoritative",
    })
    _write_json(root / "cost_calibration_capacity.json", {
        "num_kvcache_blocks": 448,
        "block_size": 256,
    })
    _write_jsonl(root / "workload_manifest.jsonl", workload_rows)
    _write_json(root / "source_evidence.json", {
        "schema_version": 1,
        "base_commit": "1" * 40,
        "tree_sha256": manifest["source_tree_sha256"],
        "files": source_files,
        "patch_size_bytes": 0,
        "patch_sha256": hashlib.sha256(b"").hexdigest(),
    })
    (root / "source.patch").write_bytes(b"")
    with tarfile.open(
        root / "source_snapshot.tar.gz",
        "w:gz",
    ) as archive:
        archive.add(source_root, arcname="source")

    for case_index, case_spec in enumerate(
        gate.build_case_matrix(manifest)
    ):
        case_dir = root / "processes" / case_spec["case_id"]
        workload = next(
            row for row in workload_rows
            if row["scenario"] == case_spec["scenario"]
        )
        start_ns = 1_000_000_000 + case_index * 10_000_000
        timeline = {
            **_timeline_row(
                workload["request_id"],
                [
                    start_ns + 100_000_000,
                    start_ns + 200_000_000,
                ],
                seq_id=case_index,
                scheduled_arrival_ns=start_ns,
                actual_arrival_ns=start_ns,
                first_scheduled_ns=start_ns + 10_000_000,
                completion_ns=start_ns + 300_000_000,
            ),
        }
        _write_jsonl(
            case_dir / "request_timeline.jsonl",
            [timeline],
        )
        scheduler_row = {
            "step_index": 0,
            "step_start_ns": start_ns,
            "step_end_ns": start_ns + 300_000_000,
        }
        if case_spec["policy"] == "P4":
            scheduler_row.update({
                "policy_branch": "adaptive_mixed_chunked_prefill",
                "scheduled": [{
                    "seq_id": case_index,
                    "is_decode": False,
                }],
                "queue_before": {
                    "adaptive_mixed_state": "inactive",
                    "adaptive_high_streak": 0,
                    "adaptive_low_streak": 0,
                    "adaptive_consecutive_mixed_steps": 0,
                    "waiting_seq_ids": [case_index],
                    "prefilling_seq_ids": [],
                    "running_seq_ids": [],
                },
                "queue_after": {
                    "adaptive_mixed_state": "inactive",
                    "adaptive_high_streak": 0,
                    "adaptive_low_streak": 0,
                    "adaptive_consecutive_mixed_steps": 0,
                    "waiting_seq_ids": [],
                    "prefilling_seq_ids": [],
                    "running_seq_ids": [],
                },
            })
        _write_jsonl(
            case_dir / "scheduler_trace.jsonl",
            [scheduler_row],
        )
        _write_jsonl(
            case_dir / "memory_trace.jsonl",
            [{
                "step_index": 0,
                "timestamp_ns": start_ns + 300_000_000,
                "cuda_allocated_bytes": 80,
                "cuda_reserved_bytes": 100,
                "used_kv_blocks": 5,
                "kv_block_bytes": 20,
            }],
        )
        _write_json(case_dir / "case_result.json", {
            "case_id": case_spec["case_id"],
            "status": "PASS",
        })
        _write_json(case_dir / "process.json", {
            "case_id": case_spec["case_id"],
            "returncode": 0,
            "tinyvllm_dist_port": 20_000 + case_index * 2,
            "master_port": 20_001 + case_index * 2,
        })
        (case_dir / "exitcode").write_text("0\n")
    return manifest


def test_finalize_artifacts_merges_classifies_and_hashes_deterministically():
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        manifest = _finalization_fixture(root)

        first = gate.finalize_artifacts(root)
        first_bytes = {
            path.name: path.read_bytes()
            for path in root.iterdir()
            if path.is_file()
        }
        second = gate.finalize_artifacts(root)

        assert second == first
        assert first_bytes == {
            path.name: path.read_bytes()
            for path in root.iterdir()
            if path.is_file()
        }
        assert first["classification"] == "NO_GO"
        case_rows = [
            json.loads(line)
            for line in (root / "case_rows.jsonl").read_text().splitlines()
        ]
        assert len(case_rows) == 54
        assert [row["case_id"] for row in case_rows] == [
            row["case_id"] for row in gate.build_case_matrix(manifest)
        ]
        for filename in (
            "request_timeline.jsonl",
            "scheduler_trace.jsonl",
            "memory_trace.jsonl",
        ):
            rows = [
                json.loads(line)
                for line in (root / filename).read_text().splitlines()
            ]
            assert len(rows) == 54
            assert all(
                {"case_id", "policy", "scenario", "repetition"}
                <= set(row)
                for row in rows
            )
        stored_manifest = json.loads(
            (root / "run_manifest.json").read_text()
        )
        assert stored_manifest["expected_case_ids"] == [
            row["case_id"] for row in gate.build_case_matrix(manifest)
        ]
        assert len(stored_manifest["process_port_pairs"]) == 54
        hashes = json.loads(
            (root / "artifact_hashes.json").read_text()
        )
        assert set(gate.COST_CALIBRATION_ARTIFACT_FILES) <= set(
            gate.FINAL_ARTIFACT_FILES
        )
        assert set(hashes) == set(
            gate.FINAL_ARTIFACT_FILES
        ) - {"artifact_hashes.json"}
        for filename, expected_hash in hashes.items():
            assert expected_hash == gate.sha256_file(root / filename)
        assert json.loads(
            (root / "summary.json").read_text()
        ) == first
        assert (root / "report.md").read_text().startswith(
            "# Production Arrival-Load Gate\n"
        )
def test_cli_exposes_separate_cost_and_workload_calibration_stages():
    completed = subprocess.run(
        ["python3", str(GATE_PATH), "--help"],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0
    for command in (
        "snapshot-source",
        "run-cost-calibration-remote",
        "run-workload-calibration-remote",
        "freeze-workload",
        "run-canonical",
        "finalize-artifacts",
        "verify-harness",
        "run-smoke",
    ):
        assert command in completed.stdout
    assert "run-calibration-remote" not in completed.stdout

    canonical = gate.build_parser().parse_args([
        "run-canonical",
        "--run-dir", "/tmp/canonical",
        "--python-bin", "/fake/python",
        "--model-path", "/fake/model",
        "--run-tag", "canonical",
        "--source-evidence", "/tmp/source.json",
        "--environment-evidence", "/tmp/environment.json",
        "--smoke-run-dir", "/tmp/smoke",
        "--cost-calibration-run-dir", "/tmp/cost",
        "--workload-calibration-run-dir", "/tmp/workload",
    ])
    assert canonical.cost_calibration_run_dir == Path("/tmp/cost")
    assert canonical.workload_calibration_run_dir == Path(
        "/tmp/workload"
    )
    assert not hasattr(canonical, "calibration_run_dir")


def test_predecessor_identity_binds_both_p4_and_p5():
    current = {
        "source_tree_sha256": "source",
        "environment_sha256": "environment",
        "policy_identity_by_name": {
            "P4": "p4",
            "P5": "p5",
        },
    }
    gate._validate_predecessor_identity(
        current,
        json.loads(json.dumps(current)),
        "cost calibration",
    )
    for policy in ("P4", "P5"):
        predecessor = json.loads(json.dumps(current))
        predecessor["policy_identity_by_name"][policy] += "-drift"
        try:
            gate._validate_predecessor_identity(
                current,
                predecessor,
                "cost calibration",
            )
        except ValueError as exc:
            assert policy in str(exc)
        else:
            raise AssertionError(
                f"{policy} predecessor identity drift accepted"
            )


def test_authoritative_cost_calibration_stage_writes_complete_bound_artifacts():
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        run_dir = root / "cost"
        smoke_dir = root / "smoke"
        smoke_dir.mkdir()
        source_sha256 = "1" * 64
        environment = {
            "gpu": "fake-gpu",
            "torch": "fake-torch",
        }
        environment_sha256 = gate.environment_identity_sha256(
            environment
        )
        smoke_manifest = {
            "run_tag": "smoke-tag",
            "source_tree_sha256": source_sha256,
            "environment_sha256": environment_sha256,
            **gate._resolved_policy_contract(
                cost_envelope=P5_COST_ENVELOPE,
            ),
        }
        _write_json(smoke_dir / "run_manifest.json", smoke_manifest)
        _write_json(smoke_dir / "summary.json", {
            "classification": "SMOKE_ONLY",
            "lifecycle_complete": True,
            "exact_outputs": True,
        })
        _write_json(root / "source.json", {
            "tree_sha256": source_sha256,
        })
        _write_json(root / "environment.json", environment)

        launches = []
        next_port = iter(range(20_000, 21_000))
        original_ports = gate.allocate_port_pair
        original_launch = getattr(
            gate,
            "_launch_cost_calibration_shape",
            None,
        )
        original_probe = getattr(
            gate,
            "_launch_cost_capacity_probe",
            None,
        )
        probe = {}

        def fake_probe(**kwargs):
            probe.update({
                "tinyvllm_dist_port": kwargs[
                    "tinyvllm_dist_port"
                ],
                "master_port": kwargs["master_port"],
            })
            calibration = gate._load_local_module(
                "arrival_load_cost_calibration_for_test",
                gate._REPO_ROOT
                / "tools"
                / "arrival_load_cost_calibration.py",
            )
            return calibration.build_capacity_evidence(
                base_engine_config=kwargs["engine_config"],
                num_kvcache_blocks=448,
                block_size=256,
            )

        def fake_launch(**kwargs):
            shape = kwargs["shape"]
            launches.append({
                "shape": dict(shape),
                "tinyvllm_dist_port": kwargs[
                    "tinyvllm_dist_port"
                ],
                "master_port": kwargs["master_port"],
            })
            base = (
                1_000_000
                if shape["kind"] == "decode"
                else 1_000_000
                + shape["prefill_tokens"] * 10_000
            )
            return [{
                **shape,
                "iteration": iteration,
                "duration_ns": base + iteration,
            } for iteration in range(
                shape["measured_iterations"]
            )]

        gate.allocate_port_pair = (
            lambda: (next(next_port), next(next_port))
        )
        gate._launch_cost_capacity_probe = fake_probe
        gate._launch_cost_calibration_shape = fake_launch
        try:
            result = gate.run_cost_calibration_remote(
                run_dir=run_dir,
                python_bin="/fake/python",
                model_path="/fake/model",
                run_tag="cost-tag",
                source_evidence_path=root / "source.json",
                environment_evidence_path=root / "environment.json",
                smoke_run_dir=smoke_dir,
            )
        finally:
            gate.allocate_port_pair = original_ports
            if original_launch is None:
                del gate._launch_cost_calibration_shape
            else:
                gate._launch_cost_calibration_shape = original_launch
            if original_probe is None:
                del gate._launch_cost_capacity_probe
            else:
                gate._launch_cost_capacity_probe = original_probe

        assert result["status"] == "PASS"
        assert result["purpose"] == "authoritative"
        assert len(launches) == 40
        assert len({
            port
            for launch in launches
            for port in (
                launch["tinyvllm_dist_port"],
                launch["master_port"],
            )
        }) == 80
        assert not {
            probe["tinyvllm_dist_port"],
            probe["master_port"],
        }.intersection({
            port
            for launch in launches
            for port in (
                launch["tinyvllm_dist_port"],
                launch["master_port"],
            )
        })
        assert all(
            (run_dir / filename).is_file()
            for filename in gate.COST_CALIBRATION_ARTIFACT_FILES
        )
        manifest_rows = gate._read_jsonl(
            run_dir / "cost_calibration_manifest.jsonl"
        )
        raw_rows = gate._read_jsonl(
            run_dir / "cost_calibration_rows.jsonl"
        )
        summary = gate._read_json(
            run_dir / "cost_calibration_summary.json"
        )
        run_manifest = gate._read_json(
            run_dir / "run_manifest.json"
        )
        assert len(manifest_rows) == 40
        assert len(raw_rows) == 40 * 7
        assert summary["status"] == "PASS"
        assert summary["source_tree_sha256"] == source_sha256
        assert summary["environment_sha256"] == environment_sha256
        assert len(summary["engine_config_sha256"]) == 64
        assert run_manifest["run_type"] == "cost_calibration"
        assert run_manifest["purpose"] == "authoritative"
        assert run_manifest["cost_calibration_artifact_sha256"] == (
            gate.sha256_file(
                run_dir / "cost_calibration_summary.json"
            )
        )


def test_workload_calibration_consumes_authoritative_not_provisional_envelope():
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        smoke_dir = root / "smoke"
        cost_dir = root / "cost"
        smoke_dir.mkdir()
        cost_dir.mkdir()
        source_sha256 = "2" * 64
        environment = {
            "gpu": "fake-gpu",
            "torch": "fake-torch",
        }
        environment_sha256 = gate.environment_identity_sha256(
            environment
        )
        provisional = dict(P5_COST_ENVELOPE)
        authoritative_coefficients = {
            "cost_intercept_ns": 7_000_000,
            "cost_per_prefill_token_ns": 200_000,
        }
        smoke_manifest = {
            "run_tag": "smoke-tag",
            "source_tree_sha256": source_sha256,
            "environment_sha256": environment_sha256,
            **gate._resolved_policy_contract(
                cost_envelope=provisional,
            ),
        }
        _write_json(smoke_dir / "run_manifest.json", smoke_manifest)
        _write_json(smoke_dir / "summary.json", {
            "classification": "SMOKE_ONLY",
            "lifecycle_complete": True,
            "exact_outputs": True,
        })
        cost_summary = {
            "status": "PASS",
            "purpose": "authoritative",
            "source_tree_sha256": source_sha256,
            "environment_sha256": environment_sha256,
            "engine_config_sha256": "3" * 64,
            "required_shape_sha256": "4" * 64,
            "raw_rows_sha256": "5" * 64,
            **authoritative_coefficients,
            "envelope_sha256": "6" * 64,
        }
        _write_json(
            cost_dir / "cost_calibration_summary.json",
            cost_summary,
        )
        artifact_sha256 = gate.sha256_file(
            cost_dir / "cost_calibration_summary.json"
        )
        authoritative = {
            "artifact_sha256": artifact_sha256,
            **authoritative_coefficients,
        }
        cost_manifest = {
            "run_tag": "cost-tag",
            "run_type": "cost_calibration",
            "purpose": "authoritative",
            "source_tree_sha256": source_sha256,
            "environment_sha256": environment_sha256,
            "cost_calibration_artifact_sha256": artifact_sha256,
            **gate._resolved_policy_contract(
                cost_envelope=authoritative,
            ),
        }
        _write_json(cost_dir / "run_manifest.json", cost_manifest)
        _write_json(root / "source.json", {
            "tree_sha256": source_sha256,
        })
        _write_json(root / "environment.json", environment)

        captured = {}
        original_initialize = gate.initialize_remote_run
        original_run = gate.run_calibration

        def fake_initialize(**kwargs):
            captured["initialize"] = kwargs
            return {
                "run_tag": kwargs["run_tag"],
                "source_tree_sha256": source_sha256,
                "environment_sha256": environment_sha256,
                **gate._resolved_policy_contract(
                    cost_envelope=kwargs["cost_envelope"],
                ),
                "smoke_verification": kwargs[
                    "smoke_verification"
                ],
            }

        def fake_run(**kwargs):
            captured["run"] = kwargs
            return {"status": "PASS"}

        gate.initialize_remote_run = fake_initialize
        gate.run_calibration = fake_run
        try:
            result = gate.run_workload_calibration_remote(
                run_dir=root / "workload",
                python_bin="/fake/python",
                model_path="/fake/model",
                run_tag="workload-tag",
                source_evidence_path=root / "source.json",
                environment_evidence_path=root / "environment.json",
                smoke_run_dir=smoke_dir,
                cost_calibration_run_dir=cost_dir,
            )
        finally:
            gate.initialize_remote_run = original_initialize
            gate.run_calibration = original_run

        assert result == {"status": "PASS"}
        assert captured["initialize"]["cost_envelope"] == authoritative
        manifest = captured["run"]["run_manifest"]
        assert manifest["resolved_policy_config_by_name"]["P5"][
            "chunked_prefill_slo_cost_intercept_ns"
        ] == 7_000_000
        assert manifest["cost_calibration_verification"] == {
            "status": "PASS",
            "run_tag": "cost-tag",
            "artifact_sha256": artifact_sha256,
            "source_tree_sha256": source_sha256,
            "environment_sha256": environment_sha256,
        }


def test_environment_identity_ignores_run_local_fields():
    first = {
        "run_tag": "smoke-tag",
        "tinyvllm_file": "/remote/smoke-tag/staging/source/tinyvllm/__init__.py",
        "gpu": "A100",
        "torch": "2.x",
    }
    second = {
        **first,
        "run_tag": "canonical-tag",
        "tinyvllm_file": (
            "/remote/canonical-tag/staging/source/tinyvllm/__init__.py"
        ),
    }
    assert gate.environment_identity_sha256(first) == (
        gate.environment_identity_sha256(second)
    )
    changed = {
        **second,
        "gpu": "different",
    }
    assert gate.environment_identity_sha256(first) != (
        gate.environment_identity_sha256(changed)
    )


def test_run_smoke_produces_p5_lifecycle_artifact():
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        source_root = root / "source"
        source_root.mkdir()
        (source_root / "marker.txt").write_text("arrival source\n")
        source_files = [{
            "path": "marker.txt",
            "size_bytes": (source_root / "marker.txt").stat().st_size,
            "sha256": gate.sha256_file(source_root / "marker.txt"),
        }]
        source_evidence = {
            "schema_version": 1,
            "base_commit": "1" * 40,
            "tree_sha256": gate.canonical_json_sha256(source_files),
            "files": source_files,
            "patch_size_bytes": 0,
            "patch_sha256": hashlib.sha256(b"").hexdigest(),
        }
        _write_json(root / "source_evidence.json", source_evidence)
        (root / "source.patch").write_bytes(b"")
        with tarfile.open(
            root / "source_snapshot.tar.gz",
            "w:gz",
        ) as archive:
            archive.add(source_root, arcname="source")
        environment = {
            "run_tag": "smoke-test",
            "gpu": "fake-gpu",
            "torch": "fake-torch",
        }
        _write_json(root / "capability.json", environment)

        original_ports = gate.allocate_port_pair
        original_run = gate.subprocess.run
        original_initialize = gate.initialize_remote_run
        original_cost = getattr(
            gate,
            "_run_cost_calibration_core",
            None,
        )
        ports = iter(((31_000, 31_001), (31_002, 31_003)))
        provisional_artifact = "e" * 64

        def initialize_with_fake_tokenizer(**kwargs):
            return original_initialize(
                **kwargs,
                tokenizer=FakeTokenizer(),
            )

        def fake_cost(**kwargs):
            assert kwargs["purpose"] == "provisional_smoke"
            provisional_dir = kwargs["run_dir"]
            provisional_dir.mkdir(parents=True)
            _write_json(
                provisional_dir / "cost_calibration_summary.json",
                {
                    "status": "PASS",
                    "purpose": "provisional_smoke",
                },
            )
            return {
                "status": "PASS",
                "purpose": "provisional_smoke",
                "cost_calibration_artifact_sha256":
                    provisional_artifact,
                "cost_intercept_ns": P5_COST_ENVELOPE[
                    "cost_intercept_ns"
                ],
                "cost_per_prefill_token_ns": P5_COST_ENVELOPE[
                    "cost_per_prefill_token_ns"
                ],
            }

        def fake_run(command, **kwargs):
            del kwargs
            output_dir = Path(
                command[command.index("--output-dir") + 1]
            )
            case_spec = json.loads(
                Path(
                    command[command.index("--case-spec") + 1]
                ).read_text()
            )
            workload = [
                json.loads(line)
                for line in Path(
                    command[
                        command.index("--workload-manifest") + 1
                    ]
                ).read_text().splitlines()
            ]
            timeline = []
            for index, request in enumerate(workload):
                scheduled = (
                    1_000_000_000
                    + request["arrival_offset_ns"]
                )
                token_times = [
                    scheduled + 100_000_000
                    + token_index * 10_000_000
                    for token_index in range(
                        request["requested_output_tokens"]
                    )
                ]
                timeline.append({
                    "request_id": request["request_id"],
                    "seq_id": index,
                    "scheduled_arrival_ns": scheduled,
                    "actual_arrival_ns": scheduled,
                    "first_scheduled_ns": scheduled + 10_000_000,
                    "first_token_ns": token_times[0],
                    "token_timestamps_ns": token_times,
                    "completion_ns": token_times[-1] + 10_000_000,
                    "output_token_ids": list(range(
                        request["requested_output_tokens"]
                    )),
                    "finish_reason": "length",
                    "error": None,
                })
            _write_jsonl(
                output_dir / "request_timeline.jsonl",
                timeline,
            )
            scheduler_rows = [{
                "step_index": 0,
                "step_start_ns": 1_000_000_000,
                "step_end_ns": 2_000_000_000,
            }]
            if case_spec["policy"] == "P5":
                scheduler_rows = [
                    {
                        "step_index": 0,
                        "step_start_ns": 1_000_000_000,
                        "step_end_ns": 1_010_000_000,
                        "demand_state_before": "inactive",
                        "demand_state_after": "active",
                        "selected_chunk_tokens": 128,
                        "actual_prefill_tokens": 128,
                        "suppression_reason": None,
                    },
                    {
                        "step_index": 1,
                        "step_start_ns": 1_010_000_000,
                        "step_end_ns": 1_020_000_000,
                        "demand_state_before": "active",
                        "demand_state_after": "active",
                        "selected_chunk_tokens": 64,
                        "actual_prefill_tokens": 64,
                        "suppression_reason": None,
                    },
                    {
                        "step_index": 2,
                        "step_start_ns": 1_020_000_000,
                        "step_end_ns": 1_030_000_000,
                        "demand_state_before": "active",
                        "demand_state_after": "draining",
                        "selected_chunk_tokens": None,
                        "actual_prefill_tokens": 0,
                        "suppression_reason": "no_slack",
                    },
                ]
            _write_jsonl(
                output_dir / "scheduler_trace.jsonl",
                scheduler_rows,
            )
            _write_jsonl(
                output_dir / "memory_trace.jsonl",
                [{
                    "step_index": 0,
                    "cuda_allocated_bytes": 100,
                    "cuda_reserved_bytes": 200,
                    "used_kv_blocks": 2,
                    "kv_block_bytes": 64,
                }],
            )
            _write_json(output_dir / "case_result.json", {
                "case_id": case_spec["case_id"],
                "status": "PASS",
            })
            (output_dir / "exitcode").write_text("0\n")
            return subprocess.CompletedProcess(
                command,
                0,
                stdout="",
                stderr="",
            )

        gate.allocate_port_pair = lambda: next(ports)
        gate.subprocess.run = fake_run
        gate.initialize_remote_run = initialize_with_fake_tokenizer
        gate._run_cost_calibration_core = fake_cost
        try:
            summary = gate.run_smoke(
                run_dir=root,
                python_bin="/fake/python",
                model_path="/fake/model",
                run_tag="smoke-test",
                source_evidence_path=root / "source_evidence.json",
                environment_evidence_path=root / "capability.json",
            )
        finally:
            gate.allocate_port_pair = original_ports
            gate.subprocess.run = original_run
            gate.initialize_remote_run = original_initialize
            if original_cost is None:
                del gate._run_cost_calibration_core
            else:
                gate._run_cost_calibration_core = original_cost

        assert summary == {
            "classification": "SMOKE_ONLY",
            "lifecycle_complete": True,
            "exact_outputs": True,
            "case_count": 2,
            "p5_smoke": {
                "demand_activation_count": 1,
                "largest_chunk_admission_count": 1,
                "smaller_chunk_admission_count": 1,
                "slo_suppression_count": 1,
                "draining_decision_count": 1,
                "distinct_selected_chunk_tokens": 2,
                "classification": "SMOKE_ONLY",
            },
        }
        manifest = gate._read_json(root / "run_manifest.json")
        assert manifest["smoke_policies"] == ["P0", "P5"]
        assert manifest["provisional_cost_calibration"] == {
            "status": "PASS",
            "purpose": "provisional_smoke",
            "artifact_sha256": provisional_artifact,
        }
        assert manifest["resolved_policy_config_by_name"]["P5"][
            "cost_calibration_artifact_sha256"
        ] == provisional_artifact


def test_run_smoke_is_incomplete_without_preregistered_p5_paths():
    scheduler_rows = _synthetic_p5_smoke_rows()
    scheduler_rows = [
        row for row in scheduler_rows
        if row["selected_chunk_tokens"] != 64
    ]
    p5_smoke = gate.summarize_p5_smoke(scheduler_rows)
    summary = gate.build_smoke_summary(
        lifecycle_complete=True,
        exact_outputs=True,
        case_count=2,
        p5_smoke=p5_smoke,
    )
    assert summary["classification"] == "INCOMPLETE"
    assert summary["p5_smoke"]["classification"] == "INCOMPLETE"


def test_smoke_workload_covers_original_ninth_token_failure_boundary():
    prompt_bank = gate.build_prompt_bank(
        FakeTokenizer(),
        model_id="fake-model",
    )

    workload = gate._smoke_workload(prompt_bank)
    assert workload
    assert all(row["requested_output_tokens"] >= 16 for row in workload)
    assert all(
        row["sampling"]["max_tokens"]
        == row["requested_output_tokens"]
        for row in workload
    )


def _workload_row(
    request_id: str,
    *,
    output_tokens: int = 3,
    bucket: str = "short__short",
) -> dict:
    prompt_class, output_class = bucket.split("__")
    return {
        "request_id": request_id,
        "warmup": False,
        "scenario": "steady_moderate",
        "prompt_token_count": 4,
        "requested_output_tokens": output_tokens,
        "prompt_class": prompt_class,
        "output_class": output_class,
        "service_time_bucket": bucket,
    }


def _timeline_row(
    request_id: str,
    token_timestamps_ns: list[int],
    *,
    seq_id: int = 7,
    scheduled_arrival_ns: int = 100,
    actual_arrival_ns: int = 120,
    first_scheduled_ns: int = 150,
    completion_ns: int | None = None,
) -> dict:
    return {
        "request_id": request_id,
        "seq_id": seq_id,
        "scheduled_arrival_ns": scheduled_arrival_ns,
        "actual_arrival_ns": actual_arrival_ns,
        "first_scheduled_ns": first_scheduled_ns,
        "first_token_ns": token_timestamps_ns[0],
        "token_timestamps_ns": token_timestamps_ns,
        "completion_ns": (
            token_timestamps_ns[-1]
            if completion_ns is None
            else completion_ns
        ),
        "output_token_ids": list(range(len(token_timestamps_ns))),
        "finish_reason": "length",
        "error": None,
    }


def test_reconstructs_scheduled_arrival_metrics_and_shared_step_tokens():
    metrics = gate.reconstruct_request_metrics(
        [_workload_row("r0")],
        [{
            **_timeline_row("r0", [200, 260, 260]),
            "output_token_ids": [11, 12, 13],
        }],
        [],
    )
    assert len(metrics) == 1
    assert metrics[0]["injection_lag_ns"] == 20
    assert metrics[0]["queue_delay_ns"] == 30
    assert metrics[0]["ttft_ns"] == 100
    assert metrics[0]["e2e_ns"] == 160
    assert metrics[0]["itl_ns"] == [60, 0]
    assert metrics[0]["maximum_decode_gap_ns"] == 60


def test_one_token_output_has_no_itl_sample():
    metrics = gate.reconstruct_request_metrics(
        [_workload_row("r0", output_tokens=1)],
        [_timeline_row("r0", [300])],
        [],
    )
    assert metrics[0]["itl_ns"] == []
    assert metrics[0]["maximum_decode_gap_ns"] is None


def test_lifecycle_reconstruction_rejects_duplicate_binding_and_bad_time():
    workload = [
        _workload_row("r0", output_tokens=1),
        _workload_row("r1", output_tokens=1),
    ]
    duplicate = [
        _timeline_row("r0", [300], seq_id=7),
        _timeline_row("r1", [310], seq_id=7),
    ]
    try:
        gate.reconstruct_request_metrics(workload, duplicate, [])
    except ValueError as exc:
        assert "duplicate sequence binding" in str(exc)
    else:
        raise AssertionError("duplicate sequence binding must fail")

    bad_time = [{
        **_timeline_row("r0", [300], seq_id=7),
        "actual_arrival_ns": 90,
    }]
    try:
        gate.reconstruct_request_metrics(
            [_workload_row("r0", output_tokens=1)],
            bad_time,
            [],
        )
    except ValueError as exc:
        assert "timestamp ordering" in str(exc)
    else:
        raise AssertionError("invalid timestamps must fail")


def test_repetition_summary_reports_percentiles_fairness_and_memory():
    workload = [
        _workload_row("r0", bucket="short__short"),
        _workload_row("r1", bucket="long__long"),
    ]
    timeline = [
        _timeline_row("r0", [200, 250, 300], seq_id=7),
        _timeline_row(
            "r1",
            [400, 500, 600],
            seq_id=8,
            scheduled_arrival_ns=110,
            actual_arrival_ns=130,
            first_scheduled_ns=170,
        ),
    ]
    metrics = gate.reconstruct_request_metrics(
        workload,
        timeline,
        [],
    )
    summary = gate.summarize_repetition(
        {
            "policy": "P0",
            "scenario": "steady_moderate",
            "repetition": 0,
            "measurement_start_ns": 100,
            "measurement_end_ns": 600,
            "required_service_buckets": [
                "short__short",
                "long__long",
            ],
        },
        metrics,
        [{
            "cuda_allocated_bytes": 100,
            "cuda_reserved_bytes": 200,
            "used_kv_blocks": 3,
            "kv_block_bytes": 64,
        }, {
            "cuda_allocated_bytes": 150,
            "cuda_reserved_bytes": 240,
            "used_kv_blocks": 4,
            "kv_block_bytes": 64,
        }],
    )
    assert summary["metrics"]["request_throughput_rps"] == 4_000_000.0
    assert summary["metrics"]["output_token_throughput_tps"] == 12_000_000.0
    assert summary["metrics"]["p95_ttft_ns"] == 290.0
    assert summary["metrics"]["p95_itl_ns"] == 100.0
    assert summary["metrics"]["peak_cuda_reserved_bytes"] == 240
    assert summary["metrics"]["peak_kv_bytes"] == 256
    assert set(summary["metrics"]["service_buckets"]) == {
        "short__short",
        "long__long",
    }
    assert 0.0 < summary["metrics"]["jain_service_rate_index"] <= 1.0


def test_case_aggregation_reports_median_and_worst_repetition():
    rows = [
        _case_row("P2", 0, throughput=110.0, ttft=90.0),
        _case_row("P2", 1, throughput=105.0, ttft=95.0),
        _case_row("P2", 2, throughput=99.0, ttft=110.0),
    ]
    aggregate = gate.aggregate_case_repetitions(rows)
    assert aggregate["median_metrics"]["request_throughput_rps"] == 105.0
    assert aggregate["median_metrics"]["p95_ttft_ns"] == 95.0
    assert aggregate["worst_repetition"]["repetition"] == 2
    assert (
        aggregate["worst_repetition"]["metrics"][
            "request_throughput_rps"
        ]
        == 99.0
    )


def test_p5_policy_counters_are_derived_from_scheduler_rows():
    rows = [{
        "selected_chunk_tokens": 128,
        "actual_prefill_tokens": 128,
        "predicted_step_ns": 20,
        "actual_step_duration_ns": 19,
        "demand_state_after": "active",
        "suppression_reason": None,
        "clock_invalid": False,
    }, {
        "selected_chunk_tokens": 64,
        "actual_prefill_tokens": 48,
        "predicted_step_ns": 10,
        "actual_step_duration_ns": 11,
        "demand_state_after": "active",
        "suppression_reason": None,
        "clock_invalid": False,
    }, {
        "selected_chunk_tokens": None,
        "actual_prefill_tokens": 16,
        "predicted_step_ns": None,
        "actual_step_duration_ns": 4,
        "demand_state_after": "active",
        "suppression_reason": None,
        "clock_invalid": False,
    }, {
        "selected_chunk_tokens": None,
        "actual_prefill_tokens": 0,
        "predicted_step_ns": None,
        "actual_step_duration_ns": 5,
        "demand_state_after": "draining",
        "suppression_reason": "no_slack",
        "clock_invalid": False,
    }, {
        "selected_chunk_tokens": None,
        "actual_prefill_tokens": 0,
        "predicted_step_ns": None,
        "actual_step_duration_ns": 5,
        "demand_state_after": "active",
        "suppression_reason": "missing_decode_progress",
        "clock_invalid": False,
    }, {
        "selected_chunk_tokens": None,
        "actual_prefill_tokens": 0,
        "predicted_step_ns": None,
        "actual_step_duration_ns": None,
        "demand_state_after": "active",
        "suppression_reason": "clock_invalid",
        "clock_invalid": True,
    }]
    assert gate.summarize_p5_policy(rows) == {
        "mixed_decision_count": 2,
        "slo_suppression_count": 1,
        "draining_decision_count": 1,
        "selected_chunk_histogram": {
            "64": 1,
            "128": 1,
        },
        "envelope_underprediction_count": 1,
        "missing_progress_count": 1,
        "clock_invalid_count": 1,
    }


def _classification_manifest() -> dict:
    contract = gate._resolved_policy_contract(
        cost_envelope=P5_COST_ENVELOPE,
    )
    return {
        "required_scenarios": ["steady_moderate"],
        "measured_repetitions": 3,
        "policy_identity_by_name": contract[
            "policy_identity_by_name"
        ],
        "canonical_policy_by_name": contract[
            "canonical_policy_by_name"
        ],
    }


def _case_row(
    policy: str,
    repetition: int,
    *,
    scenario: str = "steady_moderate",
    throughput: float = 100.0,
    ttft: float = 100.0,
    itl: float = 100.0,
    p99_ttft: float | None = None,
    p99_itl: float | None = None,
    e2e: float = 100.0,
    gap: float = 100.0,
    cuda_reserved: float = 100.0,
    kv_bytes: float = 100.0,
    bucket_p95: float = 100.0,
    p5_policy: dict | None = None,
    status: str = "PASS",
    exact_outputs: bool = True,
    complete_requests: bool = True,
    no_starvation: bool = True,
) -> dict:
    return {
        "policy": policy,
        "scenario": scenario,
        "repetition": repetition,
        "status": status,
        "correctness": {
            "exact_outputs": exact_outputs,
            "complete_requests": complete_requests,
            "no_starvation": no_starvation,
            "valid_lifecycle": True,
            "stable_p0_outputs": True,
        },
        "metrics": {
            "request_throughput_rps": throughput,
            "output_token_throughput_tps": throughput * 10.0,
            "p95_ttft_ns": ttft,
            "p95_itl_ns": itl,
            "p99_ttft_ns": (
                ttft if p99_ttft is None else p99_ttft
            ),
            "p99_itl_ns": (
                itl if p99_itl is None else p99_itl
            ),
            "p99_e2e_ns": e2e,
            "maximum_decode_gap_ns": gap,
            "peak_cuda_reserved_bytes": cuda_reserved,
            "peak_kv_bytes": kv_bytes,
            "service_buckets": {
                "short__short": {
                    "p95_e2e_ns": bucket_p95,
                    "completed_requests": 1,
                },
            },
        },
        **({"p5_policy": p5_policy} if p5_policy is not None else {}),
    }


def _rows_with_candidate(
    candidate_values: list[dict],
    *,
    p4_values: list[dict] | None = None,
) -> list[dict]:
    rows = []
    for repetition in range(3):
        rows.append(_case_row("P0", repetition))
        rows.append(_case_row(
            "P5",
            repetition,
            **candidate_values[repetition],
        ))
        rows.append(_case_row(
            "P4",
            repetition,
            **((p4_values or [{}, {}, {}])[repetition]),
        ))
    return rows


def _canonical_classification_manifest() -> dict:
    manifest = _classification_manifest()
    manifest["required_scenarios"] = list(
        gate.CANONICAL_SCENARIOS
    )
    return manifest


def _p5_policy_evidence(
    scenario: str,
    repetition: int,
) -> dict:
    histogram = {"128": 1}
    if scenario == "burst" and repetition == 0:
        histogram = {"16": 1, "64": 1, "128": 1}
    return {
        "mixed_decision_count": sum(histogram.values()),
        "slo_suppression_count": (
            1
            if scenario == "steady_moderate" and repetition == 0
            else 0
        ),
        "draining_decision_count": 1,
        "selected_chunk_histogram": histogram,
        "envelope_underprediction_count": 0,
        "missing_progress_count": 0,
        "clock_invalid_count": 0,
    }


def _canonical_p5_rows() -> list[dict]:
    rows = []
    for scenario in gate.CANONICAL_SCENARIOS:
        for repetition in range(3):
            rows.append(_case_row(
                "P0",
                repetition,
                scenario=scenario,
            ))
            rows.append(_case_row(
                "P4",
                repetition,
                scenario=scenario,
            ))
            rows.append(_case_row(
                "P5",
                repetition,
                scenario=scenario,
                throughput=(
                    125.0 if scenario == "burst" else 106.0
                ),
                p5_policy=_p5_policy_evidence(
                    scenario,
                    repetition,
                ),
            ))
    return rows


def _p5_rows(
    rows: list[dict],
    scenario: str | None = None,
) -> list[dict]:
    return [
        row for row in rows
        if row["policy"] == "P5"
        and (scenario is None or row["scenario"] == scenario)
    ]


def test_p5_canonical_promotion_requirements_pass_at_boundaries():
    summary = gate.classify_gate(
        _canonical_classification_manifest(),
        _canonical_p5_rows(),
    )
    assert summary["classification"] == "GO"
    assert summary["candidate_results"]["P5"]["guard_failures"] == []


def test_p5_canonical_tail_and_fairness_guards_are_exact():
    mutations = (
        ("p99 TTFT", lambda rows: rows[0]["metrics"].__setitem__(
            "p99_ttft_ns", 110.001
        )),
        ("p99 ITL", lambda rows: rows[0]["metrics"].__setitem__(
            "p99_itl_ns", 110.001
        )),
        ("p99 E2E", lambda rows: rows[0]["metrics"].__setitem__(
            "p99_e2e_ns", 110.001
        )),
        ("max gap", lambda rows: rows[0]["metrics"].__setitem__(
            "maximum_decode_gap_ns", 110.001
        )),
        ("mixed fairness", lambda rows: _p5_rows(
            rows, "mixed_service_fairness"
        )[0]["metrics"]["service_buckets"]["short__short"].__setitem__(
            "p95_e2e_ns", 110.001
        )),
    )
    for label, mutate in mutations:
        rows = _canonical_p5_rows()
        mutate(_p5_rows(rows))
        summary = gate.classify_gate(
            _canonical_classification_manifest(),
            rows,
        )
        assert summary["classification"] == "NO_GO", label


def test_p5_canonical_long_prompt_and_burst_guards_are_exact():
    rows = _canonical_p5_rows()
    _p5_rows(rows, "long_prompt_pressure")[0]["metrics"][
        "p95_itl_ns"
    ] = 105.001
    assert gate.classify_gate(
        _canonical_classification_manifest(),
        rows,
    )["classification"] == "NO_GO"

    rows = _canonical_p5_rows()
    for row in _p5_rows(rows, "burst")[:2]:
        row["metrics"]["request_throughput_rps"] = 124.999
    assert gate.classify_gate(
        _canonical_classification_manifest(),
        rows,
    )["classification"] == "NO_GO"


def test_p5_canonical_structural_promotion_guards_are_exact():
    rows = _canonical_p5_rows()
    for row in _p5_rows(rows, "burst"):
        row["p5_policy"]["selected_chunk_histogram"] = {
            "64": 1,
            "128": 1,
        }
    assert gate.classify_gate(
        _canonical_classification_manifest(),
        rows,
    )["classification"] == "NO_GO"

    rows = _canonical_p5_rows()
    for row in _p5_rows(rows):
        if row["scenario"] != "burst":
            row["p5_policy"]["slo_suppression_count"] = 0
    assert gate.classify_gate(
        _canonical_classification_manifest(),
        rows,
    )["classification"] == "NO_GO"

    rows = _canonical_p5_rows()
    _p5_rows(rows)[0]["p5_policy"][
        "envelope_underprediction_count"
    ] = 1
    assert gate.classify_gate(
        _canonical_classification_manifest(),
        rows,
    )["classification"] == "NO_GO"


def test_classification_throughput_boundary_is_go():
    summary = gate.classify_gate(
        _classification_manifest(),
        _rows_with_candidate([
            {"throughput": 105.0},
            {"throughput": 106.0},
            {"throughput": 107.0},
        ]),
    )
    assert summary["classification"] == "GO"
    assert summary["candidate_results"]["P5"]["benefit_path"] == (
        "throughput"
    )


def test_classification_favorable_but_subthreshold_is_promising():
    summary = gate.classify_gate(
        _classification_manifest(),
        _rows_with_candidate([
            {"throughput": 104.0},
            {"throughput": 104.5},
            {"throughput": 104.999},
        ]),
    )
    assert summary["classification"] == "PROMISING_NOT_PROVEN"


def test_classification_latency_and_memory_boundaries_are_go():
    latency = gate.classify_gate(
        _classification_manifest(),
        _rows_with_candidate([
            {"ttft": 90.0, "throughput": 98.0},
            {"ttft": 89.0, "throughput": 99.0},
            {"ttft": 88.0, "throughput": 100.0},
        ]),
    )
    assert latency["classification"] == "GO"
    assert latency["candidate_results"]["P5"]["benefit_path"] == (
        "latency"
    )

    memory = gate.classify_gate(
        _classification_manifest(),
        _rows_with_candidate([
            {
                "cuda_reserved": 95.0,
                "throughput": 98.0,
                "ttft": 102.0,
                "itl": 102.0,
            },
            {
                "cuda_reserved": 94.0,
                "throughput": 99.0,
                "ttft": 101.0,
                "itl": 101.0,
            },
            {
                "cuda_reserved": 93.0,
                "throughput": 100.0,
            },
        ]),
    )
    assert memory["classification"] == "GO"
    assert memory["candidate_results"]["P5"]["benefit_path"] == (
        "memory"
    )


def test_tail_guard_or_bad_worst_repetition_prevents_go():
    tail = gate.classify_gate(
        _classification_manifest(),
        _rows_with_candidate([
            {"throughput": 106.0, "e2e": 110.001},
            {"throughput": 106.0},
            {"throughput": 106.0},
        ]),
    )
    assert tail["classification"] == "NO_GO"
    assert tail["candidate_results"]["P5"]["guard_failures"]

    bad_worst = gate.classify_gate(
        _classification_manifest(),
        _rows_with_candidate([
            {"throughput": 110.0},
            {"throughput": 110.0},
            {"throughput": 99.0},
        ]),
    )
    assert bad_worst["classification"] == "PROMISING_NOT_PROVEN"


def test_structural_and_correctness_failures_take_precedence():
    missing = _rows_with_candidate([
        {"throughput": 106.0},
        {"throughput": 106.0},
        {"throughput": 106.0},
    ])
    missing.pop()
    incomplete = gate.classify_gate(
        _classification_manifest(),
        missing,
    )
    assert incomplete["classification"] == "INCOMPLETE"
    assert incomplete["structural_failures"]

    incorrect_rows = _rows_with_candidate([
        {"throughput": 106.0},
        {"throughput": 106.0},
        {"throughput": 106.0},
    ])
    incorrect_rows[1]["correctness"]["exact_outputs"] = False
    no_go = gate.classify_gate(
        _classification_manifest(),
        incorrect_rows,
    )
    assert no_go["classification"] == "NO_GO"
    assert no_go["correctness_failures"]


def test_p5_is_the_only_promotion_authority():
    p4_go_p5_no_go = gate.classify_gate(
        _classification_manifest(),
        _rows_with_candidate(
            [{}, {}, {}],
            p4_values=[
                {"throughput": 106.0},
                {"throughput": 106.0},
                {"throughput": 106.0},
            ],
        ),
    )
    assert p4_go_p5_no_go["candidate_results"]["P4"]["classification"] == "GO"
    assert p4_go_p5_no_go["candidate_results"]["P5"]["classification"] == "NO_GO"
    assert p4_go_p5_no_go["classification"] == "NO_GO"

    p4_no_go_p5_go = gate.classify_gate(
        _classification_manifest(),
        _rows_with_candidate([
            {"throughput": 106.0},
            {"throughput": 106.0},
            {"throughput": 106.0},
        ]),
    )
    assert p4_no_go_p5_go["candidate_results"]["P4"]["classification"] == "NO_GO"
    assert p4_no_go_p5_go["candidate_results"]["P5"]["classification"] == "GO"
    assert p4_no_go_p5_go["classification"] == "GO"


def test_smoke_workload_can_activate_p4_and_cross_ninth_token():
    workload = gate._smoke_workload(_prompt_bank())
    assert len(workload) >= 10
    assert sum(row["arrival_offset_ns"] == 0 for row in workload) >= 10
    assert sum(row["prompt_class"] == "long" for row in workload) >= 10
    assert all(row["sampling"]["temperature"] == 0.0 for row in workload)
    assert all(row["sampling"]["ignore_eos"] is True for row in workload)
    assert all(row["requested_output_tokens"] >= 16 for row in workload)


def _synthetic_p5_smoke_rows() -> list[dict]:
    return [
        {
            "demand_state_before": "inactive",
            "demand_state_after": "active",
            "selected_chunk_tokens": 128,
            "actual_prefill_tokens": 128,
            "suppression_reason": None,
        },
        {
            "demand_state_before": "active",
            "demand_state_after": "active",
            "selected_chunk_tokens": 64,
            "actual_prefill_tokens": 64,
            "suppression_reason": None,
        },
        {
            "demand_state_before": "active",
            "demand_state_after": "draining",
            "selected_chunk_tokens": None,
            "actual_prefill_tokens": 0,
            "suppression_reason": "no_slack",
        },
    ]


def test_p5_smoke_requires_all_preregistered_policy_paths():
    summary = gate.summarize_p5_smoke(
        _synthetic_p5_smoke_rows()
    )
    assert summary["demand_activation_count"] >= 1
    assert summary["largest_chunk_admission_count"] >= 1
    assert summary["smaller_chunk_admission_count"] >= 1
    assert summary["slo_suppression_count"] >= 1
    assert summary["draining_decision_count"] >= 1
    assert summary["distinct_selected_chunk_tokens"] >= 2
    assert summary["classification"] == "SMOKE_ONLY"


def test_p5_smoke_without_mixed_admission_is_incomplete():
    summary = gate.summarize_p5_smoke([
        row for row in _synthetic_p5_smoke_rows()
        if row["selected_chunk_tokens"] is None
    ])
    assert summary["classification"] == "INCOMPLETE"


def main():
    test_seeded_steady_and_burst_workloads_are_byte_stable()
    test_built_prompt_bank_is_sorted_hashed_and_valid()
    test_service_buckets_are_fixed_before_execution()
    test_p5_policy_contract_is_frozen_and_source_bound()
    test_nearest_rank_boundaries()
    test_canonical_rates_and_sampling_contract_are_frozen()
    test_invalid_lambda_and_policy_fail_closed()
    test_unexpected_candidate_policy_collision_is_rejected()
    test_prompt_bank_hash_detects_drift()
    test_calibration_manifest_is_deterministic_and_p0_only()
    test_select_lambda_ref_excludes_post_arrival_drain_from_slope()
    test_reconstruct_calibration_backlog_samples_uses_offered_window()
    test_select_lambda_ref_recomputes_tail_slope_and_uses_95_percent_ceiling()
    test_select_lambda_ref_requires_stable_point_and_higher_ceiling()
    test_select_lambda_ref_rejects_structural_or_nonfinite_rows()
    test_build_case_matrix_has_exact_interleaved_non_alias_cases()
    test_build_case_matrix_rejects_bad_alias_or_repetition_contract()
    test_allocate_port_pair_returns_distinct_ephemeral_ports()
    test_run_canonical_uses_unique_ports_and_resume_is_immutable()
    test_run_canonical_replaces_incomplete_case_and_rejects_identity_drift()
    test_run_canonical_requires_matching_smoke_and_frozen_calibration()
    test_predecessor_identity_binds_source_environment_and_p4()
    test_resolved_policy_contract_is_recomputed_from_current_source()
    test_run_canonical_remote_freezes_predecessor_before_launch()
    test_run_canonical_remote_rejects_tampered_frozen_workload()
    test_run_calibration_doubles_bisects_and_freezes_workload()
    test_snapshot_source_stages_matching_bytes_and_archive()
    test_arrival_load_artifacts_are_the_only_new_ignored_experiment_root()
    test_finalize_artifacts_merges_classifies_and_hashes_deterministically()
    test_cli_exposes_separate_cost_and_workload_calibration_stages()
    test_predecessor_identity_binds_both_p4_and_p5()
    test_authoritative_cost_calibration_stage_writes_complete_bound_artifacts()
    test_workload_calibration_consumes_authoritative_not_provisional_envelope()
    test_environment_identity_ignores_run_local_fields()
    test_run_smoke_produces_p5_lifecycle_artifact()
    test_run_smoke_is_incomplete_without_preregistered_p5_paths()
    test_smoke_workload_covers_original_ninth_token_failure_boundary()
    test_reconstructs_scheduled_arrival_metrics_and_shared_step_tokens()
    test_one_token_output_has_no_itl_sample()
    test_lifecycle_reconstruction_rejects_duplicate_binding_and_bad_time()
    test_repetition_summary_reports_percentiles_fairness_and_memory()
    test_case_aggregation_reports_median_and_worst_repetition()
    test_p5_policy_counters_are_derived_from_scheduler_rows()
    test_p5_canonical_promotion_requirements_pass_at_boundaries()
    test_p5_canonical_tail_and_fairness_guards_are_exact()
    test_p5_canonical_long_prompt_and_burst_guards_are_exact()
    test_p5_canonical_structural_promotion_guards_are_exact()
    test_classification_throughput_boundary_is_go()
    test_classification_favorable_but_subthreshold_is_promising()
    test_classification_latency_and_memory_boundaries_are_go()
    test_tail_guard_or_bad_worst_repetition_prevents_go()
    test_structural_and_correctness_failures_take_precedence()
    test_p5_is_the_only_promotion_authority()
    test_smoke_workload_can_activate_p4_and_cross_ninth_token()
    test_p5_smoke_requires_all_preregistered_policy_paths()
    test_p5_smoke_without_mixed_admission_is_incomplete()
    print("arrival load gate tests passed")


if __name__ == "__main__":
    main()
