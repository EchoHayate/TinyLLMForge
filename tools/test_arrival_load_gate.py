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


def test_policy_identity_aliases_explicit_default_only():
    defaults = {
        "chunked_prefill_decode_first": True,
        "chunked_prefill_max_consecutive_chunks": 0,
        "chunked_prefill_mixed_batch": False,
        "chunked_prefill_mixed_min_prompt_tokens": 0,
    }
    resolved = {
        name: gate.resolve_policy_config(name, defaults)
        for name in ("P0", "P1", "P2", "P3")
    }
    aliases = gate.deduplicate_policies(resolved)
    assert aliases["canonical_policy_by_name"] == {
        "P0": "P0",
        "P1": "P0",
        "P2": "P2",
        "P3": "P3",
    }
    assert len(set(aliases["identity_by_name"].values())) == 3


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
    defaults = {
        "chunked_prefill_decode_first": True,
        "chunked_prefill_max_consecutive_chunks": 0,
        "chunked_prefill_mixed_batch": False,
        "chunked_prefill_mixed_min_prompt_tokens": 0,
    }
    resolved = {
        name: gate.resolve_policy_config(name, defaults)
        for name in ("P0", "P1", "P2", "P3")
    }
    resolved["P3"] = dict(resolved["P2"])
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
    }


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
    return {
        "run_tag": "arrival-test",
        "required_scenarios": list(gate.CANONICAL_SCENARIOS),
        "measured_repetitions": 3,
        "canonical_policy_by_name": {
            "P0": "P0",
            "P1": "P0",
            "P2": "P2",
            "P3": "P3",
        },
        "policy_identity_by_name": {
            "P0": "identity-p0",
            "P1": "identity-p0",
            "P2": "identity-p2",
            "P3": "identity-p3",
        },
        "resolved_policy_config_by_name": {
            "P0": {"policy": "P0"},
            "P1": {"policy": "P1"},
            "P2": {"policy": "P2"},
            "P3": {"policy": "P3"},
        },
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
    assert all(row["policy"] != "P1" for row in matrix)
    assert len({row["case_id"] for row in matrix}) == 54
    assert all(
        row["workload_sha256"] == "workload-hash"
        and row["source_tree_sha256"] == "source-hash"
        and row["environment_sha256"] == "environment-hash"
        for row in matrix
    )


def test_build_case_matrix_rejects_bad_alias_or_repetition_contract():
    bad_alias = _case_matrix_manifest()
    bad_alias["canonical_policy_by_name"]["P2"] = "P0"
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
        _write_jsonl(
            case_dir / "scheduler_trace.jsonl",
            [{
                "step_index": 0,
                "step_start_ns": start_ns,
                "step_end_ns": start_ns + 300_000_000,
            }],
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
        verifier = gate._load_local_module(
            "arrival_load_verify_for_gate_test",
            REPO_ROOT / "tools" / "arrival_load_verify.py",
        )
        assert verifier.verify_run(
            root,
            write_output=False,
        ) == first


def test_cli_exposes_task6_subcommands():
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
        "run-calibration",
        "freeze-workload",
        "run-canonical",
        "finalize-artifacts",
        "verify-harness",
        "run-smoke",
        "run-calibration-remote",
    ):
        assert command in completed.stdout


def test_environment_identity_ignores_run_tag_only():
    first = {
        "run_tag": "smoke-tag",
        "gpu": "A100",
        "torch": "2.x",
    }
    second = {
        **first,
        "run_tag": "canonical-tag",
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


def test_run_smoke_produces_independently_verified_lifecycle_artifact():
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
        ports = iter(((31_000, 31_001), (31_002, 31_003)))

        def initialize_with_fake_tokenizer(**kwargs):
            return original_initialize(
                **kwargs,
                tokenizer=FakeTokenizer(),
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
            _write_jsonl(
                output_dir / "scheduler_trace.jsonl",
                [{
                    "step_index": 0,
                    "step_start_ns": 1_000_000_000,
                    "step_end_ns": 2_000_000_000,
                }],
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

        assert summary == {
            "classification": "SMOKE_ONLY",
            "lifecycle_complete": True,
            "exact_outputs": True,
            "case_count": 2,
        }
        verifier = gate._load_local_module(
            "arrival_load_verify_for_smoke_test",
            REPO_ROOT / "tools" / "arrival_load_verify.py",
        )
        assert verifier.verify_run(
            root,
            write_output=False,
        ) == summary


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


def _classification_manifest() -> dict:
    defaults = {
        "chunked_prefill_decode_first": True,
        "chunked_prefill_max_consecutive_chunks": 0,
        "chunked_prefill_mixed_batch": False,
        "chunked_prefill_mixed_min_prompt_tokens": 0,
    }
    resolved = {
        name: gate.resolve_policy_config(name, defaults)
        for name in ("P0", "P1", "P2", "P3")
    }
    aliases = gate.deduplicate_policies(resolved)
    return {
        "required_scenarios": ["steady_moderate"],
        "measured_repetitions": 3,
        "policy_identity_by_name": aliases["identity_by_name"],
        "canonical_policy_by_name": (
            aliases["canonical_policy_by_name"]
        ),
    }


def _case_row(
    policy: str,
    repetition: int,
    *,
    throughput: float = 100.0,
    ttft: float = 100.0,
    itl: float = 100.0,
    e2e: float = 100.0,
    gap: float = 100.0,
    cuda_reserved: float = 100.0,
    kv_bytes: float = 100.0,
    bucket_p95: float = 100.0,
    status: str = "PASS",
    exact_outputs: bool = True,
    complete_requests: bool = True,
    no_starvation: bool = True,
) -> dict:
    return {
        "policy": policy,
        "scenario": "steady_moderate",
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
            "p99_ttft_ns": ttft,
            "p99_itl_ns": itl,
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
    }


def _rows_with_candidate(candidate_values: list[dict]) -> list[dict]:
    rows = []
    for repetition in range(3):
        rows.append(_case_row("P0", repetition))
        rows.append(_case_row(
            "P2",
            repetition,
            **candidate_values[repetition],
        ))
        rows.append(_case_row("P3", repetition))
    return rows


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
    assert summary["candidate_results"]["P2"]["benefit_path"] == (
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
    assert latency["candidate_results"]["P2"]["benefit_path"] == (
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
    assert memory["candidate_results"]["P2"]["benefit_path"] == (
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
    assert tail["candidate_results"]["P2"]["guard_failures"]

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


def main():
    test_seeded_steady_and_burst_workloads_are_byte_stable()
    test_built_prompt_bank_is_sorted_hashed_and_valid()
    test_service_buckets_are_fixed_before_execution()
    test_policy_identity_aliases_explicit_default_only()
    test_nearest_rank_boundaries()
    test_canonical_rates_and_sampling_contract_are_frozen()
    test_invalid_lambda_and_policy_fail_closed()
    test_unexpected_candidate_policy_collision_is_rejected()
    test_prompt_bank_hash_detects_drift()
    test_calibration_manifest_is_deterministic_and_p0_only()
    test_select_lambda_ref_recomputes_tail_slope_and_uses_95_percent_ceiling()
    test_select_lambda_ref_requires_stable_point_and_higher_ceiling()
    test_select_lambda_ref_rejects_structural_or_nonfinite_rows()
    test_build_case_matrix_has_exact_interleaved_non_alias_cases()
    test_build_case_matrix_rejects_bad_alias_or_repetition_contract()
    test_allocate_port_pair_returns_distinct_ephemeral_ports()
    test_run_canonical_uses_unique_ports_and_resume_is_immutable()
    test_run_canonical_replaces_incomplete_case_and_rejects_identity_drift()
    test_run_canonical_requires_matching_smoke_and_frozen_calibration()
    test_run_calibration_doubles_bisects_and_freezes_workload()
    test_snapshot_source_stages_matching_bytes_and_archive()
    test_finalize_artifacts_merges_classifies_and_hashes_deterministically()
    test_cli_exposes_task6_subcommands()
    test_environment_identity_ignores_run_tag_only()
    test_run_smoke_produces_independently_verified_lifecycle_artifact()
    test_reconstructs_scheduled_arrival_metrics_and_shared_step_tokens()
    test_one_token_output_has_no_itl_sample()
    test_lifecycle_reconstruction_rejects_duplicate_binding_and_bad_time()
    test_repetition_summary_reports_percentiles_fairness_and_memory()
    test_case_aggregation_reports_median_and_worst_repetition()
    test_classification_throughput_boundary_is_go()
    test_classification_favorable_but_subthreshold_is_promising()
    test_classification_latency_and_memory_boundaries_are_go()
    test_tail_guard_or_bad_worst_repetition_prevents_go()
    test_structural_and_correctness_failures_take_precedence()
    print("arrival load gate tests passed")


if __name__ == "__main__":
    main()
