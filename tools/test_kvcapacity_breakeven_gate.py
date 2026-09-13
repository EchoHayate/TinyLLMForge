"""Tests for the latent KV capacity Stage 0 gate.

Dependency-free, laptop-runnable. These tests exist because the gate is the
entry condition for every later stage, so a silent regression in it would let a
later stage spend GPU time on a false premise.
"""

import importlib.util
import json
import math
import os
import sys
import types

import pytest

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load_gate():
    for name, path in (
        ("tinyvllm", os.path.join(_REPO_ROOT, "tinyvllm")),
        ("tinyvllm.kvcapacity", os.path.join(_REPO_ROOT, "tinyvllm", "kvcapacity")),
    ):
        if name not in sys.modules:
            module = types.ModuleType(name)
            module.__path__ = [path]
            sys.modules[name] = module
    location = os.path.join(_REPO_ROOT, "tools", "kvcapacity_breakeven_gate.py")
    spec = importlib.util.spec_from_file_location("kvcapacity_breakeven_gate", location)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


GATE = _load_gate()
MODEL = sys.modules["tinyvllm.kvcapacity.capacity_model"]


@pytest.fixture(scope="module")
def artifact():
    return GATE.build_artifact()


@pytest.fixture(scope="module")
def points():
    return GATE.build_matrix()


@pytest.fixture
def inputs():
    return GATE.build_inputs()


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "kwargs",
    [
        {"layers": 0},
        {"layers": -1},
        {"kv_heads": 0},
        {"head_dim": 0},
        {"dtype_bytes": 0},
        {"layers": True},
        {"layers": 1.5},
        {"name": ""},
    ],
)
def test_geometry_rejects_bad_inputs(kwargs):
    base = {
        "name": "x",
        "layers": 36,
        "kv_heads": 8,
        "head_dim": 128,
        "dtype_bytes": 2,
    }
    base.update(kwargs)
    with pytest.raises(ValueError):
        MODEL.ModelGeometry(**base)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"constant_ms": 0.0},
        {"constant_ms": -1.0},
        {"per_token_us": -0.1},
        {"constant_ms": float("nan")},
        {"constant_ms": float("inf")},
        {"constant_ms": True},
    ],
)
def test_fit_rejects_bad_inputs(kwargs):
    base = {"constant_ms": 13.05, "per_token_us": 0.151}
    base.update(kwargs)
    with pytest.raises(ValueError):
        MODEL.DecodeStepFit(**base)


def test_fit_accepts_zero_per_token_term():
    fit = MODEL.DecodeStepFit(constant_ms=13.05, per_token_us=0.0)
    assert fit.step_ms(16384, 8) == pytest.approx(13.05)
    assert fit.kv_attention_share(16384, 8) == pytest.approx(0.0)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"warm_prefill_seconds": -0.1},
        {"decode_steps": 0},
        {"decode_steps": 1.5},
        {"tool_seconds": -1.0},
    ],
)
def test_turn_profile_rejects_bad_inputs(kwargs):
    base = {"warm_prefill_seconds": 0.045, "decode_steps": 31, "tool_seconds": 5.0}
    base.update(kwargs)
    with pytest.raises(ValueError):
        MODEL.TurnProfile(**base)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"kv_bytes": 0},
        {"utilization_target": 0.0},
        {"utilization_target": 1.5},
        {"host_bytes": -1},
        {"host_bandwidth_bytes_per_second": -1.0},
    ],
)
def test_device_budget_rejects_bad_inputs(kwargs):
    base = {"kv_bytes": 1 << 30, "utilization_target": 0.6}
    base.update(kwargs)
    with pytest.raises(ValueError):
        MODEL.DeviceBudget(**base)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"name": ""},
        {"kv_ratio": 0.0},
        {"kv_ratio": 0.5},
        {"attention_time_ratio": -0.1},
        {"lossless": "yes"},
    ],
)
def test_compression_spec_rejects_bad_inputs(kwargs):
    base = {"name": "x"}
    base.update(kwargs)
    with pytest.raises(ValueError):
        MODEL.CompressionSpec(**base)


def test_kv_ratio_below_one_is_rejected_because_it_is_a_shrink_factor():
    with pytest.raises(ValueError):
        MODEL.CompressionSpec(name="expander", kv_ratio=0.25)


@pytest.mark.parametrize("bad", [0, -1, 1.5, True])
def test_step_ms_rejects_bad_context_length(inputs, bad):
    _geometry, fit, _turn, _budget = inputs
    with pytest.raises(ValueError):
        fit.step_ms(bad, 1)


@pytest.mark.parametrize("bad", [0, -1, 2.0])
def test_step_ms_rejects_bad_batch(inputs, bad):
    _geometry, fit, _turn, _budget = inputs
    with pytest.raises(ValueError):
        fit.step_ms(16384, bad)


def test_evaluate_point_rejects_non_bool_offload(inputs):
    geometry, fit, turn, budget = inputs
    with pytest.raises(ValueError):
        MODEL.evaluate_point(
            geometry=geometry,
            fit=fit,
            turn=turn,
            budget=budget,
            compression=MODEL.CompressionSpec(name="baseline"),
            context_length=16384,
            decode_batch=1,
            offload="yes",
        )


def test_offload_requires_host_bandwidth(inputs):
    geometry, fit, turn, _budget = inputs
    budget = MODEL.DeviceBudget(kv_bytes=1 << 34, utilization_target=0.6)
    with pytest.raises(ValueError):
        MODEL.evaluate_point(
            geometry=geometry,
            fit=fit,
            turn=turn,
            budget=budget,
            compression=MODEL.CompressionSpec(name="baseline"),
            context_length=16384,
            decode_batch=1,
            offload=True,
        )


# ---------------------------------------------------------------------------
# Closed-form correctness
# ---------------------------------------------------------------------------


def test_kv_bytes_per_token_is_two_times_layers_heads_dim_dtype(inputs):
    geometry, _fit, _turn, _budget = inputs
    assert geometry.kv_bytes_per_token == 2 * 36 * 8 * 128 * 2 == 147456


def test_step_is_affine_in_total_kv_tokens(inputs):
    _geometry, fit, _turn, _budget = inputs
    single = fit.step_ms(16384, 1)
    double = fit.step_ms(16384, 2)
    assert double - fit.constant_ms == pytest.approx(2 * (single - fit.constant_ms))
    assert fit.step_ms(32768, 1) == pytest.approx(fit.step_ms(16384, 2))


def test_attention_time_ratio_scales_only_the_kv_term(inputs):
    _geometry, fit, _turn, _budget = inputs
    plain = fit.step_ms(65536, 4)
    halved = fit.step_ms(65536, 4, attention_time_ratio=0.5)
    kv_term = plain - fit.constant_ms
    assert halved == pytest.approx(fit.constant_ms + 0.5 * kv_term)


def test_zero_attention_time_ratio_leaves_only_the_constant(inputs):
    _geometry, fit, _turn, _budget = inputs
    assert fit.step_ms(131072, 32, attention_time_ratio=0.0) == pytest.approx(
        fit.constant_ms
    )


def test_kv_share_is_between_zero_and_one(inputs):
    _geometry, fit, _turn, _budget = inputs
    for length in (1024, 16384, 131072):
        for batch in (1, 8, 32):
            share = fit.kv_attention_share(length, batch)
            assert 0.0 < share < 1.0


def test_compression_divides_bytes_per_token_exactly(inputs):
    geometry, fit, turn, budget = inputs
    spec = MODEL.CompressionSpec(name="half", kv_ratio=2.0, lossless=False)
    point = MODEL.evaluate_point(
        geometry=geometry,
        fit=fit,
        turn=turn,
        budget=budget,
        compression=spec,
        context_length=16384,
        decode_batch=1,
    )
    assert point.kv_bytes_per_token == pytest.approx(147456 / 2)
    assert point.bytes_per_agent == pytest.approx(147456 / 2 * 16384)


def test_resident_kv_is_held_for_the_whole_period(points):
    for key, point in points.items():
        if key[3] is False:
            assert point.duty_memory == 1.0
            assert point.restore_seconds == 0.0


def test_offload_releases_memory_and_charges_latency(points):
    for length in GATE.CONTEXT_LENGTHS:
        for batch in GATE.DECODE_BATCHES:
            resident = points[(length, batch, "baseline_gqa", False)]
            offloaded = points[(length, batch, "baseline_gqa", True)]
            assert offloaded.duty_memory < 1.0
            assert offloaded.restore_seconds > 0.0
            assert offloaded.user_latency_seconds > resident.user_latency_seconds
            assert offloaded.gpu_phase_seconds == pytest.approx(
                resident.gpu_phase_seconds
            )


def test_sustained_agents_matches_the_steady_state_identity(points):
    for point in points.values():
        expected = 0.6 * point.decode_batch / point.duty_compute
        assert point.sustained_agents == pytest.approx(expected)


def test_capacity_is_the_minimum_of_the_ceilings(points):
    for point in points.values():
        ceilings = [point.sustained_agents, point.memory_ceiling]
        if point.offload:
            ceilings.append(point.host_ceiling / point.duty_memory)
        assert point.concurrent_agents == pytest.approx(min(ceilings))


def test_feasibility_requires_enough_agents_to_form_the_batch(points):
    for point in points.values():
        assert point.feasible == (point.concurrent_agents >= point.decode_batch)


def test_capacity_gain_requires_matching_workload(points):
    left = points[(16384, 1, "baseline_gqa", False)]
    right = points[(32768, 1, "baseline_gqa", False)]
    other_batch = points[(16384, 2, "baseline_gqa", False)]
    with pytest.raises(ValueError):
        MODEL.capacity_gain(left, right)
    with pytest.raises(ValueError):
        MODEL.capacity_gain(left, other_batch)


def test_capacity_gain_is_reflexive(points):
    point = points[(65536, 4, "mla512_neutral", False)]
    assert MODEL.capacity_gain(point, point) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Anchors against prior measured artifacts
# ---------------------------------------------------------------------------


def test_decode_step_reproduces_measured_value(inputs):
    _geometry, fit, _turn, _budget = inputs
    assert fit.step_ms(16384, 1) == pytest.approx(15.521, abs=0.05)
    assert fit.step_ms(1024, 1) == pytest.approx(13.202, abs=0.1)


def test_kv_share_reproduces_erratum_table(inputs):
    _geometry, fit, _turn, _budget = inputs
    assert fit.kv_attention_share(16384, 1) == pytest.approx(0.159, abs=0.005)
    assert fit.kv_attention_share(65536, 1) == pytest.approx(0.431, abs=0.01)
    assert fit.kv_attention_share(131072, 1) == pytest.approx(0.603, abs=0.01)


def test_warm_demand_reproduces_erratum(points):
    point = points[(16384, 1, "baseline_gqa", False)]
    assert point.gpu_phase_seconds == pytest.approx(0.5262, abs=0.005)


def test_ceilings_reproduce_erratum(points):
    short = points[(16384, 1, "baseline_gqa", False)]
    assert short.memory_ceiling == pytest.approx(21.2, abs=0.3)
    assert short.sustained_agents == pytest.approx(6.3, abs=0.2)
    long = points[(131072, 1, "baseline_gqa", False)]
    assert long.memory_ceiling == pytest.approx(2.6, abs=0.2)
    assert long.sustained_agents == pytest.approx(3.4, abs=0.2)


def test_restore_time_reproduces_erratum(points):
    point = points[(16384, 1, "baseline_gqa", True)]
    assert point.restore_seconds == pytest.approx(0.0966, abs=0.003)


def test_mla_ratio_matches_deepseek_proportions():
    assert GATE.MLA_RATIO == pytest.approx(4096.0 / 1152.0)
    assert GATE.MLA_RATIO == pytest.approx(3.5556, abs=1e-4)


# ---------------------------------------------------------------------------
# Monotonicity and limiting behaviour
# ---------------------------------------------------------------------------


def test_capacity_orders_by_attention_time_ratio(points):
    for length in GATE.CONTEXT_LENGTHS:
        for batch in GATE.DECODE_BATCHES:
            ideal = points[(length, batch, "mla512_ideal", False)].concurrent_agents
            neutral = points[(length, batch, "mla512_neutral", False)].concurrent_agents
            adverse = points[(length, batch, "mla512_adverse", False)].concurrent_agents
            assert ideal >= neutral - 1e-12
            assert neutral >= adverse - 1e-12


def test_memory_ceiling_is_monotone_in_context_length(points):
    previous = math.inf
    for length in GATE.CONTEXT_LENGTHS:
        current = points[(length, 1, "baseline_gqa", False)].memory_ceiling
        assert current < previous
        previous = current


def test_memory_ceiling_is_independent_of_batch_when_resident(points):
    for length in GATE.CONTEXT_LENGTHS:
        values = {
            points[(length, batch, "baseline_gqa", False)].memory_ceiling
            for batch in GATE.DECODE_BATCHES
        }
        assert len(values) == 1


def test_perfect_compressor_is_bounded_and_demand_bound(points):
    for length in GATE.CONTEXT_LENGTHS:
        for batch in GATE.DECODE_BATCHES:
            point = points[(length, batch, "perfect_compressor", False)]
            assert math.isfinite(point.concurrent_agents)
            assert point.binding_constraint == "demand"


def test_perfect_compressor_step_equals_the_batch_invariant_floor(points, inputs):
    _geometry, fit, _turn, _budget = inputs
    for length in GATE.CONTEXT_LENGTHS:
        point = points[(length, 8, "perfect_compressor", False)]
        assert point.step_ms == pytest.approx(fit.constant_ms)


def test_larger_batch_raises_latency(points):
    for length in GATE.CONTEXT_LENGTHS:
        previous = 0.0
        for batch in GATE.DECODE_BATCHES:
            current = points[(length, batch, "baseline_gqa", False)].user_latency_seconds
            assert current > previous
            previous = current


def test_identity_compression_equals_baseline(inputs):
    geometry, fit, turn, budget = inputs
    baseline = MODEL.evaluate_point(
        geometry=geometry,
        fit=fit,
        turn=turn,
        budget=budget,
        compression=MODEL.CompressionSpec(name="baseline_gqa"),
        context_length=65536,
        decode_batch=4,
    )
    identity = MODEL.evaluate_point(
        geometry=geometry,
        fit=fit,
        turn=turn,
        budget=budget,
        compression=MODEL.CompressionSpec(
            name="identity", kv_ratio=1.0, attention_time_ratio=1.0, lossless=False
        ),
        context_length=65536,
        decode_batch=4,
    )
    assert identity.concurrent_agents == pytest.approx(baseline.concurrent_agents)
    assert MODEL.capacity_gain(identity, baseline) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Envelope and pre-registration
# ---------------------------------------------------------------------------


def test_preregistered_thresholds_are_unchanged():
    assert GATE.PREREGISTERED["min_capacity_gain"] == 1.30
    assert GATE.PREREGISTERED["max_latency_ratio"] == 1.05
    assert GATE.PREREGISTERED["conservative_attention_time_ratio"] == 1.0


def test_envelope_never_clears_on_an_unreachable_row(artifact):
    for entry in artifact["envelope"]:
        if not entry["feasible"]:
            assert entry["clears_threshold"] is False


def test_envelope_never_clears_on_a_latency_regression(artifact):
    cap = GATE.PREREGISTERED["max_latency_ratio"]
    for entry in artifact["envelope"]:
        if entry["latency_ratio"] > cap:
            assert entry["clears_threshold"] is False


def test_envelope_never_clears_below_the_margin(artifact):
    margin = GATE.PREREGISTERED["min_capacity_gain"]
    for entry in artifact["envelope"]:
        if entry["gain_vs_lossless"] < margin:
            assert entry["clears_threshold"] is False


def test_adverse_phi_clears_nowhere(artifact):
    assert all(
        entry["clears_threshold"] is False
        for entry in artifact["envelope"]
        if entry["compression"] == "mla512_adverse"
    )


def test_conservative_phi_does_not_win_at_batch_one(artifact):
    assert all(
        entry["clears_threshold"] is False
        for entry in artifact["envelope"]
        if entry["decode_batch"] == 1 and entry["compression"] == "mla512_neutral"
    )


def test_ideal_phi_wins_somewhere(artifact):
    crossover = artifact["crossover_batch"]["mla512_ideal"]
    assert any(value is not None for value in crossover.values())


def test_competitor_is_always_a_lossless_option(artifact):
    allowed = {"baseline_gqa", "baseline_gqa+offload"}
    assert all(entry["competitor"] in allowed for entry in artifact["envelope"])


def test_compression_raises_the_largest_reachable_batch(artifact):
    baseline = artifact["largest_reachable_batch"]["baseline_gqa"]
    compressed = artifact["largest_reachable_batch"]["mla512_neutral"]
    for length in baseline:
        assert compressed[length] >= baseline[length]


# ---------------------------------------------------------------------------
# Artifact hygiene
# ---------------------------------------------------------------------------


def test_gate_passes(artifact):
    failures = [check for check in artifact["invariants"] if not check["passed"]]
    assert failures == []
    assert artifact["status"] == "PASS"


def test_every_invariant_has_a_detail_string(artifact):
    for check in artifact["invariants"]:
        assert check["name"]
        assert check["detail"]


def test_artifact_is_deterministic():
    first = GATE.build_artifact()
    second = GATE.build_artifact()
    assert first["payload_sha256"] == second["payload_sha256"]


def test_payload_sha256_covers_the_payload(artifact):
    import hashlib

    payload = dict(artifact)
    digest = payload.pop("payload_sha256")
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    assert hashlib.sha256(canonical.encode("utf-8")).hexdigest() == digest


def test_artifact_is_json_serialisable(artifact):
    json.dumps(artifact)


def test_every_measured_input_carries_provenance(artifact):
    for key in ("decode_step_fit", "turn_profile", "device_budget", "workload"):
        block = artifact["provenance"][key]
        assert block["note"]
        if key != "workload":
            assert "source" in block or "derived_from" in block


def test_point_count_matches_the_frozen_matrix(artifact):
    expected = (
        len(GATE.CONTEXT_LENGTHS)
        * len(GATE.DECODE_BATCHES)
        * len(GATE.REPRESENTATIONS)
        * 2
    )
    assert len(artifact["points"]) == expected


def test_gate_main_returns_zero(capsys):
    assert GATE.main(["--print-summary"]) == 0
    captured = capsys.readouterr()
    assert "status PASS" in captured.out


def test_gate_writes_artifact(tmp_path):
    target = tmp_path / "nested" / "artifact.json"
    assert GATE.main(["--out", str(target)]) == 0
    with open(target, encoding="utf-8") as handle:
        payload = json.load(handle)
    assert payload["status"] == "PASS"
    assert payload["schema"] == "kvcapacity-stage0-gate/1"


def test_gate_imports_without_torch():
    assert "torch" not in sys.modules
