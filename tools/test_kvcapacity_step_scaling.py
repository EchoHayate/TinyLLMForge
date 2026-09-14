#!/usr/bin/env python3
"""Tests for the GATE A step-scaling worker helpers and verdict logic.

These tests never touch a GPU. They cover the parts that can silently produce a
plausible-looking wrong answer: the fit, the equal-product consistency check, the
rejection of cells whose batch is not what it claims, and the threshold logic.
"""

import importlib.util
import json
import math
import random
import sys
import types
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent


def _load(name, filename):
    """Import a tool module by path without importing the tinyvllm package."""
    path = HERE / filename
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


worker = _load("_gate_a_worker", "kvcapacity_step_scaling_worker.py")
verdict = _load("_gate_a_verdict", "kvcapacity_step_scaling_verdict.py")


# ---------------------------------------------------------------------------
# The worker must not drag in torch at import time.
# ---------------------------------------------------------------------------


def test_worker_import_does_not_require_torch():
    assert "torch" not in sys.modules or isinstance(sys.modules["torch"], types.ModuleType)
    source = (HERE / "kvcapacity_step_scaling_worker.py").read_text(encoding="utf-8")
    header = source.split("# ---", 1)[0]
    assert "import torch" not in header
    assert "from tinyvllm" not in header


def test_verdict_import_does_not_require_torch_or_numpy():
    source = (HERE / "kvcapacity_step_scaling_verdict.py").read_text(encoding="utf-8")
    assert "import torch" not in source
    assert "import numpy" not in source


# ---------------------------------------------------------------------------
# Grid construction and the equal-product structure it is supposed to provide.
# ---------------------------------------------------------------------------


def test_enumerate_cells_matches_grid_size():
    cells = worker.enumerate_cells()
    expected = sum(len(batches) for _context, batches in worker.CONTEXT_BATCH_GRID)
    assert len(cells) == expected
    assert len(set(cells)) == len(cells)


def test_grid_contains_batch_one_for_every_context():
    cells = worker.enumerate_cells()
    contexts = {context for context, _batch in cells}
    for context in contexts:
        assert (context, 1) in cells


def test_grid_has_at_least_three_distinct_batches():
    batches = {batch for _context, batch in worker.enumerate_cells()}
    assert len(batches) >= 3


def test_grid_provides_equal_product_collisions():
    """The grid is worthless for this purpose if no two shapes share L*B."""
    groups = worker.product_collision_groups(worker.enumerate_cells())
    assert groups, "the grid must contain at least one equal-product group"
    for product, members in groups.items():
        assert len(members) >= 2
        for context, batch in members:
            assert context * batch == product
        assert len({context for context, _batch in members}) >= 2


def test_product_collision_groups_excludes_singletons():
    groups = worker.product_collision_groups(((1024, 1), (2048, 4)))
    assert groups == {}


def test_product_collision_groups_finds_a_planted_pair():
    groups = worker.product_collision_groups(((1024, 4), (2048, 2), (4096, 1)))
    assert list(groups) == [4096]
    assert len(groups[4096]) == 3


# ---------------------------------------------------------------------------
# Capacity arithmetic.
# ---------------------------------------------------------------------------


def test_kv_bytes_scales_with_both_dimensions():
    single = worker.kv_bytes_for_cell(16384, 1)
    assert single == 16384 * worker.KV_BYTES_PER_TOKEN
    assert worker.kv_bytes_for_cell(16384, 4) == 4 * single
    assert worker.kv_bytes_for_cell(65536, 1) == 4 * single


def test_cell_fits_respects_budget_and_headroom():
    budget = worker.kv_bytes_for_cell(16384, 8)
    assert worker.cell_fits(16384, 4, budget)
    # Exactly at the budget must fail, because the guard keeps headroom.
    assert not worker.cell_fits(16384, 8, budget)
    assert worker.cell_fits(16384, 8, budget / 0.95 + 1)


def test_cell_fits_is_permissive_when_budget_unknown():
    assert worker.cell_fits(131072, 32, None)


def test_every_grid_cell_fits_the_stage0_budget():
    """The pre-registered grid must be runnable under the modelled KV budget."""
    budget = int(47.6 * (1024 ** 3))
    for context, batch in worker.enumerate_cells():
        assert worker.cell_fits(context, batch, budget), (context, batch)


def test_every_grid_context_is_within_the_model_positional_limit():
    """The first full run lost five cells to this and the gate still said PASS.

    Qwen3-8B declares max_position_embeddings = 40960, so the engine clamps
    max_model_len and rejects longer prompts outright. A grid that asks for more is
    not ambitious, it is unrunnable.
    """
    for context, _batch in worker.enumerate_cells():
        assert context <= 40960, context


def test_amended_grid_preserves_the_original_resident_token_range():
    """The amendment must not quietly shrink what is being modelled."""
    original = {context * batch for context, batch in verdict.PREREGISTERED_CELLS_V1}
    amended = {context * batch for context, batch in worker.enumerate_cells()}
    assert max(amended) >= max(original)
    assert min(amended) <= min(original)


def test_the_original_grid_is_retained_for_the_record():
    """Amending a pre-registered plan is only defensible if the original stays visible."""
    assert (65536, 1) in verdict.PREREGISTERED_CELLS_V1
    assert (131072, 2) in verdict.PREREGISTERED_CELLS_V1
    assert (65536, 1) not in verdict.PREREGISTERED_CELLS


# ---------------------------------------------------------------------------
# Prompt construction, the guard against prefix-cache sharing.
# ---------------------------------------------------------------------------


def test_prompts_have_requested_shape():
    rng = random.Random(0)
    prompts = worker.build_distinct_prompts(512, 3, 1000, rng)
    assert len(prompts) == 3
    assert all(len(prompt) == 512 for prompt in prompts)


def test_prompts_differ_in_the_first_block():
    """Block-hash prefix caching keys on whole blocks, so block zero must differ."""
    rng = random.Random(1)
    prompts = worker.build_distinct_prompts(600, 6, 5000, rng)
    first_blocks = {tuple(prompt[:256]) for prompt in prompts}
    assert len(first_blocks) == 6


def test_prompt_tokens_stay_in_range():
    rng = random.Random(2)
    prompts = worker.build_distinct_prompts(128, 2, 300, rng)
    for prompt in prompts:
        assert all(4 <= token < 299 for token in prompt)


def test_build_distinct_prompts_rejects_tiny_vocabulary():
    with pytest.raises(ValueError):
        worker.build_distinct_prompts(16, 2, 4, random.Random(3))


def test_prompt_digest_is_order_sensitive_and_stable():
    assert worker.prompt_digest([1, 2, 3]) == worker.prompt_digest([1, 2, 3])
    assert worker.prompt_digest([1, 2, 3]) != worker.prompt_digest([3, 2, 1])


# ---------------------------------------------------------------------------
# Summary statistics.
# ---------------------------------------------------------------------------


def test_summarise_reports_none_for_no_samples():
    assert worker.summarise([]) is None


def test_summarise_computes_expected_values():
    summary = worker.summarise([10.0, 12.0, 11.0, 100.0])
    assert summary["count"] == 4
    assert summary["min_ms"] == 10.0
    assert summary["max_ms"] == 100.0
    assert summary["median_ms"] == pytest.approx(11.5)
    # The median must resist the outlier that the mean absorbs.
    assert summary["mean_ms"] > summary["median_ms"]


# ---------------------------------------------------------------------------
# The fit.
# ---------------------------------------------------------------------------


def _points_from(model, cells):
    return [
        {
            "context_length": context,
            "batch": batch,
            "kv_tokens": context * batch,
            "step_ms": model(context, batch),
            "step_stdev_ms": 0.0,
            "sample_count": 24,
        }
        for context, batch in cells
    ]


def test_least_squares_recovers_a_planted_affine_law():
    designs = [[1.0, 0.0], [1.0, 1.0], [1.0, 2.0], [1.0, 3.0]]
    observations = [5.0, 7.0, 9.0, 11.0]
    intercept, slope = verdict.least_squares(designs, observations)
    assert intercept == pytest.approx(5.0)
    assert slope == pytest.approx(2.0)


def test_least_squares_rejects_underdetermined_systems():
    with pytest.raises(ValueError):
        verdict.least_squares([[1.0, 1.0]], [1.0])


def test_least_squares_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        verdict.least_squares([[1.0, 1.0], [1.0, 2.0]], [1.0])


def test_least_squares_rejects_a_singular_design():
    with pytest.raises(ValueError):
        verdict.least_squares([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]], [1.0, 2.0, 3.0])


def test_r_squared_is_one_for_an_exact_fit():
    designs = [[1.0, 1.0], [1.0, 2.0], [1.0, 3.0]]
    score, _predictions, residual = verdict.r_squared(designs, [3.0, 5.0, 7.0], [1.0, 2.0])
    assert score == pytest.approx(1.0)
    assert residual == pytest.approx(0.0)


def test_fit_models_recovers_the_stage0_law_when_it_holds():
    cells = worker.enumerate_cells()
    points = _points_from(lambda L, B: 13.05 + 0.000151 * L * B, cells)
    m1, m2 = verdict.fit_models(points)
    assert m1["c0_ms"] == pytest.approx(13.05, abs=1e-6)
    assert m1["c1_us_per_token"] == pytest.approx(0.151, abs=1e-9)
    assert m1["r_squared"] == pytest.approx(1.0)
    # M2 has a spare parameter and must drive it to zero, not invent structure.
    assert m2["a_ms_per_sequence"] == pytest.approx(0.0, abs=1e-6)


def test_fit_models_exposes_a_hidden_batch_term():
    """The point of M2: a per-sequence cost must not hide inside M1's constants."""
    cells = worker.enumerate_cells()
    points = _points_from(lambda L, B: 13.0 + 1.5 * B + 0.000151 * L * B, cells)
    m1, m2 = verdict.fit_models(points)
    assert m2["a_ms_per_sequence"] == pytest.approx(1.5, abs=1e-6)
    assert m2["r_squared"] > m1["r_squared"]
    assert m2["residual_reduction_vs_m1"] > 0.5
    # M1 absorbs the batch cost by distorting its parameters.
    assert m1["c1_us_per_token"] > 0.151


def test_fit_models_returns_no_m2_without_batch_variation():
    points = _points_from(lambda L, B: 13.0 + 0.000151 * L * B, ((16384, 1), (32768, 1), (65536, 1)))
    _m1, m2 = verdict.fit_models(points)
    assert m2 is None


def test_batch_term_share_is_reported_at_the_largest_batch():
    cells = worker.enumerate_cells()
    points = _points_from(lambda L, B: 13.0 + 1.0 * B + 0.000151 * L * B, cells)
    _m1, m2 = verdict.fit_models(points)
    share = verdict.batch_term_share(m2, points)
    assert share["batch"] == max(point["batch"] for point in points)
    assert 0.0 < share["share"] < 1.0


def test_batch_term_share_is_none_without_m2():
    assert verdict.batch_term_share(None, []) is None


# ---------------------------------------------------------------------------
# Equal-product consistency, the parameter-free test.
# ---------------------------------------------------------------------------


def test_collision_consistency_passes_when_shape_does_not_matter():
    points = _points_from(lambda L, B: 13.05 + 0.000151 * L * B, worker.enumerate_cells())
    groups = verdict.collision_consistency(points)
    assert groups
    assert all(group["within_threshold"] for group in groups)


def test_collision_consistency_catches_shape_dependence():
    """If batch is more expensive than context, equal-product cells diverge."""
    points = _points_from(
        lambda L, B: 13.05 + 0.000151 * L * B + 4.0 * B, worker.enumerate_cells()
    )
    groups = verdict.collision_consistency(points)
    assert any(not group["within_threshold"] for group in groups)


def test_collision_consistency_ignores_lonely_products():
    points = _points_from(lambda L, B: 10.0, ((1024, 1), (4096, 1)))
    assert verdict.collision_consistency(points) == []


# ---------------------------------------------------------------------------
# Point extraction: the integrity filters.
# ---------------------------------------------------------------------------


def _row(context, batch, median, **overrides):
    row = {
        "context_length": context,
        "batch": batch,
        "measured": True,
        "prefill_tokens_match": True,
        "step": {"median_ms": median, "stdev_ms": 0.1, "count": 24},
    }
    row.update(overrides)
    return row


def test_extract_points_keeps_clean_rows():
    payload = {"rows": [_row(16384, 1, 15.0), _row(16384, 2, 18.0)]}
    points, rejected = verdict.extract_points(payload)
    assert len(points) == 2
    assert rejected == []


def test_extract_points_drops_unmeasured_rows_with_a_reason():
    payload = {
        "rows": [
            _row(16384, 1, 15.0),
            {
                "context_length": 131072,
                "batch": 8,
                "measured": False,
                "step": None,
                "skipped_reason": "resident KV exceeds the device budget",
            },
        ]
    }
    points, rejected = verdict.extract_points(payload)
    assert len(points) == 1
    assert len(rejected) == 1
    assert "budget" in rejected[0]["reason"]


def test_extract_points_rejects_prefill_token_mismatch():
    """A prefill shortfall means prefix caching merged KV, so the batch is a lie.

    This is the specific failure that would make compression look good for the
    wrong reason, so it must remove the cell rather than warn about it.
    """
    payload = {
        "rows": [
            _row(16384, 1, 15.0),
            _row(
                16384,
                4,
                16.0,
                prefill_tokens_match=False,
                prefill_tokens_total=16384,
                prefill_tokens_expected=65536,
            ),
        ]
    }
    points, rejected = verdict.extract_points(payload)
    assert [point["batch"] for point in points] == [1]
    assert "shared" in rejected[0]["reason"]


def test_extract_points_computes_kv_tokens():
    payload = {"rows": [_row(16384, 4, 20.0)]}
    points, _rejected = verdict.extract_points(payload)
    assert points[0]["kv_tokens"] == 65536


# ---------------------------------------------------------------------------
# Verdict logic.
# ---------------------------------------------------------------------------


def _payload_from(model, cells=None):
    cells = cells or worker.enumerate_cells()
    return {
        "payload_sha256": "test",
        "rows": [_row(context, batch, model(context, batch)) for context, batch in cells],
    }


def test_verdict_passes_when_the_stage0_model_holds():
    report = verdict.build_report(_payload_from(lambda L, B: 13.05 + 0.000151 * L * B))
    assert report["verdict"] == "PASS"
    assert all(check["passed"] for check in report["checks"])


def test_verdict_fails_on_a_large_hidden_batch_term():
    report = verdict.build_report(
        _payload_from(lambda L, B: 13.05 + 6.0 * B + 0.000151 * L * B)
    )
    assert report["verdict"] == "FAIL"
    assert "refit" in report["consequence"]


def test_verdict_fails_when_the_step_is_superlinear_in_resident_tokens():
    report = verdict.build_report(
        _payload_from(lambda L, B: 13.05 + 0.000151 * L * B + 2e-10 * (L * B) ** 2)
    )
    assert report["verdict"] == "FAIL"


def test_verdict_is_inconclusive_with_too_few_batches():
    report = verdict.build_report(
        _payload_from(
            lambda L, B: 13.05 + 0.000151 * L * B,
            ((16384, 1), (32768, 1), (65536, 1), (131072, 1)),
        )
    )
    # Coverage is reported first, since a grid that was not covered cannot test
    # anything else. Either way this data must not yield a decision.
    assert report["verdict"] == "INCONCLUSIVE"


def test_verdict_is_inconclusive_with_too_few_points():
    report = verdict.build_report({"rows": [_row(16384, 1, 15.0)]})
    assert report["verdict"] == "INCONCLUSIVE"


def test_verdict_reports_stage0_drift_without_hiding_it():
    """A model that fits perfectly can still contradict the frozen constants."""
    report = verdict.build_report(_payload_from(lambda L, B: 26.0 + 0.000302 * L * B))
    comparison = report["stage0_comparison"]
    assert comparison["c0_ratio"] == pytest.approx(26.0 / 13.05, rel=1e-3)
    assert comparison["c1_ratio"] == pytest.approx(2.0, rel=1e-3)


def test_verdict_flags_a_vacuous_collision_check():
    report = verdict.build_report(
        _payload_from(
            lambda L, B: 13.05 + 0.000151 * L * B,
            ((16384, 1), (16384, 2), (16384, 4), (16384, 8)),
        )
    )
    collision_check = next(
        check for check in report["checks"] if check["name"] == "collision_consistency"
    )
    assert collision_check["vacuous"] is True


def test_report_is_json_serialisable_and_hashed():
    report = verdict.build_report(_payload_from(lambda L, B: 13.05 + 0.000151 * L * B))
    encoded = json.dumps(report, sort_keys=True)
    assert report["report_sha256"] in encoded or len(report["report_sha256"]) == 64
    assert json.loads(encoded)["verdict"] == "PASS"


def test_report_hash_is_deterministic():
    first = verdict.build_report(_payload_from(lambda L, B: 13.05 + 0.000151 * L * B))
    second = verdict.build_report(_payload_from(lambda L, B: 13.05 + 0.000151 * L * B))
    assert first["report_sha256"] == second["report_sha256"]


def test_report_hash_changes_with_the_measurement():
    first = verdict.build_report(_payload_from(lambda L, B: 13.05 + 0.000151 * L * B))
    second = verdict.build_report(_payload_from(lambda L, B: 13.05 + 0.000152 * L * B))
    assert first["report_sha256"] != second["report_sha256"]


def test_render_mentions_the_verdict_and_both_models():
    report = verdict.build_report(_payload_from(lambda L, B: 13.05 + 0.5 * B + 0.000151 * L * B))
    text = verdict.render(report)
    assert "GATE A" in text
    assert report["verdict"] in text
    assert "M1" in text and "M2" in text
    assert "us/token" in text


def test_render_lists_excluded_cells():
    payload = _payload_from(lambda L, B: 13.05 + 0.000151 * L * B)
    payload["rows"].append(
        {
            "context_length": 131072,
            "batch": 32,
            "measured": False,
            "step": None,
            "skipped_reason": "resident KV exceeds the device budget",
        }
    )
    text = verdict.render(verdict.build_report(payload))
    assert "excluded" in text
    assert "budget" in text


def test_thresholds_are_declared_and_within_sane_ranges():
    assert 0.9 <= verdict.MIN_R_SQUARED < 1.0
    assert 0.0 < verdict.MAX_COLLISION_SPREAD <= 0.25
    assert 0.0 < verdict.MAX_BATCH_TERM_SHARE <= 0.5
    assert verdict.STAGE0_C0_MS == 13.05
    assert verdict.STAGE0_C1_US_PER_TOKEN == 0.151


def test_exit_code_is_nonzero_unless_the_gate_passes(tmp_path, capsys):
    payload_path = tmp_path / "payload.json"
    payload_path.write_text(
        json.dumps(_payload_from(lambda L, B: 13.05 + 6.0 * B + 0.000151 * L * B)),
        encoding="utf-8",
    )
    out_path = tmp_path / "verdict.json"
    code = verdict.main(["--payload", str(payload_path), "--out", str(out_path)])
    capsys.readouterr()
    assert code == 1
    assert json.loads(out_path.read_text(encoding="utf-8"))["verdict"] == "FAIL"


def test_exit_code_is_zero_when_the_gate_passes(tmp_path, capsys):
    payload_path = tmp_path / "payload.json"
    payload_path.write_text(
        json.dumps(_payload_from(lambda L, B: 13.05 + 0.000151 * L * B)), encoding="utf-8"
    )
    code = verdict.main(["--payload", str(payload_path)])
    capsys.readouterr()
    assert code == 0


# ---------------------------------------------------------------------------
# A regression guard on the physics the gate is meant to protect.
# ---------------------------------------------------------------------------


def test_a_realistic_gqa_step_law_would_pass():
    """Sanity check with plausible A100 numbers for Qwen3-8B GQA decode."""
    report = verdict.build_report(
        _payload_from(lambda L, B: 13.05 + 0.15 * B + 0.000151 * L * B)
    )
    assert report["verdict"] == "PASS"
    share = report["batch_term_at_largest_batch"]
    assert share["share"] < verdict.MAX_BATCH_TERM_SHARE


def test_noise_does_not_flip_a_true_model():
    rng = random.Random(7)
    report = verdict.build_report(
        _payload_from(
            lambda L, B: 13.05 + 0.000151 * L * B + rng.uniform(-0.3, 0.3),
        )
    )
    assert report["verdict"] == "PASS"


def test_fit_is_not_confused_by_a_single_wild_outlier_being_excluded():
    """An unmeasurable cell must shrink the fit, never poison it."""
    payload = _payload_from(lambda L, B: 13.05 + 0.000151 * L * B)
    payload["rows"][0] = {
        "context_length": payload["rows"][0]["context_length"],
        "batch": payload["rows"][0]["batch"],
        "measured": False,
        "step": None,
        "skipped_reason": "no decode step ran at the target batch",
    }
    report = verdict.build_report(payload)
    # A dropped cell now blocks a PASS, because coverage is judged on cells that
    # produced a number. It must not become a FAIL: a gap in the measurement is not
    # evidence against the model.
    assert report["verdict"] == "INCONCLUSIVE"
    assert "no measurement" in report["consequence"]
    assert len(report["rejected_cells"]) == 1
    # The fit itself is unaffected by the exclusion.
    assert report["model_m1"]["r_squared"] == pytest.approx(1.0, abs=1e-6)


def test_math_import_is_not_needed_for_determinism():
    assert math.isfinite(verdict.MIN_R_SQUARED)


# ---------------------------------------------------------------------------
# M3, the curvature check.
#
# These tests exist to record why M3 is not redundant. Deleting it would restore
# a gate that passes data plainly violating the model it is supposed to test.
# ---------------------------------------------------------------------------


def test_r_squared_alone_cannot_detect_curvature():
    """The documented reason M3 exists, asserted rather than assumed.

    A step that is materially superlinear in resident tokens still fits a
    straight line with an R^2 far above the pre-registered threshold. If this
    test ever fails, the R^2 threshold became a sufficient curvature test and M3
    could be reconsidered. Until then it cannot.
    """
    points = _points_from(
        lambda L, B: 13.05 + 0.000151 * L * B + 2e-10 * (L * B) ** 2,
        worker.enumerate_cells(),
    )
    m1, _m2 = verdict.fit_models(points)
    assert m1["r_squared"] > verdict.MIN_R_SQUARED
    curvature = verdict.curvature_share(verdict.fit_curvature_model(points), points)
    assert curvature["share"] > verdict.MAX_CURVATURE_SHARE


def test_equal_product_check_also_cannot_detect_curvature():
    """The second reason M3 exists.

    Any function of `L * B` alone leaves equal-product cells identical, so the
    parameter-free consistency check is structurally blind to the form of the
    law. It constrains shape dependence only.
    """
    points = _points_from(
        lambda L, B: 13.05 + 0.000151 * L * B + 2e-10 * (L * B) ** 2,
        worker.enumerate_cells(),
    )
    groups = verdict.collision_consistency(points)
    assert groups
    assert all(group["within_threshold"] for group in groups)


def test_fit_curvature_model_recovers_a_planted_quadratic():
    points = _points_from(
        lambda L, B: 13.0 + 0.000151 * L * B + 3e-10 * (L * B) ** 2,
        worker.enumerate_cells(),
    )
    m3 = verdict.fit_curvature_model(points)
    assert m3["c0_ms"] == pytest.approx(13.0, abs=1e-3)
    assert m3["c1_us_per_token"] == pytest.approx(0.151, abs=1e-4)
    assert m3["c2_ms_per_token_squared"] == pytest.approx(3e-10, rel=1e-3)
    assert m3["r_squared"] == pytest.approx(1.0, abs=1e-9)


def test_fit_curvature_model_drives_c2_to_zero_on_linear_data():
    points = _points_from(lambda L, B: 13.05 + 0.000151 * L * B, worker.enumerate_cells())
    m3 = verdict.fit_curvature_model(points)
    assert m3["c2_ms_per_token_squared"] == pytest.approx(0.0, abs=1e-14)
    share = verdict.curvature_share(m3, points)
    assert share["share"] < 1e-6


def test_fit_curvature_model_needs_three_distinct_products():
    """All of the largest grid cells share one product, so this can really happen."""
    points = _points_from(lambda L, B: 13.05 + 0.000151 * L * B, ((16384, 16), (32768, 8), (65536, 4)))
    assert len({point["kv_tokens"] for point in points}) == 1
    assert verdict.fit_curvature_model(points) is None


def test_curvature_share_is_none_without_a_model():
    assert verdict.curvature_share(None, []) is None


def test_curvature_share_uses_absolute_magnitude():
    """A step that bends downward is just as much a model violation as upward."""
    points = _points_from(
        lambda L, B: 40.0 + 0.000151 * L * B - 2e-10 * (L * B) ** 2,
        worker.enumerate_cells(),
    )
    share = verdict.curvature_share(verdict.fit_curvature_model(points), points)
    assert share["curvature_term_ms"] < 0
    assert share["share"] > 0


def test_curvature_share_is_reported_at_the_largest_product():
    points = _points_from(
        lambda L, B: 13.0 + 0.000151 * L * B + 1e-10 * (L * B) ** 2,
        worker.enumerate_cells(),
    )
    share = verdict.curvature_share(verdict.fit_curvature_model(points), points)
    assert share["kv_tokens"] == max(point["kv_tokens"] for point in points)


def test_verdict_is_inconclusive_when_every_cell_shares_one_product():
    """Reachable with this grid: the four largest cells all sit at L*B = 262144.

    The tool must decline to fit rather than raise out of the linear solver. A
    gate that crashes is worse than a gate that says it does not know.
    """
    cells = ((16384, 16), (32768, 8), (65536, 4), (131072, 2))
    assert len({context * batch for context, batch in cells}) == 1
    report = verdict.build_report(
        _payload_from(lambda L, B: 13.05 + 0.000151 * L * B, cells)
    )
    assert report["verdict"] == "INCONCLUSIVE"
    assert "same resident token count" in report["consequence"]


def test_verdict_flags_a_vacuous_curvature_check():
    """Two distinct products identify a slope but cannot identify curvature."""
    cells = ((16384, 1), (16384, 2), (32768, 1))
    report = verdict.build_report(
        _payload_from(lambda L, B: 13.05 + 0.000151 * L * B, cells)
    )
    assert len({point["kv_tokens"] for point in report["measured_points"]}) == 2
    check = next(c for c in report["checks"] if c["name"] == "curvature_share")
    assert check["vacuous"] is True
    assert check["passed"] is True


def test_render_mentions_m3_when_it_was_identified():
    report = verdict.build_report(
        _payload_from(lambda L, B: 13.05 + 0.000151 * L * B, worker.enumerate_cells())
    )
    text = verdict.render(report)
    assert "M3" in text
    assert "ms/token^2" in text


def test_all_three_models_are_reported_together():
    report = verdict.build_report(
        _payload_from(lambda L, B: 13.05 + 0.000151 * L * B, worker.enumerate_cells())
    )
    assert report["model_m1"] and report["model_m2"] and report["model_m3"]
    assert report["thresholds"]["max_curvature_share"] == verdict.MAX_CURVATURE_SHARE


def test_curvature_threshold_is_declared_in_a_sane_range():
    assert 0.0 < verdict.MAX_CURVATURE_SHARE <= 0.25


def test_a_realistic_law_passes_all_three_model_checks():
    """Plausible A100 numbers, with small per-sequence overhead and no bending."""
    report = verdict.build_report(
        _payload_from(
            lambda L, B: 13.05 + 0.15 * B + 0.000151 * L * B, worker.enumerate_cells()
        )
    )
    assert report["verdict"] == "PASS"
    assert report["curvature_at_largest_product"]["share"] < verdict.MAX_CURVATURE_SHARE
    assert report["batch_term_at_largest_batch"]["share"] < verdict.MAX_BATCH_TERM_SHARE


# ---------------------------------------------------------------------------
# Grid overrides and the provenance that keeps a smoke run honest.
# ---------------------------------------------------------------------------


def test_parse_grid_spec_reads_a_well_formed_specification():
    assert worker.parse_grid_spec("1024:1,2,4;2048:1,2") == (
        (1024, (1, 2, 4)),
        (2048, (1, 2)),
    )


def test_parse_grid_spec_tolerates_whitespace_and_trailing_separators():
    assert worker.parse_grid_spec(" 1024:1,2 ; ") == ((1024, (1, 2)),)


@pytest.mark.parametrize(
    "text",
    [
        "",
        "1024",
        "1024:",
        "0:1",
        "1024:0",
        "-8:1",
        "1024:1,1",
        "1024:1;1024:2",
    ],
)
def test_parse_grid_spec_rejects_malformed_input(text):
    with pytest.raises(ValueError):
        worker.parse_grid_spec(text)


def test_format_grid_spec_round_trips():
    spec = worker.format_grid_spec(worker.CONTEXT_BATCH_GRID)
    assert worker.parse_grid_spec(spec) == worker.CONTEXT_BATCH_GRID


def test_enumerate_cells_honours_an_override_grid():
    cells = worker.enumerate_cells(((512, (1, 2)),))
    assert cells == ((512, 1), (512, 2))


def _payload_for_grid(grid, model):
    rows = [
        _row(context, batch, model(context, batch))
        for context, batch in worker.enumerate_cells(grid)
    ]
    return worker.build_payload(
        rows,
        [],
        model_path="/models/qwen3-8b",
        enforce_eager=False,
        seed=1,
        gpu_memory_utilization=0.85,
        warmup_steps=8,
        measured_steps=24,
        grid=grid,
    )


def test_payload_marks_the_preregistered_grid():
    payload = _payload_for_grid(
        worker.CONTEXT_BATCH_GRID, lambda L, B: 13.05 + 0.000151 * L * B
    )
    assert payload["grid_is_preregistered"] is True
    assert payload["grid_spec"] == payload["preregistered_grid_spec"]


def test_payload_marks_a_narrowed_grid():
    grid = ((1024, (1, 2, 4)), (2048, (1, 2)))
    payload = _payload_for_grid(grid, lambda L, B: 13.05 + 0.000151 * L * B)
    assert payload["grid_is_preregistered"] is False
    assert payload["grid_spec"] == "1024:1,2,4;2048:1,2"
    assert payload["preregistered_grid_spec"] != payload["grid_spec"]


def test_a_narrowed_grid_cannot_produce_a_pass():
    """A smoke run that fits perfectly must not be presentable as GATE A.

    This is the safeguard against the exact move that sank the previous research
    line: a technically correct number reported as though it answered a question
    it never addressed.
    """
    grid = ((1024, (1, 2, 4)), (2048, (1, 2)), (4096, (1,)))
    payload = _payload_for_grid(grid, lambda L, B: 13.05 + 0.000151 * L * B)
    report = verdict.build_report(payload)
    assert report["grid_is_preregistered"] is False
    assert report["verdict"] == "INCONCLUSIVE"
    assert "narrowed grid" in report["consequence"]
    # Every check except coverage still ran and passed; only the verdict is held back.
    for check in report["checks"]:
        if check["name"] != "grid_coverage":
            assert check["passed"], check


def test_a_narrowed_grid_cannot_produce_a_failure_either():
    """A narrow grid decides nothing in either direction.

    The smoke run made the reason concrete: a 0.6B model at contexts up to 4096
    spends about two percent of its step on KV, so the constant dominates, R^2
    collapses, and the checks return FAIL while establishing nothing about the
    model. Emitting that FAIL would teach the reader to discount a FAIL on the
    real grid, which is the verdict this gate exists to be able to deliver.
    """
    grid = ((1024, (1, 2, 4)), (2048, (1, 2)), (4096, (1,)))
    payload = _payload_for_grid(grid, lambda L, B: 13.05 + 9.0 * B + 0.000151 * L * B)
    report = verdict.build_report(payload)
    assert report["verdict"] == "INCONCLUSIVE"
    # The failing checks are still reported, so the failure is visible even though
    # the verdict withholds judgement.
    failed = [check["name"] for check in report["checks"] if not check["passed"]]
    assert "m1_r_squared" in failed


def test_render_flags_a_narrowed_grid():
    grid = ((1024, (1, 2, 4)), (2048, (1, 2)), (4096, (1,)))
    text = verdict.render(verdict.build_report(_payload_for_grid(grid, lambda L, B: 13.05 + 0.000151 * L * B)))
    assert "NOT pre-registered" in text


def test_preregistered_grid_verdict_is_not_downgraded():
    payload = _payload_for_grid(
        worker.CONTEXT_BATCH_GRID, lambda L, B: 13.05 + 0.000151 * L * B
    )
    report = verdict.build_report(payload)
    assert report["grid_is_preregistered"] is True
    assert report["verdict"] == "PASS"


def test_payload_is_hashed_and_json_serialisable():
    payload = _payload_for_grid(
        worker.CONTEXT_BATCH_GRID, lambda L, B: 13.05 + 0.000151 * L * B
    )
    assert len(payload["payload_sha256"]) == 64
    assert json.loads(json.dumps(payload))["schema"] == payload["schema"]


def test_payload_hash_tracks_the_grid():
    full = _payload_for_grid(
        worker.CONTEXT_BATCH_GRID, lambda L, B: 13.05 + 0.000151 * L * B
    )
    narrow = _payload_for_grid(
        ((1024, (1, 2)),), lambda L, B: 13.05 + 0.000151 * L * B
    )
    assert full["payload_sha256"] != narrow["payload_sha256"]


def test_cli_defaults_to_the_preregistered_grid():
    args = worker.parse_args(["--model-path", "/models/x", "--out", "/tmp/x.json"])
    assert args.grid == worker.CONTEXT_BATCH_GRID


def test_cli_accepts_a_grid_override():
    args = worker.parse_args(
        ["--model-path", "/models/x", "--out", "/tmp/x.json", "--grid-spec", "512:1,2"]
    )
    assert args.grid == ((512, (1, 2)),)


def test_cli_rejects_a_malformed_grid_override():
    with pytest.raises(SystemExit):
        worker.parse_args(
            ["--model-path", "/models/x", "--out", "/tmp/x.json", "--grid-spec", "512:"]
        )


# ---------------------------------------------------------------------------
# Merging per-context artifacts.
#
# One artifact per context length is forced by the engine: it initialises a
# torch.distributed process group on construction and refuses to do so twice in
# one process. The smoke run discovered this the direct way, with three of six
# cells lost to "trying to initialize the default process group twice".
# ---------------------------------------------------------------------------


def _worker_payload(grid, model):
    rows = [
        _row(context, batch, model(context, batch))
        for context, batch in worker.enumerate_cells(grid)
    ]
    return worker.build_payload(
        rows,
        [{"context_length": grid[0][0], "identity": {"num_kvcache_blocks": 100}}],
        model_path="/models/qwen3-8b",
        enforce_eager=False,
        seed=1,
        gpu_memory_utilization=0.85,
        warmup_steps=32,
        measured_steps=24,
        grid=grid,
    )


def test_merge_payloads_reassembles_the_preregistered_grid():
    """Per-context runs that jointly cover the grid must count as pre-registered."""
    model = lambda L, B: 13.05 + 0.000151 * L * B  # noqa: E731
    payloads = [
        _worker_payload((group,), model) for group in worker.CONTEXT_BATCH_GRID
    ]
    merged = worker.enumerate_cells()
    combined = verdict.merge_payloads(payloads)
    assert combined["grid_is_preregistered"] is True
    assert combined["missing_preregistered_cells"] == []
    assert combined["extra_cells"] == []
    assert len(combined["rows"]) == len(merged)
    assert combined["source_artifact_count"] == len(worker.CONTEXT_BATCH_GRID)


def test_merge_payloads_reports_a_lost_context_group():
    """A context length whose engine failed must leave the grid visibly incomplete."""
    model = lambda L, B: 13.05 + 0.000151 * L * B  # noqa: E731
    payloads = [
        _worker_payload((group,), model) for group in worker.CONTEXT_BATCH_GRID[:-1]
    ]
    combined = verdict.merge_payloads(payloads)
    assert combined["grid_is_preregistered"] is False
    dropped = worker.CONTEXT_BATCH_GRID[-1]
    for batch in dropped[1]:
        assert (dropped[0], batch) in combined["missing_preregistered_cells"]


def test_merge_payloads_rejects_a_duplicated_cell():
    """Two artifacts covering the same cell would double-weight it in the fit."""
    model = lambda L, B: 13.05 + 0.000151 * L * B  # noqa: E731
    payload = _worker_payload((worker.CONTEXT_BATCH_GRID[0],), model)
    with pytest.raises(ValueError, match="more than one artifact"):
        verdict.merge_payloads([payload, payload])


def test_merge_payloads_rejects_an_empty_list():
    with pytest.raises(ValueError):
        verdict.merge_payloads([])


def test_merge_payloads_notes_cells_outside_the_preregistered_grid():
    payload = _worker_payload(((1024, (1, 2)),), lambda L, B: 5.0 + 0.0001 * L * B)
    combined = verdict.merge_payloads([payload])
    assert combined["extra_cells"] == [(1024, 1), (1024, 2)]
    assert combined["grid_is_preregistered"] is False


def test_merged_preregistered_grid_can_pass():
    model = lambda L, B: 13.05 + 0.000151 * L * B  # noqa: E731
    payloads = [
        _worker_payload((group,), model) for group in worker.CONTEXT_BATCH_GRID
    ]
    report = verdict.build_report(verdict.merge_payloads(payloads))
    assert report["grid_is_preregistered"] is True
    assert report["verdict"] == "PASS"


def test_merged_partial_grid_is_downgraded_not_failed():
    model = lambda L, B: 13.05 + 0.000151 * L * B  # noqa: E731
    payloads = [
        _worker_payload((group,), model) for group in worker.CONTEXT_BATCH_GRID[:2]
    ]
    report = verdict.build_report(verdict.merge_payloads(payloads))
    assert report["verdict"] == "INCONCLUSIVE"
    assert report["missing_preregistered_cells"]


def test_preregistered_cell_list_matches_the_worker_grid():
    """The two modules must not drift apart silently."""
    assert set(verdict.PREREGISTERED_CELLS) == set(worker.enumerate_cells())


def test_cli_accepts_repeated_payload_arguments(tmp_path, capsys):
    model = lambda L, B: 13.05 + 0.000151 * L * B  # noqa: E731
    paths = []
    for index, group in enumerate(worker.CONTEXT_BATCH_GRID):
        path = tmp_path / f"part{index}.json"
        path.write_text(json.dumps(_worker_payload((group,), model)), encoding="utf-8")
        paths.append(str(path))
    argv = []
    for path in paths:
        argv += ["--payload", path]
    out = tmp_path / "verdict.json"
    code = verdict.main(argv + ["--out", str(out)])
    capsys.readouterr()
    assert code == 0
    assert json.loads(out.read_text(encoding="utf-8"))["verdict"] == "PASS"


# ---------------------------------------------------------------------------
# Window stability.
#
# The smoke run produced 3.46 ms at batch 1 and 31.3 ms at batches 2 and 4 on a
# 0.6B model, with torch.compile recompilation markers landing inside the timed
# window. Those numbers may describe compilation rather than decoding, so the
# gate must be able to say "the measurement is unsound" separately from "the
# model is wrong".
# ---------------------------------------------------------------------------


def test_measurement_drift_detects_a_window_still_warming_up():
    samples = [40.0, 38.0, 36.0, 20.0, 15.0, 15.0, 15.0, 15.0]
    drift = worker.measurement_drift(samples)
    assert drift["ratio"] < 1.0 - verdict.MAX_DRIFT_DEVIATION


def test_measurement_drift_is_near_one_for_a_settled_window():
    samples = [15.0, 15.2, 14.9, 15.1, 15.0, 14.8, 15.2, 15.0]
    drift = worker.measurement_drift(samples)
    assert abs(drift["ratio"] - 1.0) <= verdict.MAX_DRIFT_DEVIATION


def test_measurement_drift_needs_enough_samples():
    assert worker.measurement_drift([1.0, 2.0]) is None


def _row_with_drift(context, batch, median, ratio):
    row = _row(context, batch, median)
    row["drift"] = {
        "first_half_median_ms": median,
        "second_half_median_ms": median * ratio,
        "ratio": ratio,
    }
    return row


def test_verdict_is_inconclusive_when_a_window_drifts():
    """A drifting window must not be reported as a refutation of Stage 0."""
    rows = [
        _row_with_drift(context, batch, 13.05 + 0.000151 * context * batch, 1.0)
        for context, batch in worker.enumerate_cells()
    ]
    rows[3] = _row_with_drift(
        rows[3]["context_length"], rows[3]["batch"], rows[3]["step"]["median_ms"], 0.70
    )
    report = verdict.build_report({"rows": rows, "payload_sha256": "x"})
    assert report["verdict"] == "INCONCLUSIVE"
    assert "warming up" in report["consequence"]
    check = next(c for c in report["checks"] if c["name"] == "window_stability")
    assert check["passed"] is False


def test_verdict_passes_when_every_window_is_stable():
    rows = [
        _row_with_drift(context, batch, 13.05 + 0.000151 * context * batch, 1.01)
        for context, batch in worker.enumerate_cells()
    ]
    report = verdict.build_report({"rows": rows, "payload_sha256": "x"})
    assert report["verdict"] == "PASS"


def test_window_stability_check_is_vacuous_without_drift_data():
    report = verdict.build_report(
        _payload_from(lambda L, B: 13.05 + 0.000151 * L * B, worker.enumerate_cells())
    )
    check = next(c for c in report["checks"] if c["name"] == "window_stability")
    assert check["vacuous"] is True


def test_a_drifting_window_does_not_mask_a_real_model_failure():
    """Both problems can be present; the measurement complaint takes precedence.

    A drifting window makes the fitted constants untrustworthy, so reporting FAIL
    against Stage 0 on that data would be asserting more than is known.
    """
    rows = [
        _row_with_drift(context, batch, 13.05 + 9.0 * batch + 0.000151 * context * batch, 0.6)
        for context, batch in worker.enumerate_cells()
    ]
    report = verdict.build_report({"rows": rows, "payload_sha256": "x"})
    assert report["verdict"] == "INCONCLUSIVE"


def test_warmup_default_was_raised_after_the_smoke_run():
    """Eight steps let recompilation into the window; the default is now higher."""
    assert worker.WARMUP_STEPS >= 32


def test_drift_threshold_is_declared_in_a_sane_range():
    assert 0.0 < verdict.MAX_DRIFT_DEVIATION <= 0.15


# ---------------------------------------------------------------------------
# The CUDA graph regime boundary.
#
# Measured on Qwen3-8B, A100, at L=16384: 15.3 ms at batch 1 against 43.3 ms at
# batch 2, then a smooth rise to 62.1 ms at batch 16. The eager path over the same
# cells shows 40.4 ms against 42.0 ms, so the jump is graph replay, not compute.
# Decode graphs are captured for batches 1, 2, 4 and 8 but only take effect at 1.
# ---------------------------------------------------------------------------


def _point(context, batch, step_ms):
    return {
        "context_length": context,
        "batch": batch,
        "kv_tokens": context * batch,
        "step_ms": step_ms,
        "step_stdev_ms": 0.5,
        "sample_count": 24,
        "drift_ratio": 1.0,
    }


def test_regime_discontinuity_detects_the_measured_graph_boundary():
    """Reproduces the observed graph-path numbers."""
    points = [
        _point(16384, 1, 15.263),
        _point(16384, 2, 43.295),
        _point(32768, 1, 17.607),
        _point(32768, 2, 46.409),
    ]
    found = verdict.regime_discontinuity(points)
    assert found["regimes_differ"] is True
    assert found["max_ratio"] > 2.5
    assert len(found["comparisons"]) == 2


def test_regime_discontinuity_is_quiet_on_the_eager_path():
    """The eager numbers over the same cells must not trip the detector."""
    points = [
        _point(16384, 1, 40.402),
        _point(16384, 2, 41.969),
        _point(32768, 1, 43.709),
        _point(32768, 2, 47.640),
    ]
    found = verdict.regime_discontinuity(points)
    assert found["regimes_differ"] is False
    assert found["max_ratio"] < verdict.MAX_REGIME_STEP_RATIO


def test_regime_discontinuity_needs_both_batches_at_one_context():
    points = [_point(16384, 2, 43.0), _point(16384, 4, 45.0)]
    assert verdict.regime_discontinuity(points) is None


def test_regime_discontinuity_reports_the_worst_context():
    points = [
        _point(16384, 1, 15.0),
        _point(16384, 2, 45.0),
        _point(32768, 1, 40.0),
        _point(32768, 2, 44.0),
    ]
    found = verdict.regime_discontinuity(points)
    assert found["max_ratio"] == pytest.approx(3.0)


def _graph_path_rows():
    """A full V2 grid where batch 1 is on the graph path and batch >= 2 is not."""
    rows = []
    for context, batch in worker.enumerate_cells():
        if batch == 1:
            step = 13.0 + 0.000151 * context * batch
        else:
            step = 40.0 + 0.000090 * context * batch
        row = _row(context, batch, step)
        row["drift"] = {"ratio": 1.0, "first_half_median_ms": step,
                        "second_half_median_ms": step}
        rows.append(row)
    return rows


def test_fit_excludes_batch_one_when_the_regimes_differ():
    """Mixing two execution paths into one fit would describe neither."""
    report = verdict.build_report({"rows": _graph_path_rows(), "payload_sha256": "x"})
    assert report["regime_discontinuity"]["regimes_differ"] is True
    assert "batch >= 2" in report["fit_scope"]
    assert all(point["batch"] >= 2 for point in report["fitted_points"])
    # The batch-1 cells are still reported, just not fitted.
    assert any(point["batch"] == 1 for point in report["measured_points"])


def test_excluding_batch_one_does_not_weaken_coverage():
    """Coverage must still be judged on every measured cell, not the fit subset.

    Otherwise dropping batch 1 from the fit would also quietly drop it from the
    coverage requirement, and the gate would stop noticing missing cells.
    """
    report = verdict.build_report({"rows": _graph_path_rows(), "payload_sha256": "x"})
    check = next(c for c in report["checks"] if c["name"] == "grid_coverage")
    assert check["passed"] is True
    rows = [row for row in _graph_path_rows() if row["batch"] != 1]
    partial = verdict.build_report({"rows": rows, "payload_sha256": "x"})
    coverage = next(c for c in partial["checks"] if c["name"] == "grid_coverage")
    assert coverage["passed"] is False


def test_a_clean_multi_batch_regime_can_still_pass():
    """Separating the regimes must remain capable of returning PASS."""
    report = verdict.build_report({"rows": _graph_path_rows(), "payload_sha256": "x"})
    assert report["verdict"] == "PASS"
    assert report["model_m1"]["r_squared"] >= verdict.MIN_R_SQUARED


def test_the_fit_uses_the_multi_batch_constant_not_the_batch_one_constant():
    """The whole point: Stage 0's c0 came from batch 1 and is not the serving c0."""
    report = verdict.build_report({"rows": _graph_path_rows(), "payload_sha256": "x"})
    # Planted: 13.0 at batch 1, 40.0 for the multi-sequence regime.
    assert report["model_m1"]["c0_ms"] == pytest.approx(40.0, abs=0.5)
    comparison = report["stage0_comparison"]
    assert comparison["c0_ratio"] > 2.5


def test_single_regime_data_is_fitted_whole():
    """With no discontinuity, batch 1 stays in the fit."""
    rows = []
    for context, batch in worker.enumerate_cells():
        step = 40.0 + 0.000090 * context * batch
        row = _row(context, batch, step)
        row["drift"] = {"ratio": 1.0, "first_half_median_ms": step,
                        "second_half_median_ms": step}
        rows.append(row)
    report = verdict.build_report({"rows": rows, "payload_sha256": "x"})
    assert report["regime_discontinuity"]["regimes_differ"] is False
    assert report["fit_scope"] == "all measured batches"
    assert any(point["batch"] == 1 for point in report["fitted_points"])


def test_render_explains_the_regime_boundary():
    text = verdict.render(
        verdict.build_report({"rows": _graph_path_rows(), "payload_sha256": "x"})
    )
    assert "regime boundary" in text
    assert "different execution paths" in text
    assert "fitted on:" in text


def test_regime_threshold_is_declared_in_a_sane_range():
    assert 1.0 < verdict.MAX_REGIME_STEP_RATIO <= 2.0


# ---------------------------------------------------------------------------
# Sample dispersion.
#
# The second full run produced a cell with a median of 60.6 ms and a standard
# deviation of 109.7 ms, meaning at least one step took hundreds of milliseconds.
# The median absorbed it and every existing check passed. A robust statistic over
# a contaminated sample is still a contaminated measurement.
# ---------------------------------------------------------------------------


def _row_with_spread(context, batch, median, stdev):
    row = _row(context, batch, median)
    row["step"]["stdev_ms"] = stdev
    row["drift"] = {
        "ratio": 1.0,
        "first_half_median_ms": median,
        "second_half_median_ms": median,
    }
    return row


def _clean_rows():
    return [
        _row_with_spread(context, batch, 40.0 + 0.00009 * context * batch, 0.5)
        for context, batch in worker.enumerate_cells()
    ]


def test_dispersion_ratio_is_extracted_from_the_measurement():
    payload = {"rows": [_row_with_spread(8192, 2, 60.628, 109.718)]}
    points, _rejected = verdict.extract_points(payload)
    assert points[0]["dispersion_ratio"] == pytest.approx(109.718 / 60.628)


def test_verdict_is_inconclusive_when_a_cell_is_wildly_dispersed():
    """Reproduces the observed contaminated cell."""
    rows = _clean_rows()
    rows[10] = _row_with_spread(
        rows[10]["context_length"], rows[10]["batch"], 60.628, 109.718
    )
    report = verdict.build_report({"rows": rows, "payload_sha256": "x"})
    assert report["verdict"] == "INCONCLUSIVE"
    assert "steady decoding" in report["consequence"]
    check = next(c for c in report["checks"] if c["name"] == "sample_dispersion")
    assert check["passed"] is False


def test_normal_measurement_noise_does_not_trip_the_dispersion_check():
    """Observed eager spreads are around 1% of the median."""
    report = verdict.build_report({"rows": _clean_rows(), "payload_sha256": "x"})
    check = next(c for c in report["checks"] if c["name"] == "sample_dispersion")
    assert check["passed"] is True
    assert report["verdict"] == "PASS"


def test_dispersion_is_judged_on_every_measured_cell_not_just_fitted_ones():
    """A contaminated batch-1 cell must not escape by being excluded from the fit."""
    rows = []
    for context, batch in worker.enumerate_cells():
        if batch == 1:
            rows.append(_row_with_spread(context, 1, 13.0 + 0.000151 * context, 40.0))
        else:
            rows.append(
                _row_with_spread(context, batch, 40.0 + 0.00009 * context * batch, 0.5)
            )
    report = verdict.build_report({"rows": rows, "payload_sha256": "x"})
    assert "batch >= 2" in report["fit_scope"]
    assert report["verdict"] == "INCONCLUSIVE"
    check = next(c for c in report["checks"] if c["name"] == "sample_dispersion")
    assert check["passed"] is False


def test_dispersion_check_is_vacuous_without_spread_data():
    report = verdict.build_report(
        _payload_from(lambda L, B: 13.05 + 0.000151 * L * B, worker.enumerate_cells())
    )
    check = next(c for c in report["checks"] if c["name"] == "sample_dispersion")
    assert check["vacuous"] is False or check["passed"] is True


def test_dispersion_threshold_is_declared_in_a_sane_range():
    assert 0.0 < verdict.MAX_DISPERSION_RATIO <= 0.5


# ---------------------------------------------------------------------------
# Context must leave room for the tokens the worker generates.
# ---------------------------------------------------------------------------


def test_grid_contexts_leave_room_for_generated_tokens():
    """A request is rejected when prompt plus generated tokens exceeds the limit.

    The second run lost every 40960 cell to exactly this, having asked for the
    full positional limit as prompt with nothing left to decode into.
    """
    generated = worker.WARMUP_STEPS + worker.MEASURED_STEPS + 2
    for context, _batch in worker.enumerate_cells():
        assert context + generated <= 40960, (context, generated)


def test_grid_contexts_are_block_aligned():
    """The KV cache is paged at 256 tokens, so ragged contexts waste a partial block."""
    for context, _batch in worker.enumerate_cells():
        assert context % 256 == 0, context


def test_preregistered_lists_stay_in_step_after_the_context_change():
    assert set(verdict.PREREGISTERED_CELLS) == set(worker.enumerate_cells())
    assert 40448 in {context for context, _batch in worker.enumerate_cells()}


# ---------------------------------------------------------------------------
# Dispatch provenance. The first GATE A run measured an eager fallback while
# believing it measured the serving path, so the rerun has to be able to prove
# which path each step took, and to say so when it cannot.
# ---------------------------------------------------------------------------


class _Runner:
    def __init__(self, events, quest_events=None):
        self._events = list(events)
        self._quest_events = list(quest_events or [])

    def cuda_graph_dispatch_observation(self):
        return self._events.pop(0) if self._events else None

    def quest_activation_observation(self):
        return (
            self._quest_events.pop(0)
            if self._quest_events
            else None
        )


class _Engine:
    def __init__(self, runner):
        self.model_runner = runner


def test_multi_sequence_graph_kwargs_allowlists_every_batch_above_one():
    """A batch missing from the allowlist silently runs eager, which is the bug."""
    kwargs = worker.multi_sequence_graph_kwargs([1, 2, 4, 8, 16, 32])
    assert kwargs["multi_sequence_cuda_graphs"] is True
    assert kwargs["multi_sequence_cuda_graph_batch_allowlist"] == (2, 4, 8, 16, 32)
    assert kwargs["multi_sequence_cuda_graph_max_entries"] >= 5


def test_multi_sequence_graph_kwargs_is_empty_when_only_batch_one_is_measured():
    assert worker.multi_sequence_graph_kwargs([1]) == {}


def test_worker_defaults_keep_the_multi_sequence_graph_path_off():
    """The flag is opt-in, so an unflagged run must remain comparable to the old one."""
    args = worker.parse_args(["--model-path", "m", "--out", "o"])
    assert args.multi_sequence_cuda_graphs is False


def test_payload_records_whether_the_graph_path_was_requested():
    payload = worker.build_payload(
        [], [], model_path="m", enforce_eager=False, seed=1,
        gpu_memory_utilization=0.85, warmup_steps=1, measured_steps=1,
        multi_sequence_cuda_graphs=True,
    )
    assert payload["configuration"]["multi_sequence_cuda_graphs"] is True


def test_cli_accepts_kv8_quest_configuration():
    args = worker.parse_args(
        [
            "--model-path",
            "m",
            "--out",
            "o",
            "--kv-quant-bits",
            "8",
            "--quest-top-k-blocks",
            "16",
            "--quest-min-seq-len",
            "512",
            "--quest-min-saved-blocks",
            "128",
        ]
    )
    assert args.kv_quant_bits == 8
    assert args.quest_top_k_blocks == 16
    assert args.quest_min_seq_len == 512
    assert args.quest_min_saved_blocks == 128


def test_payload_records_requested_quant_and_quest_configuration():
    payload = worker.build_payload(
        [],
        [],
        model_path="m",
        enforce_eager=True,
        seed=1,
        gpu_memory_utilization=0.85,
        warmup_steps=1,
        measured_steps=1,
        kv_quant_bits=8,
        quest_top_k_blocks=16,
        quest_min_seq_len=512,
        quest_min_saved_blocks=128,
    )
    assert payload["configuration"]["kv_quant_bits"] == 8
    assert payload["configuration"]["quest_top_k_blocks"] == 16
    assert payload["configuration"]["quest_min_seq_len"] == 512
    assert payload["configuration"]["quest_min_saved_blocks"] == 128


def test_load_engine_passes_kv8_quest_configuration(monkeypatch):
    captured = {}

    class FakeLLM:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setitem(sys.modules, "tinyvllm", types.SimpleNamespace(LLM=FakeLLM))
    worker._load_engine(
        model_path="m",
        max_model_len=8192,
        enforce_eager=True,
        gpu_memory_utilization=0.85,
        max_num_seqs=19,
        kv_blocks=640,
        kv_quant_bits=8,
        quest_top_k_blocks=16,
        quest_min_seq_len=512,
        quest_min_saved_blocks=128,
    )
    assert captured["kv_quant_bits"] == 8
    assert captured["quest_top_k_blocks"] == 16
    assert captured["quest_min_seq_len"] == 512
    assert captured["quest_min_saved_blocks"] == 128


def test_runner_passes_nondefault_quest_configuration_to_worker():
    source = (
        HERE / "run_kvcapacity_step_scaling_remote.sh"
    ).read_text(encoding="utf-8")
    assert 'QUEST_TOP_K_BLOCKS="${QUEST_TOP_K_BLOCKS:--1}"' in source
    assert 'QUEST_MIN_SEQ_LEN="${QUEST_MIN_SEQ_LEN:-512}"' in source
    assert 'QUEST_MIN_SAVED_BLOCKS="${QUEST_MIN_SAVED_BLOCKS:-0}"' in source
    assert 'REMOTE_ARGS+=(--quest-top-k-blocks "${QUEST_TOP_K_BLOCKS}")' in source
    assert 'REMOTE_ARGS+=(--quest-min-seq-len "${QUEST_MIN_SEQ_LEN}")' in source
    assert (
        'REMOTE_ARGS+=(--quest-min-saved-blocks '
        '"${QUEST_MIN_SAVED_BLOCKS}")'
    ) in source


def _quest_event(
    observation_id,
    *,
    reason="active",
    resolved_top_k=16,
    saved_blocks=128,
    batch_size=8,
):
    return {
        "observation_id": observation_id,
        "requested_top_k": 16,
        "resolved_top_k": resolved_top_k,
        "min_seq_len": 512,
        "min_saved_blocks": 128,
        "saved_blocks": saved_blocks,
        "batch_size": batch_size,
        "reason": reason,
    }


def test_quest_activation_observation_reads_through_model_runner():
    event = _quest_event(1)
    engine = _Engine(_Runner([], [event]))

    assert worker.quest_activation_observation(engine) == event


def test_quest_activation_tracker_marks_repeated_and_missing_events():
    event = _quest_event(7)
    engine = _Engine(_Runner([], [event, event]))
    tracker = worker.QuestActivationTracker()

    assert tracker.observe(engine)["status"] == "valid"
    assert tracker.observe(engine)["status"] == "unpublished"
    assert tracker.observe(object()) == {"status": "unobserved"}


def test_quest_activation_tracker_marks_contradictory_event_invalid():
    event = _quest_event(
        1,
        reason="active",
        resolved_top_k=-1,
    )
    tracker = worker.QuestActivationTracker()

    observed = tracker.observe(_Engine(_Runner([], [event])))

    assert observed["status"] == "invalid"


def test_summarise_quest_activation_preserves_auditable_counts():
    events = [
        {"status": "valid", **_quest_event(1)},
        {
            "status": "valid",
            **_quest_event(
                2,
                reason="below_saved_blocks",
                resolved_top_k=-1,
                saved_blocks=96,
                batch_size=6,
            ),
        },
    ]

    summary = worker.summarise_quest_activation(events)

    assert summary == {
        "steps": 2,
        "status_counts": {"valid": 2},
        "reason_counts": {
            "active": 1,
            "below_saved_blocks": 1,
        },
        "resolved_top_k_counts": {"-1": 1, "16": 1},
        "saved_blocks_min": 96,
        "saved_blocks_max": 128,
        "all_valid": True,
    }


def test_dispatch_label_separates_eager_reasons():
    assert worker.dispatch_label({"dispatch": "graph"}) == "graph"
    assert worker.dispatch_label(
        {"dispatch": "eager", "fallback_reason": "batch_not_allowlisted"}
    ) == "eager:batch_not_allowlisted"
    assert worker.dispatch_label(
        {"dispatch": "eager", "fallback_reason": None, "cache_state": "observing"}
    ) == "eager:observing"


def test_dispatch_label_marks_an_engine_that_publishes_nothing():
    """Silence is not evidence of graph replay; it must be recorded as unobserved."""
    assert worker.dispatch_label(None) == "unobserved"


def test_dispatch_observation_survives_an_engine_without_the_hook():
    assert worker.dispatch_observation(object()) is None


def test_dispatch_observation_reads_through_the_model_runner():
    engine = _Engine(_Runner([{"dispatch": "graph"}]))
    assert worker.dispatch_observation(engine) == {"dispatch": "graph"}


def test_summarise_dispatch_flags_a_window_that_was_not_all_graph():
    summary = worker.summarise_dispatch(["graph"] * 20 + ["eager:observing"] * 4)
    assert summary["steps"] == 24
    assert summary["graph_steps"] == 20
    assert summary["all_graph"] is False
    assert summary["graph_share"] == pytest.approx(20 / 24)


def test_summarise_dispatch_confirms_a_clean_graph_window():
    summary = worker.summarise_dispatch(["graph"] * 24)
    assert summary["all_graph"] is True
    assert summary["graph_share"] == 1.0


def test_summarise_dispatch_reports_a_fully_eager_window_as_such():
    """This is the shape the previous run would have produced had it been audited."""
    summary = worker.summarise_dispatch(["eager:feature_disabled"] * 24)
    assert summary["graph_steps"] == 0
    assert summary["graph_share"] == 0.0
    assert summary["counts"] == {"eager:feature_disabled": 24}


def test_summarise_dispatch_is_none_without_steps():
    assert worker.summarise_dispatch([]) is None


def test_dispatch_tracker_marks_a_step_that_published_nothing():
    """A stale event must not be read as evidence about the current step.

    The batch-1 cell of the first msgraph smoke run returned
    `eager:unsupported_mode` left over from a prefill step while it was actually
    replaying the batch-1 graph. A label that lies in the safe direction is still
    a label that can validate the wrong run.
    """
    engine = _Engine(
        _Runner(
            [
                {"step_id": 7, "dispatch": "graph"},
                {"step_id": 7, "dispatch": "graph"},
                {"step_id": 8, "dispatch": "eager", "fallback_reason": "capture_failed"},
            ]
        )
    )
    tracker = worker.DispatchTracker()
    assert tracker.observe(engine) == "graph"
    assert tracker.observe(engine) == "unpublished"
    assert tracker.observe(engine) == "eager:capture_failed"


def test_dispatch_tracker_reports_an_engine_that_publishes_nothing_at_all():
    tracker = worker.DispatchTracker()
    assert tracker.observe(object()) == "unobserved"


def test_summarise_dispatch_does_not_count_unpublished_steps_as_graph():
    summary = worker.summarise_dispatch(["graph"] * 4 + ["unpublished"] * 8)
    assert summary["graph_share"] == pytest.approx(4 / 12)
    assert summary["all_graph"] is False


def test_multi_sequence_graph_kwargs_lifts_the_one_time_capture_budgets():
    """The default 2 s single-capture budget rejected batch 2 in the smoke run.

    The first capture in a process also pays torch.compile for the shape, so at
    the default the cheapest measured batch is the one that gets left on the
    eager path, which reads as a batch-scaling cliff instead of as a policy.
    """
    kwargs = worker.multi_sequence_graph_kwargs([2, 4, 8, 16, 32])
    assert kwargs["multi_sequence_cuda_graph_max_single_capture_ns"] > 2_000_000_000
    assert kwargs["multi_sequence_cuda_graph_max_total_capture_ns"] > 5_000_000_000
    assert kwargs["multi_sequence_cuda_graph_max_reserved_bytes"] >= 512 * 1024 * 1024


# ---------------------------------------------------------------------------
# Device provenance. The first graph-path wall sweep got 676 KV blocks where the
# same settings yield 1145 on an idle card, because roughly 25 GiB belonged to
# another process. The artifact recorded nothing about it.
# ---------------------------------------------------------------------------


def test_device_memory_snapshot_reports_all_three_fields():
    snapshot = worker.device_memory_before_load()
    assert set(snapshot) == {
        "device_free_bytes_before_load",
        "device_total_bytes",
        "device_foreign_share",
    }


def test_device_memory_snapshot_degrades_without_cuda_instead_of_raising():
    """Runs on this laptop, where importing torch or querying CUDA fails."""
    snapshot = worker.device_memory_before_load()
    if snapshot["device_total_bytes"] is None:
        assert snapshot["device_free_bytes_before_load"] is None
        assert snapshot["device_foreign_share"] is None
    else:
        assert snapshot["device_total_bytes"] > 0


def test_payload_identity_can_carry_the_device_snapshot():
    """A contaminated run has to be recognisable from the artifact alone."""
    identity = {"kv_capacity_tokens": 173056}
    identity.update(
        {
            "device_free_bytes_before_load": 55 * 2**30,
            "device_total_bytes": 80 * 2**30,
            "device_foreign_share": 0.3125,
        }
    )
    payload = worker.build_payload(
        [], [{"context_length": 2048, "identity": identity}],
        model_path="m", enforce_eager=False, seed=1,
        gpu_memory_utilization=0.85, warmup_steps=1, measured_steps=1,
    )
    recorded = payload["engines"][0]["identity"]
    assert recorded["device_foreign_share"] == pytest.approx(0.3125)


# ---------------------------------------------------------------------------
# The KV feasibility guard. Its message read as a contradiction in the sweep
# artifact ("43486543872 required, 45034242048 available" followed by a refusal),
# because it compared a figure it had not actually used.
# ---------------------------------------------------------------------------


def test_resident_bytes_charge_whole_blocks_and_generated_tokens():
    """A sequence at L=8192 decoding 34 more tokens pins 33 blocks, not 32."""
    per_token = worker.KV_BYTES_PER_TOKEN
    naive = 8192 * per_token
    real = worker.cell_resident_kv_bytes(8192, 1, generated_tokens=34)
    assert real == 33 * 256 * per_token
    assert real > naive


def test_resident_bytes_scale_with_batch():
    one = worker.cell_resident_kv_bytes(2048, 1, generated_tokens=34)
    assert worker.cell_resident_kv_bytes(2048, 144, generated_tokens=34) == 144 * one


def test_cell_fits_matches_what_the_engine_actually_did_at_the_wall():
    """Measured against the graph-path wall sweep, which is the only arbiter.

    With 305408 tokens of visible capacity, L=2048 B=128 measured cleanly at
    24/24 steps on the target batch, B=140 could not form the batch at all, and
    B=144 pins more blocks than exist. A guard that disagrees with any of those
    three is either skipping measurable cells or admitting unmeasurable ones.
    """
    available = 305408 * worker.KV_BYTES_PER_TOKEN
    assert worker.cell_fits(2048, 128, available, generated_tokens=34) is True
    assert worker.cell_fits(2048, 140, available, generated_tokens=34) is False
    assert worker.cell_fits(2048, 144, available, generated_tokens=34) is False


def test_cell_fits_admits_the_8192_wall_cell_the_percentage_guard_refused():
    """L=8192 B=40 pins 1320 of 1343 blocks; the old 5% headroom rejected it.

    That cell is the eager run's wall point, so refusing it silently removed the
    one measurement the capacity ratio needs.
    """
    available = 343808 * worker.KV_BYTES_PER_TOKEN
    assert worker.cell_fits(8192, 40, available, generated_tokens=34) is True
    assert worker.cell_fits(8192, 44, available, generated_tokens=34) is False


def test_cell_fits_is_permissive_when_capacity_is_unknown():
    """Provenance may be missing; a missing budget must not silently skip cells."""
    assert worker.cell_fits(8192, 32, None) is True
