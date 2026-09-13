#!/usr/bin/env python3
"""Stage 0 gate for the latent KV capacity line.

Analytic only. No third-party imports, no GPU, no network. Runs in well under a
second on a laptop, because every later stage re-runs it as an entry gate.

What it does
------------
1. Loads the measured inputs (decode-step fit, warm turn profile, device budget)
   from the frozen provenance block below, each carrying the artifact it came
   from. No number in this file is invented.
2. Sweeps a frozen matrix of (context length, decode batch, KV representation,
   offload) and derives serving capacity for every row.
3. Checks anchoring and structural invariants. Anchors assert that the model
   reproduces numbers already measured in prior artifacts, so a modelling
   regression is caught rather than believed. Two anchors already caught two
   real modelling errors in this gate's first run; see the plan document.
4. Emits the operating envelope: where, if anywhere, a lossy KV compressor beats
   the best *lossless* alternative by the pre-registered margin, at a reachable
   decode batch, without a latency regression.
5. Writes a deterministic JSON artifact with a payload_sha256.

Usage
-----
    python3 tools/kvcapacity_breakeven_gate.py --print-summary
    python3 tools/kvcapacity_breakeven_gate.py --out artifact.json
"""

import argparse
import hashlib
import json
import os
import sys
import types


def _install_stub_packages():
    """Import `tinyvllm.kvcapacity` without importing `tinyvllm`.

    `tinyvllm/__init__.py` pulls torch. The gate must stay dependency-light, so
    parent packages are stubbed and only the leaf module is really imported. The
    runtime code keeps ordinary absolute imports.
    """
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    for name, path in (
        ("tinyvllm", os.path.join(repo_root, "tinyvllm")),
        ("tinyvllm.kvcapacity", os.path.join(repo_root, "tinyvllm", "kvcapacity")),
    ):
        if name in sys.modules:
            continue
        module = types.ModuleType(name)
        module.__path__ = [path]
        sys.modules[name] = module


_install_stub_packages()

from tinyvllm.kvcapacity.capacity_model import (  # noqa: E402
    CompressionSpec,
    DecodeStepFit,
    DeviceBudget,
    ModelGeometry,
    TurnProfile,
    capacity_gain,
    evaluate_point,
)

BYTES_PER_GIB = 1024 ** 3

_ERRATUM = (
    "docs/superpowers/plans/"
    "2026-09-11-latent-action-speculation-stage1b-erratum.md"
)
_STEP2 = (
    "docs/superpowers/plans/"
    "2026-09-12-latent-action-speculation-stage1b-step2.md"
)

# ---------------------------------------------------------------------------
# Frozen provenance. Every measured number carries the artifact that produced it.
# ---------------------------------------------------------------------------

PROVENANCE = {
    "decode_step_fit": {
        "constant_ms": 13.05,
        "per_token_us": 0.151,
        "source": _ERRATUM,
        "derived_from": (
            "experiments/agentspec_engine_demand/"
            "engine-demand-measure-a100-20260910-2258/engine_demand.json"
        ),
        "note": (
            "Affine fit of the Qwen3-8B CUDA-graph decode step against resident "
            "KV tokens on one A100 80GB PCIe at batch 1. The batch term is an "
            "extrapolation of this fit, not a measurement, and is the first "
            "thing Stage 1a must confirm."
        ),
    },
    "turn_profile": {
        "warm_prefill_seconds": 0.045,
        "decode_steps": 31,
        "source": _ERRATUM,
        "note": (
            "Warm prefill is what the engine recomputes when block-hash prefix "
            "caching hits. decode_steps is recovered from the measured warm "
            "demand at 16384: (0.5262 - 0.045) / 0.015521 = 31.0."
        ),
    },
    "geometry": {
        "name": "Qwen3-8B",
        "layers": 36,
        "kv_heads": 8,
        "head_dim": 128,
        "dtype_bytes": 2,
        "note": "GQA baseline. 2 * 36 * 8 * 128 * 2 = 147456 bytes per token.",
    },
    "device_budget": {
        "kv_gib": 47.6,
        "host_gib": 256.0,
        "host_bandwidth_gb_per_s": 25.0,
        "source": _ERRATUM,
        "note": (
            "A100 80GB at gpu_memory_utilization 0.85 after weights and "
            "workspace. Host link is PCIe4 x16 effective one-way."
        ),
    },
    "workload": {
        "tool_seconds": 5.0,
        "utilization_target": 0.6,
        "source": _STEP2,
        "note": (
            "The declared operating point of the closed speculation line, "
            "reused unchanged so the two lines remain comparable."
        ),
    },
}

# Pre-registered decision thresholds. Fixed before this gate was implemented.
# They must not be edited to accommodate a result.
PREREGISTERED = {
    "min_capacity_gain": 1.30,
    "max_latency_ratio": 1.05,
    "context_lengths_of_interest": [16384, 32768, 65536, 131072],
    "conservative_attention_time_ratio": 1.0,
    "note": (
        "min_capacity_gain is the margin a lossy compressor must clear against "
        "the best lossless alternative, not against a do-nothing baseline. "
        "max_latency_ratio closes the metrics trap where capacity is bought by "
        "making every turn slower. conservative_attention_time_ratio is the phi "
        "assumed until Stage 1a measures it: bytes saved, no time saved."
    ),
}

# ---------------------------------------------------------------------------
# Frozen sweep matrix
# ---------------------------------------------------------------------------

CONTEXT_LENGTHS = (16384, 32768, 65536, 131072)
DECODE_BATCHES = (1, 2, 4, 8, 16, 32)

# MLA at DeepSeek-V2 proportions applied to this geometry:
#   baseline per layer per token = 2 * 8 * 128 * 2 = 4096 bytes
#   MLA      per layer per token = (d_c + d_r) * 2 = (512 + 64) * 2 = 1152 bytes
#   ratio = 4096 / 1152 = 3.5556
MLA_RATIO = 4096.0 / 1152.0

REPRESENTATIONS = (
    CompressionSpec(name="baseline_gqa", kv_ratio=1.0, attention_time_ratio=1.0),
    CompressionSpec(
        name="int8_kv", kv_ratio=2.0, attention_time_ratio=0.5, lossless=False
    ),
    CompressionSpec(
        name="mla512_ideal",
        kv_ratio=MLA_RATIO,
        attention_time_ratio=1.0 / MLA_RATIO,
        lossless=False,
    ),
    CompressionSpec(
        name="mla512_neutral",
        kv_ratio=MLA_RATIO,
        attention_time_ratio=1.0,
        lossless=False,
    ),
    CompressionSpec(
        name="mla512_adverse",
        kv_ratio=MLA_RATIO,
        attention_time_ratio=1.5,
        lossless=False,
    ),
    CompressionSpec(
        name="perfect_compressor",
        kv_ratio=1.0e6,
        attention_time_ratio=0.0,
        lossless=False,
    ),
)

LOSSY_NAMES = tuple(spec.name for spec in REPRESENTATIONS if not spec.lossless)


def build_inputs():
    geometry = ModelGeometry(
        name=PROVENANCE["geometry"]["name"],
        layers=PROVENANCE["geometry"]["layers"],
        kv_heads=PROVENANCE["geometry"]["kv_heads"],
        head_dim=PROVENANCE["geometry"]["head_dim"],
        dtype_bytes=PROVENANCE["geometry"]["dtype_bytes"],
    )
    fit = DecodeStepFit(
        constant_ms=PROVENANCE["decode_step_fit"]["constant_ms"],
        per_token_us=PROVENANCE["decode_step_fit"]["per_token_us"],
    )
    turn = TurnProfile(
        warm_prefill_seconds=PROVENANCE["turn_profile"]["warm_prefill_seconds"],
        decode_steps=PROVENANCE["turn_profile"]["decode_steps"],
        tool_seconds=PROVENANCE["workload"]["tool_seconds"],
    )
    budget = DeviceBudget(
        kv_bytes=int(PROVENANCE["device_budget"]["kv_gib"] * BYTES_PER_GIB),
        utilization_target=PROVENANCE["workload"]["utilization_target"],
        host_bytes=int(PROVENANCE["device_budget"]["host_gib"] * BYTES_PER_GIB),
        host_bandwidth_bytes_per_second=(
            PROVENANCE["device_budget"]["host_bandwidth_gb_per_s"] * 1.0e9
        ),
    )
    return geometry, fit, turn, budget


def build_matrix():
    geometry, fit, turn, budget = build_inputs()
    points = {}
    for length in CONTEXT_LENGTHS:
        for batch in DECODE_BATCHES:
            for spec in REPRESENTATIONS:
                for offload in (False, True):
                    points[(length, batch, spec.name, offload)] = evaluate_point(
                        geometry=geometry,
                        fit=fit,
                        turn=turn,
                        budget=budget,
                        compression=spec,
                        context_length=length,
                        decode_batch=batch,
                        offload=offload,
                    )
    return points


def best_lossless(points, length, batch):
    """Best *reachable* capacity obtainable without any lossy representation.

    The competitor set is deliberately the free, already-shipped, exact options:
    resident baseline KV, and baseline KV with lossless host offload. An
    unreachable configuration is not a competitor.
    """
    candidates = []
    for offload in (False, True):
        point = points[(length, batch, "baseline_gqa", offload)]
        name = "baseline_gqa+offload" if offload else "baseline_gqa"
        candidates.append((name, point))
    reachable = [item for item in candidates if item[1].feasible]
    pool = reachable if reachable else candidates
    return max(pool, key=lambda item: item[1].concurrent_agents)


def build_envelope(points):
    """Where does a lossy compressor beat the best lossless alternative?"""
    threshold = PREREGISTERED["min_capacity_gain"]
    latency_cap = PREREGISTERED["max_latency_ratio"]
    entries = []
    for length in CONTEXT_LENGTHS:
        for batch in DECODE_BATCHES:
            competitor_name, competitor = best_lossless(points, length, batch)
            reference = points[(length, batch, "baseline_gqa", False)]
            for name in LOSSY_NAMES:
                for offload in (False, True):
                    point = points[(length, batch, name, offload)]
                    gain_vs_lossless = capacity_gain(point, competitor)
                    gain_vs_baseline = capacity_gain(point, reference)
                    latency_ratio = (
                        point.user_latency_seconds / reference.user_latency_seconds
                    )
                    clears = bool(
                        point.feasible
                        and gain_vs_lossless >= threshold
                        and latency_ratio <= latency_cap
                    )
                    entries.append(
                        {
                            "context_length": length,
                            "decode_batch": batch,
                            "compression": name,
                            "offload": offload,
                            "competitor": competitor_name,
                            "competitor_agents": round(competitor.concurrent_agents, 6),
                            "competitor_feasible": competitor.feasible,
                            "agents": round(point.concurrent_agents, 6),
                            "feasible": point.feasible,
                            "gain_vs_lossless": round(gain_vs_lossless, 6),
                            "gain_vs_baseline": round(gain_vs_baseline, 6),
                            "latency_ratio": round(latency_ratio, 6),
                            "binding_constraint": point.binding_constraint,
                            "clears_threshold": clears,
                        }
                    )
    return entries


def crossover_batches(envelope, compression):
    """Smallest reachable decode batch clearing the margin, per context length."""
    result = {}
    for length in CONTEXT_LENGTHS:
        clearing = [
            entry["decode_batch"]
            for entry in envelope
            if entry["context_length"] == length
            and entry["compression"] == compression
            and not entry["offload"]
            and entry["clears_threshold"]
        ]
        result[str(length)] = min(clearing) if clearing else None
    return result


def best_over_reachable_batch(points, compression, offload=False):
    """Best capacity a representation can actually reach, over all batches.

    Supplementary to the pre-registered same-batch decision rule. The same-batch
    comparison is the decision; this table exists because "how many agents fit"
    is a batch-free question, and because it forces the latency at the chosen
    batch into the open instead of letting a capacity number hide it.
    """
    result = {}
    for length in CONTEXT_LENGTHS:
        best = None
        for batch in DECODE_BATCHES:
            point = points[(length, batch, compression, offload)]
            if not point.feasible:
                continue
            if best is None or point.concurrent_agents > best.concurrent_agents:
                best = point
        result[str(length)] = (
            None
            if best is None
            else {
                "decode_batch": best.decode_batch,
                "agents": round(best.concurrent_agents, 6),
                "user_latency_seconds": round(best.user_latency_seconds, 6),
                "binding_constraint": best.binding_constraint,
            }
        )
    return result


def reachable_batches(points, compression, offload=False):
    """Largest reachable decode batch, per context length."""
    result = {}
    for length in CONTEXT_LENGTHS:
        reachable = [
            batch
            for batch in DECODE_BATCHES
            if points[(length, batch, compression, offload)].feasible
        ]
        result[str(length)] = max(reachable) if reachable else None
    return result


# ---------------------------------------------------------------------------
# Invariants
# ---------------------------------------------------------------------------


def _close(actual, expected, tolerance):
    return abs(actual - expected) <= tolerance


def check_invariants(points, envelope):
    geometry, fit, _turn, _budget = build_inputs()
    checks = []

    def record(name, passed, detail):
        checks.append({"name": name, "passed": bool(passed), "detail": detail})

    # --- anchors against already-measured artifacts -------------------------

    kv_per_token = geometry.kv_bytes_per_token
    record(
        "anchor_kv_bytes_per_token",
        kv_per_token == 147456,
        f"{kv_per_token} bytes, expected 147456",
    )
    record(
        "anchor_kv_mib_per_token",
        _close(kv_per_token / (1024 ** 2), 0.141, 0.001),
        f"{kv_per_token / (1024 ** 2):.6f} MiB, erratum reported 0.141",
    )
    record(
        "anchor_decode_step_16384",
        _close(fit.step_ms(16384, 1), 15.521, 0.05),
        f"{fit.step_ms(16384, 1):.3f} ms, measured 15.521 ms",
    )
    record(
        "anchor_kv_attention_share_16384",
        _close(fit.kv_attention_share(16384, 1), 0.159, 0.005),
        f"{fit.kv_attention_share(16384, 1):.4f}, erratum reported 0.159",
    )
    record(
        "anchor_kv_attention_share_131072",
        _close(fit.kv_attention_share(131072, 1), 0.603, 0.01),
        f"{fit.kv_attention_share(131072, 1):.4f}, erratum reported 0.603",
    )

    base_16k = points[(16384, 1, "baseline_gqa", False)]
    record(
        "anchor_warm_demand_16384",
        _close(base_16k.gpu_phase_seconds, 0.5262, 0.005),
        f"{base_16k.gpu_phase_seconds:.4f} s, erratum reported 0.5262 s",
    )
    record(
        "anchor_memory_ceiling_16384",
        _close(base_16k.memory_ceiling, 21.2, 0.3),
        f"{base_16k.memory_ceiling:.2f} agents, erratum reported 21.2",
    )
    record(
        "anchor_demand_ceiling_16384",
        _close(base_16k.sustained_agents, 6.3, 0.2),
        f"{base_16k.sustained_agents:.2f} agents, erratum reported 6.3",
    )

    base_131k = points[(131072, 1, "baseline_gqa", False)]
    record(
        "anchor_memory_ceiling_131072",
        _close(base_131k.memory_ceiling, 2.6, 0.2),
        f"{base_131k.memory_ceiling:.2f} agents, erratum reported 2.6",
    )
    record(
        "anchor_demand_ceiling_131072",
        _close(base_131k.sustained_agents, 3.4, 0.2),
        f"{base_131k.sustained_agents:.2f} agents, erratum reported 3.4",
    )
    record(
        "anchor_restore_seconds_16384",
        _close(points[(16384, 1, "baseline_gqa", True)].restore_seconds, 0.0966, 0.003),
        f"{points[(16384, 1, 'baseline_gqa', True)].restore_seconds * 1000:.1f} ms, "
        "erratum reported 97 ms",
    )

    # --- structural invariants ---------------------------------------------

    identity = points[(16384, 4, "baseline_gqa", False)]
    record(
        "identity_representation_has_unit_gain",
        _close(capacity_gain(identity, identity), 1.0, 1e-12),
        "baseline against itself is exactly 1.0",
    )
    record(
        "resident_kv_is_held_across_the_tool_wait",
        all(
            point.duty_memory == 1.0
            for key, point in points.items()
            if key[3] is False
        ),
        "without offload, duty_memory is exactly 1 everywhere",
    )
    record(
        "offload_releases_kv_during_the_tool_wait",
        all(
            points[(length, batch, name, True)].duty_memory < 1.0
            for length in CONTEXT_LENGTHS
            for batch in DECODE_BATCHES
            for name in ("baseline_gqa",)
        ),
        "with offload, duty_memory is strictly below 1 everywhere",
    )
    record(
        "offload_never_reduces_gpu_work",
        all(
            points[(length, batch, name, True)].gpu_phase_seconds
            >= points[(length, batch, name, False)].gpu_phase_seconds - 1e-12
            for length in CONTEXT_LENGTHS
            for batch in DECODE_BATCHES
            for name in ("baseline_gqa", "mla512_neutral")
        ),
        "offload is a memory move, so it cannot shrink the GPU phase",
    )
    record(
        "offload_costs_user_latency",
        all(
            points[(length, batch, "baseline_gqa", True)].user_latency_seconds
            > points[(length, batch, "baseline_gqa", False)].user_latency_seconds
            for length in CONTEXT_LENGTHS
            for batch in DECODE_BATCHES
        ),
        "the restore transfer is charged to user latency everywhere",
    )
    record(
        "compression_reduces_bytes_everywhere",
        all(
            points[(length, batch, "mla512_neutral", False)].bytes_per_agent
            < points[(length, batch, "baseline_gqa", False)].bytes_per_agent
            for length in CONTEXT_LENGTHS
            for batch in DECODE_BATCHES
        ),
        "mla512 bytes per agent below baseline at every point",
    )
    record(
        "kv_share_rises_with_batch",
        all(
            fit.kv_attention_share(length, batch)
            <= fit.kv_attention_share(length, nxt) + 1e-12
            for length in CONTEXT_LENGTHS
            for batch, nxt in zip(DECODE_BATCHES, DECODE_BATCHES[1:])
        ),
        "KV attention share is non-decreasing in decode batch",
    )
    record(
        "capacity_orders_by_attention_time_ratio",
        all(
            points[(length, batch, "mla512_ideal", False)].concurrent_agents
            >= points[(length, batch, "mla512_neutral", False)].concurrent_agents - 1e-12
            >= points[(length, batch, "mla512_adverse", False)].concurrent_agents - 1e-12
            for length in CONTEXT_LENGTHS
            for batch in DECODE_BATCHES
        ),
        "ideal >= neutral >= adverse at every point",
    )
    record(
        "perfect_compressor_capacity_is_bounded",
        all(
            points[(length, batch, "perfect_compressor", False)].concurrent_agents
            < float("inf")
            for length in CONTEXT_LENGTHS
            for batch in DECODE_BATCHES
        ),
        "an infinitely good compressor still hits the batch-invariant step floor",
    )
    record(
        "perfect_compressor_is_demand_bound",
        all(
            points[(length, batch, "perfect_compressor", False)].binding_constraint
            == "demand"
            for length in CONTEXT_LENGTHS
            for batch in DECODE_BATCHES
        ),
        "removing all KV bytes always leaves the compute phase as the binder",
    )

    # --- findings, frozen so a later model change cannot silently erase them --

    record(
        "finding_no_win_at_batch_one_under_conservative_phi",
        all(
            entry["clears_threshold"] is False
            for entry in envelope
            if entry["decode_batch"] == 1 and entry["compression"] == "mla512_neutral"
        ),
        "at batch 1 with phi=1, mla512 clears the margin nowhere",
    )
    record(
        "finding_adverse_phi_never_wins",
        all(
            entry["clears_threshold"] is False
            for entry in envelope
            if entry["compression"] == "mla512_adverse"
        ),
        "if reconstruction costs more time than the bytes saved, nothing wins",
    )

    return checks


# ---------------------------------------------------------------------------
# Artifact
# ---------------------------------------------------------------------------


def build_artifact():
    points = build_matrix()
    envelope = build_envelope(points)
    checks = check_invariants(points, envelope)
    status = "PASS" if all(check["passed"] for check in checks) else "FAIL"

    ordered = [
        points[(length, batch, spec.name, offload)]
        for length in CONTEXT_LENGTHS
        for batch in DECODE_BATCHES
        for spec in REPRESENTATIONS
        for offload in (False, True)
    ]

    payload = {
        "schema": "kvcapacity-stage0-gate/1",
        "provenance": PROVENANCE,
        "preregistered": PREREGISTERED,
        "sweep": {
            "context_lengths": list(CONTEXT_LENGTHS),
            "decode_batches": list(DECODE_BATCHES),
            "representations": [
                {
                    "name": spec.name,
                    "kv_ratio": round(spec.kv_ratio, 6),
                    "attention_time_ratio": round(spec.attention_time_ratio, 6),
                    "lossless": spec.lossless,
                }
                for spec in REPRESENTATIONS
            ],
        },
        "points": [point.as_dict() for point in ordered],
        "envelope": envelope,
        "crossover_batch": {
            name: crossover_batches(envelope, name) for name in LOSSY_NAMES
        },
        "largest_reachable_batch": {
            "baseline_gqa": reachable_batches(points, "baseline_gqa"),
            "baseline_gqa+offload": reachable_batches(
                points, "baseline_gqa", offload=True
            ),
            "mla512_neutral": reachable_batches(points, "mla512_neutral"),
        },
        "best_over_reachable_batch": {
            "baseline_gqa": best_over_reachable_batch(points, "baseline_gqa"),
            "baseline_gqa+offload": best_over_reachable_batch(
                points, "baseline_gqa", offload=True
            ),
            "mla512_ideal": best_over_reachable_batch(points, "mla512_ideal"),
            "mla512_neutral": best_over_reachable_batch(points, "mla512_neutral"),
            "mla512_adverse": best_over_reachable_batch(points, "mla512_adverse"),
        },
        "invariants": checks,
        "status": status,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    payload["payload_sha256"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return payload


def print_summary(artifact):
    print(f"status {artifact['status']}")
    checks = artifact["invariants"]
    passed = sum(1 for check in checks if check["passed"])
    print(f"invariants {passed}/{len(checks)} passed")
    for check in checks:
        if not check["passed"]:
            print(f"  FAIL {check['name']}: {check['detail']}")

    index = {
        (row["context_length"], row["decode_batch"], row["compression"], row["offload"]):
        row
        for row in artifact["points"]
    }
    lengths = artifact["sweep"]["context_lengths"]
    batches = artifact["sweep"]["decode_batches"]

    print()
    print("baseline, resident KV, rho=0.6, tool=5s  (N_dem = agents needed to")
    print("sustain the batch, N_mem = agents the KV budget holds)")
    print(f"{'ctx':>8} {'B':>3} {'step_ms':>8} {'kv_sh':>6} {'lat_s':>7} "
          f"{'N_dem':>7} {'N_mem':>7} {'N':>7} {'bind':>7} {'reach':>6}")
    for length in lengths:
        for batch in batches:
            row = index[(length, batch, "baseline_gqa", False)]
            print(
                f"{length:>8} {batch:>3} {row['step_ms']:>8.2f} "
                f"{row['kv_attention_share']:>6.3f} "
                f"{row['user_latency_seconds']:>7.3f} "
                f"{row['sustained_agents']:>7.2f} {row['memory_ceiling']:>7.2f} "
                f"{row['concurrent_agents']:>7.2f} {row['binding_constraint']:>7} "
                f"{('yes' if row['feasible'] else 'NO'):>6}"
            )

    print()
    print("largest reachable decode batch")
    for name, table in sorted(artifact["largest_reachable_batch"].items()):
        print(f"  {name:>21}: {table}")

    lookup = {
        (row["context_length"], row["decode_batch"], row["compression"], row["offload"]):
        row
        for row in artifact["envelope"]
    }
    print()
    print("mla512 capacity gain vs best reachable lossless alternative (resident)")
    print(f"{'ctx':>8} {'B':>3} {'phi=1/r':>8} {'phi=1':>8} {'phi=1.5':>8} "
          f"{'reach':>6} {'competitor':>21}")
    for length in lengths:
        for batch in batches:
            ideal = lookup[(length, batch, "mla512_ideal", False)]
            neutral = lookup[(length, batch, "mla512_neutral", False)]
            adverse = lookup[(length, batch, "mla512_adverse", False)]
            print(
                f"{length:>8} {batch:>3} "
                f"{ideal['gain_vs_lossless']:>8.3f} "
                f"{neutral['gain_vs_lossless']:>8.3f} "
                f"{adverse['gain_vs_lossless']:>8.3f} "
                f"{('yes' if neutral['feasible'] else 'NO'):>6} "
                f"{neutral['competitor']:>21}"
            )

    print()
    print("best capacity over reachable batches (supplementary, batch-free)")
    print(f"{'ctx':>8} {'representation':>21} {'B*':>3} {'agents':>7} "
          f"{'lat_s':>7} {'bind':>7}")
    for length in lengths:
        for name in (
            "baseline_gqa",
            "baseline_gqa+offload",
            "mla512_neutral",
            "mla512_ideal",
        ):
            row = artifact["best_over_reachable_batch"][name][str(length)]
            if row is None:
                print(f"{length:>8} {name:>21} {'-':>3} {'-':>7} {'-':>7} {'-':>7}")
                continue
            print(
                f"{length:>8} {name:>21} {row['decode_batch']:>3} "
                f"{row['agents']:>7.2f} {row['user_latency_seconds']:>7.3f} "
                f"{row['binding_constraint']:>7}"
            )

    print()
    print(f"pre-registered margin {artifact['preregistered']['min_capacity_gain']}, "
          f"latency cap {artifact['preregistered']['max_latency_ratio']}")
    print("smallest reachable decode batch clearing the margin (None = never):")
    for name, table in sorted(artifact["crossover_batch"].items()):
        print(f"  {name:>19}: {table}")

    print()
    print(f"payload_sha256 {artifact['payload_sha256']}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=None, help="write the JSON artifact here")
    parser.add_argument("--print-summary", action="store_true")
    args = parser.parse_args(argv)

    artifact = build_artifact()
    if args.out:
        directory = os.path.dirname(os.path.abspath(args.out))
        if directory:
            os.makedirs(directory, exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as handle:
            json.dump(artifact, handle, indent=2, sort_keys=True)
            handle.write("\n")
    if args.print_summary or not args.out:
        print_summary(artifact)
    return 0 if artifact["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
