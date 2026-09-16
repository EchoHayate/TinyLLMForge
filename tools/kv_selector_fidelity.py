"""Selector fidelity gate: can a query-aware selector keep the attention mass that matters?

Why this file exists
--------------------
The end-to-end needle gate in `tools/eval_needle.py` scores 100.0% for every arm we
have ever run, including arms we expect to be bad. A gate that cannot fail cannot
license a decision, and the CPU-offload plan needs a decision: it wants to touch only
k of L tokens per step (k/L around 11% inside the GPU weight-read window), which means
discarding ~89% of the context. Nothing in the current harness can see that.

This tool measures the selector directly, with the model's output out of the loop, so
the measurement cannot be rescued by a task that happens to be easy for other reasons:

  recovered mass   sum of the true softmax attention probabilities that survive inside
                   the selected token set
  needle coverage  whether the tokens that actually carry the answer survive
  recall@32        how many of the true top-32 tokens survive

The same measurement runs on deliberately bad arms (random / recency-only / uniform
stride). If a bad arm scores as well as the real selector, the verdict is
NON_DISCRIMINATIVE and the numbers must not be used as evidence. That is the point:
this gate is built to be able to fail.

Two selectors are measured, not one:

  quest_per_head      the per-channel min/max upper bound applied per kv head, i.e.
                      what the Quest paper describes
  quest_shared_heads  what tinyvllm actually ships: `quest_score_kernel` sums the bound
                      over every kv head and channel, producing ONE block ranking shared
                      by all heads. Cheaper, but if heads disagree about which blocks
                      matter, this arm pays for it. The gap between the two is reported.

Inputs are a real post-RoPE Q/K dump from `tools/dump_needle_qk.py`. No synthetic
attention appears in the measurement path; synthetic tensors appear only in the unit
tests, and only to check that the estimator computes what it claims.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Sequence

import numpy as np

# A bad arm scoring within this margin of the real selector means the configuration
# cannot tell selection quality apart, so it cannot be used as evidence.
DISCRIMINATIVE_MARGIN = 0.10
# Above this, a bad arm is already "good enough" and the configuration is saturated.
SATURATION_MASS = 0.95
# Coverage is close to binary per head (the answer tokens are either in the selected set
# or they are not), so a real selector must beat the bad arms by a wide margin, not a
# statistical whisker.
COVERAGE_MARGIN = 0.50

ARMS = (
    "oracle_kvhead",
    "quest_per_head",
    "quest_shared_heads",
    "recency",
    "sink_recency",
    "uniform_stride",
    "random",
)
BAD_ARMS = ("recency", "sink_recency", "uniform_stride", "random")
TOP_M = 32


def softmax_rows(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float64)
    x = x - x.max(axis=-1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=-1, keepdims=True)


def unit_bounds(seq_len: int, granularity: int) -> np.ndarray:
    """Selection units as [start, end) rows. The last unit may be short."""
    if granularity <= 0:
        raise ValueError("granularity must be positive")
    starts = np.arange(0, seq_len, granularity, dtype=np.int64)
    ends = np.minimum(starts + granularity, seq_len)
    return np.stack([starts, ends], axis=1)


def quest_unit_scores(q_repr: np.ndarray, kmin: np.ndarray, kmax: np.ndarray) -> np.ndarray:
    """Quest's per-channel upper bound: sum_d max(q_d * kmin_d, q_d * kmax_d).

    Identical to `quest_score_kernel` in tinyvllm/layers/attention.py, which is why the
    result is a statement about the engine's selector rather than an idealized one.
    """
    return np.maximum(q_repr * kmin, q_repr * kmax).sum(axis=-1)


def unit_minmax(k_all: np.ndarray, bounds: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-unit per-channel min/max for every kv head: [n_units, n_kv_heads, dim]."""
    seq_len, n_kv, dim = k_all.shape
    n_units = bounds.shape[0]
    gran = int(bounds[0, 1] - bounds[0, 0])
    if n_units * gran == seq_len:
        view = k_all.reshape(n_units, gran, n_kv, dim)
        return view.min(axis=1), view.max(axis=1)
    kmin = np.empty((n_units, n_kv, dim), dtype=k_all.dtype)
    kmax = np.empty((n_units, n_kv, dim), dtype=k_all.dtype)
    for u, (s, e) in enumerate(bounds):
        chunk = k_all[s:e]
        kmin[u] = chunk.min(axis=0)
        kmax[u] = chunk.max(axis=0)
    return kmin, kmax


def _force_keep(order: np.ndarray, n_units: int, k: int) -> np.ndarray:
    """Keep unit 0 (attention sink) and the last unit (recency), as the engine does.

    The forced units are charged against the budget; if k < 2 they still both survive,
    so `selected_token_frac` is reported separately and is the honest cost measure.
    """
    picked = [u for u in (0, n_units - 1) if 0 <= u < n_units]
    for u in order:
        if len(picked) >= k:
            break
        if int(u) not in picked:
            picked.append(int(u))
    return np.array(sorted(set(picked)), dtype=np.int64)


class LayerCache:
    """Everything about one layer that does not depend on k or on the arm."""

    def __init__(self, q: np.ndarray, k_all: np.ndarray):
        self.q = q.astype(np.float64)
        self.k_all = k_all
        self.seq_len, self.n_kv_heads, self.dim = k_all.shape
        self.n_q_heads = q.shape[0]
        self.group = self.n_q_heads // self.n_kv_heads
        self.scale = 1.0 / np.sqrt(self.dim)
        # Engine semantics: the block-score query is the group-wise amax of q.
        self.q_repr = self.q.reshape(self.n_kv_heads, self.group, self.dim).max(axis=1)

        self.probs = np.empty((self.n_q_heads, self.seq_len), dtype=np.float64)
        for h in range(self.n_q_heads):
            g = h // self.group
            k_head = k_all[:, g, :].astype(np.float64)
            self.probs[h] = softmax_rows((self.q[h] @ k_head.T) * self.scale)
        self.top_idx = np.argsort(-self.probs, axis=1)[:, :TOP_M]
        self._minmax: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        self._unit_max_prob: dict[int, np.ndarray] = {}

    def minmax(self, granularity: int, bounds: np.ndarray):
        if granularity not in self._minmax:
            self._minmax[granularity] = unit_minmax(self.k_all, bounds)
        return self._minmax[granularity]

    def unit_max_prob(self, bounds: np.ndarray) -> np.ndarray:
        """Per kv head, the largest true probability inside each unit: [n_units, n_kv].

        Ranking by probability equals ranking by score, so the oracle needs no extra
        matmul - it reuses the exact distribution the metric is measured against.
        """
        gran = int(bounds[0, 1] - bounds[0, 0])
        if gran in self._unit_max_prob:
            return self._unit_max_prob[gran]
        n_units = bounds.shape[0]
        out = np.empty((n_units, self.n_kv_heads), dtype=np.float64)
        for g in range(self.n_kv_heads):
            group_best = self.probs[g * self.group:(g + 1) * self.group].max(axis=0)
            if n_units * gran == self.seq_len:
                out[:, g] = group_best.reshape(n_units, gran).max(axis=1)
            else:
                for u, (s, e) in enumerate(bounds):
                    out[u, g] = group_best[s:e].max()
        self._unit_max_prob[gran] = out
        return out


def select_units(arm: str, cache: LayerCache, kv_head: int, bounds: np.ndarray,
                 k_units: int, *, kmin, kmax, unit_max_prob, rng) -> np.ndarray:
    n_units = bounds.shape[0]
    k_units = int(min(max(k_units, 1), n_units))

    if arm == "recency":
        return np.arange(n_units - k_units, n_units, dtype=np.int64)
    if arm == "sink_recency":
        tail = np.arange(max(n_units - (k_units - 1), 0), n_units, dtype=np.int64) if k_units > 1 else np.array([], np.int64)
        return np.array(sorted({0} | {int(u) for u in tail}), dtype=np.int64)
    if arm == "uniform_stride":
        return np.unique(np.linspace(0, n_units - 1, num=k_units).round().astype(np.int64))
    if arm == "random":
        pool = np.arange(1, max(n_units - 1, 1), dtype=np.int64)
        take = min(max(k_units - 2, 0), pool.size)
        chosen = rng.choice(pool, size=take, replace=False) if take else np.array([], np.int64)
        return _force_keep(np.asarray(chosen, dtype=np.int64), n_units, k_units)
    if arm == "oracle_kvhead":
        return _force_keep(np.argsort(-unit_max_prob[:, kv_head]), n_units, k_units)
    if arm == "quest_per_head":
        score = quest_unit_scores(cache.q_repr[kv_head], kmin[:, kv_head, :].astype(np.float64),
                                  kmax[:, kv_head, :].astype(np.float64))
        return _force_keep(np.argsort(-score), n_units, k_units)
    if arm == "quest_shared_heads":
        score = quest_unit_scores(cache.q_repr[None, :, :], kmin.astype(np.float64),
                                  kmax.astype(np.float64)).sum(axis=1)
        return _force_keep(np.argsort(-score), n_units, k_units)
    raise ValueError(f"unknown arm: {arm}")


def selected_mask(units: Sequence[int], bounds: np.ndarray, seq_len: int) -> np.ndarray:
    mask = np.zeros(seq_len, dtype=bool)
    for u in units:
        s, e = bounds[int(u)]
        mask[s:e] = True
    return mask


def evaluate_layer(q: np.ndarray, k_all: np.ndarray, *, granularity: int, k_frac: float,
                   needle_positions: Sequence[int], seed: int, cache: LayerCache | None = None) -> dict:
    cache = cache or LayerCache(q, k_all)
    seq_len = cache.seq_len
    bounds = unit_bounds(seq_len, granularity)
    n_units = bounds.shape[0]
    k_units = max(1, int(round(k_frac * n_units)))
    kmin, kmax = cache.minmax(granularity, bounds)
    unit_max_prob = cache.unit_max_prob(bounds)
    needles = np.array([p for p in needle_positions if 0 <= p < seq_len], dtype=np.int64)

    out = {
        "seq_len": int(seq_len),
        "granularity": int(granularity),
        "k_frac_requested": float(k_frac),
        "n_units": int(n_units),
        "k_units": int(k_units),
        "k_tokens": int(k_units * granularity),
        "k_frac_effective": float(k_units / n_units),
        "arms": {},
    }
    rng = np.random.default_rng(seed)
    for arm in ARMS:
        masks = [
            selected_mask(
                select_units(arm, cache, g, bounds, k_units,
                             kmin=kmin, kmax=kmax, unit_max_prob=unit_max_prob, rng=rng),
                bounds, seq_len,
            )
            for g in range(cache.n_kv_heads)
        ]
        mass = np.empty(cache.n_q_heads)
        recall = np.empty(cache.n_q_heads)
        needle_cov = np.empty(cache.n_q_heads)
        for h in range(cache.n_q_heads):
            m = masks[h // cache.group]
            mass[h] = cache.probs[h][m].sum()
            recall[h] = float(m[cache.top_idx[h]].mean())
            needle_cov[h] = float(m[needles].mean()) if needles.size else np.nan
        out["arms"][arm] = {
            "mass_mean": float(mass.mean()),
            "mass_p05": float(np.percentile(mass, 5)),
            "mass_min": float(mass.min()),
            "recall_top32_mean": float(recall.mean()),
            "needle_coverage_mean": (None if needles.size == 0 else float(needle_cov.mean())),
            "selected_token_frac": float(np.mean([m.mean() for m in masks])),
        }
    return out


def verdict_for_cell(cell: dict) -> dict:
    """Judge the cell, preferring answer coverage over recovered attention mass.

    The first version of this gate judged on recovered mass, and the 8192-token runs
    showed why that was wrong: with the attention sink and the recency tail force-kept,
    a selector that keeps *only* those two units already recovers 0.89-0.94 of the mass
    while dropping the answer tokens entirely. Mass is therefore nearly blind to
    retrieval quality - the margin between the real selector and a known-bad arm is
    around 0.02, inside noise - whereas answer coverage separates them by 0.75-1.00.

    So coverage is the primary criterion whenever the dump carries needle positions, and
    the mass margin is retained as a diagnostic. Mass is only used to judge when there
    are no needle positions to judge with (synthetic tests, ablations).
    """
    arms = cell["arms"]
    mass = {a: arms[a]["mass_mean"] for a in ARMS}
    worst_bad_mass_name = max(BAD_ARMS, key=lambda a: mass[a])
    mass_margin = mass["quest_per_head"] - mass[worst_bad_mass_name]

    cov = {a: arms[a]["needle_coverage_mean"] for a in ARMS}
    has_coverage = cov["quest_per_head"] is not None

    out = {
        "best_bad_arm": worst_bad_mass_name,
        "margin_vs_best_bad_arm": float(mass_margin),
        "quest_per_head_mass": float(mass["quest_per_head"]),
        "quest_shared_heads_mass": float(mass["quest_shared_heads"]),
        "per_head_minus_shared": float(mass["quest_per_head"] - mass["quest_shared_heads"]),
        "oracle_headroom": float(mass["oracle_kvhead"] - mass["quest_per_head"]),
        "criterion": "coverage" if has_coverage else "mass",
    }

    failures = []
    if mass["oracle_kvhead"] + 1e-9 < mass["quest_per_head"]:
        failures.append("oracle_below_estimator_implementation_bug")

    if has_coverage:
        best_real_cov = max(cov["quest_per_head"], cov["quest_shared_heads"])
        worst_bad_cov_name = max(BAD_ARMS, key=lambda a: cov[a])
        cov_margin = best_real_cov - cov[worst_bad_cov_name]
        out.update({
            "coverage_quest_per_head": float(cov["quest_per_head"]),
            "coverage_quest_shared_heads": float(cov["quest_shared_heads"]),
            "coverage_oracle_by_mass": float(cov["oracle_kvhead"]),
            "coverage_best_bad_arm": worst_bad_cov_name,
            "coverage_margin": float(cov_margin),
            "mass_is_discriminative": bool(mass_margin >= DISCRIMINATIVE_MARGIN),
        })
        if cov[worst_bad_cov_name] >= 0.95:
            failures.append(f"saturated_bad_arm_coverage_{worst_bad_cov_name}_{cov[worst_bad_cov_name]:.3f}")
        if cov_margin < COVERAGE_MARGIN:
            failures.append(f"coverage_margin_{cov_margin:.3f}_below_{COVERAGE_MARGIN}")
    else:
        if mass[worst_bad_mass_name] >= SATURATION_MASS:
            failures.append(f"saturated_bad_arm_{worst_bad_mass_name}_{mass[worst_bad_mass_name]:.3f}")
        if mass_margin < DISCRIMINATIVE_MARGIN:
            failures.append(f"margin_{mass_margin:.3f}_below_{DISCRIMINATIVE_MARGIN}")

    out["failures"] = failures
    out["discriminative"] = not failures
    return out


def run(dump_path: str, meta_path: str, granularities: Sequence[int],
        k_fracs: Sequence[float], seed: int) -> dict:
    with np.load(dump_path) as z:
        q_all = z["q"].astype(np.float32)
        k_all = z["k"].astype(np.float32)
        layers = z["layers"].astype(int).tolist()
    with open(meta_path, encoding="utf-8") as f:
        meta = json.load(f)

    cells = []
    for li, layer in enumerate(layers):
        cache = LayerCache(q_all[li], k_all[li])
        for granularity in granularities:
            for k_frac in k_fracs:
                cell = evaluate_layer(
                    q_all[li], k_all[li],
                    granularity=int(granularity), k_frac=float(k_frac),
                    needle_positions=meta.get("answer_needle_positions", []),
                    seed=seed + 1000 * li, cache=cache,
                )
                cell["layer"] = int(layer)
                cell["verdict"] = verdict_for_cell(cell)
                cells.append(cell)
        del cache

    return {
        "dump": os.path.basename(dump_path),
        "meta": meta,
        "discriminative_margin_threshold": DISCRIMINATIVE_MARGIN,
        "saturation_mass_threshold": SATURATION_MASS,
        "classification": "DISCRIMINATIVE" if any(c["verdict"]["discriminative"] for c in cells) else "NON_DISCRIMINATIVE",
        "cells": cells,
    }


def format_report(report: dict) -> str:
    meta = report.get("meta", {})
    lines = [
        f"variant={meta.get('variant')} model={meta.get('model')} seq_len={meta.get('seq_len')} "
        f"model_answered_correctly={meta.get('model_answered_correctly')}",
        f"classification={report['classification']}",
        f"{'layer':>5} {'gran':>5} {'k/L':>6} | {'cov_ph':>7} {'cov_sh':>7} {'cov_bad':>7} {'covmrg':>7}"
        f" | {'mass_ph':>7} {'mass_bad':>8} {'massmrg':>8} | {'disc':>4}",
    ]
    for c in report["cells"]:
        a, v = c["arms"], c["verdict"]
        mass_bad = a[v["best_bad_arm"]]["mass_mean"]
        if v.get("criterion") == "coverage":
            cov_txt = (f"{v['coverage_quest_per_head']:>7.3f} {v['coverage_quest_shared_heads']:>7.3f} "
                       f"{a[v['coverage_best_bad_arm']]['needle_coverage_mean']:>7.3f} {v['coverage_margin']:>+7.3f}")
        else:
            cov_txt = f"{'n/a':>7} {'n/a':>7} {'n/a':>7} {'n/a':>7}"
        lines.append(
            f"{c['layer']:>5} {c['granularity']:>5} {c['k_frac_effective']*100:>5.1f}% | {cov_txt}"
            f" | {a['quest_per_head']['mass_mean']:>7.4f} {mass_bad:>8.4f} "
            f"{v['margin_vs_best_bad_arm']:>+8.4f} | {'yes' if v['discriminative'] else 'no':>4}"
        )
    return "\n".join(lines)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dump", required=True, help="npz from tools/dump_needle_qk.py")
    p.add_argument("--meta", required=True, help="json sidecar from tools/dump_needle_qk.py")
    p.add_argument("--granularities", type=int, nargs="+", default=[256, 64, 32],
                   help="selection unit size in tokens; 32 matches the 128 KiB gather chunk")
    p.add_argument("--k-fracs", type=float, nargs="+", default=[0.02, 0.05, 0.11, 0.25, 0.50])
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--out-json", default=None)
    args = p.parse_args()

    report = run(args.dump, args.meta, args.granularities, args.k_fracs, args.seed)
    print(format_report(report))
    if args.out_json:
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"\nwrote {args.out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
