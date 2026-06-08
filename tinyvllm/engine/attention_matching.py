"""Fast KV Compaction via Attention Matching primitives.

This module implements the paper's fast AM-HighestAttnKeys variant at the
single (layer, KV-head) tensor level:

1. choose compact keys from original keys using RMS/mean/max attention scores,
2. fit log-attention bias beta by bounded non-negative least squares so selected
   keys preserve the original unnormalized attention mass,
3. fit compact values by least squares so compact attention outputs match the
   original attention outputs on reference queries.

The functions are intentionally independent from the scheduler/KV-cache layout so
we can validate the math first and then plug it into engine-specific cache flows.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class AttentionMatchedKV:
    """Compacted KV tensors for one layer/head."""

    keys: torch.Tensor
    beta: torch.Tensor
    values: torch.Tensor
    indices: torch.Tensor


def _scaled_scores(queries: torch.Tensor, keys: torch.Tensor) -> torch.Tensor:
    head_dim = keys.shape[-1]
    return (queries @ keys.T).to(torch.float32) / (head_dim ** 0.5)


def attention_output(
    queries: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    beta: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return softmax(QK^T/sqrt(d) + beta) @ V in fp32 math."""
    scores = _scaled_scores(queries, keys)
    if beta is not None:
        scores = scores + beta.to(torch.float32).view(1, -1)
    weights = torch.softmax(scores, dim=-1)
    return weights @ values.to(torch.float32)


def highest_attention_key_indices(
    keys: torch.Tensor,
    queries: torch.Tensor,
    budget: int,
    score_method: str = "rms",
) -> torch.Tensor:
    """Select key indices with largest aggregate attention over reference queries."""
    num_keys = keys.shape[0]
    if budget <= 0:
        raise ValueError("budget must be positive")
    if budget >= num_keys:
        return torch.arange(num_keys, device=keys.device, dtype=torch.long)
    if score_method not in ("rms", "mean", "max"):
        raise ValueError("score_method must be 'rms', 'mean', or 'max'")

    weights = torch.softmax(_scaled_scores(queries, keys), dim=-1)
    if score_method == "rms":
        key_scores = torch.sqrt(torch.mean(weights * weights, dim=0))
    elif score_method == "mean":
        key_scores = torch.mean(weights, dim=0)
    else:
        key_scores = torch.max(weights, dim=0).values

    return torch.topk(key_scores, k=budget, largest=True).indices


def solve_box_nnls(
    design: torch.Tensor,
    target: torch.Tensor,
    lower_bound: float = 1e-12,
    upper_bound: float | None = None,
    iters: int = 0,
) -> torch.Tensor:
    """Solve min ||design @ x - target|| with non-negative box constraints."""
    design = design.to(torch.float32)
    target = target.to(torch.float32)
    try:
        solution = torch.linalg.lstsq(design, target.unsqueeze(1)).solution.squeeze(1)
    except Exception:
        lam = 1e-6
        if design.shape[0] < design.shape[1]:
            gram = design @ design.T
            gram = 0.5 * (gram + gram.T)
            gram.diagonal().add_(lam)
            solution = design.T @ torch.linalg.solve(gram, target)
        else:
            gram = design.T @ design
            gram = 0.5 * (gram + gram.T)
            gram.diagonal().add_(lam)
            solution = torch.linalg.solve(gram, design.T @ target)

    solution = solution.clamp_min(lower_bound)
    if upper_bound is not None:
        solution = solution.clamp_max(upper_bound)
    if iters <= 0:
        return solution

    # Projected gradient refinement. This is optional; it keeps v0 fast by default.
    probe = torch.randn(design.shape[1], device=design.device, dtype=torch.float32)
    probe = probe / probe.norm().clamp_min(1e-6)
    for _ in range(3):
        left = design @ probe
        left = left / left.norm().clamp_min(1e-6)
        probe = design.T @ left
        probe = probe / probe.norm().clamp_min(1e-6)
    lipschitz = torch.dot(probe, design.T @ (design @ probe)).clamp_min(1e-6)
    step = 1.0 / lipschitz
    for _ in range(iters):
        grad = design.T @ (design @ solution - target)
        solution = (solution - step * grad).clamp_min(lower_bound)
        if upper_bound is not None:
            solution = solution.clamp_max(upper_bound)
    return solution


def fit_attention_bias(
    keys: torch.Tensor,
    queries: torch.Tensor,
    selected_indices: torch.Tensor,
    beta_bound: float = 3.0,
    nnls_iters: int = 0,
) -> torch.Tensor:
    """Fit beta so compact keys preserve original unnormalized attention mass."""
    scores = _scaled_scores(queries, keys)
    exp_scores = torch.exp(scores - torch.max(scores, dim=1, keepdim=True).values)
    target_mass = exp_scores.sum(dim=1)
    design = exp_scores[:, selected_indices]
    upper = float(torch.exp(torch.tensor(beta_bound)))
    weights = solve_box_nnls(
        design,
        target_mass,
        lower_bound=float(torch.exp(torch.tensor(-beta_bound))),
        upper_bound=upper,
        iters=nnls_iters,
    )
    return torch.log(weights).to(keys.dtype)


def fit_compacted_values(
    keys: torch.Tensor,
    values: torch.Tensor,
    queries: torch.Tensor,
    selected_indices: torch.Tensor,
    beta: torch.Tensor,
    ridge_lambda: float = 0.0,
) -> torch.Tensor:
    """Fit compact values by least squares attention-output matching."""
    target = attention_output(queries, keys, values)
    compact_keys = keys[selected_indices]
    compact_scores = _scaled_scores(queries, compact_keys) + beta.to(torch.float32).view(1, -1)
    design = torch.softmax(compact_scores, dim=-1)
    if ridge_lambda <= 0:
        try:
            solution = torch.linalg.lstsq(design, target).solution
        except Exception:
            solution = torch.linalg.pinv(design) @ target
    else:
        gram = design.T @ design
        gram = 0.5 * (gram + gram.T)
        gram.diagonal().add_(ridge_lambda)
        solution = torch.linalg.solve(gram, design.T @ target)
    return solution.to(values.dtype)


def attention_matching_highest_keys(
    keys: torch.Tensor,
    values: torch.Tensor,
    queries: torch.Tensor,
    budget: int,
    score_method: str = "rms",
    beta_bound: float = 3.0,
    nnls_iters: int = 0,
    ridge_lambda: float = 0.0,
) -> AttentionMatchedKV:
    """Run AM-HighestAttnKeys for one layer/head."""
    selected = highest_attention_key_indices(keys, queries, budget, score_method)
    beta = fit_attention_bias(keys, queries, selected, beta_bound, nnls_iters)
    compact_values = fit_compacted_values(keys, values, queries, selected, beta, ridge_lambda)
    return AttentionMatchedKV(
        keys=keys[selected].contiguous(),
        beta=beta.contiguous(),
        values=compact_values.contiguous(),
        indices=selected.contiguous(),
    )
