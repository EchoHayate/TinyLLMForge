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


@dataclass
class AttentionMatchingCacheEntry:
    """Cached compact tensors for one sequence/KV-head selector state."""

    compact: AttentionMatchedKV
    step: int


@dataclass
class AttentionMatchingDecodeCache:
    """Tiny per-layer AM compact cache used by decode refresh experiments."""

    entries: dict[tuple, AttentionMatchingCacheEntry]
    step: int = 0
    hits: int = 0
    misses: int = 0

    def __init__(self):
        self.entries = {}
        self.step = 0
        self.hits = 0
        self.misses = 0

    def begin_step(self) -> int:
        self.step += 1
        return self.step

    def get(self, key: tuple, refresh_interval: int) -> AttentionMatchedKV | None:
        entry = self.entries.get(key)
        if entry is None:
            self.misses += 1
            return None
        if refresh_interval <= 0 or self.step - entry.step >= refresh_interval:
            self.misses += 1
            return None
        self.hits += 1
        return entry.compact

    def put(self, key: tuple, compact: AttentionMatchedKV) -> None:
        self.entries[key] = AttentionMatchingCacheEntry(compact=compact, step=self.step)

    def clear(self) -> None:
        self.entries.clear()
        self.step = 0
        self.hits = 0
        self.misses = 0


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


def _fit_selected_output_error(
    keys: torch.Tensor,
    values: torch.Tensor,
    queries: torch.Tensor,
    selected_indices: torch.Tensor,
    target: torch.Tensor,
    ridge_lambda: float,
    beta_bound: float,
) -> torch.Tensor:
    if selected_indices.numel() == 0:
        return torch.mean(target * target)
    beta = fit_attention_bias(keys, queries, selected_indices, beta_bound=beta_bound)
    compact_values = fit_compacted_values(
        keys,
        values,
        queries,
        selected_indices,
        beta,
        ridge_lambda=ridge_lambda,
    )
    pred = attention_output(queries, keys[selected_indices], compact_values, beta)
    return torch.mean((pred - target) ** 2)


def omp_attention_key_indices(
    keys: torch.Tensor,
    values: torch.Tensor,
    queries: torch.Tensor,
    budget: int,
    ridge_lambda: float = 1e-6,
    beta_bound: float = 3.0,
    score_method: str = "rms",
    candidate_pool_size: int | None = None,
) -> torch.Tensor:
    """Select compact key indices by greedy output-matching OMP.

    The objective is the same attention-output MSE used by the AM value-fitting
    stage. To keep long-context smoke runs usable, OMP only searches a small
    HighestAttnKeys candidate pool rather than every historical token. At each
    iteration we try every unselected candidate, refit beta/C_v for the temporary
    set, and keep the candidate with the lowest fitted-output error.
    """
    num_keys = keys.shape[0]
    if budget <= 0:
        raise ValueError("budget must be positive")
    if budget >= num_keys:
        return torch.arange(num_keys, device=keys.device, dtype=torch.long)

    target = attention_output(queries, keys, values)
    selected: list[int] = []
    if candidate_pool_size is None or candidate_pool_size <= 0:
        candidate_pool_size = max(budget + 4, budget * 2)
    candidate_pool_size = max(budget, min(num_keys, candidate_pool_size))
    pool = highest_attention_key_indices(
        keys,
        queries,
        budget=candidate_pool_size,
        score_method=score_method,
    )
    remaining = pool.tolist()
    for _ in range(budget):
        best_candidate = remaining[0]
        best_error: torch.Tensor | None = None
        for candidate in remaining:
            trial = torch.tensor(selected + [candidate], device=keys.device, dtype=torch.long)
            err = _fit_selected_output_error(
                keys,
                values,
                queries,
                trial,
                target,
                ridge_lambda=ridge_lambda,
                beta_bound=beta_bound,
            )
            if best_error is None or float(err.item()) < float(best_error.item()):
                best_error = err
                best_candidate = candidate
        selected.append(best_candidate)
        remaining.remove(best_candidate)
    return torch.tensor(selected, device=keys.device, dtype=torch.long)


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
    return attention_matching_compact_keys(
        keys,
        values,
        queries,
        budget=budget,
        selector="highest",
        score_method=score_method,
        beta_bound=beta_bound,
        nnls_iters=nnls_iters,
        ridge_lambda=ridge_lambda,
    )


def attention_matching_compact_keys(
    keys: torch.Tensor,
    values: torch.Tensor,
    queries: torch.Tensor,
    budget: int,
    selector: str = "highest",
    score_method: str = "rms",
    beta_bound: float = 3.0,
    nnls_iters: int = 0,
    ridge_lambda: float = 0.0,
    omp_candidate_pool_size: int = 0,
) -> AttentionMatchedKV:
    """Run Attention Matching with a configurable key selector."""
    if selector == "highest":
        selected = highest_attention_key_indices(keys, queries, budget, score_method)
    elif selector == "omp":
        selected = omp_attention_key_indices(
            keys,
            values,
            queries,
            budget,
            ridge_lambda=ridge_lambda,
            beta_bound=beta_bound,
            score_method=score_method,
            candidate_pool_size=omp_candidate_pool_size,
        )
    else:
        raise ValueError("selector must be 'highest' or 'omp'")
    beta = fit_attention_bias(keys, queries, selected, beta_bound, nnls_iters)
    compact_values = fit_compacted_values(keys, values, queries, selected, beta, ridge_lambda)
    return AttentionMatchedKV(
        keys=keys[selected].contiguous(),
        beta=beta.contiguous(),
        values=compact_values.contiguous(),
        indices=selected.contiguous(),
    )


def attention_matching_decode(
    queries: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    context_lens: torch.Tensor,
    budget: int,
    selector: str = "highest",
    score_method: str = "rms",
    beta_bound: float = 3.0,
    ridge_lambda: float = 1e-6,
    omp_candidate_pool_size: int = 0,
    cache: AttentionMatchingDecodeCache | None = None,
    cache_refresh_interval: int = 0,
    cache_signatures: tuple | list | None = None,
) -> torch.Tensor:
    """Decode attention through AM compact tensors.

    Parameters
    ----------
    queries:
        `[B, num_q_heads, head_dim]` query tensor for the current decode token.
    keys / values:
        Dense fp cache tensors after optional KV8 dequant, shaped
        `[B, max_seq_len, num_kv_heads, head_dim]`.
    context_lens:
        Valid token count per batch row. Rows may be padded in `keys/values`.
    budget:
        Number of compact KV entries per `(batch,row,kv_head)`.

    The implementation is intentionally eager and loop-based. It is the first
    integration path for `C_k / beta / C_v`; performance work can happen after
    quality is established.
    """
    if budget <= 0:
        raise ValueError("budget must be positive")
    batch, num_q_heads, head_dim = queries.shape
    _, _, num_kv_heads, _ = keys.shape
    assert num_q_heads % num_kv_heads == 0, "GQA requires q heads divisible by KV heads"
    group_size = num_q_heads // num_kv_heads
    out = torch.empty_like(queries)
    if cache is not None and cache_refresh_interval > 0:
        cache.begin_step()
    for b in range(batch):
        seq_len = int(context_lens[b].item())
        if cache_signatures is None:
            cache_signature = (b, seq_len)
        else:
            cache_signature = cache_signatures[b]
        for kv_h in range(num_kv_heads):
            q_start = kv_h * group_size
            q_end = q_start + group_size
            q_group = queries[b, q_start:q_end]
            k_seq = keys[b, :seq_len, kv_h]
            v_seq = values[b, :seq_len, kv_h]
            if budget >= seq_len:
                group_out = attention_output(q_group, k_seq, v_seq)
            else:
                cache_key = (
                    cache_signature,
                    kv_h,
                    selector,
                    budget,
                    score_method,
                    float(beta_bound),
                    float(ridge_lambda),
                    int(omp_candidate_pool_size),
                    str(k_seq.dtype),
                    str(k_seq.device),
                )
                compact = None
                if cache is not None and cache_refresh_interval > 0:
                    compact = cache.get(cache_key, cache_refresh_interval)
                if compact is None:
                    compact = attention_matching_compact_keys(
                        k_seq,
                        v_seq,
                        q_group,
                        budget=budget,
                        selector=selector,
                        score_method=score_method,
                        beta_bound=beta_bound,
                        ridge_lambda=ridge_lambda,
                        omp_candidate_pool_size=omp_candidate_pool_size,
                    )
                    if cache is not None and cache_refresh_interval > 0:
                        cache.put(cache_key, compact)
                group_out = attention_output(q_group, compact.keys, compact.values, compact.beta)
            out[b, q_start:q_end] = group_out.to(out.dtype)
    return out
