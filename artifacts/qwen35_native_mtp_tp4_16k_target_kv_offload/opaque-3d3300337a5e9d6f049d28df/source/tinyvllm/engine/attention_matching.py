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

from dataclasses import dataclass, field

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
    compacts: tuple[AttentionMatchedKV, ...] | None = None
    centroids: torch.Tensor | None = None

    def __post_init__(self):
        if self.compacts is None:
            self.compacts = (self.compact,)


@dataclass
class AttentionMatchingDecodeCache:
    """Tiny per-layer AM compact cache used by decode refresh experiments."""

    entries: dict[tuple, AttentionMatchingCacheEntry]
    step: int = 0
    hits: int = 0
    misses: int = 0
    refit_hits: int = 0
    refit_misses: int = 0
    prefill_builds: int = 0
    last_cluster_indices: list[int] = field(default_factory=list)

    def __init__(self):
        self.entries = {}
        self.refit_entries = {}
        self.step = 0
        self.hits = 0
        self.misses = 0
        self.refit_hits = 0
        self.refit_misses = 0
        self.prefill_builds = 0
        self.last_cluster_indices = []

    def begin_step(self) -> int:
        self.step += 1
        return self.step

    def get(
        self,
        key: tuple,
        refresh_interval: int,
        query: torch.Tensor | None = None,
        cluster_route_top_k: int = 1,
    ) -> AttentionMatchedKV | None:
        entry = self.entries.get(key)
        if entry is None:
            self.misses += 1
            return None
        if refresh_interval <= 0 or self.step - entry.step >= refresh_interval:
            self.misses += 1
            return None
        self.hits += 1
        if query is None or entry.centroids is None or len(entry.compacts) <= 1:
            self.last_cluster_indices.append(0)
            return entry.compact
        compact, route = _route_compact_bank(entry.compacts, entry.centroids, query, cluster_route_top_k)
        self.last_cluster_indices.append(route[0] if len(route) == 1 else route)
        return compact

    def put(self, key: tuple, compact: AttentionMatchedKV) -> None:
        self.entries[key] = AttentionMatchingCacheEntry(compact=compact, step=self.step)

    def get_refit(self, key: tuple, refresh_interval: int) -> AttentionMatchedKV | None:
        entry = self.refit_entries.get(key)
        if entry is None:
            self.refit_misses += 1
            return None
        if refresh_interval <= 1 or self.step - entry.step >= refresh_interval:
            self.refit_misses += 1
            return None
        self.refit_hits += 1
        return entry.compact

    def put_refit(self, key: tuple, compact: AttentionMatchedKV) -> None:
        self.refit_entries[key] = AttentionMatchingCacheEntry(compact=compact, step=self.step)

    def put_bank(
        self,
        key: tuple,
        compacts: tuple[AttentionMatchedKV, ...] | list[AttentionMatchedKV],
        centroids: torch.Tensor,
    ) -> None:
        compact_tuple = tuple(compacts)
        if not compact_tuple:
            raise ValueError("compact bank must contain at least one cluster")
        self.entries[key] = AttentionMatchingCacheEntry(
            compact=compact_tuple[0],
            step=self.step,
            compacts=compact_tuple,
            centroids=centroids.contiguous(),
        )

    def clear(self) -> None:
        self.entries.clear()
        self.refit_entries.clear()
        self.step = 0
        self.hits = 0
        self.misses = 0
        self.refit_hits = 0
        self.refit_misses = 0
        self.prefill_builds = 0
        self.last_cluster_indices = []


@dataclass
class _PendingDecodeRefit:
    b: int
    q_start: int
    q_end: int
    k_seq: torch.Tensor
    v_seq: torch.Tensor
    q_group: torch.Tensor
    compact: AttentionMatchedKV
    refit_key: tuple | None


def _compact_cache_key(
    cache_signature,
    kv_h: int,
    selector: str,
    budget: int,
    score_method: str,
    beta_bound: float,
    ridge_lambda: float,
    omp_candidate_pool_size: int,
    num_clusters: int,
    num_key_spans: int,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple:
    return (
        cache_signature,
        kv_h,
        selector,
        budget,
        score_method,
        float(beta_bound),
        float(ridge_lambda),
        int(omp_candidate_pool_size),
        int(num_clusters),
        int(num_key_spans),
        str(dtype),
        str(device),
    )


def _refit_cache_key(cache_key: tuple, compact: AttentionMatchedKV, mode: str) -> tuple:
    return (
        cache_key,
        "decode_refit",
        mode,
        tuple(int(i) for i in compact.indices.detach().cpu().tolist()),
    )


def _anchor_refit_indices(
    num_keys: int,
    selected: torch.Tensor,
    anchor_budget: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build a small anchor set containing `selected` for approximate refit.

    Returns `(anchors, selected_pos)` where `anchors[selected_pos] == selected`.
    The anchor set is deterministic: uniform positional anchors plus the compact
    key indices. This avoids constructing full-sequence attention targets during
    decode refit while keeping enough global position coverage for a cheap test.
    """
    num_keys = int(num_keys)
    if num_keys <= 0:
        raise ValueError("num_keys must be positive")
    selected = selected.to(dtype=torch.long)
    if selected.numel() == 0:
        raise ValueError("selected must not be empty")
    anchor_budget = max(int(anchor_budget), int(selected.numel()))
    anchor_budget = min(anchor_budget, num_keys)
    uniform = torch.linspace(
        0,
        num_keys - 1,
        anchor_budget,
        device=selected.device,
    ).round().to(torch.long)
    anchors = torch.unique(torch.cat([uniform, selected], dim=0), sorted=True)
    # `selected` was concatenated above, so every selected index must exist.
    selected_pos = (anchors[:, None] == selected[None, :]).to(torch.long).argmax(dim=0)
    return anchors, selected_pos


def _cluster_reference_queries(
    queries: torch.Tensor,
    num_clusters: int,
    iters: int = 4,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Cluster reference queries and return `(labels, centroids)`.

    This lightweight k-means is intentionally small and deterministic: centroids
    are initialized from evenly spaced reference queries, which works well for
    prefill references ordered by position while avoiding extra random state.
    """
    if queries.ndim != 2:
        raise ValueError("queries must be shaped [num_refs, head_dim]")
    num_refs = queries.shape[0]
    if num_refs == 0:
        raise ValueError("queries must contain at least one reference")
    actual_clusters = max(1, min(int(num_clusters), num_refs))
    if actual_clusters == 1:
        labels = torch.zeros(num_refs, device=queries.device, dtype=torch.long)
        return labels, queries.to(torch.float32).mean(dim=0, keepdim=True).to(queries.dtype)

    init = torch.linspace(0, num_refs - 1, actual_clusters, device=queries.device).round().to(torch.long)
    centroids = queries[init].to(torch.float32).contiguous()
    refs = queries.to(torch.float32)
    labels = torch.zeros(num_refs, device=queries.device, dtype=torch.long)
    for _ in range(iters):
        distances = torch.sum((refs[:, None, :] - centroids[None, :, :]) ** 2, dim=-1)
        labels = torch.argmin(distances, dim=1)
        next_centroids = centroids.clone()
        for cluster_idx in range(actual_clusters):
            mask = labels == cluster_idx
            if bool(mask.any().item()):
                next_centroids[cluster_idx] = refs[mask].mean(dim=0)
        centroids = next_centroids
    return labels, centroids.to(queries.dtype)


def _route_compact_bank(
    compacts: tuple[AttentionMatchedKV, ...] | list[AttentionMatchedKV],
    centroids: torch.Tensor,
    query: torch.Tensor,
    route_top_k: int = 1,
) -> tuple[AttentionMatchedKV, list[int]]:
    """Route a query to one or more compact banks.

    `route_top_k == 1` preserves the original hard routing behavior. Larger
    values build a small ensemble by concatenating the nearest compact banks and
    shifting each bank's beta by `log(route_weight)`, so the final softmax can
    mix clusters without re-fitting beta/C_v at decode time.
    """
    compact_tuple = tuple(compacts)
    if not compact_tuple:
        raise ValueError("compact bank must contain at least one cluster")
    actual_k = max(1, min(int(route_top_k), len(compact_tuple)))
    q_repr = query.to(torch.float32).mean(dim=0)
    centroids = centroids.to(device=query.device, dtype=torch.float32)
    distances = torch.sum((centroids - q_repr.view(1, -1)) ** 2, dim=1)
    route_indices = torch.topk(-distances, k=actual_k, largest=True).indices
    route = [int(i.item()) for i in route_indices]
    if actual_k == 1:
        return compact_tuple[route[0]], route

    route_scores = -distances[route_indices]
    route_weights = torch.softmax(route_scores, dim=0).to(torch.float32)
    keys = []
    values = []
    beta = []
    indices = []
    for weight, route_idx in zip(route_weights, route):
        compact = compact_tuple[route_idx]
        keys.append(compact.keys)
        values.append(compact.values)
        beta_shift = torch.log(weight.clamp_min(1e-6)).to(device=compact.beta.device, dtype=torch.float32)
        beta.append((compact.beta.to(torch.float32) + beta_shift).to(compact.beta.dtype))
        indices.append(compact.indices)
    return AttentionMatchedKV(
        keys=torch.cat(keys, dim=0).contiguous(),
        beta=torch.cat(beta, dim=0).contiguous(),
        values=torch.cat(values, dim=0).contiguous(),
        indices=torch.cat(indices, dim=0).contiguous(),
    ), route


def _span_ranges(seq_len: int, num_key_spans: int) -> list[tuple[int, int]]:
    actual_spans = max(1, min(int(num_key_spans), int(seq_len)))
    ranges: list[tuple[int, int]] = []
    for span_idx in range(actual_spans):
        start = (seq_len * span_idx) // actual_spans
        end = (seq_len * (span_idx + 1)) // actual_spans
        if end > start:
            ranges.append((start, end))
    return ranges


def _build_span_local_compact_bank(
    keys: torch.Tensor,
    values: torch.Tensor,
    queries: torch.Tensor,
    budget: int,
    num_key_spans: int,
    selector: str,
    score_method: str,
    beta_bound: float,
    ridge_lambda: float,
    omp_candidate_pool_size: int,
) -> tuple[list[AttentionMatchedKV], torch.Tensor]:
    """Build one compact KV per contiguous key span.

    The goal is coverage rather than query clustering: every span contributes a
    small local compact basis, so middle evidence spans cannot be completely
    evicted by global Highest/OMP selection.
    """
    compacts: list[AttentionMatchedKV] = []
    centroids: list[torch.Tensor] = []
    for start, end in _span_ranges(keys.shape[0], num_key_spans):
        k_span = keys[start:end]
        v_span = values[start:end]
        span_budget = min(budget, k_span.shape[0])
        compact = attention_matching_compact_keys(
            k_span,
            v_span,
            queries,
            budget=span_budget,
            selector=selector,
            score_method=score_method,
            beta_bound=beta_bound,
            ridge_lambda=ridge_lambda,
            omp_candidate_pool_size=omp_candidate_pool_size,
        )
        compact.indices = (compact.indices + start).contiguous()
        compacts.append(compact)
        centroids.append(k_span.to(torch.float32).mean(dim=0).to(keys.dtype))
    return compacts, torch.stack(centroids, dim=0)


def _refit_compact_for_queries(
    keys: torch.Tensor,
    values: torch.Tensor,
    queries: torch.Tensor,
    compact: AttentionMatchedKV,
    beta_bound: float,
    ridge_lambda: float,
    mode: str = "full",
) -> AttentionMatchedKV:
    """Refit beta/C_v for cached selected indices using current decode queries."""
    if mode not in ("full", "direct", "beta", "anchor"):
        raise ValueError("decode refit mode must be 'full', 'direct', 'beta', or 'anchor'")
    selected = compact.indices.to(device=keys.device, dtype=torch.long)
    fit_keys = keys
    fit_values = values
    fit_selected = selected
    if mode == "anchor":
        anchor_budget = max(64, int(selected.numel()) * 4)
        anchors, fit_selected = _anchor_refit_indices(keys.shape[0], selected, anchor_budget)
        fit_keys = keys[anchors]
        fit_values = values[anchors]
    beta = fit_attention_bias(fit_keys, queries, fit_selected, beta_bound=beta_bound)
    if mode == "full":
        compact_values = fit_compacted_values(
            keys,
            values,
            queries,
            selected,
            beta,
            ridge_lambda=ridge_lambda,
        )
    elif mode == "anchor":
        compact_values = fit_compacted_values(
            fit_keys,
            fit_values,
            queries,
            fit_selected,
            beta,
            ridge_lambda=ridge_lambda,
        )
    elif mode == "direct":
        compact_values = values[selected]
    else:
        compact_values = compact.values.to(device=values.device, dtype=values.dtype)
    return AttentionMatchedKV(
        keys=keys[selected].contiguous(),
        beta=beta.contiguous(),
        values=compact_values.contiguous(),
        indices=selected.contiguous(),
    )


def _gather_batched_rows(rows: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    """Gather per-system rows from `[N, L, D]` with indices `[N, M]`."""
    return torch.gather(rows, 1, indices.unsqueeze(-1).expand(-1, -1, rows.shape[-1]))


def _attention_output_batched(
    queries: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    beta: torch.Tensor | None = None,
) -> torch.Tensor:
    """Batched compact attention output for same-shaped systems.

    Shapes: `queries=[N,R,D]`, `keys/values=[N,M,D]`, `beta=[N,M]`.
    """
    head_dim = keys.shape[-1]
    scores = torch.matmul(queries.to(torch.float32), keys.to(torch.float32).transpose(1, 2)) / (head_dim ** 0.5)
    if beta is not None:
        scores = scores + beta.to(torch.float32).unsqueeze(1)
    weights = torch.softmax(scores, dim=-1)
    return torch.matmul(weights, values.to(torch.float32))


def _refit_compacts_for_query_groups_batched(
    keys: torch.Tensor,
    values: torch.Tensor,
    queries: torch.Tensor,
    selected: torch.Tensor,
    compact_values: torch.Tensor,
    beta_bound: float,
    ridge_lambda: float,
    mode: str = "full",
) -> list[AttentionMatchedKV]:
    """Batched version of `_refit_compact_for_queries` for same-shaped systems.

    Shapes:
      keys/values: `[N, L, D]`, queries: `[N, R, D]`, selected: `[N, M]`.
    The helper intentionally targets the decode case `R < M`; other shapes keep
    the scalar fallback so public semantics stay conservative.
    """
    if mode not in ("full", "direct", "beta", "anchor"):
        raise ValueError("batched decode refit mode must be 'full', 'direct', 'beta', or 'anchor'")
    if keys.ndim != 3 or values.ndim != 3 or queries.ndim != 3 or selected.ndim != 2:
        raise ValueError("batched refit expects keys/values/queries [N, *, D] and selected [N, M]")
    num_systems, _, head_dim = keys.shape
    _, num_queries, _ = queries.shape
    compact_size = selected.shape[1]
    if num_queries >= compact_size:
        raise ValueError("batched refit currently expects num_queries < compact_size")
    if mode == "full" and ridge_lambda <= 0:
        raise ValueError("batched full refit currently expects positive ridge_lambda")

    scale = head_dim ** 0.5
    fit_keys = keys
    fit_values = values
    fit_selected = selected
    if mode == "anchor":
        anchor_budget = max(64, int(compact_size) * 4)
        anchor_budget = min(anchor_budget, keys.shape[1])
        uniform = torch.linspace(0, keys.shape[1] - 1, anchor_budget, device=keys.device).round().to(torch.long)
        uniform_batch = uniform.unsqueeze(0).expand(num_systems, -1)
        anchor_indices = torch.cat([uniform_batch, selected], dim=1)
        fit_keys = _gather_batched_rows(keys, anchor_indices).contiguous()
        fit_values = _gather_batched_rows(values, anchor_indices).contiguous()
        fit_selected = torch.arange(
            anchor_budget,
            anchor_budget + compact_size,
            device=selected.device,
            dtype=torch.long,
        ).unsqueeze(0).expand(num_systems, -1)

    scores = torch.matmul(queries.to(torch.float32), fit_keys.to(torch.float32).transpose(1, 2)) / scale
    exp_scores = torch.exp(scores - torch.max(scores, dim=2, keepdim=True).values)
    target_mass = exp_scores.sum(dim=2)
    gather_idx = fit_selected[:, None, :].expand(-1, num_queries, -1)
    design = torch.gather(exp_scores, 2, gather_idx)

    lam = 1e-6
    gram = torch.matmul(design, design.transpose(1, 2))
    gram = 0.5 * (gram + gram.transpose(1, 2))
    eye = torch.eye(num_queries, device=gram.device, dtype=gram.dtype).expand(num_systems, -1, -1)
    gram = gram + eye * lam
    weights = torch.matmul(
        design.transpose(1, 2),
        torch.linalg.solve(gram, target_mass.unsqueeze(-1)),
    ).squeeze(-1)
    weights = weights.clamp_min(float(torch.exp(torch.tensor(-beta_bound))))
    weights = weights.clamp_max(float(torch.exp(torch.tensor(beta_bound))))
    beta = torch.log(weights).to(keys.dtype)

    compact_keys = _gather_batched_rows(keys, selected).contiguous()
    if mode in ("full", "anchor"):
        target = torch.matmul(torch.softmax(scores, dim=-1), fit_values.to(torch.float32))
        compact_scores = (
            torch.matmul(queries.to(torch.float32), compact_keys.to(torch.float32).transpose(1, 2)) / scale
            + beta.to(torch.float32).unsqueeze(1)
        )
        value_design = torch.softmax(compact_scores, dim=-1)
        value_gram = torch.matmul(value_design, value_design.transpose(1, 2))
        value_gram = 0.5 * (value_gram + value_gram.transpose(1, 2))
        value_gram = value_gram + eye * ridge_lambda
        fitted_values = torch.matmul(
            value_design.transpose(1, 2),
            torch.linalg.solve(value_gram, target),
        ).to(values.dtype)
    elif mode == "direct":
        fitted_values = _gather_batched_rows(values, selected).contiguous()
    else:
        fitted_values = compact_values.to(device=values.device, dtype=values.dtype).contiguous()

    return [
        AttentionMatchedKV(
            keys=compact_keys[i].contiguous(),
            beta=beta[i].contiguous(),
            values=fitted_values[i].contiguous(),
            indices=selected[i].contiguous(),
        )
        for i in range(num_systems)
    ]


def _can_batch_decode_refit(
    q_group: torch.Tensor,
    compact: AttentionMatchedKV,
    ridge_lambda: float,
    mode: str,
) -> bool:
    compact_size = int(compact.indices.numel())
    if compact_size <= 0:
        return False
    if q_group.shape[0] >= compact_size:
        return False
    if mode == "full" and ridge_lambda <= 0:
        return False
    return True


def _process_pending_decode_refits(
    pending: list[_PendingDecodeRefit],
    out: torch.Tensor,
    cache: AttentionMatchingDecodeCache | None,
    cache_refit_enabled: bool,
    beta_bound: float,
    ridge_lambda: float,
    mode: str,
) -> None:
    if not pending:
        return
    buckets: dict[tuple, list[_PendingDecodeRefit]] = {}
    fallback: list[_PendingDecodeRefit] = []
    for item in pending:
        if not _can_batch_decode_refit(item.q_group, item.compact, ridge_lambda, mode):
            fallback.append(item)
            continue
        bucket_key = (
            item.k_seq.shape[0],
            item.q_group.shape[0],
            item.compact.indices.numel(),
            item.k_seq.shape[-1],
            str(item.k_seq.dtype),
            str(item.k_seq.device),
            mode,
        )
        buckets.setdefault(bucket_key, []).append(item)

    for items in buckets.values():
        k_batch = torch.stack([item.k_seq for item in items], dim=0)
        v_batch = torch.stack([item.v_seq for item in items], dim=0)
        q_batch = torch.stack([item.q_group for item in items], dim=0)
        selected = torch.stack([
            item.compact.indices.to(device=item.k_seq.device, dtype=torch.long)
            for item in items
        ], dim=0)
        compact_values = torch.stack([
            item.compact.values.to(device=item.v_seq.device, dtype=item.v_seq.dtype)
            for item in items
        ], dim=0)
        try:
            refit_compacts = _refit_compacts_for_query_groups_batched(
                k_batch,
                v_batch,
                q_batch,
                selected,
                compact_values,
                beta_bound=beta_bound,
                ridge_lambda=ridge_lambda,
                mode=mode,
            )
        except Exception:
            fallback.extend(items)
            continue
        refit_outputs = _attention_output_batched(
            q_batch,
            torch.stack([compact.keys for compact in refit_compacts], dim=0),
            torch.stack([compact.values for compact in refit_compacts], dim=0),
            torch.stack([compact.beta for compact in refit_compacts], dim=0),
        )
        for item, refit_compact in zip(items, refit_compacts):
            if cache is not None and cache_refit_enabled and item.refit_key is not None:
                cache.put_refit(item.refit_key, refit_compact)
        for item, group_out in zip(items, refit_outputs):
            out[item.b, item.q_start:item.q_end] = group_out.to(out.dtype)

    for item in fallback:
        refit_compact = _refit_compact_for_queries(
            item.k_seq,
            item.v_seq,
            item.q_group,
            item.compact,
            beta_bound=beta_bound,
            ridge_lambda=ridge_lambda,
            mode=mode,
        )
        if cache is not None and cache_refit_enabled and item.refit_key is not None:
            cache.put_refit(item.refit_key, refit_compact)
        group_out = attention_output(
            item.q_group,
            refit_compact.keys,
            refit_compact.values,
            refit_compact.beta,
        )
        out[item.b, item.q_start:item.q_end] = group_out.to(out.dtype)


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
    if design.shape[0] < design.shape[1]:
        # Decode refit has only a few GQA queries but many compact keys. Avoid
        # repeatedly calling SVD-based lstsq on an underdetermined tiny matrix.
        lam = 1e-6
        gram = design @ design.T
        gram = 0.5 * (gram + gram.T)
        gram.diagonal().add_(lam)
        solution = design.T @ torch.linalg.solve(gram, target)
    else:
        try:
            solution = torch.linalg.lstsq(design, target.unsqueeze(1)).solution.squeeze(1)
        except Exception:
            lam = 1e-6
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
        if design.shape[0] < design.shape[1]:
            # Ridge identity: (A^T A + λI)^-1 A^T y = A^T (A A^T + λI)^-1 y.
            # For decode refit this solves a small [num_q, num_q] system instead
            # of [compact_keys, compact_keys].
            gram = design @ design.T
            gram = 0.5 * (gram + gram.T)
            gram.diagonal().add_(ridge_lambda)
            solution = design.T @ torch.linalg.solve(gram, target)
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


def build_attention_matching_prefill_cache(
    cache: AttentionMatchingDecodeCache,
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
    cache_signatures: tuple | list | None = None,
    ref_query_stride: int = 1,
    num_clusters: int = 1,
    num_key_spans: int = 1,
) -> None:
    """Build persistent compact KV entries from prefill Q/K/V references."""
    if budget <= 0:
        raise ValueError("budget must be positive")
    if ref_query_stride <= 0:
        raise ValueError("ref_query_stride must be positive")
    if num_clusters <= 0:
        raise ValueError("num_clusters must be positive")
    if num_key_spans <= 0:
        raise ValueError("num_key_spans must be positive")
    batch, _, num_q_heads, head_dim = queries.shape
    _, _, num_kv_heads, _ = keys.shape
    assert num_q_heads % num_kv_heads == 0, "GQA requires q heads divisible by KV heads"
    group_size = num_q_heads // num_kv_heads
    for b in range(batch):
        seq_len = int(context_lens[b].item())
        if seq_len <= budget:
            continue
        cache_signature = (b, seq_len) if cache_signatures is None else cache_signatures[b]
        for kv_h in range(num_kv_heads):
            q_start = kv_h * group_size
            q_end = q_start + group_size
            q_refs = queries[b, :seq_len:ref_query_stride, q_start:q_end].reshape(-1, head_dim)
            k_seq = keys[b, :seq_len, kv_h]
            v_seq = values[b, :seq_len, kv_h]
            key = _compact_cache_key(
                cache_signature,
                kv_h,
                selector,
                budget,
                score_method,
                beta_bound,
                ridge_lambda,
                omp_candidate_pool_size,
                num_clusters,
                num_key_spans,
                k_seq.dtype,
                k_seq.device,
            )
            if num_key_spans > 1:
                compacts, centroids = _build_span_local_compact_bank(
                    k_seq,
                    v_seq,
                    q_refs,
                    budget=budget,
                    num_key_spans=num_key_spans,
                    selector=selector,
                    score_method=score_method,
                    beta_bound=beta_bound,
                    ridge_lambda=ridge_lambda,
                    omp_candidate_pool_size=omp_candidate_pool_size,
                )
                if len(compacts) == 1:
                    cache.put(key, compacts[0])
                else:
                    cache.put_bank(key, compacts, centroids)
            elif num_clusters == 1:
                compact = attention_matching_compact_keys(
                    k_seq,
                    v_seq,
                    q_refs,
                    budget=budget,
                    selector=selector,
                    score_method=score_method,
                    beta_bound=beta_bound,
                    ridge_lambda=ridge_lambda,
                    omp_candidate_pool_size=omp_candidate_pool_size,
                )
                cache.put(key, compact)
            else:
                labels, centroids = _cluster_reference_queries(q_refs, num_clusters)
                compacts: list[AttentionMatchedKV] = []
                kept_centroids: list[torch.Tensor] = []
                for cluster_idx in range(centroids.shape[0]):
                    q_cluster = q_refs[labels == cluster_idx]
                    if q_cluster.numel() == 0:
                        continue
                    compacts.append(attention_matching_compact_keys(
                        k_seq,
                        v_seq,
                        q_cluster,
                        budget=budget,
                        selector=selector,
                        score_method=score_method,
                        beta_bound=beta_bound,
                        ridge_lambda=ridge_lambda,
                        omp_candidate_pool_size=omp_candidate_pool_size,
                    ))
                    kept_centroids.append(centroids[cluster_idx])
                if len(compacts) == 1:
                    cache.put(key, compacts[0])
                else:
                    cache.put_bank(key, compacts, torch.stack(kept_centroids, dim=0))
            cache.prefill_builds += 1


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
    num_clusters: int = 1,
    cluster_route_top_k: int = 1,
    num_key_spans: int = 1,
    decode_refit: bool = False,
    decode_refit_mode: str = "full",
    decode_refit_interval: int = 1,
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
    if num_clusters <= 0:
        raise ValueError("num_clusters must be positive")
    if cluster_route_top_k <= 0:
        raise ValueError("cluster_route_top_k must be positive")
    if num_key_spans <= 0:
        raise ValueError("num_key_spans must be positive")
    if decode_refit_mode not in ("full", "direct", "beta", "anchor"):
        raise ValueError("decode_refit_mode must be 'full', 'direct', 'beta', or 'anchor'")
    if decode_refit_interval <= 0:
        raise ValueError("decode_refit_interval must be positive")
    batch, num_q_heads, head_dim = queries.shape
    _, _, num_kv_heads, _ = keys.shape
    assert num_q_heads % num_kv_heads == 0, "GQA requires q heads divisible by KV heads"
    group_size = num_q_heads // num_kv_heads
    out = torch.empty_like(queries)
    pending_refits: list[_PendingDecodeRefit] = []
    cache_refit_enabled = cache is not None and cache_refresh_interval > 0 and decode_refit_interval > 1
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
                cache_key = _compact_cache_key(
                    cache_signature,
                    kv_h,
                    selector,
                    budget,
                    score_method,
                    beta_bound,
                    ridge_lambda,
                    omp_candidate_pool_size,
                    num_clusters,
                    num_key_spans,
                    k_seq.dtype,
                    k_seq.device,
                )
                compact = None
                if cache is not None and cache_refresh_interval > 0:
                    compact = cache.get(cache_key, cache_refresh_interval, q_group, cluster_route_top_k)
                if compact is None:
                    compacts: list[AttentionMatchedKV] = []
                    kept_centroids: list[torch.Tensor] = []
                    if num_key_spans > 1:
                        compacts, centroids_for_route = _build_span_local_compact_bank(
                            k_seq,
                            v_seq,
                            q_group,
                            budget=budget,
                            num_key_spans=num_key_spans,
                            selector=selector,
                            score_method=score_method,
                            beta_bound=beta_bound,
                            ridge_lambda=ridge_lambda,
                            omp_candidate_pool_size=omp_candidate_pool_size,
                        )
                        kept_centroids = [centroids_for_route[i] for i in range(centroids_for_route.shape[0])]
                        compact, _ = _route_compact_bank(compacts, centroids_for_route, q_group, cluster_route_top_k)
                    elif num_clusters == 1:
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
                    else:
                        labels, centroids = _cluster_reference_queries(q_group, num_clusters)
                        for cluster_idx in range(centroids.shape[0]):
                            q_cluster = q_group[labels == cluster_idx]
                            if q_cluster.numel() == 0:
                                continue
                            compacts.append(attention_matching_compact_keys(
                                k_seq,
                                v_seq,
                                q_cluster,
                                budget=budget,
                                selector=selector,
                                score_method=score_method,
                                beta_bound=beta_bound,
                                ridge_lambda=ridge_lambda,
                                omp_candidate_pool_size=omp_candidate_pool_size,
                            ))
                            kept_centroids.append(centroids[cluster_idx])
                        centroids_for_route = torch.stack(kept_centroids, dim=0)
                        compact, _ = _route_compact_bank(compacts, centroids_for_route, q_group, cluster_route_top_k)
                    if cache is not None and cache_refresh_interval > 0:
                        if (num_clusters == 1 and num_key_spans == 1) or len(compacts) == 1:
                            cache.put(cache_key, compact)
                        else:
                            cache.put_bank(cache_key, compacts, torch.stack(kept_centroids, dim=0))
                if decode_refit:
                    refit_key = None
                    refit_compact = None
                    if cache_refit_enabled:
                        refit_key = _refit_cache_key(cache_key, compact, decode_refit_mode)
                        refit_compact = cache.get_refit(refit_key, decode_refit_interval)
                    if refit_compact is None:
                        pending_refits.append(
                            _PendingDecodeRefit(
                                b=b,
                                q_start=q_start,
                                q_end=q_end,
                                k_seq=k_seq,
                                v_seq=v_seq,
                                q_group=q_group,
                                compact=compact,
                                refit_key=refit_key,
                            )
                        )
                        continue
                    compact = refit_compact
                group_out = attention_output(q_group, compact.keys, compact.values, compact.beta)
            out[b, q_start:q_end] = group_out.to(out.dtype)
    _process_pending_decode_refits(
        pending_refits,
        out,
        cache,
        cache_refit_enabled,
        beta_bound=beta_bound,
        ridge_lambda=ridge_lambda,
        mode=decode_refit_mode,
    )
    return out
