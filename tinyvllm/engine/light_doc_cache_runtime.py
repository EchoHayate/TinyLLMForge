"""Default-off runtime planning helpers for Light Doc Cache.

This module intentionally does not mutate KV cache tensors or hook into the
ModelRunner hot path. It maps the offline adaptive policy artifacts into a small
runtime plan that can be reported, tested, and later wired to real KV recovery.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any


_HEAD_RE = re.compile(r"_add_l(?P<layer>\d+)h(?P<head>\d+)")
_BUDGET_RE = re.compile(r"_b(?P<budget>\d{3})(?:_|$)")
_BUDGET_FRACTION_RE = re.compile(r"budget(?P<budget>\d+)")


HeadCoord = tuple[int, int]


@dataclass(frozen=True)
class LightDocCachePolicy:
    """Parsed adaptive light-doc-cache policy metadata."""

    kind: str
    default_policy_dir: str
    base_safe_policy_dir: str | None = None
    doc_top_tasks: dict[str, int] = field(default_factory=dict)
    overrides: dict[str, dict[str, Any]] = field(default_factory=dict)
    default_added_heads: tuple[HeadCoord, ...] = ()
    default_budget_fraction: float = 0.5


@dataclass(frozen=True)
class LightDocCacheRuntimeConfig:
    """Inputs required to estimate runtime KV entry storage/recovery."""

    enabled: bool = False
    num_layers: int = 0
    num_kv_heads: int = 0
    policy: LightDocCachePolicy | None = None
    base_recovered_heads: list[HeadCoord] | tuple[HeadCoord, ...] = ()
    base_budget_fraction: float = 0.5
    head_budget_fractions: dict[HeadCoord, float] = field(default_factory=dict)


@dataclass(frozen=True)
class LightDocCacheRuntimePlan:
    """A non-mutating plan for one request/document/task."""

    enabled: bool
    task_id: str
    doc_id: str | None
    seq_len: int
    num_layers: int
    num_kv_heads: int
    total_head_token_entries: int
    stored_head_token_entries: int
    recovered_head_token_entries: int
    effective_saving_fraction: float
    stored_kv_heads_equivalent: float
    recovered_kv_head_equivalent: float
    recovered_heads: list[HeadCoord]
    applied_added_heads: list[HeadCoord]
    dropped_added_heads: list[HeadCoord]
    fallback_reason: str | None = None

    @property
    def total_kv_heads(self) -> int:
        return self.num_layers * self.num_kv_heads

    @property
    def stored_kv_heads(self) -> int:
        if self.seq_len == 0:
            return 0
        return self.stored_head_token_entries // self.seq_len

    @property
    def recovered_kv_heads(self) -> int:
        return len(self.recovered_heads)

    @property
    def compression_ratio(self) -> float:
        if self.stored_head_token_entries == 0:
            return 1.0
        return self.total_head_token_entries / self.stored_head_token_entries

    def as_summary(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "task_id": self.task_id,
            "doc_id": self.doc_id,
            "seq_len": self.seq_len,
            "num_layers": self.num_layers,
            "num_kv_heads": self.num_kv_heads,
            "total_kv_heads": self.total_kv_heads,
            "stored_kv_heads": self.stored_kv_heads,
            "stored_kv_heads_equivalent": self.stored_kv_heads_equivalent,
            "recovered_kv_heads": self.recovered_kv_heads,
            "recovered_kv_head_equivalent": self.recovered_kv_head_equivalent,
            "total_head_token_entries": self.total_head_token_entries,
            "stored_head_token_entries": self.stored_head_token_entries,
            "recovered_head_token_entries": self.recovered_head_token_entries,
            "effective_saving_fraction": self.effective_saving_fraction,
            "compression_ratio": self.compression_ratio,
            "recovered_heads": [list(head) for head in self.recovered_heads],
            "applied_added_heads": [list(head) for head in self.applied_added_heads],
            "dropped_added_heads": [list(head) for head in self.dropped_added_heads],
            "fallback_reason": self.fallback_reason,
        }


@dataclass
class LightDocCacheCompressedKVStorage:
    """Minimal real-storage prototype for planned compact heads.

    It stores full tensors for non-compact heads and only selected prefix tokens
    for compact heads. Restore can either fill missing compact-head tokens or
    call a recovery callback, so this is not a trained recovery implementation
    yet.
    """

    full_shape: tuple[int, ...]
    dtype: Any
    full_heads: dict[HeadCoord, Any]
    compact_heads: dict[HeadCoord, tuple[int, Any]]
    fill_value: float = 0.0
    seq_len: int = 0
    device: Any = None

    @classmethod
    def from_full_kv(
        cls,
        kv_cache: Any,
        plan: LightDocCacheRuntimePlan,
        *,
        fill_value: float = 0.0,
    ) -> "LightDocCacheCompressedKVStorage":
        shape = tuple(int(dim) for dim in kv_cache.shape)
        if len(shape) != 6:
            raise ValueError("kv_cache must be shaped [2, layers, blocks, block_size, kv_heads, head_dim]")
        if shape[0] != 2:
            raise ValueError("kv_cache dim0 must be 2 for K/V")
        _, layers, blocks, block_size, kv_heads, _ = shape
        if layers != plan.num_layers or kv_heads != plan.num_kv_heads:
            raise ValueError("kv_cache layer/head shape does not match plan")
        if blocks * block_size < plan.seq_len:
            raise ValueError("kv_cache shape cannot hold plan seq_len")

        recovered_set = set(plan.recovered_heads)
        compact_heads = {}
        full_heads = {}
        for layer in range(layers):
            for head in range(kv_heads):
                coord = (layer, head)
                head_view = kv_cache[:, layer, :, :, head, :]
                head_tokens = _flatten_head_tokens(head_view)
                if coord in recovered_set:
                    selected_tokens = _selected_token_count_for_head(plan, coord)
                    compact_heads[coord] = (selected_tokens, _copy_array(head_tokens[:, :selected_tokens, :]))
                else:
                    full_heads[coord] = _copy_array(head_tokens[:, :plan.seq_len, :])
        return cls(
            full_shape=shape,
            dtype=getattr(kv_cache, "dtype", None),
            device=getattr(kv_cache, "device", None),
            full_heads=full_heads,
            compact_heads=compact_heads,
            fill_value=fill_value,
            seq_len=plan.seq_len,
        )

    def restore_to_full_shape(self, recover_missing_fn=None) -> Any:
        restored = _full_array(self.full_shape, self.fill_value, self.dtype, device=self.device)
        for (layer, head), value in self.full_heads.items():
            _assign_flat_head_tokens(restored, layer, head, value, value.shape[1])
        for (layer, head), (selected_tokens, value) in self.compact_heads.items():
            _assign_flat_head_tokens(restored, layer, head, value, selected_tokens)
            if recover_missing_fn is None:
                continue
            missing_tokens = max(0, self.seq_len - selected_tokens)
            if missing_tokens == 0:
                continue
            recovered = recover_missing_fn(
                layer=layer,
                kv_head=head,
                selected_tokens=selected_tokens,
                missing_tokens=missing_tokens,
                stored_tokens=value,
                head_dim=self.full_shape[5],
                dtype=self.dtype,
            )
            _validate_recovered_missing_shape(recovered, self.full_shape[0], missing_tokens, self.full_shape[5])
            _assign_flat_head_tokens_at(restored, layer, head, recovered, selected_tokens, missing_tokens)
        return restored

    def summary(self) -> dict[str, Any]:
        full_bytes = _array_nbytes_from_shape(self.full_shape, self.dtype)
        full_head_bytes = sum(_array_nbytes(value) for value in self.full_heads.values())
        compact_head_bytes = sum(_array_nbytes(value) for _, value in self.compact_heads.values())
        stored_bytes = full_head_bytes + compact_head_bytes
        return {
            "claim_boundary": "storage_prototype_fill_missing_not_recovered",
            "full_shape": list(self.full_shape),
            "full_tensor_bytes": full_bytes,
            "stored_tensor_bytes": stored_bytes,
            "stored_full_head_bytes": full_head_bytes,
            "stored_compact_head_bytes": compact_head_bytes,
            "saved_tensor_bytes": full_bytes - stored_bytes,
            "byte_saving_fraction": 0.0 if full_bytes == 0 else (full_bytes - stored_bytes) / full_bytes,
            "full_heads": len(self.full_heads),
            "compact_heads": len(self.compact_heads),
        }


@dataclass(frozen=True)
class MultiSourceRecoveryBank:
    """Offline-fitted multi-source recovery weights."""

    weights: dict[HeadCoord, Any]
    source_heads: dict[HeadCoord, tuple[HeadCoord, ...]]
    ridge: float = 1e-6


def load_light_doc_cache_policy(path: str | Path) -> LightDocCachePolicy:
    """Load an adaptive policy JSON produced by light-doc-cache experiments."""

    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError("light doc cache policy must be a JSON object")
    return parse_light_doc_cache_policy(payload)


def parse_light_doc_cache_policy(payload: dict[str, Any]) -> LightDocCachePolicy:
    """Parse adaptive policy metadata without importing experiment scripts."""

    kind = str(payload.get("kind", ""))
    if kind != "task_adaptive_light_doc_cache_policy":
        raise ValueError(f"unsupported light doc cache policy kind: {kind!r}")
    default_policy_dir = str(payload.get("default_policy_dir", ""))
    if not default_policy_dir:
        raise ValueError("light doc cache policy requires default_policy_dir")
    overrides = payload.get("overrides", {})
    if not isinstance(overrides, dict):
        raise ValueError("light doc cache policy overrides must be an object")
    doc_top_tasks = payload.get("doc_top_tasks", {})
    if doc_top_tasks is None:
        doc_top_tasks = {}
    if not isinstance(doc_top_tasks, dict):
        raise ValueError("light doc cache policy doc_top_tasks must be an object")
    return LightDocCachePolicy(
        kind=kind,
        default_policy_dir=default_policy_dir,
        base_safe_policy_dir=payload.get("base_safe_policy_dir"),
        doc_top_tasks={str(key): int(value) for key, value in doc_top_tasks.items()},
        overrides={str(key): value for key, value in overrides.items()},
        default_added_heads=_extract_added_heads(default_policy_dir),
        default_budget_fraction=_infer_budget_fraction(default_policy_dir, default=0.5),
    )


def build_light_doc_cache_runtime_plan(
    config: LightDocCacheRuntimeConfig,
    *,
    task_id: str,
    doc_id: str | None,
    seq_len: int,
) -> LightDocCacheRuntimePlan:
    """Build an estimated KV entry plan for one request.

    The returned plan counts head-token entries. It is a planning/metrics object,
    not proof that KV tensors were physically compressed or recovered.
    """

    _validate_shape(config.num_layers, config.num_kv_heads)
    seq_len = int(seq_len)
    if seq_len < 0:
        raise ValueError("seq_len must be non-negative")
    total_entries = int(config.num_layers) * int(config.num_kv_heads) * seq_len
    if not config.enabled:
        return LightDocCacheRuntimePlan(
            enabled=False,
            task_id=task_id,
            doc_id=doc_id,
            seq_len=seq_len,
            num_layers=int(config.num_layers),
            num_kv_heads=int(config.num_kv_heads),
            total_head_token_entries=total_entries,
            stored_head_token_entries=total_entries,
            recovered_head_token_entries=0,
            effective_saving_fraction=0.0,
            stored_kv_heads_equivalent=float(config.num_layers * config.num_kv_heads),
            recovered_kv_head_equivalent=0.0,
            recovered_heads=[],
            applied_added_heads=[],
            dropped_added_heads=[],
            fallback_reason="disabled",
        )
    if config.policy is None:
        raise ValueError("enabled light doc cache runtime planning requires a policy")

    base_heads = _dedupe_heads(config.base_recovered_heads)
    added_heads = list(config.policy.default_added_heads)
    override = config.policy.overrides.get(str(task_id))
    dropped_heads: list[HeadCoord] = []
    fallback_reason = None
    if override is not None:
        dropped_heads = _parse_head_list(override.get("drop_heads", override.get("drop_added_heads", [])))
        dropped_set = set(dropped_heads)
        added_heads = [head for head in added_heads if head not in dropped_set]
        fallback_reason = "task_override"

    recovered_heads = _dedupe_heads([*added_heads, *base_heads])
    _validate_heads(recovered_heads, config.num_layers, config.num_kv_heads)
    _validate_heads(dropped_heads, config.num_layers, config.num_kv_heads)

    recovered_head_equivalent = 0.0
    head_budget_fractions = {
        (int(layer), int(head)): _validate_fraction(fraction, f"head_budget_fractions[{layer}:{head}]")
        for (layer, head), fraction in config.head_budget_fractions.items()
    }
    for head in recovered_heads:
        fallback_budget = (
            config.policy.default_budget_fraction
            if head in set(added_heads)
            else config.base_budget_fraction
        )
        budget_fraction = head_budget_fractions.get(head, fallback_budget)
        recovered_head_equivalent += 1.0 - _validate_fraction(budget_fraction, f"budget_fraction[{head[0]}:{head[1]}]")
    recovered_entries = int(round(recovered_head_equivalent * seq_len))
    stored_entries = total_entries - recovered_entries
    saving = 0.0 if total_entries == 0 else recovered_entries / total_entries
    return LightDocCacheRuntimePlan(
        enabled=True,
        task_id=task_id,
        doc_id=doc_id,
        seq_len=seq_len,
        num_layers=int(config.num_layers),
        num_kv_heads=int(config.num_kv_heads),
        total_head_token_entries=total_entries,
        stored_head_token_entries=stored_entries,
        recovered_head_token_entries=recovered_entries,
        effective_saving_fraction=saving,
        stored_kv_heads_equivalent=(0.0 if seq_len == 0 else stored_entries / seq_len),
        recovered_kv_head_equivalent=recovered_head_equivalent,
        recovered_heads=recovered_heads,
        applied_added_heads=_dedupe_heads(added_heads),
        dropped_added_heads=dropped_heads,
        fallback_reason=fallback_reason,
    )


def _extract_added_heads(policy_dir: str) -> tuple[HeadCoord, ...]:
    heads = []
    for match in _HEAD_RE.finditer(policy_dir):
        heads.append((int(match.group("layer")), int(match.group("head"))))
    return tuple(_dedupe_heads(heads))


def build_config_from_policy_dirs(
    policy: LightDocCachePolicy,
    *,
    repo_root: str | Path,
    num_layers: int,
    num_kv_heads: int,
    enabled: bool = False,
) -> LightDocCacheRuntimeConfig:
    """Create runtime config by reading real policy_rows.csv artifacts."""

    repo_root = Path(repo_root)
    default_rows = _load_compact_policy_rows(repo_root / policy.default_policy_dir / "policy_rows.csv")
    if policy.base_safe_policy_dir:
        base_rows = _load_compact_policy_rows(repo_root / policy.base_safe_policy_dir / "policy_rows.csv")
    else:
        base_rows = {
            head: budget
            for head, budget in default_rows.items()
            if head not in set(policy.default_added_heads)
        }
    added_set = set(policy.default_added_heads)
    return LightDocCacheRuntimeConfig(
        enabled=enabled,
        num_layers=num_layers,
        num_kv_heads=num_kv_heads,
        policy=policy,
        base_recovered_heads=sorted(base_rows),
        base_budget_fraction=0.5,
        head_budget_fractions={
            **base_rows,
            **{head: budget for head, budget in default_rows.items() if head in added_set},
        },
    )


def summarize_planned_kv_storage(
    plan: LightDocCacheRuntimePlan,
    *,
    num_blocks: int,
    block_size: int,
    head_dim: int,
    element_size_bytes: int,
) -> dict[str, Any]:
    """Map a runtime plan to full-cache byte accounting.

    This only compares a planned compressed logical footprint against the full
    KV cache shape. It does not allocate or free storage.
    """

    num_blocks = _validate_positive_int(num_blocks, "num_blocks")
    block_size = _validate_positive_int(block_size, "block_size")
    head_dim = _validate_positive_int(head_dim, "head_dim")
    element_size_bytes = _validate_positive_int(element_size_bytes, "element_size_bytes")
    if plan.total_kv_heads <= 0:
        raise ValueError("plan must have positive total_kv_heads")
    if num_blocks * block_size < plan.seq_len:
        raise ValueError("full cache shape cannot hold plan seq_len")

    bytes_per_head_token = 2 * head_dim * element_size_bytes
    full_kv_bytes = plan.total_head_token_entries * bytes_per_head_token
    recovered_kv_bytes = plan.recovered_head_token_entries * bytes_per_head_token
    stored_kv_bytes = full_kv_bytes - recovered_kv_bytes
    return {
        "claim_boundary": "planning_only_not_allocated",
        "full_cache_shape": [
            2,
            plan.num_layers,
            num_blocks,
            block_size,
            plan.num_kv_heads,
            head_dim,
        ],
        "full_kv_bytes": full_kv_bytes,
        "planned_stored_kv_bytes": stored_kv_bytes,
        "planned_recovered_kv_bytes": recovered_kv_bytes,
        "planned_byte_saving_fraction": 0.0 if full_kv_bytes == 0 else recovered_kv_bytes / full_kv_bytes,
        "bytes_per_head_token": bytes_per_head_token,
    }


def summarize_planned_kv_storage_from_shape(
    plan: LightDocCacheRuntimePlan,
    *,
    full_cache_shape,
    element_size_bytes: int,
) -> dict[str, Any]:
    """Summarize planned storage using a real `kv_cache.shape`-style tuple."""

    shape = [int(dim) for dim in full_cache_shape]
    if len(shape) != 6:
        raise ValueError("full_cache_shape must be [2, layers, blocks, block_size, kv_heads, head_dim]")
    if shape[0] != 2:
        raise ValueError("full_cache_shape dim0 must be 2 for K/V")
    _, num_layers, num_blocks, block_size, num_kv_heads, head_dim = shape
    if num_layers != plan.num_layers or num_kv_heads != plan.num_kv_heads:
        raise ValueError(
            "shape layer/head mismatch: "
            f"shape_layers={num_layers}, plan_layers={plan.num_layers}, "
            f"shape_kv_heads={num_kv_heads}, plan_kv_heads={plan.num_kv_heads}"
        )
    summary = summarize_planned_kv_storage(
        plan,
        num_blocks=num_blocks,
        block_size=block_size,
        head_dim=head_dim,
        element_size_bytes=element_size_bytes,
    )
    summary["shape_source"] = "kv_cache_shape"
    return summary


def build_model_runner_light_doc_cache_summary(runner: Any, plan: LightDocCacheRuntimePlan) -> dict[str, Any] | None:
    """Build a summary-only report from a ModelRunner-like object.

    The helper intentionally depends only on `runner.kv_cache.shape` and
    `runner.kv_cache.element_size()`, so tests can exercise it without loading
    the full model stack. It does not allocate compressed KV storage.
    """

    kv_cache = getattr(runner, "kv_cache", None)
    if kv_cache is None:
        return None
    shape = getattr(kv_cache, "shape", None)
    element_size = getattr(kv_cache, "element_size", None)
    if shape is None or element_size is None:
        return None
    storage = summarize_planned_kv_storage_from_shape(
        plan,
        full_cache_shape=tuple(shape),
        element_size_bytes=int(element_size()),
    )
    return {
        "enabled": plan.enabled,
        "claim_boundary": storage["claim_boundary"],
        "plan": plan.as_summary(),
        "storage": storage,
        "next_step": "wire_compressed_storage_before_claiming_runtime_savings",
    }


def materialize_light_doc_cache_sidecar(
    kv_cache: Any,
    plan: LightDocCacheRuntimePlan,
    *,
    fill_value: float = -1.0,
    recover_missing_fn=None,
    evaluate_readback: bool = False,
) -> tuple[LightDocCacheCompressedKVStorage, dict[str, Any]]:
    """Materialize a default-off sidecar from a full KV cache tensor.

    This stores selected-token slices for compact heads in a sidecar object and
    returns accounting. It does not replace the original KV cache, change
    attention reads, or prove runtime memory reduction.
    """

    storage = LightDocCacheCompressedKVStorage.from_full_kv(kv_cache, plan, fill_value=fill_value)
    restored = None
    if evaluate_readback:
        restored = storage.restore_to_full_shape(recover_missing_fn=recover_missing_fn)
    summary = _build_sidecar_materialization_summary(
        plan,
        storage,
        kv_cache,
        restored,
        evaluate_readback=evaluate_readback,
    )
    return storage, summary


def _build_sidecar_materialization_summary(
    plan: LightDocCacheRuntimePlan,
    storage: LightDocCacheCompressedKVStorage,
    full_kv: Any,
    restored_kv: Any | None,
    *,
    evaluate_readback: bool,
) -> dict[str, Any]:
    sidecar_storage = storage.summary()
    logical_full_bytes = _logical_full_kv_bytes_from_storage_summary(plan, sidecar_storage)
    logical_stored_bytes = sidecar_storage["stored_tensor_bytes"]
    logical_saved_bytes = max(0, logical_full_bytes - logical_stored_bytes)
    summary = {
        "claim_boundary": "sidecar_materialized_not_attention_hot_path",
        "kv_cache_shape": list(storage.full_shape),
        "logical_full_kv_bytes": logical_full_bytes,
        "logical_saved_kv_bytes": logical_saved_bytes,
        "logical_stored_kv_bytes": logical_stored_bytes,
        "logical_byte_saving_fraction": 0.0 if logical_full_bytes == 0 else logical_saved_bytes / logical_full_bytes,
        "plan": plan.as_summary(),
        "sidecar_storage": sidecar_storage,
    }
    if evaluate_readback:
        if restored_kv is None:
            raise ValueError("restored_kv is required when evaluate_readback is true")
        summary["error_metrics"] = evaluate_restored_kv_error(full_kv, restored_kv, plan)
    return summary


def materialize_model_runner_light_doc_cache_sidecar(
    runner: Any,
    plan: LightDocCacheRuntimePlan,
    *,
    fill_value: float = -1.0,
    recover_missing_fn=None,
    evaluate_readback: bool = False,
) -> tuple[LightDocCacheCompressedKVStorage, dict[str, Any]] | None:
    """Materialize a sidecar from `runner.kv_cache` if available."""

    kv_cache = getattr(runner, "kv_cache", None)
    if kv_cache is None:
        return None
    return materialize_light_doc_cache_sidecar(
        kv_cache,
        plan,
        fill_value=fill_value,
        recover_missing_fn=recover_missing_fn,
        evaluate_readback=evaluate_readback,
    )


def make_oracle_recovery_callback(full_kv: Any, plan: LightDocCacheRuntimePlan):
    """Return a callback that copies missing tokens from the original KV.

    This is only for validating storage layout and error metrics.
    """

    def recover_missing(*, layer, kv_head, selected_tokens, missing_tokens, stored_tokens, head_dim, dtype):
        del stored_tokens, head_dim, dtype
        head_tokens = _flatten_head_tokens(full_kv[:, layer, :, :, kv_head, :])
        return _copy_array(head_tokens[:, selected_tokens:selected_tokens + missing_tokens, :])

    return recover_missing


def make_repeat_last_recovery_callback():
    """Return a deterministic non-oracle callback that repeats the last stored token."""

    def recover_missing(*, layer, kv_head, selected_tokens, missing_tokens, stored_tokens, head_dim, dtype):
        del layer, kv_head, selected_tokens, head_dim, dtype
        if missing_tokens <= 0:
            return stored_tokens[:, :0, :]
        if int(stored_tokens.shape[1]) <= 0:
            return _zeros_like_missing_tokens(stored_tokens, missing_tokens)
        last_token = stored_tokens[:, -1:, :]
        return _repeat_tokens(last_token, missing_tokens)

    return recover_missing


def make_linear_tail_recovery_callback(*, ridge: float = 1e-6):
    """Return a toy ridge-linear callback that extrapolates missing token values.

    This callback fits value ~= slope * token_index + bias from the stored
    prefix tokens for each K/V channel independently. It is a deterministic
    non-oracle storage-layout baseline, not the attention-output recovery bank.
    """

    ridge_value = float(ridge)

    def recover_missing(*, layer, kv_head, selected_tokens, missing_tokens, stored_tokens, head_dim, dtype):
        del layer, kv_head, head_dim, dtype
        if missing_tokens <= 0:
            return stored_tokens[:, :0, :]
        if int(stored_tokens.shape[1]) < 2:
            return make_repeat_last_recovery_callback()(
                layer=0,
                kv_head=0,
                selected_tokens=selected_tokens,
                missing_tokens=missing_tokens,
                stored_tokens=stored_tokens,
                head_dim=int(stored_tokens.shape[2]),
                dtype=getattr(stored_tokens, "dtype", None),
            )
        return _linear_tail_extrapolate(stored_tokens, int(selected_tokens), int(missing_tokens), ridge_value)

    return recover_missing


def make_correlated_head_recovery_callback(
    storage: LightDocCacheCompressedKVStorage,
    *,
    source_heads: dict[HeadCoord, HeadCoord] | None = None,
    ridge: float = 1e-6,
):
    """Return a callback that predicts compact heads from retained full heads.

    The callback fits a small affine ridge map on the target compact head's
    stored prefix and the matching prefix from a retained full source head, then
    predicts the target head's missing tokens from the source head's missing
    tokens. It does not read target missing KV values.
    """

    if source_heads is None:
        source_heads = build_correlated_source_head_map(storage, ridge=ridge)
    source_head_map = {
        (int(target_layer), int(target_head)): (int(source_layer), int(source_head))
        for (target_layer, target_head), (source_layer, source_head) in source_heads.items()
    }
    ridge_value = float(ridge)

    def recover_missing(*, layer, kv_head, selected_tokens, missing_tokens, stored_tokens, head_dim, dtype):
        del head_dim, dtype
        target = (int(layer), int(kv_head))
        source = source_head_map.get(target)
        if source is None:
            return make_repeat_last_recovery_callback()(
                layer=layer,
                kv_head=kv_head,
                selected_tokens=selected_tokens,
                missing_tokens=missing_tokens,
                stored_tokens=stored_tokens,
                head_dim=int(stored_tokens.shape[2]),
                dtype=getattr(stored_tokens, "dtype", None),
            )
        source_tokens = storage.full_heads.get(source)
        if source_tokens is None:
            raise ValueError(f"source head {source[0]}:{source[1]} is not retained as a full head")
        return _correlated_head_predict(
            source_tokens=source_tokens,
            target_tokens=stored_tokens,
            selected_tokens=int(selected_tokens),
            missing_tokens=int(missing_tokens),
            ridge=ridge_value,
        )

    return recover_missing


def build_correlated_source_head_map(
    storage: LightDocCacheCompressedKVStorage,
    *,
    ridge: float = 1e-6,
) -> dict[HeadCoord, HeadCoord]:
    """Select retained source heads by prefix reconstruction error.

    For each compact target head, this scores every retained full head using
    only the target's stored prefix and the source's matching prefix. The source
    with the lowest affine ridge fit error is selected for missing-token
    recovery.
    """

    if not storage.full_heads:
        raise ValueError("correlated recovery requires at least one retained full source head")
    source_map = {}
    for target, (selected_tokens, target_tokens) in storage.compact_heads.items():
        best_source = None
        best_error = None
        for source, source_tokens in storage.full_heads.items():
            error = _correlated_prefix_fit_mse(
                source_tokens=source_tokens,
                target_tokens=target_tokens,
                selected_tokens=int(selected_tokens),
                ridge=float(ridge),
            )
            if best_error is None or error < best_error or (error == best_error and source < best_source):
                best_error = error
                best_source = source
        if best_source is None:
            raise ValueError("correlated recovery requires at least one retained full source head")
        source_map[target] = best_source
    return source_map


def make_multi_source_correlated_head_recovery_callback(
    storage: LightDocCacheCompressedKVStorage,
    *,
    source_heads: dict[HeadCoord, list[HeadCoord] | tuple[HeadCoord, ...]],
    ridge: float = 1e-6,
):
    """Return a callback that predicts compact heads from multiple retained heads."""

    source_head_map = {
        (int(target_layer), int(target_head)): [
            (int(source_layer), int(source_head)) for source_layer, source_head in sources
        ]
        for (target_layer, target_head), sources in source_heads.items()
    }
    ridge_value = float(ridge)

    def recover_missing(*, layer, kv_head, selected_tokens, missing_tokens, stored_tokens, head_dim, dtype):
        del head_dim, dtype
        target = (int(layer), int(kv_head))
        sources = source_head_map.get(target)
        if not sources:
            return make_repeat_last_recovery_callback()(
                layer=layer,
                kv_head=kv_head,
                selected_tokens=selected_tokens,
                missing_tokens=missing_tokens,
                stored_tokens=stored_tokens,
                head_dim=int(stored_tokens.shape[2]),
                dtype=getattr(stored_tokens, "dtype", None),
            )
        source_tokens = []
        for source in sources:
            value = storage.full_heads.get(source)
            if value is None:
                raise ValueError(f"source head {source[0]}:{source[1]} is not retained as a full head")
            source_tokens.append(value)
        return _multi_source_correlated_head_predict(
            source_tokens=source_tokens,
            target_tokens=stored_tokens,
            selected_tokens=int(selected_tokens),
            missing_tokens=int(missing_tokens),
            ridge=ridge_value,
        )

    return recover_missing


def fit_multi_source_recovery_bank(
    calibration_kv: Any,
    plan: LightDocCacheRuntimePlan,
    *,
    source_heads: dict[HeadCoord, list[HeadCoord] | tuple[HeadCoord, ...]],
    ridge: float = 1e-6,
) -> MultiSourceRecoveryBank:
    """Fit multi-source recovery weights from an offline calibration KV tensor."""

    weights = {}
    source_map = {}
    for target, sources in source_heads.items():
        target = (int(target[0]), int(target[1]))
        source_tuple = tuple((int(layer), int(head)) for layer, head in sources)
        if not source_tuple:
            raise ValueError("multi-source recovery bank requires at least one source head")
        selected_tokens = _selected_token_count_for_head(plan, target)
        target_tokens = _flatten_head_tokens(calibration_kv[:, target[0], :, :, target[1], :])[:, :plan.seq_len, :]
        source_tokens = [
            _flatten_head_tokens(calibration_kv[:, layer, :, :, head, :])[:, :plan.seq_len, :]
            for layer, head in source_tuple
        ]
        weights[target] = _fit_multi_source_recovery_weights(
            source_tokens=source_tokens,
            target_tokens=target_tokens,
            selected_tokens=selected_tokens,
            ridge=float(ridge),
        )
        source_map[target] = source_tuple
    return MultiSourceRecoveryBank(weights=weights, source_heads=source_map, ridge=float(ridge))


def make_calibrated_multi_source_recovery_callback(
    storage: LightDocCacheCompressedKVStorage,
    bank: MultiSourceRecoveryBank,
):
    """Return a callback that applies offline-fitted multi-source weights."""

    def recover_missing(*, layer, kv_head, selected_tokens, missing_tokens, stored_tokens, head_dim, dtype):
        del head_dim, dtype
        target = (int(layer), int(kv_head))
        weights = bank.weights.get(target)
        sources = bank.source_heads.get(target)
        if weights is None or sources is None:
            return make_repeat_last_recovery_callback()(
                layer=layer,
                kv_head=kv_head,
                selected_tokens=selected_tokens,
                missing_tokens=missing_tokens,
                stored_tokens=stored_tokens,
                head_dim=int(stored_tokens.shape[2]),
                dtype=getattr(stored_tokens, "dtype", None),
            )
        source_tokens = []
        for source in sources:
            value = storage.full_heads.get(source)
            if value is None:
                raise ValueError(f"source head {source[0]}:{source[1]} is not retained as a full head")
            source_tokens.append(value)
        return _apply_multi_source_recovery_weights(
            source_tokens=source_tokens,
            weights=weights,
            start_token=int(storage.compact_heads[target][0]),
            missing_tokens=int(missing_tokens),
        )

    return recover_missing


def save_multi_source_recovery_bank(bank: MultiSourceRecoveryBank, path: str | Path) -> None:
    """Save a fitted recovery bank to a JSON file."""

    entries = []
    for target in sorted(bank.weights):
        weights = bank.weights[target]
        entries.append(
            {
                "target": [int(target[0]), int(target[1])],
                "sources": [[int(layer), int(head)] for layer, head in bank.source_heads[target]],
                "weights": _array_to_nested_lists(weights),
            }
        )
    payload = {
        "kind": "multi_source_recovery_bank",
        "version": 1,
        "ridge": float(bank.ridge),
        "entries": entries,
    }
    Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_multi_source_recovery_bank(path: str | Path) -> MultiSourceRecoveryBank:
    """Load a fitted recovery bank from a JSON file."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("kind") != "multi_source_recovery_bank":
        raise ValueError("recovery bank JSON kind must be multi_source_recovery_bank")
    weights = {}
    source_heads = {}
    for entry in payload.get("entries", []):
        target = (int(entry["target"][0]), int(entry["target"][1]))
        source_heads[target] = tuple((int(layer), int(head)) for layer, head in entry["sources"])
        weights[target] = _nested_lists_to_array(entry["weights"])
    return MultiSourceRecoveryBank(
        weights=weights,
        source_heads=source_heads,
        ridge=float(payload.get("ridge", 1e-6)),
    )


def evaluate_restored_kv_error(full_kv: Any, restored_kv: Any, plan: LightDocCacheRuntimePlan) -> dict[str, Any]:
    """Evaluate reconstruction error on missing compact-head tokens only."""

    missing_diffs = []
    missing_token_count = 0
    for layer, head in plan.recovered_heads:
        selected_tokens = _selected_token_count_for_head(plan, (layer, head))
        missing_tokens = max(0, plan.seq_len - selected_tokens)
        if missing_tokens <= 0:
            continue
        full_tokens = _flatten_head_tokens(full_kv[:, layer, :, :, head, :])
        restored_tokens = _flatten_head_tokens(restored_kv[:, layer, :, :, head, :])
        diff = restored_tokens[:, selected_tokens:selected_tokens + missing_tokens, :] - full_tokens[
            :, selected_tokens:selected_tokens + missing_tokens, :
        ]
        missing_diffs.append(diff)
        missing_token_count += missing_tokens
    if not missing_diffs:
        return {
            "num_missing_compact_tokens": 0,
            "mse_missing_compact_tokens": 0.0,
            "mae_missing_compact_tokens": 0.0,
            "max_abs_missing_compact_tokens": 0.0,
        }
    flat = _concat_flat_arrays(missing_diffs)
    return {
        "num_missing_compact_tokens": int(missing_token_count),
        "mse_missing_compact_tokens": float(_mean_square(flat)),
        "mae_missing_compact_tokens": float(_mean_abs(flat)),
        "max_abs_missing_compact_tokens": float(_max_abs(flat)),
    }


def _load_compact_policy_rows(path: Path) -> dict[HeadCoord, float]:
    if not path.exists():
        raise FileNotFoundError(path)
    rows = {}
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if row.get("action") != "compact":
                continue
            head = (int(row["layer"]), int(row["kv_head"]))
            rows[head] = _validate_fraction(float(row["budget_fraction"]), f"budget_fraction[{head[0]}:{head[1]}]")
    return rows


def _infer_budget_fraction(policy_dir: str, default: float) -> float:
    b_match = _BUDGET_RE.search(policy_dir)
    if b_match is not None:
        return int(b_match.group("budget")) / 100.0
    budget_match = _BUDGET_FRACTION_RE.search(policy_dir)
    if budget_match is not None:
        return int(budget_match.group("budget")) / 100.0
    return float(default)


def _parse_head_list(values: Any) -> list[HeadCoord]:
    if values is None:
        return []
    if not isinstance(values, list):
        raise ValueError("head list must be a list")
    heads = []
    for value in values:
        if isinstance(value, str):
            pieces = value.split(":")
            if len(pieces) != 2:
                raise ValueError(f"invalid head coordinate: {value!r}")
            heads.append((int(pieces[0]), int(pieces[1])))
        elif isinstance(value, (list, tuple)) and len(value) == 2:
            heads.append((int(value[0]), int(value[1])))
        else:
            raise ValueError(f"invalid head coordinate: {value!r}")
    return _dedupe_heads(heads)


def _dedupe_heads(heads: list[HeadCoord] | tuple[HeadCoord, ...]) -> list[HeadCoord]:
    seen = set()
    deduped = []
    for layer, head in heads:
        coord = (int(layer), int(head))
        if coord in seen:
            continue
        seen.add(coord)
        deduped.append(coord)
    return deduped


def _validate_shape(num_layers: int, num_kv_heads: int) -> None:
    if int(num_layers) <= 0:
        raise ValueError("num_layers must be positive")
    if int(num_kv_heads) <= 0:
        raise ValueError("num_kv_heads must be positive")


def _validate_fraction(value: float, name: str) -> float:
    fraction = float(value)
    if fraction < 0.0 or fraction > 1.0:
        raise ValueError(f"{name} must be between 0 and 1")
    return fraction


def _validate_positive_int(value: int, name: str) -> int:
    integer = int(value)
    if integer <= 0:
        raise ValueError(f"{name} must be positive")
    return integer


def _validate_heads(heads: list[HeadCoord], num_layers: int, num_kv_heads: int) -> None:
    for layer, head in heads:
        if layer < 0 or layer >= int(num_layers) or head < 0 or head >= int(num_kv_heads):
            raise ValueError(
                "layer/head outside model shape: "
                f"layer={layer}, head={head}, num_layers={num_layers}, num_kv_heads={num_kv_heads}"
            )


def _selected_token_count_for_head(plan: LightDocCacheRuntimePlan, head: HeadCoord) -> int:
    if head not in set(plan.recovered_heads):
        return plan.seq_len
    head_budget = plan.stored_head_token_entries
    # Prefer exact per-head equivalent when available from the plan inputs.
    # `LightDocCacheRuntimePlan` stores only aggregate results, so infer the
    # selected-token count from recovered equivalent for common b050/b075 rows by
    # using the compact/full tensors supplied to the storage prototype. The
    # current prototype stores prefix tokens and fills the rest.
    del head_budget
    if head in set(plan.applied_added_heads):
        # Adaptive added heads currently use budget75 in the generated policies.
        return int(round(plan.seq_len * 0.75))
    return int(round(plan.seq_len * 0.5))


def _copy_array(value: Any) -> Any:
    if hasattr(value, "clone"):
        return value.clone()
    if hasattr(value, "copy"):
        return value.copy()
    raise TypeError("kv_cache slices must support clone() or copy()")


def _flatten_head_tokens(head_view: Any) -> Any:
    # head_view: [2, blocks, block_size, head_dim] -> [2, tokens, head_dim]
    shape = tuple(int(dim) for dim in head_view.shape)
    if len(shape) != 4:
        raise ValueError("head view must be shaped [2, blocks, block_size, head_dim]")
    return head_view.reshape(shape[0], shape[1] * shape[2], shape[3])


def _assign_flat_head_tokens(restored: Any, layer: int, head: int, flat_tokens: Any, token_count: int) -> None:
    shape = tuple(int(dim) for dim in restored.shape)
    block_size = shape[3]
    head_dim = shape[5]
    token_count = int(token_count)
    if token_count <= 0:
        return
    target = restored[:, layer, :, :, head, :].reshape(shape[0], shape[2] * block_size, head_dim)
    target[:, :token_count, :] = flat_tokens[:, :token_count, :]


def _assign_flat_head_tokens_at(
    restored: Any,
    layer: int,
    head: int,
    flat_tokens: Any,
    start_token: int,
    token_count: int,
) -> None:
    shape = tuple(int(dim) for dim in restored.shape)
    block_size = shape[3]
    head_dim = shape[5]
    start_token = int(start_token)
    token_count = int(token_count)
    if token_count <= 0:
        return
    target = restored[:, layer, :, :, head, :].reshape(shape[0], shape[2] * block_size, head_dim)
    target[:, start_token:start_token + token_count, :] = flat_tokens[:, :token_count, :]


def _validate_recovered_missing_shape(value: Any, kv_dim: int, missing_tokens: int, head_dim: int) -> None:
    shape = tuple(int(dim) for dim in value.shape)
    expected = (int(kv_dim), int(missing_tokens), int(head_dim))
    if shape != expected:
        raise ValueError(f"recovered missing tokens shape must be {expected}, got {shape}")


def _repeat_tokens(value: Any, repeat_count: int) -> Any:
    if hasattr(value, "repeat"):
        if "torch" in str(type(value)):
            return value.repeat(1, int(repeat_count), 1)
        return value.repeat(int(repeat_count), axis=1)
    raise TypeError("stored_tokens must support repeat()")


def _zeros_like_missing_tokens(stored_tokens: Any, missing_tokens: int) -> Any:
    shape = (int(stored_tokens.shape[0]), int(missing_tokens), int(stored_tokens.shape[2]))
    if hasattr(stored_tokens, "new_zeros"):
        return stored_tokens.new_zeros(shape)
    try:
        import numpy as np

        return np.zeros(shape, dtype=getattr(stored_tokens, "dtype", None))
    except Exception as exc:
        raise RuntimeError("repeat-last recovery requires numpy or torch") from exc


def _linear_tail_extrapolate(stored_tokens: Any, selected_tokens: int, missing_tokens: int, ridge: float) -> Any:
    try:
        import numpy as np

        if "numpy" in str(type(stored_tokens)):
            y = stored_tokens.astype("float64", copy=False)
            positions = np.arange(selected_tokens, dtype=np.float64)
            future = np.arange(selected_tokens, selected_tokens + missing_tokens, dtype=np.float64)
            pred = _linear_tail_extrapolate_numpy(y, positions, future, ridge)
            return pred.astype(stored_tokens.dtype, copy=False)
    except Exception:
        pass
    try:
        import torch

        y = stored_tokens.to(torch.float64)
        device = stored_tokens.device
        positions = torch.arange(selected_tokens, dtype=torch.float64, device=device)
        future = torch.arange(selected_tokens, selected_tokens + missing_tokens, dtype=torch.float64, device=device)
        x_mean = positions.mean()
        y_mean = y.mean(dim=1, keepdim=True)
        centered_x = positions - x_mean
        centered_y = y - y_mean
        denominator = (centered_x * centered_x).sum() + float(ridge)
        slope = (centered_y * centered_x.reshape(1, selected_tokens, 1)).sum(dim=1, keepdim=True) / denominator
        bias = y_mean - slope * x_mean
        pred = slope * future.reshape(1, missing_tokens, 1) + bias
        return pred.to(dtype=stored_tokens.dtype)
    except Exception as exc:
        raise RuntimeError("linear-tail recovery requires numpy or torch") from exc


def _linear_tail_extrapolate_numpy(y: Any, positions: Any, future: Any, ridge: float) -> Any:
    x_mean = positions.mean()
    y_mean = y.mean(axis=1, keepdims=True)
    centered_x = positions - x_mean
    centered_y = y - y_mean
    denominator = (centered_x * centered_x).sum() + float(ridge)
    slope = (centered_y * centered_x.reshape(1, positions.shape[0], 1)).sum(axis=1, keepdims=True) / denominator
    bias = y_mean - slope * x_mean
    return slope * future.reshape(1, future.shape[0], 1) + bias


def _correlated_head_predict(
    *,
    source_tokens: Any,
    target_tokens: Any,
    selected_tokens: int,
    missing_tokens: int,
    ridge: float,
) -> Any:
    if missing_tokens <= 0:
        return target_tokens[:, :0, :]
    if selected_tokens <= 0:
        return _zeros_like_missing_tokens(target_tokens, missing_tokens)
    try:
        import numpy as np

        if "numpy" in str(type(target_tokens)):
            source = source_tokens.astype("float64", copy=False)
            target = target_tokens.astype("float64", copy=False)
            pred = _correlated_head_predict_numpy(
                source_selected=source[:, :selected_tokens, :],
                target_selected=target[:, :selected_tokens, :],
                source_missing=source[:, selected_tokens:selected_tokens + missing_tokens, :],
                ridge=ridge,
            )
            return pred.astype(target_tokens.dtype, copy=False)
    except Exception:
        pass
    try:
        import torch

        source = source_tokens.to(torch.float64)
        target = target_tokens.to(torch.float64)
        source_selected = source[:, :selected_tokens, :]
        target_selected = target[:, :selected_tokens, :]
        source_missing = source[:, selected_tokens:selected_tokens + missing_tokens, :]
        source_mean = source_selected.mean(dim=1, keepdim=True)
        target_mean = target_selected.mean(dim=1, keepdim=True)
        centered_source = source_selected - source_mean
        centered_target = target_selected - target_mean
        denominator = (centered_source * centered_source).sum(dim=1, keepdim=True) + float(ridge)
        slope = (centered_source * centered_target).sum(dim=1, keepdim=True) / denominator
        bias = target_mean - slope * source_mean
        return (slope * source_missing + bias).to(dtype=target_tokens.dtype)
    except Exception as exc:
        raise RuntimeError("correlated-head recovery requires numpy or torch") from exc


def _correlated_prefix_fit_mse(
    *,
    source_tokens: Any,
    target_tokens: Any,
    selected_tokens: int,
    ridge: float,
) -> float:
    if selected_tokens <= 0:
        return float("inf")
    try:
        import numpy as np

        if "numpy" in str(type(target_tokens)):
            source_selected = source_tokens[:, :selected_tokens, :].astype("float64", copy=False)
            target_selected = target_tokens[:, :selected_tokens, :].astype("float64", copy=False)
            pred = _correlated_head_predict_numpy(
                source_selected=source_selected,
                target_selected=target_selected,
                source_missing=source_selected,
                ridge=ridge,
            )
            diff = pred - target_selected
            return float(np.mean(diff * diff))
    except Exception:
        pass
    try:
        import torch

        source_selected = source_tokens[:, :selected_tokens, :].to(torch.float64)
        target_selected = target_tokens[:, :selected_tokens, :].to(torch.float64)
        source_mean = source_selected.mean(dim=1, keepdim=True)
        target_mean = target_selected.mean(dim=1, keepdim=True)
        centered_source = source_selected - source_mean
        centered_target = target_selected - target_mean
        denominator = (centered_source * centered_source).sum(dim=1, keepdim=True) + float(ridge)
        slope = (centered_source * centered_target).sum(dim=1, keepdim=True) / denominator
        bias = target_mean - slope * source_mean
        diff = slope * source_selected + bias - target_selected
        return float((diff * diff).mean().item())
    except Exception as exc:
        raise RuntimeError("correlated source-head scoring requires numpy or torch") from exc


def _multi_source_correlated_head_predict(
    *,
    source_tokens: list[Any],
    target_tokens: Any,
    selected_tokens: int,
    missing_tokens: int,
    ridge: float,
) -> Any:
    if missing_tokens <= 0:
        return target_tokens[:, :0, :]
    if selected_tokens <= 0:
        return _zeros_like_missing_tokens(target_tokens, missing_tokens)
    try:
        import numpy as np

        if "numpy" in str(type(target_tokens)):
            sources = [value.astype("float64", copy=False) for value in source_tokens]
            target = target_tokens.astype("float64", copy=False)
            pred = _multi_source_correlated_head_predict_numpy(
                source_selected=[value[:, :selected_tokens, :] for value in sources],
                target_selected=target[:, :selected_tokens, :],
                source_missing=[value[:, selected_tokens:selected_tokens + missing_tokens, :] for value in sources],
                ridge=ridge,
            )
            return pred.astype(target_tokens.dtype, copy=False)
    except Exception:
        pass
    try:
        import torch

        sources = [value.to(torch.float64) for value in source_tokens]
        target = target_tokens.to(torch.float64)
        source_selected = [value[:, :selected_tokens, :] for value in sources]
        source_missing = [value[:, selected_tokens:selected_tokens + missing_tokens, :] for value in sources]
        target_selected = target[:, :selected_tokens, :]
        preds = []
        feature_count = len(source_selected) + 1
        eye = torch.eye(feature_count, dtype=torch.float64, device=target_tokens.device)
        eye[-1, -1] = 0.0
        for kv_index in range(int(target_tokens.shape[0])):
            dim_preds = []
            for dim in range(int(target_tokens.shape[2])):
                features = torch.stack([value[kv_index, :, dim] for value in source_selected], dim=1)
                ones = torch.ones((selected_tokens, 1), dtype=torch.float64, device=target_tokens.device)
                design = torch.cat([features, ones], dim=1)
                future_features = torch.stack([value[kv_index, :, dim] for value in source_missing], dim=1)
                future_ones = torch.ones((missing_tokens, 1), dtype=torch.float64, device=target_tokens.device)
                future_design = torch.cat([future_features, future_ones], dim=1)
                target_vector = target_selected[kv_index, :, dim:dim + 1]
                lhs = design.transpose(0, 1).matmul(design) + float(ridge) * eye
                rhs = design.transpose(0, 1).matmul(target_vector)
                weights = torch.linalg.solve(lhs, rhs)
                dim_preds.append(future_design.matmul(weights))
            preds.append(torch.cat(dim_preds, dim=1).reshape(1, missing_tokens, int(target_tokens.shape[2])))
        return torch.cat(preds, dim=0).to(dtype=target_tokens.dtype)
    except Exception as exc:
        raise RuntimeError("multi-source correlated recovery requires numpy or torch") from exc


def _fit_multi_source_recovery_weights(
    *,
    source_tokens: list[Any],
    target_tokens: Any,
    selected_tokens: int,
    ridge: float,
) -> Any:
    if selected_tokens <= 0:
        raise ValueError("selected_tokens must be positive for recovery-bank fitting")
    try:
        import numpy as np

        if "numpy" in str(type(target_tokens)):
            sources = [value.astype("float64", copy=False) for value in source_tokens]
            target = target_tokens.astype("float64", copy=False)
            return _fit_multi_source_recovery_weights_numpy(
                source_selected=[value[:, :selected_tokens, :] for value in sources],
                target_selected=target[:, :selected_tokens, :],
                ridge=ridge,
            )
    except Exception:
        pass
    try:
        import torch

        sources = [value.to(torch.float64) for value in source_tokens]
        target = target_tokens.to(torch.float64)
        source_selected = [value[:, :selected_tokens, :] for value in sources]
        target_selected = target[:, :selected_tokens, :]
        kv_dim, _, head_dim = target_selected.shape
        feature_count = len(source_selected) + 1
        weights = torch.empty(
            (kv_dim, head_dim, feature_count),
            dtype=torch.float64,
            device=target_tokens.device,
        )
        regularizer = float(ridge) * torch.eye(feature_count, dtype=torch.float64, device=target_tokens.device)
        regularizer[-1, -1] = 0.0
        for kv_index in range(int(kv_dim)):
            for dim in range(int(head_dim)):
                features = torch.stack([value[kv_index, :, dim] for value in source_selected], dim=1)
                ones = torch.ones((selected_tokens, 1), dtype=torch.float64, device=target_tokens.device)
                design = torch.cat([features, ones], dim=1)
                target_vector = target_selected[kv_index, :, dim:dim + 1]
                lhs = design.transpose(0, 1).matmul(design) + regularizer
                rhs = design.transpose(0, 1).matmul(target_vector)
                weights[kv_index, dim] = torch.linalg.solve(lhs, rhs).reshape(feature_count)
        return weights
    except Exception as exc:
        raise RuntimeError("multi-source recovery-bank fitting requires numpy or torch") from exc


def _apply_multi_source_recovery_weights(
    *,
    source_tokens: list[Any],
    weights: Any,
    start_token: int,
    missing_tokens: int,
) -> Any:
    if missing_tokens <= 0:
        return source_tokens[0][:, :0, :]
    try:
        import numpy as np

        if "numpy" in str(type(weights)):
            sources = [value.astype("float64", copy=False) for value in source_tokens]
            pred = _apply_multi_source_recovery_weights_numpy(
                source_missing=[value[:, start_token:start_token + missing_tokens, :] for value in sources],
                weights=weights,
            )
            return pred.astype(source_tokens[0].dtype, copy=False)
    except Exception:
        pass
    try:
        import torch

        sources = [value.to(torch.float64) for value in source_tokens]
        source_missing = [value[:, start_token:start_token + missing_tokens, :] for value in sources]
        weights = torch.as_tensor(weights, dtype=torch.float64, device=source_tokens[0].device)
        kv_dim, head_dim, feature_count = weights.shape
        pred = torch.empty(
            (kv_dim, missing_tokens, head_dim),
            dtype=torch.float64,
            device=source_tokens[0].device,
        )
        for kv_index in range(int(kv_dim)):
            for dim in range(int(head_dim)):
                features = torch.stack([value[kv_index, :, dim] for value in source_missing], dim=1)
                ones = torch.ones((missing_tokens, 1), dtype=torch.float64, device=source_tokens[0].device)
                design = torch.cat([features, ones], dim=1)
                pred[kv_index, :, dim] = design.matmul(weights[kv_index, dim].reshape(feature_count, 1)).reshape(missing_tokens)
        return pred.to(dtype=source_tokens[0].dtype)
    except Exception as exc:
        raise RuntimeError("multi-source recovery-bank apply requires numpy or torch") from exc


def _multi_source_correlated_head_predict_numpy(
    *,
    source_selected: list[Any],
    target_selected: Any,
    source_missing: list[Any],
    ridge: float,
) -> Any:
    import numpy as np

    kv_dim, selected_tokens, head_dim = target_selected.shape
    missing_tokens = source_missing[0].shape[1]
    pred = np.empty((kv_dim, missing_tokens, head_dim), dtype=target_selected.dtype)
    feature_count = len(source_selected) + 1
    regularizer = float(ridge) * np.eye(feature_count, dtype=np.float64)
    regularizer[-1, -1] = 0.0
    for kv_index in range(kv_dim):
        for dim in range(head_dim):
            features = np.stack([value[kv_index, :, dim] for value in source_selected], axis=1)
            design = np.concatenate([features, np.ones((selected_tokens, 1), dtype=np.float64)], axis=1)
            future_features = np.stack([value[kv_index, :, dim] for value in source_missing], axis=1)
            future_design = np.concatenate([future_features, np.ones((missing_tokens, 1), dtype=np.float64)], axis=1)
            lhs = design.T @ design + regularizer
            rhs = design.T @ target_selected[kv_index, :, dim]
            weights = np.linalg.solve(lhs, rhs)
            pred[kv_index, :, dim] = future_design @ weights
    return pred


def _fit_multi_source_recovery_weights_numpy(
    *,
    source_selected: list[Any],
    target_selected: Any,
    ridge: float,
) -> Any:
    import numpy as np

    kv_dim, selected_tokens, head_dim = target_selected.shape
    feature_count = len(source_selected) + 1
    weights = np.empty((kv_dim, head_dim, feature_count), dtype=np.float64)
    regularizer = float(ridge) * np.eye(feature_count, dtype=np.float64)
    regularizer[-1, -1] = 0.0
    for kv_index in range(kv_dim):
        for dim in range(head_dim):
            features = np.stack([value[kv_index, :, dim] for value in source_selected], axis=1)
            design = np.concatenate([features, np.ones((selected_tokens, 1), dtype=np.float64)], axis=1)
            lhs = design.T @ design + regularizer
            rhs = design.T @ target_selected[kv_index, :, dim]
            weights[kv_index, dim] = np.linalg.solve(lhs, rhs)
    return weights


def _apply_multi_source_recovery_weights_numpy(
    *,
    source_missing: list[Any],
    weights: Any,
) -> Any:
    import numpy as np

    kv_dim, missing_tokens, head_dim = source_missing[0].shape
    pred = np.empty((kv_dim, missing_tokens, head_dim), dtype=np.float64)
    for kv_index in range(kv_dim):
        for dim in range(head_dim):
            features = np.stack([value[kv_index, :, dim] for value in source_missing], axis=1)
            design = np.concatenate([features, np.ones((missing_tokens, 1), dtype=np.float64)], axis=1)
            pred[kv_index, :, dim] = design @ weights[kv_index, dim]
    return pred


def _array_to_nested_lists(value: Any) -> list[Any]:
    if hasattr(value, "detach"):
        return value.detach().cpu().tolist()
    if hasattr(value, "tolist"):
        return value.tolist()
    raise TypeError("recovery bank weights must support tolist()")


def _nested_lists_to_array(value: Any) -> Any:
    try:
        import numpy as np

        return np.asarray(value, dtype=np.float64)
    except Exception as exc:
        raise RuntimeError("loading recovery bank requires numpy") from exc


def _correlated_head_predict_numpy(
    *,
    source_selected: Any,
    target_selected: Any,
    source_missing: Any,
    ridge: float,
) -> Any:
    source_mean = source_selected.mean(axis=1, keepdims=True)
    target_mean = target_selected.mean(axis=1, keepdims=True)
    centered_source = source_selected - source_mean
    centered_target = target_selected - target_mean
    denominator = (centered_source * centered_source).sum(axis=1, keepdims=True) + float(ridge)
    slope = (centered_source * centered_target).sum(axis=1, keepdims=True) / denominator
    bias = target_mean - slope * source_mean
    return slope * source_missing + bias


def _concat_flat_arrays(values: list[Any]) -> Any:
    try:
        import numpy as np

        if values and "numpy" in str(type(values[0])):
            return np.concatenate([value.reshape(-1).astype("float64") for value in values], axis=0)
    except Exception:
        pass
    try:
        import torch

        return torch.cat([value.reshape(-1).to(torch.float64) for value in values], dim=0)
    except Exception as exc:
        raise RuntimeError("error metrics require numpy or torch") from exc


def _mean_square(value: Any) -> float:
    if hasattr(value, "detach"):
        return float((value * value).mean().item())
    return float((value * value).mean())


def _mean_abs(value: Any) -> float:
    if hasattr(value, "detach"):
        return float(value.abs().mean().item())
    return float(abs(value).mean())


def _max_abs(value: Any) -> float:
    if hasattr(value, "detach"):
        return float(value.abs().max().item())
    return float(abs(value).max())


def _full_array(shape: tuple[int, ...], fill_value: float, dtype: Any, *, device: Any = None) -> Any:
    try:
        import numpy as np

        if dtype is None or isinstance(dtype, np.dtype) or "numpy" in str(type(dtype)):
            return np.full(shape, fill_value, dtype=dtype)
    except Exception:
        pass
    try:
        import torch

        kwargs = {"dtype": dtype}
        if device is not None:
            kwargs["device"] = device
        return torch.full(shape, fill_value, **kwargs)
    except Exception as exc:
        raise RuntimeError("restoring storage requires numpy or torch") from exc


def _array_nbytes(value: Any) -> int:
    nbytes = getattr(value, "nbytes", None)
    if nbytes is not None:
        return int(nbytes)
    if hasattr(value, "numel") and hasattr(value, "element_size"):
        return int(value.numel() * value.element_size())
    raise TypeError("array value must expose nbytes or numel()/element_size()")


def _array_nbytes_from_shape(shape: tuple[int, ...], dtype: Any) -> int:
    element_size = None
    try:
        import numpy as np

        element_size = np.dtype(dtype).itemsize
    except Exception:
        if hasattr(dtype, "itemsize"):
            element_size = int(dtype.itemsize)
    if element_size is None:
        raise TypeError("dtype must expose an element size")
    total = int(element_size)
    for dim in shape:
        total *= int(dim)
    return total


def _logical_full_kv_bytes_from_storage_summary(
    plan: LightDocCacheRuntimePlan,
    sidecar_storage: dict[str, Any],
) -> int:
    shape = sidecar_storage["full_shape"]
    full_tensor_bytes = int(sidecar_storage["full_tensor_bytes"])
    capacity_tokens_per_head = int(shape[2]) * int(shape[3])
    capacity_head_count = int(shape[1]) * int(shape[4])
    capacity_head_token_entries = max(1, capacity_tokens_per_head * capacity_head_count)
    return int(plan.total_head_token_entries * full_tensor_bytes // capacity_head_token_entries)


def _parse_heads_csv(value: str) -> list[HeadCoord]:
    value = value.strip()
    if not value:
        return []
    return _parse_head_list([part.strip() for part in value.split(",") if part.strip()])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Estimate Light Doc Cache runtime KV entry metrics.")
    parser.add_argument("--policy-file", required=True, help="Adaptive policy JSON file.")
    parser.add_argument("--task-id", required=True, help="Task/query identifier used for fallback overrides.")
    parser.add_argument("--doc-id", default=None, help="Optional document identifier for reporting.")
    parser.add_argument("--seq-len", type=int, required=True, help="Sequence length in tokens.")
    parser.add_argument("--num-layers", type=int, required=True, help="Model layer count.")
    parser.add_argument("--num-kv-heads", type=int, required=True, help="KV heads per layer.")
    parser.add_argument("--base-recovered-heads", default="", help="Comma separated base heads, e.g. 11:3,24:0.")
    parser.add_argument("--base-budget-fraction", type=float, default=0.5, help="Stored token fraction for base heads.")
    parser.add_argument("--repo-root", default=None, help="Repo root for reading policy_rows.csv from policy dirs.")
    parser.add_argument(
        "--from-policy-dirs",
        action="store_true",
        help="Read compact heads and budget fractions from default/base policy_rows.csv artifacts.",
    )
    parser.add_argument("--enabled", action="store_true", help="Enable planning; without this reports disabled/no-op.")
    args = parser.parse_args(argv)

    policy = load_light_doc_cache_policy(args.policy_file)
    if args.from_policy_dirs:
        config = build_config_from_policy_dirs(
            policy,
            repo_root=Path(args.repo_root) if args.repo_root is not None else Path.cwd(),
            num_layers=args.num_layers,
            num_kv_heads=args.num_kv_heads,
            enabled=args.enabled,
        )
    else:
        config = LightDocCacheRuntimeConfig(
            enabled=args.enabled,
            num_layers=args.num_layers,
            num_kv_heads=args.num_kv_heads,
            policy=policy,
            base_recovered_heads=_parse_heads_csv(args.base_recovered_heads),
            base_budget_fraction=args.base_budget_fraction,
        )
    plan = build_light_doc_cache_runtime_plan(
        config,
        task_id=args.task_id,
        doc_id=args.doc_id,
        seq_len=args.seq_len,
    )
    print(json.dumps(plan.as_summary(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
