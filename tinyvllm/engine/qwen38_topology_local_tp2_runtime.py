from __future__ import annotations

import time

import torch

from tinyvllm.engine.qwen38_topology_local_tp2_state import (
    build_qwen38_topology_local_tp2_state_owner,
)
from tinyvllm.layers.qwen38_topology_local_tp2_linear_attention import (
    Qwen38TopologyLocalTP2LinearAttention,
    build_logical_tp2_linear_attention_view,
    release_global_tp4_decode_accumulation,
)


_RUNTIME_SNAPSHOT_SCHEMA = (
    "qwen38.topology-local-tp2-runtime-snapshot.v1"
)
_FROZEN_REPOSITORY = "Qwen/Qwen3.8-27B"
_FROZEN_REVISION = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
_PAIR_GROUPS = ((0, 1), (2, 3))
_LINEAR_LAYER_INDICES = tuple(
    index for index in range(64) if index % 4 != 3
)


def _cohort_identity(leases: tuple[object, ...]) -> tuple[
    tuple[int, int, int],
    ...,
]:
    if not isinstance(leases, tuple) or not leases:
        raise ValueError("candidate leases must be a non-empty tuple")
    rows = []
    for lease in leases:
        try:
            row = (
                int(lease.slot_id),
                int(lease.generation),
                int(lease.request_id),
            )
        except (AttributeError, TypeError, ValueError) as error:
            raise ValueError(
                "candidate lease identity is invalid"
            ) from error
        if any(value < 0 for value in row):
            raise ValueError(
                "candidate lease identity must be non-negative"
            )
        rows.append(row)
    if len({row[0] for row in rows}) != len(rows):
        raise ValueError(
            "candidate leases must reference distinct slots"
        )
    return tuple(rows)


def _validate_profile(model) -> object:
    profile = getattr(model, "qwen38_text_profile", None)
    if profile is None:
        raise ValueError(
            "topology-local TP2 runtime requires Qwen3.8 profile"
        )
    if getattr(profile, "repository", None) != _FROZEN_REPOSITORY:
        raise ValueError(
            "Qwen3.8 topology-local TP2 repository mismatch"
        )
    if getattr(profile, "revision", None) != _FROZEN_REVISION:
        raise ValueError(
            "Qwen3.8 topology-local TP2 revision mismatch"
        )
    if getattr(profile, "dtype", None) != "bfloat16":
        raise ValueError(
            "Qwen3.8 topology-local TP2 requires BF16"
        )
    layer_types = tuple(getattr(profile, "layer_types", ()))
    if (
        getattr(profile, "num_hidden_layers", None) != 64
        or len(layer_types) != 64
        or tuple(
            index
            for index, layer_type in enumerate(layer_types)
            if layer_type == "linear_attention"
        ) != _LINEAR_LAYER_INDICES
        or any(
            layer_type
            not in {"linear_attention", "full_attention"}
            for layer_type in layer_types
        )
    ):
        raise ValueError(
            "Qwen3.8 topology does not match the frozen 64-layer layout"
        )
    return profile


def _validate_installation_graph(
    model,
    owner,
    pair_context,
    capacity: int,
) -> tuple[object, tuple[object, ...]]:
    if (
        isinstance(capacity, bool)
        or not isinstance(capacity, int)
        or capacity <= 0
    ):
        raise ValueError("candidate state capacity must be positive")
    if getattr(owner, "model", None) is not model:
        raise ValueError("baseline owner must own the candidate model")
    layer_stack = getattr(model, "layer_stack", None)
    if (
        layer_stack is None
        or getattr(owner, "layer_stack", None) is not layer_stack
        or getattr(owner, "state_transaction", None)
        is not getattr(layer_stack, "state_transaction", None)
    ):
        raise ValueError(
            "baseline owner must preserve one coherent model graph"
        )
    pool = getattr(owner, "pool", None)
    if pool is None or getattr(pool, "capacity", None) != capacity:
        raise ValueError(
            "candidate capacity must match the baseline state pool"
        )
    layers = tuple(getattr(layer_stack, "layers", ()))
    if len(layers) != 64:
        raise ValueError(
            "Qwen3.8 topology must contain exactly 64 layers"
        )
    linear_indices = tuple(
        index
        for index, layer in enumerate(layers)
        if getattr(layer, "block_type", None) == "linear_attention"
    )
    if (
        linear_indices != _LINEAR_LAYER_INDICES
        or tuple(getattr(layer_stack, "linear_indices", ()))
        != _LINEAR_LAYER_INDICES
    ):
        raise ValueError(
            "Qwen3.8 topology must contain exactly 48 linear layers"
        )
    if any(
        getattr(layer, "block_type", None) != "full_attention"
        for index, layer in enumerate(layers)
        if index not in _LINEAR_LAYER_INDICES
    ):
        raise ValueError(
            "Qwen3.8 topology has an invalid full-attention cadence"
        )
    pair_map = getattr(pair_context, "pair_map", None)
    if getattr(pair_map, "pair_groups", None) != _PAIR_GROUPS:
        raise ValueError(
            "topology-local TP2 pair map must be ((0, 1), (2, 3))"
        )
    identity = getattr(pair_context, "identity", None)
    if (
        identity is None
        or getattr(identity, "global_rank", None) not in range(4)
        or getattr(identity, "logical_rank", None) not in (0, 1)
        or getattr(pair_context, "pair_group", None) is None
        or tuple(getattr(pair_context, "all_pair_groups", ()))
        == ()
    ):
        raise ValueError(
            "topology-local TP2 pair context is invalid"
        )
    return layer_stack, layers


def _world_all_gather(local):
    gathered = [
        torch.empty_like(local)
        for _ in range(4)
    ]
    torch.distributed.all_gather(gathered, local)
    return tuple(gathered)


def _pair_reduce(output, pair_group) -> None:
    torch.distributed.all_reduce(output, group=pair_group)


class Qwen38TopologyLocalTP2Runtime:

    def __init__(
        self,
        *,
        model,
        baseline_owner,
        candidate_state_owner,
        pair_context,
        mixers: tuple[
            Qwen38TopologyLocalTP2LinearAttention,
            ...,
        ],
        linear_layer_indices: tuple[int, ...],
        baseline_mixers: tuple[object, ...],
    ):
        self.model = model
        self.baseline_owner = baseline_owner
        self.candidate_state_owner = candidate_state_owner
        self.pair_context = pair_context
        self.mixers = mixers
        self.linear_layer_indices = linear_layer_indices
        self._baseline_mixers = baseline_mixers
        self.phase = "tp4_prefill"
        self._fixed_cohort = None
        self._active_leases = ()
        self._release_rows: tuple[dict, ...] = ()
        self._decode_accumulation_released = False
        self._transition_count = 0
        self._last_transition_latency_ns = None
        self._candidate_state_released = False

    def prepare_decode(self, leases: tuple[object, ...]) -> dict:
        cohort = _cohort_identity(leases)
        if self._fixed_cohort is None:
            self._fixed_cohort = cohort
        elif cohort != self._fixed_cohort:
            raise RuntimeError("fixed candidate cohort changed")
        if self.phase != "tp4_prefill":
            raise RuntimeError(
                f"candidate decode phase is already active: {self.phase}"
            )

        published = False
        started_ns = time.monotonic_ns()
        try:
            migration_rows = self.candidate_state_owner.migrate(
                leases
            )
            published = True
            self.model.layer_stack.state_transaction = (
                self.candidate_state_owner.destination_transaction
            )
            for mixer in self.mixers:
                mixer.activate_tp2_decode()
            release_rows = ()
            if not self._decode_accumulation_released:
                release_rows = tuple(
                    {
                        "layer_index": layer_index,
                        **release_global_tp4_decode_accumulation(
                            baseline,
                        ),
                    }
                    for layer_index, baseline in zip(
                        self.linear_layer_indices,
                        self._baseline_mixers,
                    )
                )
            torch.cuda.synchronize()
        except Exception:
            if published:
                self.phase = "quarantined"
            raise

        self.phase = "tp2_decode"
        self._last_transition_latency_ns = (
            time.monotonic_ns() - started_ns
        )
        self._active_leases = leases
        if release_rows:
            self._release_rows = release_rows
            self._decode_accumulation_released = True
        self._transition_count += 1
        return {
            "migration_rows": migration_rows,
            "release_rows": release_rows,
            "released_layer_count": len(release_rows),
            "released_bytes": sum(
                int(row["released_bytes"])
                for row in release_rows
            ),
            "transition_latency_ns": self._last_transition_latency_ns,
            "synchronized_before_measured_decode": True,
        }

    def release_decode_cohort(
        self,
        leases: tuple[object, ...],
    ) -> dict:
        cohort = _cohort_identity(leases)
        if (
            self.phase != "tp2_decode"
            or self._fixed_cohort is None
            or cohort != self._fixed_cohort
            or cohort != _cohort_identity(self._active_leases)
        ):
            raise RuntimeError(
                "candidate release must cover the complete fixed cohort"
            )
        try:
            self.candidate_state_owner.release(leases)
            self.model.layer_stack.state_transaction = (
                self.baseline_owner.state_transaction
            )
            for mixer in self.mixers:
                mixer.activate_tp4_prefill()
        except Exception:
            self.phase = "quarantined"
            raise
        self.phase = "tp4_prefill"
        self._fixed_cohort = None
        self._active_leases = ()
        self._candidate_state_released = True
        return {
            "released_requests": len(leases),
            "phase": self.phase,
            "fixed_cohort_cleared": True,
        }

    def snapshot(self) -> dict:
        identity = self.pair_context.identity
        return {
            "schema_version": _RUNTIME_SNAPSHOT_SCHEMA,
            "enabled": True,
            "rank": int(identity.global_rank),
            "pair_id": int(identity.pair_id),
            "logical_rank": int(identity.logical_rank),
            "pair_ranks": tuple(identity.pair_ranks),
            "linear_layer_indices": self.linear_layer_indices,
            "phase": self.phase,
            "fixed_cohort": self._fixed_cohort,
            "transition_count": self._transition_count,
            "last_transition_latency_ns":
                self._last_transition_latency_ns,
            "released_layer_count": len(self._release_rows),
            "released_bytes": sum(
                int(row["released_bytes"])
                for row in self._release_rows
            ),
            "state": self.candidate_state_owner.snapshot(),
            "mixers": tuple(
                mixer.telemetry_snapshot()
                for mixer in self.mixers
            ),
        }

    def close(self) -> dict:
        if self.phase == "closed":
            raise RuntimeError(
                "topology-local TP2 runtime is already closed"
            )
        candidate_state_released = self._candidate_state_released
        if self._active_leases:
            self.candidate_state_owner.release(
                self._active_leases
            )
            self._active_leases = ()
            candidate_state_released = True
        destroyed = 0
        for group in self.pair_context.all_pair_groups:
            torch.distributed.destroy_process_group(group)
            destroyed += 1
        self.phase = "closed"
        state = self.candidate_state_owner.snapshot()
        published_remaining = sum(
            row.get("phase") == "tp2_decode"
            for row in state.get("leases", ())
        )
        return {
            "pair_groups_destroyed": destroyed,
            "candidate_state_released": candidate_state_released,
            "published_generations_remaining": published_remaining,
            "temporary_live_tensors": int(
                state.get("temporary_live_tensors", 0)
            ),
        }


def install_qwen38_topology_local_tp2_runtime(
    model,
    owner,
    pair_context,
    capacity,
) -> Qwen38TopologyLocalTP2Runtime:
    if getattr(
        model,
        "_qwen38_topology_local_tp2_runtime",
        None,
    ) is not None:
        raise RuntimeError(
            "Qwen3.8 topology-local TP2 runtime is already installed"
        )
    _validate_profile(model)
    layer_stack, layers = _validate_installation_graph(
        model,
        owner,
        pair_context,
        capacity,
    )
    hf_config = getattr(model, "qwen38_hf_config", None)
    if hf_config is None:
        raise ValueError(
            "Qwen3.8 topology-local TP2 runtime requires hf_config"
        )

    baseline_mixers = tuple(
        layers[index].linear_attention
        for index in _LINEAR_LAYER_INDICES
    )
    views = tuple(
        build_logical_tp2_linear_attention_view(
            baseline,
            logical_rank=pair_context.identity.logical_rank,
            pair_group=pair_context.pair_group,
        )
        for baseline in baseline_mixers
    )
    mixers = tuple(
        Qwen38TopologyLocalTP2LinearAttention(
            baseline=baseline,
            candidate_view=view,
            pair_reduce=_pair_reduce,
        )
        for baseline, view in zip(baseline_mixers, views)
    )
    candidate_state_owner = (
        build_qwen38_topology_local_tp2_state_owner(
            hf_config=hf_config,
            capacity=capacity,
            device=owner.pool.device,
            source_transaction=owner.state_transaction,
            pair_identity=pair_context.identity,
            all_gather=_world_all_gather,
        )
    )

    replaced = []
    try:
        for layer_index, mixer in zip(
            _LINEAR_LAYER_INDICES,
            mixers,
        ):
            layer = layers[layer_index]
            replaced.append((
                layer,
                layer.linear_attention,
            ))
            layer.linear_attention = mixer
    except Exception:
        for layer, baseline in reversed(replaced):
            layer.linear_attention = baseline
        raise

    runtime = Qwen38TopologyLocalTP2Runtime(
        model=model,
        baseline_owner=owner,
        candidate_state_owner=candidate_state_owner,
        pair_context=pair_context,
        mixers=mixers,
        linear_layer_indices=_LINEAR_LAYER_INDICES,
        baseline_mixers=baseline_mixers,
    )
    model._qwen38_topology_local_tp2_runtime = runtime
    return runtime
