from __future__ import annotations

from dataclasses import asdict
import torch
import pickle
import os
import time
import flash_attn
import hashlib
import json
from itertools import count
from types import SimpleNamespace

import torch.distributed as dist
from tinyvllm.config import Config
from tinyvllm.engine.exact_cuda_graph_cache import (
    ExactCudaGraphCache,
    ExactCudaGraphCacheConfig,
    ExactCudaGraphEntry,
)
from tinyvllm.engine.spec_verify_exact_cuda_graph_cache import (
    SpecVerifyCaptureScratchPool,
    SpecVerifyExactCudaGraphCache,
    SpecVerifyExactCudaGraphCacheConfig,
    SpecVerifyExactCudaGraphEntry,
    SpecVerifyGraphReplayError,
    SpecVerifyGraphIdentity,
    required_spec_verify_capture_scratch_blocks,
)
from tinyvllm.engine.flash_attn_split_policy import (
    FlashAttentionSplitInputs,
    build_flash_attn_263_graph_identity,
)
from tinyvllm.engine.sequence import Sequence
from tinyvllm.engine.hybrid_state import HybridStateLease
from tinyvllm.engine.model_runner_command_ack import (
    ModelRunnerCommandEnvelope,
    execute_acknowledged_command,
)
from tinyvllm.engine.model_runner_command_timeline import (
    CommandTraceIdentity,
    ModelRunnerCommandTimelineRecorder,
    read_command_clock_identity,
)
from tinyvllm.engine.h2d_slot_reuse_diagnostic import (
    H2D_SLOT_REUSE_SCHEMA,
    H2DSlotReuseDiagnostic,
)
from tinyvllm.engine.spec_verify_trace import (
    SpecVerifyTraceRecorder,
    TargetForwardTraceContext,
)
from tinyvllm.engine.tensor_parallel_greedy import (
    select_tensor_parallel_greedy_tokens,
)
from tinyvllm.engine.decode_internal_profiler import (
    DecodeInternalProfiler,
    run_profiled_step,
)
from tinyvllm.engine.qwen35_hybrid_prefix_restore_ticket import (
    Qwen35HybridPrefixRestoreParticipant,
)
from tinyvllm.engine.qwen35_hybrid_prefix_publication_ticket import (
    Qwen35HybridPrefixPublicationParticipant,
)
from tinyvllm.engine.qwen35_hybrid_prefix_owner import (
    Qwen35HybridPrefixRestoreOwner,
    build_qwen35_hybrid_prefix_restore_owner,
)
from tinyvllm.engine.qwen35_hybrid_model_owner import (
    Qwen35HybridModelOwner,
    build_qwen35_hybrid_model_owner,
)
from tinyvllm.engine.qwen35_hybrid_model_publication import (
    Qwen35HybridModelOwnerPublicationSlot,
)
from tinyvllm.engine.qwen35_hybrid_prefix_runtime_identity import (
    bind_qwen35_hybrid_prefix_runtime_identity as _bind_qwen35_hybrid_prefix_runtime_identity,
)
from tinyvllm.engine.qwen35_recurrent_capture import (
    Qwen35RecurrentCaptureSession,
)
from tinyvllm.engine.qwen35_recurrent_capture_contract import (
    CAPTURE_IDENTITY_SCHEMA_VERSION,
    validate_run_identity,
)
from tinyvllm.engine.qwen35_speculative_state import (
    Qwen35SpeculativeStateOwner,
)
from tinyvllm.models.qwen35_checkpoint_streaming import (
    Qwen35LoadedCheckpointCandidate,
    load_qwen35_fresh_checkpoint_candidate,
    move_qwen35_loaded_checkpoint_candidate_to_device,
)
from tinyvllm.models.qwen35_checkpoint_metadata import (
    Qwen35CheckpointShardIdentity,
    read_qwen35_checkpoint_metadata,
)
from tinyvllm.models.qwen35_checkpoint import (
    build_qwen35_checkpoint_tensor_plan,
)
from tinyvllm.models.qwen35_checkpoint_candidate_factory import (
    prepare_qwen35_checkpoint_candidate_target,
)
from tinyvllm.engine.qwen35_hybrid_state import (
    build_qwen35_hybrid_state_layout,
)
from tinyvllm.engine.hybrid_state import HybridStateTensorPool
from tinyvllm.layers.attention import Attention
from tinyvllm.models.qwen35_checkpoint_worker import (
    validate_qwen35_checkpoint_candidate_load_request,
)
from tinyvllm.models.qwen3 import Qwen3ForCausalLM
from tinyvllm.utils.loader import load_model
from tinyvllm.utils.cpu_offload import apply_cpu_offload
from tinyvllm.layers.linear import (
    configure_linear_execution_rows,
    set_quant_config,
)
from tinyvllm.layers.sampler import Sampler
from tinyvllm.utils.context import reset_context, set_context, get_context
from tinyvllm.engine.kv_cartridge import compress_decode_block_table_rows, should_use_kv_cartridge
from tinyvllm.engine.light_doc_cache_runtime import (
    build_model_runner_light_doc_cache_summary,
    materialize_model_runner_light_doc_cache_sidecar,
)
from tinyvllm.engine.speculative_residency import (
    KVBlockIdentityRow,
    SpeculativeResidencyParticipant,
    SpeculativeResidencyPrecommitRow,
    SpeculativeResidencyPrepareRow,
    SpeculativeResidencyResult,
)
from tinyvllm.speculative.batch_runtime import (
    FirstTargetProposalResult,
    FirstTargetResult,
)
from tinyvllm.engine.speculative_proposal_executor import (
    ModelRunnerProposalExecutorRegistry,
    ModelRunnerProposalInput,
    ProposalFinalizeRow,
    TargetPrefillObservation,
    assert_tensor_free,
    model_runner_proposal_token_context,
)
from tinyvllm.engine.proposal_kv_cache import ProposalKVCache
from tinyvllm.engine.proposal_kv_allocator import (
    DirectProposalKVAllocator,
)
from tinyvllm.engine.qwen35_mtp_registration import (
    build_qwen35_mtp_proposal_kv_allocator,
)
from tinyvllm.engine.autoregressive_draft_registration import (
    AutoregressiveDraftRegistrationCandidate,
    AutoregressiveDraftRegistrationError,
    build_autoregressive_draft_registration_status,
    build_autoregressive_draft_registration_dependencies,
    validate_autoregressive_draft_registration_consensus,
)
from tinyvllm.engine.autoregressive_draft_tp import (
    AutoregressiveDraftTensorParallelCoordinator,
)
from tinyvllm.engine.speculative_runtime import (
    ModelRunnerProposalExecutorDescriptor,
)
from tinyvllm.speculative.adapter import (
    validate_draft_capabilities,
)
from tinyvllm.speculative.verifier import (
    AttentionMode,
    SPEC_VERIFY_FLASH_ATTN_NUM_SPLITS,
    SpecVerifyBatchMetadata,
    SpecVerifyBatchResultRow,
    SpecVerifyBatchRowMetadata,
    SpecVerifyMetadata,
    SpecVerifyPlan,
    split_spec_verify_batch_target_tokens,
    validate_spec_verify_slots,
)

from multiprocessing.synchronize import Event
from multiprocessing.shared_memory import SharedMemory


DISPATCH_EVENT_FIELDS = (
    "step_id",
    "request_ids_hash",
    "mode",
    "active_batch_size",
    "page_table_width",
    "effective_num_splits",
    "graph_identity_sha256",
    "feature_enabled",
    "dispatch",
    "cache_state",
    "observation_count",
    "fallback_reason",
    "capture_attempted",
    "capture_duration_ns",
    "capture_static_bytes",
    "capture_allocated_delta_bytes",
    "capture_reserved_delta_bytes",
    "cache_ready_entries",
    "cache_static_bytes",
    "cache_reserved_delta_bytes",
    "cache_total_capture_ns",
    "source_sha256",
)


def _autoregressive_draft_registration_dependencies():
    return build_autoregressive_draft_registration_dependencies()

SPEC_VERIFY_DISPATCH_EVENT_FIELDS = (
    "step_id",
    "request_ids_hash",
    "mode",
    "active_batch_size",
    "query_len",
    "total_query_tokens",
    "page_table_width",
    "flash_attn_num_splits",
    "graph_identity_sha256",
    "feature_enabled",
    "dispatch",
    "decision",
    "fallback_reason",
    "cache_state",
    "observation_count",
    "capture_attempted",
    "capture_duration_ns",
    "capture_static_bytes",
    "capture_allocated_delta_bytes",
    "capture_reserved_delta_bytes",
    "cache_ready_entries",
    "cache_static_bytes",
    "cache_reserved_delta_bytes",
    "cache_total_capture_ns",
    "cache_hits",
    "cache_misses",
    "cache_evictions",
    "cache_quarantines",
    "transaction_authorized",
    "source_sha256",
)


def _resolve_hf_model_dtype(hf_config, torch_module):
    dtype = getattr(hf_config, "dtype", None)
    if dtype is None:
        dtype = getattr(hf_config, "torch_dtype", None)
    floating_dtypes = {
        torch_module.float16,
        torch_module.bfloat16,
        torch_module.float32,
        torch_module.float64,
    }
    if dtype not in floating_dtypes:
        if getattr(hf_config, "model_type", None) == "qwen3_5":
            dtype = torch_module.bfloat16
        else:
            dtype = torch_module.get_default_dtype()
    hf_config.torch_dtype = dtype
    return dtype


def _configure_qwen35_linear_execution(
    model,
    *,
    configure_rows,
):
    configure_rows(model, 1024)
    return model


def _initialize_model_runner_model(
    config,
    *,
    rank,
    load_legacy_model,
    load_qwen35_model,
):
    if getattr(config.hf_config, "model_type", None) == "qwen3_5":
        return load_qwen35_model(config, rank)
    return load_legacy_model(config), None


def _qwen35_checkpoint_manifest_identity(
    model_dir,
    *,
    path_type,
    json_module,
    hashlib_module,
):
    model_path = path_type(model_dir).resolve()
    manifest_path = model_path.parent / "model_manifest.json"
    payload = manifest_path.read_bytes()
    manifest_sha256 = hashlib_module.sha256(payload).hexdigest()
    manifest = json_module.loads(payload)
    files = manifest.get("files")
    if (
        manifest.get("schema_version") != 1
        or not isinstance(files, dict)
        or manifest.get("local_path") != str(model_path)
        or manifest.get("remote_model_dir") != str(model_path)
    ):
        raise ValueError("Qwen3.5 model manifest identity is invalid")
    required = (
        "config.json",
        "model.safetensors.index.json",
    )
    if any(
        not isinstance(files.get(name), dict)
        or set(files[name]) != {"sha256", "size"}
        for name in required
    ):
        raise ValueError("Qwen3.5 model manifest files are incomplete")
    shard_rows = tuple(
        (name, row)
        for name, row in sorted(files.items())
        if name.endswith(".safetensors")
    )
    if not shard_rows:
        raise ValueError("Qwen3.5 model manifest has no checkpoint shard")
    composite_payload = {
        "config_sha256": files["config.json"]["sha256"],
        "index_sha256": files[
            "model.safetensors.index.json"
        ]["sha256"],
        "shards": {
            name: {
                "sha256": row["sha256"],
                "size": row["size"],
            }
            for name, row in shard_rows
        },
    }
    composite = hashlib_module.sha256(
        json_module.dumps(
            composite_payload,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    return {
        "manifest_sha256": manifest_sha256,
        "config_sha256": files["config.json"]["sha256"],
        "index_sha256": files[
            "model.safetensors.index.json"
        ]["sha256"],
        "composite_sha256": composite,
        "shards": shard_rows,
    }


def _load_qwen35_model_runner_model(config, rank):
    from pathlib import Path

    identity = _qwen35_checkpoint_manifest_identity(
        config.model,
        path_type=Path,
        json_module=json,
        hashlib_module=hashlib,
    )
    shards = tuple(
        Qwen35CheckpointShardIdentity(
            name=name,
            size=row["size"],
            sha256=row["sha256"],
        )
        for name, row in identity["shards"]
    )
    metadata = read_qwen35_checkpoint_metadata(
        config.model,
        shards=shards,
        expected_config_sha256=identity["config_sha256"],
        expected_index_sha256=identity["index_sha256"],
        expected_config_index_header_sha256=(
            identity["composite_sha256"]
        ),
    )
    tensor_plan = build_qwen35_checkpoint_tensor_plan(
        metadata.hf_config,
        metadata.index_payload,
        metadata.shard_headers,
    )
    layout = build_qwen35_hybrid_state_layout(
        metadata.hf_config,
        tensor_parallel_size=config.tensor_parallel_size,
        dtype=torch.bfloat16,
        recurrent_dtype=torch.float32,
        speculative_tokens=1,
    )
    pool = HybridStateTensorPool(
        layout,
        capacity=config.max_num_seqs,
        device="cpu",
    )

    def build_attention_backend(
        _layer_index,
        query_heads,
        kv_heads,
        head_dim,
    ):
        return Attention(
            query_heads,
            head_dim,
            head_dim ** -0.5,
            kv_heads,
        )

    target = prepare_qwen35_checkpoint_candidate_target(
        metadata.hf_config,
        tensor_plan,
        pool=pool,
        tensor_parallel_size=config.tensor_parallel_size,
        tensor_parallel_rank=rank,
        build_attention_backend=build_attention_backend,
        parameter_device="cpu",
    )
    candidate = load_qwen35_fresh_checkpoint_candidate(
        target.take,
        config.model,
        max_tensor_bytes=1 << 30,
        model_fingerprint=identity["manifest_sha256"],
    )
    move_qwen35_loaded_checkpoint_candidate_to_device(
        candidate,
        torch.device("cuda"),
    )
    _configure_qwen35_linear_execution(
        candidate.owner.model,
        configure_rows=configure_linear_execution_rows,
    )
    return candidate.owner.model, candidate.owner


def _qwen35_mtp_registration_dependencies():
    from pathlib import Path

    from tinyvllm.engine.qwen35_mtp_executor import (
        Qwen35MTPProposalExecutor,
    )
    from tinyvllm.engine.qwen35_mtp_cuda_graph_backend import (
        Qwen35MTPCudaGraphBackend,
    )
    from tinyvllm.engine.qwen35_mtp_graph import (
        Qwen35MTPExactGraphRunner,
    )
    from tinyvllm.engine.qwen35_mtp_graph_scratch import (
        Qwen35MTPGraphScratchOwner,
    )
    from tinyvllm.models.qwen35_mtp import (
        build_qwen35_native_mtp,
    )
    from tinyvllm.models.qwen35_mtp_checkpoint import (
        bind_qwen35_mtp_checkpoint,
        build_qwen35_mtp_checkpoint_plan,
        read_qwen35_mtp_checkpoint_tensor,
    )

    def read_metadata(config):
        identity = _qwen35_checkpoint_manifest_identity(
            config.model,
            path_type=Path,
            json_module=json,
            hashlib_module=hashlib,
        )
        shards = tuple(
            Qwen35CheckpointShardIdentity(
                name=name,
                size=row["size"],
                sha256=row["sha256"],
            )
            for name, row in identity["shards"]
        )
        return read_qwen35_checkpoint_metadata(
            config.model,
            shards=shards,
            expected_config_sha256=identity["config_sha256"],
            expected_index_sha256=identity["index_sha256"],
            expected_config_index_header_sha256=(
                identity["composite_sha256"]
            ),
        )

    def build_attention_backend(
        _layer_index,
        query_heads,
        kv_heads,
        head_dim,
    ):
        return Attention(
            query_heads,
            head_dim,
            head_dim ** -0.5,
            kv_heads,
        )

    def build_graph_runner(config, _module, _cache):
        if not config.qwen35_mtp_cuda_graphs:
            return None
        if int(config.tensor_parallel_size) != 1:
            raise RuntimeError(
                "Qwen3.5 MTP CUDA graphs require TP1"
            )
        if bool(config.kv_offload_mvp0):
            raise RuntimeError(
                "Qwen3.5 MTP CUDA graphs require KV offload disabled"
            )
        weight = getattr(getattr(_module, "fc", None), "weight", None)
        if not isinstance(weight, torch.Tensor):
            raise RuntimeError(
                "Qwen3.5 MTP FC weight is unavailable"
            )
        if weight.device.type != "cuda":
            raise RuntimeError(
                "Qwen3.5 MTP CUDA graphs require CUDA weights"
            )
        if int(getattr(_module, "hidden_size", 0)) <= 0:
            raise RuntimeError(
                "Qwen3.5 MTP hidden size is unavailable"
            )
        q_allowlist = tuple(
            exact_q
            for exact_q in config.qwen35_mtp_cuda_graph_q_allowlist
            if exact_q in (2, 3, 4)
        )
        batch_allowlist = tuple(
            batch_size
            for batch_size
            in config.qwen35_mtp_cuda_graph_batch_allowlist
            if batch_size in (1, 4)
        )
        if not q_allowlist or not batch_allowlist:
            raise RuntimeError(
                "Qwen3.5 MTP CUDA graph allowlists have no "
                "supported exact family"
            )
        scratch_cache = ProposalKVCache(
            DirectProposalKVAllocator(
                _cache.entry_allocator.physical_store
            )
        )
        scratch_owner = Qwen35MTPGraphScratchOwner(
            live_cache=_cache,
            scratch_cache=scratch_cache,
        )
        block_table_width = int(config.max_model_len)
        capture_backend = Qwen35MTPCudaGraphBackend(
            module=_module,
            proposal_kv_cache=_cache,
            device=weight.device,
            compute_dtype=weight.dtype,
            hidden_size=int(_module.hidden_size),
            block_table_width=block_table_width,
        )
        return Qwen35MTPExactGraphRunner(
            enabled=True,
            q_allowlist=q_allowlist,
            batch_allowlist=batch_allowlist,
            min_observations=(
                config.qwen35_mtp_cuda_graph_min_observations
            ),
            max_entries=config.qwen35_mtp_cuda_graph_max_entries,
            max_static_bytes=(
                config.qwen35_mtp_cuda_graph_max_static_bytes
            ),
            max_reserved_bytes=(
                config.qwen35_mtp_cuda_graph_max_reserved_bytes
            ),
            max_total_capture_ns=(
                config.qwen35_mtp_cuda_graph_max_total_capture_ns
            ),
            max_single_capture_ns=(
                config.qwen35_mtp_cuda_graph_max_single_capture_ns
            ),
            device_index=(
                0
                if weight.device.index is None
                else int(weight.device.index)
            ),
            compute_dtype=str(weight.dtype),
            hidden_size=int(_module.hidden_size),
            mtp_layer_count=1,
            block_table_width=int(config.max_model_len),
            capture_backend=capture_backend,
            scratch_owner=scratch_owner,
        )

    def build_proposal_kv_allocator(config, module):
        backend = (
            module.layer.decoder_layer.full_attention
            .attention_backend
        )
        weight = getattr(getattr(module, "fc", None), "weight", None)
        if not isinstance(weight, torch.Tensor):
            raise RuntimeError(
                "Qwen3.5 MTP FC weight is unavailable"
            )
        direct_capacity = (
            int(config.num_kvcache_blocks)
            * int(config.kvcache_block_size)
        )
        offload_enabled = bool(
            config.proposal_kv_offload_enabled
        )
        return build_qwen35_mtp_proposal_kv_allocator(
            offload_enabled=offload_enabled,
            logical_entry_capacity=(
                int(config.proposal_kv_logical_entry_capacity)
                if offload_enabled
                else direct_capacity
            ),
            gpu_slot_capacity=(
                int(config.proposal_kv_gpu_slot_capacity)
                if offload_enabled
                else direct_capacity
            ),
            cpu_backing_capacity=(
                int(config.proposal_kv_cpu_backing_capacity)
                if offload_enabled
                else direct_capacity
            ),
            async_copy=bool(config.proposal_kv_async_copy),
            batch_copy=bool(config.proposal_kv_batch_copy),
            num_kv_heads=int(backend.num_kv_heads),
            head_dim=int(backend.head_dim),
            dtype=weight.dtype,
            device=weight.device,
        )

    def move_module_to_device(module, target_model):
        target_weight = getattr(
            getattr(target_model, "embed_tokens", None),
            "weight",
            None,
        )
        target_device = getattr(target_weight, "device", None)
        if target_device is None:
            raise RuntimeError(
                "Qwen3.5 target embedding device is unavailable"
            )
        module.to(device=target_device)
        if (
            module.embed_tokens is not target_model.embed_tokens
            or module.lm_head is not target_model.lm_head
        ):
            raise RuntimeError(
                "Qwen3.5 MTP shared modules lost identity "
                "during device movement"
            )

    return SimpleNamespace(
        read_metadata=read_metadata,
        build_checkpoint_plan=build_qwen35_mtp_checkpoint_plan,
        build_module=lambda hf_config, **kwargs: (
            build_qwen35_native_mtp(
                hf_config,
                build_attention_backend=build_attention_backend,
                parameter_device="cpu",
                **kwargs,
            )
        ),
        read_tensor=lambda config, tensor: (
            read_qwen35_mtp_checkpoint_tensor(
                config.model,
                tensor,
            )
        ),
        bind_checkpoint=bind_qwen35_mtp_checkpoint,
        move_module_to_device=move_module_to_device,
        build_proposal_kv_allocator=(
            build_proposal_kv_allocator
        ),
        build_proposal_kv_cache=ProposalKVCache,
        build_graph_runner=build_graph_runner,
        build_executor=lambda **kwargs: (
            Qwen35MTPProposalExecutor(**kwargs)
        ),
    )


def _run_model_runner_eager(
    model,
    *,
    input_ids,
    positions,
    input_embeds,
    active_leases,
    token_counts,
    return_hidden,
    prepare_qwen35_state: bool = False,
    initial_qwen35_candidates=None,
    capture_qwen35_prefix_states: bool = False,
):
    if prepare_qwen35_state:
        prepare_step = getattr(model, "prepare_step", None)
        if not callable(prepare_step):
            raise RuntimeError(
                "prepared Qwen3.5 execution requires prepare_step"
            )
        return prepare_step(
            active_leases,
            token_counts,
            input_ids,
            positions,
            input_embeds=input_embeds,
            initial_candidates=initial_qwen35_candidates,
            capture_prefix_states=capture_qwen35_prefix_states,
        )
    run_step = getattr(model, "run_step", None)
    if callable(run_step):
        hidden_states, logits = run_step(
            active_leases,
            token_counts,
            input_ids,
            positions,
            input_embeds=input_embeds,
        )
    else:
        hidden_states = model(
            input_ids,
            positions,
            input_embeds=input_embeds,
        )
        logits = model.compute_logits(hidden_states)
    if return_hidden:
        return logits, hidden_states
    return logits


def _qwen35_step_token_counts(
    seqs,
    *,
    is_prefill,
    batch_kind,
):
    def prefill_count(seq):
        chunk_start = getattr(
            seq,
            "prefill_chunk_start",
            seq.num_cached_tokens,
        )
        chunk_end = getattr(seq, "prefill_chunk_end", None)
        if chunk_end is None or (
            chunk_end == 0 and chunk_start == 0
        ):
            chunk_end = len(seq)
        return chunk_end - chunk_start

    if batch_kind == "mixed":
        return tuple(
            1
            if getattr(seq, "step_is_decode", False)
            else prefill_count(seq)
            for seq in seqs
        )
    if not is_prefill:
        return (1,) * len(seqs)
    return tuple(prefill_count(seq) for seq in seqs)


def _round_qwen35_final_prefill_recurrent_states(
    runtime_bridge,
    seqs,
    leases,
    *,
    is_prefill,
    batch_kind,
):
    if runtime_bridge is None:
        return
    pool = runtime_bridge.pool
    lease_by_request = {
        int(lease.request_id): lease
        for lease in leases
    }
    target_dtype_by_layer = {
        component.layer_index: component.dtype
        for component in pool.layout.components
        if component.role == "linear_convolution"
    }
    for seq in seqs:
        is_decode_row = (
            not is_prefill
            or (
                batch_kind == "mixed"
                and getattr(seq, "step_is_decode", False)
            )
        )
        lease = lease_by_request.get(int(seq.seq_id))
        if lease is None:
            continue
        if not is_decode_row:
            chunk_start = getattr(
                seq,
                "prefill_chunk_start",
                getattr(seq, "num_cached_tokens", 0),
            )
            chunk_end = getattr(seq, "prefill_chunk_end", None)
            try:
                token_count = len(seq)
            except TypeError:
                token_count = len(seq.token_ids)
            if chunk_end is None or (
                chunk_end == 0 and chunk_start == 0
            ):
                chunk_end = token_count
            if chunk_end < token_count:
                continue
        slot_id = pool.validate(lease)
        for component in pool.layout.components:
            if component.role != "linear_recurrent":
                continue
            recurrent = pool.component_tensor(
                component.layer_index,
                "linear_recurrent",
            )
            target_dtype = target_dtype_by_layer[
                component.layer_index
            ]
            recurrent[slot_id].copy_(
                recurrent[slot_id]
                .to(target_dtype)
                .to(recurrent.dtype)
            )


def _local_model_kv_heads(total_heads, tensor_parallel_size):
    if total_heads >= tensor_parallel_size:
        if total_heads % tensor_parallel_size != 0:
            raise ValueError(
                "num_key_value_heads must be divisible by "
                "tensor_parallel_size"
            )
        return total_heads // tensor_parallel_size
    if tensor_parallel_size % total_heads != 0:
        raise ValueError(
            "KV head replication requires tensor_parallel_size "
            "to be divisible by num_key_value_heads"
        )
    return 1


def _model_runner_shared_memory_name(dist_port):
    return f"tinyvllm-{int(dist_port)}"


class _ExactGraphCaptureError(RuntimeError):
    def __init__(
        self,
        reason: str,
        message: str,
        *,
        retained_reserved_bytes: int = 0,
    ):
        super().__init__(message)
        self.reason = reason
        self.retained_reserved_bytes = retained_reserved_bytes


def _resolve_kv_cache_blocks(
    requested_blocks: int,
    auto_blocks: int,
) -> int:
    if requested_blocks == -1:
        return auto_blocks
    if requested_blocks > auto_blocks:
        raise ValueError(
            "explicit num_kvcache_blocks exceeds available KV cache "
            f"capacity: requested={requested_blocks}, "
            f"available={auto_blocks}"
        )
    return requested_blocks


def resolve_exact_graph_kv_capacity(
    *,
    auto_blocks: int,
    requested_visible_blocks: int,
    feature_enabled: bool,
    scratch_blocks: int,
) -> tuple[int, int]:
    if auto_blocks <= 0:
        raise ValueError("auto_blocks must be positive")
    if not feature_enabled:
        visible_blocks = _resolve_kv_cache_blocks(
            requested_visible_blocks,
            auto_blocks,
        )
        return visible_blocks, visible_blocks
    if scratch_blocks <= 0:
        raise ValueError("scratch_blocks must be positive")
    if requested_visible_blocks == -1:
        visible_blocks = auto_blocks - scratch_blocks
        if visible_blocks <= 0:
            raise ValueError(
                "exact graph scratch blocks exhaust KV cache capacity"
            )
        return visible_blocks, auto_blocks
    physical_blocks = requested_visible_blocks + scratch_blocks
    if physical_blocks > auto_blocks:
        raise ValueError(
            "explicit scheduler-visible KV cache plus exact graph scratch "
            "exceeds available capacity: "
            f"visible={requested_visible_blocks}, "
            f"scratch={scratch_blocks}, available={auto_blocks}"
        )
    return requested_visible_blocks, physical_blocks


def partition_exact_graph_scratch_block_ids(
    *,
    visible_blocks: int,
    decode_scratch_blocks: int,
    spec_verify_scratch_blocks: int,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    for name, value in (
        ("visible_blocks", visible_blocks),
        ("decode_scratch_blocks", decode_scratch_blocks),
        ("spec_verify_scratch_blocks", spec_verify_scratch_blocks),
    ):
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or value < 0
        ):
            raise ValueError(
                f"{name} must be a non-negative integer"
            )
    decode_end = visible_blocks + decode_scratch_blocks
    spec_verify_end = decode_end + spec_verify_scratch_blocks
    return (
        tuple(range(visible_blocks, decode_end)),
        tuple(range(decode_end, spec_verify_end)),
    )


class KVOffloadMVP0:
    """Minimal logical-block -> GPU-slot KV offload prototype.

    Scope is intentionally narrow: fp16/bf16 KV cache, full attention, eager
    execution. ``Sequence.block_table`` stays logical; this manager owns the
    logical->physical slot mapping and a CPU pinned backing store.
    """

    def __init__(
        self,
        kv_cache: torch.Tensor,
        logical_blocks: int,
        block_size: int,
        async_copy: bool = True,
        batch_copy: bool = True,
        writeback_on_evict: bool = False,
        evict_policy: str = "lru_cost",
    ):
        self.kv_cache = kv_cache
        self.logical_blocks = int(logical_blocks)
        self.gpu_blocks = int(kv_cache.size(2))
        self.block_size = int(block_size)
        self.async_copy = bool(async_copy)
        self.batch_copy = bool(batch_copy)
        self.writeback_on_evict = bool(writeback_on_evict)
        self.evict_policy = evict_policy
        self.cpu_cache = torch.empty(
            (kv_cache.size(0), kv_cache.size(1), self.logical_blocks, *kv_cache.shape[3:]),
            dtype=kv_cache.dtype,
            device="cpu",
            pin_memory=True,
        )
        self.copy_stream = torch.cuda.Stream(device=kv_cache.device) if self.async_copy else None
        self.h2d_done: dict[int, torch.cuda.Event] = {}
        self.d2h_done: dict[int, torch.cuda.Event] = {}
        self.pending_wait_blocks: set[int] = set()
        self.cpu_valid = [False] * self.logical_blocks
        self.bound_generations: list[int | None] = [
            None
        ] * self.logical_blocks
        self.logical_to_slot: dict[int, int] = {}
        self.slot_to_logical: list[int | None] = [None] * self.gpu_blocks
        self.slot_last_used = [0] * self.gpu_blocks
        self.dirty_logical_blocks: set[int] = set()
        self.clock = 0
        self.stats = {
            "h2d_copies": 0,
            "d2h_copies": 0,
            "evictions": 0,
            "h2d_ms": 0.0,
            "d2h_ms": 0.0,
            "h2d_bytes": 0,
            "d2h_bytes": 0,
            "copy_waits": 0,
            "h2d_batches": 0,
            "d2h_batches": 0,
            "h2d_batch_spans": 0,
            "d2h_batch_spans": 0,
            "evict_clean": 0,
            "evict_dirty": 0,
            "prefetch_plans": 0,
            "prefetch_read_blocks": 0,
            "prefetch_write_blocks": 0,
            "decode_plan_builds": 0,
            "decode_plan_cache_hits": 0,
            "decode_plan_identity_invalidations": 0,
            "decode_windows_with_spare_capacity": 0,
            "decode_cross_layer_hint_blocks": 0,
            "decode_cross_layer_hint_resident": 0,
            "decode_cross_layer_hint_retained": 0,
            "speculative_residency_prepares": 0,
            "speculative_residency_precommits": 0,
            "speculative_residency_seals": 0,
            "speculative_residency_rollbacks": 0,
            "speculative_residency_committed_blocks": 0,
            "speculative_residency_rejected_blocks": 0,
            "speculative_residency_rejected_d2h_copies": 0,
            "peak_resident_blocks": 0,
        }
        self.block_nbytes = self.kv_cache[:, :, 0].numel() * self.kv_cache.element_size()
        self._initialize_h2d_slot_reuse_diagnostic(
            event_factory=lambda: torch.cuda.Event(
                enable_timing=True
            ),
            stream_id=lambda stream: int(stream.cuda_stream),
        )

    def _initialize_h2d_slot_reuse_diagnostic(
        self,
        *,
        event_factory,
        stream_id,
    ) -> None:
        rank = getattr(self, "rank", None)
        if rank is None:
            rank = (
                torch.distributed.get_rank()
                if torch.distributed.is_available()
                and torch.distributed.is_initialized()
                else 0
            )
        self._h2d_slot_reuse_diagnostic = H2DSlotReuseDiagnostic(
            rank=int(rank),
            slot_count=int(self.gpu_blocks),
            event_factory=event_factory,
            stream_id=stream_id,
        )
        self._h2d_slot_reuse_context = None
        self._h2d_slot_reuse_copy_batch_ordinal = 0
        self._h2d_slot_reuse_h2d_pair_inventory = []
        self._h2d_slot_reuse_h2d_span_inventory = []
        self._h2d_slot_reuse_d2h_pair_inventory = []
        self._h2d_slot_reuse_d2h_span_inventory = []

    def configure_h2d_slot_reuse_diagnostic(
        self,
        mode: str,
    ) -> dict:
        receipt = self._h2d_slot_reuse_diagnostic.configure(mode)
        self._h2d_slot_reuse_h2d_pair_inventory = []
        self._h2d_slot_reuse_h2d_span_inventory = []
        self._h2d_slot_reuse_d2h_pair_inventory = []
        self._h2d_slot_reuse_d2h_span_inventory = []
        if mode == "off":
            self._h2d_slot_reuse_context = None
            self._h2d_slot_reuse_copy_batch_ordinal = 0
        if mode != "off":
            self._h2d_slot_reuse_copy_batch_ordinal = 0
            for physical_slot, logical_block in enumerate(
                self.slot_to_logical
            ):
                if logical_block is None:
                    continue
                self._diagnostic_assign_slot(
                    physical_slot=physical_slot,
                    logical_block=int(logical_block),
                )
            self._assert_h2d_slot_reuse_diagnostic_mapping()
        return receipt

    def set_h2d_slot_reuse_context(
        self,
        *,
        engine_step: int,
        attention_stage: str,
        layer_index: int,
        window_ordinal: int,
    ) -> dict:
        receipt = self._h2d_slot_reuse_diagnostic.set_context(
            engine_step=engine_step,
            attention_stage=attention_stage,
            layer_index=layer_index,
            window_ordinal=window_ordinal,
        )
        self._h2d_slot_reuse_context = receipt
        return receipt

    def record_h2d_slot_read_window(
        self,
        *,
        engine_step,
        attention_stage: str,
        layer_index: int,
        window_ordinal: int,
        logical_blocks,
        physical_slots,
        current_stream=None,
    ) -> None:
        diagnostic = self._h2d_slot_reuse_diagnostic
        if not diagnostic.enabled:
            return
        context = self._h2d_slot_reuse_context
        if engine_step is None:
            if context is None:
                raise RuntimeError(
                    "H2D slot-reuse engine-step context is not set"
                )
            engine_step = context["engine_step"]
        diagnostic.set_context(
            engine_step=engine_step,
            attention_stage=attention_stage,
            layer_index=layer_index,
            window_ordinal=window_ordinal,
        )
        if current_stream is None:
            current_stream = torch.cuda.current_stream()
        diagnostic.record_read_window(
            engine_step=engine_step,
            attention_stage=attention_stage,
            layer_index=layer_index,
            window_ordinal=window_ordinal,
            logical_blocks=logical_blocks,
            physical_slots=physical_slots,
            bound_generations=tuple(
                self.bound_generations[int(block)]
                for block in logical_blocks
            ),
            current_stream=current_stream,
        )

    def drain_h2d_slot_reuse_diagnostic(
        self,
        *,
        timing_epsilon_ms: float,
    ) -> dict:
        return self._h2d_slot_reuse_diagnostic.drain(
            synchronize=self.synchronize_copies,
            timing_epsilon_ms=timing_epsilon_ms,
        ).as_dict()

    def h2d_slot_reuse_diagnostic_summary(self) -> dict:
        diagnostic = self._h2d_slot_reuse_diagnostic
        return {
            "rank": diagnostic.rank,
            "mode": diagnostic.mode,
            "retained_event_count": diagnostic.retained_event_count,
            "read_row_count": diagnostic.read_row_count,
            "overwrite_row_count": diagnostic.overwrite_row_count,
        }

    def _diagnostic_assign_slot(
        self,
        *,
        physical_slot: int,
        logical_block: int,
    ) -> None:
        diagnostic = self._h2d_slot_reuse_diagnostic
        if not diagnostic.enabled:
            return
        generation = self.bound_generations[logical_block]
        if generation is None:
            raise RuntimeError(
                "diagnostic assignment requires bound generation"
            )
        diagnostic.assign_slot(
            physical_slot=physical_slot,
            logical_block=logical_block,
            bound_generation=generation,
        )

    def _diagnostic_release_slot(
        self,
        *,
        physical_slot: int,
        logical_block: int,
    ) -> None:
        if self._h2d_slot_reuse_diagnostic.enabled:
            self._h2d_slot_reuse_diagnostic.release_slot(
                physical_slot=physical_slot,
                logical_block=logical_block,
            )

    def _assert_h2d_slot_reuse_diagnostic_mapping(self) -> None:
        diagnostic = self._h2d_slot_reuse_diagnostic
        if not diagnostic.enabled:
            return
        expected = tuple(
            None
            if logical_block is None
            else (
                slot,
                int(logical_block),
                int(self.bound_generations[logical_block]),
            )
            for slot, logical_block in enumerate(
                self.slot_to_logical
            )
        )
        diagnostic.assert_mapping(expected)

    def _check_logical_block(self, logical_block: int):
        if logical_block < 0 or logical_block >= self.logical_blocks:
            raise RuntimeError(
                f"KV offload logical block id {logical_block} out of range [0, {self.logical_blocks})"
            )

    def _clear_logical_block_metadata(
        self,
        logical_block: int,
    ) -> None:
        slot = self.logical_to_slot.pop(logical_block, None)
        if slot is not None:
            if self.slot_to_logical[slot] != logical_block:
                raise RuntimeError(
                    "KV offload mapping is inconsistent"
                )
            self._diagnostic_release_slot(
                physical_slot=slot,
                logical_block=logical_block,
            )
            self.slot_to_logical[slot] = None
        self.cpu_valid[logical_block] = False
        self.dirty_logical_blocks.discard(logical_block)
        self.pending_wait_blocks.discard(logical_block)
        self.h2d_done.pop(logical_block, None)
        self.d2h_done.pop(logical_block, None)
        self._assert_h2d_slot_reuse_diagnostic_mapping()

    def bind_logical_block_identity(
        self,
        logical_block: int,
        generation: int,
    ) -> None:
        self._check_logical_block(logical_block)
        if (
            isinstance(generation, bool)
            or not isinstance(generation, int)
            or generation < 0
        ):
            raise ValueError(
                "KV offload generation must be non-negative"
            )
        current = self.bound_generations[logical_block]
        if current == generation:
            return
        if current is not None and generation < current:
            raise RuntimeError(
                "KV offload generation moved backwards"
            )
        has_existing_state = (
            logical_block in self.logical_to_slot
            or self.cpu_valid[logical_block]
            or logical_block in self.dirty_logical_blocks
            or logical_block in self.pending_wait_blocks
            or logical_block in self.h2d_done
            or logical_block in self.d2h_done
        )
        if current is None and has_existing_state:
            raise RuntimeError(
                "cannot bind an unowned KV offload block "
                "with existing state"
            )
        if current is not None:
            self._clear_logical_block_metadata(logical_block)
        self.bound_generations[logical_block] = generation

    def _discard_validated_resident_block(
        self,
        logical_block: int,
    ) -> None:
        self._clear_logical_block_metadata(logical_block)

    def discard_resident_blocks(
        self,
        block_identities: tuple[tuple[int, int], ...],
        *,
        allow_dirty: bool,
    ) -> tuple[tuple[int, int], ...]:
        if not isinstance(block_identities, tuple):
            raise ValueError(
                "block_identities must be a tuple"
            )
        if not isinstance(allow_dirty, bool):
            raise ValueError("allow_dirty must be a bool")
        normalized = []
        seen = set()
        for identity in block_identities:
            if (
                not isinstance(identity, tuple)
                or len(identity) != 2
            ):
                raise ValueError(
                    "block identity must be a block/generation tuple"
                )
            logical_block, generation = identity
            if (
                isinstance(logical_block, bool)
                or not isinstance(logical_block, int)
            ):
                raise ValueError(
                    "logical block id must be an integer"
                )
            self._check_logical_block(logical_block)
            if (
                isinstance(generation, bool)
                or not isinstance(generation, int)
                or generation < 0
            ):
                raise ValueError(
                    "block generation must be non-negative"
                )
            if logical_block in seen:
                raise ValueError(
                    "block identities must be unique"
                )
            seen.add(logical_block)
            if self.bound_generations[logical_block] != generation:
                raise RuntimeError(
                    "KV offload block generation mismatch"
                )
            if logical_block not in self.logical_to_slot:
                raise RuntimeError(
                    "KV offload block is not resident"
                )
            if (
                not allow_dirty
                and logical_block in self.dirty_logical_blocks
            ):
                raise RuntimeError(
                    "KV offload dirty block cannot be discarded"
                )
            normalized.append((logical_block, generation))
        snapshot = (
            dict(self.logical_to_slot),
            list(self.slot_to_logical),
            list(self.cpu_valid),
            set(self.dirty_logical_blocks),
            set(self.pending_wait_blocks),
            dict(self.h2d_done),
            dict(self.d2h_done),
        )
        diagnostic_snapshot = (
            self._h2d_slot_reuse_diagnostic.snapshot_state()
            if self._h2d_slot_reuse_diagnostic.enabled
            else None
        )
        try:
            for logical_block, _ in normalized:
                self._discard_validated_resident_block(
                    logical_block
                )
        except BaseException:
            (
                logical_to_slot,
                slot_to_logical,
                cpu_valid,
                dirty_logical_blocks,
                pending_wait_blocks,
                h2d_done,
                d2h_done,
            ) = snapshot
            self.logical_to_slot = logical_to_slot
            self.slot_to_logical = slot_to_logical
            self.cpu_valid = cpu_valid
            self.dirty_logical_blocks = dirty_logical_blocks
            self.pending_wait_blocks = pending_wait_blocks
            self.h2d_done = h2d_done
            self.d2h_done = d2h_done
            if diagnostic_snapshot is not None:
                self._h2d_slot_reuse_diagnostic.restore_state(
                    diagnostic_snapshot
                )
            raise
        return tuple(normalized)

    def evict_clean_resident_blocks(
        self,
        block_identities: tuple[tuple[int, int], ...],
    ) -> tuple[tuple[int, int], ...]:
        if not isinstance(block_identities, tuple):
            raise ValueError(
                "block_identities must be a tuple"
            )
        normalized = []
        seen = set()
        slots = []
        for identity in block_identities:
            if (
                not isinstance(identity, tuple)
                or len(identity) != 2
            ):
                raise ValueError(
                    "block identity must be a block/generation tuple"
                )
            logical_block, generation = identity
            if (
                isinstance(logical_block, bool)
                or not isinstance(logical_block, int)
            ):
                raise ValueError(
                    "logical block id must be an integer"
                )
            self._check_logical_block(logical_block)
            if (
                isinstance(generation, bool)
                or not isinstance(generation, int)
                or generation < 0
            ):
                raise ValueError(
                    "block generation must be non-negative"
                )
            if logical_block in seen:
                raise ValueError(
                    "block identities must be unique"
                )
            seen.add(logical_block)
            if self.bound_generations[logical_block] != generation:
                raise RuntimeError(
                    "KV offload block generation mismatch"
                )
            slot = self.logical_to_slot.get(logical_block)
            if slot is None:
                raise RuntimeError(
                    "KV offload block is not resident"
                )
            if self.slot_to_logical[slot] != logical_block:
                raise RuntimeError(
                    "KV offload mapping is inconsistent"
                )
            if logical_block in self.dirty_logical_blocks:
                raise RuntimeError(
                    "KV offload block must be clean before eviction"
                )
            if not self.cpu_valid[logical_block]:
                raise RuntimeError(
                    "KV offload block must have CPU-valid backing"
                )
            if logical_block in self.pending_wait_blocks:
                raise RuntimeError(
                    "KV offload block has pending H2D"
                )
            normalized.append((logical_block, generation))
            slots.append((logical_block, slot))

        for logical_block, slot in slots:
            self._diagnostic_release_slot(
                physical_slot=slot,
                logical_block=logical_block,
            )
            self.logical_to_slot.pop(logical_block)
            self.slot_to_logical[slot] = None
            self.h2d_done.pop(logical_block, None)
            self.d2h_done.pop(logical_block, None)
        self.stats["evictions"] += len(normalized)
        self.stats["evict_clean"] += len(normalized)
        self._assert_h2d_slot_reuse_diagnostic_mapping()
        return tuple(normalized)

    def _touch(self, slot: int):
        self.clock += 1
        self.slot_last_used[slot] = self.clock

    def _coalesce_copy_pairs(self, pairs: list[tuple[int, int]]) -> list[tuple[int, int, int]]:
        if not pairs:
            return []
        if not self.batch_copy:
            return [(int(logical_block), int(slot), 1) for logical_block, slot in pairs]
        ordered = sorted((int(logical_block), int(slot)) for logical_block, slot in pairs)
        spans = []
        start_logical, start_slot = ordered[0]
        prev_logical, prev_slot = ordered[0]
        span_len = 1
        for logical_block, slot in ordered[1:]:
            if logical_block == prev_logical + 1 and slot == prev_slot + 1:
                prev_logical = logical_block
                prev_slot = slot
                span_len += 1
                continue
            spans.append((start_logical, start_slot, span_len))
            start_logical = prev_logical = logical_block
            start_slot = prev_slot = slot
            span_len = 1
        spans.append((start_logical, start_slot, span_len))
        return spans

    def _record_copy_event(self) -> torch.cuda.Event | None:
        if self.copy_stream is None:
            return None
        event = torch.cuda.Event()
        event.record(self.copy_stream)
        return event

    def _enqueue_h2d_pairs(self, pairs: list[tuple[int, int]]):
        pairs = [(int(logical_block), int(slot)) for logical_block, slot in pairs if self.cpu_valid[int(logical_block)]]
        if not pairs:
            return
        diagnostic = self._h2d_slot_reuse_diagnostic
        if diagnostic.enabled and self.copy_stream is None:
            raise RuntimeError(
                "H2D slot-reuse diagnostic requires "
                "asynchronous copy stream"
            )
        spans = self._coalesce_copy_pairs(pairs)
        if diagnostic.enabled:
            self._h2d_slot_reuse_h2d_pair_inventory.extend(
                pairs
            )
            self._h2d_slot_reuse_h2d_span_inventory.extend(
                spans
            )
        t0 = time.perf_counter()
        if self.copy_stream is None:
            for logical_start, slot_start, span_len in spans:
                self.kv_cache[:, :, slot_start:slot_start + span_len].copy_(
                    self.cpu_cache[:, :, logical_start:logical_start + span_len],
                    non_blocking=True,
                )
            torch.cuda.synchronize()
            event = None
        else:
            copy_batch_ordinal = (
                self._h2d_slot_reuse_copy_batch_ordinal
            )
            if diagnostic.enabled:
                self._h2d_slot_reuse_copy_batch_ordinal += 1
            with torch.cuda.stream(self.copy_stream):
                for copy_span_ordinal, (
                    logical_start,
                    slot_start,
                    span_len,
                ) in enumerate(spans):
                    for logical_block in range(logical_start, logical_start + span_len):
                        d2h_event = self.d2h_done.get(logical_block)
                        if d2h_event is not None:
                            self.copy_stream.wait_event(d2h_event)
                    span_pairs = tuple(
                        (
                            logical_start + offset,
                            slot_start + offset,
                        )
                        for offset in range(span_len)
                    )
                    diagnostic_handle = (
                        diagnostic.begin_h2d_span(
                            copy_batch_ordinal=(
                                copy_batch_ordinal
                            ),
                            copy_span_ordinal=(
                                copy_span_ordinal
                            ),
                            pairs=span_pairs,
                            copy_stream=self.copy_stream,
                        )
                        if diagnostic.enabled
                        else None
                    )
                    self.kv_cache[:, :, slot_start:slot_start + span_len].copy_(
                        self.cpu_cache[:, :, logical_start:logical_start + span_len],
                        non_blocking=True,
                    )
                    if diagnostic_handle is not None:
                        diagnostic.end_h2d_span(
                            diagnostic_handle,
                            copy_stream=self.copy_stream,
                        )
            event = self._record_copy_event()
            self.pending_wait_blocks.update(logical_block for logical_block, _ in pairs)
        for logical_block, _ in pairs:
            if event is not None:
                self.h2d_done[logical_block] = event
        self.stats["h2d_copies"] += len(pairs)
        self.stats["h2d_bytes"] += len(pairs) * self.block_nbytes
        self.stats["h2d_batches"] += 1
        self.stats["h2d_batch_spans"] += len(spans)
        self.stats["h2d_ms"] += (time.perf_counter() - t0) * 1000.0

    def _enqueue_d2h_pairs(self, pairs: list[tuple[int, int]]):
        pairs = [(int(logical_block), int(slot)) for logical_block, slot in pairs]
        if not pairs:
            return
        spans = self._coalesce_copy_pairs(pairs)
        if self._h2d_slot_reuse_diagnostic.enabled:
            self._h2d_slot_reuse_d2h_pair_inventory.extend(
                pairs
            )
            self._h2d_slot_reuse_d2h_span_inventory.extend(
                spans
            )
        t0 = time.perf_counter()
        if self.copy_stream is None:
            for logical_start, slot_start, span_len in spans:
                self.cpu_cache[:, :, logical_start:logical_start + span_len].copy_(
                    self.kv_cache[:, :, slot_start:slot_start + span_len],
                    non_blocking=True,
                )
            torch.cuda.synchronize()
            event = None
        else:
            self.copy_stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(self.copy_stream):
                for logical_start, slot_start, span_len in spans:
                    self.cpu_cache[:, :, logical_start:logical_start + span_len].copy_(
                        self.kv_cache[:, :, slot_start:slot_start + span_len],
                        non_blocking=True,
                    )
            event = self._record_copy_event()
        for logical_block, _ in pairs:
            self.cpu_valid[logical_block] = True
            if event is not None:
                self.d2h_done[logical_block] = event
        self.stats["d2h_copies"] += len(pairs)
        self.stats["d2h_bytes"] += len(pairs) * self.block_nbytes
        self.stats["d2h_batches"] += 1
        self.stats["d2h_batch_spans"] += len(spans)
        self.stats["d2h_ms"] += (time.perf_counter() - t0) * 1000.0

    def _copy_h2d(self, logical_block: int, slot: int):
        self._enqueue_h2d_pairs([(logical_block, slot)])

    def _copy_d2h(self, logical_block: int, slot: int):
        self._enqueue_d2h_pairs([(logical_block, slot)])

    def _victim_score(self, slot: int, future_logical_blocks: set[int]) -> float:
        logical_block = self.slot_to_logical[slot]
        if logical_block is None:
            return float("-inf")
        if self.evict_policy == "lru":
            return float(self.slot_last_used[slot])
        dirty_penalty = self.block_nbytes * 4.0 if logical_block in self.dirty_logical_blocks else 0.0
        reuse_penalty = self.block_nbytes * 8.0 if logical_block in future_logical_blocks else 0.0
        pending_h2d_penalty = self.block_nbytes * 6.0 if logical_block in self.pending_wait_blocks else 0.0
        return float(self.slot_last_used[slot]) + dirty_penalty + reuse_penalty + pending_h2d_penalty

    def _evict_slot(
        self,
        protected_logical_blocks: set[int],
        future_logical_blocks: set[int] | None = None,
        defer_dirty_writeback: bool = False,
    ) -> int:
        future_logical_blocks = future_logical_blocks or set()
        for slot, logical_block in enumerate(self.slot_to_logical):
            if logical_block is None:
                return slot

        candidates = [
            slot for slot, logical_block in enumerate(self.slot_to_logical)
            if logical_block not in protected_logical_blocks
        ]
        if not candidates:
            raise RuntimeError(
                "KV offload GPU staging slots are insufficient for one full-attention batch: "
                f"gpu_blocks={self.gpu_blocks}, required={len(protected_logical_blocks)}"
            )
        slot = min(candidates, key=lambda item: self._victim_score(item, future_logical_blocks))
        old_logical = self.slot_to_logical[slot]
        if old_logical is not None:
            if old_logical in self.dirty_logical_blocks:
                if not defer_dirty_writeback:
                    self._enqueue_d2h_pairs([(old_logical, slot)])
                    self.dirty_logical_blocks.discard(old_logical)
                self.stats["evict_dirty"] += 1
            else:
                self.stats["evict_clean"] += 1
            if not defer_dirty_writeback:
                d2h_event = self.d2h_done.get(old_logical)
                if d2h_event is not None:
                    torch.cuda.current_stream().wait_event(d2h_event)
                    if self.copy_stream is not None:
                        self.copy_stream.wait_event(d2h_event)
                    self.stats["copy_waits"] += 1
            if not defer_dirty_writeback:
                self._diagnostic_release_slot(
                    physical_slot=slot,
                    logical_block=old_logical,
                )
                self.logical_to_slot.pop(old_logical, None)
                self.slot_to_logical[slot] = None
            self.stats["evictions"] += 1
        return slot

    def ensure_resident(
        self,
        logical_blocks: list[int],
        require_valid: bool,
        future_logical_blocks: set[int] | None = None,
        protected_logical_blocks: set[int] | None = None,
        wait: bool = False,
    ) -> dict[int, int]:
        ordered = []
        seen = set()
        for block in logical_blocks:
            block = int(block)
            if block < 0 or block in seen:
                continue
            self._check_logical_block(block)
            ordered.append(block)
            seen.add(block)
        if not ordered:
            return {}
        if len(ordered) > self.gpu_blocks:
            raise RuntimeError(
                "KV offload MVP-0 uses full attention, so all visible logical blocks must fit in GPU staging: "
                f"required={len(ordered)}, gpu_blocks={self.gpu_blocks}"
            )

        protected = set(ordered)
        if protected_logical_blocks:
            protected.update(int(block) for block in protected_logical_blocks if int(block) >= 0)
        h2d_pairs = []
        deferred_d2h_pairs = []
        deferred_wait_blocks = []
        missing_blocks = []
        for logical_block in ordered:
            if self.logical_to_slot.get(logical_block) is None:
                if require_valid and not self.cpu_valid[logical_block]:
                    raise RuntimeError(f"KV offload requested unreadable logical block {logical_block}")
                missing_blocks.append(logical_block)

        assigned_missing_slots = {}
        for logical_block in missing_blocks:
            slot = self._evict_slot(
                protected,
                future_logical_blocks=future_logical_blocks,
                defer_dirty_writeback=True,
            )
            old_logical = self.slot_to_logical[slot]
            if old_logical is not None and old_logical in self.dirty_logical_blocks:
                deferred_d2h_pairs.append((old_logical, slot))
                self.dirty_logical_blocks.discard(old_logical)
            if old_logical is not None:
                deferred_wait_blocks.append(old_logical)
                self.logical_to_slot.pop(old_logical, None)
                self.slot_to_logical[slot] = None
            self.logical_to_slot[logical_block] = slot
            self.slot_to_logical[slot] = logical_block
            assigned_missing_slots[logical_block] = slot

        if len(assigned_missing_slots) > 1:
            sorted_logical_blocks = sorted(assigned_missing_slots)
            sorted_slots = sorted(assigned_missing_slots.values())
            if any(
                assigned_missing_slots[logical_block] != slot
                for logical_block, slot in zip(sorted_logical_blocks, sorted_slots)
            ):
                for logical_block in sorted_logical_blocks:
                    slot = self.logical_to_slot.pop(logical_block)
                    self.slot_to_logical[slot] = None
                for logical_block, slot in zip(sorted_logical_blocks, sorted_slots):
                    self.logical_to_slot[logical_block] = slot
                    self.slot_to_logical[slot] = logical_block

        for logical_block in sorted(assigned_missing_slots):
            self._diagnostic_assign_slot(
                physical_slot=self.logical_to_slot[logical_block],
                logical_block=logical_block,
            )
        self._assert_h2d_slot_reuse_diagnostic_mapping()

        self._record_peak_resident_blocks()
        for logical_block in ordered:
            slot = self.logical_to_slot[logical_block]
            if logical_block in assigned_missing_slots and self.cpu_valid[logical_block]:
                h2d_pairs.append((logical_block, slot))
            self._touch(slot)
        if not h2d_pairs and not deferred_d2h_pairs and not deferred_wait_blocks:
            return {logical_block: self.logical_to_slot[logical_block] for logical_block in ordered}
        if deferred_d2h_pairs:
            self._enqueue_d2h_pairs(deferred_d2h_pairs)
        waited_d2h_event_ids = set()
        for old_logical in deferred_wait_blocks:
            d2h_event = self.d2h_done.get(old_logical)
            if d2h_event is not None and id(d2h_event) not in waited_d2h_event_ids:
                torch.cuda.current_stream().wait_event(d2h_event)
                if self.copy_stream is not None:
                    self.copy_stream.wait_event(d2h_event)
                waited_d2h_event_ids.add(id(d2h_event))
                self.stats["copy_waits"] += 1
        if h2d_pairs:
            self._enqueue_h2d_pairs(h2d_pairs)
        if wait and h2d_pairs:
            waited_h2d_blocks = [logical_block for logical_block, _ in h2d_pairs]
            self.wait_for_blocks(waited_h2d_blocks)
            self.pending_wait_blocks.difference_update(int(block) for block in waited_h2d_blocks)
        return {logical_block: self.logical_to_slot[logical_block] for logical_block in ordered}

    def wait_for_blocks(self, logical_blocks: list[int], clear_pending: bool = False):
        if self.copy_stream is None:
            return
        blocks = set(int(block) for block in logical_blocks)
        if clear_pending:
            blocks &= self.pending_wait_blocks
        if not blocks:
            return
        event_blocks = [logical_block for logical_block in blocks if self.h2d_done.get(logical_block) is not None]
        if not event_blocks:
            if clear_pending:
                self.pending_wait_blocks.difference_update(blocks)
            return
        stream = torch.cuda.current_stream()
        waited_event_ids = set()
        for logical_block in event_blocks:
            event = self.h2d_done.get(logical_block)
            if event is not None and id(event) not in waited_event_ids:
                stream.wait_event(event)
                waited_event_ids.add(id(event))
                self.stats["copy_waits"] += 1
        if clear_pending:
            self.pending_wait_blocks.difference_update(blocks)
            self.pending_wait_blocks.difference_update(
                block for block in list(self.pending_wait_blocks)
                if self.h2d_done.get(block) is not None and id(self.h2d_done[block]) in waited_event_ids
            )

    def wait_for_pending(self):
        if not self.pending_wait_blocks:
            return
        self.wait_for_blocks(list(self.pending_wait_blocks), clear_pending=True)
        self.pending_wait_blocks.clear()

    def synchronize_copies(self):
        if self.copy_stream is not None:
            self.copy_stream.synchronize()

    def translate_block_rows(
        self,
        rows: list[list[int]],
        require_valid: bool = True,
        future_logical_blocks: set[int] | None = None,
    ) -> list[list[int]]:
        logical_blocks = [int(block) for row in rows for block in row if int(block) >= 0]
        mapping = self.ensure_resident(
            logical_blocks,
            require_valid=require_valid,
            future_logical_blocks=future_logical_blocks,
        )
        return self.map_block_rows(rows, mapping=mapping)

    def map_block_rows(
        self,
        rows: list[list[int]],
        mapping: dict[int, int] | None = None,
    ) -> list[list[int]]:
        mapping = self.logical_to_slot if mapping is None else mapping
        return [[mapping[int(block)] if int(block) >= 0 else -1 for block in row] for row in rows]

    def translate_slots_for_positions(
        self,
        logical_block_table: list[int],
        positions: list[int],
        require_valid: bool = False,
        future_logical_blocks: set[int] | None = None,
    ) -> list[int]:
        logical_blocks = [int(logical_block_table[pos // self.block_size]) for pos in positions]
        mapping = self.ensure_resident(
            logical_blocks,
            require_valid=require_valid,
            future_logical_blocks=future_logical_blocks,
        )
        return self.map_slots_for_positions(
            logical_block_table,
            positions,
            mapping=mapping,
        )

    def map_slots_for_positions(
        self,
        logical_block_table: list[int],
        positions: list[int],
        mapping: dict[int, int] | None = None,
    ) -> list[int]:
        mapping = self.logical_to_slot if mapping is None else mapping
        return [
            mapping[int(logical_block_table[pos // self.block_size])] * self.block_size + (pos % self.block_size)
            for pos in positions
        ]

    def mark_dirty(self, logical_blocks: list[int]):
        for logical_block in logical_blocks:
            logical_block = int(logical_block)
            if logical_block in self.logical_to_slot:
                self.dirty_logical_blocks.add(logical_block)

    def writeback_dirty(self, logical_blocks: list[int] | None = None):
        targets = set(self.dirty_logical_blocks if logical_blocks is None else logical_blocks)
        d2h_pairs = []
        for logical_block in list(targets):
            slot = self.logical_to_slot.get(int(logical_block))
            if slot is None:
                continue
            d2h_pairs.append((int(logical_block), slot))
            self.dirty_logical_blocks.discard(int(logical_block))
        self._enqueue_d2h_pairs(d2h_pairs)

    def speculative_residency_summary(self) -> dict:
        keys = (
            "speculative_residency_prepares",
            "speculative_residency_precommits",
            "speculative_residency_seals",
            "speculative_residency_rollbacks",
            "speculative_residency_committed_blocks",
            "speculative_residency_rejected_blocks",
            "speculative_residency_rejected_d2h_copies",
        )
        return {
            key: int(self.stats.get(key, 0))
            for key in keys
        }

    def summary(self) -> dict:
        return {
            **self.stats,
            "logical_blocks": self.logical_blocks,
            "gpu_blocks": self.gpu_blocks,
            "resident_blocks": len(self.logical_to_slot),
            "dirty_blocks": len(self.dirty_logical_blocks),
            "block_nbytes": self.block_nbytes,
            "async_copy": self.async_copy,
            "batch_copy": self.batch_copy,
            "writeback_on_evict": self.writeback_on_evict,
            "evict_policy": self.evict_policy,
            "h2d_pair_inventory": [
                list(pair)
                for pair in (
                    self._h2d_slot_reuse_h2d_pair_inventory
                )
            ],
            "h2d_span_inventory": [
                list(span)
                for span in (
                    self._h2d_slot_reuse_h2d_span_inventory
                )
            ],
            "d2h_pair_inventory": [
                list(pair)
                for pair in (
                    self._h2d_slot_reuse_d2h_pair_inventory
                )
            ],
            "d2h_span_inventory": [
                list(span)
                for span in (
                    self._h2d_slot_reuse_d2h_span_inventory
                )
            ],
        }

    def _record_peak_resident_blocks(self) -> None:
        resident_blocks = len(self.logical_to_slot)
        if resident_blocks > self.gpu_blocks:
            raise RuntimeError(
                "KV offload resident block count exceeds GPU capacity"
            )
        self.stats["peak_resident_blocks"] = max(
            int(self.stats.get("peak_resident_blocks", 0)),
            resident_blocks,
        )

def _unique_ints_in_order(values) -> list[int]:
    ordered = []
    seen = set()
    for value in values:
        value = int(value)
        if value in seen:
            continue
        ordered.append(value)
        seen.add(value)
    return ordered


def _resolve_blockwise_prefill_window_blocks(
    requested_window_blocks: int,
    *,
    gpu_blocks: int,
    write_blocks,
) -> int:
    unique_write_blocks = set(int(block) for block in write_blocks)
    available_read_blocks = int(gpu_blocks) - len(unique_write_blocks)
    if available_read_blocks <= 0:
        raise RuntimeError(
            "blockwise prefill has no GPU staging slot left after "
            f"reserving current write blocks: gpu_blocks={gpu_blocks}, "
            f"write_blocks={len(unique_write_blocks)}"
        )
    return min(
        max(1, int(requested_window_blocks)),
        available_read_blocks,
    )


def _stage_kv_offload_write_blocks(
    manager,
    write_blocks,
    first_write_offset_by_block: dict[int, int],
    future_blocks: set[int],
):
    write_blocks = _unique_ints_in_order(write_blocks)
    if not write_blocks:
        return
    protected_write_blocks = set(write_blocks)
    valid_write_blocks = [
        block_id for block_id in write_blocks
        if int(first_write_offset_by_block[block_id]) > 0
    ]
    fresh_write_blocks = [
        block_id for block_id in write_blocks
        if int(first_write_offset_by_block[block_id]) == 0
    ]
    manager.stats["prefetch_plans"] += 1
    manager.stats["prefetch_write_blocks"] += len(protected_write_blocks)
    if valid_write_blocks:
        manager.ensure_resident(
            valid_write_blocks,
            require_valid=True,
            future_logical_blocks=future_blocks,
            protected_logical_blocks=protected_write_blocks,
        )
    if fresh_write_blocks:
        manager.ensure_resident(
            fresh_write_blocks,
            require_valid=False,
            future_logical_blocks=future_blocks,
            protected_logical_blocks=protected_write_blocks,
        )


def _stage_kv_offload_write_positions(
    manager,
    block_table: list[int],
    positions: list[int],
    block_size: int,
    future_blocks: set[int],
) -> tuple[list[int], list[int]]:
    write_blocks = []
    first_write_offset_by_block = {}
    for pos in positions:
        block_id = int(block_table[pos // block_size])
        if block_id not in first_write_offset_by_block:
            write_blocks.append(block_id)
            first_write_offset_by_block[block_id] = pos % block_size
        else:
            first_write_offset_by_block[block_id] = min(
                first_write_offset_by_block[block_id], pos % block_size)
    _stage_kv_offload_write_blocks(
        manager,
        write_blocks,
        first_write_offset_by_block,
        future_blocks,
    )
    return manager.map_slots_for_positions(block_table, positions), write_blocks


def _stage_kv_offload_full_decode_blocks(
    manager,
    block_table_rows: list[list[int]],
    decode_write_blocks: list[int],
    decode_write_offsets: list[int],
) -> set[int]:
    future_blocks = set(int(block) for row in block_table_rows for block in row if int(block) >= 0)
    future_blocks.update(int(block) for block in decode_write_blocks)
    valid_read_blocks = []
    for row, write_block, write_offset in zip(block_table_rows, decode_write_blocks, decode_write_offsets):
        for block in row:
            block = int(block)
            if block < 0:
                continue
            if block == int(write_block) and int(write_offset) == 0:
                continue
            valid_read_blocks.append(block)
    manager.stats["prefetch_plans"] += 1
    manager.stats["prefetch_read_blocks"] += len(set(valid_read_blocks))
    manager.stats["prefetch_write_blocks"] += len(set(int(block) for block in decode_write_blocks))
    manager.ensure_resident(
        valid_read_blocks,
        require_valid=True,
        future_logical_blocks=future_blocks,
    )
    manager.ensure_resident(
        decode_write_blocks,
        require_valid=False,
        future_logical_blocks=future_blocks,
    )
    return future_blocks


def _profile_request_set_sha256(sequence_ids) -> str:
    return hashlib.sha256(
        json.dumps(
            sorted(int(value) for value in sequence_ids),
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


class ModelRunner:

    def __init__(
        self,
        config: Config,
        rank: int,
        event: Event | list[Event],
        ack_sender=None,
    ):
        self.config = config
        hf_config = config.hf_config
        self.block_size = config.kvcache_block_size
        self.enforce_eager = config.enforce_eager
        self.world_size  = config.tensor_parallel_size
        self.rank = rank
        self.event = event
        self.ack_sender = ack_sender
        self._command_ids = count()
        self._command_timeline_clock_ns = time.monotonic_ns
        self._command_timeline_max_rows = (
            config.autoregressive_draft_command_timeline_max_rows
        )
        self.command_timeline = (
            ModelRunnerCommandTimelineRecorder(
                rank=rank,
                max_rows=self._command_timeline_max_rows,
                clock_identity=read_command_clock_identity(),
            )
            if config.autoregressive_draft_command_timeline
            else ModelRunnerCommandTimelineRecorder.disabled(rank)
        )
        self._active_command_timeline_trace = (
            self._read_active_engine_step_trace
        )

        dist_port = os.environ.get("TINYVLLM_DIST_PORT", os.environ.get("MASTER_PORT", "2333"))
        self.shared_memory_name = _model_runner_shared_memory_name(
            dist_port
        )
        dist.init_process_group(
            backend="nccl",
            init_method=f"tcp://localhost:{dist_port}",       # 初始化建立连接的方法有 tcp, 共享文件系统，环境变量等
            world_size=self.world_size,
            rank=self.rank
        )
        torch.cuda.set_device(rank)
        torch.cuda.reset_peak_memory_stats()
        default_dtype = torch.get_default_dtype()
        model_dtype = _resolve_hf_model_dtype(hf_config, torch)
        torch.set_default_dtype(model_dtype)
        torch.set_default_device("cuda")
        # 注入全局量化配置（在构建模型前）
        set_quant_config(config.quantization, config.quant_group_size, config.act_quant_bits)

        def load_legacy_model(runner_config):
            model = Qwen3ForCausalLM(runner_config.hf_config)
            load_model(
                model,
                runner_config.model,
                smoothquant_scale_path=(
                    runner_config.smoothquant_scale_path
                ),
                act_quant_skip_first=(
                    runner_config.act_quant_skip_first
                ),
                act_quant_skip_last=(
                    runner_config.act_quant_skip_last
                ),
                act_quant_skip_layers=(
                    runner_config.act_quant_skip_layers
                ),
            )
            return model

        self.model, initial_qwen35_owner = (
            _initialize_model_runner_model(
                config,
                rank=rank,
                load_legacy_model=load_legacy_model,
                load_qwen35_model=_load_qwen35_model_runner_model,
            )
        )
        # 加载完成后再做 cpu-offload（量化已在 loader 内 finalize 完成）
        if config.cpu_offload:
            apply_cpu_offload(self.model, config.cpu_offload_num_layers)
        self.sampler =  Sampler()
        self._record_step_logits = False
        self._last_step_logits_cpu: torch.Tensor | None = None
        self._spec_verify_trace = SpecVerifyTraceRecorder(
            rank=rank,
            block_size=self.block_size,
        )
        self.decode_internal_profiler = (
            DecodeInternalProfiler.disabled(rank=rank)
        )
        self.hybrid_state_runtime_bridge = None
        self._last_hybrid_state_slot_ids = None
        self.qwen35_hybrid_model_owner = None
        self.qwen35_speculative_state_owner = None
        self._speculative_side_state_handle = None
        self._speculative_side_state_leases_by_sequence = {}
        self.qwen35_recurrent_capture_session = None
        self.qwen35_recurrent_capture_workload_id = None
        self.qwen35_recurrent_capture_armed = False
        self.qwen35_hybrid_prefix_runtime_identity = None
        self.qwen35_hybrid_prefix_runtime_identity_owner = None
        self.qwen35_hybrid_prefix_restore_participant = None
        self.qwen35_hybrid_prefix_publication_participant = None
        self.qwen35_hybrid_prefix_restore_owner = None
        self.qwen35_loaded_checkpoint_candidate_slot = (
            Qwen35HybridModelOwnerPublicationSlot()
        )
        self.qwen35_checkpoint_candidate_loader = None
        self.qwen35_checkpoint_candidate_loader_authorization_sha256 = (
            None
        )
        self.qwen35_checkpoint_candidate_load_configuration = None
        self.qwen35_checkpoint_candidate_load_request = None
        if initial_qwen35_owner is not None:
            self.bind_qwen35_hybrid_model_owner(
                initial_qwen35_owner
            )

        # prepare_prefill / prepare_decode 用的 pinned host buffer 池：按 (name, dtype) 复用，
        # 容量按需向上扩；避免每步 torch.tensor(list, pin_memory=True).cuda() 触发 host alloc + pin
        self._pinned_buf_cache: dict[tuple[str, torch.dtype], torch.Tensor] = {}
        self.kv_offload: KVOffloadMVP0 | None = None
        self._kv_offload_pending_dirty_blocks: list[int] = []
        self.speculative_proposal_executors = (
            ModelRunnerProposalExecutorRegistry()
        )
        self.qwen35_mtp_module = None
        self.qwen35_mtp_executor_descriptor = None
        self.qwen35_mtp_registration_error = None
        self.qwen35_mtp_physical_store = None
        self.qwen35_mtp_executor = None
        self.autoregressive_draft_model = None
        self.autoregressive_draft_physical_store = None
        self.autoregressive_draft_executor = None
        self.autoregressive_draft_executor_descriptor = None
        self.autoregressive_draft_registration_error = None
        self.autoregressive_draft_registration_consensus_sha256 = None
        self.autoregressive_draft_checkpoint_identity = None
        self.autoregressive_draft_tokenizer_contract = None
        self.autoregressive_draft_graph_components = None

        self.exact_cuda_graph_cache = ExactCudaGraphCache(
            ExactCudaGraphCacheConfig(
                enabled=config.multi_sequence_cuda_graphs,
                batch_allowlist=(
                    config.multi_sequence_cuda_graph_batch_allowlist
                ),
                min_observations=(
                    config.multi_sequence_cuda_graph_min_observations
                ),
                max_entries=config.multi_sequence_cuda_graph_max_entries,
                max_static_bytes=(
                    config.multi_sequence_cuda_graph_max_static_bytes
                ),
                max_reserved_bytes=(
                    config.multi_sequence_cuda_graph_max_reserved_bytes
                ),
                max_total_capture_ns=(
                    config.multi_sequence_cuda_graph_max_total_capture_ns
                ),
                max_single_capture_ns=(
                    config.multi_sequence_cuda_graph_max_single_capture_ns
                ),
            )
        )
        self.spec_verify_exact_cuda_graph_cache = (
            SpecVerifyExactCudaGraphCache(
                SpecVerifyExactCudaGraphCacheConfig(
                    enabled=config.spec_verify_cuda_graphs,
                    batch_allowlist=(
                        config.spec_verify_cuda_graph_batch_allowlist
                    ),
                    query_len_allowlist=(
                        config.spec_verify_cuda_graph_query_len_allowlist
                    ),
                    min_observations=(
                        config.spec_verify_cuda_graph_min_observations
                    ),
                    max_entries=(
                        config.spec_verify_cuda_graph_max_entries
                    ),
                    max_static_bytes=(
                        config.spec_verify_cuda_graph_max_static_bytes
                    ),
                    max_reserved_bytes=(
                        config.spec_verify_cuda_graph_max_reserved_bytes
                    ),
                    max_total_capture_ns=(
                        config.spec_verify_cuda_graph_max_total_capture_ns
                    ),
                    max_single_capture_ns=(
                        config.spec_verify_cuda_graph_max_single_capture_ns
                    ),
                )
            )
        )

        self.last_cuda_graph_dispatch_event = None
        self._cuda_graph_step_id = 0
        self._cuda_graph_request_ids_hash = hashlib.sha256(
            b"[]"
        ).hexdigest()
        self.last_spec_verify_cuda_graph_dispatch_event = None
        self._spec_verify_cuda_graph_step_id = 0
        self._spec_verify_cuda_graph_request_ids_hash = hashlib.sha256(
            b"[]"
        ).hexdigest()

        if initial_qwen35_owner is None:
            self.warmup_model()

        self.allocate_kv_cache()                        #预分配空间（没有具体值）
        self._maybe_register_qwen35_mtp_executor()
        self._maybe_register_autoregressive_draft_executor()
        self.speculative_residency = (
            SpeculativeResidencyParticipant(
                participant_id=self.rank,
                manager=self.kv_offload,
                block_size=self.block_size,
            )
            if self.kv_offload is not None
            else None
        )
        # cuda graph 跳过条件：
        #   1) enforce_eager：用户显式关
        #   2) kv_quant_bits == 4 (C4)：decode 反量化路径里有动态 alloc，无法 capture
        #   3) cpu_offload：layer 权重 H2D 走独立 stream + cross-stream sync，
        #      在 capture mode 下会报 "operation failed due to a previous error during capture"
        skip_cudagraph = (
            self.enforce_eager
            or config.kv_quant_bits == 4
            or config.am_compact_blocks > 0
            or config.cpu_offload
            or config.kv_offload_mvp0
            or initial_qwen35_owner is not None
        )
        if not skip_cudagraph:
            self.capture_cudagraph()
        torch.set_default_device("cpu")
        torch.set_default_dtype(default_dtype)


        if self.world_size > 1:
            if rank == 0:
                # 创建一个多卡通信的共享块
                self.shm = SharedMemory(
                    name=self.shared_memory_name,
                    create=True,                # 连接已有的块名还是重新创建
                    size=2**20                  # 大小
                )
                dist.barrier()                  #多进程同步屏障 让所有参与分布式训练的进程（通过 world_size 定义）都在这个代码位置等待，直到所有进程都执行到此处，才会继续往下运行。
            else:
                dist.barrier()
                self.shm = SharedMemory(
                    name=self.shared_memory_name
                )
                self.loop()

    def _maybe_register_qwen35_mtp_executor(
        self,
        *,
        registration_dependencies=None,
    ):
        config = self.config
        if not getattr(config, "qwen35_mtp_enabled", False):
            return None
        hf_config = getattr(config, "hf_config", None)
        if getattr(hf_config, "model_type", None) != "qwen3_5":
            return None
        text_config = getattr(
            hf_config,
            "text_config",
            hf_config,
        )
        tensor_parallel_size = getattr(
            config,
            "tensor_parallel_size",
            None,
        )
        if (
            tensor_parallel_size not in (1, 4)
            or self.world_size != tensor_parallel_size
            or self.rank < 0
            or self.rank >= self.world_size
            or getattr(
                text_config,
                "mtp_num_hidden_layers",
                None,
            ) != 1
            or getattr(
                text_config,
                "mtp_use_dedicated_embeddings",
                None,
            )
            is not False
            or getattr(
                text_config,
                "tie_word_embeddings",
                None,
            )
            is not True
        ):
            return None
        owner = getattr(
            self,
            "qwen35_hybrid_model_owner",
            None,
        )
        if owner is None or owner.model is not self.model:
            return None
        dependencies = (
            _qwen35_mtp_registration_dependencies()
            if registration_dependencies is None
            else registration_dependencies
        )
        target_model = self.model
        self.qwen35_mtp_physical_store = None
        self.qwen35_mtp_executor = None
        try:
            metadata = dependencies.read_metadata(config)
            plan = dependencies.build_checkpoint_plan(
                metadata.hf_config,
                metadata.index_payload,
                metadata.shard_headers,
            )
            module = dependencies.build_module(
                metadata.hf_config,
                embed_tokens=target_model.embed_tokens,
                lm_head=target_model.lm_head,
                tensor_parallel_size=tensor_parallel_size,
                tensor_parallel_rank=self.rank,
            )
            if (
                module.embed_tokens is not target_model.embed_tokens
                or module.lm_head is not target_model.lm_head
            ):
                raise RuntimeError(
                    "Qwen3.5 MTP shared modules lost identity"
                )
            expected_sources = tuple(sorted(
                tensor.source_name
                for tensor in plan.tensors
            ))
            if (
                len(expected_sources) != 15
                or len(set(expected_sources)) != 15
            ):
                raise RuntimeError(
                    "Qwen3.5 MTP checkpoint plan must contain "
                    "exactly 15 unique sources"
                )
            bound_sources = dependencies.bind_checkpoint(
                module,
                plan,
                lambda tensor: dependencies.read_tensor(
                    config,
                    tensor,
                ),
            )
            if bound_sources != expected_sources:
                raise RuntimeError(
                    "Qwen3.5 MTP checkpoint binding coverage "
                    "is incomplete"
                )
            dependencies.move_module_to_device(
                module,
                target_model,
            )
            entry_allocator = (
                dependencies.build_proposal_kv_allocator(
                config,
                module,
                )
            )
            physical_store = getattr(
                entry_allocator,
                "storage",
                getattr(entry_allocator, "physical_store", None),
            )
            if physical_store is None:
                raise RuntimeError(
                    "Qwen3.5 MTP proposal KV storage is unavailable"
                )
            attention_backend = (
                module.layer.decoder_layer.full_attention
                .attention_backend
            )
            physical_store.bind_attention_backend(
                attention_backend
            )
            proposal_kv_cache = (
                dependencies.build_proposal_kv_cache(
                    entry_allocator
                )
            )
            graph_runner = dependencies.build_graph_runner(
                config,
                module,
                proposal_kv_cache,
            )
            executor = dependencies.build_executor(
                module=module,
                proposal_kv_cache=proposal_kv_cache,
                max_proposal_tokens=(
                    config.qwen35_mtp_max_proposal_tokens
                ),
                graph_runner=graph_runner,
                tensor_parallel_rank=self.rank,
                tensor_parallel_size=tensor_parallel_size,
            )
            descriptor = ModelRunnerProposalExecutorDescriptor(
                executor_id="native_checkpoint_proposal",
                capabilities=executor.capabilities,
            )
            self.speculative_proposal_executors.register(
                descriptor.executor_id,
                executor,
                descriptor.capabilities,
            )
        except Exception as error:
            self.qwen35_mtp_registration_error = (
                f"{type(error).__name__}: {error}"
            )
            return None
        self.qwen35_mtp_module = module
        self.qwen35_mtp_executor_descriptor = descriptor
        self.qwen35_mtp_registration_error = None
        self.qwen35_mtp_physical_store = physical_store
        self.qwen35_mtp_executor = executor
        return descriptor

    def _maybe_register_autoregressive_draft_executor(
        self,
        *,
        registration_dependencies=None,
        tensor_parallel_coordinator=None,
    ):
        config = self.config
        if not getattr(
            config,
            "autoregressive_draft_enabled",
            False,
        ):
            return None
        tensor_parallel_size = getattr(
            config,
            "tensor_parallel_size",
            None,
        )
        if (
            tensor_parallel_size not in (1, 4)
            or self.world_size != tensor_parallel_size
            or self.rank < 0
            or self.rank >= self.world_size
        ):
            raise RuntimeError(
                "autoregressive draft requires matching "
                "TP1 or TP4 topology"
            )
        graph_enabled = bool(
            getattr(
                config,
                "autoregressive_draft_cuda_graphs",
                False,
            )
        )
        if graph_enabled and tensor_parallel_size != 4:
            raise RuntimeError(
                "autoregressive draft CUDA graph requires TP4"
            )
        if graph_enabled and bool(
            config
            .autoregressive_draft_proposal_kv_offload_enabled
        ):
            raise RuntimeError(
                "autoregressive draft CUDA graph does not support "
                "proposal KV offload"
            )
        dependencies = (
            _autoregressive_draft_registration_dependencies()
            if registration_dependencies is None
            else registration_dependencies
        )
        target_device = torch.device("cpu")
        try:
            target_device = next(self.model.parameters()).device
        except (AttributeError, StopIteration, TypeError):
            pass
        coordinator = (
            AutoregressiveDraftTensorParallelCoordinator(
                rank=self.rank,
                world_size=self.world_size,
                device=target_device,
            )
            if tensor_parallel_coordinator is None
            else tensor_parallel_coordinator
        )
        stage = "initialization"
        candidate = None
        local_error = None
        try:
            stage = "fingerprint_target_checkpoint"
            target_checkpoint = (
                dependencies.build_checkpoint_fingerprint(
                    config.model
                )
            )
            stage = "fingerprint_draft_checkpoint"
            draft_checkpoint = (
                dependencies.build_checkpoint_fingerprint(
                    config.autoregressive_draft_model
                )
            )
            stage = "load_target_tokenizer"
            target_tokenizer = dependencies.load_tokenizer(
                config.model
            )
            stage = "build_target_tokenizer_contract"
            target_tokenizer_contract = (
                dependencies.build_tokenizer_contract(
                    config.model,
                    target_tokenizer,
                )
            )
            stage = "load_draft_tokenizer"
            draft_tokenizer = dependencies.load_tokenizer(
                config.autoregressive_draft_model
            )
            stage = "build_draft_tokenizer_contract"
            draft_tokenizer_contract = (
                dependencies.build_tokenizer_contract(
                    config.autoregressive_draft_model,
                    draft_tokenizer,
                )
            )
            stage = "validate_tokenizer_compatibility"
            dependencies.validate_tokenizer_compatibility(
                target_tokenizer_contract,
                draft_tokenizer_contract,
            )
            stage = "load_draft_hf_config"
            draft_hf_config = dependencies.load_hf_config(
                config.autoregressive_draft_model
            )
            if getattr(
                draft_hf_config,
                "model_type",
                None,
            ) != "qwen3":
                raise ValueError(
                    "autoregressive draft checkpoint must use "
                    "model_type qwen3"
                )
            stage = "build_draft_model"
            draft_model = dependencies.build_model(
                draft_hf_config,
                tensor_parallel_rank=self.rank,
                tensor_parallel_size=self.world_size,
            )
            stage = "load_draft_weights"
            dependencies.load_weights(
                draft_model,
                config.autoregressive_draft_model,
            )
            stage = "move_and_eval_draft_model"
            device, dtype = dependencies.move_model_to_target(
                draft_model,
                self.model,
            )
            stage = "build_proposal_kv_allocator"
            offload_enabled = bool(
                config
                .autoregressive_draft_proposal_kv_offload_enabled
            )
            entry_allocator = (
                dependencies.build_proposal_kv_allocator(
                    draft_model,
                    offload_enabled=offload_enabled,
                    logical_entry_capacity=(
                        int(
                            config
                            .autoregressive_draft_logical_entry_capacity
                        )
                        if offload_enabled
                        else int(
                            config
                            .autoregressive_draft_gpu_slot_capacity
                        )
                    ),
                    gpu_slot_capacity=int(
                        config
                        .autoregressive_draft_gpu_slot_capacity
                    ),
                    cpu_backing_capacity=(
                        int(
                            config
                            .autoregressive_draft_cpu_backing_capacity
                        )
                        if offload_enabled
                        else int(
                            config
                            .autoregressive_draft_gpu_slot_capacity
                        )
                    ),
                    async_copy=bool(
                        config.proposal_kv_async_copy
                    ),
                    batch_copy=bool(
                        config.proposal_kv_batch_copy
                    ),
                    dtype=dtype,
                    device=device,
                )
            )
            physical_store = getattr(
                entry_allocator,
                "storage",
                getattr(
                    entry_allocator,
                    "physical_store",
                    None,
                ),
            )
            if physical_store is None:
                raise RuntimeError(
                    "autoregressive draft proposal KV storage "
                    "is unavailable"
                )
            stage = "build_proposal_kv_cache"
            proposal_kv_cache = (
                dependencies.build_proposal_kv_cache(
                    entry_allocator
                )
            )
            stage = "build_qwen3_draft_backend"
            backend = dependencies.build_backend(
                model=draft_model,
                proposal_kv_cache=proposal_kv_cache,
                backend_identity=(
                    config.autoregressive_draft_backend
                ),
                model_fingerprint=(
                    draft_checkpoint.composite_sha256
                ),
                tokenizer_fingerprint=(
                    draft_tokenizer_contract.composite_sha256
                ),
                tensor_parallel_rank=self.rank,
                tensor_parallel_size=self.world_size,
            )
            graph_components = None
            if graph_enabled:
                stage = "build_autoregressive_draft_graph"
                graph_components = (
                    dependencies.build_graph_components(
                        config=config,
                        backend=backend,
                        proposal_kv_cache=proposal_kv_cache,
                        physical_store=physical_store,
                        device=device,
                        dtype=dtype,
                    )
                )
            stage = "build_autoregressive_draft_executor"
            executor = dependencies.build_executor(
                backend=backend,
                proposal_kv_cache=proposal_kv_cache,
                max_proposal_tokens=(
                    config
                    .autoregressive_draft_max_proposal_tokens
                ),
                tensor_parallel_rank=self.rank,
                tensor_parallel_size=self.world_size,
                tensor_parallel_coordinator=coordinator,
                graph_runner=(
                    None
                    if graph_components is None
                    else graph_components.runner
                ),
            )
            stage = "build_executor_descriptor"
            descriptor = dependencies.build_descriptor(executor)
            if descriptor.executor_id != "autoregressive-draft":
                raise ValueError(
                    "autoregressive draft descriptor ID mismatch"
                )
            candidate = AutoregressiveDraftRegistrationCandidate(
                target_checkpoint=target_checkpoint,
                draft_checkpoint=draft_checkpoint,
                target_tokenizer_contract=(
                    target_tokenizer_contract
                ),
                draft_tokenizer_contract=(
                    draft_tokenizer_contract
                ),
                model=draft_model,
                physical_store=physical_store,
                proposal_kv_cache=proposal_kv_cache,
                backend=backend,
                executor=executor,
                descriptor=descriptor,
                graph_components=graph_components,
            )
            stage = "registry_preflight"
            self.speculative_proposal_executors.preflight_registration(
                descriptor.executor_id,
                executor,
                descriptor.capabilities,
            )
        except Exception as error:
            local_error = error

        local_status = (
            build_autoregressive_draft_registration_status(
                rank=self.rank,
                world_size=self.world_size,
                stage=stage,
                candidate=candidate,
                error=local_error,
            )
        )
        try:
            statuses = coordinator.collect_registration_status(
                local_status
            )
            consensus_sha256 = (
                validate_autoregressive_draft_registration_consensus(
                    statuses,
                    world_size=self.world_size,
                )
            )
        except Exception as error:
            self.autoregressive_draft_registration_error = (
                AutoregressiveDraftRegistrationError(
                    stage="registration_consensus",
                    error_type=type(error).__name__,
                    message=str(error),
                )
            )
            return None

        if candidate is None:
            self.autoregressive_draft_registration_error = (
                AutoregressiveDraftRegistrationError(
                    stage="registration_consensus",
                    error_type="RuntimeError",
                    message=(
                        "registration consensus succeeded without "
                        "a local candidate"
                    ),
                )
            )
            return None
        try:
            self.speculative_proposal_executors.register(
                candidate.descriptor.executor_id,
                candidate.executor,
                candidate.descriptor.capabilities,
            )
        except Exception as error:
            self.autoregressive_draft_registration_error = (
                AutoregressiveDraftRegistrationError(
                    stage="register_executor",
                    error_type=type(error).__name__,
                    message=str(error),
                )
            )
            return None

        self.autoregressive_draft_model = candidate.model
        self.autoregressive_draft_physical_store = (
            candidate.physical_store
        )
        self.autoregressive_draft_executor = candidate.executor
        self.autoregressive_draft_executor_descriptor = (
            candidate.descriptor
        )
        self.autoregressive_draft_registration_error = None
        self.autoregressive_draft_registration_consensus_sha256 = (
            consensus_sha256
        )
        self.autoregressive_draft_checkpoint_identity = {
            "target": candidate.target_checkpoint,
            "draft": candidate.draft_checkpoint,
        }
        self.autoregressive_draft_tokenizer_contract = {
            "target": candidate.target_tokenizer_contract,
            "draft": candidate.draft_tokenizer_contract,
        }
        self.autoregressive_draft_graph_components = (
            candidate.graph_components
        )
        return candidate.descriptor

    def autoregressive_draft_authority_snapshot(self) -> dict:
        executor = getattr(
            self,
            "autoregressive_draft_executor",
            None,
        )
        descriptor = getattr(
            self,
            "autoregressive_draft_executor_descriptor",
            None,
        )
        checkpoint_identity = getattr(
            self,
            "autoregressive_draft_checkpoint_identity",
            None,
        )
        tokenizer_contract = getattr(
            self,
            "autoregressive_draft_tokenizer_contract",
            None,
        )
        registration_error = getattr(
            self,
            "autoregressive_draft_registration_error",
            None,
        )
        registration_consensus_sha256 = getattr(
            self,
            "autoregressive_draft_registration_consensus_sha256",
            None,
        )
        executor_snapshot = (
            None
            if executor is None
            else executor.authority_snapshot()
        )
        if executor_snapshot is not None and (
            not isinstance(executor_snapshot, dict)
            or executor_snapshot.get("rank") != self.rank
            or executor_snapshot.get("world_size")
            != self.world_size
        ):
            raise RuntimeError(
                "autoregressive draft executor authority "
                "snapshot topology mismatch"
            )
        snapshot = {
            "rank": self.rank,
            "world_size": self.world_size,
            "registered": executor is not None,
            "registration_consensus_sha256": (
                registration_consensus_sha256
            ),
            "executor_descriptor": (
                None
                if descriptor is None
                else {
                    "executor_id": descriptor.executor_id,
                    "capabilities": asdict(
                        descriptor.capabilities
                    ),
                }
            ),
            "checkpoint_identity": (
                None
                if checkpoint_identity is None
                else {
                    name: asdict(identity)
                    for name, identity
                    in checkpoint_identity.items()
                }
            ),
            "tokenizer_contract": (
                None
                if tokenizer_contract is None
                else {
                    name: asdict(contract)
                    for name, contract
                    in tokenizer_contract.items()
                }
            ),
            "registration_error": (
                None
                if registration_error is None
                else asdict(registration_error)
            ),
            "executor": (
                executor_snapshot
            ),
        }
        assert_tensor_free(
            snapshot,
            name="autoregressive draft authority snapshot",
        )
        return snapshot

    def qwen35_mtp_authority_snapshot(self) -> dict:
        executor = getattr(
            self,
            "qwen35_mtp_executor",
            None,
        )
        if executor is None:
            return {
                "rank": self.rank,
                "world_size": self.world_size,
                "registered": False,
                "executor": None,
            }
        snapshot_method = getattr(
            executor,
            "tp4_authority_snapshot",
            None,
        )
        if not callable(snapshot_method):
            raise RuntimeError(
                "Qwen3.5 MTP authority snapshot is unavailable"
            )
        snapshot = snapshot_method()
        if (
            not isinstance(snapshot, dict)
            or snapshot.get("tensor_parallel_rank") != self.rank
            or snapshot.get("tensor_parallel_size")
            != self.world_size
        ):
            raise RuntimeError(
                "Qwen3.5 MTP authority snapshot topology mismatch"
            )
        module = getattr(self, "qwen35_mtp_module", None)
        physical_store = getattr(
            self,
            "qwen35_mtp_physical_store",
            None,
        )
        target_model = getattr(self, "model", None)
        text_config = getattr(
            getattr(self.config, "hf_config", None),
            "text_config",
            getattr(self.config, "hf_config", None),
        )
        query_heads = getattr(
            text_config,
            "num_attention_heads",
            None,
        )
        kv_heads = getattr(
            text_config,
            "num_key_value_heads",
            None,
        )
        if (
            module is None
            or physical_store is None
            or target_model is None
            or isinstance(query_heads, bool)
            or not isinstance(query_heads, int)
            or query_heads <= 0
            or query_heads % self.world_size != 0
            or isinstance(kv_heads, bool)
            or not isinstance(kv_heads, int)
            or kv_heads <= 0
            or (
                kv_heads >= self.world_size
                and kv_heads % self.world_size != 0
            )
            or (
                kv_heads < self.world_size
                and self.world_size % kv_heads != 0
            )
        ):
            raise RuntimeError(
                "Qwen3.5 MTP authority metadata is unavailable"
            )
        return {
            "rank": self.rank,
            "world_size": self.world_size,
            "registered": True,
            "module_type": type(module).__name__,
            "physical_store_type": type(physical_store).__name__,
            "shared_embed_tokens": (
                getattr(module, "embed_tokens", None)
                is getattr(target_model, "embed_tokens", None)
            ),
            "shared_lm_head": (
                getattr(module, "lm_head", None)
                is getattr(target_model, "lm_head", None)
            ),
            "local_query_heads": query_heads // self.world_size,
            "local_kv_heads": (
                kv_heads // self.world_size
                if kv_heads >= self.world_size
                else 1
            ),
            "executor": snapshot,
        }

    def enable_step_logits_recording(self, enabled: bool) -> dict:
        self._record_step_logits = bool(enabled)
        self._last_step_logits_cpu = None
        return {
            "rank": self.rank,
            "enabled": self._record_step_logits,
        }

    def configure_h2d_slot_reuse_diagnostic(
        self,
        mode: str,
    ) -> dict:
        if self.kv_offload is None:
            raise RuntimeError(
                "H2D slot-reuse diagnostic requires KV offload"
            )
        receipt = (
            self.kv_offload.configure_h2d_slot_reuse_diagnostic(
                mode
            )
        )
        expected = {"rank": self.rank, "mode": mode}
        if receipt != expected:
            raise RuntimeError(
                "H2D slot-reuse configure receipt mismatch"
            )
        return expected

    def set_h2d_slot_reuse_diagnostic_context(
        self,
        engine_step: int,
        attention_stage: str = "decode",
    ) -> dict:
        if self.kv_offload is None:
            raise RuntimeError(
                "H2D slot-reuse diagnostic requires KV offload"
            )
        if (
            isinstance(engine_step, bool)
            or not isinstance(engine_step, int)
            or engine_step < 0
        ):
            raise ValueError(
                "H2D slot-reuse engine_step must be nonnegative"
            )
        if attention_stage not in {"prefill", "decode"}:
            raise ValueError(
                "focused H2D slot-reuse context requires "
                "prefill or decode"
            )
        self.kv_offload.set_h2d_slot_reuse_context(
            engine_step=engine_step,
            attention_stage=attention_stage,
            layer_index=0,
            window_ordinal=0,
        )
        return {
            "rank": self.rank,
            "engine_step": engine_step,
            "attention_stage": attention_stage,
        }

    def drain_h2d_slot_reuse_diagnostic(
        self,
        timing_epsilon_ms: float,
    ) -> dict:
        if self.kv_offload is None:
            raise RuntimeError(
                "H2D slot-reuse diagnostic requires KV offload"
            )
        row = self.kv_offload.drain_h2d_slot_reuse_diagnostic(
            timing_epsilon_ms=timing_epsilon_ms,
        )
        if (
            not isinstance(row, dict)
            or row.get("rank") != self.rank
            or row.get("schema") != H2D_SLOT_REUSE_SCHEMA
            or set(row) != {
                "rank",
                "schema",
                "mode",
                "stream_inventory",
                "read_rows",
                "overwrite_rows",
            }
        ):
            raise RuntimeError(
                "H2D slot-reuse drain receipt mismatch"
            )
        return row

    def h2d_slot_reuse_diagnostic_summary(self) -> dict:
        if self.kv_offload is None:
            raise RuntimeError(
                "H2D slot-reuse diagnostic requires KV offload"
            )
        row = self.kv_offload.h2d_slot_reuse_diagnostic_summary()
        if row.get("rank") != self.rank:
            raise RuntimeError(
                "H2D slot-reuse summary rank mismatch"
            )
        return row

    def enable_spec_verify_trace_recording(
        self,
        enabled: bool,
    ) -> dict:
        return self._spec_verify_trace.enable(enabled)

    def set_spec_verify_trace_context(
        self,
        policy: str,
        batch_size: int,
        engine_step: int,
    ) -> dict:
        context = TargetForwardTraceContext(
            policy=policy,
            batch_size=batch_size,
            engine_step=engine_step,
        )
        self._spec_verify_trace.set_context(context)
        return {
            "rank": self.rank,
            "policy": policy,
            "batch_size": batch_size,
            "engine_step": engine_step,
        }

    def drain_spec_verify_trace_rows(
        self,
    ) -> tuple[dict, ...]:
        return self._spec_verify_trace.drain()

    def _trace_block_identities(
        self,
        block_table,
    ) -> tuple[tuple[int, int], ...]:
        if not self._spec_verify_trace.enabled:
            return ()
        if self.kv_offload is None:
            raise RuntimeError(
                "trace block identities require kv_offload_mvp0"
            )
        identities = []
        for block_id in block_table:
            block_id = int(block_id)
            if (
                block_id < 0
                or block_id
                >= len(self.kv_offload.bound_generations)
            ):
                raise RuntimeError(
                    "trace logical block id is out of range"
                )
            generation = (
                self.kv_offload.bound_generations[block_id]
            )
            if generation is None:
                raise RuntimeError(
                    "trace block generation is missing"
                )
            identities.append((block_id, int(generation)))
        return tuple(identities)

    def _record_spec_first_target_trace(
        self,
        *,
        seqs,
        input_ids,
        positions,
        logits,
    ) -> None:
        if not self._spec_verify_trace.enabled:
            return
        self._spec_verify_trace.record_rows(
            stage="first_target",
            execution_mode="decode",
            sequence_ids=tuple(
                int(seq.seq_id) for seq in seqs
            ),
            query_offset=0,
            query_len=1,
            input_tokens=tuple(
                int(value)
                for value in input_ids.detach().cpu().tolist()
            ),
            positions=tuple(
                int(value)
                for value in positions.detach().cpu().tolist()
            ),
            prediction_indices=tuple(
                int(seq.num_completion_tokens)
                for seq in seqs
            ),
            logical_block_identities=tuple(
                self._trace_block_identities(seq.block_table)
                for seq in seqs
            ),
            logits=logits,
        )

    def configure_qwen35_recurrent_capture(self, configuration):
        if self.qwen35_recurrent_capture_session is not None:
            raise RuntimeError("Qwen3.5 recurrent capture is already configured")
        if not isinstance(configuration, dict) or set(configuration) != {
            "capture_root",
            "model_manifest_sha256",
            "source_tree_sha256",
            "workload_manifest_sha256",
            "world_size",
            "workload_ids",
        }:
            raise ValueError("recurrent capture configuration fields mismatch")
        if (
            not isinstance(configuration["capture_root"], str)
            or not configuration["capture_root"]
            or not isinstance(configuration["workload_ids"], list)
        ):
            raise ValueError("recurrent capture configuration is invalid")
        owner = self.qwen35_hybrid_model_owner
        if owner is None:
            raise RuntimeError("Qwen3.5 hybrid model owner is not bound")
        linear_layer_indices = tuple(owner.layer_stack.linear_indices)
        if (
            len(linear_layer_indices) != 18
            or linear_layer_indices
            != tuple(sorted(set(linear_layer_indices)))
        ):
            raise ValueError(
                "Qwen3.5 recurrent capture requires 18 sorted unique layers"
            )
        identity = validate_run_identity({
            "schema_version": CAPTURE_IDENTITY_SCHEMA_VERSION,
            "model_manifest_sha256": configuration[
                "model_manifest_sha256"
            ],
            "source_tree_sha256": configuration["source_tree_sha256"],
            "workload_manifest_sha256": configuration[
                "workload_manifest_sha256"
            ],
            "world_size": configuration["world_size"],
            "workload_ids": configuration["workload_ids"],
            "linear_layer_indices": list(linear_layer_indices),
        })
        session = Qwen35RecurrentCaptureSession(
            run_identity=identity,
            rank=self.rank,
            staging_dir=configuration["capture_root"],
        )
        self.qwen35_recurrent_capture_session = session
        self.qwen35_recurrent_capture_workload_id = None
        self.qwen35_recurrent_capture_armed = False
        return {
            "rank": self.rank,
            "configured": True,
            "workload_ids": identity.workload_ids,
            "linear_layer_indices": identity.linear_layer_indices,
        }

    def arm_qwen35_recurrent_capture(self, workload_id):
        session = self.qwen35_recurrent_capture_session
        if session is None:
            raise RuntimeError("Qwen3.5 recurrent capture is not configured")
        if workload_id not in session.run_identity.workload_ids:
            raise ValueError("recurrent capture workload is not declared")
        if self.qwen35_recurrent_capture_armed:
            raise RuntimeError("Qwen3.5 recurrent capture is already armed")
        if self.qwen35_recurrent_capture_workload_id is not None:
            raise RuntimeError("Qwen3.5 recurrent capture state is inconsistent")
        self.qwen35_recurrent_capture_workload_id = workload_id
        self.qwen35_recurrent_capture_armed = True
        return {
            "rank": self.rank,
            "workload_id": workload_id,
            "armed": True,
        }

    def finish_qwen35_recurrent_capture_workload(self, workload_id):
        session = self.qwen35_recurrent_capture_session
        if session is None:
            raise RuntimeError("Qwen3.5 recurrent capture is not configured")
        if self.qwen35_recurrent_capture_workload_id != workload_id:
            raise ValueError("recurrent capture workload is not active")
        if self.qwen35_recurrent_capture_armed:
            raise RuntimeError("recurrent capture workload has not captured")
        session.finish_workload(workload_id)
        self.qwen35_recurrent_capture_workload_id = None
        return {
            "rank": self.rank,
            "workload_id": workload_id,
            "complete": True,
        }

    def _capture_qwen35_recurrent_source_state(
        self,
        seqs,
        *,
        is_prefill,
        batch_kind,
    ):
        session = self.qwen35_recurrent_capture_session
        if session is None or not self.qwen35_recurrent_capture_armed:
            return
        if not is_prefill:
            raise ValueError("recurrent capture requires prefill")
        if batch_kind == "mixed":
            raise ValueError("recurrent capture rejects mixed batches")
        if len(seqs) != 1:
            raise ValueError("recurrent capture requires one sequence")
        leases = tuple(self._last_hybrid_state_leases)
        if len(leases) != 1:
            raise ValueError("recurrent capture requires one active lease")
        seq = seqs[0]
        if not getattr(seq, "prefill_chunk_final", False):
            raise ValueError("recurrent capture requires final prefill")
        lease = leases[0]
        if (
            seq.hybrid_state_slot_id != lease.slot_id
            or seq.hybrid_state_generation != lease.generation
        ):
            raise ValueError("recurrent capture lease does not match sequence")
        owner = self.qwen35_hybrid_model_owner
        if owner is None:
            raise RuntimeError("Qwen3.5 hybrid model owner is not bound")
        workload_id = self.qwen35_recurrent_capture_workload_id
        for adapter in owner.state_transaction.adapters:
            session.capture_layer(
                workload_id=workload_id,
                layer_index=adapter.layer_index,
                tensor=adapter.recurrent[lease.slot_id],
            )
        self.qwen35_recurrent_capture_armed = False

    def last_step_logits(self) -> torch.Tensor | None:
        if self._last_step_logits_cpu is None:
            return None
        return self._last_step_logits_cpu.clone()

    def exit(self):
        draft_executor = getattr(
            self,
            "autoregressive_draft_executor",
            None,
        )
        if draft_executor is not None:
            draft_executor.close()
        if self.world_size > 1:
            self.shm.close()                   # 关闭所有rank和共享内存的连接
            dist.barrier()
            if self.rank == 0:
                self.shm.unlink()              # 删除共享内存对象
        if hasattr(self, "graphs"):
            del self.graphs, self.graph_pool
        torch.cuda.synchronize()
        dist.destroy_process_group()
        return {
            "rank": self.rank,
            "process_group_destroyed": True,
        }

    def loop(self):         #在收到exit命令之前 子进程持续执行method_name方法
        while True:
            envelope = self.read_shm()
            execute_acknowledged_command(
                envelope,
                rank=self.rank,
                target=self,
                send_ack=self.ack_sender.send,
                timeline=self.command_timeline,
                clock_ns=self._command_timeline_clock_ns,
            )
            if envelope.method_name == "exit":
                break

    @staticmethod
    def _read_active_engine_step_trace():
        try:
            from tinyvllm.engine.engine_step_timeline import (
                active_engine_step_trace,
            )
        except ModuleNotFoundError:
            return None
        return active_engine_step_trace()

    def read_shm(self):
        # 多进程环境下 避免主进程调用
        assert self.world_size > 1 and self.rank
        self.event.wait()                               # 等待主进程信号 一直等待，直到 event被set()后才会往下执行
        event_woken_monotonic_ns = (
            self._command_timeline_clock_ns()
            if self.command_timeline.enabled
            else None
        )
        n = int.from_bytes(
            self.shm.buf[0:4],                          # 这里的单位是 byte，一个字节，或者说一个char
            "little")
        payload = pickle.loads(self.shm.buf[4:n+4])
        envelope_read_monotonic_ns = (
            self._command_timeline_clock_ns()
            if self.command_timeline.enabled
            else None
        )
        self.event.clear()                              # 重置事件标志，方便下一次等待
        if isinstance(payload, ModelRunnerCommandEnvelope):
            if (
                payload.trace_identity is not None
                and self.command_timeline.enabled
            ):
                self.command_timeline.record_worker_receive(
                    payload.trace_identity,
                    event_woken_monotonic_ns=(
                        event_woken_monotonic_ns
                    ),
                    envelope_read_monotonic_ns=(
                        envelope_read_monotonic_ns
                    ),
                )
            return payload
        if (
            isinstance(payload, list)
            and payload
            and isinstance(payload[0], str)
        ):
            method_name, *args = payload
            return ModelRunnerCommandEnvelope(
                command_id=next(self._command_ids),
                method_name=method_name,
                args=tuple(args),
                requires_ack=False,
            )
        raise RuntimeError(
            "invalid ModelRunner shared-memory command payload"
        )

    # 主进程
    def write_shm(self, envelope):
        assert self.world_size > 1 and not self.rank        #not self.rank表示self.rank == 0
        if not isinstance(envelope, ModelRunnerCommandEnvelope):
            raise ValueError(
                "envelope must be a ModelRunnerCommandEnvelope"
            )
        data = pickle.dumps(envelope)
        n = len(data)
        if n + 4 > len(self.shm.buf):
            raise RuntimeError(
                "ModelRunner shared-memory command exceeds capacity"
            )
        self.shm.buf[0:4] = n.to_bytes(4, "little")     #把数据长度写入共享内存的前4字节（用小端序存储）
        self.shm.buf[4:n+4] = data
        for event in self.event:
            event.set()

    def dispatch_command(
        self,
        method_name,
        *args,
        requires_ack,
    ):
        if self.rank != 0:
            raise RuntimeError(
                "only rank 0 may dispatch ModelRunner commands"
            )
        command_id = next(self._command_ids)
        trace_identity = None
        trace_context = None
        if self.command_timeline.enabled:
            trace_context = self._active_command_timeline_trace()
        if (
            trace_context is not None
            and getattr(trace_context, "repeat_index", None) is not None
            and getattr(trace_context, "engine_step_id", None) is not None
        ):
            dispatch_started_monotonic_ns = (
                self._command_timeline_clock_ns()
            )
            dispatch_published_monotonic_ns = (
                self._command_timeline_clock_ns()
            )
            trace_identity = CommandTraceIdentity(
                command_id=command_id,
                method_name=method_name,
                requires_ack=requires_ack,
                engine_step_id=trace_context.engine_step_id,
                repeat_index=trace_context.repeat_index,
                request_set_sha256=getattr(
                    trace_context,
                    "request_set_sha256",
                    None,
                ),
                batch_kind=getattr(
                    trace_context,
                    "batch_kind",
                    None,
                ),
                speculative_selected_sequence_ids_sha256=getattr(
                    trace_context,
                    "speculative_selected_sequence_ids_sha256",
                    None,
                ),
                dispatch_started_monotonic_ns=(
                    dispatch_started_monotonic_ns
                ),
                dispatch_published_monotonic_ns=(
                    dispatch_published_monotonic_ns
                ),
            )
        envelope = ModelRunnerCommandEnvelope(
            command_id=command_id,
            method_name=method_name,
            args=tuple(args),
            requires_ack=requires_ack,
            trace_identity=trace_identity,
        )
        if self.world_size > 1:
            self.write_shm(envelope)
        if trace_identity is not None:
            self.command_timeline.record_dispatch(trace_identity)
        return envelope

    def execute_command_envelope(self, envelope):
        return execute_acknowledged_command(
            envelope,
            rank=self.rank,
            target=self,
            send_ack=lambda acknowledgement: None,
            timeline=self.command_timeline,
            clock_ns=self._command_timeline_clock_ns,
        )

    def call(self, method_name, *args):         #动态方法调用 提供一个通用接口 把主进程调用的函数推给从进程
        if self.rank == 0:
            envelope = self.dispatch_command(
                method_name,
                *args,
                requires_ack=False,
            )
            return self.execute_command_envelope(envelope)
        method = getattr(self, method_name, None)       #获取函数对象
        return method(*args)            #执行函数并返回结果

    def configure_command_timeline(self, enabled, max_rows):
        if not isinstance(enabled, bool):
            raise ValueError(
                "command timeline enabled must be a bool"
            )
        if (
            isinstance(max_rows, bool)
            or not isinstance(max_rows, int)
            or max_rows <= 0
        ):
            raise ValueError(
                "command timeline max rows must be a positive integer"
            )
        self._command_timeline_max_rows = max_rows
        self.command_timeline = (
            ModelRunnerCommandTimelineRecorder(
                rank=self.rank,
                max_rows=max_rows,
                clock_identity=read_command_clock_identity(),
            )
            if enabled
            else ModelRunnerCommandTimelineRecorder.disabled(self.rank)
        )
        return {
            "rank": self.rank,
            "enabled": enabled,
            "max_rows": max_rows,
        }

    def reset_command_timeline(self):
        return self.configure_command_timeline(
            self.command_timeline.enabled,
            self._command_timeline_max_rows,
        )

    def command_timeline_snapshot(self):
        return self.command_timeline.snapshot()

    def memory_snapshot(self):
        kv_bytes = int(
            self.kv_cache.numel() * self.kv_cache.element_size()
        )
        if self.kv_scale is not None:
            kv_bytes += int(
                self.kv_scale.numel() * self.kv_scale.element_size()
            )
        if self.kv_zero is not None:
            kv_bytes += int(
                self.kv_zero.numel() * self.kv_zero.element_size()
            )
        return {
            "cuda_allocated_bytes": int(torch.cuda.memory_allocated()),
            "cuda_reserved_bytes": int(torch.cuda.memory_reserved()),
            "cuda_peak_allocated_bytes": int(
                torch.cuda.max_memory_allocated()
            ),
            "cuda_peak_reserved_bytes": int(
                torch.cuda.max_memory_reserved()
            ),
            "kv_capacity_bytes": kv_bytes,
        }

    def reset_peak_memory_stats(self):
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        return self.memory_snapshot()

    def _prepare_hybrid_state_batch(
        self,
        seqs: list[Sequence],
        released_leases: tuple[HybridStateLease, ...],
    ):
        released_leases = tuple(released_leases)
        active_leases = []
        for seq in seqs:
            slot_id = getattr(seq, "hybrid_state_slot_id", -1)
            generation = getattr(seq, "hybrid_state_generation", 0)
            if slot_id < 0:
                if generation != 0:
                    raise RuntimeError(
                        "hybrid state sequence metadata is partially enabled"
                    )
                continue
            if generation <= 0:
                raise RuntimeError(
                    "hybrid state sequence metadata is partially enabled"
                )
            active_leases.append(HybridStateLease(
                slot_id=slot_id,
                generation=generation,
                request_id=int(seq.seq_id),
            ))
        active_leases = tuple(active_leases)
        slot_ids = tuple(lease.slot_id for lease in active_leases)
        if len(set(slot_ids)) != len(slot_ids):
            rows = tuple(
                (
                    lease.request_id,
                    lease.slot_id,
                    lease.generation,
                )
                for lease in active_leases
            )
            raise ValueError(
                "active hybrid state leases must reference distinct "
                f"slot ids: rank={self.rank}, leases={rows}"
            )
        runtime_bridge = getattr(
            self,
            "hybrid_state_runtime_bridge",
            None,
        )
        if runtime_bridge is None:
            self._last_hybrid_state_slot_ids = None
            if released_leases or active_leases:
                raise RuntimeError(
                    "hybrid state runtime bridge is not installed"
                )
            return None
        slot_ids = runtime_bridge.prepare_batch(
            released_leases,
            active_leases,
        )
        self._last_hybrid_state_slot_ids = slot_ids
        return slot_ids

    def release_hybrid_state(
        self,
        released_leases: tuple[HybridStateLease, ...],
    ) -> None:
        released_leases = tuple(released_leases)
        if not released_leases:
            return
        runtime_bridge = getattr(
            self,
            "hybrid_state_runtime_bridge",
            None,
        )
        if runtime_bridge is None:
            raise RuntimeError(
                "hybrid state runtime bridge is not installed"
            )
        runtime_bridge.release(released_leases)

    def install_qwen35_hybrid_prefix_restore_participant(
        self,
        participant,
    ) -> None:
        if not isinstance(
            participant,
            Qwen35HybridPrefixRestoreParticipant,
        ):
            raise ValueError(
                "participant must be a "
                "Qwen35HybridPrefixRestoreParticipant"
            )
        if participant.participant_id != self.rank:
            raise ValueError(
                "participant id must match ModelRunner rank"
            )
        current = self.qwen35_hybrid_prefix_restore_participant
        if current is not None:
            if current is participant:
                return
            raise RuntimeError(
                "hybrid prefix restore participant already installed"
            )
        runtime_bridge = getattr(
            self,
            "hybrid_state_runtime_bridge",
            None,
        )
        if (
            runtime_bridge is not None
            and runtime_bridge.pool is not participant.pool
        ):
            raise ValueError(
                "restore participant pool must match runtime bridge pool"
            )
        self.qwen35_hybrid_prefix_restore_participant = participant

    def install_qwen35_hybrid_prefix_publication_participant(
        self,
        participant,
    ) -> None:
        if not isinstance(
            participant,
            Qwen35HybridPrefixPublicationParticipant,
        ):
            raise ValueError(
                "participant must be a "
                "Qwen35HybridPrefixPublicationParticipant"
            )
        if participant.participant_id != self.rank:
            raise ValueError(
                "participant id must match ModelRunner rank"
            )
        current = (
            self.qwen35_hybrid_prefix_publication_participant
        )
        if current is not None:
            if current is participant:
                return
            raise RuntimeError(
                "hybrid prefix publication participant "
                "already installed"
            )
        runtime_bridge = getattr(
            self,
            "hybrid_state_runtime_bridge",
            None,
        )
        if (
            runtime_bridge is not None
            and runtime_bridge.pool is not participant.pool
        ):
            raise ValueError(
                "publication participant pool must match "
                "runtime bridge pool"
            )
        self.qwen35_hybrid_prefix_publication_participant = (
            participant
        )

    def bind_qwen35_hybrid_model_owner(self, owner):
        if type(owner) is not Qwen35HybridModelOwner:
            raise ValueError(
                "owner must be a Qwen35HybridModelOwner"
            )
        if owner.model is not self.model:
            raise ValueError(
                "owner model must be the ModelRunner current model"
            )
        if (
            owner.model.layer_stack is not owner.layer_stack
            or owner.layer_stack.state_transaction
            is not owner.state_transaction
            or owner.state_transaction.pool is not owner.pool
            or owner.runtime_bridge.pool is not owner.pool
        ):
            raise ValueError(
                "owner must preserve one coherent ownership graph"
            )
        current = getattr(
            self,
            "qwen35_hybrid_model_owner",
            None,
        )
        if current is not None:
            if current is owner:
                return
            speculative_owner = getattr(
                self,
                "qwen35_speculative_state_owner",
                None,
            )
            if (
                speculative_owner is not None
                and speculative_owner.active
            ):
                raise RuntimeError(
                    "cannot replace Qwen3.5 model owner while a "
                    "speculative side-state transaction is active"
                )
            raise RuntimeError(
                "Qwen3.5 hybrid model owner is already bound"
            )
        runtime_bridge = getattr(
            self,
            "hybrid_state_runtime_bridge",
            None,
        )
        if (
            runtime_bridge is not None
            and runtime_bridge is not owner.runtime_bridge
        ):
            raise RuntimeError(
                "a different hybrid state runtime bridge "
                "is already installed"
            )
        restore_owner = getattr(
            self,
            "qwen35_hybrid_prefix_restore_owner",
            None,
        )
        if (
            restore_owner is not None
            and restore_owner.pool is not owner.pool
        ):
            raise RuntimeError(
                "hybrid prefix restore owner pool does not match "
                "the model owner"
            )
        participant = getattr(
            self,
            "qwen35_hybrid_prefix_restore_participant",
            None,
        )
        if (
            participant is not None
            and participant.pool is not owner.pool
        ):
            raise RuntimeError(
                "hybrid prefix restore participant pool does not "
                "match the model owner"
            )
        publication_participant = getattr(
            self,
            "qwen35_hybrid_prefix_publication_participant",
            None,
        )
        if (
            publication_participant is not None
            and publication_participant.pool is not owner.pool
        ):
            raise RuntimeError(
                "hybrid prefix publication participant pool does "
                "not match the model owner"
            )
        self.qwen35_hybrid_model_owner = owner
        self.hybrid_state_runtime_bridge = owner.runtime_bridge
        self.qwen35_speculative_state_owner = (
            Qwen35SpeculativeStateOwner(
                owner.state_transaction,
            )
        )
        self.qwen35_speculative_state_owner = (
            Qwen35SpeculativeStateOwner(
                owner.state_transaction,
            )
        )

    def bind_current_qwen35_hybrid_model(self):
        owner = build_qwen35_hybrid_model_owner(self.model)
        self.bind_qwen35_hybrid_model_owner(owner)
        return {
            "participant_id": int(self.rank),
            "capacity": int(owner.pool.capacity),
            "layout_fingerprint": owner.pool.layout.fingerprint,
            "bytes_per_slot": int(owner.pool.layout.bytes_per_slot),
            "linear_layer_indices": tuple(
                owner.layer_stack.linear_indices
            ),
        }

    def bind_qwen35_hybrid_prefix_runtime_identity(
        self,
        model_fingerprint,
    ):
        owner = getattr(
            self,
            "qwen35_hybrid_model_owner",
            None,
        )
        if owner is None:
            raise RuntimeError(
                "Qwen3.5 hybrid model owner is not bound"
            )
        current = getattr(
            self,
            "qwen35_hybrid_prefix_runtime_identity",
            None,
        )
        current_owner = getattr(
            self,
            "qwen35_hybrid_prefix_runtime_identity_owner",
            None,
        )
        if current is not None:
            if current_owner is not owner:
                raise RuntimeError(
                    "Qwen3.5 hybrid model owner changed after "
                    "runtime identity binding"
                )
            if current.model_fingerprint == model_fingerprint:
                return current.rank_row(int(self.rank))
            raise RuntimeError(
                "Qwen3.5 hybrid prefix runtime identity "
                "is already bound"
            )
        identity = _bind_qwen35_hybrid_prefix_runtime_identity(
            owner,
            model_fingerprint,
        )
        self.qwen35_hybrid_prefix_runtime_identity = identity
        self.qwen35_hybrid_prefix_runtime_identity_owner = owner
        return identity.rank_row(int(self.rank))

    def bind_qwen35_loaded_checkpoint_candidate(
        self,
        candidate,
    ):
        if type(candidate) is not Qwen35LoadedCheckpointCandidate:
            raise ValueError(
                "candidate must be an exact "
                "Qwen35LoadedCheckpointCandidate"
            )
        identity = _bind_qwen35_hybrid_prefix_runtime_identity(
            candidate.owner,
            candidate.model_fingerprint,
        )
        current_owner = getattr(
            self,
            "qwen35_hybrid_model_owner",
            None,
        )
        current_identity = getattr(
            self,
            "qwen35_hybrid_prefix_runtime_identity",
            None,
        )
        identity_owner = getattr(
            self,
            "qwen35_hybrid_prefix_runtime_identity_owner",
            None,
        )
        if current_owner is not None or current_identity is not None:
            if (
                current_owner is candidate.owner
                and identity_owner is candidate.owner
                and current_identity == identity
            ):
                return identity.rank_row(int(self.rank))
            if (
                current_owner is None
                or current_identity is None
                or identity_owner is None
            ):
                raise RuntimeError(
                    "Qwen3.5 loaded checkpoint candidate "
                    "binding state is partial"
                )
            raise RuntimeError(
                "Qwen3.5 loaded checkpoint candidate "
                "is already bound"
            )
        if identity_owner is not None:
            raise RuntimeError(
                "Qwen3.5 loaded checkpoint candidate "
                "binding state is partial"
            )
        self.bind_qwen35_hybrid_model_owner(candidate.owner)
        self.qwen35_hybrid_prefix_runtime_identity = identity
        self.qwen35_hybrid_prefix_runtime_identity_owner = (
            candidate.owner
        )
        return identity.rank_row(int(self.rank))

    def publish_qwen35_loaded_checkpoint_candidate(
        self,
        candidate,
    ):
        self.qwen35_loaded_checkpoint_candidate_slot.publish(
            candidate
        )
        return candidate

    def install_qwen35_checkpoint_candidate_loader(
        self,
        loader,
        *,
        authorization_sha256,
    ):
        if not callable(loader):
            raise ValueError("loader must be callable")
        if (
            not isinstance(authorization_sha256, str)
            or len(authorization_sha256) != 64
            or any(
                character not in "0123456789abcdef"
                for character in authorization_sha256
            )
        ):
            raise ValueError(
                "authorization_sha256 must be a lowercase SHA256"
            )
        current = getattr(
            self,
            "qwen35_checkpoint_candidate_loader",
            None,
        )
        current_authorization = getattr(
            self,
            "qwen35_checkpoint_candidate_loader_authorization_sha256",
            None,
        )
        if current is not None or current_authorization is not None:
            if (
                current is loader
                and current_authorization == authorization_sha256
            ):
                return
            raise RuntimeError(
                "Qwen3.5 checkpoint candidate loader "
                "is already installed"
            )
        self.qwen35_checkpoint_candidate_loader = loader
        self.qwen35_checkpoint_candidate_loader_authorization_sha256 = (
            authorization_sha256
        )

    def load_and_publish_qwen35_checkpoint_candidate(
        self,
        request,
    ):
        operation = "load_checkpoint_candidate"

        def error_row(error):
            detail = f"{type(error).__name__}: {error}"
            encoded = detail.encode("utf-8")
            if len(encoded) > 4096:
                detail = encoded[:4096].decode(
                    "utf-8",
                    errors="ignore",
                )
            return {
                "participant_id": int(self.rank),
                "operation": operation,
                "status": "error",
                "model_fingerprint": "",
                "detail": detail,
            }

        try:
            request = (
                validate_qwen35_checkpoint_candidate_load_request(
                    request
                )
            )
            configuration = (
                request.checkpoint_dir,
                request.model_fingerprint,
                request.max_tensor_bytes,
                request.authorization_sha256,
            )
            completed = getattr(
                self,
                "qwen35_checkpoint_candidate_load_configuration",
                None,
            )
            completed_request = getattr(
                self,
                "qwen35_checkpoint_candidate_load_request",
                None,
            )
            published = (
                self.qwen35_loaded_checkpoint_candidate_slot.candidate
            )
            if completed is not None:
                completed_loader = getattr(
                    self,
                    "qwen35_checkpoint_candidate_loader",
                    None,
                )
                completed_authorization = getattr(
                    self,
                    "qwen35_checkpoint_candidate_loader_authorization_sha256",
                    None,
                )
                if (
                    completed == configuration
                    and completed_request == request
                    and callable(completed_loader)
                    and completed_authorization
                    == request.authorization_sha256
                    and type(published)
                    is Qwen35LoadedCheckpointCandidate
                    and published.model_fingerprint
                    == request.model_fingerprint
                ):
                    return {
                        "participant_id": int(self.rank),
                        "operation": operation,
                        "status": "published",
                        "model_fingerprint": (
                            request.model_fingerprint
                        ),
                        "detail": "",
                    }
                if (
                    completed == configuration
                    and completed_request == request
                ):
                    raise RuntimeError(
                        "Qwen3.5 checkpoint candidate load "
                        "completion state is incomplete"
                    )
                raise RuntimeError(
                    "Qwen3.5 checkpoint candidate load "
                    "is already completed"
                )
            if completed_request is not None or published is not None:
                raise RuntimeError(
                    "Qwen3.5 checkpoint candidate load state "
                    "is incomplete"
                )
            loader = getattr(
                self,
                "qwen35_checkpoint_candidate_loader",
                None,
            )
            authorization = getattr(
                self,
                "qwen35_checkpoint_candidate_loader_authorization_sha256",
                None,
            )
            if loader is None and authorization is None:
                raise RuntimeError(
                    "Qwen3.5 checkpoint candidate loader "
                    "is not installed"
                )
            if not callable(loader) or authorization is None:
                raise RuntimeError(
                    "Qwen3.5 checkpoint candidate loader "
                    "state is incomplete"
                )
            if request.authorization_sha256 != authorization:
                raise RuntimeError(
                    "Qwen3.5 checkpoint candidate loader "
                    "authorization does not match request"
                )
            candidate = loader(request)
            if type(candidate) is not Qwen35LoadedCheckpointCandidate:
                raise ValueError(
                    "loader must return an exact "
                    "Qwen35LoadedCheckpointCandidate"
                )
            if (
                candidate.model_fingerprint
                != request.model_fingerprint
            ):
                raise ValueError(
                    "loaded checkpoint candidate fingerprint "
                    "does not match request"
                )
            self.qwen35_loaded_checkpoint_candidate_slot.publish(
                candidate
            )
        except Exception as error:
            return error_row(error)
        self.qwen35_checkpoint_candidate_load_request = request
        self.qwen35_checkpoint_candidate_load_configuration = (
            configuration
        )
        return {
            "participant_id": int(self.rank),
            "operation": operation,
            "status": "published",
            "model_fingerprint": request.model_fingerprint,
            "detail": "",
        }

    def bind_published_qwen35_loaded_checkpoint_candidate(self):
        candidate = (
            self.qwen35_loaded_checkpoint_candidate_slot.candidate
        )
        operation = "bind_loaded_checkpoint_candidate"
        if candidate is None:
            return {
                "participant_id": int(self.rank),
                "operation": operation,
                "status": "error",
                "model_fingerprint": "",
                "layout_fingerprint": "",
                "dtype": "",
                "detail": (
                    "loaded checkpoint candidate is not published"
                ),
            }
        try:
            identity = self.bind_qwen35_loaded_checkpoint_candidate(
                candidate
            )
        except Exception as error:
            detail = f"{type(error).__name__}: {error}"
            encoded = detail.encode("utf-8")
            if len(encoded) > 4096:
                detail = encoded[:4096].decode(
                    "utf-8",
                    errors="ignore",
                )
            return {
                "participant_id": int(self.rank),
                "operation": operation,
                "status": "error",
                "model_fingerprint": "",
                "layout_fingerprint": "",
                "dtype": "",
                "detail": detail,
            }
        return {
            "participant_id": identity["participant_id"],
            "operation": operation,
            "status": "bound",
            "model_fingerprint": identity["model_fingerprint"],
            "layout_fingerprint": identity["layout_fingerprint"],
            "dtype": identity["dtype"],
            "detail": "",
        }

    def configure_qwen35_hybrid_prefix_restore_owner(
        self,
        max_entries,
        max_bytes,
        representation="exact_restore",
    ):
        current = getattr(
            self,
            "qwen35_hybrid_prefix_restore_owner",
            None,
        )
        if current is not None:
            if (
                current.max_entries == max_entries
                and current.max_bytes == max_bytes
                and getattr(
                    current,
                    "representation",
                    "exact_restore",
                ) == representation
                and getattr(
                    self,
                    "hybrid_state_runtime_bridge",
                    None,
                ) is not None
                and current.pool is (
                    self.hybrid_state_runtime_bridge.pool
                )
                and (
                    self.qwen35_hybrid_prefix_restore_participant
                    is current.participant
                )
                and (
                    self.qwen35_hybrid_prefix_publication_participant
                    is current.publication_participant
                )
            ):
                owner = current
            else:
                raise RuntimeError(
                    "Qwen3.5 hybrid prefix restore owner "
                    "is already configured"
                )
        else:
            runtime_bridge = getattr(
                self,
                "hybrid_state_runtime_bridge",
                None,
            )
            if runtime_bridge is None:
                raise RuntimeError(
                    "hybrid state runtime bridge is not installed"
                )
            if (
                getattr(
                    self,
                    "qwen35_hybrid_prefix_restore_participant",
                    None,
                )
                is not None
                or getattr(
                    self,
                    "qwen35_hybrid_prefix_publication_participant",
                    None,
                )
                is not None
            ):
                raise RuntimeError(
                    "hybrid prefix participants are already installed"
                )
            owner_kwargs = {
                "participant_id": self.rank,
                "max_entries": max_entries,
                "max_bytes": max_bytes,
            }
            if representation != "exact_restore":
                owner_kwargs["representation"] = representation
            owner = build_qwen35_hybrid_prefix_restore_owner(
                runtime_bridge.pool,
                **owner_kwargs,
            )
            self.install_qwen35_hybrid_prefix_restore_participant(
                owner.participant
            )
            self.install_qwen35_hybrid_prefix_publication_participant(
                owner.publication_participant
            )
            self.qwen35_hybrid_prefix_restore_owner = owner
        owner_representation = getattr(
            owner,
            "representation",
            representation,
        )
        owner_representation_version = getattr(
            owner,
            "representation_version",
            (
                "qwen35_hybrid_prefix_exact_v1"
                if owner_representation == "exact_restore"
                else "qwen35_hybrid_prefix_recurrent_int8_v1"
            ),
        )
        owner_codec = getattr(
            owner,
            "codec",
            (
                None
                if owner_representation == "exact_restore"
                else "qwen35_recurrent_symmetric_int8_per_row_v1"
            ),
        )
        return {
            "participant_id": int(self.rank),
            "capacity": int(owner.pool.capacity),
            "layout_fingerprint": owner.pool.layout.fingerprint,
            "bytes_per_slot": int(owner.pool.layout.bytes_per_slot),
            "max_entries": int(owner.max_entries),
            "max_bytes": int(owner.max_bytes),
            "representation": owner_representation,
            "representation_version": owner_representation_version,
            "codec": owner_codec,
        }

    def qwen35_hybrid_prefix_cache_snapshot(self):
        owner = getattr(
            self,
            "qwen35_hybrid_prefix_restore_owner",
            None,
        )
        if owner is None:
            raise RuntimeError(
                "Qwen3.5 hybrid prefix restore owner is not configured"
            )
        snapshot = owner.snapshot_cache.observation_snapshot()
        if "current_logical_bytes" not in snapshot:
            snapshot = {
                **snapshot,
                "current_logical_bytes": snapshot[
                    "current_full_fidelity_logical_bytes"
                ],
            }
        fields = (
            "current_entries",
            "current_bytes",
            "current_logical_bytes",
            "deduplicated_bytes",
            "peak_entries",
            "peak_bytes",
            "publishes",
            "hits",
            "misses",
            "evictions",
            "validation_failures",
            "failed_restores",
        )
        p2_fields = (
            "current_encoded_physical_bytes",
            "current_encoded_logical_bytes",
            "current_full_fidelity_logical_bytes",
            "current_codec_metadata_bytes",
            "current_reader_leases",
            "current_temporary_encode_workspace_bytes",
            "current_temporary_decode_workspace_bytes",
            "current_temporary_decode_cuda_allocated_bytes",
            "current_temporary_decode_cuda_reserved_bytes",
            "peak_encoded_logical_bytes",
            "peak_full_fidelity_logical_bytes",
            "peak_codec_metadata_bytes",
            "peak_reader_leases",
            "peak_temporary_encode_workspace_bytes",
            "peak_temporary_decode_workspace_bytes",
            "peak_temporary_decode_cuda_allocated_bytes",
            "peak_temporary_decode_cuda_reserved_bytes",
            "deferred_snapshot_releases",
            "quarantines",
            "decode_failures",
            "commit_failures",
            "rollback_failures",
            "fallbacks",
            "partial_restore_attempts",
            "mixed_representation_rejections",
            "missing_layer_rejections",
        )
        if not isinstance(snapshot, dict) or any(
            name not in snapshot for name in fields
        ):
            raise RuntimeError(
                "Qwen3.5 hybrid prefix cache snapshot is incomplete"
            )
        return {
            "rank": int(self.rank),
            "representation": owner.representation,
            "representation_version": owner.representation_version,
            "codec": owner.codec,
            **{
                name: int(snapshot[name])
                for name in fields
            },
            **{
                name: int(snapshot.get(name, 0))
                for name in p2_fields
            },
        }

    def qwen35_hybrid_prefix_authority_snapshot(self):
        owner = getattr(
            self,
            "qwen35_hybrid_prefix_restore_owner",
            None,
        )
        if owner is None:
            raise RuntimeError(
                "Qwen3.5 hybrid prefix restore owner is not configured"
            )
        snapshot = owner.snapshot_cache.observation_snapshot()
        fields = (
            "current_entries",
            "hits",
            "misses",
            "publication_commits",
            "invalidations",
            "clears",
        )
        if not isinstance(snapshot, dict) or any(
            name not in snapshot for name in fields
        ):
            raise RuntimeError(
                "Qwen3.5 hybrid prefix authority snapshot is incomplete"
            )
        participant = owner.publication_participant
        terminal_payloads = getattr(
            participant,
            "_terminal_payloads",
            None,
        )
        if not isinstance(terminal_payloads, dict):
            raise RuntimeError(
                "Qwen3.5 hybrid prefix publication history is unavailable"
            )
        if terminal_payloads:
            last_ticket = max(terminal_payloads)
            payload = terminal_payloads[last_ticket]
            block_identities = getattr(
                payload,
                "block_identities",
                None,
            )
            if not isinstance(block_identities, tuple):
                raise RuntimeError(
                    "Qwen3.5 hybrid prefix publication blocks are invalid"
                )
            normalized_blocks = [
                list(identity)
                for identity in block_identities
            ]
        else:
            normalized_blocks = []
        return {
            "rank": int(self.rank),
            **{
                name: int(snapshot[name])
                for name in fields
            },
            "last_publication_block_identities": normalized_blocks,
        }

    def clear_qwen35_hybrid_prefix_cache(self):
        owner = getattr(
            self,
            "qwen35_hybrid_prefix_restore_owner",
            None,
        )
        if owner is None:
            raise RuntimeError(
                "Qwen3.5 hybrid prefix restore owner is not configured"
            )
        return {
            "rank": int(self.rank),
            "cleared_entries": int(owner.snapshot_cache.clear()),
        }

    def invalidate_qwen35_hybrid_prefix_blocks(
        self,
        block_identities,
    ):
        owner = getattr(
            self,
            "qwen35_hybrid_prefix_restore_owner",
            None,
        )
        if owner is None:
            raise RuntimeError(
                "Qwen3.5 hybrid prefix restore owner is not configured"
            )
        return {
            "rank": int(self.rank),
            "invalidated_entries": int(
                owner.snapshot_cache.invalidate_blocks(
                    block_identities
                )
            ),
        }

    def _qwen35_hybrid_prefix_restore_owner(self):
        participant = getattr(
            self,
            "qwen35_hybrid_prefix_restore_participant",
            None,
        )
        if participant is None:
            raise RuntimeError(
                "Qwen3.5 hybrid prefix restore participant is not installed"
            )
        return participant

    @staticmethod
    def _qwen35_hybrid_prefix_restore_result(
        payload,
        participant_id,
        operation,
        status,
        detail="",
    ):
        return {
            "ticket_id": int(payload.ticket_id),
            "participant_id": int(participant_id),
            "operation": operation,
            "status": status,
            "detail": detail,
        }

    def prepare_hybrid_prefix_restore(self, payload):
        participant = self._qwen35_hybrid_prefix_restore_owner()
        acknowledgement = participant.prepare(payload)
        if acknowledgement.ticket_id != payload.ticket_id:
            raise ValueError(
                "restore acknowledgement ticket id mismatch"
            )
        if acknowledgement.participant_id != participant.participant_id:
            raise ValueError(
                "restore acknowledgement participant id mismatch"
            )
        if acknowledgement.status not in {
            "prepared",
            "miss",
            "error",
        }:
            raise ValueError(
                "restore acknowledgement status is invalid"
            )
        if not isinstance(acknowledgement.detail, str):
            raise ValueError(
                "restore acknowledgement detail must be a string"
            )
        return self._qwen35_hybrid_prefix_restore_result(
            payload,
            acknowledgement.participant_id,
            "prepare",
            acknowledgement.status,
            acknowledgement.detail,
        )

    def validate_hybrid_prefix_restore(self, payload):
        participant = self._qwen35_hybrid_prefix_restore_owner()
        participant.validate_prepared(payload)
        return self._qwen35_hybrid_prefix_restore_result(
            payload,
            participant.participant_id,
            "validate",
            "ok",
        )

    def commit_hybrid_prefix_restore(self, payload):
        participant = self._qwen35_hybrid_prefix_restore_owner()
        participant.commit(payload)
        return self._qwen35_hybrid_prefix_restore_result(
            payload,
            participant.participant_id,
            "commit",
            "ok",
        )

    def rollback_hybrid_prefix_restore(self, payload):
        participant = self._qwen35_hybrid_prefix_restore_owner()
        participant.rollback(payload)
        return self._qwen35_hybrid_prefix_restore_result(
            payload,
            participant.participant_id,
            "rollback",
            "ok",
        )

    def _qwen35_hybrid_prefix_publication_owner(self):
        participant = getattr(
            self,
            "qwen35_hybrid_prefix_publication_participant",
            None,
        )
        if participant is None:
            raise RuntimeError(
                "Qwen3.5 hybrid prefix publication participant "
                "is not installed"
            )
        return participant

    def _qwen35_hybrid_prefix_publication_payload(
        self,
        payload_or_payloads,
    ):
        if not isinstance(payload_or_payloads, tuple):
            return payload_or_payloads
        world_size = getattr(self, "world_size", None)
        rank = getattr(self, "rank", None)
        if (
            isinstance(world_size, bool)
            or not isinstance(world_size, int)
            or world_size <= 0
            or isinstance(rank, bool)
            or not isinstance(rank, int)
            or rank < 0
            or rank >= world_size
            or len(payload_or_payloads) != world_size
        ):
            raise ValueError(
                "publication payload matrix does not match ModelRunner ranks"
            )
        payloads_by_rank = {}
        for payload in payload_or_payloads:
            participant_id = getattr(
                payload,
                "participant_id",
                None,
            )
            if (
                isinstance(participant_id, bool)
                or not isinstance(participant_id, int)
                or participant_id < 0
                or participant_id >= world_size
                or participant_id in payloads_by_rank
            ):
                raise ValueError(
                    "publication payload participant ids are invalid"
                )
            payloads_by_rank[participant_id] = payload
        if tuple(sorted(payloads_by_rank)) != tuple(range(world_size)):
            raise ValueError(
                "publication payload participant coverage is incomplete"
            )
        return payloads_by_rank[rank]

    @staticmethod
    def _qwen35_hybrid_prefix_publication_result(
        payload,
        participant_id,
        operation,
        status,
        detail="",
    ):
        return {
            "ticket_id": int(payload.ticket_id),
            "participant_id": int(participant_id),
            "operation": operation,
            "status": status,
            "detail": detail,
        }

    def _validate_qwen35_hybrid_prefix_publication_ack(
        self,
        payload,
        participant,
        acknowledgement,
        *,
        participant_operation,
        result_operation,
        allowed_statuses,
    ):
        if acknowledgement.ticket_id != payload.ticket_id:
            raise ValueError(
                "publication acknowledgement ticket id mismatch"
            )
        if acknowledgement.participant_id != participant.participant_id:
            raise ValueError(
                "publication acknowledgement participant id mismatch"
            )
        if acknowledgement.operation != participant_operation:
            raise ValueError(
                "publication acknowledgement operation mismatch"
            )
        if acknowledgement.status not in allowed_statuses:
            raise ValueError(
                "publication acknowledgement status is invalid"
            )
        if not isinstance(acknowledgement.detail, str):
            raise ValueError(
                "publication acknowledgement detail must be a string"
            )
        return self._qwen35_hybrid_prefix_publication_result(
            payload,
            acknowledgement.participant_id,
            result_operation,
            acknowledgement.status,
            acknowledgement.detail,
        )

    def prepare_hybrid_prefix_publication(self, payload):
        payload = self._qwen35_hybrid_prefix_publication_payload(
            payload
        )
        participant = self._qwen35_hybrid_prefix_publication_owner()
        acknowledgement = participant.prepare(payload)
        return self._validate_qwen35_hybrid_prefix_publication_ack(
            payload,
            participant,
            acknowledgement,
            participant_operation="prepare",
            result_operation="prepare",
            allowed_statuses={"prepared", "rejected", "error"},
        )

    def precommit_hybrid_prefix_publication(self, payload):
        payload = self._qwen35_hybrid_prefix_publication_payload(
            payload
        )
        participant = self._qwen35_hybrid_prefix_publication_owner()
        acknowledgement = participant.precommit(payload)
        return self._validate_qwen35_hybrid_prefix_publication_ack(
            payload,
            participant,
            acknowledgement,
            participant_operation="precommit",
            result_operation="precommit",
            allowed_statuses={"precommitted", "error"},
        )

    def finalize_hybrid_prefix_publication(self, payload):
        payload = self._qwen35_hybrid_prefix_publication_payload(
            payload
        )
        participant = self._qwen35_hybrid_prefix_publication_owner()
        acknowledgement = participant.commit(payload)
        return self._validate_qwen35_hybrid_prefix_publication_ack(
            payload,
            participant,
            acknowledgement,
            participant_operation="commit",
            result_operation="finalize",
            allowed_statuses={"finalized", "error"},
        )

    def seal_hybrid_prefix_publication(self, payload):
        payload = self._qwen35_hybrid_prefix_publication_payload(
            payload
        )
        participant = self._qwen35_hybrid_prefix_publication_owner()
        acknowledgement = participant.seal(payload)
        return self._validate_qwen35_hybrid_prefix_publication_ack(
            payload,
            participant,
            acknowledgement,
            participant_operation="seal",
            result_operation="seal",
            allowed_statuses={"committed", "error"},
        )

    def rollback_hybrid_prefix_publication(self, payload):
        payload = self._qwen35_hybrid_prefix_publication_payload(
            payload
        )
        participant = self._qwen35_hybrid_prefix_publication_owner()
        acknowledgement = participant.rollback(payload)
        return self._validate_qwen35_hybrid_prefix_publication_ack(
            payload,
            participant,
            acknowledgement,
            participant_operation="rollback",
            result_operation="rollback",
            allowed_statuses={"rolled_back", "error"},
        )

    def warmup_model(self):
        torch.cuda.empty_cache()                                #[thinking]可以看一下源码的执行策略 可能会有优化的点
        torch.cuda.reset_peak_memory_stats()                    # 从新统计GPU内存使用的峰值信息
        max_num_batched_tokens, max_model_len = self.config.max_num_batched_tokens, self.config.max_model_len       #[16384, 4096]
        # num_seqs即batch_size
        num_seqs = min(max_num_batched_tokens // max_model_len, self.config.max_num_seqs)   #min(4,512) 假设每个seq都占满的情况下 batch最大只能有4个seq  这里属于边界条件
        seqs = [Sequence([0] * max_model_len) for _ in range(num_seqs)] #这里warmup是按照极限的边界情况执行的
        self.run(seqs, True)
        torch.cuda.empty_cache()

    def allocate_kv_cache(self):
        config = self.config
        hf_config = config.hf_config
        model_config = getattr(
            hf_config,
            "text_config",
            hf_config,
        )
        # 记一次 weight-only 占用（KV cache 还没分），方便外部观察 TP 是否真切到了 weight
        # 这里走 memory_allocated 而不是 mem_get_info：前者只算本进程 torch alloc，后者算整卡（含别人）
        self.weight_mem_bytes = torch.cuda.memory_allocated()
        free, total = torch.cuda.mem_get_info()
        used = total - free
        peak = torch.cuda.memory_stats()["allocated_bytes.all.peak"]
        current = torch.cuda.memory_stats()["allocated_bytes.all.current"]
        num_kv_heads = _local_model_kv_heads(
            model_config.num_key_value_heads,
            self.world_size,
        )
        head_dim = model_config.head_dim
        dtype = hf_config.torch_dtype
        elem_bytes = dtype.itemsize

        # ----- KV cache 主存储：根据 kv_quant_bits 决定字节数 -----
        # 0: fp/half 原样；4: 每 token 每 head_dim 半字节（按 int8 pack）；8: 每 token 每 head_dim 1 字节
        kvq_bits = config.kv_quant_bits
        if kvq_bits == 0:
            tokens_per_block_bytes = self.block_size * num_kv_heads * head_dim * elem_bytes
            kv_scale_bytes_per_block = 0
        elif kvq_bits == 4:
            assert head_dim % config.kv_quant_group_size == 0, \
                f"head_dim={head_dim} 必须能被 kv_quant_group_size={config.kv_quant_group_size} 整除"
            n_groups_per_token = head_dim // config.kv_quant_group_size
            # 4-bit pack 进 int8：每 byte 存 2 个 4-bit；group_size 偶数保证字节对齐
            packed_bytes = self.block_size * num_kv_heads * (head_dim // 2)
            tokens_per_block_bytes = packed_bytes
            # scale: fp16 / bf16 一份，对称量化只存 scale；非对称额外存 zero（同 dtype）
            scale_count = self.block_size * num_kv_heads * n_groups_per_token
            kv_scale_bytes_per_block = scale_count * elem_bytes * (1 if config.kv_quant_symmetric else 2)
        else:  # 8
            tokens_per_block_bytes = self.block_size * num_kv_heads * head_dim
            n_groups_per_token = max(1, head_dim // config.kv_quant_group_size)
            scale_count = self.block_size * num_kv_heads * n_groups_per_token
            kv_scale_bytes_per_block = scale_count * elem_bytes * (1 if config.kv_quant_symmetric else 2)

        block_bytes = (
            2
            * model_config.num_hidden_layers
            * tokens_per_block_bytes
        )
        kv_scale_bytes_per_block = (
            2
            * model_config.num_hidden_layers
            * kv_scale_bytes_per_block
        )

        # Quest 启用时还需要预留 per-block K min/max summary 显存（每块 2*num_kv_heads*head_dim 元素）
        quest_enabled = config.quest_top_k_blocks > 0
        summary_bytes = (2 * model_config.num_hidden_layers * num_kv_heads *
                         head_dim * elem_bytes) if quest_enabled else 0

        per_block = block_bytes + kv_scale_bytes_per_block + summary_bytes
        auto_num_blocks = int(total * config.gpu_memory_utilization - used - (peak - current)) // per_block
        assert auto_num_blocks > 0

        if config.kv_offload_mvp0:
            gpu_nb = config.kv_offload_gpu_blocks or auto_num_blocks
            logical_nb = config.kv_offload_logical_blocks or max(auto_num_blocks, gpu_nb)
            assert gpu_nb > 0
            assert logical_nb >= gpu_nb, "kv_offload_logical_blocks 必须 >= kv_offload_gpu_blocks"
            config.num_kvcache_blocks = logical_nb
            nb = gpu_nb
            self._physical_num_kvcache_blocks = gpu_nb
            self._exact_graph_scratch_block_ids = ()
            self._spec_verify_capture_scratch_block_ids = ()
            self.spec_verify_capture_scratch_pool = None
        else:
            decode_scratch_blocks = (
                max(config.multi_sequence_cuda_graph_batch_allowlist)
                if config.multi_sequence_cuda_graphs
                else 0
            )
            spec_verify_scratch_blocks = (
                required_spec_verify_capture_scratch_blocks(
                    batch_allowlist=(
                        config.spec_verify_cuda_graph_batch_allowlist
                    ),
                    query_len_allowlist=(
                        config.spec_verify_cuda_graph_query_len_allowlist
                    ),
                    block_size=self.block_size,
                )
                if config.spec_verify_cuda_graphs
                else 0
            )
            total_scratch_blocks = (
                decode_scratch_blocks
                + spec_verify_scratch_blocks
            )
            visible_blocks, physical_blocks = (
                resolve_exact_graph_kv_capacity(
                    auto_blocks=auto_num_blocks,
                    requested_visible_blocks=config.num_kvcache_blocks,
                    feature_enabled=total_scratch_blocks > 0,
                    scratch_blocks=total_scratch_blocks,
                )
            )
            config.num_kvcache_blocks = visible_blocks
            self._physical_num_kvcache_blocks = physical_blocks
            (
                self._exact_graph_scratch_block_ids,
                self._spec_verify_capture_scratch_block_ids,
            ) = partition_exact_graph_scratch_block_ids(
                visible_blocks=visible_blocks,
                decode_scratch_blocks=decode_scratch_blocks,
                spec_verify_scratch_blocks=(
                    spec_verify_scratch_blocks
                ),
            )
            self.spec_verify_capture_scratch_pool = (
                SpecVerifyCaptureScratchPool(
                    block_ids=(
                        self._spec_verify_capture_scratch_block_ids
                    ),
                    block_size=self.block_size,
                )
                if self._spec_verify_capture_scratch_block_ids
                else None
            )
            nb = physical_blocks
        L = model_config.num_hidden_layers
        if kvq_bits == 0:
            self.kv_cache = torch.zeros(2, L, nb, self.block_size, num_kv_heads, head_dim, dtype=dtype)
            self.kv_scale = None
            self.kv_zero = None
        elif kvq_bits == 4:
            # int8 pack 后，沿最后一维 head_dim/2
            self.kv_cache = torch.zeros(2, L, nb, self.block_size, num_kv_heads, head_dim // 2, dtype=torch.int8)
            n_groups = head_dim // config.kv_quant_group_size
            self.kv_scale = torch.zeros(2, L, nb, self.block_size, num_kv_heads, n_groups, dtype=dtype)
            self.kv_zero = (None if config.kv_quant_symmetric else
                            torch.zeros(2, L, nb, self.block_size, num_kv_heads, n_groups, dtype=dtype))
        else:  # 8
            self.kv_cache = torch.zeros(2, L, nb, self.block_size, num_kv_heads, head_dim, dtype=torch.int8)
            n_groups = max(1, head_dim // config.kv_quant_group_size)
            self.kv_scale = torch.zeros(2, L, nb, self.block_size, num_kv_heads, n_groups, dtype=dtype)
            self.kv_zero = (None if config.kv_quant_symmetric else
                            torch.zeros(2, L, nb, self.block_size, num_kv_heads, n_groups, dtype=dtype))

        # Quest summary：[2, num_layers, num_blocks, num_kv_heads, head_dim]，dim0 = (min, max)
        if quest_enabled:
            self.kv_summary = torch.empty(2, L, nb, num_kv_heads, head_dim, dtype=torch.float32)
            # 用 +inf / -inf 作为 min/max 的初始值，确保第一次 token 写入后被替换
            self.kv_summary[0].fill_(float("inf"))
            self.kv_summary[1].fill_(float("-inf"))
        else:
            self.kv_summary = None

        if config.kv_offload_mvp0:
            self.kv_offload = KVOffloadMVP0(
                self.kv_cache,
                config.num_kvcache_blocks,
                self.block_size,
                async_copy=config.kv_offload_async_copy,
                batch_copy=config.kv_offload_batch_copy,
                writeback_on_evict=config.kv_offload_writeback_on_evict,
                evict_policy=config.kv_offload_evict_policy,
            )

        layer_id = 0
        for module in self.model.modules():
            if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
                module.k_cache = self.kv_cache[0, layer_id]
                module.v_cache = self.kv_cache[1, layer_id]
                # 把量化辅助张量也挂上去；非量化时为 None
                if self.kv_scale is not None:
                    module.k_scale = self.kv_scale[0, layer_id]
                    module.v_scale = self.kv_scale[1, layer_id]
                else:
                    module.k_scale = module.v_scale = None
                if self.kv_zero is not None:
                    module.k_zero = self.kv_zero[0, layer_id]
                    module.v_zero = self.kv_zero[1, layer_id]
                else:
                    module.k_zero = module.v_zero = None
                module.kv_quant_bits = kvq_bits
                module.kv_quant_group_size = config.kv_quant_group_size
                module.kv_quant_symmetric = config.kv_quant_symmetric
                module.layer_idx = layer_id
                module.num_hidden_layers = L
                if quest_enabled:
                    module.k_min = self.kv_summary[0, layer_id]
                    module.k_max = self.kv_summary[1, layer_id]
                layer_id += 1

    def _exact_graph_scratch_slots(
        self,
        *,
        batch_size: int,
    ) -> tuple[int, ...]:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if batch_size > len(self._exact_graph_scratch_block_ids):
            raise ValueError(
                "exact graph scratch capacity is smaller than batch size"
            )
        return tuple(
            block_id * self.block_size
            for block_id in self._exact_graph_scratch_block_ids[:batch_size]
        )

    def _build_multi_sequence_graph_identity(
        self,
        input_ids,
        context,
    ):
        active_batch_size = int(input_ids.size(0))
        if active_batch_size <= 1:
            raise ValueError(
                "multi-sequence graph identity requires batch above one"
            )
        try:
            page_table_width = int(context.block_tables.size(1))
        except (AttributeError, IndexError, TypeError, ValueError) as exc:
            raise ValueError(
                "invalid exact graph block-table width"
            ) from exc
        hf_config = self.config.hf_config
        properties = torch.cuda.get_device_properties(
            self.kv_cache.device
        )
        inputs = FlashAttentionSplitInputs(
            batch_size=active_batch_size,
            num_query_heads=int(
                hf_config.num_attention_heads // self.world_size
            ),
            num_kv_heads=int(
                hf_config.num_key_value_heads // self.world_size
            ),
            head_dim=int(hf_config.head_dim),
            page_block_size=int(self.block_size),
            page_table_width=page_table_width,
            max_seqlen_q=1,
            multi_processor_count=int(
                properties.multi_processor_count
            ),
        )
        return build_flash_attn_263_graph_identity(
            graph_batch_size=active_batch_size,
            inputs=inputs,
            flash_attn_version=str(
                getattr(flash_attn, "__version__", "unknown")
            ),
            require_exact_batch=True,
        )

    def _spec_verify_graph_incompatible_reason(
        self,
        *,
        input_ids,
        input_embeds,
        return_hidden: bool,
        context,
        transaction_authorized: bool,
    ) -> tuple[str | None, bool]:
        del input_ids
        if not getattr(
            self.config,
            "spec_verify_cuda_graphs",
            False,
        ):
            return "feature_disabled", True
        if self.enforce_eager:
            return "enforce_eager", True
        if getattr(context, "mode", None) != "spec_verify":
            return "unsupported_mode", True
        if self.world_size != 1:
            return "tp_not_one", True
        if getattr(self.config, "kv_offload_mvp0", False):
            return "kv_offload_enabled", True
        if (
            getattr(
                self.config,
                "kv_offload_blockwise_decode",
                False,
            )
            or getattr(
                self.config,
                "kv_offload_blockwise_prefill",
                False,
            )
        ):
            return "blockwise_enabled", True
        query_lens = tuple(
            int(value)
            for value in getattr(
                context,
                "spec_verify_query_lens",
                (),
            )
        )
        active_batch_size = len(query_lens)
        if active_batch_size not in tuple(
            getattr(
                self.config,
                "spec_verify_cuda_graph_batch_allowlist",
                (),
            )
        ):
            return "batch_not_allowlisted", True
        query_len = (
            query_lens[0]
            if query_lens
            and all(value == query_lens[0] for value in query_lens)
            else None
        )
        if query_len not in tuple(
            getattr(
                self.config,
                "spec_verify_cuda_graph_query_len_allowlist",
                (),
            )
        ):
            return "query_len_not_allowlisted", True
        if input_embeds is not None:
            return "input_embeds_active", True
        if return_hidden:
            return "hidden_state_return_active", True
        if self.hybrid_state_runtime_bridge is not None:
            return "non_transactional_state", False
        if not transaction_authorized:
            return "transaction_unauthorized", False
        return None, True

    def _build_spec_verify_graph_identity(
        self,
        *,
        input_ids,
        outputs,
        context,
    ) -> SpecVerifyGraphIdentity:
        query_lens = tuple(
            int(value)
            for value in getattr(
                context,
                "spec_verify_query_lens",
                (),
            )
        )
        active_batch_size = len(query_lens)
        if (
            active_batch_size <= 0
            or any(value <= 0 for value in query_lens)
            or any(value != query_lens[0] for value in query_lens)
        ):
            raise ValueError(
                "spec-verify graph requires one exact positive query shape"
            )
        query_len = query_lens[0]
        total_query_tokens = active_batch_size * query_len
        try:
            input_shape = tuple(input_ids.size())
            slot_mapping_shape = tuple(
                context.slot_mapping.size()
            )
            context_lens_shape = tuple(
                context.context_lens.size()
            )
            block_tables_shape = tuple(
                context.block_tables.size()
            )
            outputs_shape = tuple(outputs.size())
        except (AttributeError, TypeError, ValueError) as exc:
            raise ValueError(
                "invalid spec-verify graph tensor shape"
            ) from exc
        if input_shape != (total_query_tokens,):
            raise ValueError("input_ids shape drift")
        if slot_mapping_shape != (total_query_tokens,):
            raise ValueError("slot_mapping shape drift")
        if context_lens_shape != (active_batch_size,):
            raise ValueError("context_lens shape drift")
        if (
            len(block_tables_shape) != 2
            or block_tables_shape[0] != active_batch_size
            or block_tables_shape[1] <= 0
        ):
            raise ValueError("block_tables shape drift")
        hf_config = self.config.hf_config
        num_query_heads = int(hf_config.num_attention_heads)
        num_kv_heads = int(hf_config.num_key_value_heads)
        head_dim = int(hf_config.head_dim)
        hidden_size = int(
            getattr(
                hf_config,
                "hidden_size",
                num_query_heads * head_dim,
            )
        )
        if hidden_size != num_query_heads * head_dim:
            raise ValueError(
                "spec-verify graph output width is not represented "
                "by the graph identity"
            )
        if outputs_shape != (total_query_tokens, hidden_size):
            raise ValueError("outputs shape drift")
        try:
            input_numel = int(input_ids.numel())
        except (AttributeError, TypeError, ValueError) as exc:
            raise ValueError(
                "invalid input_ids shape metadata"
            ) from exc
        if input_numel != total_query_tokens:
            raise ValueError("input_ids shape drift")
        flash_attn_num_splits = int(
            context.flash_attn_num_splits
        )
        return SpecVerifyGraphIdentity(
            active_batch_size=active_batch_size,
            query_len=query_len,
            total_query_tokens=total_query_tokens,
            page_table_width=int(block_tables_shape[1]),
            flash_attn_num_splits=flash_attn_num_splits,
            attention_backend="flash_attn",
            attention_backend_version=str(
                getattr(flash_attn, "__version__", "unknown")
            ),
            input_dtype=str(input_ids.dtype),
            output_dtype=str(outputs.dtype),
            num_query_heads=num_query_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            page_block_size=int(self.block_size),
            device_compute_capability=tuple(
                int(value)
                for value in torch.cuda.get_device_capability(
                    self.kv_cache.device
                )
            ),
        )

    def _estimate_spec_verify_graph_static_bytes(
        self,
        identity: SpecVerifyGraphIdentity,
    ) -> int:
        if not isinstance(identity, SpecVerifyGraphIdentity):
            raise ValueError(
                "identity must be SpecVerifyGraphIdentity"
            )
        dtype_bytes = {
            "torch.int32": 4,
            "torch.int64": 8,
            "torch.float16": 2,
            "torch.bfloat16": 2,
            "torch.float32": 4,
            "torch.float64": 8,
        }
        try:
            input_element_size = dtype_bytes[
                identity.input_dtype
            ]
            output_element_size = dtype_bytes[
                identity.output_dtype
            ]
        except KeyError as exc:
            raise ValueError(
                "unsupported spec-verify graph dtype"
            ) from exc
        total_query_tokens = identity.total_query_tokens
        active_batch_size = identity.active_batch_size
        output_width = (
            identity.num_query_heads * identity.head_dim
        )
        return (
            total_query_tokens * input_element_size
            + total_query_tokens * dtype_bytes["torch.int32"]
            + active_batch_size * dtype_bytes["torch.int32"]
            + (
                active_batch_size
                * identity.page_table_width
                * dtype_bytes["torch.int32"]
            )
            + (
                total_query_tokens
                * output_width
                * output_element_size
            )
        )

    def _capture_spec_verify_graph(
        self,
        *,
        identity: SpecVerifyGraphIdentity,
        live_input_ids,
        live_positions,
        live_context,
    ) -> SpecVerifyExactCudaGraphEntry:
        if not isinstance(identity, SpecVerifyGraphIdentity):
            raise ValueError(
                "identity must be SpecVerifyGraphIdentity"
            )
        pool = getattr(
            self,
            "spec_verify_capture_scratch_pool",
            None,
        )
        if pool is None:
            raise RuntimeError("scratch_unavailable")
        row_offsets = self._spec_verify_capture_row_offsets(
            identity=identity,
            live_context=live_context,
        )
        graph = torch.cuda.CUDAGraph()
        device = live_input_ids.device
        tensors = {
            "input_ids": torch.zeros(
                identity.total_query_tokens,
                dtype=live_input_ids.dtype,
                device=device,
            ),
            "positions": torch.zeros(
                identity.total_query_tokens,
                dtype=live_positions.dtype,
                device=device,
            ),
            "slot_mapping": torch.zeros(
                identity.total_query_tokens,
                dtype=live_context.slot_mapping.dtype,
                device=device,
            ),
            "context_lens": torch.zeros(
                identity.active_batch_size,
                dtype=live_context.context_lens.dtype,
                device=device,
            ),
            "block_tables": torch.zeros(
                identity.active_batch_size,
                identity.page_table_width,
                dtype=live_context.block_tables.dtype,
                device=device,
            ),
            "outputs": torch.zeros(
                identity.total_query_tokens,
                identity.num_query_heads * identity.head_dim,
                dtype=self.config.hf_config.torch_dtype,
                device=device,
            ),
        }
        tensors["input_ids"].copy_(live_input_ids)
        tensors["positions"].copy_(live_positions)
        tensors["context_lens"].copy_(live_context.context_lens)
        lease = pool.acquire(
            active_batch_size=identity.active_batch_size,
            query_len=identity.query_len,
            row_offsets=row_offsets,
        )
        row_block_counts = tuple(lease.row_block_counts)
        expected_blocks = sum(row_block_counts)
        if len(row_block_counts) != identity.active_batch_size:
            pool.rollback(lease)
            raise RuntimeError("scratch_unavailable")
        if len(lease.block_ids) != expected_blocks:
            pool.rollback(lease)
            raise RuntimeError("scratch_unavailable")
        try:
            live_block_rows = live_context.block_tables.tolist()
            live_context_lens = tuple(
                int(value)
                for value in live_context.context_lens.tolist()
            )
            if len(live_block_rows) != identity.active_batch_size:
                raise ValueError("capture block-table shape drift")
            scratch_rows = []
            scratch_slots = []
            clone_pairs = []
            lease_offset = 0
            for row_index, live_row in enumerate(live_block_rows):
                row = [int(block_id) for block_id in live_row]
                if len(row) != identity.page_table_width:
                    raise ValueError(
                        "capture block-table shape drift"
                    )
                row_block_count = row_block_counts[row_index]
                row_blocks = lease.block_ids[
                    lease_offset:
                    lease_offset + row_block_count
                ]
                lease_offset += row_block_count
                prefix_len = (
                    live_context_lens[row_index]
                    - identity.query_len
                )
                write_block_index = prefix_len // self.block_size
                if (
                    row_block_count <= 0
                    or write_block_index + row_block_count > len(row)
                ):
                    raise ValueError(
                        "capture scratch rows exceed page-table width"
                    )
                if row_offsets[row_index] != 0:
                    clone_pairs.append(
                        (
                            row[write_block_index],
                            row_blocks[0],
                        )
                    )
                row[
                    write_block_index:
                    write_block_index + row_block_count
                ] = row_blocks
                scratch_rows.append(row)
                scratch_slots.extend(
                    row_blocks[
                        (
                            row_offsets[row_index]
                            + token_index
                        )
                        // self.block_size
                    ]
                    * self.block_size
                    + (
                        row_offsets[row_index] + token_index
                    )
                    % self.block_size
                    for token_index in range(identity.query_len)
                )
            self._clone_spec_verify_capture_prefix_blocks(
                tuple(clone_pairs)
            )
            tensors["slot_mapping"].copy_(
                torch.tensor(
                    scratch_slots,
                    dtype=live_context.slot_mapping.dtype,
                    device=device,
                )
            )
            tensors["block_tables"].copy_(
                torch.tensor(
                    scratch_rows,
                    dtype=live_context.block_tables.dtype,
                    device=device,
                )
            )
            static_bytes = sum(
                int(tensor.numel() * tensor.element_size())
                for tensor in tensors.values()
            )
            allocated_before = int(torch.cuda.memory_allocated())
            reserved_before = int(torch.cuda.memory_reserved())
            capture_started_ns = time.perf_counter_ns()
            set_context(
                mode="spec_verify",
                slot_mapping=tensors["slot_mapping"],
                context_lens=tensors["context_lens"],
                block_tables=tensors["block_tables"],
                spec_verify_query_lens=(
                    (identity.query_len,)
                    * identity.active_batch_size
                ),
                flash_attn_num_splits=(
                    identity.flash_attn_num_splits
                ),
            )
            with torch.cuda.graph(
                graph,
                pool=getattr(self, "graph_pool", None),
            ):
                tensors["outputs"].copy_(
                    self.model(
                        tensors["input_ids"],
                        tensors["positions"],
                    )
                )
            torch.cuda.synchronize()
            capture_duration_ns = (
                time.perf_counter_ns() - capture_started_ns
            )
            allocated_after = int(torch.cuda.memory_allocated())
            reserved_after = int(torch.cuda.memory_reserved())
        finally:
            reset_context()
            pool.rollback(lease)
        return SpecVerifyExactCudaGraphEntry(
            identity=identity,
            identity_sha256=identity.sha256,
            graph=graph,
            tensors=tensors,
            static_bytes=static_bytes,
            capture_duration_ns=capture_duration_ns,
            allocated_delta_bytes=max(
                0,
                allocated_after - allocated_before,
            ),
            reserved_delta_bytes=max(
                0,
                reserved_after - reserved_before,
            ),
        )

    def _spec_verify_capture_row_offsets(
        self,
        *,
        identity,
        live_context,
    ) -> tuple[int, ...]:
        try:
            context_lens = tuple(
                int(value)
                for value in live_context.context_lens.tolist()
            )
        except (AttributeError, TypeError, ValueError) as exc:
            raise ValueError(
                "invalid spec-verify capture context lengths"
            ) from exc
        if len(context_lens) != identity.active_batch_size:
            raise ValueError(
                "spec-verify capture context length shape drift"
            )
        offsets = []
        for context_len in context_lens:
            prefix_len = context_len - identity.query_len
            if prefix_len < 0:
                raise ValueError(
                    "spec-verify capture context is shorter than Q"
                )
            offset = prefix_len % self.block_size
            offsets.append(offset)
        return tuple(offsets)

    def _clone_spec_verify_capture_prefix_blocks(
        self,
        clone_pairs: tuple[tuple[int, int], ...],
    ) -> None:
        for source_block_id, destination_block_id in clone_pairs:
            self.kv_cache[
                :,
                :,
                destination_block_id,
            ].copy_(
                self.kv_cache[
                    :,
                    :,
                    source_block_id,
                ]
            )
            for tensor_name in (
                "kv_scale",
                "kv_zero",
                "kv_summary",
            ):
                tensor = getattr(self, tensor_name, None)
                if tensor is None:
                    continue
                tensor[
                    :,
                    :,
                    destination_block_id,
                ].copy_(
                    tensor[
                        :,
                        :,
                        source_block_id,
                    ]
                )

    def _attempt_post_step_spec_verify_capture(
        self,
        *,
        identity: SpecVerifyGraphIdentity,
        live_input_ids,
        live_positions,
        live_context,
    ) -> SpecVerifyExactCudaGraphEntry | None:
        try:
            entry = self._capture_spec_verify_graph(
                identity=identity,
                live_input_ids=live_input_ids,
                live_positions=live_positions,
                live_context=live_context,
            )
        except Exception:
            self.spec_verify_exact_cuda_graph_cache.quarantine(
                identity,
                "capture_failed",
            )
            return None
        self.spec_verify_exact_cuda_graph_cache.commit_capture(entry)
        if entry.state != "ready":
            return None
        return entry

    def _replay_spec_verify_graph(
        self,
        entry: SpecVerifyExactCudaGraphEntry,
        *,
        input_ids,
        positions,
        context,
    ):
        if not isinstance(entry, SpecVerifyExactCudaGraphEntry):
            raise ValueError(
                "entry must be SpecVerifyExactCudaGraphEntry"
            )
        if not getattr(
            self,
            "_spec_verify_transaction_authorized",
            False,
        ):
            raise RuntimeError("transaction_unauthorized")
        cache = self.spec_verify_exact_cuda_graph_cache
        if entry.identity_sha256 != entry.identity.sha256:
            cache.quarantine(
                entry.identity,
                "identity_drift",
            )
            raise RuntimeError(
                "spec-verify CUDA Graph identity drift"
            )
        identity = self._build_spec_verify_graph_identity(
            input_ids=input_ids,
            outputs=entry.tensors["outputs"],
            context=context,
        )
        if entry.identity != identity:
            cache.quarantine(
                entry.identity,
                "identity_drift",
            )
            raise RuntimeError(
                "spec-verify CUDA Graph identity drift"
            )
        if (
            entry.state != "ready"
            or cache.ready_entries.get(
                entry.identity_sha256
            ) is not entry
        ):
            raise RuntimeError(
                "spec-verify CUDA Graph cache state drift"
            )
        if not self._spec_verify_replay_tensor_shapes_match(
            entry=entry,
            input_ids=input_ids,
            positions=positions,
            context=context,
        ):
            cache.quarantine(
                entry.identity,
                "shape_drift",
            )
            raise RuntimeError(
                "spec-verify CUDA Graph shape drift"
            )
        tensors = entry.tensors
        tensors["input_ids"].copy_(input_ids)
        tensors["positions"].copy_(positions)
        tensors["slot_mapping"].copy_(context.slot_mapping)
        tensors["context_lens"].copy_(context.context_lens)
        tensors["block_tables"].copy_(context.block_tables)
        step_id = (
            getattr(
                self,
                "_spec_verify_cuda_graph_step_id",
                0,
            )
            + 1
        )
        replay_succeeded = False
        replay_started = False
        cache.begin_replay(
            entry,
            step_id=step_id,
        )
        try:
            set_context(
                mode="spec_verify",
                slot_mapping=tensors["slot_mapping"],
                context_lens=tensors["context_lens"],
                block_tables=tensors["block_tables"],
                spec_verify_query_lens=(
                    (identity.query_len,)
                    * identity.active_batch_size
                ),
                flash_attn_num_splits=(
                    identity.flash_attn_num_splits
                ),
            )
            replay_started = True
            entry.graph.replay()
            self._synchronize_spec_verify_graph_replay()
            replay_succeeded = True
            return tensors["outputs"]
        except BaseException as error:
            if replay_started:
                cache.quarantine(
                    entry.identity,
                    "replay_failed",
                )
                raise SpecVerifyGraphReplayError(
                    entry.identity_sha256,
                    error,
                ) from error
            raise
        finally:
            reset_context()
            if entry.in_flight_replays:
                cache.finish_replay(
                    entry,
                    step_id=step_id,
                    succeeded=replay_succeeded,
                )

    def _spec_verify_replay_tensor_shapes_match(
        self,
        *,
        entry,
        input_ids,
        positions,
        context,
    ) -> bool:
        live_tensors = {
            "input_ids": input_ids,
            "positions": positions,
            "slot_mapping": context.slot_mapping,
            "context_lens": context.context_lens,
            "block_tables": context.block_tables,
        }
        for name, live_tensor in live_tensors.items():
            static_tensor = entry.tensors.get(name)
            if static_tensor is None:
                return False
            if (
                tuple(static_tensor.shape)
                != tuple(live_tensor.shape)
                or static_tensor.dtype != live_tensor.dtype
                or str(static_tensor.device)
                != str(live_tensor.device)
            ):
                return False
        return True

    def _synchronize_spec_verify_graph_replay(self) -> None:
        return None

    def _ready_spec_verify_graph_entry(
        self,
        *,
        input_ids,
        context,
    ) -> SpecVerifyExactCudaGraphEntry | None:
        for entry in tuple(
            self.spec_verify_exact_cuda_graph_cache.ready_entries.values()
        ):
            try:
                identity = self._build_spec_verify_graph_identity(
                    input_ids=input_ids,
                    outputs=entry.tensors["outputs"],
                    context=context,
                )
            except ValueError:
                continue
            if identity == entry.identity:
                return (
                    self.spec_verify_exact_cuda_graph_cache.ready_entry(
                        identity
                    )
                )
        return None

    def _publish_spec_verify_graph_dispatch_event(
        self,
        *,
        identity: SpecVerifyGraphIdentity | None,
        dispatch: str,
        decision: str,
        fallback_reason: str | None,
        cache_state: str,
        observation_count: int,
        capture_attempted: bool,
        capture_entry=None,
        transaction_authorized: bool,
    ) -> None:
        self._spec_verify_cuda_graph_step_id = (
            getattr(
                self,
                "_spec_verify_cuda_graph_step_id",
                0,
            )
            + 1
        )
        cache = getattr(
            self,
            "spec_verify_exact_cuda_graph_cache",
            None,
        )
        summary = (
            cache.summary()
            if cache is not None
            else {
                "ready_entries": (),
                "static_bytes": 0,
                "reserved_delta_bytes": 0,
                "total_capture_ns": 0,
                "hits": 0,
                "misses": 0,
                "evictions": 0,
                "quarantines": 0,
            }
        )
        event = {
            "step_id": self._spec_verify_cuda_graph_step_id,
            "request_ids_hash": getattr(
                self,
                "_spec_verify_cuda_graph_request_ids_hash",
                hashlib.sha256(b"[]").hexdigest(),
            ),
            "mode": "spec_verify",
            "active_batch_size": (
                None
                if identity is None
                else identity.active_batch_size
            ),
            "query_len": (
                None if identity is None else identity.query_len
            ),
            "total_query_tokens": (
                None
                if identity is None
                else identity.total_query_tokens
            ),
            "page_table_width": (
                None
                if identity is None
                else identity.page_table_width
            ),
            "flash_attn_num_splits": (
                None
                if identity is None
                else identity.flash_attn_num_splits
            ),
            "graph_identity_sha256": (
                None if identity is None else identity.sha256
            ),
            "feature_enabled": bool(
                getattr(
                    self.config,
                    "spec_verify_cuda_graphs",
                    False,
                )
            ),
            "dispatch": dispatch,
            "decision": decision,
            "fallback_reason": fallback_reason,
            "cache_state": cache_state,
            "observation_count": int(observation_count),
            "capture_attempted": bool(capture_attempted),
            "capture_duration_ns": (
                0
                if capture_entry is None
                else int(capture_entry.capture_duration_ns)
            ),
            "capture_static_bytes": (
                0
                if capture_entry is None
                else int(capture_entry.static_bytes)
            ),
            "capture_allocated_delta_bytes": (
                0
                if capture_entry is None
                else int(capture_entry.allocated_delta_bytes)
            ),
            "capture_reserved_delta_bytes": (
                0
                if capture_entry is None
                else int(capture_entry.reserved_delta_bytes)
            ),
            "cache_ready_entries": len(summary["ready_entries"]),
            "cache_static_bytes": int(summary["static_bytes"]),
            "cache_reserved_delta_bytes": int(
                summary["reserved_delta_bytes"]
            ),
            "cache_total_capture_ns": int(
                summary["total_capture_ns"]
            ),
            "cache_hits": int(summary["hits"]),
            "cache_misses": int(summary["misses"]),
            "cache_evictions": int(summary["evictions"]),
            "cache_quarantines": int(summary["quarantines"]),
            "transaction_authorized": bool(
                transaction_authorized
            ),
            "source_sha256": os.environ.get(
                "TINYVLLM_SOURCE_SHA256",
                "",
            ),
        }
        if tuple(event) != SPEC_VERIFY_DISPATCH_EVENT_FIELDS:
            raise RuntimeError(
                "spec-verify CUDA Graph dispatch event schema drift"
            )
        self.last_spec_verify_cuda_graph_dispatch_event = event

    def spec_verify_graph_dispatch_observation(
        self,
    ) -> dict[str, object] | None:
        event = getattr(
            self,
            "last_spec_verify_cuda_graph_dispatch_event",
            None,
        )
        return None if event is None else dict(event)

    def _estimate_exact_graph_static_bytes(
        self,
        *,
        batch_size: int,
        page_table_width: int,
    ) -> int:
        if batch_size <= 0 or page_table_width <= 0:
            raise ValueError(
                "batch size and page-table width must be positive"
            )
        hf_config = self.config.hf_config
        scalar_bytes = batch_size * (8 + 8 + 4 + 4)
        block_table_bytes = batch_size * page_table_width * 4
        output_bytes = (
            batch_size
            * int(hf_config.hidden_size)
            * int(hf_config.torch_dtype.itemsize)
        )
        return scalar_bytes + block_table_bytes + output_bytes

    def _multi_sequence_graph_incompatible_reason(
        self,
        *,
        mode: str,
        is_prefill: bool,
        input_embeds,
        return_hidden: bool,
    ) -> str | None:
        if not getattr(
            self.config,
            "multi_sequence_cuda_graphs",
            False,
        ):
            return "feature_disabled"
        if self.enforce_eager:
            return "enforce_eager"
        if is_prefill or mode != "decode":
            return "unsupported_mode"
        context = get_context()
        if (
            context.quest_top_k_blocks > 0
            or context.am_compact_blocks > 0
            or self.config.kv_quant_bits == 4
            or self.config.cpu_offload
            or self.config.kv_offload_mvp0
            or input_embeds is not None
            or return_hidden
        ):
            return "incompatible_feature"
        return None

    def _run_eager_logits(
        self,
        *,
        input_ids,
        positions,
        input_embeds=None,
        return_hidden: bool = False,
        prepare_qwen35_state: bool = False,
        initial_qwen35_candidates=None,
        capture_qwen35_prefix_states: bool = False,
    ):
        return _run_model_runner_eager(
            self.model,
            input_ids=input_ids,
            positions=positions,
            input_embeds=input_embeds,
            active_leases=getattr(
                self,
                "_last_hybrid_state_leases",
                (),
            ),
            token_counts=getattr(
                self,
                "_last_hybrid_state_token_counts",
                (),
            ),
            return_hidden=return_hidden,
            prepare_qwen35_state=prepare_qwen35_state,
            initial_qwen35_candidates=initial_qwen35_candidates,
            capture_qwen35_prefix_states=(
                capture_qwen35_prefix_states
            ),
        )

    def _publish_cuda_graph_dispatch_event(
        self,
        *,
        mode: str,
        active_batch_size: int,
        page_table_width: int | None,
        effective_num_splits: int | None,
        graph_identity_sha256: str | None,
        dispatch: str,
        cache_state: str,
        observation_count: int,
        fallback_reason: str | None,
        capture_attempted: bool,
        capture_entry=None,
    ) -> None:
        self._cuda_graph_step_id = (
            getattr(self, "_cuda_graph_step_id", 0) + 1
        )
        summary = self.exact_cuda_graph_cache.summary()
        event = {
            "step_id": self._cuda_graph_step_id,
            "request_ids_hash": getattr(
                self,
                "_cuda_graph_request_ids_hash",
                hashlib.sha256(b"[]").hexdigest(),
            ),
            "mode": mode,
            "active_batch_size": active_batch_size,
            "page_table_width": page_table_width,
            "effective_num_splits": effective_num_splits,
            "graph_identity_sha256": graph_identity_sha256,
            "feature_enabled": bool(
                getattr(
                    self.config,
                    "multi_sequence_cuda_graphs",
                    False,
                )
            ),
            "dispatch": dispatch,
            "cache_state": cache_state,
            "observation_count": observation_count,
            "fallback_reason": fallback_reason,
            "capture_attempted": capture_attempted,
            "capture_duration_ns": (
                0
                if capture_entry is None
                else int(capture_entry.capture_duration_ns)
            ),
            "capture_static_bytes": (
                0
                if capture_entry is None
                else int(capture_entry.static_bytes)
            ),
            "capture_allocated_delta_bytes": (
                0
                if capture_entry is None
                else int(capture_entry.allocated_delta_bytes)
            ),
            "capture_reserved_delta_bytes": (
                0
                if capture_entry is None
                else int(capture_entry.reserved_delta_bytes)
            ),
            "cache_ready_entries": len(summary["ready_entries"]),
            "cache_static_bytes": summary["static_bytes"],
            "cache_reserved_delta_bytes": summary[
                "reserved_delta_bytes"
            ],
            "cache_total_capture_ns": summary["total_capture_ns"],
            "source_sha256": os.environ.get(
                "TINYVLLM_SOURCE_SHA256",
                "",
            ),
        }
        if tuple(event) != DISPATCH_EVENT_FIELDS:
            raise RuntimeError("CUDA Graph dispatch event schema drift")
        self.last_cuda_graph_dispatch_event = event

    def cuda_graph_dispatch_observation(self) -> dict | None:
        event = getattr(
            self,
            "last_cuda_graph_dispatch_event",
            None,
        )
        return None if event is None else dict(event)

    def _attempt_post_step_capture(
        self,
        *,
        identity,
        input_ids,
        positions,
        context,
    ):
        try:
            entry = self._capture_exact_multi_sequence_graph(
                identity=identity,
                input_ids=input_ids,
                positions=positions,
                context=context,
            )
        except _ExactGraphCaptureError as exc:
            self.exact_cuda_graph_cache.reject(
                identity,
                exc.reason,
                retained_reserved_bytes=(
                    exc.retained_reserved_bytes
                ),
            )
            return None
        except Exception:
            self.exact_cuda_graph_cache.reject(
                identity,
                "capture_failed",
            )
            return None
        self.exact_cuda_graph_cache.commit_capture(entry)
        return entry

    def _replay_exact_multi_sequence_graph(
        self,
        entry,
        *,
        input_ids,
        positions,
        context,
    ):
        identity = self._build_multi_sequence_graph_identity(
            input_ids,
            context,
        )
        if (
            entry.identity != identity
            or entry.identity_sha256 != identity.sha256
        ):
            self.exact_cuda_graph_cache.disable_entry(
                entry.identity_sha256,
                "identity_drift",
            )
            raise RuntimeError("exact CUDA Graph identity drift")
        tensors = entry.tensors
        try:
            tensors["input_ids"].copy_(input_ids)
            tensors["positions"].copy_(positions)
            tensors["slot_mapping"].copy_(context.slot_mapping)
            tensors["context_lens"].copy_(context.context_lens)
            tensors["block_tables"].copy_(context.block_tables)
            set_context(
                False,
                slot_mapping=tensors["slot_mapping"],
                context_lens=tensors["context_lens"],
                block_tables=tensors["block_tables"],
                flash_attn_num_splits=(
                    identity.effective_num_splits
                ),
            )
            entry.graph.replay()
            logits = self.model.compute_logits(tensors["outputs"])
        except Exception:
            self.exact_cuda_graph_cache.disable_entry(
                entry.identity_sha256,
                "replay_disabled",
            )
            raise
        finally:
            reset_context()
        entry.replay_count += 1
        entry.last_replay_step = getattr(
            self,
            "_cuda_graph_step_id",
            0,
        ) + 1
        return logits
        # 假设 block_size=256（每个块存 256 个 token），其他参数不变：

        # 32 层（num_hidden_layers=32）；
        # 8 个 KV 头（num_kv_heads=8）；
        # 每个头 64 维（head_dim=64）；
        # Key+Value 共 2 组（2）。

        # 对于 1 个 token，它的 KV 数据总元素数是：
        # 2（K+V） × 32（层） × 8（头） × 64（维度） = 32768 个元素。

        # 而 1 个缓存块能存 256 个 token，因此这个块的总元素数是：
        # 256（token数） × 32768（每个token的元素数） = 8388608 个元素

    def snapshot_kv_slots(
        self,
        physical_slots: list[int],
    ) -> dict[str, torch.Tensor]:
        if self.config.kv_quant_bits != 0:
            raise RuntimeError("KV snapshot requires FP KV")
        if not physical_slots:
            raise ValueError("KV snapshot requires at least one physical slot")
        block_ids = torch.tensor(
            [slot // self.block_size for slot in physical_slots],
            device=self.kv_cache.device,
            dtype=torch.long,
        )
        offsets = torch.tensor(
            [slot % self.block_size for slot in physical_slots],
            device=self.kv_cache.device,
            dtype=torch.long,
        )
        keys = (
            self.kv_cache[0, :, block_ids, offsets]
            .detach()
            .cpu()
            .clone()
        )
        values = (
            self.kv_cache[1, :, block_ids, offsets]
            .detach()
            .cpu()
            .clone()
        )
        return {"keys": keys, "values": values}

    def restore_kv_slots(
        self,
        physical_slots: list[int],
        snapshot: dict[str, torch.Tensor],
    ) -> None:
        if not physical_slots:
            raise ValueError("KV restore requires at least one physical slot")
        block_ids = torch.tensor(
            [slot // self.block_size for slot in physical_slots],
            device=self.kv_cache.device,
            dtype=torch.long,
        )
        offsets = torch.tensor(
            [slot % self.block_size for slot in physical_slots],
            device=self.kv_cache.device,
            dtype=torch.long,
        )
        self.kv_cache[0, :, block_ids, offsets].copy_(
            snapshot["keys"].to(self.kv_cache.device)
        )
        self.kv_cache[1, :, block_ids, offsets].copy_(
            snapshot["values"].to(self.kv_cache.device)
        )
        torch.cuda.synchronize()

    def _capture_exact_multi_sequence_graph(
        self,
        *,
        identity,
        input_ids,
        positions,
        context,
    ) -> ExactCudaGraphEntry:
        if identity.graph_batch_size != identity.active_batch_size:
            raise ValueError("exact capture cannot use rounded batch size")
        batch_size = identity.active_batch_size
        if int(input_ids.size(0)) != batch_size:
            raise ValueError("capture input batch does not match identity")
        if int(context.block_tables.size(1)) != identity.page_table_width:
            raise ValueError(
                "capture page-table width does not match identity"
            )
        device = self.kv_cache.device
        hf_config = self.config.hf_config
        tensors = {
            "input_ids": torch.zeros(
                batch_size,
                dtype=torch.int64,
                device=device,
            ),
            "positions": torch.zeros(
                batch_size,
                dtype=torch.int64,
                device=device,
            ),
            "slot_mapping": torch.zeros(
                batch_size,
                dtype=torch.int32,
                device=device,
            ),
            "context_lens": torch.zeros(
                batch_size,
                dtype=torch.int32,
                device=device,
            ),
            "block_tables": torch.zeros(
                batch_size,
                identity.page_table_width,
                dtype=torch.int32,
                device=device,
            ),
            "outputs": torch.zeros(
                batch_size,
                hf_config.hidden_size,
                dtype=hf_config.torch_dtype,
                device=device,
            ),
        }
        tensors["input_ids"].copy_(input_ids)
        tensors["positions"].copy_(positions)
        tensors["context_lens"].copy_(context.context_lens)
        tensors["block_tables"].copy_(context.block_tables)
        scratch_slots = list(
            self._exact_graph_scratch_slots(batch_size=batch_size)
        )
        tensors["slot_mapping"].copy_(
            torch.tensor(
                scratch_slots,
                dtype=torch.int32,
                device=device,
            )
        )
        snapshot = self.snapshot_kv_slots(scratch_slots)
        graph = torch.cuda.CUDAGraph()
        static_bytes = sum(
            int(tensor.numel() * tensor.element_size())
            for tensor in tensors.values()
        )
        allocated_before = int(torch.cuda.memory_allocated())
        reserved_before = int(torch.cuda.memory_reserved())
        capture_started_ns = time.perf_counter_ns()
        restore_error = None
        capture_error = None
        try:
            set_context(
                False,
                slot_mapping=tensors["slot_mapping"],
                context_lens=tensors["context_lens"],
                block_tables=tensors["block_tables"],
                flash_attn_num_splits=identity.effective_num_splits,
            )
            tensors["outputs"].copy_(
                self.model(
                    tensors["input_ids"],
                    tensors["positions"],
                )
            )
            torch.cuda.synchronize()
            with torch.cuda.graph(graph, self.graph_pool):
                tensors["outputs"].copy_(
                    self.model(
                        tensors["input_ids"],
                        tensors["positions"],
                    )
                )
            torch.cuda.synchronize()
        except Exception as exc:
            capture_error = exc
        finally:
            reset_context()
            try:
                self.restore_kv_slots(scratch_slots, snapshot)
            except Exception as exc:
                restore_error = exc
        capture_duration_ns = (
            time.perf_counter_ns() - capture_started_ns
        )
        allocated_after = int(torch.cuda.memory_allocated())
        reserved_after = int(torch.cuda.memory_reserved())
        retained_reserved_bytes = max(
            0,
            reserved_after - reserved_before,
        )
        if restore_error is not None:
            raise _ExactGraphCaptureError(
                "scratch_unavailable",
                "exact CUDA Graph scratch restore failed",
                retained_reserved_bytes=retained_reserved_bytes,
            ) from restore_error
        if capture_error is not None:
            raise _ExactGraphCaptureError(
                "capture_failed",
                "exact CUDA Graph capture failed",
                retained_reserved_bytes=retained_reserved_bytes,
            ) from capture_error
        rebuilt = self._build_multi_sequence_graph_identity(
            tensors["input_ids"],
            SimpleNamespace(
                block_tables=tensors["block_tables"],
            ),
        )
        if rebuilt != identity or rebuilt.sha256 != identity.sha256:
            raise _ExactGraphCaptureError(
                "identity_drift",
                "exact CUDA Graph identity drift",
                retained_reserved_bytes=retained_reserved_bytes,
            )
        if self.graph_pool is None:
            self.graph_pool = graph.pool()
        return ExactCudaGraphEntry(
            identity=identity,
            identity_sha256=identity.sha256,
            graph=graph,
            tensors=tensors,
            static_bytes=static_bytes,
            capture_duration_ns=capture_duration_ns,
            allocated_delta_bytes=max(
                0,
                allocated_after - allocated_before,
            ),
            reserved_delta_bytes=max(
                0,
                retained_reserved_bytes,
            ),
        )


    # 每个序列（seq）的block_table是一个列表，记录该序列在 KV Cache 中使用的块编号。
    def bind_kv_block_identity_rows(
        self,
        seqs: tuple[Sequence, ...],
        rows: tuple[KVBlockIdentityRow, ...],
    ) -> None:
        if self.kv_offload is None:
            if rows:
                raise RuntimeError(
                    "KV block identities require kv_offload_mvp0"
                )
            return
        if len(rows) != len(seqs):
            raise ValueError("KV block identity row count mismatch")

        bindings = []
        for seq, row in zip(seqs, rows):
            if row.sequence_id != seq.seq_id:
                raise ValueError(
                    "KV block identity sequence mismatch"
                )
            row_block_ids = tuple(
                block_id
                for block_id, _ in row.block_identities
            )
            if row_block_ids != tuple(seq.block_table):
                raise ValueError("KV block identity table mismatch")
            bindings.extend(row.block_identities)

        for block_id, generation in bindings:
            self.kv_offload.bind_logical_block_identity(
                block_id,
                generation,
            )

    def _speculative_residency_participant(
        self,
    ) -> SpeculativeResidencyParticipant:
        participant = getattr(
            self,
            "speculative_residency",
            None,
        )
        if participant is None:
            raise RuntimeError(
                "speculative residency requires kv_offload_mvp0"
            )
        return participant

    @staticmethod
    def _speculative_residency_result_dict(
        result: SpeculativeResidencyResult,
    ) -> dict:
        return {
            "ticket_id": result.ticket_id,
            "participant_id": result.participant_id,
            "operation": result.operation,
            "status": result.status,
            "sequence_ids": result.sequence_ids,
            "committed_block_identities": (
                result.committed_block_identities
            ),
            "rejected_block_identities": (
                result.rejected_block_identities
            ),
            "detail": result.detail,
        }

    def prepare_speculative_residency_batch(
        self,
        ticket_id: int,
        rows: tuple[SpeculativeResidencyPrepareRow, ...],
    ) -> dict:
        return self._speculative_residency_result_dict(
            self._speculative_residency_participant().prepare_batch(
                ticket_id,
                rows,
                stage_all_original_blocks=not bool(
                    self.config.kv_offload_mvp0
                    and self.config.kv_offload_blockwise_decode
                ),
            )
        )

    def precommit_speculative_residency_batch(
        self,
        ticket_id: int,
        rows: tuple[SpeculativeResidencyPrecommitRow, ...],
    ) -> dict:
        return self._speculative_residency_result_dict(
            self._speculative_residency_participant().precommit_batch(
                ticket_id,
                rows,
            )
        )

    def rollback_speculative_residency_batch(
        self,
        ticket_id: int,
    ) -> dict:
        return self._speculative_residency_result_dict(
            self._speculative_residency_participant().rollback_batch(
                ticket_id
            )
        )

    def seal_speculative_residency_batch(
        self,
        ticket_id: int,
    ) -> dict:
        return self._speculative_residency_result_dict(
            self._speculative_residency_participant().seal_batch(
                ticket_id
            )
        )

    def speculative_side_state_available(self) -> bool:
        return getattr(
            self,
            "qwen35_speculative_state_owner",
            None,
        ) is not None

    def speculative_cleanup_observation(self) -> dict:
        owner = getattr(
            self,
            "qwen35_speculative_state_owner",
            None,
        )
        leases = getattr(
            self,
            "_speculative_side_state_leases_by_sequence",
            {},
        )
        return {
            "rank": self.rank,
            "active_transaction_count": int(
                owner is not None and owner.active
            ),
            "live_lease_count": len(leases),
        }

    def prepare_speculative_side_state_batch(
        self,
        seqs: tuple[Sequence, ...],
    ) -> dict:
        owner = getattr(
            self,
            "qwen35_speculative_state_owner",
            None,
        )
        if owner is None:
            raise RuntimeError(
                "speculative side-state owner is unavailable"
            )
        leases = tuple(
            HybridStateLease(
                slot_id=int(seq.hybrid_state_slot_id),
                generation=int(seq.hybrid_state_generation),
                request_id=int(seq.seq_id),
            )
            for seq in seqs
        )
        if any(lease.slot_id < 0 for lease in leases):
            raise RuntimeError(
                "speculative side state requires active hybrid leases"
            )
        handle = owner.prepare(seqs, leases)
        assert_tensor_free(
            handle,
            name="speculative side-state prepare receipt",
        )
        self._speculative_side_state_handle = handle
        self._speculative_side_state_leases_by_sequence = {
            int(seq.seq_id): lease
            for seq, lease in zip(seqs, leases)
        }
        return handle

    def _active_speculative_side_state(self):
        owner = getattr(
            self,
            "qwen35_speculative_state_owner",
            None,
        )
        handle = getattr(
            self,
            "_speculative_side_state_handle",
            None,
        )
        if owner is None or handle is None or not owner.active:
            raise RuntimeError(
                "speculative side-state transaction is not active"
            )
        return owner, handle

    def select_speculative_side_state_batch(self, rows) -> dict:
        owner, handle = self._active_speculative_side_state()
        receipt = owner.select(handle, rows)
        assert_tensor_free(
            receipt,
            name="speculative side-state select receipt",
        )
        return receipt

    def apply_speculative_side_state_batch(self) -> dict:
        owner, handle = self._active_speculative_side_state()
        receipt = owner.apply(handle)
        assert_tensor_free(
            receipt,
            name="speculative side-state apply receipt",
        )
        return receipt

    def seal_speculative_side_state_batch(self) -> dict:
        owner, handle = self._active_speculative_side_state()
        receipt = owner.seal(handle)
        assert_tensor_free(
            receipt,
            name="speculative side-state seal receipt",
        )
        self._speculative_side_state_handle = None
        self._speculative_side_state_leases_by_sequence = {}
        return receipt

    def rollback_speculative_side_state_batch(self) -> dict:
        owner = getattr(
            self,
            "qwen35_speculative_state_owner",
            None,
        )
        handle = getattr(
            self,
            "_speculative_side_state_handle",
            None,
        )
        if owner is None or handle is None:
            raise RuntimeError(
                "speculative side-state transaction is not active"
            )
        receipt = owner.rollback(handle)
        assert_tensor_free(
            receipt,
            name="speculative side-state rollback receipt",
        )
        self._speculative_side_state_handle = None
        self._speculative_side_state_leases_by_sequence = {}
        return receipt

    def prepare_block_tables(self, seqs: list[Sequence]):
        max_len = max(len(seq.block_table) for seq in seqs)
        block_tables_data = [seq.block_table + [-1] * (max_len - len(seq.block_table)) for seq in seqs]  #用-1补齐
        if self.kv_offload is not None:
            block_tables_data = self.kv_offload.translate_block_rows(block_tables_data, require_valid=True)
        return self.prepare_block_tables_from_rows(block_tables_data)

    def prepare_block_tables_from_rows(self, rows: list[list[int]], name: str = "block_tables"):
        return self._list_to_cuda_2d(rows, name, torch.int32)

    def _validate_spec_verify_transaction_authorization(
        self,
        *,
        item,
        plan: SpecVerifyPlan,
        physical_slots: tuple[int, ...],
    ) -> None:
        authorization = getattr(
            item,
            "transaction_authorization",
            None,
        )
        if authorization is None:
            raise RuntimeError(
                "spec_verify transaction authorization is missing"
            )
        sequence_id = getattr(item, "sequence_id", None)
        original_block_identities = getattr(
            item,
            "original_block_identities",
            None,
        )
        reserved_block_identities = getattr(
            item,
            "reserved_block_identities",
            None,
        )
        proxy_block_table = getattr(
            item,
            "proxy_block_table",
            None,
        )
        if (
            not isinstance(original_block_identities, tuple)
            or not isinstance(reserved_block_identities, tuple)
            or not isinstance(proxy_block_table, tuple)
        ):
            raise ValueError(
                "spec_verify transaction authorization metadata "
                "must use tuples"
            )
        authorization_original = getattr(
            authorization,
            "original_block_identities",
            None,
        )
        authorization_reserved = getattr(
            authorization,
            "reserved_block_identities",
            None,
        )
        if (
            getattr(authorization, "sequence_id", None)
            != sequence_id
            or getattr(authorization, "state", None)
            != "reserved"
            or getattr(
                authorization,
                "materialized_token_count",
                None,
            )
            != 0
            or getattr(
                authorization,
                "proposed_token_count",
                0,
            )
            < plan.query_len + 1
            or getattr(
                authorization,
                "original_num_tokens",
                None,
            )
            != plan.logical_slots[0]
            or authorization_original
            != original_block_identities
            or authorization_reserved
            != reserved_block_identities
        ):
            raise RuntimeError(
                "spec_verify transaction authorization fields mismatch"
            )
        authorized_proxy_block_table = tuple(
            block_id
            for block_id, _ in (
                authorization_original
                + authorization_reserved
            )
        )
        if proxy_block_table != authorized_proxy_block_table:
            raise RuntimeError(
                "spec_verify transaction authorization proxy mismatch"
            )
        expected_physical_slots = tuple(
            proxy_block_table[
                logical_slot // self.block_size
            ]
            * self.block_size
            + logical_slot % self.block_size
            for logical_slot in plan.logical_slots
        )
        if physical_slots != expected_physical_slots:
            raise RuntimeError(
                "spec_verify transaction authorization slot mismatch"
            )
        payload = (
            authorization.sequence_id,
            authorization.original_num_tokens,
            authorization.proposed_token_count,
            authorization.materialized_token_count,
            authorization.state,
            authorization.original_block_identities,
            authorization.reserved_block_identities,
        )
        expected_sha256 = hashlib.sha256(
            repr(payload).encode("utf-8")
        ).hexdigest()
        if (
            getattr(
                authorization,
                "authorization_sha256",
                None,
            )
            != expected_sha256
        ):
            raise RuntimeError(
                "spec_verify transaction authorization SHA mismatch"
            )

    def _validate_spec_verify_compatibility(
        self,
        *,
        seq_count: int,
        linear_draft: bool,
        greedy: bool,
        mixed_batch: bool,
        require_residency_ticket: bool = True,
        residency_ticket_id: int | None = None,
        sequence_ids: tuple[int, ...] = (),
    ) -> None:
        if (
            isinstance(seq_count, bool)
            or not isinstance(seq_count, int)
            or seq_count <= 0
        ):
            raise RuntimeError(
                "spec_verify requires at least one sequence"
            )
        if not linear_draft:
            raise RuntimeError("spec_verify requires a linear draft")
        if not greedy:
            raise RuntimeError("spec_verify requires greedy acceptance")
        if mixed_batch or self.config.chunked_prefill_mixed_batch:
            raise RuntimeError(
                "chunked_prefill_mixed_batch is unsupported by spec_verify"
            )
        if getattr(
            self,
            "hybrid_state_runtime_bridge",
            None,
        ) is not None and not (
            getattr(
                self,
                "qwen35_speculative_state_owner",
                None,
            ) is not None
            and self.qwen35_speculative_state_owner.active
        ):
            raise RuntimeError(
                "speculative verification requires transactional "
                "non-KV state"
            )
        if (
            self.config.kv_offload_mvp0
            and require_residency_ticket
        ):
            participant = getattr(
                self,
                "speculative_residency",
                None,
            )
            if (
                residency_ticket_id is None
                or participant is None
                or not participant.is_prepared_for(
                    residency_ticket_id,
                    sequence_ids,
                )
            ):
                raise RuntimeError(
                    "kv_offload_mvp0 requires a prepared "
                    "speculative residency ticket"
                )
        elif residency_ticket_id is not None:
            raise RuntimeError(
                "speculative residency ticket requires "
                "kv_offload_mvp0"
            )
        if (
            self.config.kv_offload_blockwise_decode
            and not self.config.kv_offload_mvp0
        ):
            raise RuntimeError(
                "kv_offload_blockwise_decode requires "
                "kv_offload_mvp0 for spec_verify"
            )
        unsupported = (
            ("kv_quant_bits", self.config.kv_quant_bits != 0),
            ("quest_top_k_blocks", self.config.quest_top_k_blocks > 0),
            ("am_compact_blocks", self.config.am_compact_blocks > 0),
            ("kv_cartridge_blocks", self.config.kv_cartridge_blocks > 0),
        )
        for name, active in unsupported:
            if active:
                raise RuntimeError(f"{name} is unsupported by spec_verify")

    def prepare_spec_verify_batch(
        self,
        items: tuple[object, ...],
        residency_ticket_id: int | None = None,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        SpecVerifyBatchMetadata,
    ]:
        self._spec_verify_transaction_authorized = False
        if not isinstance(items, tuple) or not items:
            raise ValueError(
                "spec_verify batch items must be a non-empty tuple"
            )
        sequence_ids = []
        query_len = None
        prepared_rows = []
        for batch_index, item in enumerate(items):
            sequence_id = getattr(
                item,
                "sequence_id",
                None,
            )
            if (
                isinstance(sequence_id, bool)
                or not isinstance(sequence_id, int)
                or sequence_id < 0
            ):
                raise ValueError(
                    "spec_verify batch sequence ID must be "
                    "a non-negative integer"
                )
            plan = getattr(item, "plan", None)
            if not isinstance(plan, SpecVerifyPlan):
                raise ValueError(
                    "spec_verify batch plan must be SpecVerifyPlan"
                )
            row_query_len = plan.query_len
            if row_query_len <= 0:
                raise ValueError(
                    "spec_verify batch query length must be > 0"
                )
            if query_len is None:
                query_len = row_query_len
            elif row_query_len != query_len:
                raise ValueError(
                    "spec_verify batch requires homogeneous "
                    "query lengths"
                )
            if (
                len(plan.positions) != row_query_len
                or len(plan.logical_slots) != row_query_len
            ):
                raise ValueError(
                    "spec_verify batch plan row lengths mismatch"
                )
            expected_slots = tuple(
                range(
                    plan.logical_slots[0],
                    plan.logical_slots[0]
                    + row_query_len,
                )
            )
            if plan.logical_slots != expected_slots:
                raise ValueError(
                    "spec_verify batch logical slots must be consecutive"
                )
            if plan.positions != plan.logical_slots:
                raise ValueError(
                    "spec_verify batch positions must match slots"
                )
            if plan.context_len != plan.logical_slots[-1] + 1:
                raise ValueError(
                    "spec_verify batch context length mismatch"
                )
            expected_visible_block_count = (
                plan.context_len + self.block_size - 1
            ) // self.block_size
            if (
                plan.visible_block_count
                != expected_visible_block_count
            ):
                raise ValueError(
                    "spec_verify batch visible block count mismatch"
                )
            proxy_block_table = getattr(
                item,
                "proxy_block_table",
                None,
            )
            if not isinstance(proxy_block_table, tuple):
                raise ValueError(
                    "spec_verify proxy block table must be a tuple"
                )
            if (
                len(proxy_block_table)
                < plan.visible_block_count
            ):
                raise ValueError(
                    "proxy block table does not cover verifier context"
                )
            visible_block_table = tuple(
                int(block_id)
                for block_id in proxy_block_table[
                    :plan.visible_block_count
                ]
            )
            if any(
                block_id < 0
                for block_id in visible_block_table
            ):
                raise ValueError(
                    "verifier block table contains an invalid block"
                )
            physical_slots = validate_spec_verify_slots(
                plan,
                list(visible_block_table),
                self.block_size,
            )
            if self.config.spec_verify_cuda_graphs:
                self._validate_spec_verify_transaction_authorization(
                    item=item,
                    plan=plan,
                    physical_slots=physical_slots,
                )
            sequence_ids.append(sequence_id)
            prepared_rows.append(
                (
                    sequence_id,
                    batch_index,
                    plan,
                    visible_block_table,
                    physical_slots,
                )
            )

        if len(set(sequence_ids)) != len(sequence_ids):
            raise ValueError(
                "spec_verify batch sequence IDs must be unique"
            )
        self._validate_spec_verify_compatibility(
            seq_count=len(items),
            linear_draft=True,
            greedy=True,
            mixed_batch=False,
            residency_ticket_id=residency_ticket_id,
            sequence_ids=tuple(sequence_ids),
        )
        blockwise_offload = bool(
            self.config.kv_offload_mvp0
            and self.config.kv_offload_blockwise_decode
        )
        blockwise_write_blocks = []
        seen_blockwise_write_blocks = set()
        if self.config.kv_offload_mvp0:
            participant = self.speculative_residency
            if blockwise_offload:
                participant.ensure_materialized_for(
                    residency_ticket_id,
                    tuple(sequence_ids),
                )
            manager = participant.manager
            mapped_rows = []
            for (
                sequence_id,
                batch_index,
                plan,
                visible_block_table,
                _,
            ) in prepared_rows:
                physical_slots = tuple(
                    manager.map_slots_for_positions(
                        list(visible_block_table),
                        list(plan.logical_slots),
                    )
                )
                if blockwise_offload:
                    prepared_block_table = visible_block_table
                    for logical_slot in plan.logical_slots:
                        block_id = int(
                            visible_block_table[
                                logical_slot // self.block_size
                            ]
                        )
                        if block_id not in seen_blockwise_write_blocks:
                            blockwise_write_blocks.append(block_id)
                            seen_blockwise_write_blocks.add(block_id)
                else:
                    prepared_block_table = tuple(
                        manager.map_block_rows(
                            [list(visible_block_table)]
                        )[0]
                    )
                mapped_rows.append(
                    (
                        sequence_id,
                        batch_index,
                        plan,
                        prepared_block_table,
                        physical_slots,
                    )
                )
            prepared_rows = mapped_rows
        block_table_width = max(
            len(row[3]) for row in prepared_rows
        )
        metadata_rows = tuple(
            SpecVerifyBatchRowMetadata(
                sequence_id=sequence_id,
                batch_index=batch_index,
                query_offset=batch_index * query_len,
                query_len=query_len,
                input_tokens=plan.input_tokens,
                positions=plan.positions,
                logical_slots=plan.logical_slots,
                physical_slots=physical_slots,
                context_len=plan.context_len,
                block_table=visible_block_table,
            )
            for (
                sequence_id,
                batch_index,
                plan,
                visible_block_table,
                physical_slots,
            ) in prepared_rows
        )
        metadata = SpecVerifyBatchMetadata(
            rows=metadata_rows,
            query_len=query_len,
            total_query_tokens=len(items) * query_len,
            block_table_width=block_table_width,
        )

        flat_input_tokens = [
            token_id
            for row in metadata.rows
            for token_id in row.input_tokens
        ]
        flat_positions = [
            position
            for row in metadata.rows
            for position in row.positions
        ]
        flat_physical_slots = [
            slot
            for row in metadata.rows
            for slot in row.physical_slots
        ]
        context_lens_data = [
            row.context_len for row in metadata.rows
        ]
        block_table_rows = [
            list(row.block_table)
            + [-1] * (
                block_table_width - len(row.block_table)
            )
            for row in metadata.rows
        ]

        input_ids = self._list_to_cuda(
            flat_input_tokens,
            "spec_verify_input_ids",
            torch.int64,
        )
        positions = self._list_to_cuda(
            flat_positions,
            "spec_verify_positions",
            torch.int64,
        )
        slot_mapping = self._list_to_cuda(
            flat_physical_slots,
            "spec_verify_slot_mapping",
            torch.int32,
        )
        context_lens = self._list_to_cuda(
            context_lens_data,
            "spec_verify_context_lens",
            torch.int32,
        )
        block_tables = self.prepare_block_tables_from_rows(
            block_table_rows,
            "spec_verify_block_tables",
        )
        blockwise_context = (
            {
                "kv_offload_manager": (
                    self.speculative_residency.manager
                ),
                "kv_offload_blockwise_decode": True,
                "kv_offload_blockwise_blocks": (
                    self.config.kv_offload_blockwise_blocks
                ),
                "kv_offload_logical_block_tables": (
                    [
                        list(row.block_table)
                        for row in metadata.rows
                    ]
                ),
                "kv_offload_context_lens": (
                    list(context_lens_data)
                ),
                "kv_offload_write_blocks": (
                    list(blockwise_write_blocks)
                ),
            }
            if blockwise_offload
            else {}
        )
        set_context(
            mode="spec_verify",
            slot_mapping=slot_mapping,
            context_lens=context_lens,
            block_tables=block_tables,
            spec_verify_query_lens=tuple(
                query_len for _ in items
            ),
            flash_attn_num_splits=(
                SPEC_VERIFY_FLASH_ATTN_NUM_SPLITS
            ),
            **blockwise_context,
        )
        self._spec_verify_transaction_authorized = bool(
            self.config.spec_verify_cuda_graphs
        )
        return input_ids, positions, metadata

    def _validate_spec_first_target_batch(
        self,
        seqs: tuple[Sequence, ...],
    ) -> None:
        if not isinstance(seqs, tuple) or not seqs:
            raise ValueError(
                "speculative first-target sequences must be "
                "a non-empty tuple"
            )
        if self.config.chunked_prefill_mixed_batch:
            raise RuntimeError(
                "chunked_prefill_mixed_batch is unsupported by "
                "speculative first-target"
            )
        sequence_ids = []
        for seq in seqs:
            sequence_id = getattr(seq, "seq_id", None)
            if (
                isinstance(sequence_id, bool)
                or not isinstance(sequence_id, int)
                or sequence_id < 0
            ):
                raise ValueError(
                    "speculative first-target sequence ID must "
                    "be a non-negative integer"
                )
            temperature = getattr(seq, "temperature", None)
            if (
                isinstance(temperature, bool)
                or not isinstance(temperature, (int, float))
            ):
                raise ValueError(
                    "speculative first-target temperature must "
                    "be numeric"
                )
            if temperature != 0:
                raise RuntimeError(
                    "speculative first-target requires greedy "
                    "temperature"
                )
            sequence_ids.append(sequence_id)
        if len(set(sequence_ids)) != len(sequence_ids):
            raise ValueError(
                "speculative first-target sequence IDs must "
                "be unique"
            )

    def _prepare_spec_first_target_hybrid_state(
        self,
        seqs: tuple[Sequence, ...],
    ) -> None:
        self._prepare_hybrid_state_batch(list(seqs), ())
        self._last_hybrid_state_leases = tuple(
            HybridStateLease(
                slot_id=seq.hybrid_state_slot_id,
                generation=seq.hybrid_state_generation,
                request_id=int(seq.seq_id),
            )
            for seq in seqs
            if getattr(seq, "hybrid_state_slot_id", -1) >= 0
        )
        self._last_hybrid_state_token_counts = (
            tuple(1 for _lease in self._last_hybrid_state_leases)
            if self._last_hybrid_state_leases
            else ()
        )

    def _run_spec_first_target_batch(
        self,
        seqs: tuple[Sequence, ...],
        return_hidden: bool = False,
        return_logits: bool = False,
        kv_block_identity_rows: tuple[
            KVBlockIdentityRow,
            ...,
        ] = (),
    ) -> tuple[FirstTargetResult, ...] | None:
        try:
            if not isinstance(return_hidden, bool):
                raise ValueError(
                    "return_hidden must be a boolean"
                )
            if not isinstance(return_logits, bool):
                raise ValueError(
                    "return_logits must be a boolean"
                )
            self._validate_spec_first_target_batch(seqs)
            self.bind_kv_block_identity_rows(
                seqs,
                kv_block_identity_rows,
            )
            self._prepare_spec_first_target_hybrid_state(seqs)
            input_ids, positions = self.prepare_decode(
                list(seqs)
            )
            self._kv_offload_before_forward()
            speculative_owner = getattr(
                self,
                "qwen35_speculative_state_owner",
                None,
            )
            prepared_state = (
                speculative_owner is not None
                and speculative_owner.active
            )
            if prepared_state:
                outputs = self.run_model(
                    input_ids,
                    positions,
                    False,
                    return_hidden=return_hidden,
                    execution_mode="decode",
                    prepare_qwen35_state=True,
                )
            else:
                outputs = self.run_model(
                    input_ids,
                    positions,
                    False,
                    return_hidden=return_hidden,
                    execution_mode="decode",
                )
            if prepared_state:
                speculative_owner.record_first_target(outputs)
                logits = outputs.logits
                hidden_states = (
                    outputs.normalized
                    if return_hidden
                    else None
                )
            elif return_hidden:
                logits, hidden_states = outputs
            else:
                logits = outputs
                hidden_states = None
            self._record_spec_first_target_trace(
                seqs=seqs,
                input_ids=input_ids,
                positions=positions,
                logits=logits,
            )
            self._kv_offload_after_forward()
            if (
                self.rank == 0
                and getattr(self, "_record_step_logits", False)
            ):
                self._last_step_logits_cpu = (
                    logits.detach().float().cpu()
                )
            else:
                self._last_step_logits_cpu = None
            if self.rank != 0:
                return None
            target_tokens = logits.argmax(dim=-1).tolist()
            return tuple(
                FirstTargetResult(
                    sequence_id=int(seq.seq_id),
                    target_token=int(
                        target_tokens[batch_index]
                    ),
                    target_hidden=(
                        hidden_states[batch_index]
                        if hidden_states is not None
                        else None
                    ),
                    target_logits=(
                        logits[batch_index]
                        if return_logits
                        else None
                    ),
                    metadata={
                        "batch_index": batch_index,
                        "execution_mode": "decode",
                    },
                )
                for batch_index, seq in enumerate(seqs)
            )
        finally:
            reset_context()

    def run_spec_first_target_batch(
        self,
        seqs: tuple[Sequence, ...],
        return_hidden: bool = False,
        return_logits: bool = False,
        kv_block_identity_rows: tuple[
            KVBlockIdentityRow,
            ...,
        ] = (),
    ) -> tuple[FirstTargetResult, ...] | None:
        return run_profiled_step(
            self.decode_internal_profiler,
            batch_kind="spec_first_target",
            is_decode=True,
            active_sequence_count=len(seqs),
            request_set_sha256=_profile_request_set_sha256(
                seq.seq_id for seq in seqs
            ),
            dispatch="eager",
            call=lambda: self._run_spec_first_target_batch(
                seqs,
                return_hidden,
                return_logits,
                kv_block_identity_rows,
            ),
        )

    def register_speculative_proposal_executor(
        self,
        executor_id,
        executor,
        capabilities,
    ) -> None:
        self.speculative_proposal_executors.register(
            executor_id,
            executor,
            capabilities,
        )

    def observe_speculative_target_prefill_batch(
        self,
        executor_id: str,
        rows: tuple[TargetPrefillObservation, ...],
    ) -> None:
        capabilities = (
            self.speculative_proposal_executors
            .capabilities_for(executor_id)
        )
        self.speculative_proposal_executors.observe_target_prefill(
            executor_id,
            rows,
            capabilities,
        )

    def prepare_speculative_proposal_finalize_batch(
        self,
        executor_id: str,
        rows: tuple[ProposalFinalizeRow, ...],
    ) -> str:
        capabilities = (
            self.speculative_proposal_executors
            .capabilities_for(executor_id)
        )
        return (
            self.speculative_proposal_executors
            .prepare_finalize_batch(
                executor_id,
                rows,
                capabilities,
            )
        )

    def commit_speculative_proposal_finalize_batch(
        self,
        executor_id: str,
        ticket_id: str,
    ) -> None:
        capabilities = (
            self.speculative_proposal_executors
            .capabilities_for(executor_id)
        )
        self.speculative_proposal_executors.commit_finalize_batch(
            executor_id,
            ticket_id,
            capabilities,
        )

    def rollback_speculative_proposal_finalize_batch(
        self,
        executor_id: str,
        ticket_id: str,
    ) -> None:
        capabilities = (
            self.speculative_proposal_executors
            .capabilities_for(executor_id)
        )
        self.speculative_proposal_executors.rollback_finalize_batch(
            executor_id,
            ticket_id,
            capabilities,
        )

    def release_speculative_proposal_sequence(
        self,
        executor_id: str,
        sequence_id: int,
        sequence_epoch: int,
    ) -> None:
        capabilities = (
            self.speculative_proposal_executors
            .capabilities_for(executor_id)
        )
        self.speculative_proposal_executors.release_sequence(
            executor_id,
            sequence_id,
            sequence_epoch,
            capabilities,
        )

    def _proposal_prefill_observation_required(self) -> bool:
        return bool(
            self.speculative_proposal_executors
            .lifecycle_executor_ids()
        )

    def _observe_proposal_target_prefill(
        self,
        seqs: list[Sequence],
        positions,
        target_hidden,
        *,
        batch_kind: str | None,
    ) -> None:
        if batch_kind == "mixed":
            raise ValueError(
                "proposal prefill observation does not support "
                "mixed batches"
            )
        executor_ids = (
            self.speculative_proposal_executors
            .lifecycle_executor_ids()
        )
        if not executor_ids:
            return
        rows = []
        row_offset = 0
        for seq in seqs:
            chunk_start = int(
                getattr(
                    seq,
                    "prefill_chunk_start",
                    getattr(seq, "num_cached_tokens", 0),
                )
            )
            chunk_end = getattr(
                seq,
                "prefill_chunk_end",
                None,
            )
            if chunk_end is None or (
                chunk_end == 0 and chunk_start == 0
            ):
                chunk_end = len(seq)
            chunk_end = int(chunk_end)
            if (
                chunk_start < 0
                or chunk_end < chunk_start
                or chunk_end > len(seq)
            ):
                raise ValueError(
                    "proposal prefill chunk bounds are invalid"
                )
            chunk_count = chunk_end - chunk_start
            next_offset = row_offset + chunk_count
            rows.append(
                TargetPrefillObservation(
                    sequence_id=int(seq.seq_id),
                    sequence_epoch=int(
                        getattr(seq, "sequence_epoch", 0)
                    ),
                    token_ids=tuple(
                        int(token_id)
                        for token_id in seq.token_ids[
                            chunk_start:chunk_end
                        ]
                    ),
                    positions=positions[
                        row_offset:next_offset
                    ],
                    target_hidden=target_hidden[
                        row_offset:next_offset
                    ],
                    is_final_chunk=bool(
                        getattr(
                            seq,
                            "prefill_chunk_final",
                            chunk_end == len(seq),
                        )
                    ),
                )
            )
            row_offset = next_offset
        position_rows = int(positions.shape[0])
        hidden_rows = int(target_hidden.shape[0])
        if row_offset != position_rows:
            raise ValueError(
                "proposal prefill positions do not match chunk rows"
            )
        if row_offset != hidden_rows:
            raise ValueError(
                "proposal prefill hidden rows do not match chunks"
            )
        normalized = tuple(rows)
        for executor_id in executor_ids:
            self.observe_speculative_target_prefill_batch(
                executor_id,
                normalized,
            )

    def run_spec_first_target_and_proposal_batch(
        self,
        seqs: tuple[Sequence, ...],
        descriptor,
        kv_block_identity_rows: tuple[
            KVBlockIdentityRow,
            ...,
        ] = (),
    ) -> tuple[FirstTargetProposalResult, ...] | None:
        try:
            if self.world_size not in (1, 4):
                raise RuntimeError(
                    "ModelRunner proposal execution supports TP1 or TP4"
                )
            executor_id = getattr(
                descriptor,
                "executor_id",
                None,
            )
            if not isinstance(executor_id, str) or not executor_id:
                raise ValueError(
                    "proposal executor ID must be non-empty"
                )
            capabilities = validate_draft_capabilities(
                getattr(
                    descriptor,
                    "capabilities",
                    None,
                ),
                expected_execution_domain="model_runner",
            )
            if (
                self.world_size > 1
                and capabilities.requires_target_logits
            ):
                raise RuntimeError(
                    "TP4 ModelRunner proposal execution cannot "
                    "require target logits"
                )
            self._validate_spec_first_target_batch(seqs)
            self.bind_kv_block_identity_rows(
                seqs,
                kv_block_identity_rows,
            )
            self._prepare_spec_first_target_hybrid_state(seqs)
            input_ids, positions = self.prepare_decode(
                list(seqs)
            )
            self._kv_offload_before_forward()
            speculative_owner = getattr(
                self,
                "qwen35_speculative_state_owner",
                None,
            )
            prepared_state = (
                speculative_owner is not None
                and speculative_owner.active
            )
            outputs = self.run_model(
                input_ids,
                positions,
                False,
                return_hidden=(
                    capabilities.requires_target_hidden
                ),
                execution_mode="decode",
                **(
                    {"prepare_qwen35_state": True}
                    if prepared_state
                    else {}
                ),
            )
            if prepared_state:
                speculative_owner.record_first_target(outputs)
                logits = outputs.logits
                hidden_states = (
                    outputs.normalized
                    if capabilities.requires_target_hidden
                    else None
                )
            elif capabilities.requires_target_hidden:
                logits, hidden_states = outputs
            else:
                logits = outputs
                hidden_states = None
            self._record_spec_first_target_trace(
                seqs=seqs,
                input_ids=input_ids,
                positions=positions,
                logits=logits,
            )
            self._kv_offload_after_forward()
            if (
                self.rank == 0
                and getattr(self, "_record_step_logits", False)
            ):
                self._last_step_logits_cpu = (
                    logits.detach().float().cpu()
                )
            else:
                self._last_step_logits_cpu = None
            target_tokens = select_tensor_parallel_greedy_tokens(
                logits,
                rank=self.rank,
                world_size=self.world_size,
                batch_size=len(seqs),
                device=input_ids.device,
            ).tolist()
            proposal_token_contexts = tuple(
                model_runner_proposal_token_context(
                    seq,
                    capabilities,
                )
                for seq in seqs
            )
            proposal_inputs = tuple(
                ModelRunnerProposalInput(
                    sequence_id=int(seq.seq_id),
                    token_ids=proposal_token_contexts[
                        batch_index
                    ][0],
                    remaining_output_tokens=max(
                        0,
                        int(seq.max_tokens)
                        - int(seq.num_completion_tokens),
                    ),
                    max_proposal_tokens=(
                        capabilities.max_proposal_tokens
                    ),
                    first_target_token=int(
                        target_tokens[batch_index]
                    ),
                    target_hidden=(
                        hidden_states[
                            batch_index:batch_index + 1
                        ]
                        if hidden_states is not None
                        else None
                    ),
                    target_logits=(
                        logits[batch_index]
                        if capabilities.requires_target_logits
                        else None
                    ),
                    context_token_count=proposal_token_contexts[
                        batch_index
                    ][1],
                )
                for batch_index, seq in enumerate(seqs)
            )
            proposals = (
                self.speculative_proposal_executors
                .execute_batch(
                    executor_id,
                    proposal_inputs,
                    capabilities,
                )
            )
            if self.rank != 0:
                return None
            rows = tuple(
                FirstTargetProposalResult(
                    sequence_id=int(seq.seq_id),
                    target_token=int(
                        target_tokens[batch_index]
                    ),
                    proposal=proposals[batch_index],
                    first_target_metadata={
                        "batch_index": batch_index,
                        "execution_mode": "decode",
                    },
                    proposal_metadata=(
                        proposals[batch_index].metadata
                    ),
                )
                for batch_index, seq in enumerate(seqs)
            )
            assert_tensor_free(
                rows,
                name="ModelRunner fused proposal result",
            )
            return rows
        finally:
            reset_context()

    def _run_spec_verify_batch(
        self,
        items: tuple[object, ...],
        residency_ticket_id: int | None = None,
    ) -> tuple[SpecVerifyBatchResultRow, ...] | None:
        try:
            prepare_kwargs = (
                {}
                if residency_ticket_id is None
                else {
                    "residency_ticket_id": (
                        residency_ticket_id
                    )
                }
            )
            prepared = self.prepare_spec_verify_batch(
                items,
                **prepare_kwargs,
            )
            input_ids, positions, metadata = prepared
            speculative_owner = getattr(
                self,
                "qwen35_speculative_state_owner",
                None,
            )
            prepared_state = (
                speculative_owner is not None
                and speculative_owner.active
            )
            if prepared_state:
                sequence_ids = tuple(
                    row.sequence_id
                    for row in metadata.rows
                )
                leases_by_sequence = getattr(
                    self,
                    "_speculative_side_state_leases_by_sequence",
                    {},
                )
                try:
                    self._last_hybrid_state_leases = tuple(
                        leases_by_sequence[sequence_id]
                        for sequence_id in sequence_ids
                    )
                except KeyError as error:
                    raise RuntimeError(
                        "speculative side-state lease inventory "
                        "is incomplete"
                    ) from error
                self._last_hybrid_state_token_counts = tuple(
                    row.query_len
                    for row in metadata.rows
                )
                initial_candidates = (
                    speculative_owner.initial_tail_candidates(
                        sequence_ids
                    )
                )
                prepared_step = self.run_model(
                    input_ids,
                    positions,
                    False,
                    execution_mode="spec_verify",
                    prepare_qwen35_state=True,
                    initial_qwen35_candidates=initial_candidates,
                    capture_qwen35_prefix_states=True,
                )
                speculative_owner.record_tail(
                    prepared_step,
                    sequence_ids,
                )
                logits = prepared_step.logits
            else:
                logits = self.run_model(
                    input_ids,
                    positions,
                    False,
                    execution_mode="spec_verify",
                )
            if self._spec_verify_trace.enabled:
                items_by_sequence = {
                    int(item.sequence.seq_id): item
                    for item in items
                }
                prediction_indices = []
                logical_block_identities = []
                for row in metadata.rows:
                    item = items_by_sequence[row.sequence_id]
                    prediction_base = (
                        int(
                            item.sequence
                            .num_completion_tokens
                        )
                        + 1
                    )
                    prediction_indices.extend(
                        prediction_base + offset
                        for offset in range(row.query_len)
                    )
                    logical_block_identities.append(
                        self._trace_block_identities(
                            row.block_table
                        )
                    )
                self._spec_verify_trace.record_rows(
                    stage="verify_tail",
                    execution_mode="spec_verify",
                    sequence_ids=tuple(
                        row.sequence_id
                        for row in metadata.rows
                    ),
                    query_offset=0,
                    query_len=metadata.query_len,
                    input_tokens=tuple(
                        token
                        for row in metadata.rows
                        for token in row.input_tokens
                    ),
                    positions=tuple(
                        position
                        for row in metadata.rows
                        for position in row.positions
                    ),
                    prediction_indices=tuple(
                        prediction_indices
                    ),
                    logical_block_identities=tuple(
                        logical_block_identities
                    ),
                    logits=logits,
                )
            if residency_ticket_id is not None:
                self._speculative_residency_participant(
                ).mark_materialized(
                    residency_ticket_id,
                    tuple(
                        row.sequence_id
                        for row in metadata.rows
                    ),
                )
            if self.rank != 0:
                return None
            flat_target_tokens = tuple(
                int(token_id)
                for token_id in logits.argmax(dim=-1).tolist()
            )
            return split_spec_verify_batch_target_tokens(
                metadata,
                flat_target_tokens,
            )
        finally:
            reset_context()

    def run_spec_verify_batch(
        self,
        items: tuple[object, ...],
        residency_ticket_id: int | None = None,
    ) -> tuple[SpecVerifyBatchResultRow, ...] | None:
        return run_profiled_step(
            self.decode_internal_profiler,
            batch_kind="spec_verify",
            is_decode=True,
            active_sequence_count=len(items),
            request_set_sha256=_profile_request_set_sha256(
                item.sequence_id for item in items
            ),
            dispatch="eager",
            call=lambda: self._run_spec_verify_batch(
                items,
                residency_ticket_id,
            ),
        )

    def prepare_spec_verify(
        self,
        seq: Sequence,
        input_tokens: list[int],
        proxy_block_table: list[int],
        slot_positions: list[int],
    ) -> tuple[torch.Tensor, torch.Tensor, SpecVerifyMetadata]:
        self._validate_spec_verify_compatibility(
            seq_count=1,
            linear_draft=True,
            greedy=True,
            mixed_batch=False,
        )
        if not input_tokens:
            raise ValueError(
                "prepare_spec_verify requires at least one tail query"
            )
        if len(input_tokens) != len(slot_positions):
            raise ValueError(
                "spec_verify input tokens and slot positions must match"
            )
        normalized_slots = [int(position) for position in slot_positions]
        expected_slots = list(
            range(
                normalized_slots[0],
                normalized_slots[0] + len(normalized_slots),
            )
        )
        if normalized_slots != expected_slots:
            raise ValueError(
                "spec_verify slot positions must be consecutive"
            )

        positions_data = list(normalized_slots)
        context_len = normalized_slots[-1] + 1
        visible_block_count = (
            context_len + self.block_size - 1
        ) // self.block_size
        visible_block_table = [
            int(block_id)
            for block_id in proxy_block_table[:visible_block_count]
        ]
        plan = SpecVerifyPlan(
            input_tokens=tuple(int(token_id) for token_id in input_tokens),
            positions=tuple(positions_data),
            logical_slots=tuple(normalized_slots),
            context_len=context_len,
            visible_block_count=visible_block_count,
        )
        physical_slots = validate_spec_verify_slots(
            plan,
            visible_block_table,
            self.block_size,
        )

        input_ids = self._list_to_cuda(
            [int(token_id) for token_id in input_tokens],
            "spec_verify_input_ids",
            torch.int64,
        )
        positions = self._list_to_cuda(
            positions_data,
            "spec_verify_positions",
            torch.int64,
        )
        slot_mapping = self._list_to_cuda(
            list(physical_slots),
            "spec_verify_slot_mapping",
            torch.int32,
        )
        context_lens = self._list_to_cuda(
            [context_len],
            "spec_verify_context_lens",
            torch.int32,
        )
        block_tables = self.prepare_block_tables_from_rows(
            [visible_block_table],
            "spec_verify_block_tables",
        )
        set_context(
            mode="spec_verify",
            slot_mapping=slot_mapping,
            context_lens=context_lens,
            block_tables=block_tables,
            flash_attn_num_splits=SPEC_VERIFY_FLASH_ATTN_NUM_SPLITS,
        )
        metadata = SpecVerifyMetadata(
            query_len=len(input_tokens),
            input_tokens=tuple(int(token_id) for token_id in input_tokens),
            positions=tuple(positions_data),
            logical_slots=tuple(normalized_slots),
            physical_slots=physical_slots,
            context_len=context_len,
            block_table=tuple(visible_block_table),
        )
        return input_ids, positions, metadata

    def _kv_offload_translate_block_rows(
        self,
        rows: list[list[int]],
        require_valid: bool = True,
        future_logical_blocks: set[int] | None = None,
    ) -> list[list[int]]:
        if self.kv_offload is None:
            return rows
        return self.kv_offload.translate_block_rows(
            rows,
            require_valid=require_valid,
            future_logical_blocks=future_logical_blocks,
        )

    def _kv_offload_translate_slots_for_positions(
        self,
        block_table: list[int],
        positions: list[int],
        require_valid: bool = False,
        future_logical_blocks: set[int] | None = None,
    ) -> list[int]:
        if self.kv_offload is None:
            return [
                block_table[pos // self.block_size] * self.block_size + (pos % self.block_size)
                for pos in positions
            ]
        return self.kv_offload.translate_slots_for_positions(
            block_table,
            positions,
            require_valid=require_valid,
            future_logical_blocks=future_logical_blocks,
        )

    def _kv_offload_mark_pending_dirty(self, block_table: list[int], positions: list[int]):
        if self.kv_offload is None:
            return
        for pos in positions:
            self._kv_offload_pending_dirty_blocks.append(block_table[pos // self.block_size])

    def _kv_offload_after_forward(self):
        if self.kv_offload is None:
            return
        if self._kv_offload_pending_dirty_blocks:
            self.kv_offload.mark_dirty(self._kv_offload_pending_dirty_blocks)
            if not self.kv_offload.writeback_on_evict:
                self.kv_offload.writeback_dirty(self._kv_offload_pending_dirty_blocks)
            self._kv_offload_pending_dirty_blocks = []

    def _kv_offload_before_forward(self):
        if self.kv_offload is None:
            return
        self.kv_offload.wait_for_pending()

    def kv_offload_summary(self) -> dict | None:
        if self.kv_offload is None:
            return None
        return self.kv_offload.summary()

    def light_doc_cache_planning_summary(self, plan) -> dict | None:
        return build_model_runner_light_doc_cache_summary(self, plan)

    def light_doc_cache_materialize_sidecar(
        self,
        plan,
        *,
        fill_value: float = -1.0,
        recover_missing_fn=None,
        evaluate_readback: bool = False,
    ) -> dict | None:
        result = materialize_model_runner_light_doc_cache_sidecar(
            self,
            plan,
            fill_value=fill_value,
            recover_missing_fn=recover_missing_fn,
            evaluate_readback=evaluate_readback,
        )
        if result is None:
            return None
        self.light_doc_cache_sidecar, summary = result
        return summary

    # ---- pinned host buffer 池：把多次小 H2D 改成 buffer 复用 + non_blocking copy ----
    def _get_pinned(self, name: str, n: int, dtype: torch.dtype) -> torch.Tensor:
        """按 (name, dtype) 拿一个长度 ≥ n 的 1D pinned host tensor；按需翻倍扩容并复用。"""
        key = (name, dtype)
        buf = self._pinned_buf_cache.get(key)
        if buf is None or buf.numel() < n:
            new_size = max(n, (buf.numel() * 2) if buf is not None else max(n, 64))
            # 显式 device="cpu" + pin_memory=True，避开 default_device 被设成 cuda 的情况
            buf = torch.empty(new_size, dtype=dtype, device="cpu", pin_memory=True)
            self._pinned_buf_cache[key] = buf
        return buf

    def _list_to_cuda(self, data: list, name: str, dtype: torch.dtype) -> torch.Tensor:
        """把 python list 写入 pinned host buffer 后 non_blocking H2D。返回 GPU tensor。"""
        n = len(data)
        host = self._get_pinned(name, n, dtype)
        host[:n].copy_(torch.tensor(data, dtype=dtype, device="cpu"))
        return host[:n].cuda(non_blocking=True)

    def _list_to_cuda_2d(self, data: list[list[int]], name: str, dtype: torch.dtype) -> torch.Tensor:
        """2D list（每行长度相同）打成 pinned host 矩阵后 non_blocking H2D。"""
        rows = len(data)
        cols = len(data[0]) if rows else 0
        n = rows * cols
        host = self._get_pinned(name, n, dtype)
        host[:n].copy_(torch.tensor(data, dtype=dtype, device="cpu").flatten())
        return host[:n].view(rows, cols).cuda(non_blocking=True)



# 收集新token输入（input_ids/positions） → 划分多序列边界（cu_seqlens） → 适配内存需求（max_seqlen） → 管理缓存块（block_tables） → 映射块内槽位（slot_mapping） → 将所有数据送GPU并设置上下文
    def prepare_prefill(self, seqs: list[Sequence]):        #输入数据收集、序列边界划分、缓存映射、内存适配
        self._kv_offload_pending_dirty_blocks = []
        input_ids = []          # 记录每个 seq 的 所有输入token id，一维[] 
        positions = []          # 记录每个 seq中 输入的 token的位置，一维[]
        cu_seqlens_q = [0]       # 以前缀和的形式，记录每个seq的长度，如 [0, 3, 5] 表示有两个seq, 一个长度为 3 = 3 - 0， 另一个长度为 2 = 5-3
        cu_seqlens_k = [0]       
        max_seqlen_q = 0        # 记录seqs(去掉缓存后)的最大长度，标量
        max_seqlen_k = 0        # 记录seqs(包含缓存长度)的最大长度
        slot_mapping = []       # 记录所有seqs每个block中的token_id 在kvcache中的位置，[token_id1, token_id2, ...token_id]
        block_tables = None     # 有前缀和的时候，才会初始化该块表
        prefill_chunk_starts = []
        prefill_chunk_ends = []
        prefill_attention_reference_lens = []
        logical_block_table_rows = []
        kv_offload_write_blocks = []
        blockwise_prefill_enabled = self.kv_offload is not None and self.config.kv_offload_blockwise_prefill
        for seq in seqs:
            seq_len = len(seq)
            chunk_start = getattr(seq, "prefill_chunk_start", seq.num_cached_tokens)
            chunk_end = getattr(seq, "prefill_chunk_end", seq_len)
            if chunk_end == 0 and chunk_start == 0:
                # warmup_model() calls ModelRunner.run() directly with fresh Sequence
                # objects, bypassing Scheduler's chunk boundary initialization.
                chunk_end = seq_len
            prefill_chunk_starts.append(int(chunk_start))
            prefill_chunk_ends.append(int(chunk_end))
            prefill_attention_reference_lens.append(
                int(seq.num_prompt_tokens)
            )
            input_ids.extend(seq[chunk_start:chunk_end])       #从已有的cache/chunk进度开始计数
            positions.extend(list(range(chunk_start, chunk_end)))   
            seqlen_q = chunk_end - chunk_start
            seqlen_k = chunk_end
            #前缀和 累计长度，用于区分不同的序列
            cu_seqlens_q.append(cu_seqlens_q[-1] + seqlen_q)
            cu_seqlens_k.append(cu_seqlens_k[-1] + seqlen_k)
            max_seqlen_q = max(max_seqlen_q, seqlen_q)
            max_seqlen_k = max(max_seqlen_k, seqlen_k)
            if not seq.block_table:
                logical_block_table_rows.append([])
                continue
            visible_blocks = (chunk_end + self.block_size - 1) // self.block_size
            logical_block_table_rows.append(list(seq.block_table[:visible_blocks]))
            
            write_positions = list(range(chunk_start, chunk_end))
            future_blocks = set(int(block_id) for block_id in seq.block_table)
            if self.kv_offload is not None and blockwise_prefill_enabled:
                mapped_slots, write_blocks = _stage_kv_offload_write_positions(
                    self.kv_offload,
                    seq.block_table,
                    write_positions,
                    self.block_size,
                    future_blocks,
                )
                slot_mapping.extend(mapped_slots)
                kv_offload_write_blocks.extend(write_blocks)
            else:
                if self.kv_offload is not None:
                    self.kv_offload.stats["prefetch_plans"] += 1
                    self.kv_offload.stats["prefetch_write_blocks"] += len(set(
                        int(seq.block_table[pos // self.block_size]) for pos in write_positions
                    ))
                slot_mapping.extend(self._kv_offload_translate_slots_for_positions(
                    seq.block_table,
                    write_positions,
                    future_logical_blocks=future_blocks,
                ))
            self._kv_offload_mark_pending_dirty(seq.block_table, write_positions)
        
        blockwise_prefill_active = bool(blockwise_prefill_enabled and cu_seqlens_k[-1] > cu_seqlens_q[-1])
        if cu_seqlens_k[-1] > cu_seqlens_q[-1]:      # 正常情况下二者是相等的，大于则说明有前缀缓存, 因此取出seq中的block_table, 拼成 blocktables表
            if self.kv_offload is not None:
                if blockwise_prefill_active:
                    max_len = max(len(row) for row in logical_block_table_rows)
                    logical_block_table_rows = [
                        row + [-1] * (max_len - len(row))
                        for row in logical_block_table_rows
                    ]
                    # blockwise prefill keeps logical rows on host. Attention.forward
                    # stages prefix windows layer-by-layer and therefore does not need
                    # a full physical block_table here.
                    block_tables = None
                else:
                    for seq in seqs:
                        chunk_start = getattr(seq, "prefill_chunk_start", seq.num_cached_tokens)
                        if chunk_start > 0:
                            self.kv_offload.translate_slots_for_positions(
                                seq.block_table, list(range(0, chunk_start)), require_valid=True)
                    max_len = max(len(seq.block_table) for seq in seqs)
                    block_table_rows = [seq.block_table + [-1] * (max_len - len(seq.block_table)) for seq in seqs]
                    future_blocks = set(int(block_id) for row in block_table_rows for block_id in row if int(block_id) >= 0)
                    block_table_rows = self._kv_offload_translate_block_rows(
                        block_table_rows,
                        require_valid=False,
                        future_logical_blocks=future_blocks,
                    )
                    block_tables = self.prepare_block_tables_from_rows(block_table_rows)
            else:
                block_tables = self.prepare_block_tables(seqs)
        
        # 将准备好的数据传输到GPU上
        input_ids = self._list_to_cuda(input_ids, "input_ids", torch.int64)
        positions = self._list_to_cuda(positions, "positions", torch.int64)
        cu_seqlens_q = self._list_to_cuda(cu_seqlens_q, "cu_seqlens_q", torch.int32)
        cu_seqlens_k = self._list_to_cuda(cu_seqlens_k, "cu_seqlens_k", torch.int32)
        slot_mapping = self._list_to_cuda(slot_mapping, "slot_mapping", torch.int32)
        am_compact_active = (
            self.config.am_compact_blocks > 0
            and self.config.am_compact_cache_refresh_interval > 0
            and min(len(seq) for seq in seqs) >= self.config.am_compact_min_seq_len
            and min(len(seq) for seq in seqs) > self.config.am_compact_blocks
        )
        prefill_window_blocks = self.config.kv_offload_blockwise_blocks
        if blockwise_prefill_active:
            prefill_window_blocks = (
                _resolve_blockwise_prefill_window_blocks(
                    prefill_window_blocks,
                    gpu_blocks=self.kv_offload.gpu_blocks,
                    write_blocks=kv_offload_write_blocks,
                )
            )
        set_context(True, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, slot_mapping, None, block_tables,
                    am_compact_blocks=(self.config.am_compact_blocks if am_compact_active else 0),
                    am_compact_selector=self.config.am_compact_selector,
                    am_compact_score_method=self.config.am_compact_score_method,
                    am_compact_beta_bound=self.config.am_compact_beta_bound,
                    am_compact_ridge_lambda=self.config.am_compact_ridge_lambda,
                    am_omp_candidate_pool_size=self.config.am_omp_candidate_pool_size,
                    am_compact_cache_refresh_interval=self.config.am_compact_cache_refresh_interval,
                    am_prefill_cache_ref_query_stride=self.config.am_prefill_cache_ref_query_stride,
                    am_compact_num_clusters=self.config.am_compact_num_clusters,
                    am_compact_route_top_k=self.config.am_compact_route_top_k,
                    am_compact_num_key_spans=self.config.am_compact_num_key_spans,
                    am_compact_decode_refit=self.config.am_compact_decode_refit,
                    am_compact_decode_refit_mode=self.config.am_compact_decode_refit_mode,
                    am_compact_decode_refit_interval=self.config.am_compact_decode_refit_interval,
                    am_compact_skip_first_layers=self.config.am_compact_skip_first_layers,
                    am_compact_skip_last_layers=self.config.am_compact_skip_last_layers,
                    am_compact_enable_layers=self.config.am_compact_enable_layers,
                    am_compact_layer_stride=self.config.am_compact_layer_stride,
                    kv_offload_manager=self.kv_offload,
                    kv_offload_blockwise_prefill=blockwise_prefill_active,
                    kv_offload_blockwise_blocks=prefill_window_blocks,
                    kv_offload_logical_block_tables=logical_block_table_rows,
                    kv_offload_context_lens=prefill_chunk_ends,
                    kv_offload_write_blocks=[int(block) for block in kv_offload_write_blocks],
                    kv_offload_prefill_chunk_starts=prefill_chunk_starts,
                    kv_offload_prefill_chunk_ends=prefill_chunk_ends,
                    prefill_attention_reference_lens=tuple(
                        prefill_attention_reference_lens
                    ))
        return input_ids, positions

    def prepare_mixed(self, seqs: list[Sequence]):
        """Prepare a mixed chunked-prefill + decode batch as varlen prefill.

        Decode rows are represented as query length 1, so they can share the
        same FlashAttention varlen prefill path with prefill chunks.
        """
        if self.kv_offload is not None:
            raise NotImplementedError("KV offload MVP-0 暂不支持 mixed prefill+decode batch")
        input_ids = []
        positions = []
        cu_seqlens_q = [0]
        cu_seqlens_k = [0]
        max_seqlen_q = 0
        max_seqlen_k = 0
        slot_mapping = []
        logits_indices = []
        block_tables = None
        for seq in seqs:
            q_start = cu_seqlens_q[-1]
            if getattr(seq, "step_is_decode", False):
                input_ids.append(seq.last_token)
                positions.append(len(seq) - 1)
                seqlen_q = 1
                seqlen_k = len(seq)
                slot_mapping.append(seq.block_table[-1] * seq.block_size + seq.last_block_num_tokens - 1)
            else:
                seq_len = len(seq)
                chunk_start = getattr(seq, "prefill_chunk_start", seq.num_cached_tokens)
                chunk_end = getattr(seq, "prefill_chunk_end", seq_len)
                if chunk_end == 0 and chunk_start == 0:
                    chunk_end = seq_len
                input_ids.extend(seq[chunk_start:chunk_end])
                positions.extend(list(range(chunk_start, chunk_end)))
                seqlen_q = chunk_end - chunk_start
                seqlen_k = chunk_end
                for pos in range(chunk_start, chunk_end):
                    block_id = seq.block_table[pos // self.block_size]
                    slot_mapping.append(block_id * self.block_size + (pos % self.block_size))

            if getattr(seq, "step_do_sample", True):
                logits_indices.append(q_start + seqlen_q - 1)

            cu_seqlens_q.append(cu_seqlens_q[-1] + seqlen_q)
            cu_seqlens_k.append(cu_seqlens_k[-1] + seqlen_k)
            max_seqlen_q = max(max_seqlen_q, seqlen_q)
            max_seqlen_k = max(max_seqlen_k, seqlen_k)

        if cu_seqlens_k[-1] > cu_seqlens_q[-1]:
            block_tables = self.prepare_block_tables(seqs)

        input_ids = self._list_to_cuda(input_ids, "input_ids", torch.int64)
        positions = self._list_to_cuda(positions, "positions", torch.int64)
        cu_seqlens_q = self._list_to_cuda(cu_seqlens_q, "cu_seqlens_q", torch.int32)
        cu_seqlens_k = self._list_to_cuda(cu_seqlens_k, "cu_seqlens_k", torch.int32)
        slot_mapping = self._list_to_cuda(slot_mapping, "slot_mapping", torch.int32)
        logits_indices = self._list_to_cuda(logits_indices, "logits_indices", torch.int64)
        set_context(True, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k,
                    slot_mapping, None, block_tables, logits_indices)
        return input_ids, positions



    # decode阶段单token输出
    def prepare_decode(
        self,
        seqs: list[Sequence],
        *,
        flash_attn_num_splits: int = 0,
    ):         #暂时跳过
        self._kv_offload_pending_dirty_blocks = []
        input_ids = []
        positions = []
        slot_mapping = []
        context_lens = []
        num_blocks_host = []
        decode_write_blocks = []
        decode_write_offsets = []
        for seq in seqs:
            # 上一次输出的最后token
            input_ids.append(seq.last_token)
            # Sequence 已包含本轮输入 token，因此使用它的零基位置。
            positions.append(len(seq) - 1)
            context_lens.append(len(seq))
            num_blocks_host.append(seq.num_blocks)
            decode_write_blocks.append(seq.block_table[-1])
            decode_write_offsets.append(seq.last_block_num_tokens - 1)

        max_blocks = max(len(seq.block_table) for seq in seqs)
        block_table_rows = [seq.block_table + [-1] * (max_blocks - len(seq.block_table)) for seq in seqs]

        cartridge_active = should_use_kv_cartridge(
            context_lens,
            num_blocks_host,
            self.config.kv_cartridge_blocks,
            self.config.kv_cartridge_min_seq_len,
        )
        if cartridge_active:
            block_table_rows, context_lens = compress_decode_block_table_rows(
                block_table_rows,
                context_lens,
                self.block_size,
                self.config.kv_cartridge_blocks,
            )

        am_compact_active = (
            self.config.am_compact_blocks > 0
            and min(context_lens) >= self.config.am_compact_min_seq_len
            and min(context_lens) > self.config.am_compact_blocks
        )
        am_compact_cache_signatures = None
        if am_compact_active:
            am_compact_cache_signatures = tuple(
                tuple(int(block_id) for block_id in row if int(block_id) >= 0)
                for row in block_table_rows
            )

        logical_block_table_rows = [list(row) for row in block_table_rows]
        context_lens_host = list(context_lens)
        if self.kv_offload is not None:
            if self.config.kv_offload_blockwise_decode:
                # Streaming/blockwise decode only stages the current write blocks
                # here. Attention.forward will stage logical read windows layer by
                # layer, so visible blocks may exceed GPU staging slots.
                future_blocks = set(int(block) for block in decode_write_blocks)
                first_write_offset_by_block = {
                    int(block): int(offset)
                    for block, offset in zip(decode_write_blocks, decode_write_offsets)
                }
                _stage_kv_offload_write_blocks(
                    self.kv_offload,
                    decode_write_blocks,
                    first_write_offset_by_block,
                    future_blocks,
                )
            else:
                # Full attention MVP-0/MVP-1: all visible logical blocks are staged
                # on GPU before the forward.
                _stage_kv_offload_full_decode_blocks(
                    self.kv_offload,
                    block_table_rows,
                    decode_write_blocks,
                    decode_write_offsets,
                )
                physical_block_table_rows = self.kv_offload.map_block_rows(block_table_rows)
                block_table_rows = physical_block_table_rows
            slot_mapping = [
                self.kv_offload.logical_to_slot[int(block)] * self.block_size + int(offset)
                for block, offset in zip(decode_write_blocks, decode_write_offsets)
            ]
            self._kv_offload_pending_dirty_blocks.extend(decode_write_blocks)
        else:
            slot_mapping = [
                int(block) * self.block_size + int(offset)
                for block, offset in zip(decode_write_blocks, decode_write_offsets)
            ]

        input_ids = self._list_to_cuda(input_ids, "input_ids", torch.int64)
        positions = self._list_to_cuda(positions, "positions", torch.int64)
        slot_mapping = self._list_to_cuda(slot_mapping, "slot_mapping", torch.int32)
        context_lens = self._list_to_cuda(context_lens, "context_lens", torch.int32)
        block_tables = self.prepare_block_tables_from_rows(block_table_rows)

        # Quest 早返回判定（host 端，避免每层 .item() 触发 GPU sync）：
        #   1) 至少一条 seq 满足 seq_len >= min_seq_len（按 min 算保守）
        #   2) num_blocks > top_k（不然 top-k 退化为 full）
        #   3) **短序列保护**：top_k * block_size 已经 >= 最长 seq * 0.8 时，
        #      Quest 能裁掉的块 <20%，selection overhead 远超收益 → 降级 full attention
        #      （kv-sparse-attention.md §5.5 #4）
        cfg_top_k = self.config.quest_top_k_blocks if not (cartridge_active or am_compact_active) else -1
        cfg_min_len = self.config.quest_min_seq_len
        if cfg_top_k > 0 and seqs:
            min_seq_len_host = min(len(s) for s in seqs)
            min_blocks_host = min(s.num_blocks for s in seqs)
            max_seq_len_host = max(len(s) for s in seqs)
            cover = cfg_top_k * self.block_size  # top-k 已能覆盖的 token 数
            short_seq_skip = cover >= max_seq_len_host * 0.8
            quest_active_top_k = cfg_top_k if (
                min_seq_len_host >= cfg_min_len
                and min_blocks_host > cfg_top_k
                and not short_seq_skip
            ) else -1
        else:
            quest_active_top_k = -1
        set_context(False, slot_mapping=slot_mapping, context_lens=context_lens, block_tables=block_tables,
                    flash_attn_num_splits=flash_attn_num_splits,
                    quest_top_k_blocks=quest_active_top_k,
                    quest_min_seq_len=cfg_min_len,
                    am_compact_blocks=(self.config.am_compact_blocks if am_compact_active else 0),
                    am_compact_selector=self.config.am_compact_selector,
                    am_compact_score_method=self.config.am_compact_score_method,
                    am_compact_beta_bound=self.config.am_compact_beta_bound,
                    am_compact_ridge_lambda=self.config.am_compact_ridge_lambda,
                    am_omp_candidate_pool_size=self.config.am_omp_candidate_pool_size,
                    am_compact_cache_refresh_interval=self.config.am_compact_cache_refresh_interval,
                    am_compact_num_clusters=self.config.am_compact_num_clusters,
                    am_compact_route_top_k=self.config.am_compact_route_top_k,
                    am_compact_num_key_spans=self.config.am_compact_num_key_spans,
                    am_compact_decode_refit=self.config.am_compact_decode_refit,
                    am_compact_decode_refit_mode=self.config.am_compact_decode_refit_mode,
                    am_compact_decode_refit_interval=self.config.am_compact_decode_refit_interval,
                    am_compact_skip_first_layers=self.config.am_compact_skip_first_layers,
                    am_compact_skip_last_layers=self.config.am_compact_skip_last_layers,
                    am_compact_enable_layers=self.config.am_compact_enable_layers,
                    am_compact_layer_stride=self.config.am_compact_layer_stride,
                    am_compact_cache_signatures=am_compact_cache_signatures,
                    kv_offload_manager=self.kv_offload,
                    kv_offload_blockwise_decode=(self.kv_offload is not None and self.config.kv_offload_blockwise_decode),
                    kv_offload_blockwise_blocks=self.config.kv_offload_blockwise_blocks,
                    kv_offload_logical_block_tables=logical_block_table_rows,
                    kv_offload_context_lens=context_lens_host,
                    kv_offload_write_blocks=[int(block) for block in decode_write_blocks])
        return input_ids, positions

    # 生成 temperatures列表，并传到GPU上
    def prepare_sample(self, seqs: list[Sequence]):
        temperatures = []
        for seq in seqs:
            temperatures.append(seq.temperature)
        temperatures = self._list_to_cuda(temperatures, "temperatures", torch.float32)    #pin_memory=True将张量存储在锁定内存（page-locked memory）中，而非普通的可分页内存
        return temperatures
        #普通可分页内存（Pageable Memory）  |	锁定内存（Page-locked Memory / Pinned Memory）
        #操作系统可将其 “分页” 到磁盘       |     被 “锁定” 在物理内存中，不允许换出到磁盘,
        # （swap） ，释放物理内存给其他进程 |     

    def _select_sample_rows(self, logits: torch.Tensor, seqs: list[Sequence],
                            batch_kind: str | None) -> tuple[torch.Tensor, list[Sequence]]:
        if batch_kind != "mixed":
            return logits, seqs
        sample_seqs = [seq for seq in seqs if getattr(seq, "step_do_sample", True)]
        return logits, sample_seqs


    @torch.inference_mode()
    #只需要前向传播 禁用梯度计算（无需反向传播），节省内存；
    # 加速推理过程（跳过与训练相关的检查和操作）。
    def run_model(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        is_prefill: bool,
        input_embeds: torch.Tensor | None = None,
        return_hidden: bool = False,
        execution_mode: AttentionMode | None = None,
        prepare_qwen35_state: bool = False,
        initial_qwen35_candidates=None,
        capture_qwen35_prefix_states: bool = False,
    ):
        mode = execution_mode or get_context().mode
        if mode == "spec_verify" and is_prefill:
            raise ValueError("spec_verify cannot use prefill execution")
        if prepare_qwen35_state:
            return self._run_eager_logits(
                input_ids=input_ids,
                positions=positions,
                input_embeds=input_embeds,
                return_hidden=return_hidden,
                prepare_qwen35_state=True,
                initial_qwen35_candidates=(
                    initial_qwen35_candidates
                ),
                capture_qwen35_prefix_states=(
                    capture_qwen35_prefix_states
                ),
            )
        spec_verify_active = mode == "spec_verify"
        if spec_verify_active:
            context = get_context()
            transaction_authorized = bool(
                getattr(
                    self,
                    "_spec_verify_transaction_authorized",
                    False,
                )
            )
            reason, eager_safe = (
                self._spec_verify_graph_incompatible_reason(
                    input_ids=input_ids,
                    input_embeds=input_embeds,
                    return_hidden=return_hidden,
                    context=context,
                    transaction_authorized=(
                        transaction_authorized
                    ),
                )
            )
            if reason is not None:
                if not eager_safe:
                    raise RuntimeError(reason)
                logits = self._run_eager_logits(
                    input_ids=input_ids,
                    positions=positions,
                    input_embeds=input_embeds,
                    return_hidden=return_hidden,
                )
                self._publish_spec_verify_graph_dispatch_event(
                    identity=None,
                    dispatch="eager",
                    decision=reason,
                    fallback_reason=reason,
                    cache_state="absent",
                    observation_count=0,
                    capture_attempted=False,
                    capture_entry=None,
                    transaction_authorized=(
                        transaction_authorized
                    ),
                )
                return logits
            entry = self._ready_spec_verify_graph_entry(
                input_ids=input_ids,
                context=context,
            )
            if entry is not None:
                try:
                    outputs = self._replay_spec_verify_graph(
                        entry,
                        input_ids=input_ids,
                        positions=positions,
                        context=context,
                    )
                except SpecVerifyGraphReplayError:
                    raise
                except Exception:
                    summary = (
                        self.spec_verify_exact_cuda_graph_cache
                        .summary()
                    )
                    quarantined = dict(
                        summary["quarantined"]
                    )
                    fallback_reason = quarantined.get(
                        entry.identity.sha256,
                        "cache_state_drift",
                    )
                    logits = self._run_eager_logits(
                        input_ids=input_ids,
                        positions=positions,
                        input_embeds=input_embeds,
                        return_hidden=return_hidden,
                    )
                    self._publish_spec_verify_graph_dispatch_event(
                        identity=entry.identity,
                        dispatch="eager",
                        decision=fallback_reason,
                        fallback_reason=fallback_reason,
                        cache_state=(
                            "quarantined"
                            if entry.identity.sha256 in quarantined
                            else "absent"
                        ),
                        observation_count=(
                            self.spec_verify_exact_cuda_graph_cache
                            .observation_counts.get(
                                entry.identity.sha256,
                                0,
                            )
                        ),
                        capture_attempted=False,
                        capture_entry=None,
                        transaction_authorized=True,
                    )
                    return logits
                logits = self.model.compute_logits(outputs)
                observation_count = (
                    self.spec_verify_exact_cuda_graph_cache
                    .observation_counts.get(
                        entry.identity_sha256,
                        0,
                    )
                )
                self._publish_spec_verify_graph_dispatch_event(
                    identity=entry.identity,
                    dispatch="graph",
                    decision="hit",
                    fallback_reason=None,
                    cache_state="ready",
                    observation_count=observation_count,
                    capture_attempted=False,
                    capture_entry=None,
                    transaction_authorized=True,
                )
                return logits
            outputs = self.model(
                input_ids,
                positions,
                input_embeds=input_embeds,
            )
            try:
                identity = self._build_spec_verify_graph_identity(
                    input_ids=input_ids,
                    outputs=outputs,
                    context=context,
                )
            except ValueError:
                logits = self.model.compute_logits(outputs)
                self._publish_spec_verify_graph_dispatch_event(
                    identity=None,
                    dispatch="eager",
                    decision="identity_invalid",
                    fallback_reason="identity_invalid",
                    cache_state="absent",
                    observation_count=0,
                    capture_attempted=False,
                    capture_entry=None,
                    transaction_authorized=True,
                )
                return logits
            decision = (
                self.spec_verify_exact_cuda_graph_cache.observe_success(
                    identity,
                    estimated_static_bytes=(
                        self._estimate_spec_verify_graph_static_bytes(
                            identity
                        )
                    ),
                    step_id=(
                        getattr(
                            self,
                            "_spec_verify_cuda_graph_step_id",
                            0,
                        )
                        + 1
                    ),
                )
            )
            capture_entry = None
            if decision.should_capture:
                capture_entry = (
                    self._attempt_post_step_spec_verify_capture(
                        identity=identity,
                        live_input_ids=input_ids,
                        live_positions=positions,
                        live_context=context,
                    )
                )
            summary = (
                self.spec_verify_exact_cuda_graph_cache.summary()
            )
            quarantined = dict(summary["quarantined"])
            fallback_reason = quarantined.get(
                identity.sha256,
                decision.fallback_reason,
            )
            cache_state = (
                "quarantined"
                if identity.sha256 in quarantined
                else (
                    "ready"
                    if (
                        identity.sha256
                        in summary["ready_entries"]
                    )
                    else decision.cache_state
                )
            )
            self._publish_spec_verify_graph_dispatch_event(
                identity=identity,
                dispatch="eager",
                decision=decision.decision,
                fallback_reason=fallback_reason,
                cache_state=cache_state,
                observation_count=decision.observation_count,
                capture_attempted=decision.should_capture,
                capture_entry=capture_entry,
                transaction_authorized=True,
            )
            return self.model.compute_logits(outputs)
        # Quest 实际启用时（context 已确认）才走 eager；否则照常走 cuda graph
        quest_active = mode == "decode" and (get_context().quest_top_k_blocks > 0)
        am_active = mode == "decode" and (get_context().am_compact_blocks > 0)
        # C4：decode 反量化每步都要 alloc，cuda graph 无法 replay，强制 eager
        c4_active = self.config.kv_quant_bits == 4
        # cpu_offload：init 阶段已跳过 capture，这里也必须走 eager（否则 self.graphs 不存在）
        offload_active = self.config.cpu_offload
        kv_offload_active = self.config.kv_offload_mvp0
        # FlashAttention decode replay is only correctness-validated for one
        # sequence. Multi-sequence captured graphs can corrupt rows after the
        # first one, so keep the batch-1 graph fast path and fail closed to
        # eager execution for larger decode batches.
        multi_sequence_decode = mode == "decode" and input_ids.size(0) > 1
        legacy_graph_state_absent = not all(
            hasattr(self, name)
            for name in ("graphs", "graph_bs", "graph_vars")
        )
        if input_ids.size(0) > 1:
            context = get_context()
            reason = self._multi_sequence_graph_incompatible_reason(
                mode=mode,
                is_prefill=is_prefill,
                input_embeds=input_embeds,
                return_hidden=return_hidden,
            )
            if reason is not None:
                logits = self._run_eager_logits(
                    input_ids=input_ids,
                    positions=positions,
                    input_embeds=input_embeds,
                    return_hidden=return_hidden,
                )
                page_table_width = (
                    None
                    if context.block_tables is None
                    else int(context.block_tables.size(1))
                )
                self._publish_cuda_graph_dispatch_event(
                    mode=mode,
                    active_batch_size=int(input_ids.size(0)),
                    page_table_width=page_table_width,
                    effective_num_splits=None,
                    graph_identity_sha256=None,
                    dispatch="eager",
                    cache_state="absent",
                    observation_count=0,
                    fallback_reason=reason,
                    capture_attempted=False,
                )
                return logits
            try:
                identity = self._build_multi_sequence_graph_identity(
                    input_ids,
                    context,
                )
            except (ValueError, RuntimeError):
                logits = self._run_eager_logits(
                    input_ids=input_ids,
                    positions=positions,
                    input_embeds=input_embeds,
                    return_hidden=return_hidden,
                )
                self._publish_cuda_graph_dispatch_event(
                    mode=mode,
                    active_batch_size=int(input_ids.size(0)),
                    page_table_width=None,
                    effective_num_splits=None,
                    graph_identity_sha256=None,
                    dispatch="eager",
                    cache_state="absent",
                    observation_count=0,
                    fallback_reason="identity_invalid",
                    capture_attempted=False,
                )
                return logits
            if (
                identity.active_batch_size
                not in self.config.multi_sequence_cuda_graph_batch_allowlist
            ):
                logits = self._run_eager_logits(
                    input_ids=input_ids,
                    positions=positions,
                    input_embeds=input_embeds,
                    return_hidden=return_hidden,
                )
                self._publish_cuda_graph_dispatch_event(
                    mode=mode,
                    active_batch_size=identity.active_batch_size,
                    page_table_width=identity.page_table_width,
                    effective_num_splits=(
                        identity.effective_num_splits
                    ),
                    graph_identity_sha256=identity.sha256,
                    dispatch="eager",
                    cache_state="absent",
                    observation_count=0,
                    fallback_reason="batch_not_allowlisted",
                    capture_attempted=False,
                )
                return logits
            entry = self.exact_cuda_graph_cache.ready_entry(identity)
            if entry is not None:
                observation_count = (
                    self.exact_cuda_graph_cache.observation_counts.get(
                        identity.sha256,
                        0,
                    )
                )
                try:
                    logits = self._replay_exact_multi_sequence_graph(
                        entry,
                        input_ids=input_ids,
                        positions=positions,
                        context=context,
                    )
                except Exception:
                    self._publish_cuda_graph_dispatch_event(
                        mode=mode,
                        active_batch_size=identity.active_batch_size,
                        page_table_width=identity.page_table_width,
                        effective_num_splits=(
                            identity.effective_num_splits
                        ),
                        graph_identity_sha256=identity.sha256,
                        dispatch="graph",
                        cache_state="rejected",
                        observation_count=observation_count,
                        fallback_reason="replay_disabled",
                        capture_attempted=False,
                    )
                    raise
                self._publish_cuda_graph_dispatch_event(
                    mode=mode,
                    active_batch_size=identity.active_batch_size,
                    page_table_width=identity.page_table_width,
                    effective_num_splits=(
                        identity.effective_num_splits
                    ),
                    graph_identity_sha256=identity.sha256,
                    dispatch="graph",
                    cache_state="ready",
                    observation_count=observation_count,
                    fallback_reason=None,
                    capture_attempted=False,
                )
                return logits
            logits = self._run_eager_logits(
                input_ids=input_ids,
                positions=positions,
                input_embeds=input_embeds,
                return_hidden=return_hidden,
            )
            decision = self.exact_cuda_graph_cache.observe_success(
                identity,
                estimated_static_bytes=(
                    self._estimate_exact_graph_static_bytes(
                        batch_size=identity.active_batch_size,
                        page_table_width=identity.page_table_width,
                    )
                ),
            )
            capture_entry = None
            if decision.should_capture:
                capture_entry = self._attempt_post_step_capture(
                    identity=identity,
                    input_ids=input_ids,
                    positions=positions,
                    context=context,
                )
            summary = self.exact_cuda_graph_cache.summary()
            fallback_reason = summary["rejected"].get(
                identity.sha256,
                decision.fallback_reason,
            )
            cache_state = (
                "rejected"
                if identity.sha256 in summary["rejected"]
                else decision.cache_state
            )
            self._publish_cuda_graph_dispatch_event(
                mode=mode,
                active_batch_size=identity.active_batch_size,
                page_table_width=identity.page_table_width,
                effective_num_splits=identity.effective_num_splits,
                graph_identity_sha256=identity.sha256,
                dispatch="eager",
                cache_state=cache_state,
                observation_count=decision.observation_count,
                fallback_reason=fallback_reason,
                capture_attempted=decision.should_capture,
                capture_entry=capture_entry,
            )
            return logits
        if (is_prefill or spec_verify_active or self.enforce_eager or multi_sequence_decode
                or quest_active or am_active or c4_active or offload_active or kv_offload_active
                or legacy_graph_state_absent or input_embeds is not None
                or return_hidden):     #动态执行 eager mode
            return self._run_eager_logits(
                input_ids=input_ids,
                positions=positions,
                input_embeds=input_embeds,
                return_hidden=return_hidden,
            )
        else:           #静态执行  graph replay
            bs = input_ids.size(0)
            context = get_context()
            graph = self.graphs[next (x for x in self.graph_bs if x >= bs)]
            graph_vars = self.graph_vars
            for k, v in graph_vars.items():
                if k != "outputs":
                    v.zero_()
            graph_vars["input_ids"][:bs] = input_ids
            graph_vars["positions"][:bs] = positions
            graph_vars["slot_mapping"][:bs] = context.slot_mapping
            graph_vars["context_lens"][:bs] = context.context_lens
            graph_vars["block_tables"][:bs, :context.block_tables.size(1)] = context.block_tables
            graph.replay()
            return self.model.compute_logits(graph_vars["outputs"][:bs])


    def _run_model_step(self, seqs:list[Sequence], is_prefill: bool, do_sample: bool = True,
            batch_kind: str | None = None,
            released_hybrid_state_leases: tuple[HybridStateLease, ...] = (),
            ) -> list[int] | None:
        self._prepare_hybrid_state_batch(
            seqs,
            released_hybrid_state_leases,
        )
        self._last_hybrid_state_leases = tuple(
            HybridStateLease(
                slot_id=seq.hybrid_state_slot_id,
                generation=seq.hybrid_state_generation,
                request_id=int(seq.seq_id),
            )
            for seq in seqs
            if seq.hybrid_state_slot_id >= 0
        )
        self._last_hybrid_state_token_counts = (
            _qwen35_step_token_counts(
                seqs,
                is_prefill=is_prefill,
                batch_kind=batch_kind,
            )
            if self._last_hybrid_state_leases
            else ()
        )
        request_ids = sorted(int(seq.seq_id) for seq in seqs)
        self._cuda_graph_request_ids_hash = hashlib.sha256(
            json.dumps(
                request_ids,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        observe_proposal_prefill = (
            is_prefill
            and self._proposal_prefill_observation_required()
        )
        if (
            observe_proposal_prefill
            and batch_kind == "mixed"
        ):
            raise ValueError(
                "proposal prefill observation does not support "
                "mixed batches"
            )
        if batch_kind == "mixed":
            input_ids, positions = self.prepare_mixed(seqs)
        else:
            input_ids, positions = self.prepare_prefill(seqs) if is_prefill else self.prepare_decode(seqs)
        self._kv_offload_before_forward()
        if observe_proposal_prefill:
            outputs = self.run_model(
                input_ids,
                positions,
                is_prefill,
                return_hidden=True,
            )
            logits, target_hidden = outputs
            self._observe_proposal_target_prefill(
                seqs,
                positions,
                target_hidden,
                batch_kind=batch_kind,
            )
        else:
            logits = self.run_model(
                input_ids,
                positions,
                is_prefill,
            )
        self._capture_qwen35_recurrent_source_state(
            seqs,
            is_prefill=is_prefill,
            batch_kind=batch_kind,
        )
        _round_qwen35_final_prefill_recurrent_states(
            self.hybrid_state_runtime_bridge,
            seqs,
            self._last_hybrid_state_leases,
            is_prefill=is_prefill,
            batch_kind=batch_kind,
        )
        self._kv_offload_after_forward()
        if not do_sample:
            self._last_step_logits_cpu = None
            reset_context()
            return None
        if self.rank == 0:
            logits, sample_seqs = self._select_sample_rows(logits, seqs, batch_kind)
            if (
                self._spec_verify_trace.enabled
                and not is_prefill
                and batch_kind != "mixed"
            ):
                self._spec_verify_trace.record_rows(
                    stage="ordinary_decode",
                    execution_mode="decode",
                    sequence_ids=tuple(
                        int(seq.seq_id)
                        for seq in sample_seqs
                    ),
                    query_offset=0,
                    query_len=1,
                    input_tokens=tuple(
                        int(seq.last_token)
                        for seq in sample_seqs
                    ),
                    positions=tuple(
                        int(seq.num_tokens - 1)
                        for seq in sample_seqs
                    ),
                    prediction_indices=tuple(
                        int(seq.num_completion_tokens)
                        for seq in sample_seqs
                    ),
                    logical_block_identities=tuple(
                        self._trace_block_identities(
                            seq.block_table
                        )
                        for seq in sample_seqs
                    ),
                    logits=logits,
                )
            if self._record_step_logits:
                self._last_step_logits_cpu = logits.detach().float().cpu()
            else:
                self._last_step_logits_cpu = None
            temperatures = self.prepare_sample(sample_seqs)    #只有主进程做采样
            token_ids = self.sampler(logits, temperatures).tolist()
        else:
            self._last_step_logits_cpu = None
            token_ids = None
        reset_context()
        return token_ids

    def configure_decode_internal_profile(
        self,
        enabled: bool,
        profile_label: str,
    ):
        if not isinstance(enabled, bool):
            raise ValueError(
                "decode internal profile enabled must be a boolean"
            )
        if (
            not isinstance(profile_label, str)
            or not profile_label
        ):
            raise ValueError(
                "decode internal profile label must be non-empty"
            )
        self.decode_internal_profiler = (
            DecodeInternalProfiler(
                rank=self.rank,
                event_factory=lambda: torch.cuda.Event(
                    enable_timing=True,
                ),
                synchronize=torch.cuda.synchronize,
                nvtx_range_factory=torch.cuda.nvtx.range,
                profile_label=profile_label,
            )
            if enabled
            else DecodeInternalProfiler.disabled(rank=self.rank)
        )
        return {
            "rank": self.rank,
            "enabled": enabled,
            "profile_label": profile_label,
        }

    def finalize_decode_internal_profile(self):
        return self.decode_internal_profiler.finalize()

    def run(self, seqs:list[Sequence], is_prefill: bool, do_sample: bool = True,
            batch_kind: str | None = None,
            released_hybrid_state_leases: tuple[HybridStateLease, ...] = (),
            kv_block_identity_rows: tuple[KVBlockIdentityRow, ...] = (),
            ) -> list[int] | None:
        self.bind_kv_block_identity_rows(
            tuple(seqs),
            kv_block_identity_rows,
        )
        request_ids = sorted(int(seq.seq_id) for seq in seqs)
        request_set_sha256 = hashlib.sha256(
            json.dumps(
                request_ids,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        is_decode = (
            not is_prefill
            if batch_kind != "mixed"
            else all(
                bool(getattr(seq, "step_is_decode", False))
                for seq in seqs
            )
        )
        return run_profiled_step(
            self.decode_internal_profiler,
            batch_kind=(
                batch_kind
                or ("prefill" if is_prefill else "decode")
            ),
            is_decode=is_decode,
            active_sequence_count=len(seqs),
            request_set_sha256=request_set_sha256,
            dispatch=(
                "eager"
                if is_prefill or len(seqs) > 1 or self.enforce_eager
                else "graph_or_eager"
            ),
            call=lambda: self._run_model_step(
                seqs,
                is_prefill,
                do_sample,
                batch_kind,
                released_hybrid_state_leases,
            ),
        )

    @torch.inference_mode()
    def capture_cudagraph(self):
        config = self.config
        hf_config = config.hf_config
        max_bs = min(self.config.max_num_seqs, 512)        # 这里的 max_batch_size默认了seq_len = 1, 因此 batch_size * seq_len = max_bs
        max_num_blocks = (config.max_model_len + self.block_size - 1) // self.block_size
        input_ids = torch.zeros(max_bs, dtype=torch.int64)
        positions = torch.zeros(max_bs, dtype=torch.int64)
        slot_mapping = torch.zeros(max_bs, dtype=torch.int32)
        context_lens = torch.zeros(max_bs, dtype=torch.int32)
        block_tables = torch.zeros(max_bs, max_num_blocks, dtype=torch.int32)
        outputs = torch.zeros(max_bs, hf_config.hidden_size)
        if config.multi_sequence_cuda_graphs:
            self.graph_bs = [1]
        else:
            self.graph_bs = [1, 2, 4, 8] + list(range(16, max_bs + 1, 16))      # 捕捉各种batch_size的cuda graph
        self.graphs = {}
        self.graph_pool = None

        # decode 阶段
        for bs in reversed(self.graph_bs):
            graph = torch.cuda.CUDAGraph()
            set_context(False, slot_mapping=slot_mapping[:bs], context_lens=context_lens[:bs], block_tables=block_tables[:bs, :])
            outputs[:bs] = self.model(input_ids[:bs], positions[:bs])       # warm up
            with torch.cuda.graph(graph, self.graph_pool):                  # 开始 capture
                outputs[:bs] = self.model(input_ids[:bs], positions[:bs])
            if self.graph_pool is None:
                self.graph_pool = graph.pool()
            self.graphs[bs] = graph
            torch.cuda.synchronize()
            reset_context()

        self.graph_vars = dict(
            input_ids=input_ids, 
            positions=positions,
            slot_mapping=slot_mapping,
            context_lens=context_lens,
            block_tables=block_tables,
            outputs=outputs
        )
