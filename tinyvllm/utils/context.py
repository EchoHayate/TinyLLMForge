from dataclasses import dataclass
import torch


@dataclass
class Context:
    is_prefill: bool = False    #prefill or decode
    cu_seqlens_q: torch.Tensor | None = None    #prefill        累积序列长度 记录批量中每个序列的起始和结束位置。cu_seqlens_q = [0, 3, 8]
    cu_seqlens_k: torch.Tensor | None = None    #prefill
    max_seqlen_q: int = 0   #prefill
    max_seqlen_k: int = 0   #prefill
    slot_mapping: torch.Tensor | None = None    #prefill or decode
    context_lens: torch.Tensor | None = None    #decode
    block_tables: torch.Tensor | None = None    #prefill or decode
    logits_indices: torch.Tensor | None = None  #prefill: rows in flattened hidden states that need logits
    # Quest 动态稀疏 attention：>0 表示开启，每个 query 只看 top-k block
    quest_top_k_blocks: int = -1
    quest_min_seq_len: int = 0
    # Attention Matching KV compaction：>0 表示 decode eager 路径按 AM 生成 C_k/beta/C_v
    am_compact_blocks: int = 0
    am_compact_selector: str = "highest"
    am_compact_score_method: str = "rms"
    am_compact_beta_bound: float = 3.0
    am_compact_ridge_lambda: float = 1e-6
    am_omp_candidate_pool_size: int = 0
    am_compact_cache_refresh_interval: int = 0
    am_prefill_cache_ref_query_stride: int = 8
    am_compact_num_clusters: int = 1
    am_compact_route_top_k: int = 1
    am_compact_num_key_spans: int = 1
    am_compact_decode_refit: bool = False
    am_compact_decode_refit_mode: str = "full"
    am_compact_decode_refit_interval: int = 1
    am_compact_skip_first_layers: int = 0
    am_compact_skip_last_layers: int = 0
    am_compact_enable_layers: tuple[int, ...] | None = None
    am_compact_layer_stride: int = 1
    am_compact_cache_signatures: tuple | None = None
    # KV offload blockwise decode：attention 层按 logical block window 触发 staging。
    kv_offload_manager: object | None = None
    kv_offload_blockwise_decode: bool = False
    kv_offload_blockwise_prefill: bool = False
    kv_offload_blockwise_blocks: int = 1
    kv_offload_logical_block_tables: list[list[int]] | None = None
    kv_offload_context_lens: list[int] | None = None
    kv_offload_write_blocks: list[int] | None = None
    kv_offload_prefill_chunk_starts: list[int] | None = None
    kv_offload_prefill_chunk_ends: list[int] | None = None

_CONTEXT = Context()

def get_context():
    return _CONTEXT


def am_compact_layer_enabled(context: Context, layer_idx: int, num_hidden_layers: int = 0) -> bool:
    """Return whether AM compact decode should run on a decoder layer.

    `am_compact_enable_layers` is an explicit allow-list and takes precedence
    over skip/stride rules. A negative `layer_idx` preserves old behavior for
    attention modules that have not been annotated by ModelRunner yet.
    """
    if context.am_compact_blocks <= 0:
        return False
    layer_idx = int(layer_idx)
    if layer_idx < 0:
        return True
    enable_layers = context.am_compact_enable_layers
    if enable_layers is not None:
        return layer_idx in enable_layers
    skip_first = int(context.am_compact_skip_first_layers)
    if layer_idx < skip_first:
        return False
    skip_last = int(context.am_compact_skip_last_layers)
    num_hidden_layers = int(num_hidden_layers)
    if num_hidden_layers > 0 and skip_last > 0 and layer_idx >= num_hidden_layers - skip_last:
        return False
    stride = max(1, int(context.am_compact_layer_stride))
    return ((layer_idx - skip_first) % stride) == 0


def am_compact_enabled_layers(context: Context, num_hidden_layers: int) -> tuple[int, ...]:
    """Return all decoder layer indices where AM compact decode is enabled."""
    return tuple(
        layer_idx for layer_idx in range(int(num_hidden_layers))
        if am_compact_layer_enabled(context, layer_idx, num_hidden_layers)
    )


def set_context(is_prefill, cu_seqlens_q=None, cu_seqlens_k=None, max_seqlen_q=0, max_seqlen_k=0,
                slot_mapping=None, context_lens=None, block_tables=None,
                logits_indices=None,
                quest_top_k_blocks: int = -1, quest_min_seq_len: int = 0,
                am_compact_blocks: int = 0, am_compact_selector: str = "highest",
                am_compact_score_method: str = "rms",
                am_compact_beta_bound: float = 3.0, am_compact_ridge_lambda: float = 1e-6,
                am_omp_candidate_pool_size: int = 0,
                am_compact_cache_refresh_interval: int = 0,
                am_prefill_cache_ref_query_stride: int = 8,
                am_compact_num_clusters: int = 1,
                am_compact_route_top_k: int = 1,
                am_compact_num_key_spans: int = 1,
                am_compact_decode_refit: bool = False,
                am_compact_decode_refit_mode: str = "full",
                am_compact_decode_refit_interval: int = 1,
                am_compact_skip_first_layers: int = 0,
                am_compact_skip_last_layers: int = 0,
                am_compact_enable_layers: tuple[int, ...] | list[int] | None = None,
                am_compact_layer_stride: int = 1,
                am_compact_cache_signatures: tuple | None = None,
                kv_offload_manager: object | None = None,
                kv_offload_blockwise_decode: bool = False,
                kv_offload_blockwise_prefill: bool = False,
                kv_offload_blockwise_blocks: int = 1,
                kv_offload_logical_block_tables: list[list[int]] | None = None,
                kv_offload_context_lens: list[int] | None = None,
                kv_offload_write_blocks: list[int] | None = None,
                kv_offload_prefill_chunk_starts: list[int] | None = None,
                kv_offload_prefill_chunk_ends: list[int] | None = None):
    global _CONTEXT
    if am_compact_enable_layers is not None and not isinstance(am_compact_enable_layers, tuple):
        am_compact_enable_layers = tuple(int(x) for x in am_compact_enable_layers)
    _CONTEXT = Context(
        is_prefill=is_prefill,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        slot_mapping=slot_mapping,
        context_lens=context_lens,
        block_tables=block_tables,
        logits_indices=logits_indices,
        quest_top_k_blocks=quest_top_k_blocks,
        quest_min_seq_len=quest_min_seq_len,
        am_compact_blocks=am_compact_blocks,
        am_compact_selector=am_compact_selector,
        am_compact_score_method=am_compact_score_method,
        am_compact_beta_bound=am_compact_beta_bound,
        am_compact_ridge_lambda=am_compact_ridge_lambda,
        am_omp_candidate_pool_size=am_omp_candidate_pool_size,
        am_compact_cache_refresh_interval=am_compact_cache_refresh_interval,
        am_prefill_cache_ref_query_stride=am_prefill_cache_ref_query_stride,
        am_compact_num_clusters=am_compact_num_clusters,
        am_compact_route_top_k=am_compact_route_top_k,
        am_compact_num_key_spans=am_compact_num_key_spans,
        am_compact_decode_refit=am_compact_decode_refit,
        am_compact_decode_refit_mode=am_compact_decode_refit_mode,
        am_compact_decode_refit_interval=am_compact_decode_refit_interval,
        am_compact_skip_first_layers=am_compact_skip_first_layers,
        am_compact_skip_last_layers=am_compact_skip_last_layers,
        am_compact_enable_layers=am_compact_enable_layers,
        am_compact_layer_stride=am_compact_layer_stride,
        am_compact_cache_signatures=am_compact_cache_signatures,
        kv_offload_manager=kv_offload_manager,
        kv_offload_blockwise_decode=kv_offload_blockwise_decode,
        kv_offload_blockwise_prefill=kv_offload_blockwise_prefill,
        kv_offload_blockwise_blocks=kv_offload_blockwise_blocks,
        kv_offload_logical_block_tables=kv_offload_logical_block_tables,
        kv_offload_context_lens=kv_offload_context_lens,
        kv_offload_write_blocks=kv_offload_write_blocks,
        kv_offload_prefill_chunk_starts=kv_offload_prefill_chunk_starts,
        kv_offload_prefill_chunk_ends=kv_offload_prefill_chunk_ends,
    )

def reset_context():
    global _CONTEXT              #声明为global变量非常重要 否则将导致修改无效
    _CONTEXT = Context()
