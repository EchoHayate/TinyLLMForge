import os
from dataclasses import dataclass
from transformers import AutoConfig


def _normalize_positive_int_tuple(
    value,
    *,
    name,
    allow_empty,
):
    if not isinstance(value, (tuple, list)):
        raise ValueError(f"{name} must be a tuple or list")
    if not value and not allow_empty:
        raise ValueError(f"{name} must be non-empty")
    if any(
        isinstance(item, bool)
        or not isinstance(item, int)
        or item <= 0
        for item in value
    ):
        raise ValueError(
            f"{name} must contain positive non-boolean integers"
        )
    return tuple(sorted(set(value)))


@dataclass   #自动配置__init__方法
class Config:
    model: str                                          # sequence includes multi-tokens   batch includes multi-sequences
    max_num_batched_tokens: int = 16384                 # 2^14 = [batch_size * seq_len]  单个批次（batch）中，所有 sequence 包含的 token 总数的最大限制 
    max_num_seqs: int = 512                             # 可以同时并行处理的最大sequence 数量                  
    max_model_len: int = 4096                           #模型可以处理的最大sequence长度
    gpu_memory_utilization: float = 0.9                 #gpu利用率 可以用来确定实际 kv cache大小
    tensor_parallel_size: int = 1                       #并行计算gpu的个数
    enforce_eager: bool = False                         # True表示以即时执行模式推理，用于debug   false表示启用cuda graph  cuda graph开启后减少kernal launch时间 可用于吞吐量测试
    replay_aware_decode_metadata: bool = False
    zero_temperature_greedy_fast_path: bool = False
    graph_resident_greedy_tail: bool = False
    exact_greedy_decode_burst: bool = False
    exact_greedy_decode_burst_continuation: bool = False
    exact_greedy_decode_burst_split_phase: bool = False
    exact_greedy_decode_burst_ragged_coalescing: bool = False
    exact_greedy_decode_burst_lease_local_delta_journal: bool = False
    exact_greedy_decode_burst_generation_sealed_identity: bool = False
    exact_greedy_decode_burst_medium_split_k: bool = False
    exact_greedy_decode_burst_elastic_k16: bool = False
    exact_greedy_decode_burst_octet_folded_graph: bool = False
    exact_greedy_decode_burst_tokens: int = 4
    prefill_cuda_graphs: bool = False
    prefill_cuda_graph_token_allowlist: tuple = (256, 2048)
    phase_stitch_profile: bool = False
    phase_stitched_exact_graph_runtime: bool = False
    multi_sequence_cuda_graphs: bool = False
    multi_sequence_cuda_graph_batch_allowlist: tuple = (2, 4, 8)
    multi_sequence_cuda_graph_min_observations: int = 3
    multi_sequence_cuda_graph_max_entries: int = 8
    multi_sequence_cuda_graph_max_static_bytes: int = 64 * 1024 * 1024
    multi_sequence_cuda_graph_max_reserved_bytes: int = 512 * 1024 * 1024
    multi_sequence_cuda_graph_max_total_capture_ns: int = 5_000_000_000
    multi_sequence_cuda_graph_max_single_capture_ns: int = 2_000_000_000
    spec_verify_cuda_graphs: bool = False
    spec_verify_cuda_graph_batch_allowlist: tuple = (1, 4)
    spec_verify_cuda_graph_query_len_allowlist: tuple = ()
    spec_verify_cuda_graph_min_observations: int = 2
    spec_verify_cuda_graph_max_entries: int = 8
    spec_verify_cuda_graph_max_static_bytes: int = 64 * 1024 * 1024
    spec_verify_cuda_graph_max_reserved_bytes: int = 512 * 1024 * 1024
    spec_verify_cuda_graph_max_total_capture_ns: int = 5_000_000_000
    spec_verify_cuda_graph_max_single_capture_ns: int = 2_000_000_000
    qwen35_mtp_cuda_graphs: bool = False
    qwen35_mtp_cuda_graph_q_allowlist: tuple = (2, 3, 4)
    qwen35_mtp_cuda_graph_batch_allowlist: tuple = (1, 2, 4)
    qwen35_mtp_cuda_graph_min_observations: int = 2
    qwen35_mtp_cuda_graph_max_entries: int = 8
    qwen35_mtp_cuda_graph_max_static_bytes: int = 64 * 1024 * 1024
    qwen35_mtp_cuda_graph_max_reserved_bytes: int = 512 * 1024 * 1024
    qwen35_mtp_cuda_graph_max_total_capture_ns: int = 5_000_000_000
    qwen35_mtp_cuda_graph_max_single_capture_ns: int = 2_000_000_000
    qwen35_mtp_enabled: bool = False
    qwen35_mtp_max_proposal_tokens: int = 4
    proposal_kv_offload_enabled: bool = False
    proposal_kv_logical_entry_capacity: int = 0
    proposal_kv_gpu_slot_capacity: int = 0
    proposal_kv_cpu_backing_capacity: int = 0
    proposal_kv_async_copy: bool = True
    proposal_kv_batch_copy: bool = True
    autoregressive_draft_enabled: bool = False
    autoregressive_draft_model: str | None = None
    autoregressive_draft_backend: str = "qwen3"
    autoregressive_draft_max_proposal_tokens: int = 4
    autoregressive_draft_gpu_slot_capacity: int = 0
    autoregressive_draft_proposal_kv_offload_enabled: bool = False
    autoregressive_draft_logical_entry_capacity: int = 0
    autoregressive_draft_cpu_backing_capacity: int = 0
    autoregressive_draft_cuda_graphs: bool = False
    autoregressive_draft_cuda_graph_q_allowlist: tuple = (4,)
    autoregressive_draft_cuda_graph_batch_allowlist: tuple = (4,)
    autoregressive_draft_cuda_graph_min_observations: int = 2
    autoregressive_draft_cuda_graph_max_entries: int = 4
    autoregressive_draft_cuda_graph_max_static_bytes: int = 64 * 1024 * 1024
    autoregressive_draft_cuda_graph_max_reserved_bytes: int = 512 * 1024 * 1024
    autoregressive_draft_cuda_graph_max_total_capture_ns: int = 5_000_000_000
    autoregressive_draft_cuda_graph_max_single_capture_ns: int = 4_000_000_000
    autoregressive_draft_command_timeline: bool = False
    autoregressive_draft_command_timeline_max_rows: int = 8192
    hf_config: AutoConfig | None = None                 # hugging face config, 加载模型的层数，隐藏层数，注意力头数
    eos: int  = -1                                      # end of sentence, 使用模型默认的句子结束符
    kvcache_block_size: int = 256                       
    num_kvcache_blocks: int = -1                        #-1代表自动计算  num_kvcache_blocks=kv cache/kvcache_block_size 

    # 量化 / cpu-offload 相关配置
    quantization: str | None = None                     # None | "int8" | "int8_bnb" | "int4" | "int2"  线性层权重量化方式
    quant_group_size: int = 128                         # 分组量化的组大小（更小的组 -> 更高精度但额外开销大）
    cpu_offload: bool = False                           # 是否启用 cpu-offload (decoder layer 粒度)
    cpu_offload_num_layers: int = -1                    # 卸载到 cpu 的 decoder 层数，-1 表示除最后 2 层外全部卸载
    qwen35_hybrid_prefix_representation: str = "exact_restore"

    # 动态稀疏 attention（Quest，page-level top-k）相关配置
    quest_top_k_blocks: int = -1                        # decode 时每个 query 选择的 block 数，-1 表示关闭 Quest
    quest_min_seq_len: int = 1024                       # 序列长度小于此值时退化为 full attention

    # KV-Cartridge v0：无训练、read-side KV block 压缩。>0 表示 decode 阶段只保留这么多历史 block。
    kv_cartridge_blocks: int = 0                        # 0 表示关闭；>0 时启用 uniform cartridge block table
    kv_cartridge_min_seq_len: int = 1024                # 序列长度小于此值时退化为 full attention
    kv_cartridge_mode: str = "uniform"                  # v0 仅支持 uniform：保首尾，中间均匀抽样

    # Attention Matching compact decode：实验性 eager-only 路径，用 C_k/beta/C_v 替代完整 KV attention
    am_compact_blocks: int = 0                          # 0 表示关闭；>0 时每个 KV head 生成这么多 compact KV
    am_compact_selector: str = "highest"                  # highest / omp：AM compact key 选择器
    am_compact_min_seq_len: int = 1024                  # 序列长度小于此值时退化为 full attention
    am_compact_score_method: str = "rms"                # AM-HighestAttnKeys score: rms / mean / max
    am_compact_beta_bound: float = 3.0                  # beta box bound：[-bound, bound]
    am_compact_ridge_lambda: float = 1e-6               # C_v least-squares ridge
    am_omp_candidate_pool_size: int = 0                 # 0 表示按 max(2*b, b+4) 自动选择 OMP 候选池
    am_compact_cache_refresh_interval: int = 0          # >0 时复用 compact KV N 个 decode step 后刷新
    am_prefill_cache_ref_query_stride: int = 8          # prefill 构建 compact cache 时每隔 N 个 query 取参考
    am_compact_num_clusters: int = 1                    # prefill 持久 compact cache 的 query cluster 数
    am_compact_route_top_k: int = 1                     # decode 时 ensemble 最近的 N 个 compact clusters
    am_compact_num_key_spans: int = 1                   # prefill 按连续 key span 构建局部 compact bank 数
    am_compact_decode_refit: bool = False               # decode 时用当前 query 重拟合 cached indices 的 beta/C_v
    am_compact_decode_refit_mode: str = "full"          # full / direct / beta：decode refit 的 C_v 策略
    am_compact_decode_refit_interval: int = 1           # decode refit 后复用 N 个 decode step；1 表示每步 refit
    am_compact_skip_first_layers: int = 0               # 前 N 层不启用 AM，走 baseline FlashAttention
    am_compact_skip_last_layers: int = 0                # 后 N 层不启用 AM，走 baseline FlashAttention
    am_compact_enable_layers: list[int] | None = None   # 显式指定启用 AM 的层；None 表示按 skip/stride 规则
    am_compact_layer_stride: int = 1                    # 每隔 N 层启用 AM；1 表示所有未 skip 的层启用

    # Chunked prefill：把长 prompt 的 prefill 拆成多个小步，避免单个超长 prefill 长时间占住调度器
    max_num_prefill_tokens_per_step: int = 0             # 0 表示关闭；>0 时每次 prefill step 最多处理这么多 prompt token
    chunked_prefill_decode_first: bool = True            # 已有 decode 请求时优先 decode，避免被新长 prompt prefill 阻塞
    chunked_prefill_max_consecutive_chunks: int = 0       # >0 时 prefill 连续 N 个 chunk 后若有 running decode，则让出 1 次 decode
    chunked_prefill_mixed_batch: bool = False             # True 时允许一个 prefill chunk 和 running decode 走同一次 varlen prefill forward
    chunked_prefill_mixed_min_prompt_tokens: int = 0       # >0 时 mixed 只接纳剩余 prompt token >= 该阈值的 prefill，短 prompt 等 decode 空闲后再 prefill
    chunked_prefill_adaptive_mixed: bool = False
    chunked_prefill_adaptive_enter_waiting: int = 8
    chunked_prefill_adaptive_exit_waiting: int = 2
    chunked_prefill_adaptive_transition_steps: int = 2
    chunked_prefill_adaptive_max_mixed_steps: int = 2
    chunked_prefill_slo_mixed: bool = False
    chunked_prefill_slo_target_gap_ns: int = 0
    chunked_prefill_slo_reserve_ns: int = 0
    chunked_prefill_slo_cost_intercept_ns: int = 0
    chunked_prefill_slo_cost_per_prefill_token_ns: int = 0
    chunked_prefill_slo_min_chunk_tokens: int = 16

    # KV cache 量化（C4 等）相关配置
    kv_quant_bits: int = 0                              # 0 / 4 / 8，KV cache 量化位宽，0 表示不量化
    kv_quant_group_size: int = 128                      # group-wise 量化的组大小，沿 head_dim 切
    kv_quant_symmetric: bool = True                     # True = 对称量化（仅 scale），False = 非对称（scale + zero）
    # KV offload MVP-0：仅支持 fp16/bf16 KV + full attention + eager decode。
    # seq.block_table 保持 logical block id；ModelRunner 在 prepare_* 中翻译到 GPU physical slot。
    kv_offload_mvp0: bool = False
    kv_offload_gpu_blocks: int = 0                      # >0 时限制 GPU staging slot 数；0 表示使用自动估算值
    kv_offload_logical_blocks: int = 0                  # >0 时覆盖 logical block 总数；0 表示使用自动估算值
    kv_offload_async_copy: bool = True                  # MVP-1：用独立 CUDA stream/event 做 H2D/D2H
    kv_offload_batch_copy: bool = True                  # MVP-1：连续 logical/slot span 合并成批量 copy
    kv_offload_writeback_on_evict: bool = False         # False 保持每次 forward 后写回；True 延迟到 eviction
    kv_offload_evict_policy: str = "lru_cost"           # lru / lru_cost
    kv_offload_blockwise_decode: bool = False           # 实验：decode attention 按 block window 流式扫描 KV
    kv_offload_blockwise_prefill: bool = False          # 实验：chunked prefill attention 按 block window 流式扫描 prefix KV
    kv_offload_blockwise_blocks: int = 1                # 每个 attention window 最多处理多少 logical KV blocks
    # Activation 量化（A8 等）相关配置（W4A8 用）
    act_quant_bits: int = 0                             # 0 / 8
    # A8 首尾层跳过：长文 W4A8+SQ 复读塌方根因——首尾若干层 outlier 最严重，
    # 让这些层保 fp 激活、中间层走 A8 能显著修复（详见 docs/qwen3-8b-fixes.md §21）
    act_quant_skip_first: int = 0                       # 前 N 层不做 A8 假量化
    act_quant_skip_last: int = 0                        # 后 N 层不做 A8 假量化
    act_quant_skip_layers: list[int] | None = None      # 显式指定不做 A8 的层（按 outlier 强度精准 skip）

    # SmoothQuant 相关配置（把激活离群值迁移到 weight 上的 per-input-channel scale）
    smoothquant_scale_path: str | None = None           # 校准产物路径；None 关闭
    smoothquant_alpha: float = 0.5                      # 仅 calibration 时用；loader 不需要

    # 在默认的构造函数之后自动启用，用于补充缺少的初始化逻辑
    def __post_init__(self):
        assert os.path.isdir(self.model)
        assert self.kvcache_block_size % 256 == 0
        assert 1 <= self.tensor_parallel_size <= 8
        if not isinstance(
            self.replay_aware_decode_metadata,
            bool,
        ):
            raise ValueError(
                "replay_aware_decode_metadata must be a bool"
            )
        if not isinstance(
            self.zero_temperature_greedy_fast_path,
            bool,
        ):
            raise ValueError(
                "zero_temperature_greedy_fast_path must be a bool"
            )
        if not isinstance(
            self.graph_resident_greedy_tail,
            bool,
        ):
            raise ValueError(
                "graph_resident_greedy_tail must be a bool"
            )
        if not isinstance(
            self.exact_greedy_decode_burst,
            bool,
        ):
            raise ValueError(
                "exact_greedy_decode_burst must be a bool"
            )
        if not isinstance(
            self.exact_greedy_decode_burst_continuation,
            bool,
        ):
            raise ValueError(
                "exact_greedy_decode_burst_continuation "
                "must be a bool"
            )
        if not isinstance(
            self.exact_greedy_decode_burst_split_phase,
            bool,
        ):
            raise ValueError(
                "exact_greedy_decode_burst_split_phase "
                "must be a bool"
            )
        if not isinstance(
            self.exact_greedy_decode_burst_ragged_coalescing,
            bool,
        ):
            raise ValueError(
                "exact_greedy_decode_burst_ragged_coalescing "
                "must be a bool"
            )
        if not isinstance(
            self.exact_greedy_decode_burst_lease_local_delta_journal,
            bool,
        ):
            raise ValueError(
                "exact_greedy_decode_burst_lease_local_delta_journal "
                "must be a bool"
            )
        if not isinstance(
            self.exact_greedy_decode_burst_generation_sealed_identity,
            bool,
        ):
            raise ValueError(
                "exact_greedy_decode_burst_generation_sealed_identity "
                "must be a bool"
            )
        if not isinstance(
            self.exact_greedy_decode_burst_medium_split_k,
            bool,
        ):
            raise ValueError(
                "exact_greedy_decode_burst_medium_split_k "
                "must be a bool"
            )
        if not isinstance(
            self.exact_greedy_decode_burst_elastic_k16,
            bool,
        ):
            raise ValueError(
                "exact_greedy_decode_burst_elastic_k16 "
                "must be a bool"
            )
        if not isinstance(
            self.exact_greedy_decode_burst_octet_folded_graph,
            bool,
        ):
            raise ValueError(
                "exact_greedy_decode_burst_octet_folded_graph "
                "must be a bool"
            )
        if (
            self.exact_greedy_decode_burst_medium_split_k
            and not self.exact_greedy_decode_burst
        ):
            raise ValueError(
                "medium split-k requires "
                "exact_greedy_decode_burst"
            )
        if (
            isinstance(self.exact_greedy_decode_burst_tokens, bool)
            or not isinstance(
                self.exact_greedy_decode_burst_tokens,
                int,
            )
            or not (
                2 <= self.exact_greedy_decode_burst_tokens <= 8
            )
        ):
            raise ValueError(
                "exact_greedy_decode_burst_tokens must be an "
                "integer in [2, 8]"
            )
        if self.exact_greedy_decode_burst_octet_folded_graph:
            if not self.exact_greedy_decode_burst:
                raise ValueError(
                    "octet-folded graph requires "
                    "exact_greedy_decode_burst"
                )
            if self.exact_greedy_decode_burst_tokens != 8:
                raise ValueError(
                    "octet-folded graph requires K8 base width"
                )
            if self.exact_greedy_decode_burst_ragged_coalescing:
                raise ValueError(
                    "octet-folded graph requires "
                    "ragged coalescing off"
                )
            if self.exact_greedy_decode_burst_split_phase:
                raise ValueError(
                    "octet-folded graph requires split phase off"
                )
        if self.exact_greedy_decode_burst_elastic_k16:
            if not self.exact_greedy_decode_burst:
                raise ValueError(
                    "elastic K16 requires "
                    "exact_greedy_decode_burst"
                )
            if self.exact_greedy_decode_burst_tokens != 8:
                raise ValueError(
                    "elastic K16 requires K8 base width"
                )
            if self.exact_greedy_decode_burst_ragged_coalescing:
                raise ValueError(
                    "elastic K16 requires ragged coalescing off"
                )
            if self.exact_greedy_decode_burst_split_phase:
                raise ValueError(
                    "elastic K16 requires split phase off"
                )
        if self.exact_greedy_decode_burst_ragged_coalescing:
            if not self.exact_greedy_decode_burst:
                raise ValueError(
                    "ragged coalescing requires "
                    "exact_greedy_decode_burst"
                )
            if not self.exact_greedy_decode_burst_split_phase:
                raise ValueError(
                    "ragged coalescing requires split phase"
                )
            if self.exact_greedy_decode_burst_tokens != 8:
                raise ValueError("ragged coalescing requires K8")
        if self.exact_greedy_decode_burst_generation_sealed_identity:
            if not self.exact_greedy_decode_burst:
                raise ValueError(
                    "generation-sealed identity requires "
                    "exact_greedy_decode_burst"
                )
            if self.exact_greedy_decode_burst_tokens != 8:
                raise ValueError(
                    "generation-sealed identity requires K8"
                )
            if not (
                self
                .exact_greedy_decode_burst_lease_local_delta_journal
            ):
                raise ValueError(
                    "generation-sealed identity requires "
                    "lease-local delta journal"
                )
        if not isinstance(self.prefill_cuda_graphs, bool):
            raise ValueError("prefill_cuda_graphs must be a bool")
        if not isinstance(self.phase_stitch_profile, bool):
            raise ValueError("phase_stitch_profile must be a bool")
        if not isinstance(
            self.phase_stitched_exact_graph_runtime,
            bool,
        ):
            raise ValueError(
                "phase_stitched_exact_graph_runtime must be a bool"
            )
        self.prefill_cuda_graph_token_allowlist = (
            _normalize_positive_int_tuple(
                self.prefill_cuda_graph_token_allowlist,
                name="prefill_cuda_graph_token_allowlist",
                allow_empty=False,
            )
        )
        if self.exact_greedy_decode_burst_lease_local_delta_journal:
            if not self.exact_greedy_decode_burst:
                raise ValueError(
                    "lease-local delta journal requires "
                    "exact_greedy_decode_burst"
                )
            if self.exact_greedy_decode_burst_tokens != 8:
                raise ValueError(
                    "lease-local delta journal requires K8"
                )
        if self.exact_greedy_decode_burst_split_phase:
            if not self.exact_greedy_decode_burst:
                raise ValueError(
                    "split phase requires exact_greedy_decode_burst"
                )
            if self.exact_greedy_decode_burst_tokens != 8:
                raise ValueError("split phase requires K8")
            if self.exact_greedy_decode_burst_continuation:
                raise ValueError(
                    "split phase cannot compose with continuation"
                )
        allowlist = self.multi_sequence_cuda_graph_batch_allowlist
        assert isinstance(allowlist, (tuple, list))
        assert allowlist
        assert all(
            isinstance(value, int)
            and not isinstance(value, bool)
            and value > 1
            for value in allowlist
        )
        self.multi_sequence_cuda_graph_batch_allowlist = tuple(
            sorted(set(allowlist))
        )
        for value in (
            self.multi_sequence_cuda_graph_min_observations,
            self.multi_sequence_cuda_graph_max_entries,
            self.multi_sequence_cuda_graph_max_static_bytes,
            self.multi_sequence_cuda_graph_max_reserved_bytes,
            self.multi_sequence_cuda_graph_max_total_capture_ns,
            self.multi_sequence_cuda_graph_max_single_capture_ns,
        ):
            assert isinstance(value, int) and not isinstance(value, bool)
            assert value > 0
        if not isinstance(self.spec_verify_cuda_graphs, bool):
            raise ValueError(
                "spec_verify_cuda_graphs must be a bool"
            )
        self.spec_verify_cuda_graph_batch_allowlist = (
            _normalize_positive_int_tuple(
                self.spec_verify_cuda_graph_batch_allowlist,
                name=(
                    "spec_verify_cuda_graph_batch_allowlist"
                ),
                allow_empty=False,
            )
        )
        self.spec_verify_cuda_graph_query_len_allowlist = (
            _normalize_positive_int_tuple(
                self.spec_verify_cuda_graph_query_len_allowlist,
                name=(
                    "spec_verify_cuda_graph_query_len_allowlist"
                ),
                allow_empty=True,
            )
        )
        for name in (
            "spec_verify_cuda_graph_min_observations",
            "spec_verify_cuda_graph_max_entries",
            "spec_verify_cuda_graph_max_static_bytes",
            "spec_verify_cuda_graph_max_reserved_bytes",
            "spec_verify_cuda_graph_max_total_capture_ns",
            "spec_verify_cuda_graph_max_single_capture_ns",
        ):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value <= 0
            ):
                raise ValueError(
                    f"{name} must be a positive integer"
                )
        if not isinstance(self.qwen35_mtp_cuda_graphs, bool):
            raise ValueError(
                "qwen35_mtp_cuda_graphs must be a bool"
            )
        if not isinstance(self.qwen35_mtp_enabled, bool):
            raise ValueError("qwen35_mtp_enabled must be a bool")
        for name in (
            "proposal_kv_offload_enabled",
            "proposal_kv_async_copy",
            "proposal_kv_batch_copy",
        ):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be a bool")
        for name in (
            "proposal_kv_logical_entry_capacity",
            "proposal_kv_gpu_slot_capacity",
            "proposal_kv_cpu_backing_capacity",
        ):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 0
            ):
                raise ValueError(
                    f"{name} must be a nonnegative integer"
                )
        if self.proposal_kv_offload_enabled:
            if not self.qwen35_mtp_enabled:
                raise ValueError(
                    "proposal KV offload requires Qwen3.5 MTP"
                )
            if (
                self.proposal_kv_logical_entry_capacity
                != self.proposal_kv_cpu_backing_capacity
                or self.proposal_kv_logical_entry_capacity
                <= self.proposal_kv_gpu_slot_capacity
                or self.proposal_kv_gpu_slot_capacity <= 0
            ):
                raise ValueError(
                    "proposal KV offload requires "
                    "logical == cpu > gpu > 0"
                )
            if self.qwen35_mtp_cuda_graphs:
                raise ValueError(
                    "proposal KV offload is incompatible with "
                    "Qwen3.5 MTP CUDA graphs"
                )
        if (
            isinstance(self.qwen35_mtp_max_proposal_tokens, bool)
            or not isinstance(
                self.qwen35_mtp_max_proposal_tokens,
                int,
            )
            or self.qwen35_mtp_max_proposal_tokens <= 0
        ):
            raise ValueError(
                "qwen35_mtp_max_proposal_tokens must be a "
                "positive integer"
            )
        if not isinstance(
            self.autoregressive_draft_enabled,
            bool,
        ):
            raise ValueError(
                "autoregressive_draft_enabled must be a bool"
            )
        if (
            self.autoregressive_draft_model is not None
            and not isinstance(
                self.autoregressive_draft_model,
                str,
            )
        ):
            raise ValueError(
                "autoregressive_draft_model must be a string or None"
            )
        if (
            self.autoregressive_draft_enabled
            and (
                not isinstance(
                    self.autoregressive_draft_model,
                    str,
                )
                or not self.autoregressive_draft_model.strip()
            )
        ):
            raise ValueError(
                "autoregressive_draft_model is required when enabled"
            )
        if (
            self.autoregressive_draft_enabled
            and self.tensor_parallel_size not in (1, 4)
        ):
            raise ValueError(
                "autoregressive draft requires "
                "tensor_parallel_size 1 or 4"
            )
        if self.autoregressive_draft_backend != "qwen3":
            raise ValueError(
                "autoregressive_draft_backend must equal qwen3"
            )
        if (
            isinstance(
                self.autoregressive_draft_max_proposal_tokens,
                bool,
            )
            or not isinstance(
                self.autoregressive_draft_max_proposal_tokens,
                int,
            )
            or not (
                1
                <= self.autoregressive_draft_max_proposal_tokens
                <= 4
            )
        ):
            raise ValueError(
                "autoregressive_draft_max_proposal_tokens "
                "must be in 1..4"
            )
        if (
            isinstance(
                self.autoregressive_draft_gpu_slot_capacity,
                bool,
            )
            or not isinstance(
                self.autoregressive_draft_gpu_slot_capacity,
                int,
            )
            or self.autoregressive_draft_gpu_slot_capacity < 0
            or (
                self.autoregressive_draft_enabled
                and self.autoregressive_draft_gpu_slot_capacity <= 0
            )
        ):
            raise ValueError(
                "autoregressive_draft_gpu_slot_capacity must be "
                "positive when enabled and nonnegative otherwise"
            )
        if not isinstance(
            self.autoregressive_draft_proposal_kv_offload_enabled,
            bool,
        ):
            raise ValueError(
                "autoregressive_draft_proposal_kv_offload_enabled "
                "must be a bool"
            )
        for name in (
            "autoregressive_draft_logical_entry_capacity",
            "autoregressive_draft_cpu_backing_capacity",
        ):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 0
            ):
                raise ValueError(
                    f"{name} must be a nonnegative integer"
                )
        if self.autoregressive_draft_proposal_kv_offload_enabled:
            if not self.autoregressive_draft_enabled:
                raise ValueError(
                    "autoregressive draft proposal KV offload "
                    "requires autoregressive draft"
                )
            if (
                self.autoregressive_draft_logical_entry_capacity
                != self.autoregressive_draft_cpu_backing_capacity
            ):
                raise ValueError(
                    "autoregressive draft proposal KV offload "
                    "requires logical == cpu"
                )
            if (
                self.autoregressive_draft_logical_entry_capacity
                <= self.autoregressive_draft_gpu_slot_capacity
                or self.autoregressive_draft_gpu_slot_capacity <= 0
            ):
                raise ValueError(
                    "autoregressive draft proposal KV offload "
                    "requires logical == cpu > gpu > 0"
                )
        if not isinstance(
            self.autoregressive_draft_cuda_graphs,
            bool,
        ):
            raise ValueError(
                "autoregressive_draft_cuda_graphs must be a bool"
            )
        if not isinstance(
            self.autoregressive_draft_command_timeline,
            bool,
        ):
            raise ValueError(
                "autoregressive_draft_command_timeline must be a bool"
            )
        if (
            isinstance(
                self.autoregressive_draft_command_timeline_max_rows,
                bool,
            )
            or not isinstance(
                self.autoregressive_draft_command_timeline_max_rows,
                int,
            )
            or self.autoregressive_draft_command_timeline_max_rows
            <= 0
        ):
            raise ValueError(
                "command timeline max rows must be a positive integer"
            )
        self.autoregressive_draft_cuda_graph_q_allowlist = (
            _normalize_positive_int_tuple(
                self.autoregressive_draft_cuda_graph_q_allowlist,
                name=(
                    "autoregressive_draft_cuda_graph_q_allowlist"
                ),
                allow_empty=False,
            )
        )
        if (
            self.autoregressive_draft_cuda_graph_q_allowlist[0]
            < 2
        ):
            raise ValueError(
                "autoregressive_draft_cuda_graph_q_allowlist "
                "must contain only values at least two"
            )
        self.autoregressive_draft_cuda_graph_batch_allowlist = (
            _normalize_positive_int_tuple(
                self.autoregressive_draft_cuda_graph_batch_allowlist,
                name=(
                    "autoregressive_draft_cuda_graph_batch_allowlist"
                ),
                allow_empty=False,
            )
        )
        for name in (
            "autoregressive_draft_cuda_graph_min_observations",
            "autoregressive_draft_cuda_graph_max_entries",
            "autoregressive_draft_cuda_graph_max_static_bytes",
            "autoregressive_draft_cuda_graph_max_reserved_bytes",
            "autoregressive_draft_cuda_graph_max_total_capture_ns",
            "autoregressive_draft_cuda_graph_max_single_capture_ns",
        ):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value <= 0
            ):
                raise ValueError(
                    f"{name} must be a positive integer"
                )
        if self.autoregressive_draft_cuda_graphs:
            if not self.autoregressive_draft_enabled:
                raise ValueError(
                    "autoregressive draft CUDA graph mode requires "
                    "autoregressive draft"
                )
            if self.tensor_parallel_size != 4:
                raise ValueError(
                    "autoregressive draft CUDA graph mode requires "
                    "tensor_parallel_size 4"
                )
            if (
                self.autoregressive_draft_proposal_kv_offload_enabled
            ):
                raise ValueError(
                    "autoregressive draft CUDA graphs are "
                    "incompatible with proposal KV offload"
                )
            if (
                self.autoregressive_draft_cuda_graph_q_allowlist
                != (4,)
            ):
                raise ValueError(
                    "autoregressive draft CUDA graphs support "
                    "Q4 only"
                )
            if (
                self.autoregressive_draft_cuda_graph_batch_allowlist
                != (4,)
            ):
                raise ValueError(
                    "autoregressive draft CUDA graphs support "
                    "batch size 4 only"
                )
            if (
                self.autoregressive_draft_max_proposal_tokens
                != 4
            ):
                raise ValueError(
                    "autoregressive draft CUDA graphs require "
                    "max proposal tokens 4"
                )
        self.qwen35_mtp_cuda_graph_q_allowlist = (
            _normalize_positive_int_tuple(
                self.qwen35_mtp_cuda_graph_q_allowlist,
                name="qwen35_mtp_cuda_graph_q_allowlist",
                allow_empty=False,
            )
        )
        if self.qwen35_mtp_cuda_graph_q_allowlist[0] < 2:
            raise ValueError(
                "qwen35_mtp_cuda_graph_q_allowlist must contain "
                "only values at least two"
            )
        self.qwen35_mtp_cuda_graph_batch_allowlist = (
            _normalize_positive_int_tuple(
                self.qwen35_mtp_cuda_graph_batch_allowlist,
                name=(
                    "qwen35_mtp_cuda_graph_batch_allowlist"
                ),
                allow_empty=False,
            )
        )
        for name in (
            "qwen35_mtp_cuda_graph_min_observations",
            "qwen35_mtp_cuda_graph_max_entries",
            "qwen35_mtp_cuda_graph_max_static_bytes",
            "qwen35_mtp_cuda_graph_max_reserved_bytes",
            "qwen35_mtp_cuda_graph_max_total_capture_ns",
            "qwen35_mtp_cuda_graph_max_single_capture_ns",
        ):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value <= 0
            ):
                raise ValueError(
                    f"{name} must be a positive integer"
                )
        assert self.quantization in (None, "int8", "int8_bnb", "int4", "int2")
        if self.qwen35_hybrid_prefix_representation not in (
            "exact_restore",
            "recurrent_int8_per_row",
        ):
            raise ValueError(
                "unsupported Qwen3.5 hybrid prefix representation: "
                f"{self.qwen35_hybrid_prefix_representation}"
            )
        assert self.kv_quant_bits in (0, 4, 8), "kv_quant_bits 仅支持 0/4/8"
        assert self.kv_offload_gpu_blocks >= 0
        assert self.kv_offload_logical_blocks >= 0
        assert self.kv_offload_evict_policy in ("lru", "lru_cost")
        assert self.kv_offload_blockwise_blocks > 0
        assert self.kv_cartridge_blocks >= 0
        assert self.kv_cartridge_min_seq_len >= 0
        assert self.kv_cartridge_mode == "uniform", "KV-Cartridge v0 仅支持 uniform 模式"
        assert not (self.kv_cartridge_blocks > 0 and self.quest_top_k_blocks > 0), \
            "KV-Cartridge v0 和 Quest 都是 decode 稀疏策略，请分开评测"
        assert self.am_compact_blocks >= 0
        assert self.am_compact_min_seq_len >= 0
        assert self.am_compact_selector in ("highest", "omp"), \
            "am_compact_selector 仅支持 highest / omp"
        assert self.am_compact_score_method in ("rms", "mean", "max")
        assert self.am_compact_beta_bound > 0.0
        assert self.am_compact_ridge_lambda >= 0.0
        assert self.am_omp_candidate_pool_size >= 0
        assert self.am_compact_cache_refresh_interval >= 0
        assert self.am_prefill_cache_ref_query_stride > 0
        assert self.am_compact_num_clusters > 0
        assert self.am_compact_route_top_k > 0
        assert self.am_compact_num_key_spans > 0
        assert self.am_compact_decode_refit_mode in ("full", "direct", "beta", "anchor")
        assert self.am_compact_decode_refit_interval > 0
        assert self.am_compact_skip_first_layers >= 0
        assert self.am_compact_skip_last_layers >= 0
        assert self.am_compact_layer_stride > 0
        if self.am_compact_enable_layers is not None:
            self.am_compact_enable_layers = sorted(set(int(x) for x in self.am_compact_enable_layers))
            assert all(x >= 0 for x in self.am_compact_enable_layers)
        assert not (self.am_compact_blocks > 0 and self.quest_top_k_blocks > 0), \
            "Attention Matching compact decode 和 Quest 请分开评测"
        assert not (self.am_compact_blocks > 0 and self.kv_cartridge_blocks > 0), \
            "Attention Matching compact decode 和 KV-Cartridge uniform 请分开评测"
        assert not (self.am_compact_blocks > 0 and self.kv_quant_bits == 4), \
            "Attention Matching compact decode v0 仅支持 fp16 KV / KV8，暂不支持 KV4"
        if self.kv_offload_mvp0:
            assert self.kv_quant_bits == 0, "KV offload MVP-0 仅支持 fp16/bf16 KV，暂不支持 KV4/KV8"
            assert self.quest_top_k_blocks <= 0, "KV offload MVP-0 先只支持 full attention，请关闭 Quest"
            assert self.kv_cartridge_blocks == 0, "KV offload MVP-0 先只支持 full attention，请关闭 KV-Cartridge"
            assert self.am_compact_blocks == 0, "KV offload MVP-0 先只支持 full attention，请关闭 AM compact"
        assert not self.kv_offload_blockwise_decode or self.kv_offload_mvp0, \
            "kv_offload_blockwise_decode 依赖 kv_offload_mvp0"
        assert not self.kv_offload_blockwise_prefill or self.kv_offload_mvp0, \
            "kv_offload_blockwise_prefill 依赖 kv_offload_mvp0"
        assert not self.kv_offload_blockwise_prefill or self.max_num_prefill_tokens_per_step > 0, \
            "kv_offload_blockwise_prefill 需要开启 chunked prefill: max_num_prefill_tokens_per_step > 0"
        assert not (self.kv_offload_mvp0 and self.chunked_prefill_mixed_batch), \
            "KV offload MVP-0 暂不支持 mixed prefill+decode batch，请关闭 chunked_prefill_mixed_batch"
        assert self.act_quant_bits in (0, 8), "act_quant_bits 仅支持 0/8"
        assert self.act_quant_skip_first >= 0 and self.act_quant_skip_last >= 0
        assert self.max_num_prefill_tokens_per_step >= 0
        assert self.chunked_prefill_max_consecutive_chunks >= 0
        assert self.chunked_prefill_mixed_min_prompt_tokens >= 0
        assert self.chunked_prefill_adaptive_enter_waiting > 0
        assert self.chunked_prefill_adaptive_exit_waiting >= 0
        assert self.chunked_prefill_adaptive_transition_steps > 0
        assert self.chunked_prefill_adaptive_max_mixed_steps > 0
        assert self.chunked_prefill_adaptive_exit_waiting < self.chunked_prefill_adaptive_enter_waiting
        assert not (self.chunked_prefill_adaptive_mixed and self.chunked_prefill_mixed_batch), \
            "adaptive mixed 和 always-on mixed 必须分开评测"
        assert not (self.chunked_prefill_adaptive_mixed and self.kv_offload_mvp0), \
            "KV offload MVP-0 暂不支持 adaptive mixed prefill+decode"
        assert not self.chunked_prefill_adaptive_mixed or self.max_num_prefill_tokens_per_step > 0, \
            "adaptive mixed 需要开启 chunked prefill"
        int64_max = (1 << 63) - 1
        for value in (
            self.chunked_prefill_slo_target_gap_ns,
            self.chunked_prefill_slo_reserve_ns,
            self.chunked_prefill_slo_cost_intercept_ns,
            self.chunked_prefill_slo_cost_per_prefill_token_ns,
        ):
            assert isinstance(value, int) and not isinstance(value, bool)
            assert 0 <= value <= int64_max
        assert (
            isinstance(self.chunked_prefill_slo_min_chunk_tokens, int)
            and not isinstance(self.chunked_prefill_slo_min_chunk_tokens, bool)
            and self.chunked_prefill_slo_min_chunk_tokens > 0
        )
        if self.chunked_prefill_slo_mixed:
            assert self.chunked_prefill_slo_target_gap_ns > 0
            assert self.chunked_prefill_slo_reserve_ns > 0
            assert (
                self.chunked_prefill_slo_reserve_ns
                < self.chunked_prefill_slo_target_gap_ns
            )
            assert self.chunked_prefill_slo_cost_intercept_ns > 0
            assert self.chunked_prefill_slo_cost_per_prefill_token_ns > 0
            assert self.max_num_prefill_tokens_per_step > 0
            assert (
                self.chunked_prefill_slo_min_chunk_tokens
                <= self.max_num_prefill_tokens_per_step
            )
            assert (
                self.max_num_prefill_tokens_per_step
                % self.chunked_prefill_slo_min_chunk_tokens
                == 0
            )
            assert not self.chunked_prefill_mixed_batch
            assert not self.chunked_prefill_adaptive_mixed
            assert not self.kv_offload_mvp0
            assert (
                self.chunked_prefill_slo_cost_per_prefill_token_ns
                <= (
                    int64_max - self.chunked_prefill_slo_cost_intercept_ns
                ) // self.max_num_prefill_tokens_per_step
            )
        assert 0.0 <= self.smoothquant_alpha <= 1.0
        if self.smoothquant_scale_path is not None:
            assert os.path.isfile(self.smoothquant_scale_path), \
                f"smoothquant_scale_path 不存在: {self.smoothquant_scale_path}"
        if self.kv_quant_bits == 4:
            # group_size 必须能整除 head_dim，且对 4-bit pack 友好（即 group_size 为偶数）
            assert self.kv_quant_group_size % 2 == 0
            assert self.kv_quant_symmetric, "非对称 KV 量化分支尚未实现，仅支持对称"
        if self.kv_quant_bits == 8:
            assert self.kv_quant_symmetric, "C8 仅实现对称量化"
        self.hf_config = AutoConfig.from_pretrained(self.model)
        if self.am_compact_enable_layers is not None:
            assert all(x < self.hf_config.num_hidden_layers for x in self.am_compact_enable_layers), \
                "am_compact_enable_layers 中存在超过模型层数的 layer index"
            assert len(self.am_compact_enable_layers) > 0 or self.am_compact_blocks == 0, \
                "am_compact_enable_layers 为空会导致没有任何层启用 AM"
        if self.am_compact_blocks > 0 and self.am_compact_enable_layers is None:
            assert self.am_compact_skip_first_layers + self.am_compact_skip_last_layers < self.hf_config.num_hidden_layers, \
                "am_compact_skip_first_layers + am_compact_skip_last_layers 覆盖了全部层"
        max_position_embeddings = getattr(
            self.hf_config,
            "max_position_embeddings",
            None,
        )
        if max_position_embeddings is not None:
            self.max_model_len = min(
                self.max_model_len,
                max_position_embeddings,
            )
        assert self.max_num_batched_tokens >= self.max_model_len
