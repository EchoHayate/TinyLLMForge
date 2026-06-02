import os
from dataclasses import dataclass
from transformers import AutoConfig

@dataclass   #自动配置__init__方法
class Config:
    model: str                                          # sequence includes multi-tokens   batch includes multi-sequences
    max_num_batched_tokens: int = 16384                 # 2^14 = [batch_size * seq_len]  单个批次（batch）中，所有 sequence 包含的 token 总数的最大限制 
    max_num_seqs: int = 512                             # 可以同时并行处理的最大sequence 数量                  
    max_model_len: int = 4096                           #模型可以处理的最大sequence长度
    gpu_memory_utilization: float = 0.9                 #gpu利用率 可以用来确定实际 kv cache大小
    tensor_parallel_size: int = 1                       #并行计算gpu的个数
    enforce_eager: bool = False                         # True表示以即时执行模式推理，用于debug   false表示启用cuda graph  cuda graph开启后减少kernal launch时间 可用于吞吐量测试
    hf_config: AutoConfig | None = None                 # hugging face config, 加载模型的层数，隐藏层数，注意力头数
    eos: int  = -1                                      # end of sentence, 使用模型默认的句子结束符
    kvcache_block_size: int = 256                       
    num_kvcache_blocks: int = -1                        #-1代表自动计算  num_kvcache_blocks=kv cache/kvcache_block_size 

    # 量化 / cpu-offload 相关配置
    quantization: str | None = None                     # None | "int8" | "int8_bnb" | "int4" | "int2"  线性层权重量化方式
    quant_group_size: int = 128                         # 分组量化的组大小（更小的组 -> 更高精度但额外开销大）
    cpu_offload: bool = False                           # 是否启用 cpu-offload (decoder layer 粒度)
    cpu_offload_num_layers: int = -1                    # 卸载到 cpu 的 decoder 层数，-1 表示除最后 2 层外全部卸载

    # 动态稀疏 attention（Quest，page-level top-k）相关配置
    quest_top_k_blocks: int = -1                        # decode 时每个 query 选择的 block 数，-1 表示关闭 Quest
    quest_min_seq_len: int = 1024                       # 序列长度小于此值时退化为 full attention

    # KV cache 量化（C4 等）相关配置
    kv_quant_bits: int = 0                              # 0 / 4 / 8，KV cache 量化位宽，0 表示不量化
    kv_quant_group_size: int = 128                      # group-wise 量化的组大小，沿 head_dim 切
    kv_quant_symmetric: bool = True                     # True = 对称量化（仅 scale），False = 非对称（scale + zero）
    # Activation 量化（A8 等）相关配置（W4A8 用）
    act_quant_bits: int = 0                             # 0 / 8
    # A8 首尾层跳过：长文 W4A8+SQ 复读塌方根因——首尾若干层 outlier 最严重，
    # 让这些层保 fp 激活、中间层走 A8 能显著修复（详见 docs/qwen3-8b-fixes.md §21）
    act_quant_skip_first: int = 0                       # 前 N 层不做 A8 假量化
    act_quant_skip_last: int = 0                        # 后 N 层不做 A8 假量化

    # SmoothQuant 相关配置（把激活离群值迁移到 weight 上的 per-input-channel scale）
    smoothquant_scale_path: str | None = None           # 校准产物路径；None 关闭
    smoothquant_alpha: float = 0.5                      # 仅 calibration 时用；loader 不需要

    # 在默认的构造函数之后自动启用，用于补充缺少的初始化逻辑
    def __post_init__(self):
        assert os.path.isdir(self.model)
        assert self.kvcache_block_size % 256 == 0
        assert 1 <= self.tensor_parallel_size <= 8
        assert self.quantization in (None, "int8", "int8_bnb", "int4", "int2")
        assert self.kv_quant_bits in (0, 4, 8), "kv_quant_bits 仅支持 0/4/8"
        assert self.act_quant_bits in (0, 8), "act_quant_bits 仅支持 0/8"
        assert self.act_quant_skip_first >= 0 and self.act_quant_skip_last >= 0
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
        self.max_model_len = min(self.max_model_len, self.hf_config.max_position_embeddings)
        assert self.max_num_batched_tokens >= self.max_model_len