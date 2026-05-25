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
    quantization: str | None = None                     # None | "int8" | "int8_bnb" | "int2"  线性层权重量化方式
    quant_group_size: int = 128                         # 分组量化的组大小（更小的组 -> 更高精度但额外开销大）
    cpu_offload: bool = False                           # 是否启用 cpu-offload (decoder layer 粒度)
    cpu_offload_num_layers: int = -1                    # 卸载到 cpu 的 decoder 层数，-1 表示除最后 2 层外全部卸载

    # 在默认的构造函数之后自动启用，用于补充缺少的初始化逻辑
    def __post_init__(self):
        assert os.path.isdir(self.model)
        assert self.kvcache_block_size % 256 == 0
        assert 1 <= self.tensor_parallel_size <= 8
        assert self.quantization in (None, "int8", "int8_bnb", "int2")
        self.hf_config = AutoConfig.from_pretrained(self.model)
        self.max_model_len = min(self.max_model_len, self.hf_config.max_position_embeddings)
        assert self.max_num_batched_tokens >= self.max_model_len