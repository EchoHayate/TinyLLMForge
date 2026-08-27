import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist                # 用于张量并行
from tinyvllm.engine.decode_internal_profiler import (
    profile_collective,
    profile_operation,
)

from tinyvllm.utils.context import get_context

# 输入embedding层
class VocabParallelEmbedding(nn.Module):        
    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__()
        self.tp_rank = dist.get_rank()              # 张量并行id, 分块的张量序号
        self.tp_size = dist.get_world_size()        # 张量并行大小
        assert num_embeddings % self.tp_size == 0 
        self.num_embeddings = num_embeddings
        self.num_embeddings_per_partition = self.num_embeddings // self.tp_size
        self.vocab_start_idx = self.num_embeddings_per_partition * self.tp_rank
        self.vocab_end_idx = self.vocab_start_idx + self.num_embeddings_per_partition
        self.weight = nn.Parameter(torch.empty(self.num_embeddings_per_partition, embedding_dim))  #parameter表示模型的 可学习参数   能够自动注册 计算梯度
        self.weight.weight_loader = self.weight_loader
        # num_embeddings 是初始化时传入的参数，明确代表整个模型的词表总大小（比如 30 万、50 万等）

    # param需要符合上文定义的nn.Parameter大小 否则会报错
    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):      #按 GPU 编号拆分完整权重，只加载当前 GPU 负责的分块 是词表并行的关键步骤
        param_data = param.data
        shard_size = param_data.size(0)     #也就是num_embeddings_per_partition
        start_idx = self.tp_rank * shard_size
        loaded_weight = loaded_weight.narrow(0, start_idx, shard_size)      #narrow表示只能连续切片 对大张量可能有潜在的性能优化 
        assert param_data.size() == loaded_weight.size()
        param_data.copy_(loaded_weight)

        # x:(batch_size, seq_len)  mask:(batch_size, seq_len)   y:(batch_size, seq_len, embedding_dim)
        # batch_size：每次输入的样本数量
    def forward(self, x: torch.Tensor):
        if self.tp_size > 1:
            mask = (x >= self.vocab_start_idx) & (x < self.vocab_end_idx)        #e.g.((1,1,0,0) (1,0,0,0) (0,0,0,0))   
            x = mask * (x - self.vocab_start_idx)           #重置当前GPU分块的token索引 让其落在当前GPU分块内 e.g 75000→0，75001→1，
        with profile_operation(
            "memory",
            "vocab_parallel_embedding_lookup",
            tensor=x,
        ):
            y = F.embedding(x, self.weight)         #根据index和权重去做lookup 计算输出形状
        if self.tp_size > 1:
            y = mask.unsqueeze(1) * y                 # 这里只是计算到了局部值
            profile_collective(
                "vocab_parallel_embedding_all_reduce",
                y,
                dist.all_reduce,
                site_role="vocab_parallel_embedding",
                collective_kind="all_reduce",
                process_group="tensor_parallel",
                execution_phase="decode_or_prefill",
                async_mode=False,
            )
        return y
    

# 在模型中，输入的embedding权重和输出头的矩阵权重共享，embedding是查表，输出则是矩阵乘
class ParallelLMHead(VocabParallelEmbedding):
    
    def __init__(self, num_embedding: int,
                 embedding_dim: int, 
                 bias: bool = False,
                 exact_full_vocab: bool = False,):
        super().__init__(num_embedding, embedding_dim)
        self.exact_full_vocab = exact_full_vocab
        if bias:
            self.bias = nn.Parameter(torch.empty(self.num_embeddings_per_partition))
            self.bias.weight_loader = self.weight_loader
        else:
            self.register_parameter("bias", None)
        # all_gather buffer 缓存（仅 rank0 用）：复用一组 [tp_size] 个 [max_n, vocab_per_part] 的 buffer
        self._gather_bufs: list[torch.Tensor] | None = None
        self._gather_buf_shape: tuple[int, ...] | None = None
        self._exact_full_weight: torch.Tensor | None = None
        self._exact_full_bias: torch.Tensor | None = None
        self._exact_weight_ready = False

    def _prepare_exact_full_parameters(self) -> None:
        if self._exact_weight_ready:
            return
        if self.tp_rank == 0:
            weight_shards = [
                torch.empty_like(self.weight)
                for _ in range(self.tp_size)
            ]
        else:
            weight_shards = None
        profile_collective(
            "lm_head_weight_gather",
            self.weight,
            lambda tensor: dist.gather(
                tensor,
                weight_shards,
                0,
            ),
            site_role="lm_head_parameter_materialization",
            collective_kind="gather",
            process_group="tensor_parallel",
            execution_phase="startup",
            async_mode=False,
            destination_rank=0,
        )
        if self.tp_rank == 0:
            self._exact_full_weight = torch.cat(weight_shards, dim=0)
        if self.bias is not None:
            if self.tp_rank == 0:
                bias_shards = [
                    torch.empty_like(self.bias)
                    for _ in range(self.tp_size)
                ]
            else:
                bias_shards = None
            profile_collective(
                "lm_head_bias_gather",
                self.bias,
                lambda tensor: dist.gather(
                    tensor,
                    bias_shards,
                    0,
                ),
                site_role="lm_head_parameter_materialization",
                collective_kind="gather",
                process_group="tensor_parallel",
                execution_phase="startup",
                async_mode=False,
                destination_rank=0,
            )
            if self.tp_rank == 0:
                self._exact_full_bias = torch.cat(bias_shards, dim=0)
        self._exact_weight_ready = True

    def forward(self, x: torch.Tensor):
        context = get_context()
        last_indices = None
        if context.is_prefill:
            last_indices = context.logits_indices
            if last_indices is None:
                last_indices = context.cu_seqlens_q[1:] - 1         #获取每个序列最后一个token的索引
        if self.exact_full_vocab and self.tp_size > 1:
            self._prepare_exact_full_parameters()
            if self.tp_rank != 0:
                return None
            logits = F.linear(
                x,
                self._exact_full_weight,
                self._exact_full_bias,
            )
            if last_indices is not None:
                logits = logits[last_indices].contiguous()
            return logits
        logits = F.linear(x, self.weight, self.bias)        # (10,1024) * (50000,1024)^T + (10,50000) = (10,50000)
        if last_indices is not None:
            logits = logits[last_indices].contiguous()
        if self.tp_size > 1:                                #将各个GPU上的logits结果进行拼接 并返回到 0号GPU
            if self.tp_rank == 0:
                if self._gather_bufs is None or self._gather_buf_shape != tuple(logits.shape) or self._gather_bufs[0].dtype != logits.dtype:
                    self._gather_bufs = [torch.empty_like(logits) for _ in range(self.tp_size)]
                    self._gather_buf_shape = tuple(logits.shape)
                all_logits = self._gather_bufs
            else:
                all_logits = None
            profile_collective(
                "vocab_parallel_lm_head_gather",
                logits,
                lambda tensor: dist.gather(
                    tensor,
                    all_logits,
                    0,
                ),
                site_role="vocab_parallel_logits_materialization",
                collective_kind="gather",
                process_group="tensor_parallel",
                execution_phase="decode_or_prefill",
                async_mode=False,
                destination_rank=0,
            )
            # logits 形状是 [N, vocab/tp]，要沿 vocab 维（dim=1）拼回 [N, vocab]，
            # 而不是沿 batch 维（dim=0）—— 否则下游 sampler 的 temperatures[N] 维度对不上
            logits = torch.cat(all_logits, 1) if self.tp_rank == 0 else None
        return logits
    
