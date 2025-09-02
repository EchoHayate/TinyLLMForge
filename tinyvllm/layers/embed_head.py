import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist                # 用于张量并行

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
    
    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):      #按 GPU 编号拆分完整权重，只加载当前 GPU 负责的分块 是词表并行的关键步骤
        param_data = param.data
        shard_size = param_data.size(0)     #也就是num_embeddings_per_partition
        start_idx = self.tp_rank * shard_size
        loaded_weight = loaded_weight.narrow(0, start_idx, shard_size)
        assert param_data.size() == loaded_weight.size()
        param_data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor):
        if self.tp_size > 1:
            mask = (x >= self.vocab_start_idx) & (x < self.vocab_end_idx)
            x = mask * (x - self.vocab_start_idx)
        y = F.embedding(x, self.weight)
        if self.tp_size > 1:
            y = mask.unsqueeze(1) * y                 # 这里只是计算到了局部值
            dist.all_reduce(y)                        # 因此需要规约广播，使得每张卡上的计算结果一致
        return y
    

# 在模型中，输入的embedding权重和输出头的矩阵权重共享，embedding是查表，输出则是矩阵乘
class ParallelLMHead(VocabParallelEmbedding):
    
    def __init__(self, num_embedding: int,
                 embedding_dim: int, 
                 bias: bool = False,):
        super().__init__(num_embedding, embedding_dim)
        if bias:
            self.bias = nn.Parameter(torch.empty(self.num_embeddings_per_partition))
            self.bias.weight_loader = self.weight_loader
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor):
        context = get_context()
        if context.is_prefill:
            last_indices = context.cu_seqlens_q[1:] - 1
            x = x[last_indices].contiguous()
        logits = F.linear(x, self.weight, self.bias)
        if self.tp_size > 1:
            # 这里的 [[logits]], 就表示在第0维进行堆叠
            all_logits = [torch.empty_like(logits) for _ in range(self.tp_size)] if self.tp_rank == 0 else None
            dist.gather(logits, all_logits, 0)
            # logits最终是二维的，因此需要使用 cat 进行降维
            logits = torch.cat(all_logits, 0) if self.tp.rank == 0 else None
        return logits
    