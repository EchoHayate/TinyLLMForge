import torch
from torch import nn
# 根据输入的预测分数（logits）和温度参数（temperatures），动态选择采样策略生成输出 token
class Sampler(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self, 
        logits: torch.Tensor,           # [batch_size, num_token_ids]
        temperatures: torch.Tensor,     # [batch_size]
    ) -> torch.Tensor:                  # [batch_size]
        logits = logits.to(torch.float32)
        greedy_tokens = logits.argmax(dim = -1)     #greedy_tokens = [batch_size] 词汇表维度取最大值的索引，即选择预测分数最高的 token
        logits.div_(temperatures.unsqueeze(dim = 1))
        probs = torch.softmax(logits, dim = -1, dtype = torch.float32)
        epsilon = 1e-10 
        # 温度 < 1：放大 logits 之间的差异（让高概率更高，低概率更低，接近贪心采样）。
        # 温度 > 1：缩小 logits 之间的差异（让概率分布更平缓，增加随机性）。
        sample_tokens = probs.div_(torch.empty_like(probs).exponential_(1) + epsilon).argmax(dim = -1)      #Gumbel-Max 采样
        return torch.where(temperatures == 0, greedy_tokens, sample_tokens)