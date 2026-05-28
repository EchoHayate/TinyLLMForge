import torch
from torch import nn
# 根据输入的预测分数（logits）和温度参数（temperatures），动态选择采样策略生成输出 token
class Sampler(nn.Module):
    def __init__(self):
        super().__init__()
        # Gumbel-Max 用的指数随机数 buffer：按需扩容并复用，避免每步 alloc 一个 [bs, vocab] 的临时张量
        self._gumbel_buf: torch.Tensor | None = None

    def _get_gumbel_buf(self, ref: torch.Tensor) -> torch.Tensor:
        buf = self._gumbel_buf
        if (buf is None or buf.shape != ref.shape
                or buf.dtype != ref.dtype or buf.device != ref.device):
            buf = torch.empty_like(ref)
            self._gumbel_buf = buf
        return buf

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
        gumbel = self._get_gumbel_buf(probs).exponential_(1)
        sample_tokens = probs.div_(gumbel + epsilon).argmax(dim = -1)      #Gumbel-Max 采样
        return torch.where(temperatures == 0, greedy_tokens, sample_tokens)