import torch
from torch import nn
import torch.nn.functional as F

class SiluAndMul(nn.Module):
    def __init__(self):
        super().__init__()
    
    # MLP 中 gate和up层的输出拼在一起，就是 x
    # 这里计算的 MLP中的第二层， silu和 linear的结果逐元素相乘
    @torch.compile
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, y = x.chunk(2, -1)
        return F.silu(x) * y         