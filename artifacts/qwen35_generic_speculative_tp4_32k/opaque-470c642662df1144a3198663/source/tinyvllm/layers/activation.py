import torch
from torch import nn
import torch.nn.functional as F
from tinyvllm.utils.torch_compile import compile_if_enabled

class SiluAndMul(nn.Module):
    def __init__(self):
        super().__init__()
    
    # MLP 中 gate和up层的输出拼在一起，就是 x
    # 这里计算的 MLP中的第二层， silu和 linear的结果逐元素相乘
    # dynamic=True：prefill 阶段 token 总数会随 batch 变化，让 inductor 走动态形状路径，
    # 避免每个新 shape 都触发重编译（首次冷启会更慢，但稳态零重编译开销）
    @compile_if_enabled(dynamic=True)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, y = x.chunk(2, -1)
        return F.silu(x) * y
