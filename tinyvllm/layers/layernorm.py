import torch
from torch import nn
# x_normalized = (x / sqrt(mean(x²) + ε)) * γ   (✔)
# RMSNorm: x / (sum(x_i^2) + epsilon) * gamma  (❌)
# epsilon：一个极小值（如 1e-6），防止分母为 0；
# γ（weight）：可学习的缩放参数（初始为 1），用于恢复特征的表达能力（避免归一化后特征被 “抹平”）
class RMSNorm(nn.Module):
# LayerNorm 计算的是 “均值中心化 + 方差缩放”（(x - mean)/sqrt(var + ε) * γ + β）；
# RMSNorm 跳过了 “均值中心化”，只做 “均方根缩放”（x / sqrt(mean(x²) + ε) * γ）。
    def __init__(
        self,
        hidden_size: int, 
        eps: float = 1e-6
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))
    
    # 图优化
    @torch.compile                          #RMS无残差
    def rms_forward(            
        self, 
        x: torch.Tensor                     # [batch_size, seq_len, hidden_size]
        ) -> torch.Tensor:
        origin_dtype = x.dtype
        x = x.to(torch.float32)     # 转为 float32 计算（避免低精度下的数值不稳定）
        var = x.pow(2).mean(dim = -1, keepdim = True)
        x.mul_(torch.rsqrt(var + self.eps))
        x = x.to(origin_dtype).mul_(self.weight)    #转 fp32 只是为了完成 RMSNorm 核心计算，计算结束后必须换回原始精度        
        return x

    @torch.compile                          #RMS有残差
    def add_rms_forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        origin_dtype = x.dtype
        x = x.to(torch.float32).add_(residual.to(torch.float32))
        residual = x.to(origin_dtype)
        var = x.pow(2).mean(dim = -1, keepdim = True)
        x.mul_(torch.rsqrt(var + self.eps))
        x = x.to(origin_dtype).mul_(self.weight)
        return x, residual

    def forward(self,
        x: torch.Tensor, 
        residual: torch.Tensor | None = None
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            return self.rms_forward(x)
        else:
            return self.add_rms_forward(x, residual)


