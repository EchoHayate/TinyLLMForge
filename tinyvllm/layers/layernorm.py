import torch
from torch import nn

# RMSNorm: x / (sum(x_i^2) + epsilon) * gamma
class RMSNorm(nn.Module):
    
    def __init__(
        self,
        hidden_size: int, 
        eps: float = 1e-6
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))
    
    @torch.compile
    def rms_forward(
        self, 
        x: torch.Tensor                     # [batch_size, seq_len, hidden_size]
        ) -> torch.Tensor:
        origin_dtype = x.dtype
        x = x.to(torch.float32)
        var = x.pow(2).mean(dim = -1, keepdim = True)
        x.mul_(torch.rsqrt(var + self.eps))
        x = x.to(origin_dtype).mul_(self.weight)
        return x

    @torch.compile
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


