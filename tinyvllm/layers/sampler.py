import torch
from torch import nn

class Sampler(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self, 
        logits: torch.Tensor,           # [batch_size, num_token_ids]
        temperatures: torch.Tensor,     # [batch_size]
    ) -> torch.Tensor:                  # [batch_size]
        logits = logits.to(torch.float32)
        greedy_tokens = logits.argmax(dim = -1)
        logits.div_(temperatures.unsqueeze(dim = 1))
        probs = torch.softmax(logits, dim = -1, dtype = torch.float32)
        epsilon = 1e-10
        sample_tokens = probs.div_(torch.empty_like(probs).exponential_(1) + epsilon).argmax(dim = -1)
        return torch.where(temperatures == 0, greedy_tokens, sample_tokens)