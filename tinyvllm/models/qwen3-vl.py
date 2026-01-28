import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PretrainedConfig
from typing import Optional, Tuple
import math

from tinyvllm.layers.activation import SiluAndMul, ACT2FN
from tinyvllm.layers.attention import Attention
from tinyvllm.layers.layernorm import RMSNorm
from tinyvllm.layers.linear import QKVParallelLinear, RowParallelLinear, MergedColumnParallelLinear
from tinyvllm.models.qwen3 import Qwen3Model, Qwen3Config

class Qwen2VLVisionConfig(PretrainedConfig):
    model_type = "qwen2_vl_vision"
    def __init__(
        self,
        depth=32,
        embed_dim=1280,
        num_heads=16,
        mlp_ratio=16,
        activation="gelu_quick",
        in_channels=3,
        hidden_size=3584,
        patch_size=14,
        spatial_merge_size=2,
        spatial_patch_size=14,
        temporal_patch_size=2,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.depth = depth
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.activation = activation
        self.in_channels = in_channels
        self.hidden_size = hidden_size
        self.patch_size = patch_size
        self.spatial_merge_size = spatial_merge_size
        self.spatial_patch_size = spatial_patch_size
        self.temporal_patch_size = temporal_patch_size

class Qwen2VLVisionRotaryEmbedding(nn.Module):
    def __init__(self, dim: int, theta: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.theta = theta

    def forward(self, seq_len: int) -> torch.Tensor:
        inv_freq = 1.0 / (self.theta ** (torch.arange(0, self.dim, 2, dtype=torch.float) / self.dim))
        return inv_freq


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb_vision(t, freqs):
    t_ = t.float().reshape(*t.shape[:-1], -1, 2)
    freqs = freqs.view(1, -1, 1, t_.shape[-2], t_.shape[-1])
    t_ = torch.view_as_complex(t_)
    freqs = torch.view_as_complex(freqs)
    x_out = torch.view_as_real(t_ * freqs).flatten(3)
    return x_out.type_as(t)

class Qwen2VLVisionAttention(nn.Module):
    def __init__(self, config: Qwen2VLVisionConfig):
        super().__init__()
        self.embed_dim = config.embed_dim
        self.num_heads = config.num_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.scaling = self.head_dim ** -0.5

        self.qkv = nn.Linear(self.embed_dim, 3 * self.embed_dim, bias=True)
        self.proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        rotary_pos_emb: torch.Tensor = None,
    ) -> torch.Tensor:
        seq_length, _ = hidden_states.shape
        qkv = self.qkv(hidden_states)
        qkv = qkv.reshape(seq_length, 3, self.num_heads, -1).permute(1, 0, 2, 3)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Apply RoPE if provided
        if rotary_pos_emb is not None:
             pass

        # Simple attention placeholder
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scaling
        attn_weights = F.softmax(attn_weights, dim=-1)
        output = torch.matmul(attn_weights, v)
        
        output = output.reshape(seq_length, -1)
        output = self.proj(output)
        return output

class Qwen2VLVisionMLP(nn.Module):
    def __init__(self, config: Qwen2VLVisionConfig):
        super().__init__()
        self.fc1 = nn.Linear(config.embed_dim, config.embed_dim * config.mlp_ratio)
        self.act = ACT2FN[config.activation]
        self.fc2 = nn.Linear(config.embed_dim * config.mlp_ratio, config.embed_dim)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(hidden_states)))

class Qwen2VLVisionBlock(nn.Module):
    def __init__(self, config: Qwen2VLVisionConfig):
        super().__init__()
        self.norm1 = RMSNorm(config.embed_dim, eps=1e-6)
        self.attn = Qwen2VLVisionAttention(config)
        self.norm2 = RMSNorm(config.embed_dim, eps=1e-6)
        self.mlp = Qwen2VLVisionMLP(config)

    def forward(self, hidden_states: torch.Tensor, cu_seqlens: torch.Tensor, rotary_pos_emb: torch.Tensor = None) -> torch.Tensor:
        hidden_states = hidden_states + self.attn(self.norm1(hidden_states), cu_seqlens, rotary_pos_emb)
        hidden_states = hidden_states + self.mlp(self.norm2(hidden_states))
        return hidden_states

class Qwen2VLVisionPatchMerger(nn.Module):
    def __init__(self, config: Qwen2VLVisionConfig):
        super().__init__()
        self.embed_dim = config.embed_dim
        self.inter_embed_dim = config.embed_dim * (config.spatial_merge_size**2)
        self.ln_q = RMSNorm(config.embed_dim, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(self.inter_embed_dim, self.inter_embed_dim),
            ACT2FN[config.activation],
            nn.Linear(self.inter_embed_dim, config.hidden_size),
        )

    def forward(self, hidden_states: torch.Tensor, rotary_pos_emb: torch.Tensor = None) -> torch.Tensor:
        h = int(math.sqrt(hidden_states.shape[0])) # Simplified assumption: square image
        w = h
        if h * w != hidden_states.shape[0]:
            print(f"Warning: Image size {hidden_states.shape[0]} is not square. Reshape will fail!")
            pass
        
        # Reshape to [H, W, C]
        x = hidden_states.view(h, w, -1)
        # Reshape to [H/2, 2, W/2, 2, C]
        x = x.view(h // 2, 2, w // 2, 2, -1)
        # Permute to [H/2, W/2, 2, 2, C]
        x = x.permute(0, 2, 1, 3, 4)
        # Flatten to [H/2, W/2, 4*C]
        x = x.contiguous().view(h // 2, w // 2, -1)
        # Flatten back to [H*W/4, 4*C]
        x = x.view(-1, self.inter_embed_dim)
        
        return self.mlp(x)

class Qwen2VLVisionTransformer(nn.Module):
    def __init__(self, config: Qwen2VLVisionConfig):
        super().__init__()
        self.config = config
        self.patch_embed = nn.Sequential(
            nn.Conv2d(
                config.in_channels,
                config.embed_dim,
                kernel_size=config.patch_size,
                stride=config.patch_size,
                bias=False
            )
        )
        self.blocks = nn.ModuleList([Qwen2VLVisionBlock(config) for _ in range(config.depth)])
        self.merger = Qwen2VLVisionPatchMerger(config)

    def forward(self, pixel_values: torch.Tensor, image_grid_thw: torch.Tensor = None) -> torch.Tensor:
        # 1. Patch Embedding Handling
        # If input is raw pixels [Batch, 3, H, W], we need to embed it.
        # If input is already flattened patches [Batch*L, EmbedDim], we skip.
        if pixel_values.dim() == 4:
            hidden_states = self.patch_embed(pixel_values)
            # [B, C, H, W] -> [B, C, L] -> [B, L, C] -> [B*L, C] (Packed)
            # Note: real Qwen2VL uses "Naive Dynamic Resolution" where images are packed.
            # Here we simplify:
            hidden_states = hidden_states.flatten(2).transpose(1, 2) # [B, L, C]
            hidden_states = hidden_states.flatten(0, 1) # [B*L, C] Packed
        else:
            hidden_states = pixel_values 
            
        rotary_pos_emb = None 
        # grid_thw would be used here to compute rotary_pos_emb specific to each image's geometry
        # To make RoPE work, we need to generate frequencies based on grid_thw
        # For this prototype we leave it as None or simple sequence based.
        
        for block in self.blocks:
            hidden_states = block(hidden_states, cu_seqlens, rotary_pos_emb)
        
        hidden_states = self.merger(hidden_states)
        return hidden_states

class Qwen3VLModel(Qwen3Model):
    def __init__(self, config: Qwen3Config, vision_config: Qwen2VLVisionConfig):
        super().__init__(config)
        self.visual = Qwen2VLVisionTransformer(vision_config)
        
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        pixel_values: torch.Tensor = None,
        image_grid_thw: torch.Tensor = None,
    ) -> torch.Tensor:
        text_embeds = self.embed_tokens(input_ids)
        
        if pixel_values is not None:
             visual_embeds = self.visual(pixel_values, image_grid_thw)
             # Vision Fusion: 
             # In Qwen2-VL, special tokens (e.g. <|image_pad|>) in input_ids are replaced by visual_embeds.
             # We assume input_ids contains a specific placeholder token ID for images.
             # For this implementation, we'll use a mask based approach.
             
             # TODO: You need to define the image_token_id. Standard Qwen2VL uses 151655.
             image_token_id = 151655 
             image_mask = (input_ids == image_token_id)
             
             # Safety check: ensure number of placeholders matches visual features
             # Note: In real scenarios, this requires careful alignment of token counts.
             # Here we perform a simple strict scatter update if counts match, 
             # or a broadcast/fill if they differ (which implies more complex logic).
             
             # Simplified replacement logic:
             # creating a clone to avoid in-place modification errors during backprop if any
             hidden_states = text_embeds.clone()
             
             if image_mask.sum() > 0:
                 # We assume visual_embeds is [Total_Visual_Tokens, Hidden_Size]
                 # and image_mask has exactly Total_Visual_Tokens trues.
                 # If shapes match perfectly:
                 if visual_embeds.shape[0] == image_mask.sum():
                    hidden_states[image_mask] = visual_embeds.to(hidden_states.dtype)
                 else:
                    # If mismatch (e.g. due to packing logic differences), 
                    # we would typically resize or error out. 
                    # For stability in this demo, we skip or print warning.
                    print(f"Warning: Visual embeds shape {visual_embeds.shape} != Mask sum {image_mask.sum()}")
        else:
             hidden_states = text_embeds
        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(positions, hidden_states, residual)
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states