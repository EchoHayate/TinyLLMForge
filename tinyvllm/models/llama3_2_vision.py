import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PretrainedConfig
from typing import Optional, Tuple, List, Union

from tinyvllm.layers.activation import ACT2FN
from tinyvllm.layers.layernorm import RMSNorm
# from tinyvllm.models.llama import LlamaModel, LlamaConfig  <-- REMOVED because file does not exist

# ============================================================================
# 0. Llama Text Model Components (Implemented locally since missing in repo)
# ============================================================================

class LlamaConfig(PretrainedConfig):
    model_type = "llama"
    def __init__(
        self,
        vocab_size=128256,
        hidden_size=4096,
        intermediate_size=14336,
        num_hidden_layers=32,
        num_attention_heads=32,
        num_key_value_heads=8,
        hidden_act="silu",
        max_position_embeddings=131072, # Llama 3.1/3.2 context
        initializer_range=0.02,
        rms_norm_eps=1e-5,
        rope_theta=500000.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta

class LlamaRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps
    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

class LlamaMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

class LlamaAttention(nn.Module):
    # Simplified Self-Attention (No GQA/RoPE details implemented for brevity, assume standard)
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, config.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, config.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
    
    def forward(self, hidden_states, position_ids=None):
        # Placeholder forward
        B, S, _ = hidden_states.shape
        q = self.q_proj(hidden_states).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(hidden_states).view(B, S, -1, self.head_dim).transpose(1, 2)
        v = self.v_proj(hidden_states).view(B, S, -1, self.head_dim).transpose(1, 2)
        # Assuming GQA expansion happened or using scaled_dot_product_attention
        weights = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        weights = F.softmax(weights, dim=-1)
        out = torch.matmul(weights, v)
        out = out.transpose(1, 2).reshape(B, S, -1)
        return self.o_proj(out)

class LlamaDecoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self_attn = LlamaAttention(config)
        self.mlp = LlamaMLP(config)
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
    
    def forward(self, hidden_states, position_ids=None):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = residual + self.self_attn(hidden_states, position_ids)
        
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = residual + self.mlp(hidden_states)
        return hidden_states


# ============================================================================
# 1. Configuration Classes
# ============================================================================

class MllamaVisionConfig(PretrainedConfig):
    model_type = "mllama_vision"
    
    def __init__(
        self,
        hidden_size=1280,
        num_hidden_layers=32,
        num_attention_heads=16,
        intermediate_size=5120,
        image_size=560,
        patch_size=14,
        activation="quick_gelu",
        num_channels=3,
        num_global_layers=8,  # Llama 3.2 often has a global encoder on top
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.image_size = image_size
        self.patch_size = patch_size
        self.activation = activation
        self.num_channels = num_channels
        self.num_global_layers = num_global_layers

class MllamaConfig(PretrainedConfig):
    model_type = "mllama"
    
    def __init__(
        self,
        vision_config=None,
        text_config=None,
        cross_attention_layers: List[int] = None, # Indices of layers with cross-attn
        **kwargs
    ):
        super().__init__(**kwargs)
        if vision_config is None:
            self.vision_config = MllamaVisionConfig()
        elif isinstance(vision_config, dict):
            self.vision_config = MllamaVisionConfig(**vision_config)
        else:
            self.vision_config = vision_config

        if text_config is None:
             self.text_config = LlamaConfig() # Default tinyvllm Llama config
        elif isinstance(text_config, dict):
             self.text_config = LlamaConfig(**text_config)
        else:
             self.text_config = text_config
             
        # Default cross-attention placement for 11B model
        if cross_attention_layers is None:
            self.cross_attention_layers = [3, 8, 13, 18, 23, 28, 33, 38]
        else:
            self.cross_attention_layers = cross_attention_layers


# ============================================================================
# 2. Vision Encoder (The Tower)
# ============================================================================

class MllamaVisionMLP(nn.Module):
    def __init__(self, config: MllamaVisionConfig):
        super().__init__()
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.act = ACT2FN[config.activation]
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))

class MllamaVisionAttention(nn.Module):
    def __init__(self, config: MllamaVisionConfig):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.scaling = self.head_dim ** -0.5

        self.qkv = nn.Linear(self.embed_dim, 3 * self.embed_dim)
        self.proj = nn.Linear(self.embed_dim, self.embed_dim)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x)
        # Reshape and Permute logic for Multi-head attention
        qkv = qkv.reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scaling
        attn_weights = F.softmax(attn_weights, dim=-1)
        output = torch.matmul(attn_weights, v)
        
        output = output.transpose(1, 2).reshape(B, N, C)
        return self.proj(output)

class MllamaVisionBlock(nn.Module):
    def __init__(self, config: MllamaVisionConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.hidden_size, eps=1e-6) # Llama vision uses LayerNorm usually
        self.attn = MllamaVisionAttention(config)
        self.ln2 = nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.mlp = MllamaVisionMLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

class MllamaVisionEncoder(nn.Module):
    def __init__(self, config: MllamaVisionConfig):
        super().__init__()
        self.config = config
        
        # Patch Embedding
        self.patch_embed = nn.Conv2d(
            config.num_channels, 
            config.hidden_size, 
            kernel_size=config.patch_size, 
            stride=config.patch_size,
            bias=False 
        )
        
        # Position Embedding (Global or Learnable)
        # For simplicity we assume learnable or sinusoidal here, 
        # actual implementation might use Gated logic.
        self.class_embedding = nn.Parameter(torch.randn(1, 1, config.hidden_size))
        self.position_embedding = nn.Parameter(torch.randn(1, (config.image_size // config.patch_size)**2 + 1, config.hidden_size))
        
        self.ln_pre = nn.LayerNorm(config.hidden_size)
        
        # Transformer Blocks
        self.layers = nn.ModuleList([MllamaVisionBlock(config) for _ in range(config.num_hidden_layers)])
        
        self.ln_post = nn.LayerNorm(config.hidden_size)
        
    def forward(self, pixel_values):
        # pixel_values: [B, C, H, W]
        B = pixel_values.shape[0]
        x = self.patch_embed(pixel_values) # [B, C, H/P, W/P]
        x = x.flatten(2).transpose(1, 2)   # [B, N, C]
        
        # Add CLS token
        cls_tokens = self.class_embedding.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        
        # Add Positional Embedding
        # Note: real implementation needs interpolation for different resolutions
        x = x + self.position_embedding[:, :x.shape[1], :]
        
        x = self.ln_pre(x)
        
        for layer in self.layers:
            x = layer(x)
            
        x = self.ln_post(x)
        return x


# ============================================================================
# 3. Cross Attention Components (The Mllama Special Sauce)
# ============================================================================

class MllamaCrossAttention(nn.Module):
    """
    Gated Cross Attention Layer.
    Text Hidden States act as Query.
    Vision Features act as Key/Value.
    """
    def __init__(self, config: MllamaConfig):
        super().__init__()
        dim = config.text_config.hidden_size
        self.num_heads = config.text_config.num_attention_heads
        self.head_dim = dim // self.num_heads
        self.scaling = self.head_dim ** -0.5
        
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False) # Note: dimension must match vision output projection
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.o_proj = nn.Linear(dim, dim, bias=False)
        
        # Gating parameter (tanh gate usually)
        self.gate = nn.Parameter(torch.tensor([0.0])) 

    def forward(self, hidden_states, vision_states):
        # hidden_states: [B, SeqLen, Dim] (Text)
        # vision_states: [B, VisLen, Dim] (Vision) - Assumed projected to matches Dim text
        
        B, S, _ = hidden_states.shape
        
        q = self.q_proj(hidden_states).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        
        # For simplicity, assuming vision_states are already projected to match Text Dim
        # In real implementation there is a projection layer before this
        k = self.k_proj(vision_states).view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(vision_states).view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Attention
        weights = torch.matmul(q, k.transpose(-2, -1)) * self.scaling
        weights = F.softmax(weights, dim=-1)
        output = torch.matmul(weights, v)
        
        output = output.transpose(1, 2).contiguous().reshape(B, S, -1)
        output = self.o_proj(output)
        
        # Gated Residual
        # y = x + tanh(gate) * attention_output
        return hidden_states + torch.tanh(self.gate) * output


# ============================================================================
# 4. Model Wrapper (The Integrator)
# ============================================================================

class MllamaForConditionalGeneration(nn.Module):
    def __init__(self, config: MllamaConfig):
        super().__init__()
        self.config = config
        
        # 1. Vision Tower
        self.vision_model = MllamaVisionEncoder(config.vision_config)
        
        # 2. Vision Projection (Align Vision Dim -> Text Dim)
        self.multi_modal_projector = nn.Sequential(
            nn.Linear(config.vision_config.hidden_size, config.text_config.hidden_size),
            nn.GELU(),
            nn.Linear(config.text_config.hidden_size, config.text_config.hidden_size)
        )
        
        # 3. Text Model
        self.vocab_size = config.text_config.vocab_size
        self.embed_tokens = nn.Embedding(config.text_config.vocab_size, config.text_config.hidden_size)
        
        self.layers = nn.ModuleList()
        self.cross_attention_layers = nn.ModuleList()
        
        # Indices where cross attention happens
        cross_attn_indices = set(config.cross_attention_layers)
        
        for i in range(config.text_config.num_hidden_layers):
            self.layers.append(LlamaDecoderLayer(config.text_config))
            
            if i in cross_attn_indices:
                # Add a cross attention layer for this depth
                self.cross_attention_layers.append(MllamaCrossAttention(config))
            else:
                self.cross_attention_layers.append(nn.Identity()) # Placeholder
                
        self.norm = LlamaRMSNorm(config.text_config.hidden_size, eps=config.text_config.rms_norm_eps)
        self.lm_head = nn.Linear(config.text_config.hidden_size, config.text_config.vocab_size, bias=False)

    def forward(
        self,
        input_ids: torch.Tensor,
        pixel_values: Optional[torch.Tensor] = None,
        positions: Optional[torch.Tensor] = None, 
    ):
        # 1. Vision Forward
        vision_features = None
        if pixel_values is not None:
            vision_outputs = self.vision_model(pixel_values) # [B, N_vis, VisDim]
            vision_features = self.multi_modal_projector(vision_outputs) # [B, N_vis, TextDim]

        # 2. Text Embedding
        hidden_states = self.embed_tokens(input_ids)
        
        # 3. Layer Loop (Interleaved)
        cross_attn_indices = set(self.config.cross_attention_layers)
        
        for i, layer in enumerate(self.layers):
            # Self Attention
            hidden_states = layer(hidden_states, positions)
            
            # Cross Attention
            if i in cross_attn_indices:
                if vision_features is not None:
                     cross_layer = self.cross_attention_layers[i]
                     if not isinstance(cross_layer, nn.Identity):
                        hidden_states = cross_layer(hidden_states, vision_features)
                
        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states)
        return logits

