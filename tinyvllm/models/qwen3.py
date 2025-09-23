import torch
from torch import nn
import torch.distributed as dist
from transformers import Qwen3Config

from tinyvllm.layers.activation import SiluAndMul
from tinyvllm.layers.attention import Attention
from tinyvllm.layers.layernorm import RMSNorm
from tinyvllm.layers.linear import QKVParallelLinear, MergedColumnParallelLinear, RowParallelLinear
from tinyvllm.layers.rotary_embedding import get_rope
from tinyvllm.layers.embed_head import VocabParallelEmbedding, ParallelLMHead


class QWen3Attention(nn.Module):
    
    def __init__(
        self, 
        hidden_size: int, 
        num_heads: int,
        num_kv_heads: int,
        max_position: int = 4096 * 32, 
        head_dim: int | None = None,
        rms_norm_eps: float = 1e-6, 
        qkv_bias: bool = False,
        rope_theta: float = 10000, 
        rope_scaling: tuple | None = None
    ):
        super().__init__()
        tp_size = dist.get_world_size()
        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size

        self.total_num_kv_heads = num_kv_heads
        assert self.total_num_kv_heads % tp_size == 0
        self.num_kv_heads = self.total_num_kv_heads // tp_size

        self.head_dim = head_dim or hidden_size // self.total_num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim ** (-0.5)
        
        self.qkv_proj = QKVParallelLinear(
            hidden_size, 
            self.head_dim, 
            self.total_num_heads, 
            self.total_num_kv_heads, 
            bias = qkv_bias)

        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size, 
            bias = False
        )

        self.rotary_emb = get_rope(
            head_size = self.head_dim,
            rotary_dim = self.head_dim,
            max_position = max_position,
            base = rope_theta,
            rope_scaling = rope_scaling             # 缩放旋转角度，增强超长文本的处理能力, 这个参数其实没用到
        )

        self.attn = Attention(
            self.num_heads, 
            self.head_dim,
            self.scaling, 
            self.num_kv_heads
        )

        self.q_norm = RMSNorm(self.head_dim, eps = rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps = rms_norm_eps)

    def forward(self,
                positions: torch.Tensor,                       # [batch_size * seq_len]
                hidden_states: torch.Tensor                    # [batch_size * seq_len, num_kv_heads * head_dim] = [16384, 8 * 128]
        ) -> torch.Tensor:
        qkv = self.qkv_proj(hidden_states)                     #[batch_size×seq_len, q_size + 2×kv_size]   [16384, (2 *8 + 16) * 128] = [16384, 4096]
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim = -1)    # q = [16384, 16 * 128]
        q_by_head = q.view(-1, self.num_heads, self.head_dim)           # q_by_head = [16384, 16, 128]
        k_by_head = k.view(-1, self.num_kv_heads, self.head_dim)
        q_by_head = self.q_norm(q_by_head)
        k_by_head = self.k_norm(k_by_head)
# 注意力权重 = softmax( (Q @ K^T) * 缩放因子 )
# 注意力输出 = 注意力权重 @ V
# V 的数值范围对最终结果的影响远小于 Q/K
# 即使 V 的数值有波动，注意力权重本身已经是 “归一化的概率分布”（sum=1），加权求和后会自然 “平滑” V 的数值差异；
# 退一步说，即使 V 有较大数值，后续通常还有线性层（如代码中的 o_proj）和后续的归一化层（如 Transformer 块的输出归一化），可以进一步调整输出分布，无需在 V 本身额外加归一化。
        q = q_by_head.view(q.shape)         #q变回原来的 q = [16384, 16 * 128]
        k = k_by_head.view(k.shape)

        q, k = self.rotary_emb(positions, q, k)

        o = self.attn(q, k, v)
        output = self.o_proj(o) #1.前面拆分过num_heads 所以需要合并   2.o只是多个注意力头的简单拼接  linear融合 all_reduce聚合 “多头简单拼接的原始特征” 转换为 “整合了所有头信息的优化特征”
        return output
    

class Qwen3MLP(nn.Module):

    def __init__(
        self, 
        hidden_size: int,               # 1024
        intermediate_size: int,         # 3072 = 1024 * 3, 即gate up输出的维度，
        hidden_act: str,                # 激活函数名称，这里仅支持 SiLU
        ): 
        super().__init__()
    # 这里gate和up做Column parallel   down做row parallel
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size, 
            [intermediate_size]*2, 
            bias = False
        )
        #这里 gate_up的输出，在单卡上是 gate分块 + up分块，
        # 因此刚好契合 down_proj的分块，所以不用通信就可以直接计算
        self.down_proj = RowParallelLinear(
            intermediate_size, 
            hidden_size, 
            bias = False)
        assert hidden_act == "silu"
        self.act_fn = SiluAndMul()
        
    def forward(
        self, 
        x: torch.Tensor                   # [16384, 1024] = [batch_size*seq_len, head_num * head_size]
    ) -> torch.Tensor:                    # [16384, 1024]
        gate_up = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x = self.down_proj(x)
        return x
    

class Qwen3DecoderLayer(nn.Module):
    def __init__(
        self,
        config: Qwen3Config
    ):
        super().__init__()
        self.self_attn = QWen3Attention(
            hidden_size = config.hidden_size,
            num_heads = config.num_attention_heads,
            num_kv_heads = config.num_key_value_heads,
            max_position = config.max_position_embeddings, 
            rms_norm_eps=config.rms_norm_eps, 
            qkv_bias=getattr(config, 'attention_bias', False),
            head_dim=getattr(config, 'head_dim', None),
            rope_theta=getattr(config, 'rope_theta', None),
            rope_scaling=getattr(config, 'rope_scaling', None)
        )
        self.mlp = Qwen3MLP(
            hidden_size=config.hidden_size, 
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act
        )
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(                                    #这是整个llama架构的核心
        self,
        positions: torch.Tensor, 
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None
    )-> tuple[torch.Tensor, torch.Tensor]:
        if residual is None:                        #没有残差表示是第一层的attention
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        hidden_states = self.self_attn(positions, hidden_states)
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual
    

class Qwen3Model(nn.Module):
    def __init__(
        self, 
        config: Qwen3Config, 
    ) -> None:
        super().__init__()
        self.embed_tokens = VocabParallelEmbedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([Qwen3DecoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)

    def forward(
        self, 
        input_ids: torch.Tensor,                            # [16384 = batch_size * seq_len], 记录 token 的 id
        positions: torch.Tensor,                            # [16384 = batch_size * seq_ken], 记录每个 token 在 seq 中的位置
    ) -> torch.Tensor:
        hidden_states = self.embed_tokens(input_ids)
        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(positions, hidden_states, residual)
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states
    

class Qwen3ForCausalLM(nn.Module):
    # 打包权重 机制  为了适配tensor parallel 读取一个大文件比读取多个小文件更快
    packed_modules_mapping = {
        "q_proj":("qkv_proj", "q"), 
        "k_proj":("qkv_proj", "k"),
        "v_proj":("qkv_proj", "v"), 
        "gate_proj":("gate_up_proj", 0), 
        "up_proj":("gate_up_proj", 1), 
    }

    def __init__(
        self,
        config: Qwen3Config, 
    ):
        super().__init__()
        self.model = Qwen3Model(config) 
        self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size)
        if config.tie_word_embeddings:              # 和最初的embedding层共用同一权重
            self.lm_head.weight.data = self.model.embed_tokens.weight.data
    
    #这里是在Module里面默认要求重写的 module里面很多函数都是以hook实现 
    # _call_impl 是核心调用方式 主要流程：
    # 前向预处理钩子（forward_pre_hooks）→ 执行 forward → 前向钩子（forward_hooks）[对 forward 的输出（result）做后处理]→ 反向传播钩子准备
    def forward(
        self, 
        input_ids: torch.Tensor, 
        positions: torch.Tensor, 
    ) -> torch.Tensor:
        hidden_states = self.model(input_ids, positions)
        return hidden_states
    
    def compute_logits(
        self, 
        hidden_states: torch.Tensor, 
    ) -> torch.Tensor:
        logits = self.lm_head(hidden_states)
        return logits







