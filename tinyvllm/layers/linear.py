import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist

from tinyvllm.layers.quantization import (
    quantize_weight,
    dequantize_weight,
    fake_quantize_act_int8,
)

# bnb 是可选依赖：只有走 int8_bnb fused GEMM 路径才需要
try:
    import bitsandbytes.functional as _bnbF                       # type: ignore
except Exception:                                                 # noqa
    _bnbF = None


def _bnb_int8_matmul(x: torch.Tensor, qweight: torch.Tensor,
                     w_scales: torch.Tensor, bias: torch.Tensor | None) -> torch.Tensor:
    """fused W8A16: 输入 fp16/bf16 -> 行量化 int8 -> int8 GEMM -> dequant fp16 -> 还原原 dtype。

    qweight: [out, in] int8 (per-row 量化)
    w_scales: [out] float32 (= max_abs/127)
    """
    if _bnbF is None or x.dtype not in (torch.float16, torch.bfloat16):
        return None
    orig_dtype = x.dtype
    orig_shape = x.shape
    # bnb 的 int8_vectorwise_quant 只支持 fp16 输入
    x2d = x.reshape(-1, orig_shape[-1])
    if orig_dtype != torch.float16:
        x2d = x2d.to(torch.float16)
    x2d = x2d.contiguous()
    qx, sx, _ = _bnbF.int8_vectorwise_quant(x2d)                  # qx int8, sx fp32 [M]
    out_i32 = _bnbF.int8_linear_matmul(qx, qweight)               # [M, out_features] int32
    # bnb 的 col_stats 期望 weight per-row absmax；我们存的是 absmax/127，因此乘回 127
    col_stats = (w_scales.to(torch.float32) * 127.0)
    y = _bnbF.int8_mm_dequant(out_i32, sx, col_stats, bias=None)  # [M, out] fp16
    y = y.reshape(*orig_shape[:-1], qweight.shape[0])
    if orig_dtype != torch.float16:
        y = y.to(orig_dtype)
    if bias is not None:
        y = y + bias
    return y

# ------------------------------------------------------------------
# 全局量化 / cpu-offload 设置
# 由 ModelRunner 在构建模型前通过 set_quant_config() 注入。
# 这样可以避免改动每个层的构造签名（侵入性最小）。
# ------------------------------------------------------------------
_QUANT_METHOD: str | None = None
_QUANT_GROUP_SIZE: int = 128
_ACT_QUANT_BITS: int = 0


def set_quant_config(method: str | None, group_size: int = 128, act_bits: int = 0):
    global _QUANT_METHOD, _QUANT_GROUP_SIZE, _ACT_QUANT_BITS
    _QUANT_METHOD = method
    _QUANT_GROUP_SIZE = group_size
    _ACT_QUANT_BITS = act_bits


def get_quant_method() -> str | None:
    return _QUANT_METHOD


def divide(numerator, denominator):
    assert numerator % denominator == 0
    return numerator // denominator


# ------------------------------------------------------------------
# Quant 工具：将一个普通 Parameter 替换为 (qweight, scales) buffer 对，
# 同时保留一个轻量 Parameter “shadow” 以兼容 weight_loader 流程。
# 加载流程：loader 把 fp 权重写到 self.weight.data -> 在 finalize_quantization()
# 中真正做量化并丢掉 fp 权重。
# ------------------------------------------------------------------
class _QuantMixin:
    """为线性层提供量化辅助方法（在 self 上挂 qweight / scales buffer）。"""

    quant_method: str | None = None
    quant_group_size: int = 128
    act_quant_bits: int = 0

    def _maybe_init_quant(self):
        self.quant_method = _QUANT_METHOD
        self.quant_group_size = _QUANT_GROUP_SIZE
        self.act_quant_bits = _ACT_QUANT_BITS

    def finalize_quantization(self):
        """把当前 self.weight 量化为 (qweight, scales)，释放 fp weight。"""
        if self.quant_method is None:
            return
        weight = self.weight.data
        qweight, scales = quantize_weight(weight, self.quant_method, self.quant_group_size)
        # 用 buffer 注册（不参与梯度，且 .to(device) 时会跟随移动）
        self.register_buffer("qweight", qweight, persistent=False)
        self.register_buffer("scales", scales, persistent=False)
        # 释放 fp 权重：把 Parameter 设为 None（nn.Module 允许）
        self.weight = None

    def _get_dequantized_weight(self, dtype: torch.dtype) -> torch.Tensor:
        return dequantize_weight(
            self.qweight, self.scales,
            self.quant_method, self.quant_group_size, dtype,
        )

    def _linear_forward(self, x: torch.Tensor, bias: torch.Tensor | None) -> torch.Tensor:
        # A8 假量化（对所有量化路径都生效，包括 None；naive W4A8 用，没有 weight 量化时也允许 A8 验 act 噪声单独的影响）
        if self.act_quant_bits == 8:
            x = fake_quantize_act_int8(x)
        if self.quant_method is None:
            return F.linear(x, self.weight, bias)
        # int8_bnb：fused W8A16 GEMM（仅 fp16 走快路径，bf16 fallback）
        if self.quant_method == "int8_bnb":
            y = _bnb_int8_matmul(x, self.qweight, self.scales, bias)
            if y is not None:
                return y
        # weight-only 量化：临时反量化（同设备：x.device 即 weight 当前所在设备）
        w = self._get_dequantized_weight(x.dtype)
        return F.linear(x, w, bias)


class LinearBase(nn.Module, _QuantMixin):
    def __init__(self, 
        input_size: int,        #用 input_size（输入维度）和 output_size（输出维度）对应线性层权重矩阵的列数和行数
        output_size: int,
        tp_dim: int | None = None,
        ):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.tp_dim = tp_dim                        # 张量并行的维度，0维，1维...
        self.tp_rank = dist.get_rank()
        self.tp_size = dist.get_world_size()        # 张量并行的数量
        self._maybe_init_quant()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
    

class ReplicatedLinear(LinearBase):

    def __init__(
        self, 
        input_size: int, 
        output_size: int, 
        bias: bool = False
    ):
        super().__init__(input_size, output_size)
        self.weight = nn.Parameter(torch.empty(self.output_size, self.input_size))
        self.weight.weight_loader = self.weight_loader
        if bias:
            self.bias = nn.Parameter(torch.empty(self.output_size))   #节省显存 后面会有广播机制拓展为 [batch_size,output_size]
            self.bias.weight_loader = self.weight_loader
        else:
            self.register_parameter("bias", None)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        param.data.copy_(loaded_weight)     #param 是 带身份的参数包装器，param.data 是这个包装器里 实际存储数值的张量
        

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._linear_forward(x, self.bias)

#列切分最后需要concatenate tp_dim = 0 对output_size切割就是列切割
class ColumnParallelLinear(LinearBase):

    def __init__(
        self, 
        input_size: int, 
        output_size: int, 
        bias: bool = False
    ):
        super().__init__(input_size, output_size, 0)                            
        # F.linear(x, weight) = x @ weight^T， 因此weight = [output_size, input_size]，   x = [batch_size, input_size]
        # tp_dim = 0, 所有只有 ouput_size被切割, w^T, weight会有一个转置 则output_size就是列拆分
        #为了保证线性变换的输入[batch_size, input_size] 输出[batch_size, output_size]  所以框架存储约定 定义的weight维度[output_size, input_size]
        self.input_size_per_partition = input_size
        self.output_size_per_partition = divide(output_size, self.tp_size) 
        self.weight = nn.Parameter(torch.empty(self.output_size_per_partition, self.input_size))
        self.weight.weight_loader = self.weight_loader

        if bias:
            self.bias = nn.Parameter(torch.empty(self.output_size_per_partition))
            self.bias.weight_loader = self.weight_loader
        else:
            self.register_parameter("bias", None)           #注册参数 防止后面调用报错

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        param_data = param.data
        shard_size = param_data.size(self.tp_dim)
        start_idx = self.tp_rank * shard_size
        loaded_weight = loaded_weight.narrow(self.tp_dim, start_idx, shard_size)
        param_data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._linear_forward(x, self.bias)
    

class MergedColumnParallelLinear(ColumnParallelLinear):         #针对FFN的gate up做切分
    def __init__(
            self, 
            input_size: int, 
            output_sizes: list[int],        #只有gate和up
            bias: bool = False):
        self.output_sizes = output_sizes
        super().__init__(input_size, sum(output_sizes), bias = bias)
    #param narrow把原来能放x的长度大小给切分成y的片段 然后weightloader又把原来要放在一个位置上的权重平分成worldsize块 在放到大小刚刚合适的y尺寸的param上
    #gate和up拼接起来的大矩阵进行分片
    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor, loaded_shared_id: int):   #这里的loaded_shared_id将gate和up分开处理 所以不会导致gate up混合切片
        param_data = param.data
        # 在每个节点上，gate和up权重是交替的，即每个节点都有部分 gate 权重和 up 权重
        # loaded_shared_id = 0/1, 0表示取出 gate 的部分权重，1表示取出 up 的部分权重
        # para_data是单卡的权重数据，不是所有的
        shard_offset = sum(self.output_sizes[:loaded_shared_id]) // self.tp_size    #gate_size // tp_size  分片中的起始位   对于当前模型 只有gate up两维度 可以不加sum
        shard_size = self.output_sizes[loaded_shared_id] // self.tp_size            #步长
        param_data = param_data.narrow(self.tp_dim, shard_offset, shard_size)       #按照tp_dim进行分片
        loaded_weight = loaded_weight.chunk(self.tp_size, self.tp_dim)[self.tp_rank]        #narrow() 是 “在单个张量中截取连续片段”，而 chunk() 是 “将张量平均拆分成多个子张量”
        param_data.copy_(loaded_weight)
        #param表示空容器 loaded_weight表示实际权重



# q = W_q @ hidden_states，k = W_k @ hidden_states，v = W_v @ hidden_states
# W_q、W_k、W_v 这 3 个权重矩阵横向拼接成一个更大的矩阵 W_qkv，合并的目的是减少矩阵乘法的调用次数（GPU 对单次大矩阵乘法的优化更好）
# [hidden_size, (num_q_heads + num_k_heads + num_v_heads) * head_dim]，然后用一次矩阵乘法就能同时算出 q、k、v 的拼接结果
class QKVParallelLinear(ColumnParallelLinear):                      #针对attention的QKV做切分
    def __init__(
        self, 
        hidden_size: int, 
        head_size: int, 
        total_num_heads: int,
        total_num_kv_heads: int | None, 
        bias: bool = False, 
    ):
        self.head_size = head_size      #for qwen3 0.6b  128
        self.total_num_heads = total_num_heads      #for qwen3 0.6b  16
        self.total_num_kv_heads = total_num_kv_heads or total_num_heads     #for qwen3 0.6b  8 or 16
        tp_size = dist.get_world_size()
        self.num_heads = divide(self.total_num_heads, tp_size)      #for qwen3  16/tp_size
        self.num_kv_heads = divide(self.total_num_kv_heads, tp_size)    #for qwen3  8/tp_size

        input_size = hidden_size
        # q + k + v = q + 2 * k/v
        output_size = (self.num_heads + 2 * self.num_kv_heads) * head_size
        super().__init__(input_size, output_size, bias)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor, loaded_shared_id: str):
        param_data = param.data
        assert loaded_shared_id in ["q", "k", "v"]
        if loaded_shared_id == "q":
            # 一张卡上并不是一个头，可能也是多头
            shard_size = self.num_heads * self.head_size
            shard_offset = 0
        elif loaded_shared_id == "k":
            shard_size = self.num_kv_heads * self.head_size
            shard_offset = self.num_heads * self.head_size
        else:
            shard_size = self.num_kv_heads * self.head_size
            shard_offset = self.num_heads * self.head_size + self.num_kv_heads * self.head_size
        
        param_data = param_data.narrow(self.tp_dim, shard_offset, shard_size)
        loaded_weight = loaded_weight.chunk(self.tp_size, self.tp_dim)[self.tp_rank]
        param_data.copy_(loaded_weight)

#行切分最后需要all_reduce  tp_dim = 1
class RowParallelLinear(LinearBase):

    def __init__(self, 
        input_size: int, 
        output_size: int,
        bias: bool = False
    ):
        super().__init__(input_size, output_size, 1)
        self.input_size_per_partition = divide(input_size, self.tp_size)

        self.weight = nn.Parameter(torch.empty(output_size, self.input_size_per_partition))
        self.weight.weight_loader = self.weight_loader
        if bias:
            self.bias = nn.Parameter(torch.empty(output_size))
            self.bias.weight_loader = self.weight_loader
        else:
            self.register_parameter("bias", None)
        

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        param_data = param.data
        shard_size = param_data.size(self.tp_dim)
        start_idx = self.tp_rank * shard_size
        loaded_weight = loaded_weight.narrow(self.tp_dim, start_idx, shard_size)
        param_data.copy_(loaded_weight)
        

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bias = self.bias if self.tp_rank == 0 else None
        y = self._linear_forward(x, bias)                       #简单的矩阵乘
        if self.tp_size > 1:
            dist.all_reduce(y)
        return y
