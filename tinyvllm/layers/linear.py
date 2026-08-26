import importlib

import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist
from tinyvllm.engine.decode_internal_profiler import (
    profile_collective,
    profile_operation,
)

from tinyvllm.layers.quantization import (
    quantize_weight,
    dequantize_weight,
    fake_quantize_act_int8,
)

_BNB_FUNCTIONAL = None
_BNB_IMPORT_ATTEMPTED = False


def _load_bitsandbytes_functional():
    global _BNB_FUNCTIONAL, _BNB_IMPORT_ATTEMPTED
    if not _BNB_IMPORT_ATTEMPTED:
        _BNB_IMPORT_ATTEMPTED = True
        try:
            _BNB_FUNCTIONAL = importlib.import_module(
                "bitsandbytes.functional"
            )
        except Exception:
            _BNB_FUNCTIONAL = None
    return _BNB_FUNCTIONAL


def _bnb_int8_matmul(x: torch.Tensor, qweight: torch.Tensor,
                     w_scales: torch.Tensor, bias: torch.Tensor | None) -> torch.Tensor:
    """fused W8A16: 输入 fp16/bf16 -> 行量化 int8 -> int8 GEMM -> dequant fp16 -> 还原原 dtype。

    qweight: [out, in] int8 (per-row 量化)
    w_scales: [out] float32 (= max_abs/127)
    """
    if x.dtype not in (torch.float16, torch.bfloat16):
        return None
    bnb_functional = _load_bitsandbytes_functional()
    if bnb_functional is None:
        return None
    orig_dtype = x.dtype
    orig_shape = x.shape
    # bnb 的 int8_vectorwise_quant 只支持 fp16 输入
    x2d = x.reshape(-1, orig_shape[-1])
    if orig_dtype != torch.float16:
        x2d = x2d.to(torch.float16)
    x2d = x2d.contiguous()
    qx, sx, _ = bnb_functional.int8_vectorwise_quant(x2d)         # qx int8, sx fp32 [M]
    out_i32 = bnb_functional.int8_linear_matmul(qx, qweight)      # [M, out_features] int32
    # bnb 的 col_stats 期望 weight per-row absmax；我们存的是 absmax/127，因此乘回 127
    col_stats = (w_scales.to(torch.float32) * 127.0)
    y = bnb_functional.int8_mm_dequant(                            # [M, out] fp16
        out_i32,
        sx,
        col_stats,
        bias=None,
    )
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
        self.linear_execution_rows = 0

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

    def _linear_forward_unpartitioned(
        self,
        x: torch.Tensor,
        bias: torch.Tensor | None,
    ) -> torch.Tensor:
        # SmoothQuant：x = x / s 把激活离群值"除掉"（s 已经在 loader 中乘进 weight）
        # buffer 仅在校准 + loader 注入路径下存在；getattr 默认 None 保证未启用零开销
        ss = getattr(self, "smooth_scale", None)
        if ss is not None:
            # 防呆：input-channel 维必须对得上（最后一维），否则除法会广播到错位
            assert ss.shape[-1] == x.shape[-1], (
                f"[smoothquant] smooth_scale dim {ss.shape[-1]} != x last dim {x.shape[-1]}"
            )
            x = x / ss
        # A8 假量化（对所有量化路径都生效，包括 None；naive W4A8 用，没有 weight 量化时也允许 A8 验 act 噪声单独的影响）
        if self.act_quant_bits == 8:
            x = fake_quantize_act_int8(x)
        operation_name = f"{type(self).__name__}.linear"
        if self.quant_method is None:
            with profile_operation(
                "gemm",
                operation_name,
                tensor=x,
            ):
                return F.linear(x, self.weight, bias)
        # int8_bnb：fused W8A16 GEMM（仅 fp16 走快路径，bf16 fallback）
        if self.quant_method == "int8_bnb":
            with profile_operation(
                "gemm",
                operation_name,
                tensor=x,
            ):
                y = _bnb_int8_matmul(
                    x,
                    self.qweight,
                    self.scales,
                    bias,
                )
            if y is not None:
                return y
        # weight-only 量化：临时反量化（同设备：x.device 即 weight 当前所在设备）
        w = self._get_dequantized_weight(x.dtype)
        with profile_operation(
            "gemm",
            operation_name,
            tensor=x,
        ):
            return F.linear(x, w, bias)

    def _linear_forward(
        self,
        x: torch.Tensor,
        bias: torch.Tensor | None,
    ) -> torch.Tensor:
        rows = self.linear_execution_rows
        if x.ndim != 2 or rows <= 0 or x.shape[0] <= rows:
            return self._linear_forward_unpartitioned(x, bias)
        return torch.cat([
            self._linear_forward_unpartitioned(chunk, bias)
            for chunk in x.split(rows, dim=0)
        ])

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


class ReplicatedLocalOutputLinear(LinearBase):

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
    ):
        super().__init__(input_size, output_size, 0)
        self.output_size_per_partition = divide(
            output_size,
            self.tp_size,
        )
        self.weight = nn.Parameter(
            torch.empty(self.output_size, self.input_size)
        )
        self.weight.weight_loader = self.weight_loader
        if bias:
            self.bias = nn.Parameter(torch.empty(self.output_size))
            self.bias.weight_loader = self.weight_loader
        else:
            self.register_parameter("bias", None)

    def weight_loader(
        self,
        param: nn.Parameter,
        loaded_weight: torch.Tensor,
    ):
        param.data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        full_output = self._linear_forward_unpartitioned(
            x.unsqueeze(0),
            self.bias,
        ).squeeze(0)
        return full_output.narrow(
            -1,
            self.tp_rank * self.output_size_per_partition,
            self.output_size_per_partition,
        )


class ReplicatedColumnParallelLinear(ReplicatedLinear):

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
    ):
        if output_size % dist.get_world_size() != 0:
            raise ValueError(
                "output_size must be divisible by tensor parallel size"
            )
        super().__init__(input_size, output_size, bias=bias)
        self.output_size_per_partition = output_size // self.tp_size
        self.requires_unpartitioned_linear_execution = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        full_output = self._linear_forward_unpartitioned(x, self.bias)
        return full_output.narrow(
            -1,
            self.tp_rank * self.output_size_per_partition,
            self.output_size_per_partition,
        )


class ReplicatedSegmentedColumnParallelLinear(ReplicatedLinear):

    def __init__(
        self,
        input_size: int,
        output_sizes: list[int] | tuple[int, ...],
        bias: bool = False,
    ):
        if not output_sizes:
            raise ValueError("output_sizes must contain at least one segment")
        tp_size = dist.get_world_size()
        if any(
            isinstance(output_size, bool)
            or not isinstance(output_size, int)
            or output_size <= 0
            or output_size % tp_size != 0
            for output_size in output_sizes
        ):
            raise ValueError(
                "output_sizes must contain positive TP-divisible integers"
            )
        self.output_sizes = tuple(output_sizes)
        self.local_output_sizes = tuple(
            output_size // tp_size for output_size in self.output_sizes
        )
        super().__init__(
            input_size,
            sum(self.output_sizes),
            bias=bias,
        )
        self.requires_unpartitioned_linear_execution = True

    def weight_loader(
        self,
        param: nn.Parameter,
        loaded_weight: torch.Tensor,
        loaded_segment_id: int | None = None,
    ):
        if loaded_segment_id is None:
            if loaded_weight.shape != param.shape:
                raise ValueError("fused loaded weight shape is invalid")
            param.data.copy_(loaded_weight)
            return
        if (
            isinstance(loaded_segment_id, bool)
            or not isinstance(loaded_segment_id, int)
            or loaded_segment_id < 0
            or loaded_segment_id >= len(self.output_sizes)
        ):
            raise ValueError("loaded_segment_id must select a valid segment")
        size = self.output_sizes[loaded_segment_id]
        if loaded_weight.shape != (size, self.input_size):
            raise ValueError("loaded segment shape is invalid")
        offset = sum(self.output_sizes[:loaded_segment_id])
        param.data.narrow(0, offset, size).copy_(loaded_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        full_output = self._linear_forward_unpartitioned(x, self.bias)
        outputs = []
        global_offset = 0
        for output_size, local_output_size in zip(
            self.output_sizes,
            self.local_output_sizes,
        ):
            outputs.append(full_output.narrow(
                -1,
                global_offset + self.tp_rank * local_output_size,
                local_output_size,
            ))
            global_offset += output_size
        return torch.cat(outputs, dim=-1)


class ReplicatedHeadPairedColumnParallelLinear(ReplicatedLinear):

    def __init__(
        self,
        input_size: int,
        num_heads: int,
        head_dim: int,
        bias: bool = False,
    ):
        if (
            isinstance(num_heads, bool)
            or not isinstance(num_heads, int)
            or num_heads <= 0
        ):
            raise ValueError("num_heads must be a positive integer")
        if (
            isinstance(head_dim, bool)
            or not isinstance(head_dim, int)
            or head_dim <= 0
        ):
            raise ValueError("head_dim must be a positive integer")
        tp_size = dist.get_world_size()
        if num_heads % tp_size != 0:
            raise ValueError(
                f"num_heads {num_heads} must be divisible by "
                f"tensor parallel size {tp_size}"
            )
        self.num_heads = num_heads
        self.local_num_heads = num_heads // tp_size
        self.head_dim = head_dim
        super().__init__(
            input_size,
            num_heads * 2 * head_dim,
            bias=bias,
        )
        self.local_output_size = (
            self.local_num_heads * 2 * self.head_dim
        )
        self.requires_unpartitioned_linear_execution = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        full_output = self._linear_forward_unpartitioned(x, self.bias)
        return full_output.narrow(
            -1,
            self.tp_rank * self.local_output_size,
            self.local_output_size,
        )


class ReplicatedKVHeadParallelLinear(ReplicatedLinear):

    def __init__(
        self,
        input_size: int,
        total_num_kv_heads: int,
        head_dim: int,
        bias: bool = False,
    ):
        if (
            isinstance(total_num_kv_heads, bool)
            or not isinstance(total_num_kv_heads, int)
            or total_num_kv_heads <= 0
        ):
            raise ValueError(
                "total_num_kv_heads must be a positive integer"
            )
        if (
            isinstance(head_dim, bool)
            or not isinstance(head_dim, int)
            or head_dim <= 0
        ):
            raise ValueError("head_dim must be a positive integer")
        tp_size = dist.get_world_size()
        if total_num_kv_heads >= tp_size:
            if total_num_kv_heads % tp_size != 0:
                raise ValueError(
                    "KV-head sharding requires total_num_kv_heads "
                    "to be divisible by tensor parallel size"
                )
            self.local_num_kv_heads = total_num_kv_heads // tp_size
            self.num_kv_head_replicas = 1
        else:
            if tp_size % total_num_kv_heads != 0:
                raise ValueError(
                    "KV-head replication requires tensor parallel "
                    "size to be divisible by total_num_kv_heads"
                )
            self.local_num_kv_heads = 1
            self.num_kv_head_replicas = (
                tp_size // total_num_kv_heads
            )
        self.total_num_kv_heads = total_num_kv_heads
        self.head_dim = head_dim
        super().__init__(
            input_size,
            total_num_kv_heads * head_dim,
            bias=bias,
        )
        self.source_kv_rank = (
            self.tp_rank // self.num_kv_head_replicas
        )
        self.local_output_size = (
            self.local_num_kv_heads * self.head_dim
        )
        self.requires_unpartitioned_linear_execution = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        full_output = self._linear_forward_unpartitioned(x, self.bias)
        return full_output.narrow(
            -1,
            self.source_kv_rank * self.local_output_size,
            self.local_output_size,
        )


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
    

class KVHeadParallelLinear(LinearBase):

    def __init__(
        self,
        input_size: int,
        total_num_kv_heads: int,
        head_dim: int,
        bias: bool = False,
    ):
        if (
            isinstance(total_num_kv_heads, bool)
            or not isinstance(total_num_kv_heads, int)
            or total_num_kv_heads <= 0
        ):
            raise ValueError(
                "total_num_kv_heads must be a positive integer"
            )
        if (
            isinstance(head_dim, bool)
            or not isinstance(head_dim, int)
            or head_dim <= 0
        ):
            raise ValueError("head_dim must be a positive integer")
        super().__init__(
            input_size,
            total_num_kv_heads * head_dim,
            0,
        )
        self.total_num_kv_heads = total_num_kv_heads
        self.head_dim = head_dim
        if total_num_kv_heads >= self.tp_size:
            if total_num_kv_heads % self.tp_size != 0:
                raise ValueError(
                    "KV-head sharding requires total_num_kv_heads "
                    "to be divisible by tensor parallel size"
                )
            self.local_num_kv_heads = (
                total_num_kv_heads // self.tp_size
            )
            self.num_kv_head_replicas = 1
        else:
            if self.tp_size % total_num_kv_heads != 0:
                raise ValueError(
                    "KV-head replication requires tensor parallel "
                    "size to be divisible by total_num_kv_heads"
                )
            self.local_num_kv_heads = 1
            self.num_kv_head_replicas = (
                self.tp_size // total_num_kv_heads
            )
        self.source_kv_rank = (
            self.tp_rank // self.num_kv_head_replicas
        )
        local_output_size = self.local_num_kv_heads * head_dim
        self.weight = nn.Parameter(
            torch.empty(local_output_size, input_size)
        )
        self.weight.weight_loader = self.weight_loader
        if bias:
            self.bias = nn.Parameter(torch.empty(local_output_size))
            self.bias.weight_loader = self.weight_loader
        else:
            self.register_parameter("bias", None)

    def weight_loader(
        self,
        param: nn.Parameter,
        loaded_weight: torch.Tensor,
    ):
        shard_size = param.data.size(self.tp_dim)
        start = self.source_kv_rank * shard_size
        local_weight = loaded_weight.narrow(
            self.tp_dim,
            start,
            shard_size,
        )
        param.data.copy_(local_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._linear_forward(x, self.bias)


class HeadPairedColumnParallelLinear(ColumnParallelLinear):

    def __init__(
        self,
        input_size: int,
        num_heads: int,
        head_dim: int,
        bias: bool = False,
    ):
        if (
            isinstance(num_heads, bool)
            or not isinstance(num_heads, int)
            or num_heads <= 0
        ):
            raise ValueError("num_heads must be a positive integer")
        if (
            isinstance(head_dim, bool)
            or not isinstance(head_dim, int)
            or head_dim <= 0
        ):
            raise ValueError("head_dim must be a positive integer")
        tp_size = dist.get_world_size()
        if num_heads % tp_size != 0:
            raise ValueError(
                f"num_heads {num_heads} must be divisible by "
                f"tensor parallel size {tp_size}"
            )
        self.num_heads = num_heads
        self.local_num_heads = num_heads // tp_size
        self.head_dim = head_dim
        super().__init__(
            input_size,
            num_heads * 2 * head_dim,
            bias=bias,
        )

    def split_query_gate(
        self,
        projected: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if projected.ndim != 2:
            raise ValueError("projected tensor must be rank two")
        if not projected.is_floating_point():
            raise ValueError("projected tensor must use a floating point dtype")
        expected_width = self.local_num_heads * 2 * self.head_dim
        if projected.shape[1] != expected_width:
            raise ValueError(
                f"projected feature dimension must equal {expected_width}"
            )
        paired = projected.view(
            projected.shape[0],
            self.local_num_heads,
            2 * self.head_dim,
        )
        query, gate = paired.chunk(2, dim=-1)
        return query, gate.reshape(projected.shape[0], -1)


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


class ReplicatedMergedColumnParallelLinear(ReplicatedLinear):

    def __init__(
        self,
        input_size: int,
        output_sizes: list[int],
        bias: bool = False,
    ):
        self.output_sizes = output_sizes
        super().__init__(input_size, sum(output_sizes), bias=bias)
        self.requires_unpartitioned_linear_execution = True

    def weight_loader(
        self,
        param: nn.Parameter,
        loaded_weight: torch.Tensor,
        loaded_shared_id: int,
    ):
        if (
            isinstance(loaded_shared_id, bool)
            or not isinstance(loaded_shared_id, int)
            or loaded_shared_id < 0
            or loaded_shared_id >= len(self.output_sizes)
        ):
            raise ValueError("loaded_shared_id is out of range")
        size = self.output_sizes[loaded_shared_id]
        if tuple(loaded_weight.shape) != (size, self.input_size):
            raise ValueError("loaded segment shape is invalid")
        offset = sum(self.output_sizes[:loaded_shared_id])
        param.data.narrow(0, offset, size).copy_(loaded_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = []
        offset = 0
        for size in self.output_sizes:
            weight = self.weight.narrow(0, offset, size)
            bias = (
                None
                if self.bias is None
                else self.bias.narrow(0, offset, size)
            )
            with profile_operation(
                "gemm",
                f"{type(self).__name__}.linear",
                tensor=x,
            ):
                outputs.append(F.linear(x, weight, bias))
            offset += size
        return torch.cat(outputs, dim=-1)


class SegmentedColumnParallelLinear(ColumnParallelLinear):

    def __init__(
        self,
        input_size: int,
        output_sizes: list[int] | tuple[int, ...],
        bias: bool = False,
    ):
        if not output_sizes:
            raise ValueError("output_sizes must contain at least one segment")
        validated_sizes = []
        tp_size = dist.get_world_size()
        for segment_index, output_size in enumerate(output_sizes):
            if (
                isinstance(output_size, bool)
                or not isinstance(output_size, int)
                or output_size <= 0
            ):
                raise ValueError(
                    f"output_sizes[{segment_index}] must be a positive integer"
                )
            if output_size % tp_size != 0:
                raise ValueError(
                    f"output_sizes[{segment_index}]={output_size} must be "
                    f"divisible by tensor parallel size {tp_size}"
                )
            validated_sizes.append(output_size)
        self.output_sizes = tuple(validated_sizes)
        self.local_output_sizes = tuple(
            output_size // tp_size for output_size in self.output_sizes
        )
        super().__init__(input_size, sum(self.output_sizes), bias=bias)

    def _validate_source(
        self,
        param: nn.Parameter,
        loaded_weight: torch.Tensor,
    ):
        if param.ndim not in (1, 2):
            raise ValueError("segmented parameters must be rank 1 or rank 2")
        if loaded_weight.ndim != param.ndim:
            raise ValueError(
                "loaded tensor rank must match segmented parameter rank"
            )
        if param.ndim == 2 and loaded_weight.shape[1] != self.input_size:
            raise ValueError(
                f"loaded input dimension {loaded_weight.shape[1]} must match "
                f"input_size {self.input_size}"
            )
        if loaded_weight.dtype != param.dtype:
            raise ValueError(
                "loaded tensor dtype must match segmented parameter dtype"
            )
        if loaded_weight.device != param.device:
            raise ValueError(
                "loaded tensor device must match segmented parameter device"
            )

    def weight_loader(
        self,
        param: nn.Parameter,
        loaded_weight: torch.Tensor,
        loaded_segment_id: int | None = None,
    ):
        self._validate_source(param, loaded_weight)
        expected_local_output = sum(self.local_output_sizes)
        if param.shape[0] != expected_local_output:
            raise ValueError(
                f"local parameter output dimension {param.shape[0]} "
                f"must match {expected_local_output}"
            )

        if loaded_segment_id is not None:
            if (
                isinstance(loaded_segment_id, bool)
                or not isinstance(loaded_segment_id, int)
                or loaded_segment_id < 0
                or loaded_segment_id >= len(self.output_sizes)
            ):
                raise ValueError("loaded_segment_id must select a valid segment")
            global_size = self.output_sizes[loaded_segment_id]
            local_size = self.local_output_sizes[loaded_segment_id]
            if loaded_weight.shape[0] != global_size:
                raise ValueError(
                    f"segment {loaded_segment_id} loaded output dimension "
                    f"{loaded_weight.shape[0]} must match {global_size}"
                )
            destination_offset = sum(
                self.local_output_sizes[:loaded_segment_id]
            )
            source_shard = loaded_weight.narrow(
                0, self.tp_rank * local_size, local_size
            )
            destination = param.data.narrow(
                0, destination_offset, local_size
            )
            destination.copy_(source_shard)
            return

        expected_global_output = sum(self.output_sizes)
        if loaded_weight.shape[0] != expected_global_output:
            raise ValueError(
                f"fused loaded output dimension {loaded_weight.shape[0]} "
                f"must match {expected_global_output}"
            )

        source_shards = []
        global_offset = 0
        for global_size, local_size in zip(
            self.output_sizes, self.local_output_sizes
        ):
            segment = loaded_weight.narrow(0, global_offset, global_size)
            source_shards.append(
                segment.narrow(0, self.tp_rank * local_size, local_size)
            )
            global_offset += global_size
        local_weight = torch.cat(source_shards, dim=0)
        param.data.copy_(local_weight)



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
        # 注意：传给 super().__init__ 的是 total（未切分）output_size，
        # ColumnParallelLinear 会自己按 tp_size 切。如果这里传切完的（self.num_heads * ...），
        # ColumnParallel 会再切一次，导致 param 大小只剩 1/tp_size²，weight_loader narrow 越界。
        output_size = (self.total_num_heads + 2 * self.total_num_kv_heads) * head_size
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
        bias: bool = False,
        accumulation_dtype: torch.dtype | None = None,
        preserve_dense_prefill: bool = False,
    ):
        super().__init__(input_size, output_size, 1)
        if (
            accumulation_dtype is not None
            and not torch.empty(
                (),
                dtype=accumulation_dtype,
            ).is_floating_point()
        ):
            raise ValueError(
                "accumulation_dtype must be floating point"
            )
        if not isinstance(preserve_dense_prefill, bool):
            raise ValueError(
                "preserve_dense_prefill must be a boolean"
            )
        self.accumulation_dtype = accumulation_dtype
        self.preserve_dense_prefill = preserve_dense_prefill
        self.register_buffer(
            "accumulation_weight",
            None,
            persistent=False,
        )
        self.register_buffer(
            "prefill_weight",
            None,
            persistent=False,
        )
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
        full_weight = loaded_weight
        loaded_weight = full_weight.narrow(
            self.tp_dim,
            start_idx,
            shard_size,
        )
        param_data.copy_(loaded_weight)
        if (
            param is self.weight
            and self.accumulation_dtype is not None
        ):
            self.accumulation_weight = loaded_weight.to(
                device=param.device,
                dtype=self.accumulation_dtype,
            ).contiguous()
        if (
            param is self.weight
            and self.preserve_dense_prefill
        ):
            self.prefill_weight = full_weight.to(
                device=param.device,
                dtype=param.dtype,
            ).contiguous()

    def _local_output(
        self,
        x: torch.Tensor,
        bias: torch.Tensor | None,
    ) -> torch.Tensor:
        if self.accumulation_dtype is None or self.tp_size == 1:
            return self._linear_forward_unpartitioned(x, bias)
        if self.quant_method is not None:
            raise ValueError(
                "accumulation_dtype does not support quantized weights"
            )
        if self.accumulation_weight is None:
            raise RuntimeError(
                "accumulation weight is not loaded"
            )
        with profile_operation(
            "gemm",
            f"{type(self).__name__}.linear",
            tensor=x,
        ):
            output = F.linear(
                x.to(dtype=self.accumulation_dtype),
                self.accumulation_weight,
            )
        if bias is not None:
            output = output + bias.to(
                dtype=self.accumulation_dtype
            )
        return output

    def forward_prefill(self, x: torch.Tensor) -> torch.Tensor:
        if not self.preserve_dense_prefill:
            return self.forward(x)
        if self.quant_method is not None:
            raise ValueError(
                "dense prefill preservation does not support "
                "quantized weights"
            )
        if self.prefill_weight is None:
            raise RuntimeError("prefill weight is not loaded")
        if self.tp_size > 1:
            gathered = [
                torch.empty_like(x)
                for _ in range(self.tp_size)
            ]
            profile_collective(
                "row_parallel_prefill_all_gather",
                x,
                lambda tensor: dist.all_gather(gathered, tensor),
                collective_kind="all_gather",
                process_group="tensor_parallel",
                async_mode=False,
            )
            x = torch.cat(gathered, dim=-1)
        with profile_operation(
            "gemm",
            f"{type(self).__name__}.prefill_linear",
            tensor=x,
        ):
            return F.linear(x, self.prefill_weight, self.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output_dtype = x.dtype
        rows = self.linear_execution_rows
        if x.ndim == 2 and rows > 0 and x.shape[0] > rows:
            outputs = []
            for chunk in x.split(rows, dim=0):
                bias = self.bias if self.tp_rank == 0 else None
                output = self._local_output(
                    chunk,
                    bias,
                )
                if self.tp_size > 1:
                    profile_collective(
                        "row_parallel_all_reduce",
                        output,
                        dist.all_reduce,
                        collective_kind="all_reduce",
                        process_group="tensor_parallel",
                        async_mode=False,
                    )
                if output.dtype != output_dtype:
                    output = output.to(dtype=output_dtype)
                outputs.append(output)
            return torch.cat(outputs)
        bias = self.bias if self.tp_rank == 0 else None
        y = self._local_output(x, bias)
        if self.tp_size > 1:
            profile_collective(
                "row_parallel_all_reduce",
                y,
                dist.all_reduce,
                collective_kind="all_reduce",
                process_group="tensor_parallel",
                async_mode=False,
            )
        if y.dtype != output_dtype:
            y = y.to(dtype=output_dtype)
        return y


class ReplicatedWeightRowParallelLinear(LinearBase):

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
    ):
        super().__init__(input_size, output_size, 1)
        self.input_size_per_partition = divide(
            input_size,
            self.tp_size,
        )
        self.weight = nn.Parameter(
            torch.empty(self.output_size, self.input_size)
        )
        self.weight.weight_loader = self.weight_loader
        if bias:
            self.bias = nn.Parameter(torch.empty(self.output_size))
            self.bias.weight_loader = self.weight_loader
        else:
            self.register_parameter("bias", None)

    def weight_loader(
        self,
        param: nn.Parameter,
        loaded_weight: torch.Tensor,
    ):
        param.data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.tp_size > 1:
            gathered = [
                torch.empty_like(x)
                for _ in range(self.tp_size)
            ]
            profile_collective(
                "replicated_weight_row_parallel_all_gather",
                x,
                lambda tensor: dist.all_gather(gathered, tensor),
                collective_kind="all_gather",
                process_group="tensor_parallel",
                async_mode=False,
            )
            x = torch.cat(gathered, dim=-1)
        return self._linear_forward_unpartitioned(
            x.unsqueeze(0),
            self.bias,
        ).squeeze(0)


def configure_linear_execution_rows(
    model: nn.Module,
    rows: int,
) -> nn.Module:
    if (
        isinstance(rows, bool)
        or not isinstance(rows, int)
        or rows <= 0
    ):
        raise ValueError("rows must be a positive integer")
    for module in model.modules():
        if isinstance(module, LinearBase):
            module.linear_execution_rows = (
                0
                if getattr(
                    module,
                    "requires_unpartitioned_linear_execution",
                    False,
                )
                else rows
            )
    return model
