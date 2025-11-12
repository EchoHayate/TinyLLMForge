import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist

def divide(numerator, denominator):
    assert numerator % denominator == 0
    return numerator // denominator

class LinearBase(nn.Module):
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
        return F.linear(x, self.weight, self.bias)

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
        return F.linear(x, self.weight, self.bias)
    

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
        y = F.linear(x, self.weight, self.bias if self.tp_rank == 0 else None)      #简单的矩阵乘
        if self.tp_size > 1:
            dist.all_reduce(y)
        return y