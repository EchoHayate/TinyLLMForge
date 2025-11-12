import torch
import pickle

import torch.distributed as dist
from tinyvllm.config import Config
from tinyvllm.engine.sequence import Sequence
from tinyvllm.models.qwen3 import Qwen3ForCausalLM
from tinyvllm.utils.loader import load_model
from tinyvllm.layers.sampler import Sampler
from tinyvllm.utils.context import reset_context, set_context, get_context

from multiprocessing.synchronize import Event
from multiprocessing.shared_memory import SharedMemory

class ModelRunner:

    def __init__(self, config: Config, rank: int, event: Event | list[Event]):
        self.config = config
        hf_config = config.hf_config
        self.block_size = config.kvcache_block_size
        self.enforce_eager = config.enforce_eager
        self.world_size  = config.tensor_parallel_size
        self.rank = rank
        self.event = event

        dist.init_process_group(
            backend="nccl", 
            init_method="tcp://localhost:2333",               # 初始化建立连接的方法有 tcp, 共享文件系统，环境变量等
            world_size=self.world_size, 
            rank=self.rank
        )
        torch.cuda.set_device(rank)
        default_dtype = torch.get_default_dtype()
        torch.set_default_dtype(hf_config.torch_dtype)
        torch.set_default_device("cuda")
        self.model = Qwen3ForCausalLM(hf_config)        #这里会自动触发Module中的__call__
        load_model(self.model, config.model)            #涉及到一些qwen里面的
        self.sampler =  Sampler()
        
        self.warmup_model()                             #暂时跳过

        self.allocate_kv_cache()                        #预分配空间（没有具体值）
        if not self.enforce_eager:
            self.capture_cudagraph()                    #暂时跳过
        torch.set_default_device("cpu")
        torch.set_default_dtype(default_dtype)

        if self.world_size > 1:
            if rank == 0:
                # 创建一个多卡通信的共享块
                self.shm = SharedMemory(
                    name="tinyvllm",            # 块名，供查询
                    create=True,                # 连接已有的块名还是重新创建
                    size=2**20                  # 大小
                )
                dist.barrier()                  #多进程同步屏障 让所有参与分布式训练的进程（通过 world_size 定义）都在这个代码位置等待，直到所有进程都执行到此处，才会继续往下运行。
            else:
                dist.barrier()
                self.shm = SharedMemory(name="tinyvllm")
                self.loop()

    def exit(self):
        if self.world_size > 1:
            self.shm.close()                   # 关闭所有rank和共享内存的连接
            dist.barrier()              
            if self.rank == 0:
                self.shm.unlink()              # 删除共享内存对象
        if not self.enforce_eager:
            del self.graphs, self.graph_pool
        torch.cuda.synchronize()
        dist.destroy_process_group()

    def loop(self):         #在收到exit命令之前 子进程持续执行method_name方法
        while True:
            method_name, args = self.read_shm()
            self.call(method_name, *args)
            if method_name == "exit":
                break

    def read_shm(self):
        # 多进程环境下 避免主进程调用
        assert self.world_size > 1 and self.rank        
        self.event.wait()                               # 等待主进程信号 一直等待，直到 event被set()后才会往下执行
        n = int.from_bytes(
            self.shm.buf[0:4],                          # 这里的单位是 byte，一个字节，或者说一个char
            "little")
        method_name, *args = pickle.loads(self.shm.buf[4:n+4])
        self.event.clear()                              # 重置事件标志，方便下一次等待
        return method_name, args

    # 主进程    
    def write_shm(self, method_name, *args):
        assert self.world_size > 1 and not self.rank        #not self.rank表示self.rank == 0
        data = pickle.dumps([method_name, *args])
        n = len(data)
        self.shm.buf[0:4] = n.to_bytes(4, "little")     #把数据长度写入共享内存的前4字节（用小端序存储）
        self.shm.buf[4:n+4] = data
        for event in self.event:
            event.set()
    
    def call(self, method_name, *args):         #动态方法调用 提供一个通用接口 把主进程调用的函数推给从进程
        if self.world_size > 1 and self.rank == 0:
            # 主进程调用的函数会被写入共享块中，供从进程调用, 这样可以自动实现 主进程调用 -> 从进程调用
            self.write_shm(method_name, args)
        method = getattr(self, method_name, None)       #获取函数对象
        return method(*args)            #执行函数并返回结果

    def warmup_model(self): 
        torch.cuda.empty_cache()                                #[thinking]可以看一下源码的执行策略 可能会有优化的点  
        torch.cuda.reset_peak_memory_stats()                    # 从新统计GPU内存使用的峰值信息
        max_num_batched_tokens, max_model_len = self.config.max_num_batched_tokens, self.config.max_model_len       #[16384, 4096]
        # num_seqs即batch_size   
        num_seqs = min(max_num_batched_tokens // max_model_len, self.config.max_num_seqs)   #min(4,512) 假设每个seq都占满的情况下 batch最大只能有4个seq  这里属于边界条件
        seqs = [Sequence([0] * max_model_len) for _ in range(num_seqs)] #这里warmup是按照极限的边界情况执行的
        self.run(seqs, True) 
        torch.cuda.empty_cache() 

    def allocate_kv_cache(self):
        config = self.config
        hf_config = config.hf_config
        free, total = torch.cuda.mem_get_info()
        used = total - free
        peak = torch.cuda.memory_stats()["allocated_bytes.all.peak"]
        current = torch.cuda.memory_stats()["allocated_bytes.all.current"]
        num_kv_heads = hf_config.num_key_value_heads // self.world_size
        block_bytes = 2 * hf_config.num_hidden_layers * self.block_size * num_kv_heads * \
            hf_config.head_dim * hf_config.torch_dtype.itemsize             
        # 2: key 和 value各占一块   num_hidden_layers:attention总层数     block_size：单个缓存块能存储的 token 数量
        # num_kv_heads：键值注意力头的数量  head_dim：每个注意力头的维度    torch_dtype.itemsize：单个数据元素的字节数
        config.num_kvcache_blocks = int(total * config.gpu_memory_utilization - used - (peak - current)) // block_bytes
        assert config.num_kvcache_blocks > 0
        self.kv_cache = torch.zeros(2, hf_config.num_hidden_layers, 
                config.num_kvcache_blocks, self.block_size, num_kv_heads, hf_config.head_dim)
        layer_id = 0
        for module in self.model.modules():
            if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
                module.k_cache = self.kv_cache[0, layer_id]
                module.v_cache = self.kv_cache[1, layer_id]
                layer_id += 1
        # 假设 block_size=256（每个块存 256 个 token），其他参数不变：

        # 32 层（num_hidden_layers=32）；
        # 8 个 KV 头（num_kv_heads=8）；
        # 每个头 64 维（head_dim=64）；
        # Key+Value 共 2 组（2）。

        # 对于 1 个 token，它的 KV 数据总元素数是：
        # 2（K+V） × 32（层） × 8（头） × 64（维度） = 32768 个元素。

        # 而 1 个缓存块能存 256 个 token，因此这个块的总元素数是：
        # 256（token数） × 32768（每个token的元素数） = 8388608 个元素

    
    # 每个序列（seq）的block_table是一个列表，记录该序列在 KV Cache 中使用的块编号。
    def prepare_block_tables(self, seqs: list[Sequence]):
        max_len = max(len(seq.block_table) for seq in seqs)
        block_tables = [seq.block_table + [-1] * (max_len - len(seq.block_table)) for seq in seqs]  #用-1补齐
        block_tables = torch.tensor(block_tables, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        return block_tables



# 收集新token输入（input_ids/positions） → 划分多序列边界（cu_seqlens） → 适配内存需求（max_seqlen） → 管理缓存块（block_tables） → 映射块内槽位（slot_mapping） → 将所有数据送GPU并设置上下文
    def prepare_prefill(self, seqs: list[Sequence]):        #输入数据收集、序列边界划分、缓存映射、内存适配
        input_ids = []          # 记录每个 seq 的 所有输入token id，一维[] 
        positions = []          # 记录每个 seq中 输入的 token的位置，一维[]
        cu_seqlens_q = [0]       # 以前缀和的形式，记录每个seq的长度，如 [0, 3, 5] 表示有两个seq, 一个长度为 3 = 3 - 0， 另一个长度为 2 = 5-3
        cu_seqlens_k = [0]       
        max_seqlen_q = 0        # 记录seqs(去掉缓存后)的最大长度，标量
        max_seqlen_k = 0        # 记录seqs(包含缓存长度)的最大长度
        slot_mapping = []       # 记录所有seqs每个block中的token_id 在kvcache中的位置，[token_id1, token_id2, ...token_id]
        block_tables = None     # 有前缀和的时候，才会初始化该块表
        for seq in seqs:
            seq_len = len(seq)
            input_ids.extend(seq[seq.num_cached_tokens:])       #从已有的cache开始计数
            positions.extend(list(range(seq.num_cached_tokens, seq_len)))   
            seqlen_q = seq_len - seq.num_cached_tokens
            seqlen_k = seq_len
            #前缀和 累计长度，用于区分不同的序列
            cu_seqlens_q.append(cu_seqlens_q[-1] + seqlen_q)
            cu_seqlens_k.append(cu_seqlens_k[-1] + seqlen_k)
            max_seqlen_q = max(max_seqlen_q, seqlen_q)
            max_seqlen_k = max(max_seqlen_k, seqlen_k)
            if not seq.block_table:
                continue
            
            for i in range(seq.num_cached_blocks, seq.num_blocks):
                start = seq.block_table[i] * self.block_size
                if i != seq.num_blocks - 1:
                    end = start + seq.block_size
                else:
                    end = start + seq.last_block_num_tokens
                slot_mapping.extend(list(range(start, end)))
        
        if cu_seqlens_k[-1] > cu_seqlens_q[-1]:      # 正常情况下二者是相等的，大于则说明有前缀缓存, 因此取出seq中的block_table, 拼成 blocktables表
            block_tables = self.prepare_block_tables(seqs)
        
        # 将准备好的数据传输到GPU上
        input_ids = torch.tensor(data=input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(data=positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_q = torch.tensor(data=cu_seqlens_q, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_k = torch.tensor(data=cu_seqlens_k, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(data=slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        set_context(True, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, slot_mapping, None, block_tables)
        return input_ids, positions



    # decode阶段单token输出
    def prepare_decode(self, seqs: list[Sequence]):         #暂时跳过
        input_ids = []
        positions = []
        slot_mapping = []
        context_lens = []
        for seq in seqs:
            # 上一次输出的最后token
            input_ids.append(seq.last_token)
            # 下一个token的位置
            positions.append(len(seq))
            context_lens.append(len(seq))
            slot_mapping.append(seq.block_table[-1] * seq.block_size + seq.last_block_num_tokens - 1)   #
        input_ids = torch.tensor(data=input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(data=positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(data=slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        context_lens = torch.tensor(data=context_lens, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        block_tables = self.prepare_block_tables(seqs)
        set_context(False, slot_mapping=slot_mapping, context_lens=context_lens, block_tables=block_tables)
        return input_ids, positions

    # 生成 temperatures列表，并传到GPU上
    def prepare_sample(self, seqs: list[Sequence]):
        temperatures = []
        for seq in seqs:
            temperatures.append(seq.temperature)
        temperatures = torch.tensor(data=temperatures, dtype=torch.float32, pin_memory=True).cuda(non_blocking=True)    #pin_memory=True将张量存储在锁定内存（page-locked memory）中，而非普通的可分页内存
        return temperatures
        #普通可分页内存（Pageable Memory）  |	锁定内存（Page-locked Memory / Pinned Memory）
        #操作系统可将其 “分页” 到磁盘       |     被 “锁定” 在物理内存中，不允许换出到磁盘,
        # （swap） ，释放物理内存给其他进程 |     

    @torch.inference_mode()
    #只需要前向传播 禁用梯度计算（无需反向传播），节省内存；
    # 加速推理过程（跳过与训练相关的检查和操作）。
    def run_model(self, input_ids: torch.Tensor, positions: torch.Tensor, is_prefill: bool):
        if is_prefill or self.enforce_eager or input_ids.size(0) > 512:     #动态执行 eager mode    input_ids.size(0) > 512：大批量输入的形状不固定，预编译静态图的收益有限，动态执行更灵活。
            return self.model.compute_logits(self.model(input_ids, positions))
        else:           #静态执行  graph replay
            bs = input_ids.size(0)
            context = get_context()
            graph = self.graphs[next (x for x in self.graph_bs if x >= bs)]
            graph_vars = self.graph_vars
            for k, v in graph_vars.items():
                if k != "outputs":
                    v.zero_()
            graph_vars["input_ids"][:bs] = input_ids
            graph_vars["positions"][:bs] = positions
            graph_vars["slot_mapping"][:bs] = context.slot_mapping
            graph_vars["context_lens"][:bs] = context.context_lens
            graph_vars["block_tables"][:bs, :context.block_tables.size(1)] = context.block_tables
            graph.replay()
            return self.model.compute_logits(graph_vars["outputs"][:bs])


    def run(self, seqs:list[Sequence], is_prefill: bool) -> list[int]:
        input_ids, positions = self.prepare_prefill(seqs) if is_prefill else self.prepare_decode(seqs)
        temperatures = self.prepare_sample(seqs) if self.rank == 0 else None    #只有主进程做采样
        logits = self.run_model(input_ids, positions, is_prefill)
        token_ids = self.sampler(logits, temperatures).tolist() if self.rank == 0 else None
        reset_context()
        return token_ids

    @torch.inference_mode()
    def capture_cudagraph(self):
        config = self.config
        hf_config = config.hf_config
        max_bs = min(self.config.max_num_seqs, 512)        # 这里的 max_batch_size默认了seq_len = 1, 因此 batch_size * seq_len = max_bs
        max_num_blocks = (config.max_model_len + self.block_size - 1) // self.block_size
        input_ids = torch.zeros(max_bs, dtype=torch.int64)
        positions = torch.zeros(max_bs, dtype=torch.int64)
        slot_mapping = torch.zeros(max_bs, dtype=torch.int32)
        context_lens = torch.zeros(max_bs, dtype=torch.int32)
        block_tables = torch.zeros(max_bs, max_num_blocks, dtype=torch.int32)
        outputs = torch.zeros(max_bs, hf_config.hidden_size)
        self.graph_bs = [1, 2, 4, 8] + list(range(16, max_bs + 1, 16))      # 捕捉各种batch_size的cuda graph
        self.graphs = {}
        self.graph_pool = None

        # decode 阶段
        for bs in reversed(self.graph_bs):
            graph = torch.cuda.CUDAGraph()
            set_context(False, slot_mapping=slot_mapping[:bs], context_lens=context_lens[:bs], block_tables=block_tables[:bs, :])
            outputs[:bs] = self.model(input_ids[:bs], positions[:bs])       # warm up
            with torch.cuda.graph(graph, self.graph_pool):                  # 开始 capture
                outputs[:bs] = self.model(input_ids[:bs], positions[:bs])
            if self.graph_pool is None:
                self.graph_pool = graph.pool()
            self.graphs[bs] = graph
            torch.cuda.synchronize()
            reset_context()

        self.graph_vars = dict(
            input_ids=input_ids, 
            positions=positions,
            slot_mapping=slot_mapping,
            context_lens=context_lens,
            block_tables=block_tables,
            outputs=outputs
        )

