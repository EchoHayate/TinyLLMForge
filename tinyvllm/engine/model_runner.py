from __future__ import annotations

import torch
import pickle
import os
import time

import torch.distributed as dist
from tinyvllm.config import Config
from tinyvllm.engine.sequence import Sequence
from tinyvllm.models.qwen3 import Qwen3ForCausalLM
from tinyvllm.utils.loader import load_model
from tinyvllm.utils.cpu_offload import apply_cpu_offload
from tinyvllm.layers.linear import set_quant_config
from tinyvllm.layers.sampler import Sampler
from tinyvllm.utils.context import reset_context, set_context, get_context
from tinyvllm.engine.kv_cartridge import compress_decode_block_table_rows, should_use_kv_cartridge

from multiprocessing.synchronize import Event
from multiprocessing.shared_memory import SharedMemory


class KVOffloadMVP0:
    """Minimal logical-block -> GPU-slot KV offload prototype.

    Scope is intentionally narrow: fp16/bf16 KV cache, full attention, eager
    execution. ``Sequence.block_table`` stays logical; this manager owns the
    logical->physical slot mapping and a CPU pinned backing store.
    """

    def __init__(
        self,
        kv_cache: torch.Tensor,
        logical_blocks: int,
        block_size: int,
        async_copy: bool = True,
        batch_copy: bool = True,
        writeback_on_evict: bool = False,
        evict_policy: str = "lru_cost",
    ):
        self.kv_cache = kv_cache
        self.logical_blocks = int(logical_blocks)
        self.gpu_blocks = int(kv_cache.size(2))
        self.block_size = int(block_size)
        self.async_copy = bool(async_copy)
        self.batch_copy = bool(batch_copy)
        self.writeback_on_evict = bool(writeback_on_evict)
        self.evict_policy = evict_policy
        self.cpu_cache = torch.empty(
            (kv_cache.size(0), kv_cache.size(1), self.logical_blocks, *kv_cache.shape[3:]),
            dtype=kv_cache.dtype,
            device="cpu",
            pin_memory=True,
        )
        self.copy_stream = torch.cuda.Stream(device=kv_cache.device) if self.async_copy else None
        self.h2d_done: dict[int, torch.cuda.Event] = {}
        self.d2h_done: dict[int, torch.cuda.Event] = {}
        self.pending_wait_blocks: set[int] = set()
        self.cpu_valid = [False] * self.logical_blocks
        self.logical_to_slot: dict[int, int] = {}
        self.slot_to_logical: list[int | None] = [None] * self.gpu_blocks
        self.slot_last_used = [0] * self.gpu_blocks
        self.dirty_logical_blocks: set[int] = set()
        self.clock = 0
        self.stats = {
            "h2d_copies": 0,
            "d2h_copies": 0,
            "evictions": 0,
            "h2d_ms": 0.0,
            "d2h_ms": 0.0,
            "h2d_bytes": 0,
            "d2h_bytes": 0,
            "copy_waits": 0,
            "h2d_batches": 0,
            "d2h_batches": 0,
            "h2d_batch_spans": 0,
            "d2h_batch_spans": 0,
            "evict_clean": 0,
            "evict_dirty": 0,
            "prefetch_plans": 0,
            "prefetch_read_blocks": 0,
            "prefetch_write_blocks": 0,
        }
        self.block_nbytes = self.kv_cache[:, :, 0].numel() * self.kv_cache.element_size()

    def _check_logical_block(self, logical_block: int):
        if logical_block < 0 or logical_block >= self.logical_blocks:
            raise RuntimeError(
                f"KV offload logical block id {logical_block} out of range [0, {self.logical_blocks})"
            )

    def _touch(self, slot: int):
        self.clock += 1
        self.slot_last_used[slot] = self.clock

    def _coalesce_copy_pairs(self, pairs: list[tuple[int, int]]) -> list[tuple[int, int, int]]:
        if not pairs:
            return []
        if not self.batch_copy:
            return [(int(logical_block), int(slot), 1) for logical_block, slot in pairs]
        ordered = sorted((int(logical_block), int(slot)) for logical_block, slot in pairs)
        spans = []
        start_logical, start_slot = ordered[0]
        prev_logical, prev_slot = ordered[0]
        span_len = 1
        for logical_block, slot in ordered[1:]:
            if logical_block == prev_logical + 1 and slot == prev_slot + 1:
                prev_logical = logical_block
                prev_slot = slot
                span_len += 1
                continue
            spans.append((start_logical, start_slot, span_len))
            start_logical = prev_logical = logical_block
            start_slot = prev_slot = slot
            span_len = 1
        spans.append((start_logical, start_slot, span_len))
        return spans

    def _record_copy_event(self) -> torch.cuda.Event | None:
        if self.copy_stream is None:
            return None
        event = torch.cuda.Event()
        event.record(self.copy_stream)
        return event

    def _enqueue_h2d_pairs(self, pairs: list[tuple[int, int]]):
        pairs = [(int(logical_block), int(slot)) for logical_block, slot in pairs if self.cpu_valid[int(logical_block)]]
        if not pairs:
            return
        spans = self._coalesce_copy_pairs(pairs)
        t0 = time.perf_counter()
        if self.copy_stream is None:
            for logical_start, slot_start, span_len in spans:
                self.kv_cache[:, :, slot_start:slot_start + span_len].copy_(
                    self.cpu_cache[:, :, logical_start:logical_start + span_len],
                    non_blocking=True,
                )
            torch.cuda.synchronize()
            event = None
        else:
            with torch.cuda.stream(self.copy_stream):
                for logical_start, slot_start, span_len in spans:
                    for logical_block in range(logical_start, logical_start + span_len):
                        d2h_event = self.d2h_done.get(logical_block)
                        if d2h_event is not None:
                            self.copy_stream.wait_event(d2h_event)
                    self.kv_cache[:, :, slot_start:slot_start + span_len].copy_(
                        self.cpu_cache[:, :, logical_start:logical_start + span_len],
                        non_blocking=True,
                    )
            event = self._record_copy_event()
            self.pending_wait_blocks.update(logical_block for logical_block, _ in pairs)
        for logical_block, _ in pairs:
            if event is not None:
                self.h2d_done[logical_block] = event
        self.stats["h2d_copies"] += len(pairs)
        self.stats["h2d_bytes"] += len(pairs) * self.block_nbytes
        self.stats["h2d_batches"] += 1
        self.stats["h2d_batch_spans"] += len(spans)
        self.stats["h2d_ms"] += (time.perf_counter() - t0) * 1000.0

    def _enqueue_d2h_pairs(self, pairs: list[tuple[int, int]]):
        pairs = [(int(logical_block), int(slot)) for logical_block, slot in pairs]
        if not pairs:
            return
        spans = self._coalesce_copy_pairs(pairs)
        t0 = time.perf_counter()
        if self.copy_stream is None:
            for logical_start, slot_start, span_len in spans:
                self.cpu_cache[:, :, logical_start:logical_start + span_len].copy_(
                    self.kv_cache[:, :, slot_start:slot_start + span_len],
                    non_blocking=True,
                )
            torch.cuda.synchronize()
            event = None
        else:
            self.copy_stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(self.copy_stream):
                for logical_start, slot_start, span_len in spans:
                    self.cpu_cache[:, :, logical_start:logical_start + span_len].copy_(
                        self.kv_cache[:, :, slot_start:slot_start + span_len],
                        non_blocking=True,
                    )
            event = self._record_copy_event()
        for logical_block, _ in pairs:
            self.cpu_valid[logical_block] = True
            if event is not None:
                self.d2h_done[logical_block] = event
        self.stats["d2h_copies"] += len(pairs)
        self.stats["d2h_bytes"] += len(pairs) * self.block_nbytes
        self.stats["d2h_batches"] += 1
        self.stats["d2h_batch_spans"] += len(spans)
        self.stats["d2h_ms"] += (time.perf_counter() - t0) * 1000.0

    def _copy_h2d(self, logical_block: int, slot: int):
        self._enqueue_h2d_pairs([(logical_block, slot)])

    def _copy_d2h(self, logical_block: int, slot: int):
        self._enqueue_d2h_pairs([(logical_block, slot)])

    def _victim_score(self, slot: int, future_logical_blocks: set[int]) -> float:
        logical_block = self.slot_to_logical[slot]
        if logical_block is None:
            return float("-inf")
        if self.evict_policy == "lru":
            return float(self.slot_last_used[slot])
        dirty_penalty = self.block_nbytes * 4.0 if logical_block in self.dirty_logical_blocks else 0.0
        reuse_penalty = self.block_nbytes * 8.0 if logical_block in future_logical_blocks else 0.0
        pending_h2d_penalty = self.block_nbytes * 6.0 if logical_block in self.pending_wait_blocks else 0.0
        return float(self.slot_last_used[slot]) + dirty_penalty + reuse_penalty + pending_h2d_penalty

    def _evict_slot(
        self,
        protected_logical_blocks: set[int],
        future_logical_blocks: set[int] | None = None,
        defer_dirty_writeback: bool = False,
    ) -> int:
        future_logical_blocks = future_logical_blocks or set()
        for slot, logical_block in enumerate(self.slot_to_logical):
            if logical_block is None:
                return slot

        candidates = [
            slot for slot, logical_block in enumerate(self.slot_to_logical)
            if logical_block not in protected_logical_blocks
        ]
        if not candidates:
            raise RuntimeError(
                "KV offload GPU staging slots are insufficient for one full-attention batch: "
                f"gpu_blocks={self.gpu_blocks}, required={len(protected_logical_blocks)}"
            )
        slot = min(candidates, key=lambda item: self._victim_score(item, future_logical_blocks))
        old_logical = self.slot_to_logical[slot]
        if old_logical is not None:
            if old_logical in self.dirty_logical_blocks:
                if not defer_dirty_writeback:
                    self._enqueue_d2h_pairs([(old_logical, slot)])
                    self.dirty_logical_blocks.discard(old_logical)
                self.stats["evict_dirty"] += 1
            else:
                self.stats["evict_clean"] += 1
            if not defer_dirty_writeback:
                d2h_event = self.d2h_done.get(old_logical)
                if d2h_event is not None:
                    torch.cuda.current_stream().wait_event(d2h_event)
                    if self.copy_stream is not None:
                        self.copy_stream.wait_event(d2h_event)
                    self.stats["copy_waits"] += 1
            if not defer_dirty_writeback:
                self.logical_to_slot.pop(old_logical, None)
                self.slot_to_logical[slot] = None
            self.stats["evictions"] += 1
        return slot

    def ensure_resident(
        self,
        logical_blocks: list[int],
        require_valid: bool,
        future_logical_blocks: set[int] | None = None,
        protected_logical_blocks: set[int] | None = None,
        wait: bool = False,
    ) -> dict[int, int]:
        ordered = []
        seen = set()
        for block in logical_blocks:
            block = int(block)
            if block < 0 or block in seen:
                continue
            self._check_logical_block(block)
            ordered.append(block)
            seen.add(block)
        if len(ordered) > self.gpu_blocks:
            raise RuntimeError(
                "KV offload MVP-0 uses full attention, so all visible logical blocks must fit in GPU staging: "
                f"required={len(ordered)}, gpu_blocks={self.gpu_blocks}"
            )

        protected = set(ordered)
        if protected_logical_blocks:
            protected.update(int(block) for block in protected_logical_blocks if int(block) >= 0)
        h2d_pairs = []
        deferred_d2h_pairs = []
        deferred_wait_blocks = []
        for logical_block in ordered:
            slot = self.logical_to_slot.get(logical_block)
            if slot is None:
                if require_valid and not self.cpu_valid[logical_block]:
                    raise RuntimeError(f"KV offload requested unreadable logical block {logical_block}")
                slot = self._evict_slot(
                    protected,
                    future_logical_blocks=future_logical_blocks,
                    defer_dirty_writeback=True,
                )
                old_logical = self.slot_to_logical[slot]
                if old_logical is not None and old_logical in self.dirty_logical_blocks:
                    deferred_d2h_pairs.append((old_logical, slot))
                    self.dirty_logical_blocks.discard(old_logical)
                if old_logical is not None:
                    deferred_wait_blocks.append(old_logical)
                    self.logical_to_slot.pop(old_logical, None)
                    self.slot_to_logical[slot] = None
                self.logical_to_slot[logical_block] = slot
                self.slot_to_logical[slot] = logical_block
                if self.cpu_valid[logical_block]:
                    h2d_pairs.append((logical_block, slot))
            self._touch(slot)
        self._enqueue_d2h_pairs(deferred_d2h_pairs)
        waited_d2h_event_ids = set()
        for old_logical in deferred_wait_blocks:
            d2h_event = self.d2h_done.get(old_logical)
            if d2h_event is not None and id(d2h_event) not in waited_d2h_event_ids:
                torch.cuda.current_stream().wait_event(d2h_event)
                if self.copy_stream is not None:
                    self.copy_stream.wait_event(d2h_event)
                waited_d2h_event_ids.add(id(d2h_event))
                self.stats["copy_waits"] += 1
        self._enqueue_h2d_pairs(h2d_pairs)
        if wait:
            waited_h2d_blocks = [logical_block for logical_block, _ in h2d_pairs]
            self.wait_for_blocks(waited_h2d_blocks)
            self.pending_wait_blocks.difference_update(int(block) for block in waited_h2d_blocks)
        return {logical_block: self.logical_to_slot[logical_block] for logical_block in ordered}

    def wait_for_blocks(self, logical_blocks: list[int], clear_pending: bool = False):
        if self.copy_stream is None:
            return
        blocks = set(int(block) for block in logical_blocks)
        stream = torch.cuda.current_stream()
        waited_event_ids = set()
        for logical_block in blocks:
            event = self.h2d_done.get(logical_block)
            if event is not None and id(event) not in waited_event_ids:
                stream.wait_event(event)
                waited_event_ids.add(id(event))
                self.stats["copy_waits"] += 1
        if clear_pending:
            self.pending_wait_blocks.difference_update(blocks)

    def wait_for_pending(self):
        if not self.pending_wait_blocks:
            return
        self.wait_for_blocks(list(self.pending_wait_blocks), clear_pending=True)
        self.pending_wait_blocks.clear()

    def synchronize_copies(self):
        if self.copy_stream is not None:
            self.copy_stream.synchronize()

    def translate_block_rows(
        self,
        rows: list[list[int]],
        require_valid: bool = True,
        future_logical_blocks: set[int] | None = None,
    ) -> list[list[int]]:
        logical_blocks = [int(block) for row in rows for block in row if int(block) >= 0]
        mapping = self.ensure_resident(
            logical_blocks,
            require_valid=require_valid,
            future_logical_blocks=future_logical_blocks,
        )
        return [[mapping[int(block)] if int(block) >= 0 else -1 for block in row] for row in rows]

    def translate_slots_for_positions(
        self,
        logical_block_table: list[int],
        positions: list[int],
        require_valid: bool = False,
        future_logical_blocks: set[int] | None = None,
    ) -> list[int]:
        logical_blocks = [int(logical_block_table[pos // self.block_size]) for pos in positions]
        mapping = self.ensure_resident(
            logical_blocks,
            require_valid=require_valid,
            future_logical_blocks=future_logical_blocks,
        )
        return [
            mapping[int(logical_block_table[pos // self.block_size])] * self.block_size + (pos % self.block_size)
            for pos in positions
        ]

    def mark_dirty(self, logical_blocks: list[int]):
        for logical_block in logical_blocks:
            logical_block = int(logical_block)
            if logical_block in self.logical_to_slot:
                self.dirty_logical_blocks.add(logical_block)

    def writeback_dirty(self, logical_blocks: list[int] | None = None):
        targets = set(self.dirty_logical_blocks if logical_blocks is None else logical_blocks)
        d2h_pairs = []
        for logical_block in list(targets):
            slot = self.logical_to_slot.get(int(logical_block))
            if slot is None:
                continue
            d2h_pairs.append((int(logical_block), slot))
            self.dirty_logical_blocks.discard(int(logical_block))
        self._enqueue_d2h_pairs(d2h_pairs)

    def summary(self) -> dict:
        return {
            **self.stats,
            "logical_blocks": self.logical_blocks,
            "gpu_blocks": self.gpu_blocks,
            "resident_blocks": len(self.logical_to_slot),
            "dirty_blocks": len(self.dirty_logical_blocks),
            "block_nbytes": self.block_nbytes,
            "async_copy": self.async_copy,
            "batch_copy": self.batch_copy,
            "writeback_on_evict": self.writeback_on_evict,
            "evict_policy": self.evict_policy,
        }

class ModelRunner:

    def __init__(self, config: Config, rank: int, event: Event | list[Event]):
        self.config = config
        hf_config = config.hf_config
        self.block_size = config.kvcache_block_size
        self.enforce_eager = config.enforce_eager
        self.world_size  = config.tensor_parallel_size
        self.rank = rank
        self.event = event

        dist_port = os.environ.get("TINYVLLM_DIST_PORT", os.environ.get("MASTER_PORT", "2333"))
        dist.init_process_group(
            backend="nccl", 
            init_method=f"tcp://localhost:{dist_port}",       # 初始化建立连接的方法有 tcp, 共享文件系统，环境变量等
            world_size=self.world_size, 
            rank=self.rank
        )
        torch.cuda.set_device(rank)
        default_dtype = torch.get_default_dtype()
        torch.set_default_dtype(hf_config.torch_dtype)
        torch.set_default_device("cuda")
        # 注入全局量化配置（在构建模型前）
        set_quant_config(config.quantization, config.quant_group_size, config.act_quant_bits)
        self.model = Qwen3ForCausalLM(hf_config)        #这里会自动触发Module中的__call__
        load_model(self.model, config.model,
                   smoothquant_scale_path=config.smoothquant_scale_path,
                   act_quant_skip_first=config.act_quant_skip_first,
                   act_quant_skip_last=config.act_quant_skip_last,
                   act_quant_skip_layers=config.act_quant_skip_layers)            #涉及到一些qwen里面的
        # 加载完成后再做 cpu-offload（量化已在 loader 内 finalize 完成）
        if config.cpu_offload:
            apply_cpu_offload(self.model, config.cpu_offload_num_layers)
        self.sampler =  Sampler()

        # prepare_prefill / prepare_decode 用的 pinned host buffer 池：按 (name, dtype) 复用，
        # 容量按需向上扩；避免每步 torch.tensor(list, pin_memory=True).cuda() 触发 host alloc + pin
        self._pinned_buf_cache: dict[tuple[str, torch.dtype], torch.Tensor] = {}
        self.kv_offload: KVOffloadMVP0 | None = None
        self._kv_offload_pending_dirty_blocks: list[int] = []

        self.warmup_model()                             #暂时跳过

        self.allocate_kv_cache()                        #预分配空间（没有具体值）
        # cuda graph 跳过条件：
        #   1) enforce_eager：用户显式关
        #   2) kv_quant_bits == 4 (C4)：decode 反量化路径里有动态 alloc，无法 capture
        #   3) cpu_offload：layer 权重 H2D 走独立 stream + cross-stream sync，
        #      在 capture mode 下会报 "operation failed due to a previous error during capture"
        skip_cudagraph = (
            self.enforce_eager
            or config.kv_quant_bits == 4
            or config.am_compact_blocks > 0
            or config.cpu_offload
            or config.kv_offload_mvp0
        )
        if not skip_cudagraph:
            self.capture_cudagraph()
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
        if hasattr(self, "graphs"):
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
            # 注意：必须 *args 展开，否则 worker 端解包出来 args 会是一层多余的 tuple，
            # 导致 method(*args) 传给目标函数的实参只有一个 tuple（缺位置参数）
            self.write_shm(method_name, *args)
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
        # 记一次 weight-only 占用（KV cache 还没分），方便外部观察 TP 是否真切到了 weight
        # 这里走 memory_allocated 而不是 mem_get_info：前者只算本进程 torch alloc，后者算整卡（含别人）
        self.weight_mem_bytes = torch.cuda.memory_allocated()
        free, total = torch.cuda.mem_get_info()
        used = total - free
        peak = torch.cuda.memory_stats()["allocated_bytes.all.peak"]
        current = torch.cuda.memory_stats()["allocated_bytes.all.current"]
        num_kv_heads = hf_config.num_key_value_heads // self.world_size
        head_dim = hf_config.head_dim
        dtype = hf_config.torch_dtype
        elem_bytes = dtype.itemsize

        # ----- KV cache 主存储：根据 kv_quant_bits 决定字节数 -----
        # 0: fp/half 原样；4: 每 token 每 head_dim 半字节（按 int8 pack）；8: 每 token 每 head_dim 1 字节
        kvq_bits = config.kv_quant_bits
        if kvq_bits == 0:
            tokens_per_block_bytes = self.block_size * num_kv_heads * head_dim * elem_bytes
            kv_scale_bytes_per_block = 0
        elif kvq_bits == 4:
            assert head_dim % config.kv_quant_group_size == 0, \
                f"head_dim={head_dim} 必须能被 kv_quant_group_size={config.kv_quant_group_size} 整除"
            n_groups_per_token = head_dim // config.kv_quant_group_size
            # 4-bit pack 进 int8：每 byte 存 2 个 4-bit；group_size 偶数保证字节对齐
            packed_bytes = self.block_size * num_kv_heads * (head_dim // 2)
            tokens_per_block_bytes = packed_bytes
            # scale: fp16 / bf16 一份，对称量化只存 scale；非对称额外存 zero（同 dtype）
            scale_count = self.block_size * num_kv_heads * n_groups_per_token
            kv_scale_bytes_per_block = scale_count * elem_bytes * (1 if config.kv_quant_symmetric else 2)
        else:  # 8
            tokens_per_block_bytes = self.block_size * num_kv_heads * head_dim
            n_groups_per_token = max(1, head_dim // config.kv_quant_group_size)
            scale_count = self.block_size * num_kv_heads * n_groups_per_token
            kv_scale_bytes_per_block = scale_count * elem_bytes * (1 if config.kv_quant_symmetric else 2)

        block_bytes = 2 * hf_config.num_hidden_layers * tokens_per_block_bytes
        kv_scale_bytes_per_block = 2 * hf_config.num_hidden_layers * kv_scale_bytes_per_block

        # Quest 启用时还需要预留 per-block K min/max summary 显存（每块 2*num_kv_heads*head_dim 元素）
        quest_enabled = config.quest_top_k_blocks > 0
        summary_bytes = (2 * hf_config.num_hidden_layers * num_kv_heads *
                         head_dim * elem_bytes) if quest_enabled else 0

        per_block = block_bytes + kv_scale_bytes_per_block + summary_bytes
        auto_num_blocks = int(total * config.gpu_memory_utilization - used - (peak - current)) // per_block
        assert auto_num_blocks > 0

        if config.kv_offload_mvp0:
            gpu_nb = config.kv_offload_gpu_blocks or auto_num_blocks
            logical_nb = config.kv_offload_logical_blocks or max(auto_num_blocks, gpu_nb)
            assert gpu_nb > 0
            assert logical_nb >= gpu_nb, "kv_offload_logical_blocks 必须 >= kv_offload_gpu_blocks"
            config.num_kvcache_blocks = logical_nb
            nb = gpu_nb
        else:
            config.num_kvcache_blocks = auto_num_blocks
            nb = config.num_kvcache_blocks
        L = hf_config.num_hidden_layers
        if kvq_bits == 0:
            self.kv_cache = torch.zeros(2, L, nb, self.block_size, num_kv_heads, head_dim, dtype=dtype)
            self.kv_scale = None
            self.kv_zero = None
        elif kvq_bits == 4:
            # int8 pack 后，沿最后一维 head_dim/2
            self.kv_cache = torch.zeros(2, L, nb, self.block_size, num_kv_heads, head_dim // 2, dtype=torch.int8)
            n_groups = head_dim // config.kv_quant_group_size
            self.kv_scale = torch.zeros(2, L, nb, self.block_size, num_kv_heads, n_groups, dtype=dtype)
            self.kv_zero = (None if config.kv_quant_symmetric else
                            torch.zeros(2, L, nb, self.block_size, num_kv_heads, n_groups, dtype=dtype))
        else:  # 8
            self.kv_cache = torch.zeros(2, L, nb, self.block_size, num_kv_heads, head_dim, dtype=torch.int8)
            n_groups = max(1, head_dim // config.kv_quant_group_size)
            self.kv_scale = torch.zeros(2, L, nb, self.block_size, num_kv_heads, n_groups, dtype=dtype)
            self.kv_zero = (None if config.kv_quant_symmetric else
                            torch.zeros(2, L, nb, self.block_size, num_kv_heads, n_groups, dtype=dtype))

        # Quest summary：[2, num_layers, num_blocks, num_kv_heads, head_dim]，dim0 = (min, max)
        if quest_enabled:
            self.kv_summary = torch.empty(2, L, nb, num_kv_heads, head_dim, dtype=torch.float32)
            # 用 +inf / -inf 作为 min/max 的初始值，确保第一次 token 写入后被替换
            self.kv_summary[0].fill_(float("inf"))
            self.kv_summary[1].fill_(float("-inf"))
        else:
            self.kv_summary = None

        if config.kv_offload_mvp0:
            self.kv_offload = KVOffloadMVP0(
                self.kv_cache,
                config.num_kvcache_blocks,
                self.block_size,
                async_copy=config.kv_offload_async_copy,
                batch_copy=config.kv_offload_batch_copy,
                writeback_on_evict=config.kv_offload_writeback_on_evict,
                evict_policy=config.kv_offload_evict_policy,
            )

        layer_id = 0
        for module in self.model.modules():
            if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
                module.k_cache = self.kv_cache[0, layer_id]
                module.v_cache = self.kv_cache[1, layer_id]
                # 把量化辅助张量也挂上去；非量化时为 None
                if self.kv_scale is not None:
                    module.k_scale = self.kv_scale[0, layer_id]
                    module.v_scale = self.kv_scale[1, layer_id]
                else:
                    module.k_scale = module.v_scale = None
                if self.kv_zero is not None:
                    module.k_zero = self.kv_zero[0, layer_id]
                    module.v_zero = self.kv_zero[1, layer_id]
                else:
                    module.k_zero = module.v_zero = None
                module.kv_quant_bits = kvq_bits
                module.kv_quant_group_size = config.kv_quant_group_size
                module.kv_quant_symmetric = config.kv_quant_symmetric
                module.layer_idx = layer_id
                module.num_hidden_layers = L
                if quest_enabled:
                    module.k_min = self.kv_summary[0, layer_id]
                    module.k_max = self.kv_summary[1, layer_id]
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
        block_tables_data = [seq.block_table + [-1] * (max_len - len(seq.block_table)) for seq in seqs]  #用-1补齐
        if self.kv_offload is not None:
            block_tables_data = self.kv_offload.translate_block_rows(block_tables_data, require_valid=True)
        return self.prepare_block_tables_from_rows(block_tables_data)

    def prepare_block_tables_from_rows(self, rows: list[list[int]], name: str = "block_tables"):
        return self._list_to_cuda_2d(rows, name, torch.int32)

    def _kv_offload_translate_block_rows(
        self,
        rows: list[list[int]],
        require_valid: bool = True,
        future_logical_blocks: set[int] | None = None,
    ) -> list[list[int]]:
        if self.kv_offload is None:
            return rows
        return self.kv_offload.translate_block_rows(
            rows,
            require_valid=require_valid,
            future_logical_blocks=future_logical_blocks,
        )

    def _kv_offload_translate_slots_for_positions(
        self,
        block_table: list[int],
        positions: list[int],
        require_valid: bool = False,
        future_logical_blocks: set[int] | None = None,
    ) -> list[int]:
        if self.kv_offload is None:
            return [
                block_table[pos // self.block_size] * self.block_size + (pos % self.block_size)
                for pos in positions
            ]
        return self.kv_offload.translate_slots_for_positions(
            block_table,
            positions,
            require_valid=require_valid,
            future_logical_blocks=future_logical_blocks,
        )

    def _kv_offload_mark_pending_dirty(self, block_table: list[int], positions: list[int]):
        if self.kv_offload is None:
            return
        for pos in positions:
            self._kv_offload_pending_dirty_blocks.append(block_table[pos // self.block_size])

    def _kv_offload_after_forward(self):
        if self.kv_offload is None:
            return
        if self._kv_offload_pending_dirty_blocks:
            self.kv_offload.mark_dirty(self._kv_offload_pending_dirty_blocks)
            if not self.kv_offload.writeback_on_evict:
                self.kv_offload.writeback_dirty(self._kv_offload_pending_dirty_blocks)
            self._kv_offload_pending_dirty_blocks = []

    def _kv_offload_before_forward(self):
        if self.kv_offload is None:
            return
        self.kv_offload.wait_for_pending()

    def kv_offload_summary(self) -> dict | None:
        if self.kv_offload is None:
            return None
        return self.kv_offload.summary()

    # ---- pinned host buffer 池：把多次小 H2D 改成 buffer 复用 + non_blocking copy ----
    def _get_pinned(self, name: str, n: int, dtype: torch.dtype) -> torch.Tensor:
        """按 (name, dtype) 拿一个长度 ≥ n 的 1D pinned host tensor；按需翻倍扩容并复用。"""
        key = (name, dtype)
        buf = self._pinned_buf_cache.get(key)
        if buf is None or buf.numel() < n:
            new_size = max(n, (buf.numel() * 2) if buf is not None else max(n, 64))
            # 显式 device="cpu" + pin_memory=True，避开 default_device 被设成 cuda 的情况
            buf = torch.empty(new_size, dtype=dtype, device="cpu", pin_memory=True)
            self._pinned_buf_cache[key] = buf
        return buf

    def _list_to_cuda(self, data: list, name: str, dtype: torch.dtype) -> torch.Tensor:
        """把 python list 写入 pinned host buffer 后 non_blocking H2D。返回 GPU tensor。"""
        n = len(data)
        host = self._get_pinned(name, n, dtype)
        host[:n].copy_(torch.tensor(data, dtype=dtype, device="cpu"))
        return host[:n].cuda(non_blocking=True)

    def _list_to_cuda_2d(self, data: list[list[int]], name: str, dtype: torch.dtype) -> torch.Tensor:
        """2D list（每行长度相同）打成 pinned host 矩阵后 non_blocking H2D。"""
        rows = len(data)
        cols = len(data[0]) if rows else 0
        n = rows * cols
        host = self._get_pinned(name, n, dtype)
        host[:n].copy_(torch.tensor(data, dtype=dtype, device="cpu").flatten())
        return host[:n].view(rows, cols).cuda(non_blocking=True)



# 收集新token输入（input_ids/positions） → 划分多序列边界（cu_seqlens） → 适配内存需求（max_seqlen） → 管理缓存块（block_tables） → 映射块内槽位（slot_mapping） → 将所有数据送GPU并设置上下文
    def prepare_prefill(self, seqs: list[Sequence]):        #输入数据收集、序列边界划分、缓存映射、内存适配
        self._kv_offload_pending_dirty_blocks = []
        input_ids = []          # 记录每个 seq 的 所有输入token id，一维[] 
        positions = []          # 记录每个 seq中 输入的 token的位置，一维[]
        cu_seqlens_q = [0]       # 以前缀和的形式，记录每个seq的长度，如 [0, 3, 5] 表示有两个seq, 一个长度为 3 = 3 - 0， 另一个长度为 2 = 5-3
        cu_seqlens_k = [0]       
        max_seqlen_q = 0        # 记录seqs(去掉缓存后)的最大长度，标量
        max_seqlen_k = 0        # 记录seqs(包含缓存长度)的最大长度
        slot_mapping = []       # 记录所有seqs每个block中的token_id 在kvcache中的位置，[token_id1, token_id2, ...token_id]
        block_tables = None     # 有前缀和的时候，才会初始化该块表
        prefill_chunk_starts = []
        prefill_chunk_ends = []
        logical_block_table_rows = []
        kv_offload_write_blocks = []
        blockwise_prefill_enabled = self.kv_offload is not None and self.config.kv_offload_blockwise_prefill
        for seq in seqs:
            seq_len = len(seq)
            chunk_start = getattr(seq, "prefill_chunk_start", seq.num_cached_tokens)
            chunk_end = getattr(seq, "prefill_chunk_end", seq_len)
            if chunk_end == 0 and chunk_start == 0:
                # warmup_model() calls ModelRunner.run() directly with fresh Sequence
                # objects, bypassing Scheduler's chunk boundary initialization.
                chunk_end = seq_len
            prefill_chunk_starts.append(int(chunk_start))
            prefill_chunk_ends.append(int(chunk_end))
            input_ids.extend(seq[chunk_start:chunk_end])       #从已有的cache/chunk进度开始计数
            positions.extend(list(range(chunk_start, chunk_end)))   
            seqlen_q = chunk_end - chunk_start
            seqlen_k = chunk_end
            #前缀和 累计长度，用于区分不同的序列
            cu_seqlens_q.append(cu_seqlens_q[-1] + seqlen_q)
            cu_seqlens_k.append(cu_seqlens_k[-1] + seqlen_k)
            max_seqlen_q = max(max_seqlen_q, seqlen_q)
            max_seqlen_k = max(max_seqlen_k, seqlen_k)
            if not seq.block_table:
                logical_block_table_rows.append([])
                continue
            visible_blocks = (chunk_end + self.block_size - 1) // self.block_size
            logical_block_table_rows.append(list(seq.block_table[:visible_blocks]))
            
            write_positions = list(range(chunk_start, chunk_end))
            future_blocks = set(int(block_id) for block_id in seq.block_table)
            if self.kv_offload is not None and blockwise_prefill_enabled:
                write_blocks = []
                first_write_offset_by_block = {}
                for pos in write_positions:
                    block_id = int(seq.block_table[pos // self.block_size])
                    if block_id not in first_write_offset_by_block:
                        write_blocks.append(block_id)
                        first_write_offset_by_block[block_id] = pos % self.block_size
                    else:
                        first_write_offset_by_block[block_id] = min(
                            first_write_offset_by_block[block_id], pos % self.block_size)
                valid_write_blocks = [
                    block_id for block_id in write_blocks
                    if int(first_write_offset_by_block[block_id]) > 0
                ]
                fresh_write_blocks = [
                    block_id for block_id in write_blocks
                    if int(first_write_offset_by_block[block_id]) == 0
                ]
                self.kv_offload.stats["prefetch_plans"] += 1
                self.kv_offload.stats["prefetch_write_blocks"] += len(set(write_blocks))
                protected_write_blocks = set(write_blocks)
                self.kv_offload.ensure_resident(
                    valid_write_blocks,
                    require_valid=True,
                    future_logical_blocks=future_blocks,
                    protected_logical_blocks=protected_write_blocks,
                )
                self.kv_offload.ensure_resident(
                    fresh_write_blocks,
                    require_valid=False,
                    future_logical_blocks=future_blocks,
                    protected_logical_blocks=protected_write_blocks,
                )
                slot_mapping.extend([
                    self.kv_offload.logical_to_slot[int(seq.block_table[pos // self.block_size])] * self.block_size
                    + (pos % self.block_size)
                    for pos in write_positions
                ])
                kv_offload_write_blocks.extend(write_blocks)
            else:
                if self.kv_offload is not None:
                    self.kv_offload.stats["prefetch_plans"] += 1
                    self.kv_offload.stats["prefetch_write_blocks"] += len(set(
                        int(seq.block_table[pos // self.block_size]) for pos in write_positions
                    ))
                slot_mapping.extend(self._kv_offload_translate_slots_for_positions(
                    seq.block_table,
                    write_positions,
                    future_logical_blocks=future_blocks,
                ))
            self._kv_offload_mark_pending_dirty(seq.block_table, write_positions)
        
        blockwise_prefill_active = bool(blockwise_prefill_enabled and cu_seqlens_k[-1] > cu_seqlens_q[-1])
        if cu_seqlens_k[-1] > cu_seqlens_q[-1]:      # 正常情况下二者是相等的，大于则说明有前缀缓存, 因此取出seq中的block_table, 拼成 blocktables表
            if self.kv_offload is not None:
                if blockwise_prefill_active:
                    max_len = max(len(row) for row in logical_block_table_rows)
                    logical_block_table_rows = [
                        row + [-1] * (max_len - len(row))
                        for row in logical_block_table_rows
                    ]
                    # blockwise prefill keeps logical rows on host. Attention.forward
                    # stages prefix windows layer-by-layer and therefore does not need
                    # a full physical block_table here.
                    block_tables = None
                else:
                    for seq in seqs:
                        chunk_start = getattr(seq, "prefill_chunk_start", seq.num_cached_tokens)
                        if chunk_start > 0:
                            self.kv_offload.translate_slots_for_positions(
                                seq.block_table, list(range(0, chunk_start)), require_valid=True)
                    max_len = max(len(seq.block_table) for seq in seqs)
                    block_table_rows = [seq.block_table + [-1] * (max_len - len(seq.block_table)) for seq in seqs]
                    future_blocks = set(int(block_id) for row in block_table_rows for block_id in row if int(block_id) >= 0)
                    block_table_rows = self._kv_offload_translate_block_rows(
                        block_table_rows,
                        require_valid=False,
                        future_logical_blocks=future_blocks,
                    )
                    block_tables = self.prepare_block_tables_from_rows(block_table_rows)
            else:
                block_tables = self.prepare_block_tables(seqs)
        
        # 将准备好的数据传输到GPU上
        input_ids = self._list_to_cuda(input_ids, "input_ids", torch.int64)
        positions = self._list_to_cuda(positions, "positions", torch.int64)
        cu_seqlens_q = self._list_to_cuda(cu_seqlens_q, "cu_seqlens_q", torch.int32)
        cu_seqlens_k = self._list_to_cuda(cu_seqlens_k, "cu_seqlens_k", torch.int32)
        slot_mapping = self._list_to_cuda(slot_mapping, "slot_mapping", torch.int32)
        am_compact_active = (
            self.config.am_compact_blocks > 0
            and self.config.am_compact_cache_refresh_interval > 0
            and min(len(seq) for seq in seqs) >= self.config.am_compact_min_seq_len
            and min(len(seq) for seq in seqs) > self.config.am_compact_blocks
        )
        set_context(True, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, slot_mapping, None, block_tables,
                    am_compact_blocks=(self.config.am_compact_blocks if am_compact_active else 0),
                    am_compact_selector=self.config.am_compact_selector,
                    am_compact_score_method=self.config.am_compact_score_method,
                    am_compact_beta_bound=self.config.am_compact_beta_bound,
                    am_compact_ridge_lambda=self.config.am_compact_ridge_lambda,
                    am_omp_candidate_pool_size=self.config.am_omp_candidate_pool_size,
                    am_compact_cache_refresh_interval=self.config.am_compact_cache_refresh_interval,
                    am_prefill_cache_ref_query_stride=self.config.am_prefill_cache_ref_query_stride,
                    am_compact_num_clusters=self.config.am_compact_num_clusters,
                    am_compact_route_top_k=self.config.am_compact_route_top_k,
                    am_compact_num_key_spans=self.config.am_compact_num_key_spans,
                    am_compact_decode_refit=self.config.am_compact_decode_refit,
                    am_compact_decode_refit_mode=self.config.am_compact_decode_refit_mode,
                    am_compact_decode_refit_interval=self.config.am_compact_decode_refit_interval,
                    am_compact_skip_first_layers=self.config.am_compact_skip_first_layers,
                    am_compact_skip_last_layers=self.config.am_compact_skip_last_layers,
                    am_compact_enable_layers=self.config.am_compact_enable_layers,
                    am_compact_layer_stride=self.config.am_compact_layer_stride,
                    kv_offload_manager=self.kv_offload,
                    kv_offload_blockwise_prefill=blockwise_prefill_active,
                    kv_offload_blockwise_blocks=self.config.kv_offload_blockwise_blocks,
                    kv_offload_logical_block_tables=logical_block_table_rows,
                    kv_offload_context_lens=prefill_chunk_ends,
                    kv_offload_write_blocks=[int(block) for block in kv_offload_write_blocks],
                    kv_offload_prefill_chunk_starts=prefill_chunk_starts,
                    kv_offload_prefill_chunk_ends=prefill_chunk_ends)
        return input_ids, positions

    def prepare_mixed(self, seqs: list[Sequence]):
        """Prepare a mixed chunked-prefill + decode batch as varlen prefill.

        Decode rows are represented as query length 1, so they can share the
        same FlashAttention varlen prefill path with prefill chunks.
        """
        if self.kv_offload is not None:
            raise NotImplementedError("KV offload MVP-0 暂不支持 mixed prefill+decode batch")
        input_ids = []
        positions = []
        cu_seqlens_q = [0]
        cu_seqlens_k = [0]
        max_seqlen_q = 0
        max_seqlen_k = 0
        slot_mapping = []
        logits_indices = []
        block_tables = None
        for seq in seqs:
            q_start = cu_seqlens_q[-1]
            if getattr(seq, "step_is_decode", False):
                input_ids.append(seq.last_token)
                positions.append(len(seq))
                seqlen_q = 1
                seqlen_k = len(seq)
                slot_mapping.append(seq.block_table[-1] * seq.block_size + seq.last_block_num_tokens - 1)
            else:
                seq_len = len(seq)
                chunk_start = getattr(seq, "prefill_chunk_start", seq.num_cached_tokens)
                chunk_end = getattr(seq, "prefill_chunk_end", seq_len)
                if chunk_end == 0 and chunk_start == 0:
                    chunk_end = seq_len
                input_ids.extend(seq[chunk_start:chunk_end])
                positions.extend(list(range(chunk_start, chunk_end)))
                seqlen_q = chunk_end - chunk_start
                seqlen_k = chunk_end
                for pos in range(chunk_start, chunk_end):
                    block_id = seq.block_table[pos // self.block_size]
                    slot_mapping.append(block_id * self.block_size + (pos % self.block_size))

            if getattr(seq, "step_do_sample", True):
                logits_indices.append(q_start + seqlen_q - 1)

            cu_seqlens_q.append(cu_seqlens_q[-1] + seqlen_q)
            cu_seqlens_k.append(cu_seqlens_k[-1] + seqlen_k)
            max_seqlen_q = max(max_seqlen_q, seqlen_q)
            max_seqlen_k = max(max_seqlen_k, seqlen_k)

        if cu_seqlens_k[-1] > cu_seqlens_q[-1]:
            block_tables = self.prepare_block_tables(seqs)

        input_ids = self._list_to_cuda(input_ids, "input_ids", torch.int64)
        positions = self._list_to_cuda(positions, "positions", torch.int64)
        cu_seqlens_q = self._list_to_cuda(cu_seqlens_q, "cu_seqlens_q", torch.int32)
        cu_seqlens_k = self._list_to_cuda(cu_seqlens_k, "cu_seqlens_k", torch.int32)
        slot_mapping = self._list_to_cuda(slot_mapping, "slot_mapping", torch.int32)
        logits_indices = self._list_to_cuda(logits_indices, "logits_indices", torch.int64)
        set_context(True, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k,
                    slot_mapping, None, block_tables, logits_indices)
        return input_ids, positions



    # decode阶段单token输出
    def prepare_decode(self, seqs: list[Sequence]):         #暂时跳过
        self._kv_offload_pending_dirty_blocks = []
        input_ids = []
        positions = []
        slot_mapping = []
        context_lens = []
        num_blocks_host = []
        decode_write_blocks = []
        decode_write_offsets = []
        for seq in seqs:
            # 上一次输出的最后token
            input_ids.append(seq.last_token)
            # 下一个token的位置
            positions.append(len(seq))
            context_lens.append(len(seq))
            num_blocks_host.append(seq.num_blocks)
            decode_write_blocks.append(seq.block_table[-1])
            decode_write_offsets.append(seq.last_block_num_tokens - 1)

        max_blocks = max(len(seq.block_table) for seq in seqs)
        block_table_rows = [seq.block_table + [-1] * (max_blocks - len(seq.block_table)) for seq in seqs]

        cartridge_active = should_use_kv_cartridge(
            context_lens,
            num_blocks_host,
            self.config.kv_cartridge_blocks,
            self.config.kv_cartridge_min_seq_len,
        )
        if cartridge_active:
            block_table_rows, context_lens = compress_decode_block_table_rows(
                block_table_rows,
                context_lens,
                self.block_size,
                self.config.kv_cartridge_blocks,
            )

        am_compact_active = (
            self.config.am_compact_blocks > 0
            and min(context_lens) >= self.config.am_compact_min_seq_len
            and min(context_lens) > self.config.am_compact_blocks
        )
        am_compact_cache_signatures = None
        if am_compact_active:
            am_compact_cache_signatures = tuple(
                tuple(int(block_id) for block_id in row if int(block_id) >= 0)
                for row in block_table_rows
            )

        logical_block_table_rows = [list(row) for row in block_table_rows]
        context_lens_host = list(context_lens)
        if self.kv_offload is not None:
            if self.config.kv_offload_blockwise_decode:
                # Streaming/blockwise decode only stages the current write blocks
                # here. Attention.forward will stage logical read windows layer by
                # layer, so visible blocks may exceed GPU staging slots.
                valid_write_blocks = [
                    int(block) for block, offset in zip(decode_write_blocks, decode_write_offsets)
                    if int(offset) > 0
                ]
                fresh_write_blocks = [
                    int(block) for block, offset in zip(decode_write_blocks, decode_write_offsets)
                    if int(offset) == 0
                ]
                future_blocks = set(int(block) for block in decode_write_blocks)
                protected_write_blocks = set(int(block) for block in decode_write_blocks)
                self.kv_offload.stats["prefetch_plans"] += 1
                self.kv_offload.stats["prefetch_write_blocks"] += len(future_blocks)
                self.kv_offload.ensure_resident(
                    valid_write_blocks,
                    require_valid=True,
                    future_logical_blocks=future_blocks,
                    protected_logical_blocks=protected_write_blocks,
                )
                self.kv_offload.ensure_resident(
                    fresh_write_blocks,
                    require_valid=False,
                    future_logical_blocks=future_blocks,
                    protected_logical_blocks=protected_write_blocks,
                )
            else:
                # Full attention MVP-0/MVP-1: all visible logical blocks are staged
                # on GPU before the forward.
                future_blocks = set(int(block) for row in block_table_rows for block in row if int(block) >= 0)
                future_blocks.update(int(block) for block in decode_write_blocks)
                valid_read_blocks = []
                for row, write_block, write_offset in zip(block_table_rows, decode_write_blocks, decode_write_offsets):
                    for block in row:
                        block = int(block)
                        if block < 0:
                            continue
                        if block == int(write_block) and int(write_offset) == 0:
                            continue
                        valid_read_blocks.append(block)
                self.kv_offload.stats["prefetch_plans"] += 1
                self.kv_offload.stats["prefetch_read_blocks"] += len(set(valid_read_blocks))
                self.kv_offload.stats["prefetch_write_blocks"] += len(set(int(block) for block in decode_write_blocks))
                self.kv_offload.ensure_resident(
                    valid_read_blocks,
                    require_valid=True,
                    future_logical_blocks=future_blocks,
                )
                self.kv_offload.ensure_resident(
                    decode_write_blocks,
                    require_valid=False,
                    future_logical_blocks=future_blocks,
                )
                physical_block_table_rows = self._kv_offload_translate_block_rows(
                    block_table_rows,
                    require_valid=False,
                    future_logical_blocks=future_blocks,
                )
                block_table_rows = physical_block_table_rows
            slot_mapping = [
                self.kv_offload.logical_to_slot[int(block)] * self.block_size + int(offset)
                for block, offset in zip(decode_write_blocks, decode_write_offsets)
            ]
            self._kv_offload_pending_dirty_blocks.extend(decode_write_blocks)
        else:
            slot_mapping = [
                int(block) * self.block_size + int(offset)
                for block, offset in zip(decode_write_blocks, decode_write_offsets)
            ]

        input_ids = self._list_to_cuda(input_ids, "input_ids", torch.int64)
        positions = self._list_to_cuda(positions, "positions", torch.int64)
        slot_mapping = self._list_to_cuda(slot_mapping, "slot_mapping", torch.int32)
        context_lens = self._list_to_cuda(context_lens, "context_lens", torch.int32)
        block_tables = self.prepare_block_tables_from_rows(block_table_rows)

        # Quest 早返回判定（host 端，避免每层 .item() 触发 GPU sync）：
        #   1) 至少一条 seq 满足 seq_len >= min_seq_len（按 min 算保守）
        #   2) num_blocks > top_k（不然 top-k 退化为 full）
        #   3) **短序列保护**：top_k * block_size 已经 >= 最长 seq * 0.8 时，
        #      Quest 能裁掉的块 <20%，selection overhead 远超收益 → 降级 full attention
        #      （kv-sparse-attention.md §5.5 #4）
        cfg_top_k = self.config.quest_top_k_blocks if not (cartridge_active or am_compact_active) else -1
        cfg_min_len = self.config.quest_min_seq_len
        if cfg_top_k > 0 and seqs:
            min_seq_len_host = min(len(s) for s in seqs)
            min_blocks_host = min(s.num_blocks for s in seqs)
            max_seq_len_host = max(len(s) for s in seqs)
            cover = cfg_top_k * self.block_size  # top-k 已能覆盖的 token 数
            short_seq_skip = cover >= max_seq_len_host * 0.8
            quest_active_top_k = cfg_top_k if (
                min_seq_len_host >= cfg_min_len
                and min_blocks_host > cfg_top_k
                and not short_seq_skip
            ) else -1
        else:
            quest_active_top_k = -1
        set_context(False, slot_mapping=slot_mapping, context_lens=context_lens, block_tables=block_tables,
                    quest_top_k_blocks=quest_active_top_k,
                    quest_min_seq_len=cfg_min_len,
                    am_compact_blocks=(self.config.am_compact_blocks if am_compact_active else 0),
                    am_compact_selector=self.config.am_compact_selector,
                    am_compact_score_method=self.config.am_compact_score_method,
                    am_compact_beta_bound=self.config.am_compact_beta_bound,
                    am_compact_ridge_lambda=self.config.am_compact_ridge_lambda,
                    am_omp_candidate_pool_size=self.config.am_omp_candidate_pool_size,
                    am_compact_cache_refresh_interval=self.config.am_compact_cache_refresh_interval,
                    am_compact_num_clusters=self.config.am_compact_num_clusters,
                    am_compact_route_top_k=self.config.am_compact_route_top_k,
                    am_compact_num_key_spans=self.config.am_compact_num_key_spans,
                    am_compact_decode_refit=self.config.am_compact_decode_refit,
                    am_compact_decode_refit_mode=self.config.am_compact_decode_refit_mode,
                    am_compact_decode_refit_interval=self.config.am_compact_decode_refit_interval,
                    am_compact_skip_first_layers=self.config.am_compact_skip_first_layers,
                    am_compact_skip_last_layers=self.config.am_compact_skip_last_layers,
                    am_compact_enable_layers=self.config.am_compact_enable_layers,
                    am_compact_layer_stride=self.config.am_compact_layer_stride,
                    am_compact_cache_signatures=am_compact_cache_signatures,
                    kv_offload_manager=self.kv_offload,
                    kv_offload_blockwise_decode=(self.kv_offload is not None and self.config.kv_offload_blockwise_decode),
                    kv_offload_blockwise_blocks=self.config.kv_offload_blockwise_blocks,
                    kv_offload_logical_block_tables=logical_block_table_rows,
                    kv_offload_context_lens=context_lens_host,
                    kv_offload_write_blocks=[int(block) for block in decode_write_blocks])
        return input_ids, positions

    # 生成 temperatures列表，并传到GPU上
    def prepare_sample(self, seqs: list[Sequence]):
        temperatures = []
        for seq in seqs:
            temperatures.append(seq.temperature)
        temperatures = self._list_to_cuda(temperatures, "temperatures", torch.float32)    #pin_memory=True将张量存储在锁定内存（page-locked memory）中，而非普通的可分页内存
        return temperatures
        #普通可分页内存（Pageable Memory）  |	锁定内存（Page-locked Memory / Pinned Memory）
        #操作系统可将其 “分页” 到磁盘       |     被 “锁定” 在物理内存中，不允许换出到磁盘,
        # （swap） ，释放物理内存给其他进程 |     

    def _select_sample_rows(self, logits: torch.Tensor, seqs: list[Sequence],
                            batch_kind: str | None) -> tuple[torch.Tensor, list[Sequence]]:
        if batch_kind != "mixed":
            return logits, seqs
        sample_seqs = [seq for seq in seqs if getattr(seq, "step_do_sample", True)]
        return logits, sample_seqs


    @torch.inference_mode()
    #只需要前向传播 禁用梯度计算（无需反向传播），节省内存；
    # 加速推理过程（跳过与训练相关的检查和操作）。
    def run_model(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        is_prefill: bool,
        input_embeds: torch.Tensor | None = None,
        return_hidden: bool = False,
    ):
        # Quest 实际启用时（context 已确认）才走 eager；否则照常走 cuda graph
        quest_active = (not is_prefill) and (get_context().quest_top_k_blocks > 0)
        am_active = (not is_prefill) and (get_context().am_compact_blocks > 0)
        # C4：decode 反量化每步都要 alloc，cuda graph 无法 replay，强制 eager
        c4_active = self.config.kv_quant_bits == 4
        # cpu_offload：init 阶段已跳过 capture，这里也必须走 eager（否则 self.graphs 不存在）
        offload_active = self.config.cpu_offload
        kv_offload_active = self.config.kv_offload_mvp0
        if (is_prefill or self.enforce_eager or input_ids.size(0) > 512
                or quest_active or am_active or c4_active or offload_active or kv_offload_active
                or input_embeds is not None or return_hidden):     #动态执行 eager mode
            hidden_states = self.model(input_ids, positions, input_embeds=input_embeds)
            logits = self.model.compute_logits(hidden_states)
            if return_hidden:
                return logits, hidden_states
            return logits
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


    def run(self, seqs:list[Sequence], is_prefill: bool, do_sample: bool = True,
            batch_kind: str | None = None) -> list[int] | None:
        if batch_kind == "mixed":
            input_ids, positions = self.prepare_mixed(seqs)
        else:
            input_ids, positions = self.prepare_prefill(seqs) if is_prefill else self.prepare_decode(seqs)
        self._kv_offload_before_forward()
        logits = self.run_model(input_ids, positions, is_prefill)
        self._kv_offload_after_forward()
        if not do_sample:
            reset_context()
            return None
        if self.rank == 0:
            logits, sample_seqs = self._select_sample_rows(logits, seqs, batch_kind)
            temperatures = self.prepare_sample(sample_seqs)    #只有主进程做采样
            token_ids = self.sampler(logits, temperatures).tolist()
        else:
            token_ids = None
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
