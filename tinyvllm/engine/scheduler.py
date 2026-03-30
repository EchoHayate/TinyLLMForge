from collections import deque

from tinyvllm.config import Config
from tinyvllm.engine.sequence import Sequence, SequenceStatus
from tinyvllm.engine.block_manager import BlockManager, CPUBlockAllocator

class Scheduler:

    def __init__(self, config: Config):
        self.max_num_seqs = config.max_num_seqs
        self.max_num_batched_tokens = config.max_num_batched_tokens
        self.eos = config.eos
        self.block_manager = BlockManager(config.num_kvcache_blocks, config.kvcache_block_size)
        self.cpu_block_manager = CPUBlockAllocator(getattr(config, "num_cpu_kvcache_blocks", 0))
        self.waiting: deque[Sequence] = deque()     # 未分配 KV 缓存块
        self.running: deque[Sequence] = deque()     # 已分配 KV 缓存块  参与decode阶段生成
        self.swapped: deque[Sequence] = deque()     # 已分配 CPU 缓存块 被换出的序列
    
    def is_finished(self):
        return not self.waiting and not self.running and not self.swapped

    def add(self, seq: Sequence):
        self.waiting.append(seq)

    def _swap_out(self, seq: Sequence) -> dict[int, int]:
        if not self.block_manager.can_swap_out(seq, self.cpu_block_manager):
            return {}
        mapping = self.block_manager.swap_out(seq, self.cpu_block_manager)
        seq.status = SequenceStatus.SWAPPED
        self.swapped.append(seq)
        return mapping

    def _swap_in(self, seq: Sequence) -> dict[int, int]:
        if not self.block_manager.can_swap_in(seq):
            return {}
        mapping = self.block_manager.swap_in(seq, self.cpu_block_manager)
        return mapping

    def schedule(self) -> tuple[list[Sequence], bool, dict[int, int], dict[int, int]]:
        swap_in_map = {}
        swap_out_map = {}
        scheduled_seqs = []
        
        # 1. Swap in phase
        while self.swapped:
            seq = self.swapped[0]
            if self.block_manager.can_swap_in(seq):
                mapping = self._swap_in(seq)
                swap_in_map.update(mapping)
                seq.status = SequenceStatus.WAITING
                self.swapped.popleft()
                self.waiting.appendleft(seq) # 换入后优先级最高，放入waiting队首
            else:
                break
                
        # 2. prefill, 从 waiting 队列中取出 seq
        num_seqs = 0        #number of sequence in the current batch
        num_batched_tokens = 0
        while self.waiting and num_seqs < self.max_num_seqs:
            seq = self.waiting[0]
            if num_batched_tokens + len(seq) > self.max_num_batched_tokens or not self.block_manager.can_allocate(seq):
                break
            num_seqs += 1
            self.block_manager.allocate(seq)
            num_batched_tokens += len(seq) - seq.num_cached_tokens
            seq.status = SequenceStatus.RUNNING
            self.waiting.popleft()
            self.running.append(seq)
            scheduled_seqs.append(seq)
        if scheduled_seqs:
            return scheduled_seqs, True, swap_in_map, swap_out_map

        # 3. decode，从 running 队列中取出 seq
        while self.running and num_seqs < self.max_num_seqs:        
            seq = self.running.popleft()
            while not self.block_manager.can_append(seq):
                victim = self.running.pop() if self.running else seq
                mapping = self._swap_out(victim)
                if mapping:
                    swap_out_map.update(mapping)
                else:
                    self.preempt(victim)
                if victim is seq:
                    break
            else:
                num_seqs += 1
                self.block_manager.may_append(seq)
                scheduled_seqs.append(seq)
        if scheduled_seqs:
            self.running.extendleft(reversed(scheduled_seqs))
        return scheduled_seqs, False, swap_in_map, swap_out_map    

    def preempt(self, seq: Sequence):       #将正在running队列中的seq给“踢”出去 
        seq.status = SequenceStatus.WAITING
        self.block_manager.deallocate(seq)
        self.waiting.appendleft(seq)

    def postprocess(self, seqs: list[Sequence], token_ids: list[int]):
        for seq, token_id in zip(seqs, token_ids):
            seq.append_token(token_id)
            # 如果不能忽略句子终止符号，并且遇到了终止符号
            # 或者生成的长度已经达到了最大值
            if (not seq.ignore_eos and token_id == self.eos) or seq.num_completion_tokens == seq.max_tokens:
                seq.status = SequenceStatus.FINISHED
                self.block_manager.deallocate(seq)
                self.running.remove(seq)
