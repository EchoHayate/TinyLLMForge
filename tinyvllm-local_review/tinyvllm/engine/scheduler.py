from collections import deque

from tinyvllm.config import Config
from tinyvllm.engine.sequence import Sequence, SequenceStatus
from tinyvllm.engine.block_manager import BlockManager

class Scheduler:

    def __init__(self, config: Config):
        self.max_num_seqs = config.max_num_seqs
        self.max_num_batched_tokens = config.max_num_batched_tokens
        self.eos = config.eos
        self.block_manager = BlockManager(config.num_kvcache_blocks, config.kvcache_block_size)
        self.waiting: deque[Sequence] = deque()     #未分配 KV 缓存块
        self.running: deque[Sequence] = deque()     #已分配 KV 缓存块  参与decode阶段生成
    def is_finished(self):
        return not self.waiting and not self.running

    def add(self, seq: Sequence):
        self.waiting.append(seq)

    def schedule(self) -> tuple[list[Sequence], bool]:
        # prefill, 从 waiting 队列中取出 seq   prefill阶段：处理输入 prompt 的所有 token（批量计算，生成初始 KV 缓存）。
        scheduled_seqs = [] #scheduled_seqs和waiting队列的区别：scheduled_seqs 是从 waiting 队列中筛选出来的、满足调度条件的序列集合
        num_seqs = 0        #number of sequence in the current batch
        num_batched_tokens = 0
        while self.waiting and num_seqs < self.max_num_seqs:
            seq = self.waiting[0]                   # 这里不使用 popleft的原因是 waiting 队列不一定调度成功（如下if判断） 如果调度不成功 这个token就不在waiting队列里了
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
            return scheduled_seqs, True

        # decode，从 running 队列中取出 seq   Decode 阶段：逐 token 生成（利用已有 KV 缓存，每次生成一个新 token）。
        while self.running and num_seqs < self.max_num_seqs:        
            seq = self.running.popleft();          # 这里是preempt抢占资源保证 running队列一定调度成功
            #[thinking] 这里可能有一个能够优化的点 就是在抢占资源的时候默认是t出running的第一个 但是第一个腾出来的空间未必够新的seq使用 所以可以考虑合理规划选一个大小相近的seq去剔除
            while not self.block_manager.can_append(seq):
                if self.running:
                    self.preempt(self.running.pop())
                else:
                    self.preempt(seq)
                    break
            else:
                num_seqs += 1
                self.block_manager.may_append(seq)
                scheduled_seqs.append(seq)
        assert scheduled_seqs
        self.running.extendleft(reversed(scheduled_seqs))       #当前step结束 但未到达终止条件 所以需要在返回running队列
        return scheduled_seqs, False    

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
