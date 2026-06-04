from __future__ import annotations

from collections import deque

from tinyvllm.config import Config
from tinyvllm.engine.sequence import Sequence, SequenceStatus
from tinyvllm.engine.block_manager import BlockManager

class Scheduler:

    def __init__(self, config: Config):
        self.max_num_seqs = config.max_num_seqs
        self.max_num_batched_tokens = config.max_num_batched_tokens
        self.max_num_prefill_tokens_per_step = getattr(config, "max_num_prefill_tokens_per_step", 0)
        self.chunked_prefill_decode_first = getattr(config, "chunked_prefill_decode_first", True)
        self.chunked_prefill_max_consecutive_chunks = getattr(config, "chunked_prefill_max_consecutive_chunks", 0)
        self.chunked_prefill_mixed_batch = getattr(config, "chunked_prefill_mixed_batch", False)
        self._consecutive_prefill_chunks = 0
        self.eos = config.eos
        self.block_manager = BlockManager(config.num_kvcache_blocks, config.kvcache_block_size)
        self.waiting: deque[Sequence] = deque()     #未分配 KV 缓存块
        self.prefilling: deque[Sequence] = deque()  #chunked prefill 中，已分配 KV 但 prompt 未完整算完
        self.running: deque[Sequence] = deque()     #已分配 KV 缓存块  参与decode阶段生成

    @property
    def chunked_prefill_enabled(self) -> bool:
        return self.max_num_prefill_tokens_per_step > 0

    def is_finished(self):
        return not self.waiting and not self.prefilling and not self.running

    def add(self, seq: Sequence):
        self.waiting.append(seq)

    def schedule(self) -> tuple[list[Sequence], bool, bool]:
        if self.chunked_prefill_enabled:
            if self.chunked_prefill_decode_first and self.running:
                self._consecutive_prefill_chunks = 0
                return (*self._schedule_decode(), True)
            if (self.running
                    and self.chunked_prefill_max_consecutive_chunks > 0
                    and self._consecutive_prefill_chunks >= self.chunked_prefill_max_consecutive_chunks):
                self._consecutive_prefill_chunks = 0
                return (*self._schedule_decode(), True)
            if self.chunked_prefill_mixed_batch and self.running:
                mixed = self._schedule_mixed_prefill_decode()
                if mixed is not None:
                    if len(mixed) == 4:
                        self._consecutive_prefill_chunks = 0
                    else:
                        self._consecutive_prefill_chunks += 1
                    return mixed
            prefill = self._schedule_chunked_prefill()
            if prefill is not None:
                self._consecutive_prefill_chunks += 1
                return prefill
            self._consecutive_prefill_chunks = 0
            return (*self._schedule_decode(), True)

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
            seq.num_computed_tokens = len(seq)
            seq.prefill_chunk_start = seq.num_cached_tokens
            seq.prefill_chunk_end = len(seq)
            seq.prefill_chunk_final = True
            num_batched_tokens += len(seq) - seq.num_cached_tokens
            seq.status = SequenceStatus.RUNNING
            self.waiting.popleft()
            self.running.append(seq)
            scheduled_seqs.append(seq)
        if scheduled_seqs:
            return scheduled_seqs, True, True

        # decode，从 running 队列中取出 seq   Decode 阶段：逐 token 生成（利用已有 KV 缓存，每次生成一个新 token）。
        return (*self._schedule_decode(), True)

    def _schedule_decode(self) -> tuple[list[Sequence], bool]:
        scheduled_seqs = []
        num_seqs = 0
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

    def _schedule_chunked_prefill(self, max_prefill_seqs: int | None = None) -> tuple[list[Sequence], bool, bool] | None:
        max_prefill_seqs = self.max_num_seqs if max_prefill_seqs is None else max_prefill_seqs
        if self.prefilling:
            seq = self.prefilling.popleft()
            return self._schedule_one_prefill_chunk(seq)

        if not self.waiting:
            return None

        candidate = self.waiting[0]
        if not self.block_manager.can_allocate(candidate):
            return None
        seq = self.waiting.popleft()
        self.block_manager.allocate(seq, publish_hashes=False)
        seq.status = SequenceStatus.PREFILLING
        first = self._schedule_one_prefill_chunk(seq)
        if first is None:
            return None
        scheduled, is_prefill, do_sample = first
        if not do_sample:
            return first

        num_batched_tokens = scheduled[0].prefill_chunk_end - scheduled[0].prefill_chunk_start
        while self.waiting and len(scheduled) < max_prefill_seqs:
            candidate = self.waiting[0]
            # Conservative short-prompt batching: only admit prompts that finish
            # in one chunk without relying on prefix-cache state discovered after allocation.
            if len(candidate) > self.max_num_prefill_tokens_per_step:
                break
            if num_batched_tokens + len(candidate) > self.max_num_batched_tokens:
                break
            if not self.block_manager.can_allocate(candidate):
                break
            seq = self.waiting.popleft()
            self.block_manager.allocate(seq, publish_hashes=False)
            seq.status = SequenceStatus.PREFILLING
            one = self._schedule_one_prefill_chunk(seq)
            if one is None or not one[2]:
                self.prefilling.appendleft(seq)
                break
            scheduled.append(seq)
            num_batched_tokens += seq.prefill_chunk_end - seq.prefill_chunk_start
        return scheduled, is_prefill, do_sample

    def _schedule_one_prefill_chunk(self, seq: Sequence) -> tuple[list[Sequence], bool, bool] | None:
        if seq.num_computed_tokens >= len(seq):
            # 全 prompt 命中 prefix cache 时仍需重算最后一个 prompt token 拿 logits，采样首个输出 token。
            seq.prefill_chunk_start = max(0, len(seq) - 1)
            seq.prefill_chunk_end = len(seq)
            seq.prefill_chunk_final = True
            return [seq], True, True

        start = seq.num_computed_tokens
        chunk_len = min(self.max_num_prefill_tokens_per_step, len(seq) - start)
        end = start + chunk_len
        seq.prefill_chunk_start = start
        seq.prefill_chunk_end = end
        seq.prefill_chunk_final = (end == len(seq))
        return [seq], True, seq.prefill_chunk_final

    def _schedule_mixed_prefill_decode(self) -> tuple[list[Sequence], bool, bool, str] | None:
        prefill_slots = max(1, self.max_num_seqs - 1)
        prefill = self._schedule_chunked_prefill(max_prefill_seqs=prefill_slots)
        if prefill is None:
            return None
        prefill_seqs, is_prefill, prefill_do_sample = prefill
        assert is_prefill
        decode_seqs = []
        while self.running and len(prefill_seqs) + len(decode_seqs) < self.max_num_seqs:
            seq = self.running.popleft()
            while not self.block_manager.can_append(seq):
                if self.running:
                    self.preempt(self.running.pop())
                else:
                    self.preempt(seq)
                    seq = None
                    break
            if seq is None:
                continue
            self.block_manager.may_append(seq)
            decode_seqs.append(seq)

        if not decode_seqs:
            return prefill

        for seq in prefill_seqs:
            seq.step_is_decode = False
            seq.step_do_sample = prefill_do_sample
        for seq in decode_seqs:
            seq.step_is_decode = True
            seq.step_do_sample = True
        return prefill_seqs + decode_seqs, True, True, "mixed"

    def preempt(self, seq: Sequence):       #将正在running队列中的seq给“踢”出去 
        seq.status = SequenceStatus.WAITING
        self.block_manager.deallocate(seq)
        self.waiting.appendleft(seq)

    def postprocess(self, seqs: list[Sequence], token_ids: list[int] | None,
                    is_prefill: bool = False, do_sample: bool = True,
                    batch_kind: str | None = None):
        if batch_kind == "mixed":
            self._postprocess_mixed(seqs, token_ids)
            return
        if is_prefill and self.chunked_prefill_enabled:
            self._postprocess_chunked_prefill(seqs, token_ids, do_sample)
            return
        for seq, token_id in zip(seqs, token_ids):
            seq.append_token(token_id)
            # 如果不能忽略句子终止符号，并且遇到了终止符号
            # 或者生成的长度已经达到了最大值
            if (not seq.ignore_eos and token_id == self.eos) or seq.num_completion_tokens == seq.max_tokens:
                seq.status = SequenceStatus.FINISHED
                self.block_manager.deallocate(seq)
                self.running.remove(seq)

    def _postprocess_chunked_prefill(self, seqs: list[Sequence], token_ids: list[int] | None, do_sample: bool):
        token_iter = iter(token_ids or [])
        for seq in seqs:
            old_end = seq.num_computed_tokens
            new_end = max(seq.num_computed_tokens, seq.prefill_chunk_end)
            self.block_manager.commit_prefill(seq, old_end, new_end)
            seq.num_computed_tokens = new_end

            if not do_sample:
                seq.status = SequenceStatus.PREFILLING
                self.prefilling.append(seq)
                continue

            token_id = next(token_iter)
            seq.append_token(token_id)
            if (not seq.ignore_eos and token_id == self.eos) or seq.num_completion_tokens == seq.max_tokens:
                seq.status = SequenceStatus.FINISHED
                self.block_manager.deallocate(seq)
            else:
                seq.status = SequenceStatus.RUNNING
                self.running.append(seq)

    def _postprocess_mixed(self, seqs: list[Sequence], token_ids: list[int] | None):
        token_iter = iter(token_ids or [])
        for seq in seqs:
            if getattr(seq, "step_is_decode", False):
                token_id = next(token_iter)
                seq.append_token(token_id)
                if (not seq.ignore_eos and token_id == self.eos) or seq.num_completion_tokens == seq.max_tokens:
                    seq.status = SequenceStatus.FINISHED
                    self.block_manager.deallocate(seq)
                else:
                    seq.status = SequenceStatus.RUNNING
                    self.running.append(seq)
                seq.step_is_decode = False
                seq.step_do_sample = True
                continue

            old_end = seq.num_computed_tokens
            new_end = max(seq.num_computed_tokens, seq.prefill_chunk_end)
            self.block_manager.commit_prefill(seq, old_end, new_end)
            seq.num_computed_tokens = new_end
            if not getattr(seq, "step_do_sample", True):
                seq.status = SequenceStatus.PREFILLING
                self.prefilling.append(seq)
            else:
                token_id = next(token_iter)
                seq.append_token(token_id)
                if (not seq.ignore_eos and token_id == self.eos) or seq.num_completion_tokens == seq.max_tokens:
                    seq.status = SequenceStatus.FINISHED
                    self.block_manager.deallocate(seq)
                else:
                    seq.status = SequenceStatus.RUNNING
                    self.running.append(seq)
            seq.step_is_decode = False
            seq.step_do_sample = True
