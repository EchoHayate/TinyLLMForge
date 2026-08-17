from copy import copy
from enum import Enum, auto
from itertools import count
import math
from tinyvllm.sampling_params import SamplingParams

class SequenceStatus(Enum):
    WAITING = auto()            # 1
    RUNNING = auto()            # 2
    FINISHED = auto()           # 3
    PREFILLING = auto()         # 4，chunked prefill 中：KV 已分配但 prompt 尚未完整算完

class Sequence:
    block_size = 256            #通过block管理token  不同 Seq 的 KV 缓存数据是严格隔离的
    counter = count()           # 每次返回值后，+1,  [0, 1, 2, 3, ...]

    def __init__(self, token_ids: list[int], sampling_params = SamplingParams()):
        self.seq_id = next(Sequence.counter)
        self.status = SequenceStatus.WAITING            # 默认进入等待队列
        self.token_ids = copy(token_ids)                # 记录每次生成+prompt的所有token id
        self.last_token = token_ids[-1]                 # 记录每次生成后的最后一个token id
        self.num_tokens = len(self.token_ids)           # 记录每次生成+prompt的所有token 数量
        self.num_prompt_tokens = len(token_ids)         # 记录prompt的token数量 传入时就确定了
        self.num_cached_tokens = 0                      # 记录prefix cache的 token数量
        self.num_computed_tokens = 0                    # chunked prefill 已经写入 KV cache 的 prompt token 数
        self.prefill_chunk_start = 0                    # 当前 prefill chunk 的起始 token 位置（含）
        self.prefill_chunk_end = 0                      # 当前 prefill chunk 的结束 token 位置（不含）
        self.prefill_chunk_final = False                # 当前 chunk 是否覆盖 prompt 末尾，并需要采样首个输出 token
        self.step_is_decode = False                     # mixed batch 中当前 step 是否按 decode token 处理
        self.step_do_sample = True                      # mixed batch 中当前 step 是否需要消费一个 sampled token
        self.block_table = []                           # 记录当前语句用到的 块id
        self.hybrid_state_slot_id = -1                  # request-indexed fixed state slot; -1 means disabled/unallocated
        self.hybrid_state_generation = 0                # increments whenever a released slot is reused
        self.temperature = sampling_params.temperature  # 记录该语句的采样温度
        self.max_tokens = sampling_params.max_tokens    # 记录该语句的最大生成长度
        self.ignore_eos = sampling_params.ignore_eos    # 记录是否忽略句子的结束符号


    def __len__(self):                                  # 声明 __len__ 函数, 使得可以调用 len(Sequence) 返回长度
        return self.num_tokens
    
    def __getitem__(self, key):                         # 声明 __getitem__ 函数, 使得可以调用 Sequence[key] 获取对应索引的 token_id
        return self.token_ids[key]
    
    @property                                           # property将一个类方法伪装成属性，可以不用括号就可以调用
    def is_finished(self):                              # 判断当前语句是否生成完成
        return self.status == SequenceStatus.FINISHED  
    
    @property
    def num_completion_tokens(self):                    # 计算生成的 token 数量
        return self.num_tokens - self.num_prompt_tokens       

    @property
    def prompt_token_ids(self):                         # 返回初始的提示prompts
        return self.token_ids[:self.num_prompt_tokens]       

    @property
    def completion_token_ids(self):                     # 返回生成的 token id
        return self.token_ids[self.num_prompt_tokens:]
    
    @property
    def num_cached_blocks(self):                        # 计算缓存的 block 数量
        return self.num_cached_tokens // self.block_size
    

    @property                   
    def num_blocks(self):                               # 计算当前语句消耗的 block 数量
        return (self.num_tokens + self.block_size - 1) // self.block_size

    @property
    def last_block_num_tokens(self):                    # 计算最后一个块中的 token 数量             
        return self.num_tokens - (self.num_blocks - 1) * self.block_size
    
    def block(self, i):                                 # 返回 block[i]中的 token_ids列表
        assert 0 <= i < self.num_blocks
        return self.token_ids[i * self.block_size : (i + 1) * self.block_size]
    
    def append_token(self, token_id: int):              # 在 token_ids 列表后添加一个 token, 并更新相应状态
        self.token_ids.append(token_id)
        self.last_token = token_id
        self.num_tokens += 1

    # 由于是多卡，涉及通信发送，需要将sequence进行序列化，这个函数是决定将哪些 Sequence的属性进行序列化传输
    # 增加这个魔术方法后，pickle模块会自动调用该函数，将 Sequence 数据进行序列化
    def __getstate__(self):                             
        return (self.num_tokens, self.num_prompt_tokens, self.num_cached_blocks, self.block_table,
                self.num_computed_tokens, self.prefill_chunk_start, self.prefill_chunk_end,
                self.prefill_chunk_final, self.step_is_decode, self.step_do_sample,
                self.seq_id, self.hybrid_state_slot_id, self.hybrid_state_generation,
                self.temperature, self.max_tokens,
                self.token_ids if self.num_completion_tokens == 0 else self.last_token)
    
    # 由于是多卡，涉及通信接收，需要对序列化的 Sequence 进行解析，该函数和 getstate函数一一对应
    def __setstate__(self, state):
        # 新 state 是 16 元组：(num_tokens, num_prompt_tokens, num_cached_blocks, block_table,
        # num_computed_tokens, prefill_chunk_start, prefill_chunk_end, prefill_chunk_final,
        # step_is_decode, step_do_sample, seq_id, hybrid_state_slot_id, hybrid_state_generation,
        # temperature, max_tokens, token_ids 或 last_token)。最后一项按 num_completion_tokens 分支：
        # 旧 15 元组没有 max_tokens，恢复为 0。
        # 旧 14 元组没有 temperature，恢复为 greedy 0.0。
        # 旧 11 元组没有 hybrid state 字段，也恢复为 disabled sentinel。
        #   - 还在 prefill（completion=0）：last item 是完整 token_ids
        #   - 已进入 decode（completion>0）：last item 是 last_token（int）
        # 注意：num_cached_blocks 是 @property，不能直接赋值，反推回 num_cached_tokens。
        self.num_tokens, self.num_prompt_tokens, num_cached_blocks, self.block_table = state[:4]
        self.seq_id = -1
        self.step_is_decode = False
        self.step_do_sample = True
        self.hybrid_state_slot_id = -1
        self.hybrid_state_generation = 0
        self.temperature = 0.0
        self.max_tokens = 0
        self.num_cached_tokens = num_cached_blocks * self.block_size
        if len(state) >= 9:
            (self.num_computed_tokens, self.prefill_chunk_start,
             self.prefill_chunk_end, self.prefill_chunk_final) = state[4:8]
        else:
            self.num_computed_tokens = self.num_cached_tokens
            self.prefill_chunk_start = self.num_cached_tokens
            self.prefill_chunk_end = self.num_tokens
            self.prefill_chunk_final = True
        if len(state) >= 11:
            self.step_is_decode, self.step_do_sample = state[8:10]
        if len(state) >= 14:
            self.seq_id = state[10]
            self.hybrid_state_slot_id = state[11]
            self.hybrid_state_generation = state[12]
        elif len(state) == 13:
            self.hybrid_state_slot_id = state[10]
            self.hybrid_state_generation = state[11]
        if len(state) >= 15:
            temperature = state[13]
            if (
                isinstance(temperature, bool)
                or not isinstance(temperature, (int, float))
                or not math.isfinite(temperature)
            ):
                raise ValueError("temperature must be a finite number")
            self.temperature = float(temperature)
        if len(state) >= 16:
            max_tokens = state[14]
            if (
                isinstance(max_tokens, bool)
                or not isinstance(max_tokens, int)
                or max_tokens < 0
            ):
                raise ValueError(
                    "max_tokens must be a non-negative integer"
                )
            self.max_tokens = max_tokens
        if self.num_completion_tokens == 0:
            self.token_ids = state[-1]
        else:
            self.last_token = state[-1]
    
