from copy import copy
from enum import Enum, auto
from itertools import count
from tinyvllm.sampling_params import SamplingParams

class SequenceStatus(Enum):
    WAITING = auto()            # 1
    RUNNING = auto()            # 2
    FINISHED = auto()           # 3

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
        self.block_table = []                           # 记录当前语句用到的 块id
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
                 self.token_ids if self.num_completion_tokens == 0 else self.last_token)
    
    # 由于是多卡，涉及通信接收，需要对序列化的 Sequence 进行解析，该函数和 getstate函数一一对应
    def __setstate__(self, state):
        self.num_tokens, self.num_prompt_tokens, self.num_cached_blocks, self.block_table = state[-1]
        if self.num_completion_tokens == 0:
            self.token_ids = state[-1]
        else:
            self.last_token = state[-1]
    