from collections import deque
import xxhash
#[thinking] 这里是把token_ids转换成hash值 然后去做处理 这样做：用哈希处理 token_ids 能减轻 KV 缓存负担  有没有别的方式呢
import numpy as np

from tinyvllm.engine.sequence import Sequence

class Block:
    def __init__(self, block_id):           # 单个block块的属性
        self.block_id = block_id            # 块id
        self.ref_count = 0                  # 引用数量 主要涉及相同前缀的引用
        self.hash = -1                      # hash值 用于比较block大小  -1表示无效 
        self.token_ids = []                 # 包含的 token_id
    
    def update(self, hash: int, token_ids: list[int]):      # 记录该 block 的哈希值和所有 block_id
        self.hash = hash
        self.token_ids = token_ids

    def reset(self):                       # 重置 block状态，注意 ref_count 初始化为 1  为下一次BlockManager.allocate()做准备
        self.ref_count = 1
        self.hash = -1
        self.token_ids = []
    

class BlockManager:
    def __init__(self, num_blocks: int, block_size: int):
        assert num_blocks > 0
        self.block_size = block_size
        self.blocks: list[Block] = [Block(i) for i in range(num_blocks)]   
        self.hash_to_block_id: dict[int, int] = dict()              #键代表某个block的token序列的哈希值 值代表这个哈希值对应的kv缓存块的id（block_id） 用于快速查找和复用内容相同的 KV 缓存块
        self.free_block_ids: deque[int] = deque(range(num_blocks))  #双向队列分配和回收元素
        self.used_block_ids: set[int] = set()                       #跟踪所有正在被使用的block_id 查找的时间复杂度O（1） 如果使用deque 查找的时间复杂度为O（n） 

#block只有占满的时候 才会计算hash
    # 以整数形式，返回计算出的哈希值
    @classmethod                           # 对标c++中的static, 第一个参数为类本身，cls, class self
    def compute_hash(cls, token_ids: list[int], prefix: int = -1):   #prefix表示是否依赖于上一个block的hash 若为-1 表示当前是第一个block 不需要
        h = xxhash.xxh64()
        if prefix != -1:
            h.update(prefix.to_bytes(8, "little"))   #小端字节序处理 prefix消除平台差异
        h.update(np.array(token_ids).tobytes())
        return h.intdigest()

    # 分配对应id的block, 重置状态，并且更新 free_block_ids 队列 和 used_block_ids 集合
    def _allocate_block(self, block_id: int) -> Block:
        block = self.blocks[block_id]
        assert block.ref_count == 0
        block.reset()
        self.free_block_ids.remove(block_id)
        self.used_block_ids.add(block_id)
        return self.blocks[block_id]

    # [warning!] 隐含错误 定义了返回类型 但是没有return 实际返回 None
    #将块从 “使用中” 状态转为 “空闲” 状态（例如从 used_block_ids 移到 free_block_ids），但不会清除该块的哈希映射（hash_to_block_id 中 h→block_id 的关联）和块本身存储的 token_ids。
    def _deallocate_block(self, block_id: int) -> Block:
        assert self.blocks[block_id].ref_count == 0
        self.used_block_ids.remove(block_id)
        self.free_block_ids.append(block_id)

    # can_allocate 和 allocate 函数都是在prefill阶段调用
    # allocate, deallocate函数，都是针对一条 sequence 语句来说的
    def can_allocate(self, seq: Sequence) -> bool:
        return len(self.free_block_ids) >= seq.num_blocks

    # allocate  blocks for the sequences, update the block table and hash table
    def allocate(self, seq: Sequence):    
        assert not seq.block_table
        h = -1
        cache_miss = False
        for i in range(seq.num_blocks):
            token_ids = seq.block(i)     #token_ids：块中包含的 token 编号列表  核心作用是用于计算当前块的哈希值
            # 未填满的块（非完整块）的哈希值为 -1，不纳入缓存（因为复用价值低）
            h = self.compute_hash(token_ids, h) if len(token_ids) == self.block_size else -1   #计算hash的前提是当前block_size能被占满 如果占不满说明当前sequence结束
            block_id = self.hash_to_block_id.get(h, -1)
            # 没有缓存或者缓存未命中
            if block_id == -1 or self.blocks[block_id].token_ids != token_ids:    #self.blocks[block_id].token_ids != token_ids 确保内容没有变动过
                cache_miss = True
            
            # 没有缓存或者缓存为命中，那么就从空闲块表的头部，取出一块进行分配
            if cache_miss:
                block_id = self.free_block_ids[0]
                block = self._allocate_block(block_id)
            # 缓存命中
            else:
                seq.num_cached_tokens += self.block_size
                if block_id in self.used_block_ids:   #可复用的块
                    block = self.blocks[block_id]
                    block.ref_count += 1
                # 由于deallocate 并没有清除字典的hash， 也没有清除 block.token_id 列表。
                # 因此通过字典映射的 block_id， 可能已经被_deallocate了，但是由于 token_id还在，因此也可以用于缓存
                # 所以需要 _allocate_block回来
                else:    #曾经用过但已释放的块  保留哈希映射和块内容，让这些块能被再次快速复用
                    block = self._allocate_block(block_id)

            if h != -1:      #相同的 token_ids 序列（通过哈希 h 标识）始终对应到同一个 block_id
                block.update(h, token_ids)   #对这一步的操作不是很明白 为什么要更新
                self.hash_to_block_id[h] = block_id
            seq.block_table.append(block_id)
            
    def deallocate(self, seq: Sequence):
        for block_id in reversed(seq.block_table):   #先释放末尾的 “独有块”（引用计数容易降为 0） 再处理可能被共享的 “前缀块”
            block = self.blocks[block_id]
            block.ref_count -= 1
            if block.ref_count == 0:
                self._deallocate_block(block_id)    #这里对应上面的block = self._allocate_block(block_id) 虽然这里清理了blocks里面的blockid 但是hash和 block.token_ids 还在
        seq.num_cached_tokens = 0
        seq.block_table.clear()
    
    # can_append 和 may_append 函数都是在decode阶段调用
    def can_append(self, seq: Sequence)-> bool:
        # 只有在 len(seq) % block_size == 0，并且有新的token需要空间时，也就是len(seq) % block_size的余数为1时，才需要一个 新的block块，
        # 因此这里的 条件是 len(seq) % self.block_size == 1，其次是 free_block_ids >= 1, 即保证有多的一块就行
        #[thinking] 这里给提供了一个思路 当局部代码看不懂时 去看看调用的地方  
        return len(self.free_block_ids) >= (len(seq) % self.block_size == 1)


    def may_append(self, seq: Sequence):    #prepare for append   核心作用：为序列（seq）追加新 token 做准备
        block_table = seq.block_table
        last_block = self.blocks[block_table[-1]]   #拿到最后一个block
        
        # __allocate_block是外部cpu端用于标记的，真正的 gpu 端已经提前分配好了该 block（如果不这样的话，动态分配VRAM会造成非常大的延迟），
        # 所以需要在 == 1时更新外部的标记block
        if len(seq) % self.block_size == 1:   #如果当前序列长度是block_size的整数倍+1 说明需要一个新块
            assert last_block.hash != -1
            block_id = self.free_block_ids[0]
            self._allocate_block(block_id)
            block_table.append(block_id)
        
        # 最后一个块在分配的时候，h是-1，没有计算哈希值写入字典用于缓存
        # 因此当最后一个块空间用光时，需要计算哈希值，用于前缀缓存
        elif len(seq) % self.block_size == 0:   #最后一个块刚被填满
            assert last_block.hash == -1 
            token_ids = seq.block(seq.num_blocks - 1)       #最后一个seq列表 因为从0开始计数
            prefix = self.blocks[block_table[-2]].hash if len(block_table) > 1 else -1  #这个边界条件很重要
            h = self.compute_hash(token_ids, prefix)
            last_block.update(h, token_ids)
            self.hash_to_block_id[h] = last_block.block_id
        else:   #最后一个块未填满，h是-1，没有计算哈希值写入字典用于缓存
            assert last_block.hash == -1
            