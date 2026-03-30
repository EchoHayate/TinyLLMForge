from collections import deque, OrderedDict
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
    

class CPUBlockAllocator:
    def __init__(self, num_blocks: int):
        self.num_blocks = num_blocks
        self.free_block_ids = list(range(num_blocks))
        
    def allocate(self) -> int:
        if not self.free_block_ids:
            raise MemoryError("No free CPU blocks available for swap!")
        return self.free_block_ids.pop()
        
    def deallocate(self, block_id: int):
        self.free_block_ids.append(block_id)
        
    def deallocate_many(self, block_ids: list[int]):
        self.free_block_ids.extend(block_ids)
        
    def can_allocate(self, num_blocks: int) -> bool:
        return len(self.free_block_ids) >= num_blocks

class BlockManager:
    def __init__(self, num_blocks: int, block_size: int):
        assert num_blocks > 0
        self.block_size = block_size
        self.blocks: list[Block] = [Block(i) for i in range(num_blocks)]   
        self.hash_to_block_id: dict[int, int] = dict()              
        
        # 使用 OrderedDict 管理空闲块，实现 LRU 淘汰策略
        # Key: block_id, Value: None (只利用 Order 属性)
        # 头部 (Front): 最旧的空闲块 (Least Recently Freed / Eviction Victim)
        # 尾部 (Back): 最近释放的块 (Most Recently Freed)
        self.free_block_ids: OrderedDict[int, None] = OrderedDict.fromkeys(range(num_blocks)) 

    #block只有占满的时候 才会计算hash
    # 以整数形式，返回计算出的哈希值
    @classmethod                           # 对标c++中的static, 第一个参数为类本身，cls, class self
    def compute_hash(cls, token_ids: list[int], prefix: int = -1):   #prefix表示是否依赖于上一个block的hash 若为-1 表示当前是第一个block 不需要
        h = xxhash.xxh64()
        if prefix != -1:
            h.update(prefix.to_bytes(8, "little"))   #小端字节序处理 prefix消除平台差异
        h.update(np.array(token_ids).tobytes())
        return h.intdigest()

    # 分配对应id的block, 重置状态
    def _allocate_block(self, block_id: int) -> Block:
        block = self.blocks[block_id]
        
        # [Fix] Clean up stale hash mapping if this block was previously cached
        if block.hash != -1:
            # Check if this block is actually the one mapped in hash_to_block_id
            # (In rare cases with hash collisions or logic errors, it might differ, but safe to check)
            if self.hash_to_block_id.get(block.hash) == block_id:
                del self.hash_to_block_id[block.hash]

        assert block.ref_count == 0
        block.reset()
        
        # Remove from free_blocks if present. 
        # Since this can be called on cache hit (block in free_blocks) or cache miss (popped from free_blocks),
        # we try to remove it cautiously or rely on caller logic.
        # Actually, in this refactor, let's assume caller handles removal from free_blocks logic OR we handle it here.
        # To be safe and idempotent:
        if block_id in self.free_block_ids:
             del self.free_block_ids[block_id]
             
        return self.blocks[block_id]

    # 将块从 “使用中” 状态转为 “空闲” 状态
    def _deallocate_block(self, block_id: int):
        assert self.blocks[block_id].ref_count == 0
        # Add to end of OrderedDict (Most Recently Freed)
        self.free_block_ids[block_id] = None

    # can_allocate 和 allocate 函数都是在prefill阶段调用
    # allocate, deallocate函数，都是针对一条 sequence 语句来说的
    def can_allocate(self, seq: Sequence) -> bool:
        # We can allocate if we have enough free blocks
        # Note: Cached blocks are technically "free" in free_block_ids until allocated.
        return len(self.free_block_ids) >= seq.num_blocks

    # allocate  blocks for the sequences, update the block table and hash table
    def allocate(self, seq: Sequence):    
        assert not seq.block_table
        h = -1
        for i in range(seq.num_blocks):
            token_ids = seq.block(i)     #token_ids：块中包含的 token 编号列表  核心作用是用于计算当前块的哈希值
            # 未填满的块（非完整块）的哈希值为 -1，不纳入缓存（因为复用价值低）
            h = self.compute_hash(token_ids, h) if len(token_ids) == self.block_size else -1   #计算hash的前提是当前block_size能被占满 如果占不满说明当前sequence结束
            block_id = self.hash_to_block_id.get(h, -1)
            
            # Cache Hit Condition:
            # 1. Block ID exists in hash map
            # 2. Block content matches (collision check)
            # 3. Block is EITHER in free_blocks (resurrecting cached block) OR already referenced (shared prefix)
            
            cache_hit = False
            if block_id != -1:
                block = self.blocks[block_id]
                if block.token_ids == token_ids:
                    cache_hit = True

            if cache_hit:
                seq.num_cached_tokens += self.block_size
                block = self.blocks[block_id]
                block.ref_count += 1
                
                # If block was in free list (cached but currently unused), remove it (resurrect)
                if block_id in self.free_block_ids:
                     del self.free_block_ids[block_id]
                     
            else: # Cache Miss
                # Allocation:
                # 1. If we have free blocks, take one.
                # 2. Strategy: Take from FRONT of OrderedDict (Oldest / Least Recently Used)
                if not self.free_block_ids:
                    raise MemoryError("No free blocks available!") # Should be checked by scheduler
                    
                block_id, _ = self.free_block_ids.popitem(last=False) # pop from front
                block = self._allocate_block(block_id)
            
            # Update Hash Mapping if valid
            if h != -1:
                block.update(h, token_ids)
                self.hash_to_block_id[h] = block.block_id  #使用实体去更新hash_to_block_id
                
            seq.block_table.append(block.block_id)

    def deallocate(self, seq: Sequence):
        for block_id in reversed(seq.block_table):
            block = self.blocks[block_id]
            block.ref_count -= 1
            if block.ref_count == 0:
                self._deallocate_block(block_id)
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
            block_id, _ = self.free_block_ids.popitem(last=False)
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

    def can_swap_out(self, seq: Sequence, cpu_allocator: CPUBlockAllocator) -> bool:
        return cpu_allocator.can_allocate(len(seq.block_table))

    def swap_out(self, seq: Sequence, cpu_allocator: CPUBlockAllocator) -> dict[int, int]:
        """将序列的所有物理块换出到CPU，释放GPU块。返回 GPU_ID -> CPU_ID 的映射关系。"""
        assert self.can_swap_out(seq, cpu_allocator)
        mapping = {}
        cpu_blocks = []
        for block_id in seq.block_table:
            cpu_block_id = cpu_allocator.allocate()
            cpu_blocks.append(cpu_block_id)
            mapping[block_id] = cpu_block_id
            
        self.deallocate(seq)
        seq.cpu_block_table = cpu_blocks
        return mapping

    def can_swap_in(self, seq: Sequence) -> bool:
        return len(self.free_block_ids) >= len(seq.cpu_block_table)

    def swap_in(self, seq: Sequence, cpu_allocator: CPUBlockAllocator) -> dict[int, int]:
        """从CPU读取并分配对应的GPU块，重组序列逻辑块，恢复Hash。返回 CPU_ID -> GPU_ID 的映射关系。"""
        assert self.can_swap_in(seq)
        mapping = {}
        seq.block_table = []
        seq.num_cached_tokens = 0
        
        h = -1
        # 按照顺序重新分配内存块，不走复用逻辑（保证私有），但是计算新Hash注册到系统，供未来别人复用
        for i, cpu_block_id in enumerate(seq.cpu_block_table):
            block_id, _ = self.free_block_ids.popitem(last=False)
            block = self._allocate_block(block_id)
            
            token_ids = seq.block(i)
            h = self.compute_hash(token_ids, h) if len(token_ids) == self.block_size else -1
            if h != -1:
                block.update(h, token_ids)
                self.hash_to_block_id[h] = block.block_id
                seq.num_cached_tokens += self.block_size
                
            block.ref_count = 1
            seq.block_table.append(block_id)
            mapping[cpu_block_id] = block_id
            
        cpu_allocator.deallocate_many(seq.cpu_block_table)
        seq.cpu_block_table = []
        return mapping
            