from collections import deque
import xxhash
import numpy as np

from tinyvllm.engine.sequence import Sequence

class Block:
    def __init__(self, block_id):           # 单个block块的属性
        self.block_id = block_id            # 块id
        self.ref_count = 0                  # 引用数量
        self.hash = -1                      # hash值
        self.token_ids = []                 # 包含的 token_id
    
    def update(self, hash: int, token_ids: list[int]):      # 更换该 block 的哈希值和所有 block_id
        self.hash = hash
        self.token_ids = token_ids

    def reset(self):                       # 重置 block状态，注意 ref_count 初始化为 1
        self.ref_count = 1
        self.hash = -1
        self.token_ids = []
    

class BlockManager:
    def __init__(self, num_blocks: int, block_size: int):
        assert num_blocks > 0
        self.block_size = block_size
        self.blocks: list[Block] = [Block(i) for i in range(num_blocks)]
        self.hash_to_block_id: dict[int, int] = dict()
        self.free_block_ids: deque[int] = deque(range(num_blocks))
        self.used_block_ids: set[int] = set()

    # 以整数形式，返回计算出的哈希值
    @classmethod                           # 对标c++中的static, 第一个参数为类本身，cls, class self
    def compute_hash(cls, token_ids: list[int], prefix: int = -1):
        h = xxhash.xxh64()
        if prefix != -1:
            h.update(prefix.to_bytes(8, "little"))
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

    def _deallocate_block(self, block_id: int) -> Block:
        assert self.blocks[block_id].ref_count == 0
        self.used_block_ids.remove(block_id)
        self.free_block_ids.append(block_id)

    # allocate, deallocate函数，都是针对一条 sequence 语句来说的
    def can_allocate(self, seq: Sequence) -> bool:
        return len(self.free_block_ids) >= seq.num_blocks
    
    def allocate(self, seq: Sequence):
        assert not seq.block_table
        h = -1
        cache_miss = False
        for i in range(seq.num_blocks):
            token_ids = seq.block(i)
            h = self.compute_hash(token_ids, h) if len(token_ids) == self.block_size else -1
            block_id = self.hash_to_block_id.get(h, -1)
            # 没有缓存或者缓存未命中
            if block_id == -1 or self.blocks[block_id].token_ids != token_ids:
                cache_miss = True
            
            # 没有缓存或者缓存为命中，那么就从空闲块表的头部，取出一块进行分配
            if cache_miss:
                block_id = self.free_block_ids[0]
                block = self._allocate_block(block_id)
            # 缓存命中
            else:
                seq.num_cached_tokens += self.block_size
                if block_id in self.used_block_ids:
                    block = self.blocks[block_id]
                    block.ref_count += 1
                # 由于deallocate 并没有清除字典的hash， 也没有清除 block.token_id列表。
                # 因此通过字典映射的 block_id， 可能已经被_deallocate了，但是由于 token_id还在，因此也可以用于缓存
                # 所以需要 _allocate_block回来
                else:
                    block = self._allocate_block(block_id)

            if h != -1:
                block.update(h, token_ids)
                self.hash_to_block_id[h] = block_id
            seq.block_table.append(block_id)
            
    def deallocate(self, seq: Sequence):
        for block_id in reversed(seq.block_table):
            block = self.blocks[block_id]
            block.ref_count -= 1
            if block.ref_count == 0:
                self._deallocate_block(block_id)
        seq.num_cached_tokens = 0
        seq.block_table.clear()
            
    def can_append(self, seq: Sequence)-> bool:
        # 只有在 len(seq) % block_size == 0，并且有新的token需要空间时，才需要一个 新的block块，
        # 因此这里的 条件是 len(seq) % self.block_size == 1，其次是 free_block_ids >= 1, 即保证有多的一块就行
        return len(self.free_block_ids) >= (len(seq) % self.block_size == 1)


    def may_append(self, seq: Sequence):
        block_table = seq.block_table
        last_block = self.blocks[block_table[-1]]
        
        # __allocate_block是外部cpu端用于标记的，真正的 gpu 端已经提前分配好了该 block，
        # 所以需要在 == 1时更新外部的标记block
        if len(seq) % self.block_size == 1:
            assert last_block.hash != -1
            block_id = self.free_block_ids[0]
            self._allocate_block(block_id)
            block_table.append(block_id)
        
        # 最后一个块在分配的时候，h是-1，没有计算哈希值写入字典用于缓存
        # 因此当最后一个块空间用光时，需要计算哈希值，用于前缀缓存
        elif len(seq) % self.block_size == 0:
            assert last_block.hash == -1
            token_ids = seq.block(seq.num_blocks - 1)
            prefix = self.blocks[block_table[-2]].hash if len(block_table) > 1 else -1
            h = self.compute_hash(token_ids, prefix)
            last_block.update(h, token_ids)
            self.hash_to_block_id[h] = last_block.block_id
        else:
            assert last_block.hash == -1
            