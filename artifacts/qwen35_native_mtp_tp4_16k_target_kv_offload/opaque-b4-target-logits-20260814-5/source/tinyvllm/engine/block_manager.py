from collections import deque
from dataclasses import dataclass
import hashlib
from typing import Optional
import xxhash
#[thinking] 这里是把token_ids转换成hash值 然后去做处理 这样做：用哈希处理 token_ids 能减轻 KV 缓存负担  有没有别的方式呢
import numpy as np

from tinyvllm.engine.sequence import Sequence


@dataclass
class PrefixBlockReservation:
    block_ids: tuple[int, ...]
    block_identities: tuple[tuple[int, int, int], ...]
    token_count: int
    owner_count: int
    state: str = "reserved"


@dataclass
class SequenceBlockReservation:
    block_ids: tuple[int, ...]
    block_identities: tuple[tuple[int, int, int], ...]
    cached_tokens: int
    prefix_block_count: int
    new_block_count: int
    state: str = "reserved"


@dataclass
class SpeculativeKVTransaction:
    sequence_id: int
    original_num_tokens: int
    original_last_token: int
    original_block_table: tuple[int, ...]
    original_block_generations: tuple[int, ...]
    reserved_block_ids: tuple[int, ...]
    reserved_block_generations: tuple[int, ...]
    proposed_token_count: int
    materialized_token_count: int = 0
    state: str = "reserved"


@dataclass(frozen=True)
class SpeculativeKVTransactionAuthorization:
    sequence_id: int
    original_num_tokens: int
    proposed_token_count: int
    materialized_token_count: int
    state: str
    original_block_identities: tuple[tuple[int, int], ...]
    reserved_block_identities: tuple[tuple[int, int], ...]
    authorization_sha256: str


@dataclass(frozen=True)
class SpeculativeKVCachePublication:
    block_id: int
    block_hash: int
    token_ids: tuple[int, ...]


@dataclass(frozen=True)
class SpeculativeKVCommitPlan:
    sequence_id: int
    sequence: Sequence
    transaction: SpeculativeKVTransaction
    accepted_tokens: tuple[int, ...]
    committed_block_ids: tuple[int, ...]
    unused_block_ids: tuple[int, ...]
    materialized_end: int
    publications: tuple[SpeculativeKVCachePublication, ...]


class Block:
    def __init__(self, block_id):           # 单个block块的属性
        self.block_id = block_id            # 块id
        self.ref_count = 0                  # 引用数量 主要涉及相同前缀的引用
        self.hash = -1                      # hash值 用于比较block大小  -1表示无效 
        self.token_ids = []                 # 包含的 token_id
        self.generation = 0
    
    def update(self, hash: int, token_ids: list[int]):      # 记录该 block 的哈希值和所有 block_id
        self.hash = hash
        self.token_ids = token_ids

    def reset(self):                       # 重置 block状态，注意 ref_count 初始化为 1  为下一次BlockManager.allocate()做准备
        self.generation += 1
        self.ref_count = 1
        self.hash = -1
        self.token_ids = []
    

class BlockManager:
    def __init__(self, num_blocks: int, block_size: int):
        assert num_blocks > 0
        self.block_size = block_size
        self.blocks: list[Block] = [Block(i) for i in range(num_blocks)]   
        self.hash_to_block_id: dict[int, int] = dict()              #键代表某个block的token序列的哈希值 值代表这个哈希值对应的kv缓存块的id（block_id） 用于快速查找和复用内容相同的 KV 缓存块
        self.hash_to_block_ids: dict[int, set[int]] = dict()
        self.free_block_ids: deque[int] = deque(range(num_blocks))  #双向队列分配和回收元素
        self.used_block_ids: set[int] = set()                       #跟踪所有正在被使用的block_id 查找的时间复杂度O（1） 如果使用deque 查找的时间复杂度为O（n） 

    def block_identities(
        self,
        block_ids: tuple[int, ...],
    ) -> tuple[tuple[int, int], ...]:
        if not isinstance(block_ids, tuple):
            raise ValueError("block_ids must be a tuple")
        if len(set(block_ids)) != len(block_ids):
            raise ValueError("block_ids must be unique")
        identities = []
        for block_id in block_ids:
            if (
                isinstance(block_id, bool)
                or not isinstance(block_id, int)
                or block_id < 0
                or block_id >= len(self.blocks)
            ):
                raise ValueError("block id is out of range")
            block = self.blocks[block_id]
            if (
                block_id not in self.used_block_ids
                or block.ref_count <= 0
            ):
                raise RuntimeError(
                    "block identity ownership is stale"
                )
            identities.append((block_id, block.generation))
        return tuple(identities)

#block只有占满的时候 才会计算hash
    # 以整数形式，返回计算出的哈希值
    @classmethod                           # 对标c++中的static, 第一个参数为类本身，cls, class self
    def compute_hash(cls, token_ids: list[int], prefix: int = -1):   #prefix表示是否依赖于上一个block的hash 若为-1 表示当前是第一个block 不需要
        h = xxhash.xxh64()
        if prefix != -1:
            h.update(prefix.to_bytes(8, "little"))   #小端字节序处理 prefix消除平台差异
        h.update(np.array(token_ids).tobytes())
        return h.intdigest()

    def _register_cached_block(
        self,
        block_id: int,
        h: int,
        token_ids: list[int],
    ):
        block = self.blocks[block_id]
        block.update(h, token_ids)
        self.hash_to_block_ids.setdefault(h, set()).add(block_id)
        self.hash_to_block_id[h] = block_id

    def _unregister_cached_block(self, block_id: int):
        block = self.blocks[block_id]
        h = block.hash
        if h == -1:
            return
        block_ids = self.hash_to_block_ids.get(h)
        if block_ids is not None:
            block_ids.discard(block_id)
            if not block_ids:
                del self.hash_to_block_ids[h]
            elif self.hash_to_block_id.get(h) == block_id:
                self.hash_to_block_id[h] = next(iter(block_ids))
        if (
            self.hash_to_block_id.get(h) == block_id
            and h not in self.hash_to_block_ids
        ):
            del self.hash_to_block_id[h]

    def _find_cached_block_id(
        self,
        h: int,
        token_ids: list[int],
    ) -> int:
        primary = self.hash_to_block_id.get(h, -1)
        if (
            primary != -1
            and self.blocks[primary].token_ids == token_ids
        ):
            return primary
        for block_id in self.hash_to_block_ids.get(h, ()):
            if self.blocks[block_id].token_ids == token_ids:
                self.hash_to_block_id[h] = block_id
                return block_id
        return -1

    # 分配对应id的block, 重置状态，并且更新 free_block_ids 队列 和 used_block_ids 集合
    def _allocate_block(self, block_id: int) -> Block:
        block = self.blocks[block_id]
        assert block.ref_count == 0
        self._unregister_cached_block(block_id)
        block.reset()
        self.free_block_ids.remove(block_id)
        self.used_block_ids.add(block_id)
        return self.blocks[block_id]

    def _activate_cached_block(
        self,
        block_id: int,
        owner_count: int = 1,
    ) -> Block:
        block = self.blocks[block_id]
        assert block.ref_count == 0
        assert block.hash != -1
        assert owner_count > 0
        block.ref_count = owner_count
        self.free_block_ids.remove(block_id)
        self.used_block_ids.add(block_id)
        return block

    # [warning!] 隐含错误 定义了返回类型 但是没有return 实际返回 None
    #将块从 “使用中” 状态转为 “空闲” 状态（例如从 used_block_ids 移到 free_block_ids），但不会清除该块的哈希映射（hash_to_block_id 中 h→block_id 的关联）和块本身存储的 token_ids。
    def _deallocate_block(self, block_id: int) -> Block:
        assert self.blocks[block_id].ref_count == 0
        self.used_block_ids.remove(block_id)
        self.free_block_ids.append(block_id)

    # can_allocate 和 allocate 函数都是在prefill阶段调用
    # allocate, deallocate函数，都是针对一条 sequence 语句来说的
    def _reusable_prefix_block_ids(self, seq: Sequence) -> list[int]:
        max_cached_blocks = self.max_reusable_tokens(seq) // self.block_size
        block_ids = []
        h = -1
        for i in range(min(seq.num_blocks, max_cached_blocks)):
            token_ids = seq.block(i)
            if len(token_ids) != self.block_size:
                break
            h = self.compute_hash(token_ids, h)
            block_id = self._find_cached_block_id(h, token_ids)
            if block_id == -1:
                break
            block_ids.append(block_id)
        return block_ids

    def estimate_admission(self, seq: Sequence) -> tuple[int, int]:
        reusable_block_ids = self._reusable_prefix_block_ids(seq)
        live_prefix_blocks = sum(
            block_id in self.used_block_ids
            for block_id in reusable_block_ids
        )
        reusable_tokens = len(reusable_block_ids) * self.block_size
        required_free_blocks = seq.num_blocks - live_prefix_blocks
        return reusable_tokens, required_free_blocks

    def can_allocate(self, seq: Sequence) -> bool:
        _, required_free_blocks = self.estimate_admission(seq)
        return len(self.free_block_ids) >= required_free_blocks

    def max_reusable_tokens(self, seq: Sequence) -> int:
        """Return the full-block prefix cap that leaves one query token."""
        if len(seq) <= 1:
            return 0
        return ((len(seq) - 1) // self.block_size) * self.block_size

    # allocate  blocks for the sequences, update the block table and hash table
    def allocate(
        self,
        seq: Sequence,
        publish_hashes: bool = True,
        max_cached_tokens: Optional[int] = None,
    ):
        assert not seq.block_table
        if max_cached_tokens is None:
            max_cached_tokens = len(seq)
        max_cached_tokens = max(
            0,
            min(int(max_cached_tokens), len(seq)),
        )
        max_cached_blocks = max_cached_tokens // self.block_size
        h = -1
        cache_miss = False
        for i in range(seq.num_blocks):
            token_ids = seq.block(i)     #token_ids：块中包含的 token 编号列表  核心作用是用于计算当前块的哈希值
            # 未填满的块（非完整块）的哈希值为 -1，不纳入缓存（因为复用价值低）
            h = self.compute_hash(token_ids, h) if len(token_ids) == self.block_size else -1   #计算hash的前提是当前block_size能被占满 如果占不满说明当前sequence结束
            block_id = (
                self._find_cached_block_id(h, token_ids)
                if i < max_cached_blocks
                else -1
            )
            # 没有缓存或者缓存未命中
            if block_id == -1:
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
                    block = self._activate_cached_block(block_id)
                # chunked prefill 关闭 publish_hashes 只是不发布“未计算的新 block”。
                # 对 prefix-cache 命中的 block，KV 已经存在，必须恢复 hash/token_ids 元数据，
                # 否则后续 commit_prefill 计算下一块 hash 时 prefix 链会断。
                if h != -1:
                    self._register_cached_block(block_id, h, token_ids)

            if h != -1 and publish_hashes:      #相同的 token_ids 序列（通过哈希 h 标识）始终对应到同一个 block_id
                self._register_cached_block(block_id, h, token_ids)
            seq.block_table.append(block_id)
        seq.num_computed_tokens = seq.num_cached_tokens

    @staticmethod
    def _validate_prefix_reservation(
        reservation: PrefixBlockReservation,
    ) -> None:
        if not isinstance(reservation, PrefixBlockReservation):
            raise ValueError(
                "reservation must be a PrefixBlockReservation"
            )

    def reserve_exact_prefix(
        self,
        token_ids: tuple[int, ...],
        *,
        owner_count: int = 1,
    ) -> Optional[PrefixBlockReservation]:
        if not isinstance(token_ids, tuple):
            raise ValueError("token_ids must be a tuple")
        if not token_ids or len(token_ids) % self.block_size != 0:
            raise ValueError(
                "token_ids must be positive and full-block aligned"
            )
        if any(
            isinstance(token_id, bool)
            or not isinstance(token_id, int)
            or token_id < 0
            for token_id in token_ids
        ):
            raise ValueError(
                "token_ids must contain non-negative integers"
            )
        if (
            isinstance(owner_count, bool)
            or not isinstance(owner_count, int)
            or owner_count <= 0
        ):
            raise ValueError("owner_count must be a positive integer")

        block_ids = []
        block_hashes = []
        prefix_hash = -1
        for start in range(0, len(token_ids), self.block_size):
            block_tokens = list(
                token_ids[start:start + self.block_size]
            )
            prefix_hash = self.compute_hash(
                block_tokens,
                prefix_hash,
            )
            block_id = self._find_cached_block_id(
                prefix_hash,
                block_tokens,
            )
            if block_id == -1 or block_id in block_ids:
                return None
            block_ids.append(block_id)
            block_hashes.append(prefix_hash)

        acquired = []
        try:
            for block_id in block_ids:
                block = self.blocks[block_id]
                if block_id in self.used_block_ids:
                    block.ref_count += owner_count
                else:
                    self._activate_cached_block(
                        block_id,
                        owner_count=owner_count,
                    )
                acquired.append(block_id)
        except BaseException:
            self._release_prefix_references(
                acquired,
                owner_count,
            )
            raise

        identities = tuple(
            (
                block_id,
                self.blocks[block_id].generation,
                block_hash,
            )
            for block_id, block_hash in zip(
                block_ids,
                block_hashes,
            )
        )
        return PrefixBlockReservation(
            block_ids=tuple(block_ids),
            block_identities=identities,
            token_count=len(token_ids),
            owner_count=owner_count,
        )

    def _release_prefix_references(
        self,
        block_ids,
        owner_count: int,
    ) -> None:
        for block_id in reversed(tuple(block_ids)):
            block = self.blocks[block_id]
            if block.ref_count < owner_count:
                raise RuntimeError(
                    "prefix reservation refcount underflow"
                )
            block.ref_count -= owner_count
            if block.ref_count == 0:
                self._deallocate_block(block_id)

    def attach_prefix_reservation(
        self,
        reservation: PrefixBlockReservation,
        sequences: tuple[Sequence, ...],
    ) -> None:
        self._validate_prefix_reservation(reservation)
        if reservation.state != "reserved":
            raise RuntimeError(
                "prefix reservation is not attachable: "
                f"{reservation.state}"
            )
        if not isinstance(sequences, tuple):
            raise ValueError("sequences must be a tuple")
        if len(sequences) != reservation.owner_count:
            raise ValueError(
                "sequence count must match reservation owner count"
            )
        if any(not isinstance(seq, Sequence) for seq in sequences):
            raise ValueError("sequences must contain Sequence values")
        if len({id(seq) for seq in sequences}) != len(sequences):
            raise ValueError("sequences must contain unique objects")
        if len({seq.seq_id for seq in sequences}) != len(sequences):
            raise ValueError("sequences must contain unique seq_id values")
        for seq in sequences:
            if (
                seq.block_table
                or seq.num_cached_tokens != 0
                or seq.num_computed_tokens != 0
            ):
                raise ValueError(
                    "destination sequence already owns KV metadata"
                )

        block_table = list(reservation.block_ids)
        for seq in sequences:
            seq.block_table = list(block_table)
            seq.num_cached_tokens = reservation.token_count
            seq.num_computed_tokens = reservation.token_count
        reservation.state = "attached"

    def release_prefix_reservation(
        self,
        reservation: PrefixBlockReservation,
    ) -> None:
        self._validate_prefix_reservation(reservation)
        if reservation.state != "reserved":
            raise RuntimeError(
                "prefix reservation is not releasable: "
                f"{reservation.state}"
            )
        self._release_prefix_references(
            reservation.block_ids,
            reservation.owner_count,
        )
        reservation.state = "released"

    @staticmethod
    def _validate_sequence_reservation(
        reservation: SequenceBlockReservation,
    ) -> None:
        if not isinstance(reservation, SequenceBlockReservation):
            raise ValueError(
                "reservation must be a SequenceBlockReservation"
            )

    def _validate_sequence_reservation_structure(
        self,
        reservation: SequenceBlockReservation,
    ) -> None:
        self._validate_sequence_reservation(reservation)
        for value, name in (
            (reservation.cached_tokens, "cached_tokens"),
            (reservation.prefix_block_count, "prefix_block_count"),
            (reservation.new_block_count, "new_block_count"),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 0
            ):
                raise ValueError(
                    f"reservation {name} must be a non-negative integer"
                )
        if not isinstance(reservation.block_ids, tuple):
            raise ValueError(
                "reservation block_ids must be a tuple"
            )
        if not reservation.block_ids:
            raise ValueError(
                "reservation block_ids must be non-empty"
            )
        if any(
            isinstance(block_id, bool)
            or not isinstance(block_id, int)
            or block_id < 0
            or block_id >= len(self.blocks)
            for block_id in reservation.block_ids
        ):
            raise ValueError(
                "reservation block id is out of range"
            )
        if len(set(reservation.block_ids)) != len(
            reservation.block_ids
        ):
            raise ValueError(
                "reservation block ids must be unique"
            )
        if not isinstance(reservation.block_identities, tuple):
            raise ValueError(
                "reservation block_identities must be a tuple"
            )
        if (
            reservation.prefix_block_count
            + reservation.new_block_count
            != len(reservation.block_ids)
        ):
            raise ValueError(
                "reservation block counts are inconsistent"
            )
        if reservation.cached_tokens != (
            reservation.prefix_block_count * self.block_size
        ):
            raise ValueError(
                "reservation cached token count is inconsistent"
            )
        if len(reservation.block_identities) != (
            reservation.prefix_block_count
        ):
            raise ValueError(
                "reservation prefix identity count is inconsistent"
            )

    def reserve_sequence_blocks(
        self,
        seq: Sequence,
        *,
        max_cached_tokens: Optional[int] = None,
    ) -> SequenceBlockReservation:
        if not isinstance(seq, Sequence):
            raise ValueError("seq must be a Sequence")
        if (
            seq.block_table
            or seq.num_cached_tokens != 0
            or seq.num_computed_tokens != 0
        ):
            raise ValueError(
                "sequence already owns KV metadata"
            )
        if max_cached_tokens is None:
            max_cached_tokens = self.max_reusable_tokens(seq)
        if (
            isinstance(max_cached_tokens, bool)
            or not isinstance(max_cached_tokens, int)
        ):
            raise ValueError(
                "max_cached_tokens must be an integer"
            )
        max_cached_tokens = max(
            0,
            min(max_cached_tokens, len(seq)),
        )
        max_cached_blocks = max_cached_tokens // self.block_size

        prefix_block_ids = []
        prefix_hashes = []
        prefix_hash = -1
        for block_index in range(
            min(seq.num_blocks, max_cached_blocks)
        ):
            block_tokens = seq.block(block_index)
            if len(block_tokens) != self.block_size:
                break
            prefix_hash = self.compute_hash(
                block_tokens,
                prefix_hash,
            )
            block_id = self._find_cached_block_id(
                prefix_hash,
                block_tokens,
            )
            if (
                block_id == -1
                or block_id in prefix_block_ids
            ):
                break
            prefix_block_ids.append(block_id)
            prefix_hashes.append(prefix_hash)

        prefix_block_count = len(prefix_block_ids)
        new_block_count = seq.num_blocks - prefix_block_count
        live_prefix_blocks = sum(
            block_id in self.used_block_ids
            for block_id in prefix_block_ids
        )
        required_free_blocks = (
            seq.num_blocks - live_prefix_blocks
        )
        if len(self.free_block_ids) < required_free_blocks:
            raise RuntimeError(
                "insufficient KV blocks for sequence reservation"
            )

        free_before = tuple(self.free_block_ids)
        suffix_candidate_ids = []
        remaining_free_ids = list(self.free_block_ids)
        prefix_idle_ids = set(
            prefix_block_ids
        ).difference(self.used_block_ids)
        for block_id in remaining_free_ids:
            if block_id in prefix_idle_ids:
                continue
            suffix_candidate_ids.append(block_id)
            if len(suffix_candidate_ids) == new_block_count:
                break
        suffix_block_snapshots = {
            block_id: (
                self.blocks[block_id].ref_count,
                self.blocks[block_id].generation,
                self.blocks[block_id].hash,
                list(self.blocks[block_id].token_ids),
            )
            for block_id in suffix_candidate_ids
        }
        affected_hashes = {
            block_hash
            for _, _, block_hash, _ in suffix_block_snapshots.values()
            if block_hash != -1
        }
        hash_index_snapshots = {
            block_hash: (
                (
                    set(self.hash_to_block_ids[block_hash])
                    if block_hash in self.hash_to_block_ids
                    else None
                ),
                self.hash_to_block_id.get(block_hash),
            )
            for block_hash in affected_hashes
        }
        acquired_prefix_ids = []
        new_block_ids = []
        try:
            for block_id in prefix_block_ids:
                block = self.blocks[block_id]
                if block_id in self.used_block_ids:
                    block.ref_count += 1
                else:
                    self._activate_cached_block(block_id)
                acquired_prefix_ids.append(block_id)
            for _ in range(new_block_count):
                block_id = self.free_block_ids[0]
                self._allocate_block(block_id)
                new_block_ids.append(block_id)
        except BaseException:
            self._release_prefix_references(new_block_ids, 1)
            self._release_prefix_references(
                acquired_prefix_ids,
                1,
            )
            for block_id in suffix_block_snapshots:
                self.used_block_ids.discard(block_id)
            for block_id, (
                ref_count,
                generation,
                block_hash,
                token_ids,
            ) in suffix_block_snapshots.items():
                block = self.blocks[block_id]
                block.ref_count = ref_count
                block.generation = generation
                block.hash = block_hash
                block.token_ids = token_ids
            for block_hash, (
                hash_block_ids,
                primary_block_id,
            ) in hash_index_snapshots.items():
                if hash_block_ids is None:
                    self.hash_to_block_ids.pop(block_hash, None)
                else:
                    self.hash_to_block_ids[block_hash] = hash_block_ids
                if primary_block_id is None:
                    self.hash_to_block_id.pop(block_hash, None)
                else:
                    self.hash_to_block_id[block_hash] = primary_block_id
            self.free_block_ids = deque(free_before)
            raise

        identities = tuple(
            (
                block_id,
                self.blocks[block_id].generation,
                block_hash,
            )
            for block_id, block_hash in zip(
                prefix_block_ids,
                prefix_hashes,
            )
        )
        return SequenceBlockReservation(
            block_ids=tuple(prefix_block_ids + new_block_ids),
            block_identities=identities,
            cached_tokens=prefix_block_count * self.block_size,
            prefix_block_count=prefix_block_count,
            new_block_count=new_block_count,
        )

    def attach_sequence_reservation(
        self,
        reservation: SequenceBlockReservation,
        seq: Sequence,
    ) -> None:
        self._validate_sequence_reservation_structure(reservation)
        if reservation.state != "reserved":
            raise RuntimeError(
                "sequence reservation is not attachable: "
                f"{reservation.state}"
            )
        if not isinstance(seq, Sequence):
            raise ValueError("seq must be a Sequence")
        if (
            seq.block_table
            or seq.num_cached_tokens != 0
            or seq.num_computed_tokens != 0
        ):
            raise ValueError(
                "destination sequence already owns KV metadata"
            )
        if len(reservation.block_ids) != seq.num_blocks:
            raise ValueError(
                "reservation block count must match sequence"
            )
        if reservation.cached_tokens > self.max_reusable_tokens(seq):
            raise ValueError(
                "reservation exceeds sampleable prefix cap"
            )
        prefix_hash = -1
        for block_index, identity in enumerate(
            reservation.block_identities
        ):
            if not isinstance(identity, tuple) or len(identity) != 3:
                raise ValueError(
                    "reservation block identity is malformed"
                )
            block_id, generation, block_hash = identity
            if reservation.block_ids[block_index] != block_id:
                raise ValueError(
                    "reservation identity order is inconsistent"
                )
            block_tokens = seq.block(block_index)
            prefix_hash = self.compute_hash(
                block_tokens,
                prefix_hash,
            )
            block = self.blocks[block_id]
            if (
                block.generation != generation
                or block.hash != block_hash
                or block_hash != prefix_hash
                or block.token_ids != block_tokens
            ):
                raise RuntimeError(
                    "sequence reservation prefix identity is stale"
                )
        for block_id in reservation.block_ids:
            block = self.blocks[block_id]
            if (
                block_id not in self.used_block_ids
                or block.ref_count <= 0
            ):
                raise RuntimeError(
                    "sequence reservation block ownership is stale"
                )

        seq.block_table = list(reservation.block_ids)
        seq.num_cached_tokens = reservation.cached_tokens
        seq.num_computed_tokens = reservation.cached_tokens
        reservation.state = "attached"

    def release_sequence_reservation(
        self,
        reservation: SequenceBlockReservation,
    ) -> None:
        self._validate_sequence_reservation_structure(reservation)
        if reservation.state != "reserved":
            raise RuntimeError(
                "sequence reservation is not releasable: "
                f"{reservation.state}"
            )
        for block_id in reservation.block_ids:
            block = self.blocks[block_id]
            if (
                block_id not in self.used_block_ids
                or block.ref_count <= 0
            ):
                raise RuntimeError(
                    "sequence reservation block ownership is stale"
                )
        self._release_prefix_references(
            reservation.block_ids,
            1,
        )
        reservation.state = "released"

    def clear_reusable_cache(self) -> int:
        """Drop only idle prefix metadata; never mutate live blocks."""
        self.hash_to_block_id.clear()
        self.hash_to_block_ids.clear()
        cleared = 0
        for block in self.blocks:
            if block.ref_count != 0:
                if block.hash != -1:
                    self._register_cached_block(
                        block.block_id,
                        block.hash,
                        block.token_ids,
                    )
                continue
            if block.hash != -1 or block.token_ids:
                block.hash = -1
                block.token_ids = []
                cleared += 1
        return cleared

    def allocate_ephemeral(self, seq: Sequence):
        """Allocate scratch KV blocks without prefix-cache lookup or publication.

        This is used by speculative target-verification dry-runs. The temporary
        sequence may write KV into these blocks during a prefill forward, but the
        blocks are not attached to any live request and no hash entry is
        published, so deallocating the sequence makes the scratch KV unreachable.
        """
        assert not seq.block_table
        assert len(self.free_block_ids) >= seq.num_blocks
        for _ in range(seq.num_blocks):
            block_id = self.free_block_ids[0]
            self._allocate_block(block_id)
            seq.block_table.append(block_id)
        seq.num_cached_tokens = 0
        seq.num_computed_tokens = 0

    def _validate_speculative_kv_transaction_structure(
        self,
        transaction: SpeculativeKVTransaction,
    ) -> None:
        if not isinstance(transaction, SpeculativeKVTransaction):
            raise ValueError(
                "transaction must be a SpeculativeKVTransaction"
            )
        for value, name, minimum in (
            (transaction.sequence_id, "sequence_id", 0),
            (
                transaction.original_num_tokens,
                "original_num_tokens",
                1,
            ),
            (
                transaction.proposed_token_count,
                "proposed_token_count",
                1,
            ),
            (
                transaction.materialized_token_count,
                "materialized_token_count",
                0,
            ),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < minimum
            ):
                raise ValueError(
                    f"transaction {name} must be an integer >= {minimum}"
                )
        if (
            isinstance(transaction.original_last_token, bool)
            or not isinstance(transaction.original_last_token, int)
        ):
            raise ValueError(
                "transaction original_last_token must be an integer"
            )
        if not isinstance(transaction.original_block_table, tuple):
            raise ValueError(
                "transaction original_block_table must be a tuple"
            )
        if not isinstance(
            transaction.original_block_generations,
            tuple,
        ):
            raise ValueError(
                "transaction original_block_generations must be a tuple"
            )
        if len(transaction.original_block_table) != len(
            transaction.original_block_generations
        ):
            raise ValueError(
                "transaction original block identities are inconsistent"
            )
        if not isinstance(transaction.reserved_block_ids, tuple):
            raise ValueError(
                "transaction reserved_block_ids must be a tuple"
            )
        if not isinstance(
            transaction.reserved_block_generations,
            tuple,
        ):
            raise ValueError(
                "transaction reserved_block_generations must be a tuple"
            )
        if len(transaction.reserved_block_ids) != len(
            transaction.reserved_block_generations
        ):
            raise ValueError(
                "transaction reserved block identities are inconsistent"
            )
        if len(set(transaction.reserved_block_ids)) != len(
            transaction.reserved_block_ids
        ):
            raise ValueError(
                "transaction reserved block ids must be unique"
            )
        if len(set(transaction.original_block_table)) != len(
            transaction.original_block_table
        ):
            raise ValueError(
                "transaction original block ids must be unique"
            )
        if set(transaction.original_block_table).intersection(
            transaction.reserved_block_ids
        ):
            raise ValueError(
                "transaction original and reserved block ids overlap"
            )
        for block_id in (
            transaction.original_block_table
            + transaction.reserved_block_ids
        ):
            if (
                isinstance(block_id, bool)
                or not isinstance(block_id, int)
                or block_id < 0
                or block_id >= len(self.blocks)
            ):
                raise ValueError(
                    "transaction block id is out of range"
                )
        for generation in (
            transaction.original_block_generations
            + transaction.reserved_block_generations
        ):
            if (
                isinstance(generation, bool)
                or not isinstance(generation, int)
                or generation < 0
            ):
                raise ValueError(
                    "transaction block generation is invalid"
                )
        if transaction.materialized_token_count > (
            transaction.proposed_token_count - 1
        ):
            raise ValueError(
                "transaction materialized token count exceeds proposal"
            )
        if transaction.state not in (
            "reserved",
            "materialized",
            "committed",
            "rolled_back",
        ):
            raise ValueError("transaction state is invalid")

    def _validate_speculative_reserved_blocks(
        self,
        transaction: SpeculativeKVTransaction,
    ) -> None:
        for block_id, generation in zip(
            transaction.reserved_block_ids,
            transaction.reserved_block_generations,
        ):
            block = self.blocks[block_id]
            if (
                block_id not in self.used_block_ids
                or block.generation != generation
                or block.ref_count != 1
                or block.hash != -1
                or block.token_ids
            ):
                raise RuntimeError(
                    "speculative KV transaction block ownership is stale"
                )

    def begin_speculative_kv_transaction(
        self,
        seq: Sequence,
        proposed_token_count: int,
    ) -> SpeculativeKVTransaction:
        if not isinstance(seq, Sequence):
            raise ValueError("seq must be a Sequence")
        if (
            isinstance(proposed_token_count, bool)
            or not isinstance(proposed_token_count, int)
            or proposed_token_count <= 0
        ):
            raise ValueError(
                "proposed_token_count must be a positive integer"
            )
        if (
            isinstance(seq.seq_id, bool)
            or not isinstance(seq.seq_id, int)
            or seq.seq_id < 0
        ):
            raise ValueError(
                "sequence must have a non-negative integer seq_id"
            )
        if len(seq) <= 0:
            raise ValueError("sequence must contain at least one token")
        if not seq.block_table:
            raise ValueError("sequence must own a KV block table")
        if len(set(seq.block_table)) != len(seq.block_table):
            raise ValueError(
                "sequence block table must contain unique block ids"
            )
        for block_id in seq.block_table:
            if (
                isinstance(block_id, bool)
                or not isinstance(block_id, int)
                or block_id < 0
                or block_id >= len(self.blocks)
            ):
                raise ValueError("sequence block id is out of range")
            block = self.blocks[block_id]
            if (
                block_id not in self.used_block_ids
                or block.ref_count <= 0
            ):
                raise RuntimeError(
                    "sequence KV block ownership is stale"
                )

        materialized_end = len(seq) + max(
            0,
            proposed_token_count - 1,
        )
        required_blocks = (
            materialized_end + self.block_size - 1
        ) // self.block_size
        missing_blocks = max(
            0,
            required_blocks - len(seq.block_table),
        )
        if len(self.free_block_ids) < missing_blocks:
            raise RuntimeError(
                "insufficient KV blocks for speculative transaction"
            )

        free_before = tuple(self.free_block_ids)
        candidate_ids = tuple(
            list(self.free_block_ids)[:missing_blocks]
        )
        block_snapshots = {
            block_id: (
                self.blocks[block_id].ref_count,
                self.blocks[block_id].generation,
                self.blocks[block_id].hash,
                list(self.blocks[block_id].token_ids),
            )
            for block_id in candidate_ids
        }
        affected_hashes = {
            block_hash
            for _, _, block_hash, _ in block_snapshots.values()
            if block_hash != -1
        }
        hash_index_snapshots = {
            block_hash: (
                (
                    set(self.hash_to_block_ids[block_hash])
                    if block_hash in self.hash_to_block_ids
                    else None
                ),
                self.hash_to_block_id.get(block_hash),
            )
            for block_hash in affected_hashes
        }
        reserved_block_ids = []
        try:
            for _ in range(missing_blocks):
                block_id = self.free_block_ids[0]
                self._allocate_block(block_id)
                reserved_block_ids.append(block_id)
        except BaseException:
            for block_id in candidate_ids:
                self.used_block_ids.discard(block_id)
            for block_id, (
                ref_count,
                generation,
                block_hash,
                token_ids,
            ) in block_snapshots.items():
                block = self.blocks[block_id]
                block.ref_count = ref_count
                block.generation = generation
                block.hash = block_hash
                block.token_ids = token_ids
            for block_hash, (
                hash_block_ids,
                primary_block_id,
            ) in hash_index_snapshots.items():
                if hash_block_ids is None:
                    self.hash_to_block_ids.pop(block_hash, None)
                else:
                    self.hash_to_block_ids[block_hash] = hash_block_ids
                if primary_block_id is None:
                    self.hash_to_block_id.pop(block_hash, None)
                else:
                    self.hash_to_block_id[block_hash] = primary_block_id
            self.free_block_ids = deque(free_before)
            raise

        return SpeculativeKVTransaction(
            sequence_id=seq.seq_id,
            original_num_tokens=len(seq),
            original_last_token=int(seq.last_token),
            original_block_table=tuple(seq.block_table),
            original_block_generations=tuple(
                self.blocks[block_id].generation
                for block_id in seq.block_table
            ),
            reserved_block_ids=tuple(reserved_block_ids),
            reserved_block_generations=tuple(
                self.blocks[block_id].generation
                for block_id in reserved_block_ids
            ),
            proposed_token_count=proposed_token_count,
        )

    def mark_speculative_kv_materialized(
        self,
        transaction: SpeculativeKVTransaction,
        materialized_token_count: int,
    ) -> None:
        self._validate_speculative_kv_transaction_structure(
            transaction
        )
        if transaction.state != "reserved":
            raise RuntimeError(
                "speculative KV transaction is not materializable: "
                f"{transaction.state}"
            )
        if (
            isinstance(materialized_token_count, bool)
            or not isinstance(materialized_token_count, int)
            or materialized_token_count < 0
            or materialized_token_count > (
                transaction.proposed_token_count - 1
            )
        ):
            raise ValueError(
                "materialized_token_count is outside the proposal"
            )
        self._validate_speculative_original_blocks(transaction)
        self._validate_speculative_reserved_blocks(transaction)
        transaction.materialized_token_count = materialized_token_count
        transaction.state = "materialized"

    def _validate_speculative_sequence_owner(
        self,
        transaction: SpeculativeKVTransaction,
        seq: Sequence,
        *,
        require_snapshot: bool,
    ) -> None:
        if not isinstance(seq, Sequence):
            raise ValueError("seq must be a Sequence")
        if seq.seq_id != transaction.sequence_id:
            raise ValueError(
                "speculative KV transaction belongs to a different sequence"
            )
        if require_snapshot and (
            len(seq) != transaction.original_num_tokens
            or seq.last_token != transaction.original_last_token
            or tuple(seq.block_table)
            != transaction.original_block_table
        ):
            raise RuntimeError(
                "speculative KV transaction sequence snapshot is stale"
            )

    def _validate_speculative_original_blocks(
        self,
        transaction: SpeculativeKVTransaction,
    ) -> None:
        for block_id, generation in zip(
            transaction.original_block_table,
            transaction.original_block_generations,
        ):
            block = self.blocks[block_id]
            if (
                block_id not in self.used_block_ids
                or block.ref_count <= 0
                or block.generation != generation
            ):
                raise RuntimeError(
                    "speculative KV transaction original block ownership is stale"
                )

    def authorize_speculative_kv_write(
        self,
        transaction: SpeculativeKVTransaction,
        seq: Sequence,
    ) -> SpeculativeKVTransactionAuthorization:
        self._validate_speculative_kv_transaction_structure(
            transaction
        )
        if transaction.state != "reserved":
            raise RuntimeError(
                "speculative KV transaction is not authorizable: "
                f"{transaction.state}"
            )
        if transaction.materialized_token_count != 0:
            raise RuntimeError(
                "speculative KV transaction already has materialized KV"
            )
        self._validate_speculative_sequence_owner(
            transaction,
            seq,
            require_snapshot=True,
        )
        self._validate_speculative_original_blocks(transaction)
        self._validate_speculative_reserved_blocks(transaction)
        original_block_identities = tuple(zip(
            transaction.original_block_table,
            transaction.original_block_generations,
        ))
        reserved_block_identities = tuple(zip(
            transaction.reserved_block_ids,
            transaction.reserved_block_generations,
        ))
        payload = (
            transaction.sequence_id,
            transaction.original_num_tokens,
            transaction.proposed_token_count,
            transaction.materialized_token_count,
            transaction.state,
            original_block_identities,
            reserved_block_identities,
        )
        return SpeculativeKVTransactionAuthorization(
            sequence_id=transaction.sequence_id,
            original_num_tokens=(
                transaction.original_num_tokens
            ),
            proposed_token_count=(
                transaction.proposed_token_count
            ),
            materialized_token_count=(
                transaction.materialized_token_count
            ),
            state=transaction.state,
            original_block_identities=(
                original_block_identities
            ),
            reserved_block_identities=(
                reserved_block_identities
            ),
            authorization_sha256=hashlib.sha256(
                repr(payload).encode("utf-8")
            ).hexdigest(),
        )

    def prepare_speculative_kv_commit(
        self,
        transaction: SpeculativeKVTransaction,
        seq: Sequence,
        accepted_tokens: tuple[int, ...],
    ) -> SpeculativeKVCommitPlan:
        self._validate_speculative_kv_transaction_structure(
            transaction
        )
        if transaction.state != "materialized":
            raise RuntimeError(
                "speculative KV transaction is not committable: "
                f"{transaction.state}"
            )
        self._validate_speculative_sequence_owner(
            transaction,
            seq,
            require_snapshot=True,
        )
        self._validate_speculative_original_blocks(transaction)
        self._validate_speculative_reserved_blocks(transaction)
        if not isinstance(accepted_tokens, tuple):
            raise ValueError(
                "accepted_tokens must be a tuple"
            )
        if any(
            isinstance(token_id, bool)
            or not isinstance(token_id, int)
            for token_id in accepted_tokens
        ):
            raise ValueError(
                "each accepted token must be an integer"
            )
        accepted_count = len(accepted_tokens)
        if accepted_count > transaction.proposed_token_count:
            raise ValueError(
                "accepted token count exceeds proposal"
            )
        accepted_materialized_tokens = max(
            0,
            accepted_count - 1,
        )
        if accepted_materialized_tokens > (
            transaction.materialized_token_count
        ):
            raise RuntimeError(
                "accepted token prefix exceeds materialized KV"
            )

        materialized_end = (
            transaction.original_num_tokens
            + accepted_materialized_tokens
        )
        required_blocks = (
            materialized_end + self.block_size - 1
        ) // self.block_size
        committed_reserved_count = max(
            0,
            required_blocks
            - len(transaction.original_block_table),
        )
        if committed_reserved_count > len(
            transaction.reserved_block_ids
        ):
            raise RuntimeError(
                "speculative KV transaction lacks committed capacity"
            )
        committed_block_ids = (
            transaction.reserved_block_ids[
                :committed_reserved_count
            ]
        )
        unused_block_ids = (
            transaction.reserved_block_ids[
                committed_reserved_count:
            ]
        )
        planned_block_table = (
            transaction.original_block_table
            + committed_block_ids
        )
        planned_token_ids = (
            tuple(seq.token_ids)
            + accepted_tokens
        )
        publications = []
        prefix_hash = -1
        full_block_count = (
            materialized_end // self.block_size
        )
        for block_index in range(full_block_count):
            block_id = planned_block_table[block_index]
            start = block_index * self.block_size
            token_ids = tuple(
                planned_token_ids[
                    start:start + self.block_size
                ]
            )
            if len(token_ids) != self.block_size:
                raise RuntimeError(
                    "speculative KV publication block is incomplete"
                )
            block = self.blocks[block_id]
            if block.hash != -1:
                if tuple(block.token_ids) != token_ids:
                    raise RuntimeError(
                        "speculative KV cached block token mismatch"
                    )
                prefix_hash = block.hash
                continue
            block_hash = self.compute_hash(
                list(token_ids),
                prefix_hash,
            )
            publications.append(
                SpeculativeKVCachePublication(
                    block_id=block_id,
                    block_hash=block_hash,
                    token_ids=token_ids,
                )
            )
            prefix_hash = block_hash
        return SpeculativeKVCommitPlan(
            sequence_id=seq.seq_id,
            sequence=seq,
            transaction=transaction,
            accepted_tokens=accepted_tokens,
            committed_block_ids=committed_block_ids,
            unused_block_ids=unused_block_ids,
            materialized_end=materialized_end,
            publications=tuple(publications),
        )

    def _apply_speculative_kv_commit_plan(
        self,
        plan: SpeculativeKVCommitPlan,
    ) -> None:
        plan.sequence.block_table.extend(
            plan.committed_block_ids
        )
        for publication in plan.publications:
            block = self.blocks[publication.block_id]
            if block.hash == -1:
                self._register_cached_block(
                    publication.block_id,
                    publication.block_hash,
                    list(publication.token_ids),
                )
            elif (
                block.hash != publication.block_hash
                or tuple(block.token_ids)
                != publication.token_ids
            ):
                raise RuntimeError(
                    "speculative KV publication became stale"
                )
        self.release_reserved_blocks(
            list(plan.unused_block_ids)
        )
        plan.transaction.state = "committed"

    def commit_speculative_kv_commit_batch(
        self,
        plans: tuple[SpeculativeKVCommitPlan, ...],
    ) -> None:
        if not isinstance(plans, tuple) or not plans:
            raise ValueError(
                "speculative KV commit plans must be a non-empty tuple"
            )
        sequence_ids = []
        transaction_ids = []
        reserved_block_ids = []
        for plan in plans:
            if not isinstance(plan, SpeculativeKVCommitPlan):
                raise ValueError(
                    "speculative KV commit rows must be plans"
                )
            expected = self.prepare_speculative_kv_commit(
                plan.transaction,
                plan.sequence,
                plan.accepted_tokens,
            )
            if plan != expected:
                raise ValueError(
                    "speculative KV commit plan is stale"
                )
            sequence_ids.append(plan.sequence_id)
            transaction_ids.append(id(plan.transaction))
            reserved_block_ids.extend(
                plan.transaction.reserved_block_ids
            )
        if len(set(sequence_ids)) != len(sequence_ids):
            raise ValueError(
                "speculative KV commit sequence IDs must be unique"
            )
        if len(set(transaction_ids)) != len(transaction_ids):
            raise ValueError(
                "speculative KV commit transactions must be unique"
            )
        if len(set(reserved_block_ids)) != len(
            reserved_block_ids
        ):
            raise ValueError(
                "speculative KV commit reserved blocks must be disjoint"
            )

        free_before = tuple(self.free_block_ids)
        used_before = set(self.used_block_ids)
        block_snapshots = tuple(
            (
                block.ref_count,
                block.generation,
                block.hash,
                list(block.token_ids),
            )
            for block in self.blocks
        )
        hash_to_block_id_before = dict(
            self.hash_to_block_id
        )
        hash_to_block_ids_before = {
            block_hash: set(block_ids)
            for block_hash, block_ids
            in self.hash_to_block_ids.items()
        }
        sequence_tables_before = tuple(
            (
                plan.sequence,
                list(plan.sequence.block_table),
            )
            for plan in plans
        )
        transaction_states_before = tuple(
            (
                plan.transaction,
                plan.transaction.state,
            )
            for plan in plans
        )
        try:
            for plan in plans:
                self._apply_speculative_kv_commit_plan(plan)
        except BaseException:
            self.free_block_ids = deque(free_before)
            self.used_block_ids = set(used_before)
            self.hash_to_block_id = hash_to_block_id_before
            self.hash_to_block_ids = hash_to_block_ids_before
            for block, snapshot in zip(
                self.blocks,
                block_snapshots,
            ):
                (
                    block.ref_count,
                    block.generation,
                    block.hash,
                    token_ids,
                ) = snapshot
                block.token_ids = token_ids
            for sequence, block_table in (
                sequence_tables_before
            ):
                sequence.block_table = block_table
            for transaction, state in (
                transaction_states_before
            ):
                transaction.state = state
            raise

    def commit_speculative_kv_transaction(
        self,
        transaction: SpeculativeKVTransaction,
        seq: Sequence,
        accepted_tokens: list[int],
    ) -> None:
        self._validate_speculative_kv_transaction_structure(
            transaction
        )
        if transaction.state != "materialized":
            raise RuntimeError(
                "speculative KV transaction is not committable: "
                f"{transaction.state}"
            )
        self._validate_speculative_sequence_owner(
            transaction,
            seq,
            require_snapshot=True,
        )
        self._validate_speculative_original_blocks(transaction)
        self._validate_speculative_reserved_blocks(transaction)
        if not isinstance(accepted_tokens, list):
            raise ValueError("accepted_tokens must be a list")
        if any(
            isinstance(token_id, bool)
            or not isinstance(token_id, int)
            for token_id in accepted_tokens
        ):
            raise ValueError(
                "each accepted token must be an integer"
            )
        accepted_count = len(accepted_tokens)
        if accepted_count > transaction.proposed_token_count:
            raise ValueError(
                "accepted token count exceeds proposal"
            )
        accepted_materialized_tokens = max(
            0,
            accepted_count - 1,
        )
        if accepted_materialized_tokens > (
            transaction.materialized_token_count
        ):
            raise RuntimeError(
                "accepted token prefix exceeds materialized KV"
            )

        materialized_end = (
            transaction.original_num_tokens
            + accepted_materialized_tokens
        )
        required_blocks = (
            materialized_end + self.block_size - 1
        ) // self.block_size
        committed_reserved_count = max(
            0,
            required_blocks
            - len(transaction.original_block_table),
        )
        if committed_reserved_count > len(
            transaction.reserved_block_ids
        ):
            raise RuntimeError(
                "speculative KV transaction lacks committed capacity"
            )
        committed_block_ids = list(
            transaction.reserved_block_ids[
                :committed_reserved_count
            ]
        )
        unused_block_ids = list(
            transaction.reserved_block_ids[
                committed_reserved_count:
            ]
        )

        seq.block_table.extend(committed_block_ids)
        for token_id in accepted_tokens:
            seq.append_token(token_id)
        self.publish_full_blocks(
            seq,
            materialized_tokens=materialized_end,
        )
        self.release_reserved_blocks(unused_block_ids)
        transaction.state = "committed"

    def rollback_speculative_kv_transaction(
        self,
        transaction: SpeculativeKVTransaction,
        seq: Sequence,
    ) -> None:
        self._validate_speculative_kv_transaction_structure(
            transaction
        )
        if transaction.state not in ("reserved", "materialized"):
            raise RuntimeError(
                "speculative KV transaction is not rollbackable: "
                f"{transaction.state}"
            )
        self._validate_speculative_sequence_owner(
            transaction,
            seq,
            require_snapshot=False,
        )
        self._validate_speculative_reserved_blocks(transaction)
        self.release_reserved_blocks(
            list(transaction.reserved_block_ids)
        )
        transaction.state = "rolled_back"

    def reserve_append_blocks(self, seq: Sequence, num_new_tokens: int) -> list[int]:
        """Reserve extra blocks needed to store speculative appended tokens.

        The returned block ids are allocated but are not added to ``seq`` yet.
        Callers may expose the ids through a temporary verifier Sequence; after
        verification they must either pass them to ``commit_accepted_tokens`` or
        release them with ``release_reserved_blocks``.
        """
        if num_new_tokens <= 0:
            return []
        final_len = len(seq) + num_new_tokens
        needed_blocks = (final_len + self.block_size - 1) // self.block_size
        missing_blocks = max(0, needed_blocks - len(seq.block_table))
        assert len(self.free_block_ids) >= missing_blocks
        reserved = []
        for _ in range(missing_blocks):
            block_id = self.free_block_ids[0]
            self._allocate_block(block_id)
            reserved.append(block_id)
        return reserved

    def release_reserved_blocks(self, block_ids: list[int]):
        """Release scratch blocks that were reserved but not committed."""
        for block_id in block_ids:
            block = self.blocks[block_id]
            assert block.ref_count == 1
            assert block.hash == -1
            block.ref_count = 0
            self._deallocate_block(block_id)

    def publish_full_blocks(self, seq: Sequence, materialized_tokens=None):
        """Publish prefix-cache hashes for all fully materialized blocks."""
        if materialized_tokens is None:
            materialized_tokens = len(seq)
        for i, block_id in enumerate(seq.block_table):
            if (i + 1) * self.block_size > materialized_tokens:
                continue
            token_ids = seq.block(i)
            if len(token_ids) != self.block_size:
                continue
            block = self.blocks[block_id]
            if block.hash != -1:
                continue
            prefix = self.blocks[seq.block_table[i - 1]].hash if i > 0 else -1
            h = self.compute_hash(token_ids, prefix)
            self._register_cached_block(block_id, h, token_ids)

    def commit_accepted_tokens(self, seq: Sequence, accepted_tokens: list[int], reserved_block_ids: list[int]):
        """Commit accepted speculative tokens and expose only needed blocks.

        This updates Sequence metadata and prefix-cache hashes only. It assumes
        the caller has already written KV for accepted token positions into the
        corresponding current/reserved block slots.
        """
        if not accepted_tokens:
            self.release_reserved_blocks(list(reserved_block_ids))
            return

        final_len = len(seq) + len(accepted_tokens)
        materialized_tokens = final_len - 1
        needed_blocks = (materialized_tokens + self.block_size - 1) // self.block_size
        missing_blocks = max(0, needed_blocks - len(seq.block_table))
        assert missing_blocks <= len(reserved_block_ids)
        committed_blocks = list(reserved_block_ids[:missing_blocks])
        unused_blocks = list(reserved_block_ids[missing_blocks:])
        seq.block_table.extend(committed_blocks)
        for token_id in accepted_tokens:
            seq.append_token(token_id)
        self.publish_full_blocks(seq, materialized_tokens=materialized_tokens)
        self.release_reserved_blocks(unused_blocks)

    def commit_prefill(self, seq: Sequence, old_end: int, new_end: int):
        """Publish prefix-cache hashes only for blocks whose KV has been computed.

        Chunked prefill may allocate all blocks up front, but future blocks must not
        be visible through hash_to_block_id until their KV slots have actually been
        written by a completed prefill chunk.
        """
        if new_end <= old_end:
            return
        first_block = max(0, old_end // self.block_size)
        last_full_block = new_end // self.block_size - 1
        for i in range(first_block, last_full_block + 1):
            token_ids = seq.block(i)
            if len(token_ids) != self.block_size:
                continue
            block_id = seq.block_table[i]
            block = self.blocks[block_id]
            if block.hash != -1:
                continue
            prefix = self.blocks[seq.block_table[i - 1]].hash if i > 0 else -1
            h = self.compute_hash(token_ids, prefix)
            self._register_cached_block(block_id, h, token_ids)
            
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
            self._register_cached_block(
                last_block.block_id,
                h,
                token_ids,
            )
        else:   #最后一个块未填满，h是-1，没有计算哈希值写入字典用于缓存
            assert last_block.hash == -1
            
