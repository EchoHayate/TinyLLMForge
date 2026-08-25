from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class BlockTableIdentitySeal:
    sequence_id: int
    table_revision: int
    ownership_generation: int
    block_count: int
    write_block_index: int
    write_block_id: int
    write_block_generation: int
    predecessor_block_id: Optional[int]
    predecessor_block_generation: Optional[int]
    identity_sha256: str
