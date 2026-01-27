
import sys
import unittest.mock as mock
from collections import OrderedDict, deque

# Mock everything!
def dynamic_hash():
    # A simple mock hash that changes based on what was updated
    m = mock.MagicMock()
    # We'll use a local list to store updated values
    m._data = []
    m.update.side_effect = lambda d: m._data.append(hash(d))
    m.intdigest.side_effect = lambda: sum(m._data) if m._data else 12345
    return m

mock_xxhash = mock.MagicMock()
mock_xxhash.xxh64.side_effect = dynamic_hash
sys.modules['xxhash'] = mock_xxhash

mock_numpy = mock.MagicMock()
sys.modules['numpy'] = mock_numpy

import numpy as np
import xxhash

# 1. Mock the tinyvllm module to prevent import chain errors
mock_tinyvllm = mock.MagicMock()
sys.modules['tinyvllm'] = mock_tinyvllm
sys.modules['tinyvllm.engine'] = mock_tinyvllm.engine
sys.modules['tinyvllm.sampling_params'] = mock_tinyvllm.sampling_params

# 2. Re-implement minimal Sequence for testing since we can't easily import it
class MockSequence:
    def __init__(self, token_ids):
        self.token_ids = token_ids
        self.num_tokens = len(token_ids)
        self.num_cached_tokens = 0
        self.block_table = []
        self.block_size = 1 # Simple for testing
        
    def __len__(self):
        return self.num_tokens

    @property
    def num_blocks(self):
        return (self.num_tokens + self.block_size - 1) // self.block_size

    def block(self, i):
        return self.token_ids[i * self.block_size : (i + 1) * self.block_size]

# Now 'BlockManager' and 'Block' should be in globals()
# Inject MockSequence as Sequence for type hints
globals()['Sequence'] = MockSequence
globals()['Any'] = mock.MagicMock()

with open("tinyvllm/engine/block_manager.py", "r", encoding="utf-8") as f:
    code = f.read()
    # Remove all imports to avoid any side effects
    lines = code.split("\n")
    safe_lines = [l for l in lines if not l.startswith("import ") and not l.startswith("from ")]
    exec("\n".join(safe_lines), globals())

# Now 'BlockManager' and 'Block' should be in globals()

def test_lru_eviction():
    print("Testing LRU Eviction...")
    # 3 blocks total, block_size=1
    bm = BlockManager(num_blocks=3, block_size=1)
    
    # 验证初始状态：free_block_ids 应该是 0, 1, 2
    assert list(bm.free_block_ids.keys()) == [0, 1, 2]
    
    s1 = MockSequence([1])
    s2 = MockSequence([2])
    s3 = MockSequence([3])
    
    bm.allocate(s1)
    bm.allocate(s2)
    bm.allocate(s3)
    
    # Verify s1, s2, s3 got 0, 1, 2
    assert s1.block_table == [0]
    assert s2.block_table == [1]
    assert s3.block_table == [2]
    assert len(bm.free_block_ids) == 0
    print("- All blocks allocated in order.")

    # Deallocate s1 (block 0), then s2 (block 1).
    # Expected free_block_ids: {0: None, 1: None} (0 is older, 1 is newer)
    bm.deallocate(s1)
    bm.deallocate(s2)
    assert list(bm.free_block_ids.keys()) == [0, 1]
    
    # Allocate s4. Should take from Front (0).
    s4 = MockSequence([4])
    bm.allocate(s4)
    assert s4.block_table == [0]
    assert list(bm.free_block_ids.keys()) == [1]
    print("- LRU Allocation (s4 took block 0) successful.")
    
    # Deallocate s4 (block 0). Now 1 is OLDER than 0.
    # Expected: {1: None, 0: None}
    bm.deallocate(s4)
    assert list(bm.free_block_ids.keys()) == [1, 0]
    
    # Allocate s5. Should take from Front (1).
    s5 = MockSequence([5])
    bm.allocate(s5)
    assert s5.block_table == [1]
    print("- LRU Order Update (s5 took block 1) successful.")
    print("LRU Test Passed!")

def test_stale_hash_cleanup():
    print("\nTesting Stale Hash Cleanup...")
    bm = BlockManager(num_blocks=1, block_size=1)
    
    s1 = MockSequence([100])
    bm.allocate(s1)
    h100 = bm.blocks[0].hash
    assert bm.hash_to_block_id[h100] == 0
    
    bm.deallocate(s1)
    # Hash remains
    assert h100 in bm.hash_to_block_id
    
    # Reuse block 0 for s2 [200]
    s2 = MockSequence([200])
    bm.allocate(s2)
    h200 = bm.blocks[0].hash
    
    # Old hash h100 MUST be removed
    assert h100 not in bm.hash_to_block_id
    assert bm.hash_to_block_id[h200] == 0
    print("- Stale hash [100] correctly removed when block was reused for [200].")
    print("Stale Hash Test Passed!")

if __name__ == "__main__":
    try:
        test_lru_eviction()
        test_stale_hash_cleanup()
        print("\nALL TESTS PASSED SUCCESSFULLY!")
    except AssertionError as e:
        print(f"\nTEST FAILED! {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\nAN ERROR OCCURRED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
