"""KV-Cartridge v0 helper tests.

跑法：python3 tools/test_kv_cartridge.py
"""

import os
import sys
import importlib.util

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_THIS_DIR)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

_KV_CARTRIDGE_PATH = os.path.join(_REPO_ROOT, "tinyvllm", "engine", "kv_cartridge.py")
_SPEC = importlib.util.spec_from_file_location("kv_cartridge_under_test", _KV_CARTRIDGE_PATH)
kv_cartridge = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(kv_cartridge)

compress_decode_block_table_rows = kv_cartridge.compress_decode_block_table_rows
select_uniform_cartridge_indices = kv_cartridge.select_uniform_cartridge_indices
should_use_kv_cartridge = kv_cartridge.should_use_kv_cartridge


def test_uniform_cartridge_preserves_first_last_and_spreads_middle():
    indices = select_uniform_cartridge_indices(num_blocks=10, budget=4)

    assert indices == [0, 3, 6, 9]


def test_uniform_cartridge_returns_full_range_when_budget_covers_sequence():
    indices = select_uniform_cartridge_indices(num_blocks=4, budget=8)

    assert indices == [0, 1, 2, 3]


def test_compress_decode_rows_preserves_physical_block_ids_and_compact_lens():
    rows, lens = compress_decode_block_table_rows(
        block_table_rows=[list(range(10, 20)), list(range(20, 30))],
        context_lens=[10 * 256, 9 * 256 + 17],
        block_size=256,
        budget=4,
    )

    assert rows == [[10, 13, 16, 19], [20, 23, 26, 29]]
    assert lens == [4 * 256, 3 * 256 + 17]


def test_should_use_kv_cartridge_requires_all_rows_to_benefit():
    assert should_use_kv_cartridge(
        seq_lens=[4096, 8192], num_blocks=[16, 32], budget=8, min_seq_len=1024
    )
    assert not should_use_kv_cartridge(
        seq_lens=[768, 8192], num_blocks=[3, 32], budget=8, min_seq_len=1024
    )
    assert not should_use_kv_cartridge(
        seq_lens=[4096, 8192], num_blocks=[8, 32], budget=8, min_seq_len=1024
    )


def main():
    test_uniform_cartridge_preserves_first_last_and_spreads_middle()
    test_uniform_cartridge_returns_full_range_when_budget_covers_sequence()
    test_compress_decode_rows_preserves_physical_block_ids_and_compact_lens()
    test_should_use_kv_cartridge_requires_all_rows_to_benefit()
    print("KV-Cartridge helper tests passed")


if __name__ == "__main__":
    main()
