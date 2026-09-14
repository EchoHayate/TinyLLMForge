"""Round-trip tests for KV slot snapshot/restore under quantised KV.

Multi-sequence capture borrows scratch KV slots and must hand them back
byte-identical. Quantised KV was refused outright by `snapshot_kv_slots`, which
disabled capture whenever kv_quant_bits was set, so any attempt to measure KV
compression on the graph path silently measured the eager path instead.

These tests exercise the real methods against a minimal holder rather than a
loaded model, because the thing worth pinning is the indexing: payload and scales
must be gathered and scattered at the same (block, offset) coordinates.
"""

from __future__ import annotations

import os
import sys
import types
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import torch
except ImportError:  # pragma: no cover - exercised on hosts without torch
    torch = None


@unittest.skipUnless(torch is not None, "requires torch")
class QuantisedKVSnapshotTests(unittest.TestCase):
    LAYERS = 2
    BLOCKS = 3
    BLOCK_SIZE = 4
    HEADS = 2
    HEAD_DIM = 8
    GROUPS = 2

    def _holder(self, *, quantised):
        from tinyvllm.engine.model_runner import ModelRunner

        holder = types.SimpleNamespace()
        holder.block_size = self.BLOCK_SIZE
        holder.config = types.SimpleNamespace(kv_quant_bits=8 if quantised else 0)
        shape = (
            2,
            self.LAYERS,
            self.BLOCKS,
            self.BLOCK_SIZE,
            self.HEADS,
            self.HEAD_DIM,
        )
        if quantised:
            holder.kv_cache = torch.randint(
                -128, 127, shape, dtype=torch.int8
            )
            holder.kv_scale = torch.randn(
                2,
                self.LAYERS,
                self.BLOCKS,
                self.BLOCK_SIZE,
                self.HEADS,
                self.GROUPS,
            )
        else:
            holder.kv_cache = torch.randn(*shape)
            holder.kv_scale = None
        holder.snapshot_kv_slots = types.MethodType(
            ModelRunner.snapshot_kv_slots, holder
        )
        holder.restore_kv_slots = types.MethodType(
            ModelRunner.restore_kv_slots, holder
        )
        return holder

    def test_quantised_snapshot_is_no_longer_refused(self):
        holder = self._holder(quantised=True)
        snapshot = holder.snapshot_kv_slots([0, 5])
        self.assertIn("keys", snapshot)
        self.assertIn("key_scales", snapshot)
        self.assertIn("value_scales", snapshot)

    def test_scales_are_carried_and_restored_byte_exact(self):
        """Payload-only restore would pair integers with foreign scales."""
        holder = self._holder(quantised=True)
        slots = [1, 6, 11]
        snapshot = holder.snapshot_kv_slots(slots)
        before_cache = holder.kv_cache.clone()
        before_scale = holder.kv_scale.clone()

        holder.kv_cache.random_(-128, 127)
        holder.kv_scale.normal_()
        holder.restore_kv_slots(slots, snapshot)

        for slot in slots:
            block, offset = slot // self.BLOCK_SIZE, slot % self.BLOCK_SIZE
            self.assertTrue(
                torch.equal(
                    holder.kv_cache[:, :, block, offset],
                    before_cache[:, :, block, offset],
                )
            )
            self.assertTrue(
                torch.equal(
                    holder.kv_scale[:, :, block, offset],
                    before_scale[:, :, block, offset],
                )
            )

    def test_restore_touches_only_the_named_slots(self):
        holder = self._holder(quantised=True)
        snapshot = holder.snapshot_kv_slots([0])
        untouched = holder.kv_cache[:, :, 0, 1].clone()
        holder.restore_kv_slots([0], snapshot)
        self.assertTrue(torch.equal(holder.kv_cache[:, :, 0, 1], untouched))

    def test_a_payload_only_snapshot_is_rejected_by_a_quantised_cache(self):
        holder = self._holder(quantised=True)
        snapshot = holder.snapshot_kv_slots([0])
        del snapshot["key_scales"]
        with self.assertRaises(RuntimeError):
            holder.restore_kv_slots([0], snapshot)

    def test_unquantised_round_trip_still_works(self):
        holder = self._holder(quantised=False)
        slots = [2, 7]
        snapshot = holder.snapshot_kv_slots(slots)
        self.assertNotIn("key_scales", snapshot)
        before = holder.kv_cache.clone()
        holder.kv_cache.normal_()
        holder.restore_kv_slots(slots, snapshot)
        for slot in slots:
            block, offset = slot // self.BLOCK_SIZE, slot % self.BLOCK_SIZE
            self.assertTrue(
                torch.equal(
                    holder.kv_cache[:, :, block, offset],
                    before[:, :, block, offset],
                )
            )


if __name__ == "__main__":
    unittest.main()
