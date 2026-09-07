from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

import torch


ROOT = Path(__file__).resolve().parents[1]


class _TorchFacade:
    cuda = SimpleNamespace(synchronize=lambda: None)


def _load_restore_kv_slots():
    path = ROOT / "tinyvllm/engine/model_runner.py"
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    class_node = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "ModelRunner"
    )
    method_node = next(
        node
        for node in class_node.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "restore_kv_slots"
    )
    method_node.decorator_list = []
    namespace = {"torch": _TorchFacade}
    exec(
        compile(
            ast.fix_missing_locations(
                ast.Module(body=[method_node], type_ignores=[])
            ),
            str(path),
            "exec",
        ),
        namespace,
    )
    return namespace["restore_kv_slots"]


def test_restore_kv_slots_restores_selected_slots_only():
    restore_kv_slots = _load_restore_kv_slots()
    block_size = 2
    kv_cache = torch.arange(
        2 * 2 * 3 * block_size * 2 * 2,
        dtype=torch.float32,
    ).reshape(2, 2, 3, block_size, 2, 2)
    physical_slots = [1, 4]
    selected = [
        (slot // block_size, slot % block_size)
        for slot in physical_slots
    ]
    snapshot = {
        "keys": torch.stack(
            [
                kv_cache[0, :, block_id, offset].clone()
                for block_id, offset in selected
            ],
            dim=1,
        ),
        "values": torch.stack(
            [
                kv_cache[1, :, block_id, offset].clone()
                for block_id, offset in selected
            ],
            dim=1,
        ),
    }

    for block_id, offset in selected:
        kv_cache[:, :, block_id, offset].fill_(-1)
    before_restore = kv_cache.clone()
    runner = SimpleNamespace(block_size=block_size, kv_cache=kv_cache)

    restore_kv_slots(runner, physical_slots, snapshot)

    for ordinal, (block_id, offset) in enumerate(selected):
        assert torch.equal(
            kv_cache[0, :, block_id, offset],
            snapshot["keys"][:, ordinal],
        )
        assert torch.equal(
            kv_cache[1, :, block_id, offset],
            snapshot["values"][:, ordinal],
        )
    selected_set = set(selected)
    for block_id in range(kv_cache.size(2)):
        for offset in range(block_size):
            if (block_id, offset) in selected_set:
                continue
            assert torch.equal(
                kv_cache[:, :, block_id, offset],
                before_restore[:, :, block_id, offset],
            )
