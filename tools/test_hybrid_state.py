import importlib.util
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]


def load_module():
    path = ROOT / "tinyvllm/engine/hybrid_state.py"
    spec = importlib.util.spec_from_file_location("hybrid_state_test_target", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


hybrid_state = load_module()
HybridStateComponentSpec = hybrid_state.HybridStateComponentSpec
HybridStateLayout = hybrid_state.HybridStateLayout
HybridStateLease = hybrid_state.HybridStateLease
HybridStateSlotAllocator = hybrid_state.HybridStateSlotAllocator
HybridStateTensorPool = hybrid_state.HybridStateTensorPool
HybridStateRuntimeBridge = hybrid_state.HybridStateRuntimeBridge


def qwen35_layout(dtype=torch.bfloat16):
    return HybridStateLayout(tuple(
        component
        for layer_index in range(18)
        for component in (
            HybridStateComponentSpec(
                layer_index=layer_index,
                role="linear_convolution",
                shape=(6144, 4),
                dtype=dtype,
            ),
            HybridStateComponentSpec(
                layer_index=layer_index,
                role="linear_recurrent",
                shape=(16, 128, 128),
                dtype=dtype,
            ),
        )
    ))


def test_layout_accounts_reference_bytes_and_fingerprint():
    bf16 = qwen35_layout()
    fp32 = qwen35_layout(torch.float32)
    assert bf16.bytes_per_slot == 10_321_920
    assert bf16.bytes_by_role == {
        "linear_convolution": 884_736,
        "linear_recurrent": 9_437_184,
    }
    assert fp32.bytes_per_slot == 2 * bf16.bytes_per_slot
    assert bf16.fingerprint == qwen35_layout().fingerprint
    changed = HybridStateLayout((
        HybridStateComponentSpec(
            layer_index=0,
            role="linear_recurrent",
            shape=(16, 128, 127),
            dtype=torch.bfloat16,
        ),
    ))
    assert changed.fingerprint != bf16.fingerprint


def test_layout_rejects_invalid_components():
    invalid = (
        HybridStateComponentSpec(-1, "linear_recurrent", (1,), torch.float32),
        HybridStateComponentSpec(0, "bad", (1,), torch.float32),
        HybridStateComponentSpec(0, "linear_recurrent", (), torch.float32),
        HybridStateComponentSpec(0, "linear_recurrent", (0,), torch.float32),
        HybridStateComponentSpec(0, "linear_recurrent", (1,), torch.int8),
    )
    for component in invalid:
        try:
            HybridStateLayout((component,))
        except ValueError:
            pass
        else:
            raise AssertionError(f"invalid component accepted: {component}")
    duplicate = HybridStateComponentSpec(
        0, "linear_recurrent", (1,), torch.float32
    )
    try:
        HybridStateLayout((duplicate, duplicate))
    except ValueError:
        pass
    else:
        raise AssertionError("duplicate component accepted")


def test_allocator_reuses_slot_with_new_generation():
    allocator = HybridStateSlotAllocator(2)
    first = allocator.allocate(11)
    second = allocator.allocate(12)
    assert (first.slot_id, first.generation) == (0, 1)
    assert (second.slot_id, second.generation) == (1, 1)
    assert not allocator.can_allocate()
    try:
        allocator.allocate(13)
    except RuntimeError:
        pass
    else:
        raise AssertionError("exhausted allocator accepted request")
    try:
        allocator.allocate(11)
    except RuntimeError:
        pass
    else:
        raise AssertionError("duplicate request allocation accepted")
    allocator.release(first)
    reused = allocator.allocate(13)
    assert (reused.slot_id, reused.generation) == (0, 2)
    assert allocator.lease_for_request(13) == reused
    snapshot = allocator.observation_snapshot()
    assert snapshot["capacity"] == 2
    assert snapshot["free_slots"] == 0
    assert snapshot["used_slots"] == 2
    assert snapshot["owners"] == {"0": 13, "1": 12}
    assert snapshot["generations"] == {"0": 2, "1": 1}


def test_allocator_rejects_invalid_capacity_request_and_slot():
    for capacity in (0, -1, True):
        try:
            HybridStateSlotAllocator(capacity)
        except ValueError:
            pass
        else:
            raise AssertionError(f"invalid capacity accepted: {capacity}")
    allocator = HybridStateSlotAllocator(1)
    for request_id in (-1, True, "request"):
        try:
            allocator.allocate(request_id)
        except ValueError:
            pass
        else:
            raise AssertionError(f"invalid request id accepted: {request_id}")
    try:
        allocator.release(HybridStateLease(1, 1, 1))
    except RuntimeError:
        pass
    else:
        raise AssertionError("out-of-range allocator slot accepted")


def test_allocator_rejects_wrong_stale_and_double_release():
    allocator = HybridStateSlotAllocator(1)
    first = allocator.allocate(21)
    wrong_owner = HybridStateLease(first.slot_id, first.generation, 22)
    for invalid in (wrong_owner,):
        try:
            allocator.release(invalid)
        except RuntimeError:
            pass
        else:
            raise AssertionError("wrong-owner release accepted")
    allocator.release(first)
    second = allocator.allocate(22)
    for invalid in (first, HybridStateLease(second.slot_id, 0, 22)):
        try:
            allocator.release(invalid)
        except RuntimeError:
            pass
        else:
            raise AssertionError("stale release accepted")
    allocator.release(second)
    try:
        allocator.release(second)
    except RuntimeError:
        pass
    else:
        raise AssertionError("double release accepted")


def test_tensor_pool_zeroes_and_validates_leases():
    layout = HybridStateLayout((
        HybridStateComponentSpec(
            0, "linear_convolution", (2, 3), torch.float32
        ),
        HybridStateComponentSpec(
            0, "linear_recurrent", (2, 2), torch.float32
        ),
    ))
    pool = HybridStateTensorPool(layout, capacity=2, device="cpu")
    allocator = HybridStateSlotAllocator(2)
    first = allocator.allocate(31)
    second = allocator.allocate(32)
    convolution = pool.component_tensor(0, "linear_convolution")
    recurrent = pool.component_tensor(0, "linear_recurrent")
    assert convolution.shape == (2, 2, 3)
    assert recurrent.shape == (2, 2, 2)
    addresses = (convolution.data_ptr(), recurrent.data_ptr())
    convolution[first.slot_id].fill_(7)
    recurrent[first.slot_id].fill_(9)
    pool.activate(first)
    assert torch.count_nonzero(convolution[first.slot_id]).item() == 0
    assert torch.count_nonzero(recurrent[first.slot_id]).item() == 0
    convolution[first.slot_id].fill_(5)
    recurrent[first.slot_id].fill_(6)
    pool.activate(first)
    assert torch.all(convolution[first.slot_id] == 5)
    assert torch.all(recurrent[first.slot_id] == 6)
    pool.activate(second)
    assert pool.validate(first) == first.slot_id
    assert pool.slot_ids([second, first]).tolist() == [1, 0]
    convolution[first.slot_id].fill_(3)
    recurrent[first.slot_id].fill_(4)
    pool.release(first)
    assert torch.count_nonzero(convolution[first.slot_id]).item() == 0
    assert torch.count_nonzero(recurrent[first.slot_id]).item() == 0
    allocator.release(first)
    reused = allocator.allocate(33)
    pool.activate(reused)
    for operation in (
        lambda: pool.validate(first),
        lambda: pool.release(first),
    ):
        try:
            operation()
        except RuntimeError:
            pass
        else:
            raise AssertionError("stale tensor-pool lease accepted")
    assert addresses == (convolution.data_ptr(), recurrent.data_ptr())
    assert pool.logical_bytes == layout.bytes_per_slot * 2
    assert pool.physical_storage_bytes == layout.bytes_per_slot * 2


def test_tensor_pool_rejects_conflicting_activation():
    layout = HybridStateLayout((
        HybridStateComponentSpec(
            0, "linear_recurrent", (1,), torch.float32
        ),
    ))
    pool = HybridStateTensorPool(layout, capacity=1, device="cpu")
    first = HybridStateLease(0, 1, 41)
    pool.activate(first)
    for invalid in (
        HybridStateLease(0, 1, 42),
        HybridStateLease(1, 1, 41),
    ):
        try:
            pool.activate(invalid)
        except (RuntimeError, ValueError):
            pass
        else:
            raise AssertionError("conflicting tensor-pool activation accepted")


def test_runtime_bridge_releases_before_reused_generation_activation():
    layout = HybridStateLayout((
        HybridStateComponentSpec(
            0, "linear_recurrent", (2,), torch.float32
        ),
    ))
    pool = HybridStateTensorPool(layout, capacity=1, device="cpu")
    bridge = HybridStateRuntimeBridge(pool)
    first = HybridStateLease(0, 1, 51)
    second = HybridStateLease(0, 2, 52)
    pool.activate(first)
    tensor = pool.component_tensor(0, "linear_recurrent")
    tensor[0].fill_(7)
    slot_ids = bridge.prepare_batch((first,), (second,))
    assert slot_ids.tolist() == [0]
    assert torch.count_nonzero(tensor[0]).item() == 0
    assert pool.validate(second) == 0


def test_runtime_bridge_idempotence_order_and_isolation():
    layout = HybridStateLayout((
        HybridStateComponentSpec(
            0, "linear_recurrent", (1,), torch.float32
        ),
    ))
    pool = HybridStateTensorPool(layout, capacity=2, device="cpu")
    bridge = HybridStateRuntimeBridge(pool)
    first = HybridStateLease(0, 1, 61)
    second = HybridStateLease(1, 1, 62)
    assert bridge.prepare_batch((), (second, first)).tolist() == [1, 0]
    tensor = pool.component_tensor(0, "linear_recurrent")
    tensor[0].fill_(3)
    tensor[1].fill_(5)
    assert bridge.prepare_batch((), (first, second)).tolist() == [0, 1]
    assert tensor[:, 0].tolist() == [3.0, 5.0]
    bridge.release((first,))
    assert tensor[:, 0].tolist() == [0.0, 5.0]


def test_runtime_bridge_rejects_stale_duplicate_and_wrong_owner_release():
    layout = HybridStateLayout((
        HybridStateComponentSpec(
            0, "linear_recurrent", (1,), torch.float32
        ),
    ))
    pool = HybridStateTensorPool(layout, capacity=1, device="cpu")
    bridge = HybridStateRuntimeBridge(pool)
    first = HybridStateLease(0, 1, 71)
    second = HybridStateLease(0, 2, 72)
    pool.activate(first)
    bridge.prepare_batch((first,), (second,))
    invalid_releases = (
        first,
        HybridStateLease(0, 2, 73),
    )
    for invalid in invalid_releases:
        try:
            bridge.release((invalid,))
        except RuntimeError:
            pass
        else:
            raise AssertionError("invalid runtime release was accepted")
    bridge.release((second,))
    try:
        bridge.release((second,))
    except RuntimeError:
        pass
    else:
        raise AssertionError("duplicate runtime release was accepted")


if __name__ == "__main__":
    test_layout_accounts_reference_bytes_and_fingerprint()
    test_layout_rejects_invalid_components()
    test_allocator_reuses_slot_with_new_generation()
    test_allocator_rejects_invalid_capacity_request_and_slot()
    test_allocator_rejects_wrong_stale_and_double_release()
    test_tensor_pool_zeroes_and_validates_leases()
    test_tensor_pool_rejects_conflicting_activation()
    test_runtime_bridge_releases_before_reused_generation_activation()
    test_runtime_bridge_idempotence_order_and_isolation()
    test_runtime_bridge_rejects_stale_duplicate_and_wrong_owner_release()
    print("hybrid state tests passed")
