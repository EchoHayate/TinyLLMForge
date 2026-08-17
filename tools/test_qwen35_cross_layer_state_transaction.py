import importlib.util
from pathlib import Path
import sys
import types

import torch

ROOT = Path(__file__).resolve().parents[1]


def _load_module(module_name: str, relative_path: str):
    path = ROOT / relative_path
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


for package_name in ("tinyvllm", "tinyvllm.engine"):
    package = types.ModuleType(package_name)
    package.__path__ = [str(ROOT / package_name.replace(".", "/"))]
    sys.modules[package_name] = package

hybrid = _load_module(
    "tinyvllm.engine.hybrid_state",
    "tinyvllm/engine/hybrid_state.py",
)
adapter_module = _load_module(
    "tinyvllm.engine.qwen35_layer_state",
    "tinyvllm/engine/qwen35_layer_state.py",
)
transaction_module = _load_module(
    "tinyvllm.engine.qwen35_state_transaction",
    "tinyvllm/engine/qwen35_state_transaction.py",
)

HybridStateComponentSpec = hybrid.HybridStateComponentSpec
HybridStateLayout = hybrid.HybridStateLayout
HybridStateLease = hybrid.HybridStateLease
HybridStateTensorPool = hybrid.HybridStateTensorPool
Qwen35LayerStateAdapter = adapter_module.Qwen35LayerStateAdapter
Qwen35CrossLayerStateTransaction = (
    transaction_module.Qwen35CrossLayerStateTransaction
)


def _layout(layer_indices=(1, 3)):
    components = []
    for layer_index in layer_indices:
        components.extend((
            HybridStateComponentSpec(
                layer_index,
                "linear_convolution",
                (4, 3),
                torch.float32,
            ),
            HybridStateComponentSpec(
                layer_index,
                "linear_recurrent",
                (2, 3, 2),
                torch.float32,
            ),
        ))
    return HybridStateLayout(tuple(components))


def _fixture():
    pool = HybridStateTensorPool(_layout(), capacity=3, device="cpu")
    leases = (
        HybridStateLease(0, 1, 17),
        HybridStateLease(1, 1, 18),
        HybridStateLease(2, 1, 19),
    )
    for lease in leases:
        pool.activate(lease)
    adapters = (
        Qwen35LayerStateAdapter(pool, 1),
        Qwen35LayerStateAdapter(pool, 3),
    )
    for layer_offset, layer_index in enumerate((1, 3)):
        convolution = pool.component_tensor(
            layer_index,
            "linear_convolution",
        )
        recurrent = pool.component_tensor(
            layer_index,
            "linear_recurrent",
        )
        for slot_id in range(3):
            convolution[slot_id].copy_(
                torch.arange(12).reshape(4, 3)
                + layer_offset * 10000
                + slot_id * 100
            )
            recurrent[slot_id].copy_(
                torch.arange(12).reshape(2, 3, 2)
                + layer_offset * 20000
                + slot_id * 1000
            )
    transaction = Qwen35CrossLayerStateTransaction(adapters)
    return pool, leases, adapters, transaction


def _snapshots(pool):
    return tuple(
        (
            pool.component_tensor(
                layer_index,
                "linear_convolution",
            ).clone(),
            pool.component_tensor(
                layer_index,
                "linear_recurrent",
            ).clone(),
        )
        for layer_index in (1, 3)
    )


def test_gather_preserves_layer_and_lease_order_with_clone_isolation():
    pool, leases, _, transaction = _fixture()
    gathered = transaction.gather((leases[2], leases[0]))
    assert len(gathered) == 2
    for layer_offset, layer_index in enumerate((1, 3)):
        convolution, recurrent = gathered[layer_offset]
        assert convolution.shape == (2, 4, 3)
        assert recurrent.shape == (2, 2, 3, 2)
        torch.testing.assert_close(
            convolution[0],
            pool.component_tensor(
                layer_index,
                "linear_convolution",
            )[2],
        )
        torch.testing.assert_close(
            recurrent[1],
            pool.component_tensor(
                layer_index,
                "linear_recurrent",
            )[0],
        )
        convolution.add_(99999)
        recurrent.add_(99999)
    current = _snapshots(pool)
    assert max(tensor.max().item() for pair in current for tensor in pair) < 99999


def test_commit_updates_all_selected_layers_and_preserves_unselected_rows():
    pool, leases, _, transaction = _fixture()
    original = _snapshots(pool)
    candidates = (
        (
            torch.full((2, 4, 3), 10.0),
            torch.full((2, 2, 3, 2), 11.0),
        ),
        (
            torch.full((2, 4, 3), 20.0),
            torch.full((2, 2, 3, 2), 21.0),
        ),
    )
    transaction.commit((leases[2], leases[0]), candidates)
    current = _snapshots(pool)
    for layer_index in range(2):
        torch.testing.assert_close(current[layer_index][0][2], candidates[layer_index][0][0])
        torch.testing.assert_close(current[layer_index][0][0], candidates[layer_index][0][1])
        torch.testing.assert_close(current[layer_index][1][2], candidates[layer_index][1][0])
        torch.testing.assert_close(current[layer_index][1][0], candidates[layer_index][1][1])
        torch.testing.assert_close(current[layer_index][0][1], original[layer_index][0][1])
        torch.testing.assert_close(current[layer_index][1][1], original[layer_index][1][1])


def _expect_error(function, error_type, message):
    try:
        function()
    except error_type as error:
        assert message in str(error), str(error)
    else:
        raise AssertionError(
            f"expected {error_type.__name__} containing {message!r}"
        )


def test_prevalidation_rejects_stale_and_malformed_without_writes():
    pool, leases, _, transaction = _fixture()
    original = _snapshots(pool)
    valid = (
        (torch.ones(2, 4, 3), torch.ones(2, 2, 3, 2)),
        (torch.ones(2, 4, 3), torch.ones(2, 2, 3, 2)),
    )
    stale = HybridStateLease(0, 0, 17)
    cases = (
        (
            lambda: transaction.commit((stale, leases[1]), valid),
            RuntimeError,
            "lease mismatch",
        ),
        (
            lambda: transaction.commit((leases[0], leases[1]), valid[:1]),
            ValueError,
            "candidate",
        ),
        (
            lambda: transaction.commit(
                (leases[0], leases[1]),
                (valid[0], (torch.ones(2, 4, 2), valid[1][1])),
            ),
            ValueError,
            "shape",
        ),
        (
            lambda: transaction.commit(
                (leases[0], leases[1]),
                (valid[0], "not a pair"),
            ),
            ValueError,
            "pair",
        ),
    )
    for function, error_type, message in cases:
        _expect_error(function, error_type, message)
        current = _snapshots(pool)
        for current_pair, original_pair in zip(current, original):
            torch.testing.assert_close(current_pair[0], original_pair[0])
            torch.testing.assert_close(current_pair[1], original_pair[1])


def test_later_layer_copy_failure_rolls_back_every_layer():
    pool, leases, adapters, transaction = _fixture()
    original = _snapshots(pool)
    calls = []
    original_copies = tuple(adapter._copy_component for adapter in adapters)

    def first_copy(destination, source):
        calls.append(("layer0", destination))
        return original_copies[0](destination, source)

    def second_copy(destination, source):
        calls.append(("layer1", destination))
        if len(calls) == 7:
            raise RuntimeError("injected cross-layer copy failure")
        return original_copies[1](destination, source)

    adapters[0]._copy_component = first_copy
    adapters[1]._copy_component = second_copy
    candidates = (
        (
            torch.full((2, 4, 3), 10.0),
            torch.full((2, 2, 3, 2), 11.0),
        ),
        (
            torch.full((2, 4, 3), 20.0),
            torch.full((2, 2, 3, 2), 21.0),
        ),
    )
    _expect_error(
        lambda: transaction.commit(
            (leases[2], leases[0]),
            candidates,
        ),
        RuntimeError,
        "cross-layer copy failure",
    )
    assert len(calls) == 7
    current = _snapshots(pool)
    for current_pair, original_pair in zip(current, original):
        torch.testing.assert_close(current_pair[0], original_pair[0])
        torch.testing.assert_close(current_pair[1], original_pair[1])


def test_constructor_rejects_invalid_duplicate_and_mixed_pool_adapters():
    pool, _, adapters, _ = _fixture()
    for value, message in (
        ([], "non-empty tuple"),
        ((adapters[0], "not an adapter"), "adapter"),
        ((adapters[0], adapters[0]), "unique"),
    ):
        _expect_error(
            lambda value=value: Qwen35CrossLayerStateTransaction(value),
            ValueError,
            message,
        )
    other_pool = HybridStateTensorPool(
        _layout((5,)),
        capacity=3,
        device="cpu",
    )
    mixed = Qwen35LayerStateAdapter(other_pool, 5)
    _expect_error(
        lambda: Qwen35CrossLayerStateTransaction((adapters[0], mixed)),
        ValueError,
        "same pool",
    )
    assert pool.capacity == 3


def main():
    test_gather_preserves_layer_and_lease_order_with_clone_isolation()
    test_commit_updates_all_selected_layers_and_preserves_unselected_rows()
    test_prevalidation_rejects_stale_and_malformed_without_writes()
    test_later_layer_copy_failure_rolls_back_every_layer()
    test_constructor_rejects_invalid_duplicate_and_mixed_pool_adapters()
    print("qwen35 cross layer state transaction tests passed")


if __name__ == "__main__":
    main()
