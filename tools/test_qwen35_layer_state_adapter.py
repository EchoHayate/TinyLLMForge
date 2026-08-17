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
HybridStateComponentSpec = hybrid.HybridStateComponentSpec
HybridStateLayout = hybrid.HybridStateLayout
HybridStateLease = hybrid.HybridStateLease
HybridStateTensorPool = hybrid.HybridStateTensorPool
Qwen35LayerStateAdapter = adapter_module.Qwen35LayerStateAdapter


def _fixture():
    layout = HybridStateLayout((
        HybridStateComponentSpec(
            3,
            "linear_convolution",
            (4, 3),
            torch.float32,
        ),
        HybridStateComponentSpec(
            3,
            "linear_recurrent",
            (2, 3, 2),
            torch.float32,
        ),
    ))
    pool = HybridStateTensorPool(layout, capacity=1, device="cpu")
    lease = HybridStateLease(0, 1, 17)
    pool.activate(lease)
    adapter = Qwen35LayerStateAdapter(pool, layer_index=3)
    return pool, lease, adapter


def _batch_fixture():
    layout = HybridStateLayout((
        HybridStateComponentSpec(
            3,
            "linear_convolution",
            (4, 3),
            torch.float32,
        ),
        HybridStateComponentSpec(
            3,
            "linear_recurrent",
            (2, 3, 2),
            torch.float32,
        ),
    ))
    pool = HybridStateTensorPool(layout, capacity=3, device="cpu")
    leases = (
        HybridStateLease(0, 1, 17),
        HybridStateLease(1, 1, 18),
        HybridStateLease(2, 1, 19),
    )
    for lease in leases:
        pool.activate(lease)
    convolution = pool.component_tensor(3, "linear_convolution")
    recurrent = pool.component_tensor(3, "linear_recurrent")
    for slot_id in range(3):
        convolution[slot_id].copy_(
            torch.arange(12).reshape(4, 3) + slot_id * 100
        )
        recurrent[slot_id].copy_(
            torch.arange(12).reshape(2, 3, 2) + slot_id * 1000
        )
    adapter = Qwen35LayerStateAdapter(pool, layer_index=3)
    return pool, leases, adapter


def test_gather_returns_clones_and_commit_updates_both_components() -> None:
    pool, lease, adapter = _fixture()
    convolution = pool.component_tensor(3, "linear_convolution")
    recurrent = pool.component_tensor(3, "linear_recurrent")
    convolution[0].copy_(torch.arange(12).reshape(4, 3))
    recurrent[0].copy_(torch.arange(12).reshape(2, 3, 2))
    gathered_conv, gathered_recurrent = adapter.gather(lease)
    assert gathered_conv.data_ptr() != convolution[0].data_ptr()
    assert gathered_recurrent.data_ptr() != recurrent[0].data_ptr()
    gathered_conv.add_(100)
    gathered_recurrent.add_(200)
    assert torch.max(convolution[0]).item() == 11
    assert torch.max(recurrent[0]).item() == 11
    adapter.commit(lease, gathered_conv, gathered_recurrent)
    torch.testing.assert_close(convolution[0], gathered_conv)
    torch.testing.assert_close(recurrent[0], gathered_recurrent)


def test_stale_and_invalid_candidates_leave_pool_unchanged() -> None:
    pool, lease, adapter = _fixture()
    convolution = pool.component_tensor(3, "linear_convolution")
    recurrent = pool.component_tensor(3, "linear_recurrent")
    convolution[0].fill_(3)
    recurrent[0].fill_(5)
    original_conv = convolution.clone()
    original_recurrent = recurrent.clone()
    stale = HybridStateLease(0, 0, 17)
    try:
        adapter.commit(
            stale,
            torch.ones(4, 3),
            torch.ones(2, 3, 2),
        )
    except RuntimeError:
        pass
    else:
        raise AssertionError("stale lease was accepted")
    for candidate_conv, candidate_recurrent in (
        (torch.ones(4, 2), torch.ones(2, 3, 2)),
        (torch.ones(4, 3), torch.ones(2, 2, 3)),
        (
            torch.ones(4, 3, dtype=torch.float64),
            torch.ones(2, 3, 2),
        ),
        (
            torch.ones(4, 3),
            torch.ones(2, 3, 2, device="meta"),
        ),
    ):
        try:
            adapter.commit(lease, candidate_conv, candidate_recurrent)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid candidate state was accepted")
    torch.testing.assert_close(convolution, original_conv)
    torch.testing.assert_close(recurrent, original_recurrent)


def test_second_copy_failure_rolls_back_both_pool_rows() -> None:
    pool, lease, adapter = _fixture()
    convolution = pool.component_tensor(3, "linear_convolution")
    recurrent = pool.component_tensor(3, "linear_recurrent")
    convolution[0].fill_(7)
    recurrent[0].fill_(9)
    original_conv = convolution.clone()
    original_recurrent = recurrent.clone()
    original_copy = adapter._copy_component
    calls = []

    def failing_copy(destination, source):
        calls.append(destination)
        if len(calls) == 2:
            raise RuntimeError("injected second copy failure")
        return original_copy(destination, source)

    adapter._copy_component = failing_copy
    try:
        adapter.commit(
            lease,
            torch.full((4, 3), 1.5),
            torch.full((2, 3, 2), 2.5),
        )
    except RuntimeError as error:
        assert "second copy failure" in str(error)
    else:
        raise AssertionError("injected copy failure was swallowed")
    torch.testing.assert_close(convolution, original_conv)
    torch.testing.assert_close(recurrent, original_recurrent)


def test_constructor_rejects_invalid_or_missing_layer() -> None:
    pool, _, _ = _fixture()
    for layer_index in (True, -1, 1.5):
        try:
            Qwen35LayerStateAdapter(pool, layer_index)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid layer index accepted")
    try:
        Qwen35LayerStateAdapter(pool, 4)
    except KeyError:
        pass
    else:
        raise AssertionError("missing layer components accepted")


def test_batch_gather_preserves_order_and_returns_contiguous_clones() -> None:
    pool, leases, adapter = _batch_fixture()
    convolution = pool.component_tensor(3, "linear_convolution")
    recurrent = pool.component_tensor(3, "linear_recurrent")
    gathered_conv, gathered_recurrent = adapter.gather_batch(
        (leases[2], leases[0]),
    )
    assert gathered_conv.shape == (2, 4, 3)
    assert gathered_recurrent.shape == (2, 2, 3, 2)
    assert gathered_conv.is_contiguous()
    assert gathered_recurrent.is_contiguous()
    torch.testing.assert_close(gathered_conv[0], convolution[2])
    torch.testing.assert_close(gathered_conv[1], convolution[0])
    torch.testing.assert_close(gathered_recurrent[0], recurrent[2])
    torch.testing.assert_close(gathered_recurrent[1], recurrent[0])
    gathered_conv.add_(5000)
    gathered_recurrent.add_(6000)
    assert torch.max(convolution).item() < 5000
    assert torch.max(recurrent).item() < 6000


def test_batch_commit_updates_selected_rows_only() -> None:
    pool, leases, adapter = _batch_fixture()
    convolution = pool.component_tensor(3, "linear_convolution")
    recurrent = pool.component_tensor(3, "linear_recurrent")
    untouched_conv = convolution[1].clone()
    untouched_recurrent = recurrent[1].clone()
    candidate_conv = torch.stack((
        torch.full((4, 3), 20.0),
        torch.full((4, 3), 10.0),
    ))
    candidate_recurrent = torch.stack((
        torch.full((2, 3, 2), 40.0),
        torch.full((2, 3, 2), 30.0),
    ))
    adapter.commit_batch(
        (leases[2], leases[0]),
        candidate_conv,
        candidate_recurrent,
    )
    torch.testing.assert_close(convolution[2], candidate_conv[0])
    torch.testing.assert_close(convolution[0], candidate_conv[1])
    torch.testing.assert_close(recurrent[2], candidate_recurrent[0])
    torch.testing.assert_close(recurrent[0], candidate_recurrent[1])
    torch.testing.assert_close(convolution[1], untouched_conv)
    torch.testing.assert_close(recurrent[1], untouched_recurrent)


def _expect_batch_error(function, error_type, message: str) -> None:
    try:
        function()
    except error_type as error:
        assert message in str(error), str(error)
    else:
        raise AssertionError(
            f"expected {error_type.__name__} containing {message!r}"
        )


def test_batch_lease_validation_fails_before_writes() -> None:
    pool, leases, adapter = _batch_fixture()
    convolution = pool.component_tensor(3, "linear_convolution")
    recurrent = pool.component_tensor(3, "linear_recurrent")
    original_conv = convolution.clone()
    original_recurrent = recurrent.clone()
    candidate_conv = torch.zeros(2, 4, 3)
    candidate_recurrent = torch.zeros(2, 2, 3, 2)
    stale = HybridStateLease(2, 0, 19)
    _expect_batch_error(
        lambda: adapter.commit_batch(
            (leases[0], stale),
            candidate_conv,
            candidate_recurrent,
        ),
        RuntimeError,
        "lease mismatch",
    )
    for batch in (
        (),
        [leases[0]],
        (leases[0], "not a lease"),
        (leases[0], leases[0]),
    ):
        _expect_batch_error(
            lambda batch=batch: adapter.gather_batch(batch),
            ValueError,
            "leases",
        )
        _expect_batch_error(
            lambda batch=batch: adapter.commit_batch(
                batch,
                candidate_conv[:len(batch)],
                candidate_recurrent[:len(batch)],
            ),
            ValueError,
            "leases",
        )
    torch.testing.assert_close(convolution, original_conv)
    torch.testing.assert_close(recurrent, original_recurrent)


def test_batch_candidate_validation_fails_before_writes() -> None:
    pool, leases, adapter = _batch_fixture()
    convolution = pool.component_tensor(3, "linear_convolution")
    recurrent = pool.component_tensor(3, "linear_recurrent")
    original_conv = convolution.clone()
    original_recurrent = recurrent.clone()
    cases = (
        (
            "not a tensor",
            torch.ones(2, 2, 3, 2),
            "convolution_states must be a tensor",
        ),
        (
            torch.ones(1, 4, 3),
            torch.ones(2, 2, 3, 2),
            "convolution_states shape",
        ),
        (
            torch.ones(2, 4, 2),
            torch.ones(2, 2, 3, 2),
            "convolution_states shape",
        ),
        (
            torch.ones(2, 4, 3),
            torch.ones(2, 2, 2, 3),
            "recurrent_states shape",
        ),
        (
            torch.ones(2, 4, 3, dtype=torch.float64),
            torch.ones(2, 2, 3, 2),
            "convolution_states dtype",
        ),
        (
            torch.ones(2, 4, 3),
            torch.ones(2, 2, 3, 2, device="meta"),
            "recurrent_states device",
        ),
    )
    for candidate_conv, candidate_recurrent, message in cases:
        _expect_batch_error(
            lambda candidate_conv=candidate_conv,
            candidate_recurrent=candidate_recurrent: adapter.commit_batch(
                (leases[2], leases[0]),
                candidate_conv,
                candidate_recurrent,
            ),
            ValueError,
            message,
        )
    torch.testing.assert_close(convolution, original_conv)
    torch.testing.assert_close(recurrent, original_recurrent)


def test_late_batch_copy_failure_rolls_back_all_selected_rows() -> None:
    pool, leases, adapter = _batch_fixture()
    convolution = pool.component_tensor(3, "linear_convolution")
    recurrent = pool.component_tensor(3, "linear_recurrent")
    original_conv = convolution.clone()
    original_recurrent = recurrent.clone()
    original_copy = adapter._copy_component
    calls = []

    def failing_copy(destination, source):
        calls.append(destination)
        if len(calls) == 4:
            raise RuntimeError("injected late batch copy failure")
        return original_copy(destination, source)

    adapter._copy_component = failing_copy
    try:
        adapter.commit_batch(
            (leases[2], leases[0]),
            torch.full((2, 4, 3), 7.5),
            torch.full((2, 2, 3, 2), 8.5),
        )
    except RuntimeError as error:
        assert "late batch copy failure" in str(error)
    else:
        raise AssertionError("injected batch copy failure was swallowed")
    assert len(calls) == 4
    torch.testing.assert_close(convolution, original_conv)
    torch.testing.assert_close(recurrent, original_recurrent)


def main() -> None:
    test_gather_returns_clones_and_commit_updates_both_components()
    test_stale_and_invalid_candidates_leave_pool_unchanged()
    test_second_copy_failure_rolls_back_both_pool_rows()
    test_constructor_rejects_invalid_or_missing_layer()
    test_batch_gather_preserves_order_and_returns_contiguous_clones()
    test_batch_commit_updates_selected_rows_only()
    test_batch_lease_validation_fails_before_writes()
    test_batch_candidate_validation_fails_before_writes()
    test_late_batch_copy_failure_rolls_back_all_selected_rows()
    print("qwen35 layer state adapter tests passed")


if __name__ == "__main__":
    main()
