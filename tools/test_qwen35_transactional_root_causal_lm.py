from __future__ import annotations

import importlib.util
from contextlib import contextmanager
from pathlib import Path
import sys
import types

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]


def _load_module(module_name, relative_path):
    path = ROOT / relative_path
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


for package_name in (
    "tinyvllm",
    "tinyvllm.engine",
    "tinyvllm.layers",
    "tinyvllm.models",
):
    package = types.ModuleType(package_name)
    package.__path__ = [str(ROOT / package_name.replace(".", "/"))]
    sys.modules[package_name] = package

hybrid_module = _load_module(
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
decoder_module = _load_module(
    "tinyvllm.layers.qwen35_decoder_layer",
    "tinyvllm/layers/qwen35_decoder_layer.py",
)
stack_module = _load_module(
    "tinyvllm.layers.qwen35_packed_layer_stack",
    "tinyvllm/layers/qwen35_packed_layer_stack.py",
)
root_module = _load_module(
    "tinyvllm.models.qwen35_packed",
    "tinyvllm/models/qwen35_packed.py",
)

HybridStateComponentSpec = hybrid_module.HybridStateComponentSpec
HybridStateLayout = hybrid_module.HybridStateLayout
HybridStateLease = hybrid_module.HybridStateLease
HybridStateTensorPool = hybrid_module.HybridStateTensorPool
Qwen35LayerStateAdapter = adapter_module.Qwen35LayerStateAdapter
Qwen35CrossLayerStateTransaction = (
    transaction_module.Qwen35CrossLayerStateTransaction
)
Qwen35DecoderLayerShell = decoder_module.Qwen35DecoderLayerShell
Qwen35PackedHeterogeneousLayerStack = (
    stack_module.Qwen35PackedHeterogeneousLayerStack
)
Qwen35PackedForCausalLM = root_module.Qwen35PackedForCausalLM


class _Identity(nn.Module):
    def forward(self, tensor):
        return tensor


class _Zero(nn.Module):
    def forward(self, tensor):
        return torch.zeros_like(tensor)


class _LinearMixer(nn.Module):
    def __init__(self):
        super().__init__()
        self.fail = False

    def forward(self, hidden, convolution, recurrent):
        if self.fail:
            raise RuntimeError("injected layer failure")
        delta = hidden.sum().to(convolution.dtype)
        return (
            hidden * 0.5,
            convolution + delta,
            recurrent + delta * 0.25,
        )


def _fixture():
    layout = HybridStateLayout((
        HybridStateComponentSpec(
            0,
            "linear_convolution",
            (2, 2),
            torch.float32,
        ),
        HybridStateComponentSpec(
            0,
            "linear_recurrent",
            (2, 2, 2),
            torch.float32,
        ),
    ))
    pool = HybridStateTensorPool(layout, 2, "cpu")
    leases = (
        HybridStateLease(0, 1, 101),
        HybridStateLease(1, 1, 102),
    )
    for lease in leases:
        pool.activate(lease)
    pool.component_tensor(0, "linear_convolution").copy_(
        torch.arange(8, dtype=torch.float32).reshape(2, 2, 2)
    )
    pool.component_tensor(0, "linear_recurrent").copy_(
        torch.arange(16, dtype=torch.float32).reshape(2, 2, 2, 2)
    )
    adapter = Qwen35LayerStateAdapter(pool, 0)
    transaction = Qwen35CrossLayerStateTransaction((adapter,))
    mixer = _LinearMixer()
    decoder = Qwen35DecoderLayerShell(
        block_type="linear_attention",
        input_layernorm=_Identity(),
        post_attention_layernorm=_Identity(),
        mlp=_Zero(),
        linear_attention=mixer,
    )
    stack = Qwen35PackedHeterogeneousLayerStack(
        (decoder,),
        transaction,
    )
    return pool, leases, stack, mixer


def _snapshot(pool):
    return (
        pool.component_tensor(0, "linear_convolution").clone(),
        pool.component_tensor(0, "linear_recurrent").clone(),
    )


def _ownership_snapshot(pool):
    return {
        "bindings": dict(pool._bindings),
        "tensors": {
            key: {
                "object_id": id(tensor),
                "storage_ptr": tensor.untyped_storage().data_ptr(),
                "storage_offset": tensor.storage_offset(),
                "version": tensor._version,
                "value": tensor.clone(),
            }
            for key, tensor in pool._tensors.items()
        },
    }


def _assert_ownership_snapshot(pool, snapshot):
    assert pool._bindings == snapshot["bindings"]
    assert set(pool._tensors) == set(snapshot["tensors"])
    for key, tensor in pool._tensors.items():
        expected = snapshot["tensors"][key]
        assert id(tensor) == expected["object_id"]
        assert tensor.untyped_storage().data_ptr() == expected["storage_ptr"]
        assert tensor.storage_offset() == expected["storage_offset"]
        assert tensor._version == expected["version"]
        torch.testing.assert_close(tensor, expected["value"])


def test_staged_prepare_is_read_only_and_commit_applies_candidates():
    pool, leases, stack, _ = _fixture()
    hidden = torch.arange(12, dtype=torch.float32).reshape(4, 3)
    positions = torch.arange(4, dtype=torch.int64)
    counts = (1, 3)
    before = _snapshot(pool)

    prepared_hidden, candidates = stack.prepare(
        leases,
        counts,
        positions,
        hidden,
    )

    torch.testing.assert_close(prepared_hidden, hidden * 1.5)
    current = _snapshot(pool)
    torch.testing.assert_close(current[0], before[0])
    torch.testing.assert_close(current[1], before[1])
    assert len(candidates) == 1

    stack.commit(leases, candidates)

    current = _snapshot(pool)
    torch.testing.assert_close(current[0], candidates[0][0])
    torch.testing.assert_close(current[1], candidates[0][1])


def test_existing_forward_composes_prepare_then_commit():
    pool, leases, stack, _ = _fixture()
    hidden = torch.arange(12, dtype=torch.float32).reshape(4, 3)
    positions = torch.arange(4, dtype=torch.int64)
    counts = (1, 3)
    calls = []
    original_prepare = stack.prepare
    original_commit = stack.commit

    def recording_prepare(*args):
        calls.append("prepare")
        return original_prepare(*args)

    def recording_commit(*args):
        calls.append("commit")
        return original_commit(*args)

    stack.prepare = recording_prepare
    stack.commit = recording_commit

    output = stack(leases, counts, positions, hidden)

    torch.testing.assert_close(output, hidden * 1.5)
    assert calls == ["prepare", "commit"]
    current = _snapshot(pool)
    assert not torch.equal(current[0], torch.arange(
        8,
        dtype=torch.float32,
    ).reshape(2, 2, 2))


class _Embedding(nn.Module):
    def __init__(self, events):
        super().__init__()
        self.events = events
        self.fail = False
        self.invalid = None

    def forward(self, input_ids):
        self.events.append("embed")
        if self.fail:
            raise RuntimeError("injected embedding failure")
        values = input_ids.to(torch.float32)
        output = torch.stack((values, values + 1, values + 2), dim=-1)
        if self.invalid == "rank":
            return output[:, 0]
        if self.invalid == "integer":
            return output.to(torch.int64)
        return output


class _FinalNorm(nn.Module):
    def __init__(self, events):
        super().__init__()
        self.events = events
        self.fail = False
        self.invalid = None

    def forward(self, hidden):
        self.events.append("norm")
        if self.fail:
            raise RuntimeError("injected norm failure")
        if self.invalid == "rank":
            return hidden[:, 0]
        if self.invalid == "width":
            return hidden[:, :2]
        if self.invalid == "integer":
            return hidden.to(torch.int64)
        return hidden * 2 + 1


class _LMHead(nn.Module):
    def __init__(self, events, observe=None):
        super().__init__()
        self.events = events
        self.observe = observe
        self.fail = False
        self.invalid = None
        self.selected_rows = None
        self.output_none = False

    def forward(self, hidden):
        self.events.append("head")
        if self.observe is not None:
            self.observe()
        if self.fail:
            raise RuntimeError("injected head failure")
        if self.output_none:
            return None
        if self.invalid == "rank":
            return hidden[:, 0]
        if self.invalid == "empty_vocab":
            return hidden[:, :0]
        if self.invalid == "integer":
            return hidden.to(torch.int64)
        logits = torch.stack((
            hidden.sum(dim=-1),
            hidden[:, 0] - hidden[:, 1],
        ), dim=-1)
        if self.invalid == "zero_rows":
            return logits[:0]
        if self.invalid == "excess_rows":
            return torch.cat((logits, logits[:1]), dim=0)
        if self.selected_rows is not None:
            logits = logits[self.selected_rows]
        return logits


def _root_fixture():
    pool, leases, stack, mixer = _fixture()
    events = []
    before = _snapshot(pool)
    embedding = _Embedding(events)
    final_norm = _FinalNorm(events)
    head = _LMHead(
        events,
        observe=lambda: (
            torch.testing.assert_close(_snapshot(pool)[0], before[0]),
            torch.testing.assert_close(_snapshot(pool)[1], before[1]),
        ),
    )
    root = Qwen35PackedForCausalLM(
        embedding,
        stack,
        final_norm,
        head,
    )
    original_prepare = stack.prepare
    original_commit = stack.commit

    def recording_prepare(*args):
        events.append("prepare")
        return original_prepare(*args)

    def recording_commit(*args):
        events.append("commit")
        return original_commit(*args)

    stack.prepare = recording_prepare
    stack.commit = recording_commit
    return (
        pool,
        leases,
        stack,
        mixer,
        embedding,
        final_norm,
        head,
        root,
        events,
    )


def _root_inputs():
    return (
        (1, 3),
        torch.tensor([2, 4, 6, 8], dtype=torch.int64),
        torch.arange(4, dtype=torch.int64),
    )


def _expect_error(function, error_type, message):
    try:
        function()
    except error_type as error:
        assert message in str(error), str(error)
    else:
        raise AssertionError(f"expected {error_type.__name__}: {message}")


@contextmanager
def _distributed_role(*, initialized, world_size=1, rank=0):
    distributed = torch.distributed
    originals = {
        "is_available": distributed.is_available,
        "is_initialized": distributed.is_initialized,
        "get_world_size": distributed.get_world_size,
        "get_rank": distributed.get_rank,
    }
    distributed.is_available = lambda: True
    distributed.is_initialized = lambda: initialized
    distributed.get_world_size = lambda: world_size
    distributed.get_rank = lambda: rank
    try:
        yield
    finally:
        for name, value in originals.items():
            setattr(distributed, name, value)


def test_root_retains_components_orders_logits_before_commit_and_matches_math():
    (
        pool,
        leases,
        stack,
        _,
        embedding,
        final_norm,
        head,
        root,
        events,
    ) = _root_fixture()
    counts, input_ids, positions = _root_inputs()
    before = _snapshot(pool)

    hidden, logits = root.run_step(
        leases,
        counts,
        input_ids,
        positions,
    )

    embedded = torch.stack((
        input_ids.to(torch.float32),
        input_ids.to(torch.float32) + 1,
        input_ids.to(torch.float32) + 2,
    ), dim=-1)
    expected_hidden = embedded * 1.5 * 2 + 1
    expected_logits = torch.stack((
        expected_hidden.sum(dim=-1),
        expected_hidden[:, 0] - expected_hidden[:, 1],
    ), dim=-1)
    torch.testing.assert_close(hidden, expected_hidden)
    torch.testing.assert_close(logits, expected_logits)
    assert events == ["embed", "prepare", "norm", "head", "commit"]
    assert root.embed_tokens is embedding
    assert root.layer_stack is stack
    assert root.final_norm is final_norm
    assert root.lm_head is head
    current = _snapshot(pool)
    assert not torch.equal(current[0], before[0])
    assert not torch.equal(current[1], before[1])


def test_root_input_embeds_override_skips_embedding():
    (
        _,
        leases,
        _,
        _,
        _,
        _,
        _,
        root,
        events,
    ) = _root_fixture()
    counts, input_ids, positions = _root_inputs()
    input_embeds = torch.arange(
        12,
        dtype=torch.float32,
    ).reshape(4, 3)

    hidden, _ = root.run_step(
        leases,
        counts,
        input_ids,
        positions,
        input_embeds=input_embeds,
    )

    torch.testing.assert_close(hidden, input_embeds * 1.5 * 2 + 1)
    assert events == ["prepare", "norm", "head", "commit"]


def test_root_accepts_prefill_selected_logits_rows_before_commit():
    (
        pool,
        leases,
        _,
        _,
        _,
        _,
        head,
        root,
        events,
    ) = _root_fixture()
    counts, input_ids, positions = _root_inputs()
    before = _snapshot(pool)
    head.selected_rows = torch.tensor([3], dtype=torch.int64)

    hidden, logits = root.run_step(
        leases,
        counts,
        input_ids,
        positions,
    )

    assert hidden.shape == (4, 3)
    assert logits.shape == (1, 2)
    assert events == ["embed", "prepare", "norm", "head", "commit"]
    current = _snapshot(pool)
    assert not torch.equal(current[0], before[0])
    assert not torch.equal(current[1], before[1])


def test_tp4_non_root_accepts_none_and_commits():
    (
        pool,
        leases,
        _,
        _,
        _,
        _,
        head,
        root,
        events,
    ) = _root_fixture()
    counts, input_ids, positions = _root_inputs()
    before = _snapshot(pool)
    head.output_none = True

    with _distributed_role(initialized=True, world_size=4, rank=1):
        hidden, logits = root.run_step(
            leases,
            counts,
            input_ids,
            positions,
        )

    assert hidden.shape == (4, 3)
    assert logits is None
    assert events == ["embed", "prepare", "norm", "head", "commit"]
    current = _snapshot(pool)
    assert not torch.equal(current[0], before[0])
    assert not torch.equal(current[1], before[1])


def test_tp4_rank_zero_rejects_none_without_commit():
    (
        pool,
        leases,
        _,
        _,
        _,
        _,
        head,
        root,
        events,
    ) = _root_fixture()
    counts, input_ids, positions = _root_inputs()
    before = _snapshot(pool)
    head.output_none = True

    with _distributed_role(initialized=True, world_size=4, rank=0):
        _expect_error(
            lambda: root.run_step(
                leases,
                counts,
                input_ids,
                positions,
            ),
            ValueError,
            "rank zero",
        )

    current = _snapshot(pool)
    torch.testing.assert_close(current[0], before[0])
    torch.testing.assert_close(current[1], before[1])
    assert "commit" not in events


def test_tp4_non_root_rejects_tensor_without_commit():
    (
        pool,
        leases,
        _,
        _,
        _,
        _,
        _,
        root,
        events,
    ) = _root_fixture()
    counts, input_ids, positions = _root_inputs()
    before = _snapshot(pool)

    with _distributed_role(initialized=True, world_size=4, rank=3):
        _expect_error(
            lambda: root.run_step(
                leases,
                counts,
                input_ids,
                positions,
            ),
            ValueError,
            "non-root",
        )

    current = _snapshot(pool)
    torch.testing.assert_close(current[0], before[0])
    torch.testing.assert_close(current[1], before[1])
    assert "commit" not in events


def test_tp1_and_uninitialized_distributed_still_require_tensor():
    for initialized in (False, True):
        (
            pool,
            leases,
            _,
            _,
            _,
            _,
            head,
            root,
            events,
        ) = _root_fixture()
        counts, input_ids, positions = _root_inputs()
        before = _snapshot(pool)
        head.output_none = True

        with _distributed_role(
            initialized=initialized,
            world_size=1,
            rank=0,
        ):
            _expect_error(
                lambda: root.run_step(
                    leases,
                    counts,
                    input_ids,
                    positions,
                ),
                ValueError,
                "tensor",
            )

        current = _snapshot(pool)
        torch.testing.assert_close(current[0], before[0])
        torch.testing.assert_close(current[1], before[1])
        assert "commit" not in events


def test_root_failures_before_commit_preserve_state():
    for failure in ("embedding", "layer", "norm", "head"):
        (
            pool,
            leases,
            _,
            mixer,
            embedding,
            final_norm,
            head,
            root,
            events,
        ) = _root_fixture()
        counts, input_ids, positions = _root_inputs()
        before = _snapshot(pool)
        if failure == "embedding":
            embedding.fail = True
        elif failure == "layer":
            mixer.fail = True
        elif failure == "norm":
            final_norm.fail = True
        else:
            head.fail = True

        _expect_error(
            lambda: root.run_step(
                leases,
                counts,
                input_ids,
                positions,
            ),
            RuntimeError,
            f"injected {failure}",
        )

        current = _snapshot(pool)
        torch.testing.assert_close(current[0], before[0])
        torch.testing.assert_close(current[1], before[1])
        assert "commit" not in events


def test_head_failure_preserves_exact_pool_ownership_and_success_releases():
    (
        pool,
        leases,
        _,
        _,
        _,
        _,
        head,
        root,
        events,
    ) = _root_fixture()
    counts, input_ids, positions = _root_inputs()
    before = _ownership_snapshot(pool)
    head.fail = True

    _expect_error(
        lambda: root.run_step(
            leases,
            counts,
            input_ids,
            positions,
        ),
        RuntimeError,
        "injected head failure",
    )

    _assert_ownership_snapshot(pool, before)
    assert "commit" not in events

    head.fail = False
    root.run_step(
        leases,
        counts,
        input_ids,
        positions,
    )
    assert any(
        not torch.equal(
            tensor,
            before["tensors"][key]["value"],
        )
        for key, tensor in pool._tensors.items()
    )
    for lease in leases:
        pool.release(lease)
    assert pool._bindings == {}
    assert all(
        not bool(torch.count_nonzero(tensor))
        for tensor in pool._tensors.values()
    )


def test_root_commit_failure_rolls_back_state():
    (
        pool,
        leases,
        stack,
        _,
        _,
        _,
        _,
        root,
        events,
    ) = _root_fixture()
    counts, input_ids, positions = _root_inputs()
    before = _snapshot(pool)
    adapter = stack.state_transaction.adapters[0]
    original_copy = adapter._copy_component
    copy_calls = 0

    def failing_copy(destination, source):
        nonlocal copy_calls
        copy_calls += 1
        if copy_calls == 2:
            raise RuntimeError("injected commit failure")
        return original_copy(destination, source)

    adapter._copy_component = failing_copy

    _expect_error(
        lambda: root.run_step(
            leases,
            counts,
            input_ids,
            positions,
        ),
        RuntimeError,
        "injected commit failure",
    )

    current = _snapshot(pool)
    torch.testing.assert_close(current[0], before[0])
    torch.testing.assert_close(current[1], before[1])
    assert events[-1] == "commit"


def test_root_rejects_invalid_components_inputs_and_outputs():
    pool, leases, stack, _ = _fixture()
    events = []
    embedding = _Embedding(events)
    norm = _FinalNorm(events)
    head = _LMHead(events)
    for components, message in (
        ((object(), stack, norm, head), "embed_tokens"),
        ((embedding, object(), norm, head), "layer_stack"),
        ((embedding, stack, object(), head), "final_norm"),
        ((embedding, stack, norm, object()), "lm_head"),
    ):
        _expect_error(
            lambda components=components: Qwen35PackedForCausalLM(
                *components
            ),
            ValueError,
            message,
        )

    root = Qwen35PackedForCausalLM(
        embedding,
        stack,
        norm,
        head,
    )
    counts, input_ids, positions = _root_inputs()
    invalid_calls = (
        (
            lambda: root.run_step(
                leases,
                counts,
                input_ids.to(torch.float32),
                positions,
            ),
            "integer dtype",
        ),
        (
            lambda: root.run_step(
                leases,
                (1, 2),
                input_ids,
                positions,
            ),
            "token count",
        ),
        (
            lambda: root.run_step(
                leases,
                counts,
                input_ids,
                positions,
                input_embeds=torch.ones(4),
            ),
            "input_embeds",
        ),
    )
    for call, message in invalid_calls:
        _expect_error(call, ValueError, message)

    before = _snapshot(pool)
    for invalid in ("rank", "integer"):
        embedding.invalid = invalid
        _expect_error(
            lambda: root.run_step(
                leases,
                counts,
                input_ids,
                positions,
            ),
            ValueError,
            "embed_tokens",
        )
    embedding.invalid = None
    for invalid in ("rank", "width", "integer"):
        norm.invalid = invalid
        _expect_error(
            lambda: root.run_step(
                leases,
                counts,
                input_ids,
                positions,
            ),
            ValueError,
            "final_norm",
        )
    norm.invalid = None
    for invalid, message in (
        ("rank", "lm_head"),
        ("empty_vocab", "vocabulary"),
        ("integer", "floating point"),
        ("zero_rows", "logit row count"),
        ("excess_rows", "logit row count"),
    ):
        head.invalid = invalid
        _expect_error(
            lambda: root.run_step(
                leases,
                counts,
                input_ids,
                positions,
            ),
            ValueError,
            message,
        )
    current = _snapshot(pool)
    torch.testing.assert_close(current[0], before[0])
    torch.testing.assert_close(current[1], before[1])


def test_root_shell_is_not_selected_or_auto_invoked():
    model_runner = (
        ROOT / "tinyvllm/engine/model_runner.py"
    ).read_text()
    engine = (ROOT / "tinyvllm/engine/llm_engine.py").read_text()
    scheduler = (ROOT / "tinyvllm/engine/scheduler.py").read_text()
    assert "self.model = Qwen3ForCausalLM(hf_config)" in model_runner
    assert "Qwen35PackedForCausalLM" not in model_runner
    assert "Qwen35PackedForCausalLM" not in engine
    assert "Qwen35PackedForCausalLM" not in scheduler
    assert (
        "hybrid prefix reuse requires aligned state snapshot"
        in scheduler
    )


def _run():
    tests = [
        value
        for name, value in sorted(globals().items())
        if name.startswith("test_") and callable(value)
    ]
    for test in tests:
        test()
    print(
        "qwen35 transactional root causal lm tests passed "
        f"({len(tests)} tests)"
    )


if __name__ == "__main__":
    _run()
