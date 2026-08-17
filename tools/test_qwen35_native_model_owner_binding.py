from __future__ import annotations

import ast
import importlib.util
from pathlib import Path
import pickle
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
owner_module = _load_module(
    "tinyvllm.engine.qwen35_hybrid_model_owner",
    "tinyvllm/engine/qwen35_hybrid_model_owner.py",
)
identity_module = _load_module(
    "tinyvllm.engine.qwen35_hybrid_prefix_runtime_identity",
    "tinyvllm/engine/qwen35_hybrid_prefix_runtime_identity.py",
)

HybridStateComponentSpec = hybrid_module.HybridStateComponentSpec
HybridStateLayout = hybrid_module.HybridStateLayout
HybridStateRuntimeBridge = hybrid_module.HybridStateRuntimeBridge
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
Qwen35HybridModelOwner = owner_module.Qwen35HybridModelOwner
build_qwen35_hybrid_model_owner = (
    owner_module.build_qwen35_hybrid_model_owner
)
Qwen35HybridPrefixRuntimeIdentity = (
    identity_module.Qwen35HybridPrefixRuntimeIdentity
)
bind_qwen35_hybrid_prefix_runtime_identity = (
    identity_module.bind_qwen35_hybrid_prefix_runtime_identity
)


class Qwen35LoadedCheckpointCandidate:

    def __init__(self, owner, model_fingerprint):
        self.owner = owner
        self.model_fingerprint = model_fingerprint


class _Identity(nn.Module):
    def forward(self, tensor):
        return tensor


class _Linear(nn.Module):
    def forward(self, hidden, convolution, recurrent):
        return hidden, convolution, recurrent


class _Full(nn.Module):
    def forward(self, positions, hidden):
        return hidden


def _decoder(block_type):
    return Qwen35DecoderLayerShell(
        block_type=block_type,
        input_layernorm=_Identity(),
        post_attention_layernorm=_Identity(),
        mlp=_Identity(),
        full_attention=(
            _Full() if block_type == "full_attention" else None
        ),
        linear_attention=(
            _Linear() if block_type == "linear_attention" else None
        ),
    )


def _fixture():
    layout = HybridStateLayout(tuple(
        component
        for layer_index in (0, 2)
        for component in (
            HybridStateComponentSpec(
                layer_index,
                "linear_convolution",
                (2, 3),
                torch.float32,
            ),
            HybridStateComponentSpec(
                layer_index,
                "linear_recurrent",
                (2, 2, 2),
                torch.float32,
            ),
        )
    ))
    pool = HybridStateTensorPool(layout, 4, "cpu")
    adapters = (
        Qwen35LayerStateAdapter(pool, 0),
        Qwen35LayerStateAdapter(pool, 2),
    )
    transaction = Qwen35CrossLayerStateTransaction(adapters)
    stack = Qwen35PackedHeterogeneousLayerStack(
        (
            _decoder("linear_attention"),
            _decoder("full_attention"),
            _decoder("linear_attention"),
        ),
        transaction,
    )
    root = Qwen35PackedForCausalLM(
        _Identity(),
        stack,
        _Identity(),
        _Identity(),
    )
    return pool, transaction, stack, root


def test_model_owner_reuses_exact_transaction_pool_and_storage():
    pool, transaction, stack, model = _fixture()
    storage = {
        key: tensor.untyped_storage().data_ptr()
        for key, tensor in pool._tensors.items()
    }

    owner = build_qwen35_hybrid_model_owner(model)

    assert isinstance(owner, Qwen35HybridModelOwner)
    assert owner.model is model
    assert owner.layer_stack is stack
    assert owner.state_transaction is transaction
    assert owner.pool is pool
    assert owner.runtime_bridge.pool is pool
    assert {
        key: tensor.untyped_storage().data_ptr()
        for key, tensor in pool._tensors.items()
    } == storage


def test_model_owner_rejects_non_root_model():
    stack = _fixture()[2]
    for model in (object(), nn.Identity(), stack):
        try:
            build_qwen35_hybrid_model_owner(model)
        except ValueError as error:
            assert "root" in str(error)
        else:
            raise AssertionError("non-root model owner was accepted")


def test_model_owner_requires_exact_model_and_transaction_types():
    class DerivedModel(Qwen35PackedForCausalLM):
        pass

    class DerivedTransaction(Qwen35CrossLayerStateTransaction):
        pass

    _, transaction, stack, _ = _fixture()
    derived_model = DerivedModel(
        _Identity(),
        stack,
        _Identity(),
        _Identity(),
    )
    try:
        build_qwen35_hybrid_model_owner(derived_model)
    except ValueError as error:
        assert "root" in str(error)
    else:
        raise AssertionError("derived root model was accepted")

    derived_transaction = DerivedTransaction(transaction.adapters)
    exact_stack = Qwen35PackedHeterogeneousLayerStack(
        (
            _decoder("linear_attention"),
            _decoder("full_attention"),
            _decoder("linear_attention"),
        ),
        derived_transaction,
    )
    exact_model = Qwen35PackedForCausalLM(
        _Identity(),
        exact_stack,
        _Identity(),
        _Identity(),
    )
    try:
        build_qwen35_hybrid_model_owner(exact_model)
    except ValueError as error:
        assert "transaction" in str(error)
    else:
        raise AssertionError("derived state transaction was accepted")


def test_model_owner_rejects_incoherent_transaction_graph():
    _, transaction, stack, model = _fixture()
    stack.linear_indices = (0,)
    try:
        build_qwen35_hybrid_model_owner(model)
    except ValueError as error:
        assert "misaligned" in str(error)
    else:
        raise AssertionError("misaligned model transaction was accepted")

    pool, transaction, _, model = _fixture()
    other_pool = _fixture()[0]
    transaction.adapters = (
        transaction.adapters[0],
        Qwen35LayerStateAdapter(other_pool, 2),
    )
    assert transaction.pool is pool
    try:
        build_qwen35_hybrid_model_owner(model)
    except ValueError as error:
        assert "one pool" in str(error)
    else:
        raise AssertionError("multi-pool model transaction was accepted")


def _load_runner_method(name):
    path = ROOT / "tinyvllm/engine/model_runner.py"
    tree = ast.parse(path.read_text(), filename=str(path))
    class_node = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "ModelRunner"
    )
    method_node = next(
        node
        for node in class_node.body
        if isinstance(node, ast.FunctionDef) and node.name == name
    )
    method_node.decorator_list = []
    namespace = {
        "Qwen35HybridModelOwner": Qwen35HybridModelOwner,
        "Qwen35LoadedCheckpointCandidate": (
            Qwen35LoadedCheckpointCandidate
        ),
        "build_qwen35_hybrid_model_owner": (
            build_qwen35_hybrid_model_owner
        ),
        "_bind_qwen35_hybrid_prefix_runtime_identity": (
            bind_qwen35_hybrid_prefix_runtime_identity
        ),
    }
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
    return namespace[name]


def _runner(model):
    return types.SimpleNamespace(
        model=model,
        rank=1,
        hybrid_state_runtime_bridge=None,
        qwen35_hybrid_model_owner=None,
        qwen35_hybrid_prefix_restore_owner=None,
        qwen35_hybrid_prefix_restore_participant=None,
        qwen35_hybrid_prefix_publication_participant=None,
        qwen35_hybrid_prefix_runtime_identity=None,
        qwen35_hybrid_prefix_runtime_identity_owner=None,
    )


def _candidate(model_fingerprint="a" * 64):
    pool, _, _, model = _fixture()
    return (
        model,
        Qwen35LoadedCheckpointCandidate(
            build_qwen35_hybrid_model_owner(model),
            model_fingerprint,
        ),
        pool,
    )


def test_model_runner_atomically_binds_loaded_candidate():
    bind_owner = _load_runner_method(
        "bind_qwen35_hybrid_model_owner"
    )
    bind_candidate = _load_runner_method(
        "bind_qwen35_loaded_checkpoint_candidate"
    )
    model, candidate, pool = _candidate()
    runner = _runner(model)
    runner.bind_qwen35_hybrid_model_owner = (
        lambda owner: bind_owner(runner, owner)
    )

    first = bind_candidate(runner, candidate)
    second = bind_candidate(runner, candidate)

    assert first == second == {
        "participant_id": 1,
        "model_fingerprint": "a" * 64,
        "layout_fingerprint": pool.layout.fingerprint,
        "dtype": "float32",
    }
    assert runner.qwen35_hybrid_model_owner is candidate.owner
    assert runner.hybrid_state_runtime_bridge is (
        candidate.owner.runtime_bridge
    )
    assert runner.qwen35_hybrid_prefix_runtime_identity == (
        Qwen35HybridPrefixRuntimeIdentity(
            model_fingerprint="a" * 64,
            layout_fingerprint=pool.layout.fingerprint,
            dtype=torch.float32,
        )
    )
    assert runner.qwen35_hybrid_prefix_runtime_identity_owner is (
        candidate.owner
    )


def test_model_runner_candidate_binding_failures_are_pristine():
    bind_owner = _load_runner_method(
        "bind_qwen35_hybrid_model_owner"
    )
    bind_candidate = _load_runner_method(
        "bind_qwen35_loaded_checkpoint_candidate"
    )
    model, candidate, _ = _candidate()

    invalid = (
        object(),
        Qwen35LoadedCheckpointCandidate(
            candidate.owner,
            "bad",
        ),
    )
    for value in invalid:
        runner = _runner(model)
        runner.bind_qwen35_hybrid_model_owner = (
            lambda owner, runner=runner: bind_owner(runner, owner)
        )
        try:
            bind_candidate(runner, value)
        except (ValueError, RuntimeError):
            pass
        else:
            raise AssertionError("invalid loaded candidate was accepted")
        assert runner.qwen35_hybrid_model_owner is None
        assert runner.hybrid_state_runtime_bridge is None
        assert runner.qwen35_hybrid_prefix_runtime_identity is None

    _, other_candidate, _ = _candidate()
    runner = _runner(model)
    runner.bind_qwen35_hybrid_model_owner = (
        lambda owner: bind_owner(runner, owner)
    )
    try:
        bind_candidate(runner, other_candidate)
    except ValueError:
        pass
    else:
        raise AssertionError("candidate for another model was accepted")
    assert runner.qwen35_hybrid_model_owner is None
    assert runner.qwen35_hybrid_prefix_runtime_identity is None


def test_model_runner_candidate_binding_rejects_partial_or_conflicting_state():
    bind_owner = _load_runner_method(
        "bind_qwen35_hybrid_model_owner"
    )
    bind_candidate = _load_runner_method(
        "bind_qwen35_loaded_checkpoint_candidate"
    )
    model, candidate, _ = _candidate()

    runner = _runner(model)
    runner.bind_qwen35_hybrid_model_owner = (
        lambda owner: bind_owner(runner, owner)
    )
    bind_owner(runner, candidate.owner)
    try:
        bind_candidate(runner, candidate)
    except RuntimeError as error:
        assert "partial" in str(error)
    else:
        raise AssertionError("partial owner state was upgraded")
    assert runner.qwen35_hybrid_prefix_runtime_identity is None

    runner = _runner(model)
    runner.bind_qwen35_hybrid_model_owner = (
        lambda owner: bind_owner(runner, owner)
    )
    runner.qwen35_hybrid_prefix_runtime_identity = (
        bind_qwen35_hybrid_prefix_runtime_identity(
            candidate.owner,
            candidate.model_fingerprint,
        )
    )
    runner.qwen35_hybrid_prefix_runtime_identity_owner = candidate.owner
    try:
        bind_candidate(runner, candidate)
    except RuntimeError as error:
        assert "partial" in str(error)
    else:
        raise AssertionError("partial identity state was upgraded")
    assert runner.qwen35_hybrid_model_owner is None

    runner = _runner(model)
    runner.bind_qwen35_hybrid_model_owner = (
        lambda owner: bind_owner(runner, owner)
    )
    bind_candidate(runner, candidate)
    _, different, _ = _candidate()
    try:
        bind_candidate(runner, different)
    except RuntimeError as error:
        assert "already bound" in str(error)
    else:
        raise AssertionError("different candidate replaced binding")
    assert runner.qwen35_hybrid_model_owner is candidate.owner


def test_model_runner_binding_installs_exact_owner_and_bridge_once():
    bind = _load_runner_method("bind_qwen35_hybrid_model_owner")
    pool, _, _, model = _fixture()
    owner = build_qwen35_hybrid_model_owner(model)
    runner = _runner(model)

    bind(runner, owner)
    bind(runner, owner)

    assert runner.qwen35_hybrid_model_owner is owner
    assert runner.hybrid_state_runtime_bridge is owner.runtime_bridge
    assert runner.hybrid_state_runtime_bridge.pool is pool

    _, _, _, other_model = _fixture()
    other_owner = build_qwen35_hybrid_model_owner(other_model)
    try:
        bind(runner, other_owner)
    except (ValueError, RuntimeError):
        pass
    else:
        raise AssertionError("different model owner replaced binding")


def test_model_runner_binding_requires_exact_owner_type():
    class DerivedOwner(Qwen35HybridModelOwner):
        pass

    bind = _load_runner_method("bind_qwen35_hybrid_model_owner")
    _, _, _, model = _fixture()
    owner = build_qwen35_hybrid_model_owner(model)
    derived_owner = DerivedOwner(
        model=owner.model,
        layer_stack=owner.layer_stack,
        state_transaction=owner.state_transaction,
        pool=owner.pool,
        runtime_bridge=owner.runtime_bridge,
    )
    runner = _runner(model)

    try:
        bind(runner, derived_owner)
    except ValueError as error:
        assert "Qwen35HybridModelOwner" in str(error)
    else:
        raise AssertionError("derived model owner was accepted")
    assert runner.qwen35_hybrid_model_owner is None
    assert runner.hybrid_state_runtime_bridge is None


def test_model_runner_binding_rejects_forged_owner_graph():
    bind = _load_runner_method("bind_qwen35_hybrid_model_owner")
    pool, transaction, stack, model = _fixture()
    other_pool, other_transaction, other_stack, _ = _fixture()
    forged_owners = (
        Qwen35HybridModelOwner(
            model=model,
            layer_stack=other_stack,
            state_transaction=other_transaction,
            pool=other_pool,
            runtime_bridge=HybridStateRuntimeBridge(other_pool),
        ),
        Qwen35HybridModelOwner(
            model=model,
            layer_stack=stack,
            state_transaction=transaction,
            pool=other_pool,
            runtime_bridge=HybridStateRuntimeBridge(other_pool),
        ),
        Qwen35HybridModelOwner(
            model=model,
            layer_stack=stack,
            state_transaction=transaction,
            pool=pool,
            runtime_bridge=HybridStateRuntimeBridge(other_pool),
        ),
    )

    for owner in forged_owners:
        runner = _runner(model)
        try:
            bind(runner, owner)
        except ValueError as error:
            assert "ownership graph" in str(error)
        else:
            raise AssertionError("forged model owner graph was accepted")
        assert runner.qwen35_hybrid_model_owner is None
        assert runner.hybrid_state_runtime_bridge is None


def test_model_runner_binding_rejects_wrong_model_bridge_and_restore_pool():
    bind = _load_runner_method("bind_qwen35_hybrid_model_owner")
    pool, _, _, model = _fixture()
    owner = build_qwen35_hybrid_model_owner(model)
    _, _, _, other_model = _fixture()

    runner = _runner(other_model)
    try:
        bind(runner, owner)
    except ValueError as error:
        assert "current model" in str(error)
    else:
        raise AssertionError("owner for another model was bound")

    runner = _runner(model)
    runner.hybrid_state_runtime_bridge = HybridStateRuntimeBridge(
        _fixture()[0]
    )
    try:
        bind(runner, owner)
    except RuntimeError as error:
        assert "runtime bridge" in str(error)
    else:
        raise AssertionError("different runtime bridge was replaced")

    runner = _runner(model)
    runner.qwen35_hybrid_prefix_restore_owner = types.SimpleNamespace(
        pool=_fixture()[0]
    )
    try:
        bind(runner, owner)
    except RuntimeError as error:
        assert "restore owner" in str(error)
    else:
        raise AssertionError("different restore owner pool was accepted")

    runner = _runner(model)
    runner.qwen35_hybrid_prefix_restore_participant = (
        types.SimpleNamespace(pool=_fixture()[0])
    )
    try:
        bind(runner, owner)
    except RuntimeError as error:
        assert "participant" in str(error)
    else:
        raise AssertionError("different participant pool was accepted")


def test_bind_current_model_returns_identity_and_non_qwen35_is_pristine():
    bind_owner = _load_runner_method(
        "bind_qwen35_hybrid_model_owner"
    )
    bind_current = _load_runner_method(
        "bind_current_qwen35_hybrid_model"
    )
    pool, _, _, model = _fixture()
    runner = _runner(model)
    runner.bind_qwen35_hybrid_model_owner = (
        lambda owner: bind_owner(runner, owner)
    )

    identity = bind_current(runner)

    assert identity == {
        "participant_id": 1,
        "capacity": 4,
        "layout_fingerprint": pool.layout.fingerprint,
        "bytes_per_slot": pool.layout.bytes_per_slot,
        "linear_layer_indices": (0, 2),
    }
    assert pickle.loads(pickle.dumps(identity)) == identity

    runner = _runner(object())
    runner.bind_qwen35_hybrid_model_owner = (
        lambda owner: bind_owner(runner, owner)
    )
    try:
        bind_current(runner)
    except ValueError:
        pass
    else:
        raise AssertionError("non-Qwen3.5 current model was bound")
    assert runner.qwen35_hybrid_model_owner is None
    assert runner.hybrid_state_runtime_bridge is None


def test_scheduler_guard_and_engine_step_remain_unchanged():
    scheduler_source = (
        ROOT / "tinyvllm/engine/scheduler.py"
    ).read_text()
    assert (
        "hybrid prefix reuse requires aligned state snapshot"
        in scheduler_source
    )
    engine_source = (
        ROOT / "tinyvllm/engine/llm_engine.py"
    ).read_text()
    tree = ast.parse(engine_source)
    class_node = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "LLMEngine"
    )
    step_node = next(
        node
        for node in class_node.body
        if isinstance(node, ast.FunctionDef) and node.name == "step"
    )
    step_source = ast.get_source_segment(engine_source, step_node)
    assert "bind_current_qwen35_hybrid_model" not in step_source


def _run():
    tests = [
        value
        for name, value in sorted(globals().items())
        if name.startswith("test_") and callable(value)
    ]
    for test in tests:
        test()
    print(
        "qwen35 native model owner binding tests passed "
        f"({len(tests)} tests)"
    )


if __name__ == "__main__":
    _run()
