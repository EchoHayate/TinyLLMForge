from __future__ import annotations

import ast
import importlib.util
from pathlib import Path
import pickle
import sys
import types

import torch

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


class _PackedModel:
    pass


class _PackedStack:
    pass


class _StateTransaction:
    pass


packed_module = types.ModuleType("tinyvllm.models.qwen35_packed")
packed_module.Qwen35PackedForCausalLM = _PackedModel
sys.modules[packed_module.__name__] = packed_module

stack_module = types.ModuleType(
    "tinyvllm.layers.qwen35_packed_layer_stack"
)
stack_module.Qwen35PackedHeterogeneousLayerStack = _PackedStack
sys.modules[stack_module.__name__] = stack_module

transaction_module = types.ModuleType(
    "tinyvllm.engine.qwen35_state_transaction"
)
transaction_module.Qwen35CrossLayerStateTransaction = _StateTransaction
sys.modules[transaction_module.__name__] = transaction_module

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
HybridStateTensorPool = hybrid_module.HybridStateTensorPool
Qwen35HybridModelOwner = owner_module.Qwen35HybridModelOwner
Qwen35HybridPrefixRuntimeIdentity = (
    identity_module.Qwen35HybridPrefixRuntimeIdentity
)
bind_qwen35_hybrid_prefix_runtime_identity = (
    identity_module.bind_qwen35_hybrid_prefix_runtime_identity
)

MODEL_SHA = "a" * 64


def test_llm_engine_imports_torch_for_runtime_identity_dtype_mapping():
    source = (ROOT / "tinyvllm/engine/llm_engine.py").read_text()
    tree = ast.parse(source)
    assert any(
        isinstance(node, ast.Import)
        and any(alias.name == "torch" for alias in node.names)
        for node in tree.body
    )


def _pool(dtype=torch.float32, recurrent_dtype=None):
    recurrent_dtype = (
        dtype if recurrent_dtype is None else recurrent_dtype
    )
    layout = HybridStateLayout((
        HybridStateComponentSpec(
            layer_index=0,
            role="linear_convolution",
            shape=(2, 2),
            dtype=dtype,
        ),
        HybridStateComponentSpec(
            layer_index=0,
            role="linear_recurrent",
            shape=(2, 2),
            dtype=recurrent_dtype,
        ),
    ))
    return HybridStateTensorPool(layout, 2, "cpu")


def _owner(pool=None):
    pool = _pool() if pool is None else pool
    return Qwen35HybridModelOwner(
        model=object(),
        layer_stack=object(),
        state_transaction=object(),
        pool=pool,
        runtime_bridge=types.SimpleNamespace(pool=pool),
    )


def _expect_error(function, message):
    try:
        function()
    except (ValueError, RuntimeError) as error:
        assert message in str(error), str(error)
    else:
        raise AssertionError(f"expected error containing {message!r}")


def test_identity_derives_exact_layout_and_dtype():
    owner = _owner()

    identity = bind_qwen35_hybrid_prefix_runtime_identity(
        owner,
        MODEL_SHA,
    )

    assert identity == Qwen35HybridPrefixRuntimeIdentity(
        model_fingerprint=MODEL_SHA,
        layout_fingerprint=owner.pool.layout.fingerprint,
        dtype=torch.float32,
    )
    row = identity.rank_row(1)
    assert row == {
        "participant_id": 1,
        "model_fingerprint": MODEL_SHA,
        "layout_fingerprint": owner.pool.layout.fingerprint,
        "dtype": "float32",
    }
    assert pickle.loads(pickle.dumps(row)) == row


def test_identity_accepts_stable_recurrent_dtype_and_rejects_invalid_inputs():
    _expect_error(
        lambda: bind_qwen35_hybrid_prefix_runtime_identity(
            object(),
            MODEL_SHA,
        ),
        "owner",
    )
    for value in ("", "A" * 64, "g" * 64, "a" * 63, None):
        _expect_error(
            lambda value=value: (
                bind_qwen35_hybrid_prefix_runtime_identity(
                    _owner(),
                    value,
                )
            ),
            "SHA256",
        )
    owner = _owner(_pool(torch.bfloat16, torch.float32))
    identity = bind_qwen35_hybrid_prefix_runtime_identity(
        owner,
        MODEL_SHA,
    )
    assert identity.dtype == torch.bfloat16
    assert identity.layout_fingerprint == owner.pool.layout.fingerprint


def _load_class_method(relative_path, class_name, method_name, namespace):
    path = ROOT / relative_path
    source = path.read_text()
    tree = ast.parse(source, filename=str(path))
    class_node = next(
        node for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == class_name
    )
    method_node = next(
        node for node in class_node.body
        if isinstance(node, ast.FunctionDef)
        and node.name == method_name
    )
    method_node.decorator_list = []
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
    return namespace[method_name]


def _runner_method():
    return _load_class_method(
        "tinyvllm/engine/model_runner.py",
        "ModelRunner",
        "bind_qwen35_hybrid_prefix_runtime_identity",
        {
            "_bind_qwen35_hybrid_prefix_runtime_identity": (
                bind_qwen35_hybrid_prefix_runtime_identity
            ),
        },
    )


def _runner(owner=None, rank=1):
    return types.SimpleNamespace(
        rank=rank,
        qwen35_hybrid_model_owner=_owner() if owner is None else owner,
        qwen35_hybrid_prefix_runtime_identity=None,
        qwen35_hybrid_prefix_runtime_identity_owner=None,
    )


def test_model_runner_binds_once_and_repeats_exact_identity():
    bind = _runner_method()
    runner = _runner()

    first = bind(runner, MODEL_SHA)
    second = bind(runner, MODEL_SHA)

    assert first == second
    assert runner.qwen35_hybrid_prefix_runtime_identity == (
        Qwen35HybridPrefixRuntimeIdentity(
            model_fingerprint=MODEL_SHA,
            layout_fingerprint=(
                runner.qwen35_hybrid_model_owner.pool.layout.fingerprint
            ),
            dtype=torch.float32,
        )
    )
    assert runner.qwen35_hybrid_prefix_runtime_identity_owner is (
        runner.qwen35_hybrid_model_owner
    )


def test_model_runner_rejects_replacement_and_owner_drift():
    bind = _runner_method()
    runner = _runner()
    bind(runner, MODEL_SHA)

    _expect_error(
        lambda: bind(runner, "b" * 64),
        "already bound",
    )
    runner.qwen35_hybrid_model_owner = _owner()
    _expect_error(
        lambda: bind(runner, MODEL_SHA),
        "owner changed",
    )

    missing = _runner(owner=None)
    missing.qwen35_hybrid_model_owner = None
    _expect_error(
        lambda: bind(missing, MODEL_SHA),
        "owner",
    )


class _WorkerAck:

    def __init__(self, rank, result):
        self.rank = rank
        self.result = result


class _Collector:

    def __init__(self):
        self.poison_reasons = []

    def poison(self, reason):
        self.poison_reasons.append(reason)


def _identity_row(
    rank,
    *,
    model_fingerprint=MODEL_SHA,
    layout_fingerprint="layout-a",
    dtype="float32",
):
    return {
        "participant_id": rank,
        "model_fingerprint": model_fingerprint,
        "layout_fingerprint": layout_fingerprint,
        "dtype": dtype,
    }


def _engine(rows=None):
    rows = (
        (_identity_row(0), _identity_row(1))
        if rows is None else rows
    )
    calls = []
    engine = types.SimpleNamespace(
        model_runner=types.SimpleNamespace(world_size=2),
        model_runner_ack_collector=_Collector(),
        qwen35_hybrid_prefix_runtime_identity=None,
        qwen35_hybrid_prefix_runtime_identity_configuration=None,
    )

    def acknowledged(method_name, *args, timeout_s):
        calls.append((method_name, args, timeout_s))
        return rows[0], (_WorkerAck(1, rows[1]),)

    engine.call_model_runner_acknowledged = acknowledged
    poison = _load_class_method(
        "tinyvllm/engine/llm_engine.py",
        "LLMEngine",
        "_poison_model_runner_ack_collector",
        {},
    )
    engine._poison_model_runner_ack_collector = (
        lambda reason: poison(engine, reason)
    )
    engine._calls = calls
    return engine


def _engine_method():
    return _load_class_method(
        "tinyvllm/engine/llm_engine.py",
        "LLMEngine",
        "configure_qwen35_hybrid_prefix_runtime_identity",
        {
            "torch": torch,
            "Qwen35HybridPrefixRuntimeIdentity": (
                Qwen35HybridPrefixRuntimeIdentity
            ),
            "validate_qwen35_model_fingerprint": (
                identity_module.validate_qwen35_model_fingerprint
            ),
        },
    )


def test_engine_aggregates_all_rank_identity_and_is_idempotent():
    configure = _engine_method()
    engine = _engine()

    identity = configure(
        engine,
        model_fingerprint=MODEL_SHA,
        timeout_s=0.5,
    )
    repeated = configure(
        engine,
        model_fingerprint=MODEL_SHA,
        timeout_s=0.5,
    )

    assert identity is repeated
    assert identity == Qwen35HybridPrefixRuntimeIdentity(
        model_fingerprint=MODEL_SHA,
        layout_fingerprint="layout-a",
        dtype=torch.float32,
    )
    assert engine._calls == [(
        "bind_qwen35_hybrid_prefix_runtime_identity",
        (MODEL_SHA,),
        0.5,
    )]


def test_engine_identity_mismatch_poisons_without_installing():
    configure = _engine_method()
    scenarios = (
        (_identity_row(0), _identity_row(0)),
        (_identity_row(0), _identity_row(1, model_fingerprint="b" * 64)),
        (_identity_row(0), _identity_row(1, layout_fingerprint="layout-b")),
        (_identity_row(0), _identity_row(1, dtype="bfloat16")),
        (_identity_row(0), {**_identity_row(1), "extra": 1}),
    )
    for rows in scenarios:
        engine = _engine(rows)
        _expect_error(
            lambda: configure(
                engine,
                model_fingerprint=MODEL_SHA,
                timeout_s=0.5,
            ),
            "identity",
        )
        assert engine.model_runner_ack_collector.poison_reasons
        assert engine.qwen35_hybrid_prefix_runtime_identity is None


def test_engine_identity_validates_inputs_and_rejects_reconfigure():
    configure = _engine_method()
    for model_fingerprint, timeout_s in (
        ("bad", 0.5),
        (MODEL_SHA, None),
        (MODEL_SHA, True),
        (MODEL_SHA, 0),
    ):
        engine = _engine()
        _expect_error(
            lambda model_fingerprint=model_fingerprint,
            timeout_s=timeout_s: configure(
                engine,
                model_fingerprint=model_fingerprint,
                timeout_s=timeout_s,
            ),
            "SHA256" if model_fingerprint == "bad" else "timeout",
        )
        assert engine._calls == []

    engine = _engine()
    configure(
        engine,
        model_fingerprint=MODEL_SHA,
        timeout_s=0.5,
    )
    _expect_error(
        lambda: configure(
            engine,
            model_fingerprint="b" * 64,
            timeout_s=0.5,
        ),
        "already configured",
    )


def test_engine_step_remains_identity_and_publication_free():
    source = (ROOT / "tinyvllm/engine/llm_engine.py").read_text()
    tree = ast.parse(source)
    class_node = next(
        node for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "LLMEngine"
    )
    step = next(
        node for node in class_node.body
        if isinstance(node, ast.FunctionDef) and node.name == "step"
    )
    step_source = ast.unparse(step)
    assert "configure_qwen35_hybrid_prefix_runtime_identity" not in step_source
    assert "qwen35_hybrid_prefix_runtime_identity" not in step_source


def _run():
    tests = [
        value
        for name, value in sorted(globals().items())
        if name.startswith("test_") and callable(value)
    ]
    for test in tests:
        test()
    print(
        "qwen35 runtime prefix identity binding tests passed "
        f"({len(tests)} tests)"
    )


if __name__ == "__main__":
    _run()
