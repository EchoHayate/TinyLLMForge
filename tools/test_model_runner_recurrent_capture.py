import ast
from pathlib import Path
import sys
import types
from types import SimpleNamespace

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
for package_name in ("tinyvllm", "tinyvllm.engine"):
    package = types.ModuleType(package_name)
    package.__path__ = [str(ROOT / package_name.replace(".", "/"))]
    sys.modules.setdefault(package_name, package)

from tinyvllm.engine.qwen35_recurrent_capture import (
    Qwen35RecurrentCaptureSession,
)
from tinyvllm.engine.qwen35_recurrent_capture_contract import (
    CAPTURE_IDENTITY_SCHEMA_VERSION,
    validate_run_identity,
)

MODEL_RUNNER_PATH = ROOT / "tinyvllm/engine/model_runner.py"
METHOD_NAMES = (
    "configure_qwen35_recurrent_capture",
    "arm_qwen35_recurrent_capture",
    "finish_qwen35_recurrent_capture_workload",
    "_capture_qwen35_recurrent_source_state",
)
ENGINE_METHOD_NAMES = (
    "_collect_qwen35_recurrent_capture_rows",
    "configure_qwen35_recurrent_capture",
    "arm_qwen35_recurrent_capture",
    "finish_qwen35_recurrent_capture_workload",
)


def _load_model_runner_shell():
    tree = ast.parse(
        MODEL_RUNNER_PATH.read_text(encoding="utf-8"),
        filename=str(MODEL_RUNNER_PATH),
    )
    model_runner = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "ModelRunner"
    )
    methods = {
        node.name: node
        for node in model_runner.body
        if isinstance(node, ast.FunctionDef) and node.name in METHOD_NAMES
    }
    assert set(methods) == set(METHOD_NAMES)
    shell = ast.ClassDef(
        name="ModelRunnerShell",
        bases=[],
        keywords=[],
        body=[methods[name] for name in METHOD_NAMES],
        decorator_list=[],
    )
    module = ast.fix_missing_locations(
        ast.Module(body=[shell], type_ignores=[])
    )
    namespace = {
        "Qwen35RecurrentCaptureSession": Qwen35RecurrentCaptureSession,
        "validate_run_identity": validate_run_identity,
        "CAPTURE_IDENTITY_SCHEMA_VERSION": (
            CAPTURE_IDENTITY_SCHEMA_VERSION
        ),
    }
    exec(compile(module, str(MODEL_RUNNER_PATH), "exec"), namespace)
    return namespace["ModelRunnerShell"]


ModelRunnerShell = _load_model_runner_shell()


def _load_llm_engine_shell():
    engine_path = ROOT / "tinyvllm/engine/llm_engine.py"
    tree = ast.parse(
        engine_path.read_text(encoding="utf-8"),
        filename=str(engine_path),
    )
    llm_engine = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "LLMEngine"
    )
    methods = {
        node.name: node
        for node in llm_engine.body
        if isinstance(node, ast.FunctionDef)
        and node.name in ENGINE_METHOD_NAMES
    }
    assert set(methods) == set(ENGINE_METHOD_NAMES)
    shell = ast.ClassDef(
        name="LLMEngineShell",
        bases=[],
        keywords=[],
        body=[methods[name] for name in ENGINE_METHOD_NAMES],
        decorator_list=[],
    )
    module = ast.fix_missing_locations(
        ast.Module(body=[shell], type_ignores=[])
    )
    namespace = {}
    exec(compile(module, str(engine_path), "exec"), namespace)
    return namespace["LLMEngineShell"]


LLMEngineShell = _load_llm_engine_shell()


class FakeSession:
    def __init__(self, *, workload_ids=("w0",), fail_layer=None):
        self.run_identity = SimpleNamespace(workload_ids=workload_ids)
        self.fail_layer = fail_layer
        self.calls = []
        self.finished = []

    def capture_layer(self, *, workload_id, layer_index, tensor):
        self.calls.append((workload_id, layer_index, tensor))
        if layer_index == self.fail_layer:
            raise OSError("synthetic capture failure")

    def finish_workload(self, workload_id):
        self.finished.append(workload_id)


class CountingOwner:
    def __init__(self, adapters, linear_indices=None):
        self._transaction = SimpleNamespace(adapters=tuple(adapters))
        self.access_count = 0
        self.layer_stack = SimpleNamespace(
            linear_indices=(
                tuple(adapter.layer_index for adapter in adapters)
                if linear_indices is None
                else tuple(linear_indices)
            )
        )

    @property
    def state_transaction(self):
        self.access_count += 1
        return self._transaction


def _runner(*, session, armed=True, adapters=()):
    runner = object.__new__(ModelRunnerShell)
    runner.rank = 0
    runner.qwen35_hybrid_model_owner = CountingOwner(adapters)
    runner.qwen35_recurrent_capture_session = session
    runner.qwen35_recurrent_capture_workload_id = "w0"
    runner.qwen35_recurrent_capture_armed = armed
    runner._last_hybrid_state_leases = (
        SimpleNamespace(slot_id=3, generation=1, request_id=17),
    )
    return runner


def _seq(**overrides):
    values = {
        "hybrid_state_slot_id": 3,
        "hybrid_state_generation": 1,
        "prefill_chunk_final": True,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _adapters(layer_indices=(1, 3)):
    rows = []
    for layer_index in layer_indices:
        recurrent = torch.arange(
            4 * 2 * 3,
            dtype=torch.float32,
        ).reshape(4, 2, 3)
        recurrent.add_(layer_index * 100)
        rows.append(
            SimpleNamespace(
                layer_index=layer_index,
                recurrent=recurrent,
            )
        )
    return tuple(rows)


def test_disabled_hook_does_not_touch_owner():
    runner = _runner(session=None, adapters=_adapters())

    runner._capture_qwen35_recurrent_source_state(
        [_seq()],
        is_prefill=True,
        batch_kind="prefill",
    )

    assert runner.qwen35_hybrid_model_owner.access_count == 0


def test_unarmed_hook_does_not_touch_owner():
    runner = _runner(
        session=FakeSession(),
        armed=False,
        adapters=_adapters(),
    )

    runner._capture_qwen35_recurrent_source_state(
        [_seq()],
        is_prefill=True,
        batch_kind="prefill",
    )

    assert runner.qwen35_hybrid_model_owner.access_count == 0


@pytest.mark.parametrize(
    ("seqs", "is_prefill", "batch_kind", "leases", "message"),
    (
        ([_seq()], False, None, None, "prefill"),
        ([_seq()], True, "mixed", None, "mixed"),
        ([_seq()], True, "prefill", (1, 2), "lease"),
        (
            [_seq(prefill_chunk_final=False)],
            True,
            "prefill",
            None,
            "final",
        ),
        (
            [_seq(hybrid_state_slot_id=2)],
            True,
            "prefill",
            None,
            "lease",
        ),
        (
            [_seq(hybrid_state_generation=2)],
            True,
            "prefill",
            None,
            "lease",
        ),
    ),
)
def test_armed_hook_rejects_invalid_capture_boundary(
    seqs,
    is_prefill,
    batch_kind,
    leases,
    message,
):
    runner = _runner(session=FakeSession(), adapters=_adapters())
    if leases is not None:
        runner._last_hybrid_state_leases = tuple(
            SimpleNamespace(slot_id=value, generation=1, request_id=value)
            for value in leases
        )

    with pytest.raises(ValueError, match=message):
        runner._capture_qwen35_recurrent_source_state(
            seqs,
            is_prefill=is_prefill,
            batch_kind=batch_kind,
        )
    assert runner.qwen35_recurrent_capture_armed is True


def test_hook_captures_exact_adapter_views_and_disarms():
    adapters = _adapters()
    session = FakeSession()
    runner = _runner(session=session, adapters=adapters)
    originals = tuple(adapter.recurrent.clone() for adapter in adapters)
    identities = tuple(id(adapter.recurrent) for adapter in adapters)

    runner._capture_qwen35_recurrent_source_state(
        [_seq()],
        is_prefill=True,
        batch_kind="prefill",
    )

    assert tuple(
        (workload_id, layer_index)
        for workload_id, layer_index, _ in session.calls
    ) == (("w0", 1), ("w0", 3))
    assert tuple(
        tensor.data_ptr()
        for _, _, tensor in session.calls
    ) == tuple(adapter.recurrent[3].data_ptr() for adapter in adapters)
    assert tuple(id(adapter.recurrent) for adapter in adapters) == identities
    for adapter, original in zip(adapters, originals, strict=True):
        torch.testing.assert_close(adapter.recurrent, original)
    assert runner.qwen35_recurrent_capture_armed is False


def test_hook_stays_armed_when_any_layer_capture_fails():
    session = FakeSession(fail_layer=3)
    runner = _runner(session=session, adapters=_adapters())

    with pytest.raises(OSError, match="synthetic"):
        runner._capture_qwen35_recurrent_source_state(
            [_seq()],
            is_prefill=True,
            batch_kind="prefill",
        )

    assert runner.qwen35_recurrent_capture_armed is True


def test_hook_observes_values_before_final_prefill_rounding():
    adapter = _adapters((1,))[0]
    adapter.recurrent[3].copy_(
        torch.tensor(
            [[0.0011, 0.0022, 0.0033], [0.0044, 0.0055, 0.0066]]
        )
    )
    observed = []

    class ObservingSession(FakeSession):
        def capture_layer(self, *, workload_id, layer_index, tensor):
            observed.append(tensor.clone())

    runner = _runner(session=ObservingSession(), adapters=(adapter,))

    runner._capture_qwen35_recurrent_source_state(
        [_seq()],
        is_prefill=True,
        batch_kind="prefill",
    )
    adapter.recurrent[3].copy_(
        adapter.recurrent[3].to(torch.bfloat16).to(torch.float32)
    )

    assert not torch.equal(observed[0], adapter.recurrent[3])


def _configuration(**overrides):
    value = {
        "capture_root": "/tmp/qwen35-capture",
        "model_manifest_sha256": "a" * 64,
        "source_tree_sha256": "b" * 64,
        "workload_manifest_sha256": "c" * 64,
        "world_size": 2,
        "workload_ids": ["w0", "w1"],
    }
    value.update(overrides)
    return value


def _configuration_runner(linear_indices=tuple(range(18))):
    runner = object.__new__(ModelRunnerShell)
    runner.rank = 1
    runner.qwen35_hybrid_model_owner = CountingOwner(
        (),
        linear_indices=linear_indices,
    )
    runner.qwen35_recurrent_capture_session = None
    runner.qwen35_recurrent_capture_workload_id = None
    runner.qwen35_recurrent_capture_armed = False
    return runner


def test_configure_derives_model_layers_and_builds_rank_session(
    tmp_path,
    monkeypatch,
):
    created = []

    class Session:
        def __init__(self, **kwargs):
            created.append(kwargs)
            self.run_identity = kwargs["run_identity"]

    monkeypatch.setitem(
        ModelRunnerShell.configure_qwen35_recurrent_capture.__globals__,
        "Qwen35RecurrentCaptureSession",
        Session,
    )
    runner = _configuration_runner()
    configuration = _configuration(capture_root=str(tmp_path))

    result = runner.configure_qwen35_recurrent_capture(configuration)

    assert result == {
        "rank": 1,
        "configured": True,
        "workload_ids": ("w0", "w1"),
        "linear_layer_indices": tuple(range(18)),
    }
    identity = created[0]["run_identity"]
    assert identity.linear_layer_indices == tuple(range(18))
    assert created[0]["rank"] == 1
    assert created[0]["staging_dir"] == str(tmp_path)


@pytest.mark.parametrize(
    "configuration",
    (
        _configuration(extra=True),
        _configuration(workload_ids=("w0", "w1")),
        _configuration(world_size=True),
    ),
)
def test_configure_rejects_malformed_configuration(configuration):
    runner = _configuration_runner()

    with pytest.raises(ValueError):
        runner.configure_qwen35_recurrent_capture(configuration)


@pytest.mark.parametrize(
    "linear_indices",
    (
        tuple(range(17)),
        tuple(range(17)) + (16,),
        tuple(reversed(range(18))),
    ),
)
def test_configure_requires_exactly_18_sorted_unique_model_layers(
    linear_indices,
):
    runner = _configuration_runner(linear_indices)

    with pytest.raises(ValueError, match="18|layer"):
        runner.configure_qwen35_recurrent_capture(_configuration())


def test_configure_rejects_reconfiguration():
    runner = _configuration_runner()
    runner.qwen35_recurrent_capture_session = object()

    with pytest.raises(RuntimeError, match="configured"):
        runner.configure_qwen35_recurrent_capture(_configuration())


def test_arm_and_finish_lifecycle():
    session = FakeSession(workload_ids=("w0", "w1"))
    runner = _runner(session=session, armed=False)
    runner.qwen35_recurrent_capture_workload_id = None

    armed = runner.arm_qwen35_recurrent_capture("w1")
    assert armed == {"rank": 0, "workload_id": "w1", "armed": True}
    assert runner.qwen35_recurrent_capture_armed is True
    runner._capture_qwen35_recurrent_source_state(
        [_seq()],
        is_prefill=True,
        batch_kind="prefill",
    )

    complete = runner.finish_qwen35_recurrent_capture_workload("w1")
    assert complete == {"rank": 0, "workload_id": "w1", "complete": True}
    assert session.finished == ["w1"]
    assert runner.qwen35_recurrent_capture_armed is False
    assert runner.qwen35_recurrent_capture_workload_id is None


def test_run_calls_capture_between_forward_and_rounding():
    source = MODEL_RUNNER_PATH.read_text(encoding="utf-8")
    run_start = source.index("    def run(self, seqs:list[Sequence]")
    run_end = source.index("\n    @torch.inference_mode()", run_start)
    run_source = source[run_start:run_end]

    forward = run_source.index(
        "logits = self.run_model(input_ids, positions, is_prefill)"
    )
    capture = run_source.index(
        "self._capture_qwen35_recurrent_source_state("
    )
    rounding = run_source.index(
        "_round_qwen35_final_prefill_recurrent_states("
    )
    offload = run_source.index("self._kv_offload_after_forward()")

    assert forward < capture < rounding < offload


def _engine(*, world_size=2, response=None, error=None):
    calls = []

    def call_ack(method_name, *args, timeout_s):
        calls.append((method_name, args, timeout_s))
        if error is not None:
            raise error
        return response

    engine = object.__new__(LLMEngineShell)
    engine.model_runner = SimpleNamespace(world_size=world_size)
    engine.call_model_runner_acknowledged = call_ack
    return engine, calls


def _acks(*rows):
    return tuple(
        SimpleNamespace(rank=row["rank"], result=row)
        for row in rows
    )


@pytest.mark.parametrize(
    ("local_rank", "outer_rank", "inner_rank"),
    (
        (False, 1, 1),
        (0, True, 1),
        (0, 1, True),
        (0, 1.0, 1),
        (0, 1, 1.0),
    ),
)
def test_engine_capture_transport_rejects_non_exact_integer_ranks(
    local_rank,
    outer_rank,
    inner_rank,
):
    local = {
        "rank": local_rank,
        "workload_id": "w0",
        "armed": True,
    }
    worker = {
        "rank": inner_rank,
        "workload_id": "w0",
        "armed": True,
    }
    worker_acks = (
        SimpleNamespace(rank=outer_rank, result=worker),
    )
    engine, _ = _engine(response=(local, worker_acks))

    with pytest.raises(ValueError, match="rank"):
        engine.arm_qwen35_recurrent_capture(
            "w0",
            timeout_s=120.0,
        )


def test_engine_capture_transport_rejects_outer_inner_rank_rebinding():
    local = {
        "rank": 0,
        "workload_id": "w0",
        "armed": True,
    }
    worker = {
        "rank": 2,
        "workload_id": "w0",
        "armed": True,
    }
    worker_acks = (
        SimpleNamespace(rank=1, result=worker),
    )
    engine, _ = _engine(response=(local, worker_acks))

    with pytest.raises(ValueError, match="rank"):
        engine.arm_qwen35_recurrent_capture(
            "w0",
            timeout_s=120.0,
        )


def test_engine_configure_capture_returns_rank_ordered_rows():
    local = {
        "rank": 0,
        "configured": True,
        "workload_ids": ("w0", "w1"),
        "linear_layer_indices": tuple(range(18)),
    }
    worker = {
        **local,
        "rank": 1,
    }
    engine, calls = _engine(response=(local, _acks(worker)))

    rows = engine.configure_qwen35_recurrent_capture(
        capture_root="/tmp/capture",
        model_manifest_sha256="a" * 64,
        source_tree_sha256="b" * 64,
        workload_manifest_sha256="c" * 64,
        world_size=2,
        workload_ids=["w0", "w1"],
        timeout_s=120.0,
    )

    assert rows == (local, worker)
    assert calls == [(
        "configure_qwen35_recurrent_capture",
        ({
            "capture_root": "/tmp/capture",
            "model_manifest_sha256": "a" * 64,
            "source_tree_sha256": "b" * 64,
            "workload_manifest_sha256": "c" * 64,
            "world_size": 2,
            "workload_ids": ["w0", "w1"],
        },),
        120.0,
    )]


@pytest.mark.parametrize(
    ("method_name", "local", "workers", "match"),
    (
        (
            "arm_qwen35_recurrent_capture",
            {"rank": 0, "workload_id": "w0", "armed": True},
            (),
            "inventory",
        ),
        (
            "arm_qwen35_recurrent_capture",
            {"rank": 0, "workload_id": "w0", "armed": True},
            (
                {"rank": 1, "workload_id": "w0", "armed": True},
                {"rank": 1, "workload_id": "w0", "armed": True},
            ),
            "rank",
        ),
        (
            "arm_qwen35_recurrent_capture",
            {"rank": 0, "workload_id": "w0", "armed": True},
            (
                {"rank": 1, "workload_id": "w1", "armed": True},
            ),
            "mismatch",
        ),
        (
            "arm_qwen35_recurrent_capture",
            {"rank": 0, "workload_id": "w0", "armed": True},
            (
                {"rank": 1, "workload_id": "w0", "armed": False},
            ),
            "mismatch",
        ),
        (
            "finish_qwen35_recurrent_capture_workload",
            {"rank": 0, "workload_id": "w0", "complete": True},
            (
                {
                    "rank": 1,
                    "workload_id": "w0",
                    "complete": True,
                    "extra": True,
                },
            ),
            "fields",
        ),
    ),
)
def test_engine_capture_transport_rejects_invalid_rank_results(
    method_name,
    local,
    workers,
    match,
):
    engine, _ = _engine(response=(local, _acks(*workers)))
    method = getattr(engine, method_name)

    with pytest.raises(ValueError, match=match):
        method("w0", timeout_s=120.0)


@pytest.mark.parametrize(
    ("method_name", "rows", "match"),
    (
        (
            "arm_qwen35_recurrent_capture",
            (
                {
                    "rank": 0,
                    "workload_id": "wrong",
                    "armed": True,
                },
                {
                    "rank": 1,
                    "workload_id": "wrong",
                    "armed": True,
                },
            ),
            "workload",
        ),
        (
            "arm_qwen35_recurrent_capture",
            (
                {
                    "rank": 0,
                    "workload_id": "w0",
                    "armed": False,
                },
                {
                    "rank": 1,
                    "workload_id": "w0",
                    "armed": False,
                },
            ),
            "status",
        ),
        (
            "finish_qwen35_recurrent_capture_workload",
            (
                {
                    "rank": 0,
                    "workload_id": "w0",
                    "complete": False,
                },
                {
                    "rank": 1,
                    "workload_id": "w0",
                    "complete": False,
                },
            ),
            "status",
        ),
    ),
)
def test_engine_capture_transport_rejects_consistent_wrong_semantics(
    method_name,
    rows,
    match,
):
    engine, _ = _engine(response=(rows[0], _acks(rows[1])))

    with pytest.raises(ValueError, match=match):
        getattr(engine, method_name)("w0", timeout_s=120.0)


def test_engine_configure_rejects_runtime_world_size_mismatch():
    engine, calls = _engine(world_size=2, response=None)

    with pytest.raises(ValueError, match="world_size"):
        engine.configure_qwen35_recurrent_capture(
            capture_root="/tmp/capture",
            model_manifest_sha256="a" * 64,
            source_tree_sha256="b" * 64,
            workload_manifest_sha256="c" * 64,
            world_size=3,
            workload_ids=["w0"],
            timeout_s=120.0,
        )

    assert calls == []


def test_engine_arm_and_finish_capture_return_rank_ordered_rows():
    armed = (
        {"rank": 0, "workload_id": "w0", "armed": True},
        {"rank": 1, "workload_id": "w0", "armed": True},
    )
    engine, calls = _engine(response=(armed[0], _acks(armed[1])))

    assert engine.arm_qwen35_recurrent_capture(
        "w0",
        timeout_s=120.0,
    ) == armed
    assert calls[-1] == (
        "arm_qwen35_recurrent_capture",
        ("w0",),
        120.0,
    )

    complete = (
        {"rank": 0, "workload_id": "w0", "complete": True},
        {"rank": 1, "workload_id": "w0", "complete": True},
    )
    engine.call_model_runner_acknowledged = (
        lambda method_name, *args, timeout_s: (
            complete[0],
            _acks(complete[1]),
        )
    )
    assert engine.finish_qwen35_recurrent_capture_workload(
        "w0",
        timeout_s=120.0,
    ) == complete


@pytest.mark.parametrize(
    "error",
    (
        TimeoutError("worker acknowledgement timed out"),
        RuntimeError("acknowledgement collector is poisoned"),
    ),
)
def test_engine_capture_transport_propagates_ack_failures(error):
    engine, _ = _engine(error=error)

    with pytest.raises(type(error), match=str(error)):
        engine.arm_qwen35_recurrent_capture(
            "w0",
            timeout_s=120.0,
        )
