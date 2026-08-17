from __future__ import annotations

import __future__
import ast
import gc
import importlib.util
import os
import sys
from types import SimpleNamespace

import torch


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_module(module_name, relative_path):
    spec = importlib.util.spec_from_file_location(
        module_name,
        os.path.join(ROOT, relative_path),
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def load_class_method(relative_path, class_name, method_name, globals_=None):
    path = os.path.join(ROOT, relative_path)
    tree = ast.parse(open(path).read(), filename=path)
    class_node = next(
        node for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == class_name
    )
    method_node = next(
        node for node in class_node.body
        if isinstance(node, ast.FunctionDef) and node.name == method_name
    )
    function_node = ast.FunctionDef(
        name=method_node.name,
        args=method_node.args,
        body=method_node.body,
        decorator_list=[],
        returns=method_node.returns,
        type_comment=method_node.type_comment,
    )
    namespace = dict(globals_ or {})
    exec(
        compile(
            ast.fix_missing_locations(ast.Module(
                body=[function_node],
                type_ignores=[],
            )),
            path,
            "exec",
            flags=__future__.annotations.compiler_flag,
        ),
        namespace,
    )
    return namespace[method_name]


def load_function(relative_path, function_name, globals_=None):
    path = os.path.join(ROOT, relative_path)
    tree = ast.parse(open(path).read(), filename=path)
    function_node = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == function_name
    )
    namespace = dict(globals_ or {})
    exec(
        compile(
            ast.fix_missing_locations(ast.Module(
                body=[function_node],
                type_ignores=[],
            )),
            path,
            "exec",
            flags=__future__.annotations.compiler_flag,
        ),
        namespace,
    )
    return namespace[function_name]


hybrid_state = load_module(
    "tinyvllm.engine.hybrid_state",
    "tinyvllm/engine/hybrid_state.py",
)
HybridStateComponentSpec = hybrid_state.HybridStateComponentSpec
HybridStateLayout = hybrid_state.HybridStateLayout
HybridStateLease = hybrid_state.HybridStateLease
HybridStateRuntimeBridge = hybrid_state.HybridStateRuntimeBridge
HybridStateTensorPool = hybrid_state.HybridStateTensorPool

step = load_class_method(
    "tinyvllm/engine/llm_engine.py",
    "LLMEngine",
    "step",
)
exit_engine = load_class_method(
    "tinyvllm/engine/llm_engine.py",
    "LLMEngine",
    "exit",
    {
        "gc": gc,
        "torch": torch,
    },
)
prepare_hybrid_state_batch = load_class_method(
    "tinyvllm/engine/model_runner.py",
    "ModelRunner",
    "_prepare_hybrid_state_batch",
    {"HybridStateLease": HybridStateLease},
)
release_hybrid_state = load_class_method(
    "tinyvllm/engine/model_runner.py",
    "ModelRunner",
    "release_hybrid_state",
)
try:
    flush_pending_hybrid_state_releases = load_class_method(
        "tinyvllm/engine/llm_engine.py",
        "LLMEngine",
        "flush_pending_hybrid_state_releases",
    )
except StopIteration:
    flush_pending_hybrid_state_releases = None
try_qwen35_hybrid_prefix_restore = load_function(
    "tinyvllm/engine/llm_engine.py",
    "_try_qwen35_hybrid_prefix_restore",
)


class IntegerClock:
    def __init__(self, values):
        self.values = iter(values)

    def __call__(self):
        return next(self.values)


def make_sequence():
    return SimpleNamespace(
        seq_id=17,
        hybrid_state_slot_id=0,
        hybrid_state_generation=2,
        completion_token_ids=[],
        prefill_chunk_start=0,
        prefill_chunk_end=2,
        prefill_chunk_final=True,
        step_is_decode=False,
        step_do_sample=True,
        is_finished=False,
    )


class FakeScheduler:
    last_policy_branch = "legacy_prefill"

    def __init__(self, sequence, released):
        self.sequence = sequence
        self.released = released
        self.restored = []
        self.snapshot_index = 0

    def observation_snapshot(self):
        self.snapshot_index += 1
        return {"snapshot": self.snapshot_index}

    def schedule(self, decision_now_ns):
        assert decision_now_ns == 10
        return [self.sequence], True, True

    def drain_hybrid_state_release_events(self):
        released = self.released
        self.released = ()
        return released

    def restore_hybrid_state_release_events(self, leases):
        self.restored.append(tuple(leases))

    def postprocess(
        self,
        seqs,
        token_ids,
        is_prefill,
        do_sample,
        batch_kind,
        *,
        decision_now_ns,
        step_end_ns,
    ):
        assert seqs == [self.sequence]
        assert token_ids == [91]
        self.sequence.completion_token_ids.append(91)
        self.sequence.is_finished = True

    def last_slo_observation(self):
        return {}


class RecordingRunner:
    def __init__(self, fail=False):
        self.calls = []
        self.fail = fail

    def call(self, method_name, *args):
        self.calls.append((method_name, args))
        if self.fail:
            raise RuntimeError("injected dispatch failure")
        if method_name == "exit":
            return {
                "rank": 0,
                "process_group_destroyed": True,
            }
        return [91]

    def memory_snapshot(self):
        return {}


def make_engine(sequence, released, runner):
    return SimpleNamespace(
        _clock_ns=IntegerClock([10, 20]),
        scheduler=FakeScheduler(sequence, released),
        model_runner=runner,
        last_batch_kind=None,
        last_scheduled_seqs=[],
        last_step_observation=None,
    )


def test_llm_engine_forwards_release_events_with_run():
    sequence = make_sequence()
    released = (HybridStateLease(0, 1, 16),)
    runner = RecordingRunner()
    engine = make_engine(sequence, released, runner)
    outputs, num_tokens = step(engine)
    assert outputs == [(17, [91])]
    assert num_tokens == 2
    assert runner.calls == [(
        "run",
        ([sequence], True, True, None, released),
    )]
    assert engine.scheduler.restored == []


def test_llm_engine_restores_events_when_dispatch_fails():
    sequence = make_sequence()
    released = (HybridStateLease(0, 1, 16),)
    runner = RecordingRunner(fail=True)
    engine = make_engine(sequence, released, runner)
    try:
        step(engine)
    except RuntimeError as error:
        assert str(error) == "injected dispatch failure"
    else:
        raise AssertionError("dispatch failure was swallowed")
    assert engine.scheduler.restored == [released]


def test_llm_engine_flushes_pending_releases_with_acknowledged_rpc():
    assert callable(flush_pending_hybrid_state_releases)
    released = (HybridStateLease(0, 1, 16),)
    scheduler = FakeScheduler(make_sequence(), released)
    calls = []
    engine = SimpleNamespace(scheduler=scheduler)

    def call_acknowledged(method_name, *args, timeout_s):
        calls.append((method_name, args, timeout_s))
        return None, (SimpleNamespace(result=None),)

    engine.call_model_runner_acknowledged = call_acknowledged

    result = flush_pending_hybrid_state_releases(
        engine,
        timeout_s=7.5,
    )

    assert result == released
    assert calls == [(
        "release_hybrid_state",
        (released,),
        7.5,
    )]
    assert scheduler.restored == []


def test_llm_engine_restores_pending_releases_when_acknowledged_rpc_fails():
    assert callable(flush_pending_hybrid_state_releases)
    released = (HybridStateLease(0, 1, 16),)
    scheduler = FakeScheduler(make_sequence(), released)
    engine = SimpleNamespace(scheduler=scheduler)

    def call_acknowledged(method_name, *args, timeout_s):
        raise RuntimeError("injected release acknowledgement failure")

    engine.call_model_runner_acknowledged = call_acknowledged

    try:
        flush_pending_hybrid_state_releases(engine, timeout_s=7.5)
    except RuntimeError as error:
        assert str(error) == (
            "injected release acknowledgement failure"
        )
    else:
        raise AssertionError("release acknowledgement failure was swallowed")

    assert scheduler.restored == [released]


def test_hybrid_prefix_restore_flushes_pending_releases_before_acquire():
    events = []

    class FakeBlockManager:
        block_size = 1

        @staticmethod
        def max_reusable_tokens(sequence):
            return 1

        @staticmethod
        def compute_hash(token_ids, previous_hash):
            assert token_ids == [31]
            assert previous_hash == -1
            return 44

    class FakeKey:
        def __init__(self, **kwargs):
            self.fields = kwargs

    sequence = SimpleNamespace(
        token_ids=[31],
        hybrid_prefix_restore_attempted=False,
        hybrid_prefix_restore_hit=False,
    )
    coordinator = SimpleNamespace(timeout_s=9.0)
    engine = SimpleNamespace(
        qwen35_hybrid_prefix_engine_restore_coordinator=coordinator,
        qwen35_hybrid_prefix_runtime_identity=SimpleNamespace(
            model_fingerprint="model",
            layout_fingerprint="layout",
            dtype="float16",
        ),
        scheduler=SimpleNamespace(block_manager=FakeBlockManager()),
        model_runner=SimpleNamespace(world_size=4),
    )

    def flush_pending_hybrid_state_releases(*, timeout_s):
        events.append(("flush", timeout_s))

    def acquire_qwen35_hybrid_prefix(sequence_arg, key, token_ids):
        events.append(("acquire", tuple(token_ids)))
        assert sequence_arg is sequence
        assert key.fields["token_hash"] == 44
        return True

    engine.flush_pending_hybrid_state_releases = (
        flush_pending_hybrid_state_releases
    )
    engine.acquire_qwen35_hybrid_prefix = acquire_qwen35_hybrid_prefix

    assert try_qwen35_hybrid_prefix_restore(
        engine,
        sequence,
        key_type=FakeKey,
    )
    assert events == [
        ("flush", 9.0),
        ("acquire", (31,)),
    ]


def test_llm_engine_exit_drains_release_events_before_exit():
    released = (HybridStateLease(0, 1, 16),)
    runner = RecordingRunner()

    class ExitScheduler:
        def drain_hybrid_state_release_events(self):
            return released

    class FakeProcess:
        def __init__(self):
            self.joined = False
            self.exitcode = 0

        def join(self):
            self.joined = True

        def is_alive(self):
            return False

    process = FakeProcess()
    engine = SimpleNamespace(
        scheduler=ExitScheduler(),
        model_runner=runner,
        ps=[process],
        model_runner_ack_collector=None,
    )
    engine.call_model_runner_acknowledged = lambda method_name, timeout_s: (
        runner.call(method_name),
        (
            SimpleNamespace(
                result={
                    "rank": 1,
                    "process_group_destroyed": True,
                },
            ),
        ),
    )
    lifecycle_calls = []
    original_collect = gc.collect
    original_empty_cache = torch.cuda.empty_cache
    gc.collect = lambda: lifecycle_calls.append("gc")
    torch.cuda.empty_cache = lambda: lifecycle_calls.append("empty_cache")
    try:
        receipt = exit_engine(engine)
        repeated_receipt = exit_engine(engine)
    finally:
        gc.collect = original_collect
        torch.cuda.empty_cache = original_empty_cache
    assert runner.calls == [
        ("release_hybrid_state", (released,)),
        ("exit", ()),
    ]
    assert process.joined
    assert not hasattr(engine, "model_runner")
    assert lifecycle_calls == ["gc", "empty_cache"]
    assert receipt == {
        "process_group_destroyed": True,
        "rank_exit_codes": [0, 0],
        "owned_children_remaining": [],
        "rank_cleanup_receipts": [
            {
                "rank": 0,
                "process_group_destroyed": True,
            },
            {
                "rank": 1,
                "process_group_destroyed": True,
            },
        ],
    }
    assert repeated_receipt == receipt


def make_cpu_bridge():
    layout = HybridStateLayout((
        HybridStateComponentSpec(
            0,
            "linear_recurrent",
            (1,),
            torch.float32,
        ),
    ))
    pool = HybridStateTensorPool(layout, capacity=1, device="cpu")
    return pool, HybridStateRuntimeBridge(pool)


def test_model_runner_helper_is_optional_and_fails_closed():
    runner = SimpleNamespace(
        hybrid_state_runtime_bridge=None,
        _last_hybrid_state_slot_ids=None,
    )
    assert prepare_hybrid_state_batch(runner, [], ()) is None
    active = make_sequence()
    try:
        prepare_hybrid_state_batch(runner, [active], ())
    except RuntimeError as error:
        assert "runtime bridge is not installed" in str(error)
    else:
        raise AssertionError("active hybrid lease without bridge was accepted")
    try:
        prepare_hybrid_state_batch(
            runner,
            [],
            (HybridStateLease(0, 1, 16),),
        )
    except RuntimeError as error:
        assert "runtime bridge is not installed" in str(error)
    else:
        raise AssertionError("release event without bridge was accepted")


def test_model_runner_helper_releases_before_activation():
    pool, bridge = make_cpu_bridge()
    first = HybridStateLease(0, 1, 16)
    pool.activate(first)
    tensor = pool.component_tensor(0, "linear_recurrent")
    tensor[0].fill_(8)
    runner = SimpleNamespace(
        hybrid_state_runtime_bridge=bridge,
        _last_hybrid_state_slot_ids=None,
    )
    sequence = make_sequence()
    result = prepare_hybrid_state_batch(runner, [sequence], (first,))
    assert result.tolist() == [0]
    assert runner._last_hybrid_state_slot_ids is result
    assert torch.count_nonzero(tensor[0]).item() == 0
    assert pool.validate(HybridStateLease(0, 2, 17)) == 0


def test_model_runner_helper_rejects_duplicate_active_slots_with_rank_context():
    layout = HybridStateLayout((
        HybridStateComponentSpec(
            0,
            "linear_recurrent",
            (1,),
            torch.float32,
        ),
    ))
    pool = HybridStateTensorPool(layout, capacity=2, device="cpu")
    bridge = HybridStateRuntimeBridge(pool)
    runner = SimpleNamespace(
        rank=3,
        hybrid_state_runtime_bridge=bridge,
        _last_hybrid_state_slot_ids=None,
    )
    first = make_sequence()
    first.seq_id = 21
    first.hybrid_state_slot_id = 0
    first.hybrid_state_generation = 1
    second = make_sequence()
    second.seq_id = 22
    second.hybrid_state_slot_id = 0
    second.hybrid_state_generation = 1

    try:
        prepare_hybrid_state_batch(runner, [first, second], ())
    except ValueError as error:
        detail = str(error)
        assert "rank=3" in detail
        assert "(21, 0, 1)" in detail
        assert "(22, 0, 1)" in detail
    else:
        raise AssertionError("duplicate active hybrid slots were accepted")
    assert pool._bindings == {}


def test_model_runner_release_rpc_forwards_exact_events():
    pool, bridge = make_cpu_bridge()
    lease = HybridStateLease(0, 1, 21)
    pool.activate(lease)
    runner = SimpleNamespace(hybrid_state_runtime_bridge=bridge)
    release_hybrid_state(runner, (lease,))
    try:
        pool.validate(lease)
    except RuntimeError:
        pass
    else:
        raise AssertionError("release RPC did not unbind the lease")


if __name__ == "__main__":
    test_llm_engine_forwards_release_events_with_run()
    test_llm_engine_restores_events_when_dispatch_fails()
    test_llm_engine_flushes_pending_releases_with_acknowledged_rpc()
    test_llm_engine_restores_pending_releases_when_acknowledged_rpc_fails()
    test_hybrid_prefix_restore_flushes_pending_releases_before_acquire()
    test_llm_engine_exit_drains_release_events_before_exit()
    test_model_runner_helper_is_optional_and_fails_closed()
    test_model_runner_helper_releases_before_activation()
    test_model_runner_helper_rejects_duplicate_active_slots_with_rank_context()
    test_model_runner_release_rpc_forwards_exact_events()
    print("hybrid state runtime bridge tests passed")
