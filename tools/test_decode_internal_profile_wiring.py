from __future__ import annotations

import ast
import copy
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
MODEL_RUNNER = ROOT / "tinyvllm/engine/model_runner.py"
LLM_ENGINE = ROOT / "tinyvllm/engine/llm_engine.py"
LINEAR = ROOT / "tinyvllm/layers/linear.py"
EMBED_HEAD = ROOT / "tinyvllm/layers/embed_head.py"
QWEN35_COMPONENTS = ROOT / "tinyvllm/models/qwen35_components.py"


def _class_method(path, class_name, method_name):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    class_node = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef)
        and node.name == class_name
    )
    return next(
        node
        for node in class_node.body
        if isinstance(node, ast.FunctionDef)
        and node.name == method_name
    )


def _called_names(node):
    names = []
    for child in ast.walk(node):
        if not isinstance(child, ast.Call):
            continue
        if isinstance(child.func, ast.Name):
            names.append(child.func.id)
        elif isinstance(child.func, ast.Attribute):
            names.append(child.func.attr)
    return names


def _string_constants(node):
    return {
        child.value
        for child in ast.walk(node)
        if isinstance(child, ast.Constant)
        and isinstance(child.value, str)
    }


def _compile_method(path, class_name, method_name):
    node = copy.deepcopy(
        _class_method(path, class_name, method_name)
    )
    node.decorator_list = []
    module = ast.Module(body=[node], type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {}
    exec(compile(module, str(path), "exec"), namespace)
    return namespace[method_name]


def test_model_runner_exposes_profile_lifecycle_and_wraps_run():
    configure = _class_method(
        MODEL_RUNNER,
        "ModelRunner",
        "configure_decode_internal_profile",
    )
    finalize = _class_method(
        MODEL_RUNNER,
        "ModelRunner",
        "finalize_decode_internal_profile",
    )
    reset = _class_method(
        MODEL_RUNNER,
        "ModelRunner",
        "reset_decode_internal_profile",
    )
    run = _class_method(MODEL_RUNNER, "ModelRunner", "run")

    assert "DecodeInternalProfiler" in _called_names(configure)
    assert "configure_decode_internal_profile" in _called_names(reset)
    assert "finalize" in _called_names(finalize)
    assert "already_synchronized" in {
        argument.arg
        for argument in finalize.args.args
        + finalize.args.kwonlyargs
    }
    assert "run_profiled_step" in _called_names(run)
    assert "_run_model_step" in _called_names(run)


def test_llm_engine_exposes_acknowledged_profile_lifecycle():
    configure = _class_method(
        LLM_ENGINE,
        "LLMEngine",
        "configure_decode_internal_profile",
    )
    finalize = _class_method(
        LLM_ENGINE,
        "LLMEngine",
        "finalize_decode_internal_profile",
    )
    reset = _class_method(
        LLM_ENGINE,
        "LLMEngine",
        "reset_decode_internal_profile",
    )

    assert "call_model_runner_acknowledged" in _called_names(configure)
    assert "call_model_runner_acknowledged" in _called_names(reset)
    assert "call_model_runner_acknowledged" in _called_names(finalize)
    assert "configure_decode_internal_profile" in _string_constants(
        configure
    )
    assert "reset_decode_internal_profile" in _string_constants(reset)
    assert "finalize_decode_internal_profile" in _string_constants(
        finalize
    )


def test_rank_aware_profile_finalization_only_reuses_rank_zero_sync():
    finalize = _compile_method(
        MODEL_RUNNER,
        "ModelRunner",
        "finalize_decode_internal_profile",
    )
    profiler_calls = []
    profiler_syncs = []

    class FakeProfiler:
        def __init__(self, rank):
            self.rank = rank

        def finalize(self, *, already_synchronized):
            profiler_calls.append((self.rank, already_synchronized))
            if not already_synchronized:
                profiler_syncs.append(self.rank)
            return {"rank": self.rank}

    class FakeRunner:
        def __init__(self, rank):
            self.rank = rank
            self.world_size = 4
            self.decode_internal_profiler = FakeProfiler(rank)

    results = [
        finalize(
            FakeRunner(rank),
            already_synchronized_rank=0,
        )
        for rank in range(4)
    ]

    assert results == [{"rank": rank} for rank in range(4)]
    assert profiler_calls == [
        (0, True),
        (1, False),
        (2, False),
        (3, False),
    ]
    assert profiler_syncs == [1, 2, 3]


def test_engine_rank_aware_profile_finalization_is_acknowledged():
    calls = []

    def finalized_rank(rank):
        return {
            "rank": rank,
            "enabled": True,
            "finalization_status": "complete",
            "steps": [{"rank": rank}],
            "collectives": [],
        }

    class FakeEngine:
        model_runner = type("ModelRunner", (), {"world_size": 4})()

        def call_model_runner_acknowledged(self, *args, **kwargs):
            calls.append((args, kwargs))
            return (
                finalized_rank(0),
                tuple(
                    type(
                        "Ack",
                        (),
                        {"rank": rank, "result": finalized_rank(rank)},
                    )()
                    for rank in range(1, 4)
                ),
            )

    finalize = _compile_method(
        LLM_ENGINE,
        "LLMEngine",
        "finalize_decode_internal_profile",
    )
    result = finalize(
        FakeEngine(),
        already_synchronized_rank=0,
        timeout_s=60.0,
    )

    assert calls == [
        (
            ("finalize_decode_internal_profile", False, 0),
            {"timeout_s": 60.0},
        )
    ]
    assert result["rank_inventory"] == [0, 1, 2, 3]


@pytest.mark.parametrize(
    ("method_name", "args", "kwargs"),
    (
        (
            "configure_decode_internal_profile",
            (True, "diagnostic"),
            {"timeout_s": 60.0},
        ),
        (
            "reset_decode_internal_profile",
            (),
            {"timeout_s": 60.0},
        ),
        (
            "finalize_decode_internal_profile",
            (),
            {
                "already_synchronized_rank": 0,
                "timeout_s": 60.0,
            },
        ),
    ),
)
def test_command_timeline_profile_helpers_propagate_all_rank_failure(
    method_name,
    args,
    kwargs,
):
    sentinel = RuntimeError("rank 3 failed")

    class FakeEngine:
        def call_model_runner_acknowledged(self, *call_args, **call_kwargs):
            raise sentinel

    method = _compile_method(
        LLM_ENGINE,
        "LLMEngine",
        method_name,
    )
    with pytest.raises(RuntimeError) as exc_info:
        method(FakeEngine(), *args, **kwargs)

    assert exc_info.value is sentinel


def test_collective_call_sites_use_profile_helper():
    linear_source = LINEAR.read_text(encoding="utf-8")
    embed_source = EMBED_HEAD.read_text(encoding="utf-8")

    assert linear_source.count(
        'profile_collective(\n                        "row_parallel_all_reduce"'
    ) == 1
    assert linear_source.count(
        'profile_collective(\n                "row_parallel_all_reduce"'
    ) == 1
    assert (
        'profile_collective(\n'
        '                "vocab_parallel_embedding_all_reduce"'
        in embed_source
    )
    assert (
        'profile_collective(\n'
        '                "replicated_weight_row_parallel_all_gather"'
        in linear_source
    )


def test_qwen35_output_projections_use_true_row_parallel_layout():
    components_source = QWEN35_COMPONENTS.read_text(encoding="utf-8")

    assert "ReplicatedWeightRowParallelLinear(" not in components_source
    assert components_source.count("RowParallelLinear(") >= 2
