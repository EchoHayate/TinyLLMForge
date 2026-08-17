from __future__ import annotations

import ast
from pathlib import Path
import types


ROOT = Path(__file__).resolve().parents[1]
ENGINE_PATH = ROOT / "tinyvllm/engine/llm_engine.py"
MODEL_FINGERPRINT = "a" * 64


def _validate_model_fingerprint(value):
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(
            character not in "0123456789abcdef"
            for character in value
        )
    ):
        raise ValueError("model_fingerprint must be a lowercase SHA256")
    return value


class _Ack:

    def __init__(self, rank, result):
        self.rank = rank
        self.result = result


def _row(
    rank,
    *,
    status="bound",
    model_fingerprint=MODEL_FINGERPRINT,
    layout_fingerprint="layout-a",
    dtype="float32",
    detail="",
):
    if status == "error":
        model_fingerprint = ""
        layout_fingerprint = ""
        dtype = ""
    return {
        "participant_id": rank,
        "operation": "bind_loaded_checkpoint_candidate",
        "status": status,
        "model_fingerprint": model_fingerprint,
        "layout_fingerprint": layout_fingerprint,
        "dtype": dtype,
        "detail": detail,
    }


def _load_engine_method(name):
    source = ENGINE_PATH.read_text()
    tree = ast.parse(source, filename=str(ENGINE_PATH))
    class_node = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "LLMEngine"
    )
    method_node = next(
        node
        for node in class_node.body
        if isinstance(node, ast.FunctionDef) and node.name == name
    )
    method_node.decorator_list = []
    module = ast.Module(body=[method_node], type_ignores=[])
    namespace = {
        "validate_qwen35_model_fingerprint": (
            _validate_model_fingerprint
        ),
    }
    exec(
        compile(
            ast.fix_missing_locations(module),
            str(ENGINE_PATH),
            "exec",
        ),
        namespace,
    )
    return namespace[name]


class _Engine:

    def __init__(self, rows):
        self.model_runner = types.SimpleNamespace(
            world_size=len(rows)
        )
        self.rows = list(rows)
        self.calls = []
        self.qwen35_loaded_checkpoint_candidate_binding_configuration = (
            None
        )
        self.qwen35_loaded_checkpoint_candidate_binding_rows = None

    def call_model_runner_acknowledged(
        self,
        method_name,
        *args,
        timeout_s,
    ):
        self.calls.append(
            (method_name, args, float(timeout_s))
        )
        local = self.rows[0]
        worker_acks = tuple(
            _Ack(rank, self.rows[rank])
            for rank in range(1, len(self.rows))
        )
        return local, worker_acks


def _bind(engine, timeout_s=0.25):
    method = _load_engine_method(
        "bind_qwen35_loaded_checkpoint_candidates"
    )
    return method(engine, timeout_s=timeout_s)


def _expect_error(callback, message):
    try:
        callback()
    except (ValueError, RuntimeError) as error:
        assert message in str(error), str(error)
    else:
        raise AssertionError(f"expected error containing {message!r}")


def test_engine_validates_timeout_before_dispatch():
    for timeout_s in (True, 0, -1, float("inf")):
        engine = _Engine((_row(0),))

        _expect_error(
            lambda timeout_s=timeout_s: _bind(
                engine,
                timeout_s,
            ),
            "timeout_s",
        )

        assert engine.calls == []


def test_engine_dispatches_zero_payload_and_stores_ranked_rows():
    rows = (_row(0), _row(1))
    engine = _Engine(rows)

    result = _bind(engine)

    assert result == rows
    assert engine.calls == [
        (
            "bind_published_qwen35_loaded_checkpoint_candidate",
            (),
            0.25,
        )
    ]
    assert (
        engine.qwen35_loaded_checkpoint_candidate_binding_rows
        == rows
    )
    assert (
        engine.qwen35_loaded_checkpoint_candidate_binding_configuration
        == (
            MODEL_FINGERPRINT,
            "layout-a",
            "float32",
            0.25,
        )
    )


def test_engine_completed_exact_repeat_has_zero_dispatch():
    rows = (_row(0), _row(1))
    engine = _Engine(rows)
    first = _bind(engine)
    engine.calls.clear()

    second = _bind(engine)

    assert second is first
    assert engine.calls == []


def test_engine_participant_error_leaves_completion_unset_and_retries():
    engine = _Engine((
        _row(0),
        _row(
            1,
            status="error",
            detail="candidate missing",
        ),
    ))

    _expect_error(lambda: _bind(engine), "rank=1")

    assert (
        engine.qwen35_loaded_checkpoint_candidate_binding_configuration
        is None
    )
    assert (
        engine.qwen35_loaded_checkpoint_candidate_binding_rows
        is None
    )

    engine.rows[1] = _row(1)
    result = _bind(engine)

    assert result == (_row(0), _row(1))
    assert len(engine.calls) == 2


def test_engine_rejects_invalid_ranked_result_without_completion():
    invalid_cases = (
        (
            (_row(0),),
            "result count",
            2,
        ),
        (
            (_row(1), _row(1)),
            "participant",
            2,
        ),
        (
            (
                _row(0),
                {
                    **_row(1),
                    "extra": True,
                },
            ),
            "fields",
            2,
        ),
        (
            (
                _row(0),
                {
                    **_row(1),
                    "operation": "other",
                },
            ),
            "operation",
            2,
        ),
        (
            (
                _row(0),
                {
                    **_row(1),
                    "status": "unknown",
                },
            ),
            "status",
            2,
        ),
    )
    for rows, message, world_size in invalid_cases:
        engine = _Engine(rows)
        engine.model_runner.world_size = world_size

        _expect_error(lambda: _bind(engine), message)

        assert (
            engine
            .qwen35_loaded_checkpoint_candidate_binding_configuration
            is None
        )


def test_engine_rejects_noncanonical_bound_identity():
    cases = (
        (
            _row(0, model_fingerprint="not-a-sha256"),
            "model_fingerprint",
        ),
        (
            _row(0, dtype="float64"),
            "dtype",
        ),
    )
    for row, message in cases:
        engine = _Engine((row,))

        _expect_error(lambda: _bind(engine), message)

        assert (
            engine
            .qwen35_loaded_checkpoint_candidate_binding_configuration
            is None
        )


def test_engine_rejects_cross_rank_provenance_mismatch():
    cases = (
        (
            _row(1, model_fingerprint="b" * 64),
            "model_fingerprint",
        ),
        (
            _row(1, layout_fingerprint="layout-b"),
            "layout_fingerprint",
        ),
        (
            _row(1, dtype="bfloat16"),
            "dtype",
        ),
    )
    for second, message in cases:
        engine = _Engine((_row(0), second))

        _expect_error(lambda: _bind(engine), message)

        assert (
            engine
            .qwen35_loaded_checkpoint_candidate_binding_configuration
            is None
        )


def test_engine_completed_conflict_or_damage_fails_before_dispatch():
    rows = (_row(0), _row(1))
    engine = _Engine(rows)
    _bind(engine)
    engine.calls.clear()

    _expect_error(
        lambda: _bind(engine, timeout_s=0.5),
        "already configured",
    )
    assert engine.calls == []

    engine.qwen35_loaded_checkpoint_candidate_binding_rows = None
    _expect_error(lambda: _bind(engine), "completion state")
    assert engine.calls == []

    engine.qwen35_loaded_checkpoint_candidate_binding_rows = (
        _row(0),
        _row(1, model_fingerprint="b" * 64),
    )
    _expect_error(lambda: _bind(engine), "completion state")
    assert engine.calls == []


def test_engine_step_remains_candidate_binding_free():
    source = ENGINE_PATH.read_text()
    tree = ast.parse(source, filename=str(ENGINE_PATH))
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
    step_source = ast.unparse(step_node)
    assert (
        "bind_qwen35_loaded_checkpoint_candidates"
        not in step_source
    )
    assert (
        "qwen35_loaded_checkpoint_candidate_binding_configuration"
        not in step_source
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
        "engine all-rank loaded checkpoint candidate binding tests "
        f"passed ({len(tests)} tests)"
    )


if __name__ == "__main__":
    _run()
