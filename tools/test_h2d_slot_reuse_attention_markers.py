"""Dependency-light source contract tests for blockwise read markers."""

from __future__ import annotations

import ast
import os

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_THIS_DIR)
_ATTENTION_PATH = os.path.join(
    _REPO_ROOT,
    "tinyvllm",
    "layers",
    "attention.py",
)


def _function(name):
    tree = ast.parse(open(_ATTENTION_PATH).read())
    return next(
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == name
    )


def _marker_calls(function):
    return [
        node
        for node in ast.walk(function)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "record_h2d_slot_read_window"
    ]


def _cache_read_lines(function):
    lines = []
    for node in ast.walk(function):
        if not isinstance(node, ast.Subscript):
            continue
        value = node.value
        if isinstance(value, ast.Name) and value.id in {
            "k_cache",
            "v_cache",
        }:
            lines.append(node.lineno)
    return lines


def _dense_float_lines(function):
    lines = []
    for node in ast.walk(function):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "to"
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id in {"k_dense", "v_dense"}
        ):
            lines.append(node.lineno)
    return lines


def _assert_marker_contract(function_name, stage):
    function = _function(function_name)
    calls = _marker_calls(function)
    assert len(calls) == 1
    call = calls[0]
    keywords = {
        keyword.arg: keyword.value
        for keyword in call.keywords
    }
    assert ast.literal_eval(keywords["attention_stage"]) == stage
    assert isinstance(keywords["layer_index"], ast.Call)
    assert isinstance(keywords["window_ordinal"], ast.Name)
    assert keywords["window_ordinal"].id == "window_ordinal"
    assert max(_cache_read_lines(function)) < call.lineno
    assert call.lineno < min(_dense_float_lines(function))
    return function, call


def test_decode_marker_is_after_all_slot_reads_and_before_compute():
    function, call = _assert_marker_contract(
        "_blockwise_online_decode_attention",
        "decode",
    )
    assert any(
        isinstance(node, ast.For)
        and isinstance(node.target, ast.Tuple)
        and any(
            isinstance(element, ast.Name)
            and element.id == "window_ordinal"
            for element in node.target.elts
        )
        and isinstance(node.iter, ast.Call)
        and isinstance(node.iter.func, ast.Name)
        and node.iter.func.id == "enumerate"
        for node in ast.walk(function)
    )
    assert isinstance(
        {
            keyword.arg: keyword.value
            for keyword in call.keywords
        }["logical_blocks"],
        ast.Call,
    )


def test_spec_verify_marker_is_after_all_slot_reads_and_before_compute():
    _assert_marker_contract(
        "_blockwise_online_spec_verify_attention",
        "spec_verify",
    )


def test_prefill_marker_covers_only_historical_prefix_windows():
    function, call = _assert_marker_contract(
        "_blockwise_online_prefill_attention",
        "prefill",
    )
    marker_loop = min(
        (
            node
            for node in ast.walk(function)
            if isinstance(node, ast.For)
            and any(child is call for child in ast.walk(node))
        ),
        key=lambda node: node.end_lineno - node.lineno,
    )
    assert isinstance(marker_loop.iter, ast.Call) is False
    assert isinstance(marker_loop.iter, ast.Subscript)
    assert ast.literal_eval(marker_loop.iter.slice) == "windows"
    assert call.lineno < next(
        node.lineno
        for node in ast.walk(function)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name)
            and target.id == "k_local"
            for target in node.targets
        )
    )


def test_prefill_accepts_and_forward_passes_layer_index():
    function = _function("_blockwise_online_prefill_attention")
    assert function.args.args[-1].arg == "layer_idx"
    assert ast.literal_eval(function.args.defaults[-1]) == -1
    forward = next(
        node
        for node in ast.parse(open(_ATTENTION_PATH).read()).body
        if isinstance(node, ast.ClassDef)
        and node.name == "Attention"
    )
    calls = [
        node
        for node in ast.walk(forward)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_blockwise_online_prefill_attention"
    ]
    assert len(calls) == 1
    assert "layer_idx" in {
        keyword.arg for keyword in calls[0].keywords
    }
