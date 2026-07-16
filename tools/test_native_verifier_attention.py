"""Dependency-light dispatch tests for native verifier attention."""

from __future__ import annotations

import __future__
import os
import sys
import types

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_THIS_DIR)
_ATTENTION_PATH = os.path.join(
    _REPO_ROOT,
    "tinyvllm",
    "layers",
    "attention.py",
)


class FakeTensor:
    def __init__(self, shape, values=None):
        self.shape = tuple(shape)
        self.values = values

    def unsqueeze(self, dim):
        shape = list(self.shape)
        shape.insert(dim, 1)
        return FakeTensor(shape, self.values)

    def view_as(self, other):
        return FakeTensor(other.shape, self.values)

    def numel(self):
        count = 1
        for size in self.shape:
            count *= size
        return count

    def size(self, dim):
        return self.shape[dim]

    def tolist(self):
        return self.values


def _install_module(name: str, **attributes):
    module = types.ModuleType(name)
    for key, value in attributes.items():
        setattr(module, key, value)
    sys.modules[name] = module
    return module


def _load_attention_module():
    torch_module = _install_module("torch", Tensor=FakeTensor)
    nn_module = _install_module(
        "torch.nn",
        Module=type("Module", (), {}),
    )
    torch_module.nn = nn_module

    triton_module = _install_module(
        "triton",
        jit=lambda function: function,
        cdiv=lambda value, divisor: (value + divisor - 1) // divisor,
        next_power_of_2=lambda value: 1 << (value - 1).bit_length(),
    )
    tl_module = _install_module(
        "triton.language",
        constexpr=object,
    )
    extra_module = _install_module("triton.language.extra")
    libdevice_module = _install_module("triton.language.extra.libdevice")
    triton_module.language = tl_module
    tl_module.extra = extra_module
    extra_module.libdevice = libdevice_module

    _install_module(
        "flash_attn",
        flash_attn_varlen_func=lambda *args, **kwargs: None,
        flash_attn_with_kvcache=lambda *args, **kwargs: None,
    )
    _install_module("tinyvllm")
    _install_module("tinyvllm.utils")
    _install_module(
        "tinyvllm.utils.context",
        am_compact_layer_enabled=lambda *args, **kwargs: False,
        get_context=lambda: None,
    )
    _install_module("tinyvllm.engine")
    _install_module(
        "tinyvllm.engine.attention_matching",
        AttentionMatchingDecodeCache=type(
            "AttentionMatchingDecodeCache",
            (),
            {},
        ),
        attention_matching_decode=lambda *args, **kwargs: None,
        build_attention_matching_prefill_cache=lambda *args, **kwargs: None,
    )

    module = types.ModuleType("native_verifier_attention_under_test")
    module.__file__ = _ATTENTION_PATH
    sys.modules[module.__name__] = module
    source = open(_ATTENTION_PATH).read()
    code = compile(
        source,
        _ATTENTION_PATH,
        "exec",
        flags=__future__.annotations.compiler_flag,
    )
    exec(code, module.__dict__)
    return module


attention = _load_attention_module()


def test_spec_verify_helper_uses_single_multi_query_row():
    captured = {}

    def fake_flash(q, k_cache, v_cache, **kwargs):
        captured["q_shape"] = q.shape
        captured["cache_seqlens"] = kwargs["cache_seqlens"].tolist()
        captured["block_table"] = kwargs["block_table"].tolist()
        captured["causal"] = kwargs["causal"]
        captured["num_splits"] = kwargs["num_splits"]
        return q

    attention.flash_attn_with_kvcache = fake_flash
    q = FakeTensor((3, 4, 8))
    cache = FakeTensor((2, 256, 2, 8))
    context = types.SimpleNamespace(
        context_lens=FakeTensor((1,), [55]),
        block_tables=FakeTensor((1, 1), [[0]]),
        flash_attn_num_splits=16,
    )

    output = attention._flash_attn_spec_verify(
        q,
        cache,
        cache,
        context,
        0.125,
    )

    assert output.shape == (3, 4, 8)
    assert captured == {
        "q_shape": (1, 3, 4, 8),
        "cache_seqlens": [55],
        "block_table": [[0]],
        "causal": True,
        "num_splits": 16,
    }


def test_spec_verify_helper_rejects_invalid_rows():
    q = FakeTensor((3, 4, 8))
    cache = FakeTensor((2, 256, 2, 8))
    invalid_contexts = (
        types.SimpleNamespace(
            context_lens=None,
            block_tables=FakeTensor((1, 1), [[0]]),
        ),
        types.SimpleNamespace(
            context_lens=FakeTensor((2,), [55, 55]),
            block_tables=FakeTensor((1, 1), [[0]]),
        ),
        types.SimpleNamespace(
            context_lens=FakeTensor((1,), [55]),
            block_tables=None,
        ),
        types.SimpleNamespace(
            context_lens=FakeTensor((1,), [55]),
            block_tables=FakeTensor((2, 1), [[0], [1]]),
        ),
    )

    for context in invalid_contexts:
        try:
            attention._flash_attn_spec_verify(
                q,
                cache,
                cache,
                context,
                0.125,
            )
        except RuntimeError:
            pass
        else:
            raise AssertionError(context)


def main():
    test_spec_verify_helper_uses_single_multi_query_row()
    test_spec_verify_helper_rejects_invalid_rows()
    print("native verifier attention dispatch tests passed")
    print("CUDA numerical capability cases deferred to remote gate")


if __name__ == "__main__":
    main()
