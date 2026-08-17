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
_MISSING_MODULE = object()
_SAVED_MODULES = {}


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

    def view(self, *shape):
        return FakeTensor(shape, self.values)

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
    if name not in _SAVED_MODULES:
        _SAVED_MODULES[name] = sys.modules.get(
            name,
            _MISSING_MODULE,
        )
    module = types.ModuleType(name)
    for key, value in attributes.items():
        setattr(module, key, value)
    sys.modules[name] = module
    return module


def _restore_modules():
    for name, previous in reversed(tuple(_SAVED_MODULES.items())):
        if previous is _MISSING_MODULE:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = previous
    _SAVED_MODULES.clear()


def _load_attention_module():
    try:
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
        libdevice_module = _install_module(
            "triton.language.extra.libdevice"
        )
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
            build_attention_matching_prefill_cache=(
                lambda *args, **kwargs: None
            ),
        )

        module = types.ModuleType(
            "native_verifier_attention_under_test"
        )
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
    finally:
        _restore_modules()


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
        spec_verify_query_lens=(3,),
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


def test_spec_verify_helper_batches_multiple_rows_in_one_call():
    captured = []

    def fake_flash(q, k_cache, v_cache, **kwargs):
        captured.append({
            "q_shape": q.shape,
            "cache_seqlens": kwargs[
                "cache_seqlens"
            ].tolist(),
            "block_table": kwargs["block_table"].tolist(),
            "causal": kwargs["causal"],
            "num_splits": kwargs["num_splits"],
        })
        return q

    attention.flash_attn_with_kvcache = fake_flash
    q = FakeTensor((4, 4, 8))
    cache = FakeTensor((16, 4, 2, 8))
    context = types.SimpleNamespace(
        context_lens=FakeTensor((2,), [6, 10]),
        block_tables=FakeTensor(
            (2, 3),
            [[5, 6, -1], [10, 11, 12]],
        ),
        spec_verify_query_lens=(2, 2),
        flash_attn_num_splits=16,
    )

    output = attention._flash_attn_spec_verify(
        q,
        cache,
        cache,
        context,
        0.125,
    )

    assert output.shape == (4, 4, 8)
    assert captured == [{
        "q_shape": (2, 2, 4, 8),
        "cache_seqlens": [6, 10],
        "block_table": [[5, 6, -1], [10, 11, 12]],
        "causal": True,
        "num_splits": 16,
    }]


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
            spec_verify_query_lens=(3, 3),
        ),
        types.SimpleNamespace(
            context_lens=FakeTensor((1,), [55]),
            block_tables=None,
            spec_verify_query_lens=(3,),
        ),
        types.SimpleNamespace(
            context_lens=FakeTensor((1,), [55]),
            block_tables=FakeTensor((2, 1), [[0], [1]]),
            spec_verify_query_lens=(3,),
        ),
        types.SimpleNamespace(
            context_lens=FakeTensor((1,), [55]),
            block_tables=FakeTensor((1, 1), [[0]]),
            spec_verify_query_lens=(),
        ),
        types.SimpleNamespace(
            context_lens=FakeTensor((2,), [55, 56]),
            block_tables=FakeTensor((2, 1), [[0], [1]]),
            spec_verify_query_lens=(1, 2),
        ),
        types.SimpleNamespace(
            context_lens=FakeTensor((1,), [55]),
            block_tables=FakeTensor((1, 1), [[0]]),
            spec_verify_query_lens=(0,),
        ),
        types.SimpleNamespace(
            context_lens=FakeTensor((2,), [55, 56]),
            block_tables=FakeTensor((2, 1), [[0], [1]]),
            spec_verify_query_lens=(2, 2),
        ),
    )

    for index, context in enumerate(invalid_contexts):
        candidate_q = (
            FakeTensor((3, 4, 8))
            if index != len(invalid_contexts) - 1
            else FakeTensor((3, 4, 8))
        )
        try:
            attention._flash_attn_spec_verify(
                candidate_q,
                cache,
                cache,
                context,
                0.125,
            )
        except RuntimeError:
            pass
        else:
            raise AssertionError(context)


def test_blockwise_spec_verify_forward_dispatches_to_online_helper():
    captured = {}

    def fake_blockwise(
        q,
        k_cache,
        v_cache,
        context,
        num_heads,
        head_dim,
        scale,
        layer_idx=-1,
    ):
        captured["q_shape"] = q.shape
        captured["num_heads"] = num_heads
        captured["head_dim"] = head_dim
        captured["scale"] = scale
        captured["layer_idx"] = layer_idx
        return q

    attention._blockwise_online_spec_verify_attention = fake_blockwise
    attention._flash_attn_spec_verify = lambda *args, **kwargs: (
        _ for _ in ()
    ).throw(AssertionError("flash verifier should not run"))
    attention.store_kvcache = lambda *args, **kwargs: None
    context = types.SimpleNamespace(
        mode="spec_verify",
        slot_mapping=FakeTensor((4,), [0, 1, 2, 3]),
        kv_offload_blockwise_decode=True,
    )
    attention.get_context = lambda: context
    layer = attention.Attention(
        num_heads=4,
        head_dim=8,
        scale=0.125,
        num_kv_heads=2,
    )
    layer.layer_idx = 3
    layer.k_cache = FakeTensor((12, 4, 2, 8))
    layer.v_cache = FakeTensor((12, 4, 2, 8))

    output = layer.forward(
        FakeTensor((4, 32)),
        FakeTensor((4, 16)),
        FakeTensor((4, 16)),
    )

    assert output.shape == (-1, 32)
    assert captured == {
        "q_shape": (-1, 4, 8),
        "num_heads": 4,
        "head_dim": 8,
        "scale": 0.125,
        "layer_idx": 3,
    }


def test_decode_small_page_gathers_dense_before_flash():
    captured = {}

    def fake_gather(k_cache, v_cache, block_tables):
        captured["gather"] = {
            "k_shape": k_cache.shape,
            "v_shape": v_cache.shape,
            "block_tables": block_tables.tolist(),
        }
        return (
            FakeTensor((2, 3, 2, 8)),
            FakeTensor((2, 3, 2, 8)),
        )

    def fake_flash(q, k_cache, v_cache, **kwargs):
        captured["flash"] = {
            "q_shape": q.shape,
            "k_shape": k_cache.shape,
            "v_shape": v_cache.shape,
            "cache_seqlens": kwargs["cache_seqlens"].tolist(),
            "block_table": kwargs["block_table"],
            "causal": kwargs["causal"],
            "num_splits": kwargs["num_splits"],
        }
        return q

    attention.store_kvcache = lambda *args, **kwargs: None
    attention.gather_kv_cache_dense = fake_gather
    attention.flash_attn_with_kvcache = fake_flash
    context = types.SimpleNamespace(
        mode="decode",
        slot_mapping=FakeTensor((2,), [2, 5]),
        context_lens=FakeTensor((2,), [2, 3]),
        block_tables=FakeTensor(
            (2, 3),
            [[0, 2, -1], [1, 4, 5]],
        ),
        kv_offload_blockwise_decode=False,
        quest_top_k_blocks=-1,
        am_compact_blocks=0,
        flash_attn_num_splits=16,
    )
    attention.get_context = lambda: context
    layer = attention.Attention(
        num_heads=4,
        head_dim=8,
        scale=0.125,
        num_kv_heads=2,
    )
    layer.k_cache = FakeTensor((8, 1, 2, 8))
    layer.v_cache = FakeTensor((8, 1, 2, 8))

    output = layer.forward(
        FakeTensor((2, 32)),
        FakeTensor((2, 16)),
        FakeTensor((2, 16)),
    )

    assert output.shape == (-1, 32)
    assert captured == {
        "gather": {
            "k_shape": (8, 1, 2, 8),
            "v_shape": (8, 1, 2, 8),
            "block_tables": [[0, 2, -1], [1, 4, 5]],
        },
        "flash": {
            "q_shape": (-1, 1, 4, 8),
            "k_shape": (2, 3, 2, 8),
            "v_shape": (2, 3, 2, 8),
            "cache_seqlens": [2, 3],
            "block_table": None,
            "causal": True,
            "num_splits": 16,
        },
    }


def test_decode_supported_page_keeps_paged_flash():
    captured = {}

    def fake_gather(*args, **kwargs):
        raise AssertionError("supported paged cache should not gather")

    def fake_flash(q, k_cache, v_cache, **kwargs):
        captured["flash"] = {
            "q_shape": q.shape,
            "k_shape": k_cache.shape,
            "v_shape": v_cache.shape,
            "cache_seqlens": kwargs["cache_seqlens"].tolist(),
            "block_table": kwargs["block_table"].tolist(),
            "causal": kwargs["causal"],
            "num_splits": kwargs["num_splits"],
        }
        return q

    attention.store_kvcache = lambda *args, **kwargs: None
    attention.gather_kv_cache_dense = fake_gather
    attention.flash_attn_with_kvcache = fake_flash
    context = types.SimpleNamespace(
        mode="decode",
        slot_mapping=FakeTensor((2,), [2, 5]),
        context_lens=FakeTensor((2,), [2, 3]),
        block_tables=FakeTensor(
            (2, 2),
            [[0, -1], [1, -1]],
        ),
        kv_offload_blockwise_decode=False,
        quest_top_k_blocks=-1,
        am_compact_blocks=0,
        flash_attn_num_splits=16,
    )
    attention.get_context = lambda: context
    layer = attention.Attention(
        num_heads=4,
        head_dim=8,
        scale=0.125,
        num_kv_heads=2,
    )
    layer.k_cache = FakeTensor((8, 256, 2, 8))
    layer.v_cache = FakeTensor((8, 256, 2, 8))

    output = layer.forward(
        FakeTensor((2, 32)),
        FakeTensor((2, 16)),
        FakeTensor((2, 16)),
    )

    assert output.shape == (-1, 32)
    assert captured == {
        "flash": {
            "q_shape": (-1, 1, 4, 8),
            "k_shape": (8, 256, 2, 8),
            "v_shape": (8, 256, 2, 8),
            "cache_seqlens": [2, 3],
            "block_table": [[0, -1], [1, -1]],
            "causal": True,
            "num_splits": 16,
        },
    }


def main():
    test_spec_verify_helper_uses_single_multi_query_row()
    test_spec_verify_helper_batches_multiple_rows_in_one_call()
    test_spec_verify_helper_rejects_invalid_rows()
    test_blockwise_spec_verify_forward_dispatches_to_online_helper()
    test_decode_small_page_gathers_dense_before_flash()
    test_decode_supported_page_keeps_paged_flash()
    print("native verifier attention dispatch tests passed")
    print("CUDA numerical capability cases deferred to remote gate")


if __name__ == "__main__":
    main()
