import base64
import importlib.util
import io
import math
from pathlib import Path
import sys
import zlib

import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]


def _load_module():
    path = ROOT / "tinyvllm/layers/gated_delta.py"
    spec = importlib.util.spec_from_file_location(
        "qwen35_gated_delta_test_target", path
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


gated_delta = _load_module()
qwen35_causal_depthwise_conv = gated_delta.qwen35_causal_depthwise_conv
qwen35_gated_delta_recurrent = gated_delta.qwen35_gated_delta_recurrent
qwen35_gated_rmsnorm = gated_delta.qwen35_gated_rmsnorm
qwen35_l2norm = gated_delta.qwen35_l2norm

_R397_MINIMAL_CUDA_FIXTURE = (
    "eNqtVwtwVNUZPtk8XfIiPIRgI2ixCY+Q3U02yd5zNmQJS+saDFvBJDMSN8ma7LgQ"
    "TvYmHewrBSGQaiiO4qMamYpaSlVsobC59xxs0daqfRgcC6WlD0YZBqr0NdIX0+/u"
    "jZndy9CM0M38+c/5/+/7///855y7d5sC6RmE5OSQyz6FpIiEeju6I/3hJZ0hNVS"
    "+4d6o35dPWlM+A7Yvc1LanAkC7wv3buRpHWoPWOVtfWokGrO39Ybb+yLRzjY1v"
    "D7W09vW77RzW2lpczYIMSBDXWGeblLsPn+0J6Q63J837Xae0ZwGWAXPbE6H7tjQ"
    "x7MCAyrPXhUggTT8DQzynEZiCwwE0gb5dds7Onqi0XCHGulZH7Pf3tsZ7g13NkQ"
    "6VDu3lwX5FJXnBnleItS94Y08vxt1dKd3m0kcvKA7EbwwKfjUieBF27uNGNNUP"
    "j3IZySW2x+K9oX5zJQoTn49othUPuvjKLZBPnsiSrEZZY7KbwjyTyUYIV6SEsHF"
    "b0SENJXPHY+wlc8LFBrqJpN8s8o/HeTzE+B2fksKuZJ/xiSXTpDLTPICk7xQ5Yu"
    "CfHGi/vq2aE8XLzcDjO9AYgMm2r8kEbSKV5hBHYmgW7jT+OcyA1aqvCrI3Ynt7"
    "FTb2iOhGK9OqcnNa0x67TjdY/xTTDpVOQtyb6KemBpSw7xunGzmruZLu7MaCdj1"
    "5opsxrb4GslSs6HLzDANKl8e5P6+8qZAds6fRm7pGrYRYkjTFU/3VOKaON3tG9V"
    "wj3Fc/L4lluM96ScaUdVo2Mi7hZ2enYXIhlw5bz7xptyqJRV+n+eTJk36tMivyx"
    "zxCj1Jz9AKWiCInCFe0NLFZnqfXOg9yv4yeol+i65mM+R1yneVerlWdLIRrZvmy"
    "UE5JDaLRvY428NG2DbZTs8pd8m39Vqvv26zN48NyznsRXaHHGQR2S93sx2yyFsi"
    "LtA13l52lmVoi7x/F2/SHeIEbZD5+rART65SssQ8GtHq2HmRwzK8g/KCsMtmtpUp"
    "IkcEZbtsoGvlXHGWHWYrvAfFsDzEIsg9jFWMCMnieqX3dbFdPiVmsPe8u6hTDgC"
    "zSZ5XhjwXtXz9WfGgzGeb5YherFeyTrlANzq/e174n8Q4Lmn/u/MZqZ13+H2EtN"
    "BH6KCoE2X0X+L31bfRqH5OeMWvxc2yWHuMnlJe0qd4x7RcZYOmiaNaqecZdp6e"
    "Z0XUTffGh4QSv0cE9JAeEPVysXif7tPeExX0JfrT+A/ErzwZcg4tpOv0u+V0fa/"
    "4Ax2SzfQu5UktLl4XW9lutoj+jt0vuthJbb+4X2bSdnpcPCjcYo94Wf+Rto0NxX"
    "+rPMMeEzH2NI2hvy9r39ZaRLOyXO+Ts2nw8Hr5c+VuutnroJdGH1ZOsy+Imzwz9"
    "RZWj/EO7+dYGfu3tpWt1I7V7tW+KL85OkfU09uVk+IIHWMH6DGllpUn+jf/Z8f/"
    "ehX9cxr9y9X3USNG31dufQ1EkjHJ6V+aGsPl99FrOP2trZvqjOxTXrNfwnUntkm"
    "y+1KzV/p93mvK3tq644iR/56aG567ivxV154f3yCJCs4s/FLWVfTffa39f+EVI/"
    "u5ouk/vIr1V/8/1p/pIUQeIqRAIeTNOCFfGyXkWYyd0P+B78+H8bTXCDmAeR+0"
    "A7ILuOfgSwNOYDxVRxyMl8JXDVwxJWQmxiuAscHegrEX9t9Angc+G/pQLSE7IXa"
    "Mz8K2cdSM/S44j0J/AN8a5JaYByBezLuhS+Cbg3pXIMcp+D8L3voaQuZCd0F2Qh"
    "5GvC3APgDOL6D3YZ4J/BOoxa6bazwI3xhircU4E9KPGBmYLwBmDL4W2B5STF497"
    "PshP4btVsg0xCxGzH8A94ixJsyvh54FTBicQci78H0DMgv2pxE7B7YBjIuAPQg9"
    "BH4VdB44T8I3Bdh5kItY0xHYNwA3FfYXYXsI8wHg7ZhfAH4nxmH4e4DdBd8wMC"
    "egvw/7EDDDirmWOGw7gLkT2BH0jMD2BMYrgX+qxqy/GZj90Gfga4B8B/7D4DdA"
    "3oHvNug3FHMdMegQMMeB+SBu7ud+9HMTNfO1wf8AZBVsWz3m2nqM/Yd/E8YfYt"
    "wP+1vI90fMo+Avhj4Imx36mMeM3w/5KmR53MQZ+2XzmPU3QrZhfhQyBvkeMHvg"
    "E5B0xH8V+vFR8/x+ZHBQyxvQ9dDrwJ2CWuZCv4oaZsJ2UjHvgB9x8qE3GPsBbj"
    "X8R42eAPM32FuAK4K8hb7lauZZi0E6IC74T0M3wb9SM89VNmxnELMENg/mNuQth"
    "m0d5gXQq6FrRs1e7tLNekoMDmxrgD8ByR/fk0dh0+GbZuwfsLONsw49LW6ed+M8"
    "lSHeFsjzRr8gZZD5mC8D7wDGdyL+R0Y/cR5csC/XjBoS31+FHd1k/OF35adPAVk"
    "28fTpD/fG8CvB76u7xsdPq8tuVPD2SNfqyZ9/JcQ5UUF54gEYC/dGQtHIfSHjN0"
    "tbpNPvK/+EBTmczhpnRY2z0u2qqHC7Kh0OZ02FAxO3y+GqqXU43QC4qiqMKm/s/"
    "8mHpaijNFFlmlHsRJ3Jb82Fl1Vu/R1oZSe/+05NYZ6ykcvfs6305Be4/BT6mnRi"
    "eV22cpNfXlK5v8ywcB1WbvJLSyq3MMvCdVq5ya8cqdw7rFyXlZv8upDKHbNyK63"
    "c5C/6VG4w28KtsnKTv6ZTue9Yue7L+px0ySx5cyzcais3+XoUpO5RLrFeSis5+d"
    "SWpCbOI5Pcp6ZAVtaij29f+uIJZq5Fv28z9cU8U4OXnTi941chLWHLzDI5uQm8i"
    "f0vicxZuA=="
)


def _manual_recurrent(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    recurrent_state_v_k: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    key_dim = query.shape[-1]
    normalized_query = query * torch.rsqrt(
        (query * query).sum(dim=-1, keepdim=True) + 1e-6
    )
    normalized_query = normalized_query.float() / math.sqrt(key_dim)
    normalized_key = key * torch.rsqrt(
        (key * key).sum(dim=-1, keepdim=True) + 1e-6
    )
    normalized_key = normalized_key.float()
    state_k_v = recurrent_state_v_k.float().transpose(-1, -2).clone()
    outputs = []
    for token_index in range(query.shape[0]):
        beta = torch.sigmoid(b[token_index]).float()
        log_decay = -torch.exp(A_log.float()) * torch.nn.functional.softplus(
            a[token_index].float() + dt_bias.float()
        )
        state_k_v = state_k_v * torch.exp(log_decay)[:, None, None]
        memory = torch.einsum(
            "hk,hkv->hv", normalized_key[token_index], state_k_v
        )
        delta = (value[token_index].float() - memory) * beta[:, None]
        state_k_v = state_k_v + torch.einsum(
            "hk,hv->hkv", normalized_key[token_index], delta
        )
        outputs.append(
            torch.einsum(
                "hk,hkv->hv", normalized_query[token_index], state_k_v
            )
        )
    return torch.stack(outputs).to(query.dtype), state_k_v.transpose(-1, -2).to(
        recurrent_state_v_k.dtype
    )


def _official_fallback_recurrent(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    recurrent_state_v_k: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    key_dim = query.shape[-1]
    initial_dtype = query.dtype
    query = qwen35_l2norm(query).float() / math.sqrt(key_dim)
    key = qwen35_l2norm(key).float()
    value = value.float()
    beta = torch.sigmoid(b).float()
    decay = (
        -torch.exp(A_log.float())
        * F.softplus(a.float() + dt_bias.float())
    )
    state = (
        recurrent_state_v_k.float()
        .transpose(-1, -2)
        .contiguous()
    )
    outputs = []
    for token_index in range(query.shape[0]):
        state = (
            state
            * decay[token_index].exp().unsqueeze(-1).unsqueeze(-1)
        )
        memory = (
            state * key[token_index].unsqueeze(-1)
        ).sum(dim=-2)
        delta = (
            value[token_index] - memory
        ) * beta[token_index].unsqueeze(-1)
        state = (
            state
            + key[token_index].unsqueeze(-1)
            * delta.unsqueeze(-2)
        )
        outputs.append(
            (
                state * query[token_index].unsqueeze(-1)
            ).sum(dim=-2)
        )
    return (
        torch.stack(outputs).to(initial_dtype),
        state.transpose(-1, -2).to(recurrent_state_v_k.dtype),
    )


def test_l2norm_matches_explicit_formula() -> None:
    tensor = torch.tensor([[3.0, 4.0], [0.0, 0.0]])
    expected = tensor / torch.sqrt(
        torch.sum(tensor * tensor, dim=-1, keepdim=True) + 1e-6
    )
    actual = qwen35_l2norm(tensor)
    torch.testing.assert_close(actual, expected)


def test_gated_rmsnorm_matches_norm_before_silu_gate_formula() -> None:
    core = torch.tensor(
        [[1.0, -2.0, 3.0], [0.5, 0.25, -0.75]],
        dtype=torch.float32,
    )
    gate = torch.tensor(
        [[0.0, 1.0, -1.0], [2.0, -2.0, 0.5]],
        dtype=torch.float32,
    )
    weight = torch.tensor([1.5, -0.5, 0.25], dtype=torch.float32)
    original_core = core.clone()
    original_gate = gate.clone()
    original_weight = weight.clone()
    expected = core.float() * torch.rsqrt(
        core.float().pow(2).mean(dim=-1, keepdim=True) + 1e-6
    )
    expected = expected * weight.float()
    expected = expected * F.silu(gate.float())
    actual = qwen35_gated_rmsnorm(core, gate, weight)
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(core, original_core)
    torch.testing.assert_close(gate, original_gate)
    torch.testing.assert_close(weight, original_weight)


def test_gated_rmsnorm_bfloat16_and_fail_closed_contracts() -> None:
    core = torch.tensor(
        [[1.0, -2.0], [0.5, 3.0]],
        dtype=torch.bfloat16,
    )
    gate = torch.tensor(
        [[0.25, -0.5], [1.0, -1.5]],
        dtype=torch.bfloat16,
    )
    weight = torch.tensor([1.25, -0.75], dtype=torch.bfloat16)
    actual = qwen35_gated_rmsnorm(core, gate, weight)
    expected = core.float() * torch.rsqrt(
        core.float().pow(2).mean(dim=-1, keepdim=True) + 1e-6
    )
    expected = (
        expected.to(core.dtype) * weight * F.silu(gate.float())
    ).to(torch.bfloat16)
    assert actual.dtype == torch.bfloat16
    torch.testing.assert_close(actual.float(), expected.float())

    for function, message in (
        (
            lambda: qwen35_gated_rmsnorm(
                torch.ones(2, 2, dtype=torch.int64),
                torch.ones(2, 2),
                torch.ones(2),
            ),
            "floating point",
        ),
        (
            lambda: qwen35_gated_rmsnorm(
                torch.ones(2, 2),
                torch.ones(1, 2),
                torch.ones(2),
            ),
            "shape",
        ),
        (
            lambda: qwen35_gated_rmsnorm(
                torch.ones(2, 2),
                torch.ones(2, 2, dtype=torch.float64),
                torch.ones(2),
            ),
            "dtype",
        ),
        (
            lambda: qwen35_gated_rmsnorm(
                torch.ones(2, 2),
                torch.ones(2, 2),
                torch.ones(3),
            ),
            "weight",
        ),
        (
            lambda: qwen35_gated_rmsnorm(
                torch.ones(2, 2),
                torch.ones(2, 2),
                torch.ones(2),
                eps=0,
            ),
            "eps",
        ),
        (
            lambda: qwen35_gated_rmsnorm(
                torch.ones(2, 2),
                torch.ones(2, 2),
                torch.ones(2, device="meta"),
            ),
            "device",
        ),
    ):
        _expect_value_error(function, message)


def test_gated_rmsnorm_preserves_float32_stable_weight() -> None:
    core = torch.tensor(
        [[1.0, -2.0], [0.5, 3.0]],
        dtype=torch.bfloat16,
    )
    gate = torch.tensor(
        [[0.25, -0.5], [1.0, -1.5]],
        dtype=torch.bfloat16,
    )
    weight = torch.tensor([1.25, -0.75], dtype=torch.float32)
    original_core = core.clone()
    original_gate = gate.clone()
    original_weight = weight.clone()
    expected = core.float() * torch.rsqrt(
        core.float().pow(2).mean(dim=-1, keepdim=True) + 1e-6
    )
    expected = (
        expected.to(core.dtype) * weight * F.silu(gate.float())
    ).to(torch.bfloat16)

    actual = qwen35_gated_rmsnorm(core, gate, weight)

    assert actual.dtype == torch.bfloat16
    assert weight.dtype == torch.float32
    torch.testing.assert_close(actual.float(), expected.float())
    torch.testing.assert_close(core, original_core)
    torch.testing.assert_close(gate, original_gate)
    torch.testing.assert_close(weight, original_weight)


def test_recurrent_matches_asymmetric_manual_oracle() -> None:
    query = torch.tensor(
        [[[0.5, -1.0]], [[1.5, 0.25]]],
        dtype=torch.float32,
    )
    key = torch.tensor(
        [[[1.0, 2.0]], [[-0.75, 0.5]]],
        dtype=torch.float32,
    )
    value = torch.tensor(
        [[[0.25, -0.5, 1.0]], [[1.25, 0.75, -0.25]]],
        dtype=torch.float32,
    )
    a = torch.tensor([[-0.2], [0.4]], dtype=torch.float32)
    b = torch.tensor([[0.3], [-0.7]], dtype=torch.float32)
    A_log = torch.tensor([-0.35], dtype=torch.float32)
    dt_bias = torch.tensor([0.15], dtype=torch.float32)
    recurrent_state_v_k = torch.tensor(
        [[[0.1, 0.2], [-0.3, 0.4], [0.5, -0.6]]],
        dtype=torch.float32,
    )
    original_state = recurrent_state_v_k.clone()

    expected_output, expected_state = _manual_recurrent(
        query,
        key,
        value,
        a,
        b,
        A_log,
        dt_bias,
        recurrent_state_v_k,
    )
    actual_output, actual_state = qwen35_gated_delta_recurrent(
        query,
        key,
        value,
        a,
        b,
        A_log,
        dt_bias,
        recurrent_state_v_k,
    )

    torch.testing.assert_close(actual_output, expected_output)
    torch.testing.assert_close(actual_state, expected_state)
    assert actual_state.shape == (1, 3, 2)
    torch.testing.assert_close(recurrent_state_v_k, original_state)


def test_recurrent_bitwise_matches_transformers_fallback_reductions() -> None:
    generator = torch.Generator().manual_seed(0)
    tensors = (
        torch.randn(1, 4, 16, generator=generator),
        torch.randn(1, 4, 16, generator=generator),
        torch.randn(1, 4, 16, generator=generator),
        torch.randn(1, 4, generator=generator),
        torch.randn(1, 4, generator=generator),
        torch.randn(4, generator=generator),
        torch.randn(4, generator=generator),
        torch.randn(4, 16, 16, generator=generator),
    )
    expected_output, expected_state = _official_fallback_recurrent(
        *tensors
    )
    actual_output, actual_state = qwen35_gated_delta_recurrent(
        *tensors
    )

    assert torch.equal(actual_output, expected_output)
    assert torch.equal(actual_state, expected_state)


def test_recurrent_cuda_matches_official_contiguous_state_reduction() -> None:
    if not torch.cuda.is_available():
        return
    fixture = torch.load(
        io.BytesIO(
            zlib.decompress(
                base64.b64decode(_R397_MINIMAL_CUDA_FIXTURE)
            )
        ),
        map_location="cuda",
        weights_only=True,
    )
    query = (
        qwen35_l2norm(fixture["query"]).float()
        / math.sqrt(fixture["query"].shape[-1])
    )
    key = qwen35_l2norm(fixture["key"]).float()
    value = fixture["value"].float()
    beta = torch.sigmoid(fixture["b"]).float()
    decay = (
        -torch.exp(fixture["A_log"].float())
        * F.softplus(
            fixture["a"].float()
            + fixture["dt_bias"].float()
        )
    )
    state = (
        fixture["state"].float()
        .transpose(-1, -2)
        .contiguous()
    )
    state = state * decay[0].exp()[:, None, None]
    memory = (state * key[0].unsqueeze(-1)).sum(dim=-2)
    delta = (value[0] - memory) * beta[0].unsqueeze(-1)
    state = state + key[0].unsqueeze(-1) * delta.unsqueeze(-2)
    expected_output = (
        (state * query[0].unsqueeze(-1))
        .sum(dim=-2)
        .unsqueeze(0)
        .to(fixture["query"].dtype)
    )
    expected_state = state.transpose(-1, -2).to(
        fixture["state"].dtype
    )
    actual_output, actual_state = qwen35_gated_delta_recurrent(
        fixture["query"],
        fixture["key"],
        fixture["value"],
        fixture["a"],
        fixture["b"],
        fixture["A_log"],
        fixture["dt_bias"],
        fixture["state"],
    )

    assert torch.equal(actual_output, expected_output)
    assert torch.equal(actual_state, expected_state)


def _recurrent_fixture(
    *,
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, ...]:
    generator = torch.Generator().manual_seed(17)
    query = torch.randn(4, 2, 2, generator=generator).to(dtype)
    key = torch.randn(4, 2, 2, generator=generator).to(dtype)
    value = torch.randn(4, 2, 3, generator=generator).to(dtype)
    a = torch.randn(4, 2, generator=generator).to(dtype)
    b = torch.randn(4, 2, generator=generator).to(dtype)
    A_log = torch.randn(2, generator=generator).to(dtype)
    dt_bias = torch.randn(2, generator=generator).to(dtype)
    state = torch.randn(2, 3, 2, generator=generator).to(dtype)
    return query, key, value, a, b, A_log, dt_bias, state


def test_recurrent_split_continuation_matches_one_shot() -> None:
    query, key, value, a, b, A_log, dt_bias, state = _recurrent_fixture()
    full_output, full_state = qwen35_gated_delta_recurrent(
        query, key, value, a, b, A_log, dt_bias, state
    )
    first_output, first_state = qwen35_gated_delta_recurrent(
        query[:2],
        key[:2],
        value[:2],
        a[:2],
        b[:2],
        A_log,
        dt_bias,
        state,
    )
    second_output, second_state = qwen35_gated_delta_recurrent(
        query[2:],
        key[2:],
        value[2:],
        a[2:],
        b[2:],
        A_log,
        dt_bias,
        first_state,
    )
    torch.testing.assert_close(
        torch.cat((first_output, second_output)), full_output
    )
    torch.testing.assert_close(second_state, full_state)


def test_recurrent_request_states_remain_isolated() -> None:
    query, key, value, a, b, A_log, dt_bias, state = _recurrent_fixture()
    request_zero_state = state[0:1].clone()
    request_one_state = state[1:2].clone()
    _, updated_zero = qwen35_gated_delta_recurrent(
        query[:, 0:1],
        key[:, 0:1],
        value[:, 0:1],
        a[:, 0:1],
        b[:, 0:1],
        A_log[0:1],
        dt_bias[0:1],
        request_zero_state,
    )
    _, updated_one = qwen35_gated_delta_recurrent(
        query[:, 1:2],
        key[:, 1:2],
        value[:, 1:2],
        a[:, 1:2],
        b[:, 1:2],
        A_log[1:2],
        dt_bias[1:2],
        request_one_state,
    )
    _, batched_state = qwen35_gated_delta_recurrent(
        query, key, value, a, b, A_log, dt_bias, state
    )
    torch.testing.assert_close(updated_zero, batched_state[0:1])
    torch.testing.assert_close(updated_one, batched_state[1:2])
    torch.testing.assert_close(state[0:1], request_zero_state)
    torch.testing.assert_close(state[1:2], request_one_state)


def test_recurrent_bfloat16_preserves_contract_with_fp32_accumulation() -> None:
    tensors = _recurrent_fixture(dtype=torch.bfloat16)
    output, state = qwen35_gated_delta_recurrent(*tensors)
    expected_output, expected_state = _manual_recurrent(*tensors)
    assert output.dtype == torch.bfloat16
    assert state.dtype == torch.bfloat16
    torch.testing.assert_close(output.float(), expected_output.float())
    torch.testing.assert_close(state.float(), expected_state.float())


def _expect_value_error(function, message: str) -> None:
    try:
        function()
    except ValueError as error:
        assert message in str(error), str(error)
    else:
        raise AssertionError(f"expected ValueError containing {message!r}")


def test_recurrent_rejects_invalid_shapes() -> None:
    query, key, value, a, b, A_log, dt_bias, state = _recurrent_fixture()
    _expect_value_error(
        lambda: qwen35_gated_delta_recurrent(
            query, key[:-1], value, a, b, A_log, dt_bias, state
        ),
        "query and key",
    )
    _expect_value_error(
        lambda: qwen35_gated_delta_recurrent(
            query, key, value, a[:-1], b, A_log, dt_bias, state
        ),
        "a and b",
    )
    _expect_value_error(
        lambda: qwen35_gated_delta_recurrent(
            query, key, value, a, b, A_log[:-1], dt_bias, state
        ),
        "A_log and dt_bias",
    )
    _expect_value_error(
        lambda: qwen35_gated_delta_recurrent(
            query, key, value, a, b, A_log, dt_bias, state.transpose(-1, -2)
        ),
        "physical recurrent state",
    )


def test_reference_primitives_reject_non_floating_tensors() -> None:
    _expect_value_error(
        lambda: qwen35_l2norm(torch.ones(2, 2, dtype=torch.int64)),
        "floating point",
    )
    query, key, value, a, b, A_log, dt_bias, state = _recurrent_fixture()
    _expect_value_error(
        lambda: qwen35_gated_delta_recurrent(
            query.to(torch.int64),
            key,
            value,
            a,
            b,
            A_log,
            dt_bias,
            state,
        ),
        "floating point",
    )
    _expect_value_error(
        lambda: qwen35_causal_depthwise_conv(
            torch.ones(2, 3, dtype=torch.int64),
            torch.ones(3, 4, dtype=torch.int64),
            torch.ones(3, 4, dtype=torch.int64),
        ),
        "floating point",
    )


def _manual_causal_depthwise_conv(
    projected_qkv: torch.Tensor,
    conv_state: torch.Tensor,
    weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    state = conv_state.clone()
    outputs = []
    for token in projected_qkv:
        state = torch.cat((state[:, 1:], token[:, None]), dim=-1)
        outputs.append(F.silu(torch.sum(state * weight, dim=-1)))
    return torch.stack(outputs), state


def test_causal_depthwise_conv_split_matches_one_shot() -> None:
    projected_qkv = torch.tensor(
        [
            [1.0, -2.0, 0.5],
            [0.25, 1.5, -1.0],
            [-0.75, 0.5, 2.0],
            [1.25, -0.25, 0.75],
        ]
    )
    conv_state = torch.tensor(
        [
            [0.1, 0.2, 0.3],
            [-0.4, 0.5, -0.6],
            [0.7, -0.8, 0.9],
        ]
    )
    weight = torch.tensor(
        [
            [1.0, 2.0, 3.0],
            [-1.0, 0.5, 2.5],
            [0.25, -0.75, 1.5],
        ]
    )
    expected_output, expected_state = _manual_causal_depthwise_conv(
        projected_qkv, conv_state, weight
    )
    full_output, full_state = qwen35_causal_depthwise_conv(
        projected_qkv, conv_state, weight
    )
    first_output, first_state = qwen35_causal_depthwise_conv(
        projected_qkv[:2], conv_state, weight
    )
    second_output, second_state = qwen35_causal_depthwise_conv(
        projected_qkv[2:], first_state, weight
    )
    torch.testing.assert_close(full_output, expected_output)
    torch.testing.assert_close(full_state, expected_state)
    torch.testing.assert_close(
        torch.cat((first_output, second_output)), full_output
    )
    torch.testing.assert_close(second_state, full_state)


def test_causal_depthwise_conv_preserves_latest_window_and_inputs() -> None:
    projected_qkv = torch.tensor(
        [[1.0, 10.0], [2.0, 20.0], [3.0, 30.0], [4.0, 40.0]],
        dtype=torch.bfloat16,
    )
    conv_state = torch.tensor(
        [[-2.0, -1.0, 0.0], [-20.0, -10.0, 0.0]],
        dtype=torch.bfloat16,
    )
    weight = torch.ones_like(conv_state)
    original_projected = projected_qkv.clone()
    original_state = conv_state.clone()
    original_weight = weight.clone()

    output, state = qwen35_causal_depthwise_conv(
        projected_qkv, conv_state, weight
    )

    assert output.dtype == torch.bfloat16
    assert state.dtype == torch.bfloat16
    torch.testing.assert_close(
        state,
        torch.tensor(
            [[2.0, 3.0, 4.0], [20.0, 30.0, 40.0]],
            dtype=torch.bfloat16,
        ),
    )
    torch.testing.assert_close(projected_qkv, original_projected)
    torch.testing.assert_close(conv_state, original_state)
    torch.testing.assert_close(weight, original_weight)


def test_causal_depthwise_conv_rejects_invalid_contracts() -> None:
    projected_qkv = torch.ones(2, 3)
    conv_state = torch.ones(3, 4)
    weight = torch.ones(3, 4)
    _expect_value_error(
        lambda: qwen35_causal_depthwise_conv(
            projected_qkv[:, :2], conv_state, weight
        ),
        "channels",
    )
    _expect_value_error(
        lambda: qwen35_causal_depthwise_conv(
            projected_qkv, conv_state, weight[:, :-1]
        ),
        "weight",
    )
    _expect_value_error(
        lambda: qwen35_causal_depthwise_conv(
            projected_qkv, conv_state, weight, activation="relu"
        ),
        "activation",
    )


def main() -> None:
    test_l2norm_matches_explicit_formula()
    test_gated_rmsnorm_matches_norm_before_silu_gate_formula()
    test_gated_rmsnorm_bfloat16_and_fail_closed_contracts()
    test_gated_rmsnorm_preserves_float32_stable_weight()
    test_recurrent_matches_asymmetric_manual_oracle()
    test_recurrent_bitwise_matches_transformers_fallback_reductions()
    test_recurrent_cuda_matches_official_contiguous_state_reduction()
    test_recurrent_split_continuation_matches_one_shot()
    test_recurrent_request_states_remain_isolated()
    test_recurrent_bfloat16_preserves_contract_with_fp32_accumulation()
    test_recurrent_rejects_invalid_shapes()
    test_reference_primitives_reject_non_floating_tensors()
    test_causal_depthwise_conv_split_matches_one_shot()
    test_causal_depthwise_conv_preserves_latest_window_and_inputs()
    test_causal_depthwise_conv_rejects_invalid_contracts()
    print("qwen35 gated delta reference tests passed")


if __name__ == "__main__":
    main()
