import math

import torch
from torch.nn import functional as F

from tinyvllm.layers.gated_delta import (
    qwen35_causal_depthwise_conv,
    qwen35_gated_delta_chunk,
)


def _official_chunk_reference(
    query,
    key,
    value,
    a,
    b,
    A_log,
    dt_bias,
    recurrent_state,
):
    initial_dtype = query.dtype
    query = query * torch.rsqrt(
        (query * query).sum(dim=-1, keepdim=True) + 1e-6
    )
    key = key * torch.rsqrt(
        (key * key).sum(dim=-1, keepdim=True) + 1e-6
    )
    beta = b.sigmoid()
    decay = -A_log.float().exp() * F.softplus(
        a.float() + dt_bias.float()
    )
    query, key, value, beta, decay = [
        tensor.transpose(0, 1).unsqueeze(0).contiguous().float()
        for tensor in (query, key, value, beta, decay)
    ]
    _, heads, sequence_length, key_dim = key.shape
    value_dim = value.shape[-1]
    chunk_size = 64
    pad_size = (
        chunk_size - sequence_length % chunk_size
    ) % chunk_size
    query = F.pad(query, (0, 0, 0, pad_size))
    key = F.pad(key, (0, 0, 0, pad_size))
    value = F.pad(value, (0, 0, 0, pad_size))
    beta = F.pad(beta, (0, pad_size))
    decay = F.pad(decay, (0, pad_size))
    total_length = sequence_length + pad_size
    query = query * (1 / math.sqrt(key_dim))
    value_beta = value * beta.unsqueeze(-1)
    key_beta = key * beta.unsqueeze(-1)
    query, key, value, key_beta, value_beta = [
        tensor.reshape(
            tensor.shape[0],
            tensor.shape[1],
            -1,
            chunk_size,
            tensor.shape[-1],
        )
        for tensor in (
            query,
            key,
            value,
            key_beta,
            value_beta,
        )
    ]
    decay = decay.reshape(
        decay.shape[0],
        decay.shape[1],
        -1,
        chunk_size,
    )
    diagonal_mask = torch.triu(
        torch.ones(
            chunk_size,
            chunk_size,
            dtype=torch.bool,
            device=query.device,
        ),
        diagonal=0,
    )
    decay = decay.cumsum(dim=-1)
    decay_mask = (
        (
            decay.unsqueeze(-1) - decay.unsqueeze(-2)
        ).tril().exp().float()
    ).tril()
    attention = -(
        (key_beta @ key.transpose(-1, -2)) * decay_mask
    ).masked_fill(diagonal_mask, 0)
    for index in range(1, chunk_size):
        row = attention[..., index, :index].clone()
        sub = attention[..., :index, :index].clone()
        attention[..., index, :index] = (
            row + (row.unsqueeze(-1) * sub).sum(-2)
        )
    attention = attention + torch.eye(
        chunk_size,
        dtype=attention.dtype,
        device=attention.device,
    )
    value = attention @ value_beta
    key_cumulative_decay = attention @ (
        key_beta * decay.exp().unsqueeze(-1)
    )
    state = recurrent_state.float().transpose(
        -1, -2
    ).unsqueeze(0)
    output = torch.zeros_like(value)
    for index in range(total_length // chunk_size):
        query_chunk = query[:, :, index]
        key_chunk = key[:, :, index]
        value_chunk = value[:, :, index]
        intra = (
            query_chunk @ key_chunk.transpose(-1, -2)
        ) * decay_mask[:, :, index]
        previous = key_cumulative_decay[:, :, index] @ state
        value_new = value_chunk - previous
        inter = (
            query_chunk
            * decay[:, :, index, :, None].exp()
        ) @ state
        output[:, :, index] = inter + intra @ value_new
        state = (
            state
            * decay[:, :, index, -1, None, None].exp()
            + (
                key_chunk
                * (
                    decay[:, :, index, -1, None]
                    - decay[:, :, index]
                ).exp()[..., None]
            ).transpose(-1, -2)
            @ value_new
        )
    output = output.reshape(
        output.shape[0],
        output.shape[1],
        -1,
        value_dim,
    )[:, :, :sequence_length]
    return (
        output.transpose(1, 2).squeeze(0).to(initial_dtype),
        state.squeeze(0).transpose(-1, -2),
    )


def test_multi_token_depthwise_conv_matches_official_fallback() -> None:
    torch.manual_seed(91)
    projected = torch.randn(
        65,
        12,
        device="cuda",
        dtype=torch.bfloat16,
    )
    state = torch.randn(
        12,
        4,
        device="cuda",
        dtype=torch.bfloat16,
    )
    weight = torch.randn(
        12,
        4,
        device="cuda",
        dtype=torch.bfloat16,
    )
    combined = torch.cat(
        (state.unsqueeze(0), projected.T.unsqueeze(0)),
        dim=-1,
    ).to(weight.dtype)
    expected_state = combined[:, :, -4:].squeeze(0)
    expected = F.silu(F.conv1d(
        combined,
        weight.unsqueeze(1),
        padding=0,
        groups=12,
    )[:, :, -65:]).transpose(1, 2).squeeze(0).to(
        projected.dtype
    )

    actual, actual_state = qwen35_causal_depthwise_conv(
        projected,
        state,
        weight,
    )

    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)
    torch.testing.assert_close(
        actual_state,
        expected_state,
        rtol=0.0,
        atol=0.0,
    )


def test_chunk_gated_delta_matches_official_fallback() -> None:
    torch.manual_seed(93)
    query = torch.randn(
        65,
        2,
        8,
        device="cuda",
        dtype=torch.bfloat16,
    )
    key = torch.randn_like(query)
    value = torch.randn_like(query)
    a = torch.randn(
        65,
        2,
        device="cuda",
        dtype=torch.bfloat16,
    )
    b = torch.randn_like(a)
    A_log = torch.randn(
        2,
        device="cuda",
        dtype=torch.float32,
    )
    dt_bias = torch.randn(
        2,
        device="cuda",
        dtype=torch.bfloat16,
    )
    recurrent_state = torch.randn(
        2,
        8,
        8,
        device="cuda",
        dtype=torch.float32,
    )
    expected, expected_state = _official_chunk_reference(
        query,
        key,
        value,
        a,
        b,
        A_log,
        dt_bias,
        recurrent_state,
    )

    actual, actual_state = qwen35_gated_delta_chunk(
        query,
        key,
        value,
        a,
        b,
        A_log,
        dt_bias,
        recurrent_state,
    )

    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)
    torch.testing.assert_close(
        actual_state,
        expected_state,
        rtol=0.0,
        atol=0.0,
    )
