import math

import torch
from torch.nn import functional as F


def _require_floating_tensors(**tensors: torch.Tensor) -> None:
    non_floating = [
        name for name, tensor in tensors.items()
        if not tensor.is_floating_point()
    ]
    if non_floating:
        raise ValueError(
            "reference math requires floating point tensors: "
            + ", ".join(non_floating)
        )


def qwen35_l2norm(
    tensor: torch.Tensor,
    *,
    eps: float = 1e-6,
) -> torch.Tensor:
    if tensor.ndim == 0:
        raise ValueError("tensor must have at least one dimension")
    if eps <= 0:
        raise ValueError("eps must be positive")
    _require_floating_tensors(tensor=tensor)
    normalized = tensor * torch.rsqrt(
        torch.sum(tensor * tensor, dim=-1, keepdim=True) + eps
    )
    return normalized


def qwen35_gated_rmsnorm(
    core: torch.Tensor,
    gate: torch.Tensor,
    weight: torch.Tensor,
    *,
    eps: float = 1e-6,
) -> torch.Tensor:
    _require_floating_tensors(core=core, gate=gate, weight=weight)
    if core.ndim == 0:
        raise ValueError("core must have at least one dimension")
    if gate.shape != core.shape:
        raise ValueError("core and gate shape must exactly match")
    if weight.ndim != 1 or weight.shape[0] != core.shape[-1]:
        raise ValueError(
            "weight must be rank one and match the core last dimension"
        )
    if core.dtype != gate.dtype:
        raise ValueError("core and gate dtype must match")
    if core.device != gate.device or core.device != weight.device:
        raise ValueError("core, gate, and weight device must match")
    if not isinstance(eps, (int, float)) or not math.isfinite(eps) or eps <= 0:
        raise ValueError("eps must be a positive finite number")

    core_fp32 = core.float()
    normalized = core_fp32 * torch.rsqrt(
        core_fp32.pow(2).mean(dim=-1, keepdim=True) + float(eps)
    )
    gated = (
        normalized.to(core.dtype)
        * weight
        * F.silu(gate.float())
    )
    return gated.to(core.dtype)


def qwen35_causal_depthwise_conv(
    projected_qkv: torch.Tensor,
    conv_state: torch.Tensor,
    weight: torch.Tensor,
    *,
    activation: str = "silu",
) -> tuple[torch.Tensor, torch.Tensor]:
    _require_floating_tensors(
        projected_qkv=projected_qkv,
        conv_state=conv_state,
        weight=weight,
    )
    if projected_qkv.ndim != 2:
        raise ValueError("projected_qkv must have [tokens, channels] shape")
    if conv_state.ndim != 2:
        raise ValueError("conv_state must have [channels, kernel_width] shape")
    if projected_qkv.shape[1] != conv_state.shape[0]:
        raise ValueError(
            "projected_qkv channels must match conv_state channels"
        )
    if weight.shape != conv_state.shape:
        raise ValueError("weight must match conv_state [channels, kernel_width]")
    if conv_state.shape[1] < 1:
        raise ValueError("kernel width must be positive")
    if activation != "silu":
        raise ValueError("activation must be 'silu'")

    token_count = projected_qkv.shape[0]
    if token_count == 0:
        return projected_qkv.clone(), conv_state.clone()
    combined = torch.cat(
        (
            conv_state.unsqueeze(0),
            projected_qkv.transpose(0, 1).unsqueeze(0),
        ),
        dim=-1,
    ).to(weight.dtype)
    candidate_state = combined[
        :, :, -conv_state.shape[1]:
    ].squeeze(0)
    output = F.silu(F.conv1d(
        combined,
        weight.unsqueeze(1),
        bias=None,
        padding=0,
        groups=weight.shape[0],
    )[:, :, -token_count:])
    return (
        output.transpose(1, 2).squeeze(0).to(
            projected_qkv.dtype
        ),
        candidate_state.to(conv_state.dtype),
    )


def qwen35_causal_depthwise_conv_prefix_trace(
    projected_qkv: torch.Tensor,
    conv_state: torch.Tensor,
    weight: torch.Tensor,
    *,
    activation: str = "silu",
) -> tuple[torch.Tensor, torch.Tensor]:
    if projected_qkv.shape[0] <= 0:
        raise ValueError(
            "convolution prefix trace requires at least one token"
        )
    state = conv_state.clone()
    outputs = []
    states = []
    for token_index in range(projected_qkv.shape[0]):
        output, state = qwen35_causal_depthwise_conv(
            projected_qkv[token_index:token_index + 1],
            state,
            weight,
            activation=activation,
        )
        outputs.append(output)
        states.append(state)
    return torch.cat(outputs, dim=0), torch.stack(states)


def qwen35_gated_delta_chunk(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    recurrent_state_v_k: torch.Tensor,
    *,
    chunk_size: int = 64,
) -> tuple[torch.Tensor, torch.Tensor]:
    _require_floating_tensors(
        query=query,
        key=key,
        value=value,
        a=a,
        b=b,
        A_log=A_log,
        dt_bias=dt_bias,
        recurrent_state_v_k=recurrent_state_v_k,
    )
    if query.ndim != 3 or key.shape != query.shape:
        raise ValueError(
            "query and key must have matching [tokens, heads, key_dim] shapes"
        )
    if value.ndim != 3 or value.shape[:2] != query.shape[:2]:
        raise ValueError(
            "value must match query token and head dimensions"
        )
    tokens, heads, key_dim = query.shape
    value_dim = value.shape[-1]
    if a.shape != (tokens, heads) or b.shape != (tokens, heads):
        raise ValueError(
            "a and b must have [tokens, heads] shapes matching query"
        )
    if A_log.shape != (heads,) or dt_bias.shape != (heads,):
        raise ValueError(
            "A_log and dt_bias must have [heads] shapes matching query"
        )
    expected_state_shape = (heads, value_dim, key_dim)
    if recurrent_state_v_k.shape != expected_state_shape:
        raise ValueError(
            "physical recurrent state must have "
            f"[heads, value_dim, key_dim] shape {expected_state_shape}"
        )
    if (
        isinstance(chunk_size, bool)
        or not isinstance(chunk_size, int)
        or chunk_size <= 0
    ):
        raise ValueError("chunk_size must be a positive integer")

    initial_dtype = query.dtype
    query = qwen35_l2norm(query)
    key = qwen35_l2norm(key)
    beta = torch.sigmoid(b)
    decay = -torch.exp(A_log.float()) * F.softplus(
        a.float() + dt_bias.float()
    )
    query, key, value, beta, decay = [
        tensor.transpose(0, 1).unsqueeze(0).contiguous().float()
        for tensor in (query, key, value, beta, decay)
    ]
    pad_size = (chunk_size - tokens % chunk_size) % chunk_size
    query = F.pad(query, (0, 0, 0, pad_size))
    key = F.pad(key, (0, 0, 0, pad_size))
    value = F.pad(value, (0, 0, 0, pad_size))
    beta = F.pad(beta, (0, pad_size))
    decay = F.pad(decay, (0, pad_size))
    total_tokens = tokens + pad_size
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
    for token_index in range(1, chunk_size):
        row = attention[
            ..., token_index, :token_index
        ].clone()
        sub = attention[
            ..., :token_index, :token_index
        ].clone()
        attention[..., token_index, :token_index] = (
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
    state = recurrent_state_v_k.float().transpose(
        -1, -2
    ).unsqueeze(0)
    output = torch.zeros_like(value)
    for chunk_index in range(total_tokens // chunk_size):
        query_chunk = query[:, :, chunk_index]
        key_chunk = key[:, :, chunk_index]
        value_chunk = value[:, :, chunk_index]
        intra = (
            query_chunk @ key_chunk.transpose(-1, -2)
        ) * decay_mask[:, :, chunk_index]
        previous = key_cumulative_decay[:, :, chunk_index] @ state
        value_new = value_chunk - previous
        inter = (
            query_chunk
            * decay[:, :, chunk_index, :, None].exp()
        ) @ state
        output[:, :, chunk_index] = inter + intra @ value_new
        state = (
            state
            * decay[:, :, chunk_index, -1, None, None].exp()
            + (
                key_chunk
                * (
                    decay[:, :, chunk_index, -1, None]
                    - decay[:, :, chunk_index]
                ).exp()[..., None]
            ).transpose(-1, -2)
            @ value_new
        )
    output = output.reshape(
        output.shape[0],
        output.shape[1],
        -1,
        value_dim,
    )[:, :, :tokens]
    return (
        output.transpose(1, 2).squeeze(0).to(initial_dtype),
        state.squeeze(0).transpose(-1, -2).to(
            recurrent_state_v_k.dtype
        ),
    )


def qwen35_gated_delta_recurrent(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    recurrent_state_v_k: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    _require_floating_tensors(
        query=query,
        key=key,
        value=value,
        a=a,
        b=b,
        A_log=A_log,
        dt_bias=dt_bias,
        recurrent_state_v_k=recurrent_state_v_k,
    )
    if query.ndim != 3 or key.ndim != 3 or query.shape != key.shape:
        raise ValueError(
            "query and key must have matching [tokens, heads, key_dim] shapes"
        )
    if value.ndim != 3 or value.shape[:2] != query.shape[:2]:
        raise ValueError(
            "value must match query token and head dimensions"
        )
    tokens, heads, key_dim = query.shape
    value_dim = value.shape[-1]
    if a.shape != (tokens, heads) or b.shape != (tokens, heads):
        raise ValueError(
            "a and b must have [tokens, heads] shapes matching query"
        )
    if A_log.shape != (heads,) or dt_bias.shape != (heads,):
        raise ValueError(
            "A_log and dt_bias must have [heads] shapes matching query"
        )
    expected_state_shape = (heads, value_dim, key_dim)
    if recurrent_state_v_k.shape != expected_state_shape:
        raise ValueError(
            "physical recurrent state must have "
            f"[heads, value_dim, key_dim] shape {expected_state_shape}"
        )

    query_fp32 = qwen35_l2norm(query).float() / math.sqrt(key_dim)
    key_fp32 = qwen35_l2norm(key).float()
    value_fp32 = value.float()
    a_fp32 = a.float()
    b_fp32 = b.float()
    A_log_fp32 = A_log.float()
    dt_bias_fp32 = dt_bias.float()
    state_k_v = (
        recurrent_state_v_k.float()
        .transpose(-1, -2)
        .contiguous()
    )
    outputs = []

    for token_index in range(tokens):
        beta = torch.sigmoid(b[token_index]).float()
        log_decay = -torch.exp(A_log_fp32) * F.softplus(
            a_fp32[token_index] + dt_bias_fp32
        )
        state_k_v = state_k_v * torch.exp(log_decay)[:, None, None]
        memory = (
            state_k_v
            * key_fp32[token_index].unsqueeze(-1)
        ).sum(dim=-2)
        delta = (value_fp32[token_index] - memory) * beta[:, None]
        state_k_v = (
            state_k_v
            + key_fp32[token_index].unsqueeze(-1)
            * delta.unsqueeze(-2)
        )
        outputs.append(
            (
                state_k_v
                * query_fp32[token_index].unsqueeze(-1)
            ).sum(dim=-2)
        )

    if outputs:
        output = torch.stack(outputs).to(query.dtype)
    else:
        output = torch.empty(
            (0, heads, value_dim),
            dtype=query.dtype,
            device=query.device,
        )
    physical_state = state_k_v.transpose(-1, -2).to(
        recurrent_state_v_k.dtype
    )
    return output, physical_state


def qwen35_gated_delta_prefix_trace(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    recurrent_state_v_k: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if query.shape[0] <= 0:
        raise ValueError(
            "gated-delta prefix trace requires at least one token"
        )
    state = recurrent_state_v_k.clone()
    outputs = []
    states = []
    for token_index in range(query.shape[0]):
        output, state = qwen35_gated_delta_recurrent(
            query[token_index:token_index + 1],
            key[token_index:token_index + 1],
            value[token_index:token_index + 1],
            a[token_index:token_index + 1],
            b[token_index:token_index + 1],
            A_log,
            dt_bias,
            state,
        )
        outputs.append(output)
        states.append(state)
    return torch.cat(outputs, dim=0), torch.stack(states)
