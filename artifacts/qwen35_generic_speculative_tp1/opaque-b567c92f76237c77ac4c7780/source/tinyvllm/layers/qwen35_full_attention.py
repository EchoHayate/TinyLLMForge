import torch
from torch import nn

from tinyvllm.layers.qwen35_primitives import qwen35_apply_query_gate
from tinyvllm.utils.context import get_context


def qwen35_prefill_eager_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    context,
    *,
    num_heads: int,
    head_dim: int,
    scale: float,
    key_cache: torch.Tensor | None = None,
    value_cache: torch.Tensor | None = None,
) -> torch.Tensor:
    if (
        isinstance(key_cache, torch.Tensor)
        and isinstance(value_cache, torch.Tensor)
        and key_cache.numel() > 0
        and value_cache.numel() > 0
    ):
        block_size = key_cache.shape[1]
        slots = context.slot_mapping.to(torch.long)
        blocks = torch.div(slots, block_size, rounding_mode="floor")
        offsets = slots.remainder(block_size)
        key_cache[blocks, offsets] = key
        value_cache[blocks, offsets] = value
    cu_seqlens_q = getattr(context, "cu_seqlens_q", None)
    if isinstance(cu_seqlens_q, torch.Tensor):
        offsets = cu_seqlens_q.tolist()
    else:
        offsets = [0, query.shape[0]]
    reference_lens = getattr(
        context,
        "prefill_attention_reference_lens",
        None,
    )
    outputs = []
    for index in range(len(offsets) - 1):
        start = int(offsets[index])
        end = int(offsets[index + 1])
        segment_query = query[start:end]
        segment_key = key[start:end]
        segment_value = value[start:end]
        token_count = end - start
        reference_count = (
            int(reference_lens[index])
            if reference_lens is not None
            else token_count
        )
        if reference_count < token_count:
            raise ValueError(
                "prefill attention reference length must not be smaller "
                "than the query length"
            )
        if reference_count > token_count:
            padding = reference_count - token_count
            segment_query = torch.cat(
                (
                    segment_query,
                    segment_query.new_zeros(
                        padding,
                        *segment_query.shape[1:],
                    ),
                ),
                dim=0,
            )
            segment_key = torch.cat(
                (
                    segment_key,
                    segment_key.new_zeros(
                        padding,
                        *segment_key.shape[1:],
                    ),
                ),
                dim=0,
            )
            segment_value = torch.cat(
                (
                    segment_value,
                    segment_value.new_zeros(
                        padding,
                        *segment_value.shape[1:],
                    ),
                ),
                dim=0,
            )
        repeats = num_heads // segment_key.shape[1]
        segment_key = segment_key.repeat_interleave(repeats, dim=1)
        segment_value = segment_value.repeat_interleave(repeats, dim=1)
        row_query = segment_query.transpose(0, 1).unsqueeze(0)
        row_key = segment_key.transpose(0, 1).unsqueeze(0)
        row_value = segment_value.transpose(0, 1).unsqueeze(0)
        scores = torch.matmul(
            row_query,
            row_key.transpose(2, 3),
        ) * scale
        positions = torch.arange(reference_count, device=query.device)
        causal = positions.unsqueeze(0) > positions.unsqueeze(1)
        mask = torch.zeros_like(scores).masked_fill(
            causal.view(1, 1, reference_count, reference_count),
            torch.finfo(scores.dtype).min,
        )
        probabilities = torch.softmax(
            scores + mask,
            dim=-1,
            dtype=torch.float32,
        ).to(query.dtype)
        outputs.append(
            torch.matmul(probabilities, row_value)
            .transpose(1, 2)[:, :token_count]
            .reshape(token_count, num_heads * head_dim)
        )
    return torch.cat(outputs, dim=0)


def qwen35_cached_prefill_eager_attention(
    query: torch.Tensor,
    current_key: torch.Tensor,
    current_value: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    context,
    *,
    num_heads: int,
    head_dim: int,
    scale: float,
) -> torch.Tensor:
    block_size = key_cache.shape[1]
    slots = context.slot_mapping.to(torch.long)
    blocks = torch.div(slots, block_size, rounding_mode="floor")
    offsets = slots.remainder(block_size)
    key_cache[blocks, offsets] = current_key
    value_cache[blocks, offsets] = current_value
    q_offsets = context.cu_seqlens_q.tolist()
    k_offsets = context.cu_seqlens_k.tolist()
    block_tables = context.block_tables
    outputs = []
    for index in range(len(q_offsets) - 1):
        q_start = int(q_offsets[index])
        q_end = int(q_offsets[index + 1])
        q_length = q_end - q_start
        k_length = int(k_offsets[index + 1]) - int(k_offsets[index])
        block_count = (k_length + block_size - 1) // block_size
        block_ids = block_tables[index, :block_count].to(torch.long)
        key = key_cache[block_ids].reshape(
            -1,
            key_cache.shape[2],
            key_cache.shape[3],
        )[:k_length]
        value = value_cache[block_ids].reshape(
            -1,
            value_cache.shape[2],
            value_cache.shape[3],
        )[:k_length]
        repeats = num_heads // key.shape[1]
        key = key.repeat_interleave(repeats, dim=1)
        value = value.repeat_interleave(repeats, dim=1)
        row_query = query[q_start:q_end]
        if k_length > q_length:
            row_query = torch.cat(
                (
                    row_query.new_zeros(
                        k_length - q_length,
                        *row_query.shape[1:],
                    ),
                    row_query,
                ),
                dim=0,
            )
        row_query = row_query.transpose(0, 1).unsqueeze(0)
        key = key.transpose(0, 1).unsqueeze(0)
        value = value.transpose(0, 1).unsqueeze(0)
        scores = torch.matmul(row_query, key.transpose(2, 3)) * scale
        query_positions = torch.arange(k_length, device=query.device)
        key_positions = query_positions
        causal_mask = (
            key_positions.unsqueeze(0) > query_positions.unsqueeze(1)
        )
        scores = scores.masked_fill(
            causal_mask.view(1, 1, k_length, k_length),
            float("-inf"),
        )
        probabilities = torch.softmax(
            scores,
            dim=-1,
            dtype=torch.float32,
        ).to(query.dtype)
        outputs.append(
            torch.matmul(probabilities, value)
            .transpose(1, 2)[:, -q_length:]
            .reshape(q_length, num_heads * head_dim)
        )
    return torch.cat(outputs, dim=0)


def qwen35_cached_decode_eager_attention(
    query: torch.Tensor,
    current_key: torch.Tensor,
    current_value: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    context,
    *,
    num_heads: int,
    head_dim: int,
    scale: float,
) -> torch.Tensor:
    block_size = key_cache.shape[1]
    slots = context.slot_mapping.to(torch.long)
    blocks = torch.div(slots, block_size, rounding_mode="floor")
    offsets = slots.remainder(block_size)
    key_cache[blocks, offsets] = current_key
    value_cache[blocks, offsets] = current_value
    outputs = []
    tensor_parallel_size = (
        torch.distributed.get_world_size()
        if torch.distributed.is_initialized()
        else 1
    )
    tensor_parallel_rank = (
        torch.distributed.get_rank()
        if tensor_parallel_size > 1
        else 0
    )
    for index, context_length_value in enumerate(context.context_lens):
        context_length = int(context_length_value.item())
        block_count = (context_length + block_size - 1) // block_size
        block_ids = context.block_tables[
            index, :block_count
        ].to(torch.long)
        key = key_cache[block_ids].reshape(
            -1,
            key_cache.shape[2],
            key_cache.shape[3],
        )[:context_length]
        value = value_cache[block_ids].reshape(
            -1,
            value_cache.shape[2],
            value_cache.shape[3],
        )[:context_length]
        repeats = num_heads // key.shape[1]
        key = key.repeat_interleave(repeats, dim=1)
        value = value.repeat_interleave(repeats, dim=1)
        row_query = query[index:index + 1].transpose(0, 1).unsqueeze(0)
        key = key.transpose(0, 1).unsqueeze(0)
        value = value.transpose(0, 1).unsqueeze(0)
        scores = torch.matmul(
            row_query,
            key.transpose(2, 3),
        ) * scale
        if tensor_parallel_size > 1:
            head_start = tensor_parallel_rank * num_heads
            scores = scores.repeat(
                1,
                tensor_parallel_size,
                1,
                1,
            )
        probabilities = torch.softmax(
            scores,
            dim=-1,
            dtype=torch.float32,
        ).to(query.dtype)
        if tensor_parallel_size > 1:
            global_value = value.repeat(
                1,
                tensor_parallel_size,
                1,
                1,
            )
            attention_output = torch.matmul(
                probabilities,
                global_value,
            )[:, head_start:head_start + num_heads]
        else:
            attention_output = torch.matmul(probabilities, value)
        outputs.append(
            attention_output.transpose(1, 2)
            .reshape(1, num_heads * head_dim)
        )
    return torch.cat(outputs, dim=0)


def qwen35_cached_decode_graph_attention(
    query: torch.Tensor,
    current_key: torch.Tensor,
    current_value: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    context,
    *,
    num_heads: int,
    head_dim: int,
    scale: float,
) -> torch.Tensor:
    if key_cache.shape[1] != 1 or value_cache.shape[1] != 1:
        raise ValueError(
            "graph decode requires block-size-one K/V cache"
        )
    slots = context.slot_mapping.to(torch.long)
    key_cache[slots, 0] = current_key
    value_cache[slots, 0] = current_value
    block_ids = context.block_tables.to(torch.long)
    key = key_cache[block_ids, 0]
    value = value_cache[block_ids, 0]
    repeats = num_heads // key.shape[2]
    key = key.repeat_interleave(repeats, dim=2)
    value = value.repeat_interleave(repeats, dim=2)
    row_query = query.unsqueeze(2)
    key = key.transpose(1, 2)
    value = value.transpose(1, 2)
    scores = torch.matmul(
        row_query,
        key.transpose(2, 3),
    ) * scale
    positions = torch.arange(
        block_ids.shape[1],
        device=query.device,
    )
    valid = positions.unsqueeze(0) < context.context_lens.unsqueeze(1)
    scores = scores.masked_fill(
        ~valid.unsqueeze(1).unsqueeze(1),
        torch.finfo(scores.dtype).min,
    )
    probabilities = torch.softmax(
        scores,
        dim=-1,
        dtype=torch.float32,
    ).to(query.dtype)
    return torch.matmul(
        probabilities,
        value,
    ).transpose(1, 2).reshape(
        query.shape[0],
        num_heads * head_dim,
    )


class Qwen35FullAttentionShell(nn.Module):

    def __init__(
        self,
        *,
        head_dim: int,
        local_query_heads: int,
        local_kv_heads: int,
        q_projection: nn.Module,
        k_projection: nn.Module,
        v_projection: nn.Module,
        q_norm: nn.Module,
        k_norm: nn.Module,
        rotary: nn.Module,
        attention_backend: nn.Module,
        output_projection: nn.Module,
    ):
        super().__init__()
        for name, value in (
            ("head_dim", head_dim),
            ("local_query_heads", local_query_heads),
            ("local_kv_heads", local_kv_heads),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value <= 0
            ):
                raise ValueError(f"{name} must be a positive integer")

        self.head_dim = head_dim
        self.local_query_heads = local_query_heads
        self.local_kv_heads = local_kv_heads
        self.q_projection = q_projection
        self.k_projection = k_projection
        self.v_projection = v_projection
        self.q_norm = q_norm
        self.k_norm = k_norm
        self.rotary = rotary
        self.attention_backend = attention_backend
        self.output_projection = output_projection

    @staticmethod
    def _validate_tensor(
        tensor: torch.Tensor,
        *,
        name: str,
        token_count: int,
        feature_count: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        if not isinstance(tensor, torch.Tensor):
            raise ValueError(f"{name} output must be a tensor")
        if tensor.ndim != 2:
            raise ValueError(f"{name} output must be rank two")
        if not tensor.is_floating_point():
            raise ValueError(
                f"{name} output must use a floating point dtype"
            )
        if tensor.shape[0] != token_count:
            raise ValueError(
                f"{name} token count must equal {token_count}"
            )
        if tensor.shape[1] != feature_count:
            raise ValueError(
                f"{name} feature dimension must equal {feature_count}"
            )
        if tensor.dtype != dtype:
            raise ValueError(f"{name} dtype must match hidden_states")
        if tensor.device != device:
            raise ValueError(f"{name} device must match hidden_states")

    @staticmethod
    def _validate_transform(
        tensor: torch.Tensor,
        reference: torch.Tensor,
        name: str,
    ) -> None:
        if not isinstance(tensor, torch.Tensor):
            raise ValueError(f"{name} output must be a tensor")
        if not tensor.is_floating_point():
            raise ValueError(
                f"{name} output must use a floating point dtype"
            )
        if tensor.shape != reference.shape:
            raise ValueError(f"{name} shape must remain unchanged")
        if tensor.dtype != reference.dtype:
            raise ValueError(f"{name} dtype must remain unchanged")
        if tensor.device != reference.device:
            raise ValueError(f"{name} device must remain unchanged")

    @staticmethod
    def _transform_context_rows(
        transform: nn.Module,
        tensor: torch.Tensor,
    ) -> torch.Tensor:
        context = get_context()
        forward = transform
        prefill_forward = getattr(
            transform,
            "forward_prefill",
            None,
        )
        if context.is_prefill and callable(prefill_forward):
            forward = prefill_forward
        if not context.is_prefill:
            return forward(tensor)
        cu_seqlens_q = getattr(context, "cu_seqlens_q", None)
        cu_seqlens_k = getattr(context, "cu_seqlens_k", None)
        if (
            isinstance(cu_seqlens_q, torch.Tensor)
            and isinstance(cu_seqlens_k, torch.Tensor)
            and cu_seqlens_q.numel() == cu_seqlens_k.numel()
            and cu_seqlens_q.numel() > 1
        ):
            q_offsets = cu_seqlens_q.tolist()
            k_offsets = cu_seqlens_k.tolist()
            outputs = []
            for index in range(len(q_offsets) - 1):
                q_start = int(q_offsets[index])
                q_end = int(q_offsets[index + 1])
                q_length = q_end - q_start
                k_length = int(k_offsets[index + 1]) - int(k_offsets[index])
                segment = tensor[q_start:q_end]
                if k_length > q_length:
                    segment = torch.cat(
                        (
                            segment.new_zeros(
                                k_length - q_length,
                                *segment.shape[1:],
                            ),
                            segment,
                        ),
                        dim=0,
                    )
                outputs.append(forward(segment)[-q_length:])
            return torch.cat(outputs, dim=0)
        target_rows = int(getattr(context, "max_seqlen_k", 0))
        if target_rows <= tensor.shape[0]:
            return forward(tensor)
        padded = torch.cat(
            (
                tensor.new_zeros(
                    target_rows - tensor.shape[0],
                    *tensor.shape[1:],
                ),
                tensor,
            ),
            dim=0,
        )
        return forward(padded)[-tensor.shape[0]:]

    def forward(
        self,
        position_ids: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        if not isinstance(hidden_states, torch.Tensor):
            raise ValueError("hidden_states must be a tensor")
        if hidden_states.ndim != 2:
            raise ValueError("hidden_states must be rank two")
        if not hidden_states.is_floating_point():
            raise ValueError(
                "hidden_states must use a floating point dtype"
            )

        token_count = hidden_states.shape[0]
        query_width = self.local_query_heads * self.head_dim
        kv_width = self.local_kv_heads * self.head_dim

        paired_query_gate = self._transform_context_rows(
            self.q_projection,
            hidden_states,
        )
        key = self._transform_context_rows(
            self.k_projection,
            hidden_states,
        )
        value = self._transform_context_rows(
            self.v_projection,
            hidden_states,
        )

        self._validate_tensor(
            paired_query_gate,
            name="q_projection",
            token_count=token_count,
            feature_count=2 * query_width,
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
        self._validate_tensor(
            key,
            name="k_projection",
            token_count=token_count,
            feature_count=kv_width,
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
        self._validate_tensor(
            value,
            name="v_projection",
            token_count=token_count,
            feature_count=kv_width,
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )

        paired_query_gate = paired_query_gate.view(
            token_count,
            self.local_query_heads,
            2 * self.head_dim,
        )
        query, query_gate = paired_query_gate.chunk(2, dim=-1)
        query_gate = query_gate.reshape(token_count, query_width)
        key = key.view(token_count, self.local_kv_heads, self.head_dim)

        normalized_query = self._transform_context_rows(
            self.q_norm,
            query,
        )
        normalized_key = self._transform_context_rows(
            self.k_norm,
            key,
        )
        self._validate_transform(normalized_query, query, "q_norm")
        self._validate_transform(normalized_key, key, "k_norm")

        rotary_output = self.rotary(
            position_ids,
            normalized_query.reshape(token_count, query_width),
            normalized_key.reshape(token_count, kv_width),
        )
        if (
            not isinstance(rotary_output, tuple)
            or len(rotary_output) != 2
        ):
            raise ValueError("rotary must return a query and key tuple")
        rotated_query, rotated_key = rotary_output
        query_reference = normalized_query.reshape(
            token_count,
            query_width,
        )
        key_reference = normalized_key.reshape(token_count, kv_width)
        self._validate_transform(
            rotated_query,
            query_reference,
            "rotary query",
        )
        self._validate_transform(
            rotated_key,
            key_reference,
            "rotary key",
        )

        context = get_context()
        cached_prefill_eager = (
            context.is_prefill
            and getattr(context, "block_tables", None) is not None
            and isinstance(
                getattr(self.attention_backend, "k_cache", None),
                torch.Tensor,
            )
            and isinstance(
                getattr(self.attention_backend, "v_cache", None),
                torch.Tensor,
            )
            and self.attention_backend.k_cache.numel() > 0
            and self.attention_backend.v_cache.numel() > 0
            and getattr(self.attention_backend, "kv_quant_bits", 0) == 0
        )
        cached_decode_eager = (
            getattr(
                context,
                "mode",
                "prefill" if context.is_prefill else "decode",
            ) == "decode"
            and getattr(context, "block_tables", None) is not None
            and getattr(context, "context_lens", None) is not None
            and getattr(context, "slot_mapping", None) is not None
            and isinstance(
                getattr(self.attention_backend, "k_cache", None),
                torch.Tensor,
            )
            and isinstance(
                getattr(self.attention_backend, "v_cache", None),
                torch.Tensor,
            )
            and self.attention_backend.k_cache.numel() > 0
            and self.attention_backend.v_cache.numel() > 0
            and getattr(self.attention_backend, "kv_quant_bits", 0) == 0
            and getattr(context, "quest_top_k_blocks", -1) <= 0
            and getattr(context, "am_compact_blocks", 0) <= 0
            and not getattr(context, "kv_offload_blockwise_decode", False)
            and not getattr(context, "force_attention_backend", False)
        )
        cached_decode_graph = (
            getattr(
                context,
                "mode",
                "prefill" if context.is_prefill else "decode",
            ) == "decode"
            and getattr(context, "block_tables", None) is not None
            and getattr(context, "context_lens", None) is not None
            and getattr(context, "slot_mapping", None) is not None
            and isinstance(
                getattr(self.attention_backend, "k_cache", None),
                torch.Tensor,
            )
            and isinstance(
                getattr(self.attention_backend, "v_cache", None),
                torch.Tensor,
            )
            and self.attention_backend.k_cache.numel() > 0
            and self.attention_backend.v_cache.numel() > 0
            and self.attention_backend.k_cache.shape[1] == 1
            and self.attention_backend.v_cache.shape[1] == 1
            and getattr(self.attention_backend, "kv_quant_bits", 0) == 0
            and getattr(context, "quest_top_k_blocks", -1) <= 0
            and getattr(context, "am_compact_blocks", 0) <= 0
            and not getattr(context, "kv_offload_blockwise_decode", False)
            and getattr(context, "force_attention_backend", False)
        )
        if cached_prefill_eager:
            attention_output = qwen35_cached_prefill_eager_attention(
                rotated_query.view(
                    token_count,
                    self.local_query_heads,
                    self.head_dim,
                ),
                rotated_key.view(
                    token_count,
                    self.local_kv_heads,
                    self.head_dim,
                ),
                value.view(
                    token_count,
                    self.local_kv_heads,
                    self.head_dim,
                ),
                self.attention_backend.k_cache,
                self.attention_backend.v_cache,
                context,
                num_heads=self.local_query_heads,
                head_dim=self.head_dim,
                scale=self.head_dim ** -0.5,
            )
        elif (
            context.is_prefill
            and getattr(self.attention_backend, "kv_quant_bits", 0) == 0
        ):
            attention_output = qwen35_prefill_eager_attention(
                rotated_query.view(
                    token_count,
                    self.local_query_heads,
                    self.head_dim,
                ),
                rotated_key.view(
                    token_count,
                    self.local_kv_heads,
                    self.head_dim,
                ),
                value.view(
                    token_count,
                    self.local_kv_heads,
                    self.head_dim,
                ),
                context,
                num_heads=self.local_query_heads,
                head_dim=self.head_dim,
                scale=self.head_dim ** -0.5,
                key_cache=getattr(self.attention_backend, "k_cache", None),
                value_cache=getattr(self.attention_backend, "v_cache", None),
            )
        elif cached_decode_eager:
            attention_output = qwen35_cached_decode_eager_attention(
                rotated_query.view(
                    token_count,
                    self.local_query_heads,
                    self.head_dim,
                ),
                rotated_key.view(
                    token_count,
                    self.local_kv_heads,
                    self.head_dim,
                ),
                value.view(
                    token_count,
                    self.local_kv_heads,
                    self.head_dim,
                ),
                self.attention_backend.k_cache,
                self.attention_backend.v_cache,
                context,
                num_heads=self.local_query_heads,
                head_dim=self.head_dim,
                scale=self.head_dim ** -0.5,
            )
        elif cached_decode_graph:
            attention_output = qwen35_cached_decode_graph_attention(
                rotated_query.view(
                    token_count,
                    self.local_query_heads,
                    self.head_dim,
                ),
                rotated_key.view(
                    token_count,
                    self.local_kv_heads,
                    self.head_dim,
                ),
                value.view(
                    token_count,
                    self.local_kv_heads,
                    self.head_dim,
                ),
                self.attention_backend.k_cache,
                self.attention_backend.v_cache,
                context,
                num_heads=self.local_query_heads,
                head_dim=self.head_dim,
                scale=self.head_dim ** -0.5,
            )
        else:
            attention_output = self.attention_backend(
                rotated_query,
                rotated_key,
                value,
            )
        self._validate_tensor(
            attention_output,
            name="attention_backend",
            token_count=token_count,
            feature_count=query_width,
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
        gated_output = qwen35_apply_query_gate(
            attention_output,
            query_gate,
        )
        output = self._transform_context_rows(
            self.output_projection,
            gated_output,
        )
        if not isinstance(output, torch.Tensor):
            raise ValueError("output_projection output must be a tensor")
        if output.ndim != 2:
            raise ValueError("output_projection output must be rank two")
        if not output.is_floating_point():
            raise ValueError(
                "output_projection output must use a floating point dtype"
            )
        if output.shape[0] != token_count:
            raise ValueError(
                f"output_projection token count must equal {token_count}"
            )
        if output.dtype != hidden_states.dtype:
            raise ValueError(
                "output_projection dtype must match hidden_states"
            )
        if output.device != hidden_states.device:
            raise ValueError(
                "output_projection device must match hidden_states"
            )
        return output
