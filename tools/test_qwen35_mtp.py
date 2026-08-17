from pathlib import Path
import sys
import types

import torch
from torch import nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
for package_name in (
    "tinyvllm",
    "tinyvllm.layers",
    "tinyvllm.models",
):
    package = types.ModuleType(package_name)
    package.__path__ = [str(ROOT / package_name.replace(".", "/"))]
    sys.modules.setdefault(package_name, package)

from tinyvllm.layers.qwen35_decoder_layer import Qwen35DecoderLayerShell
from tinyvllm.layers.qwen35_packed_full_decoder_layer import (
    Qwen35PackedFullDecoderLayer,
)
from tinyvllm.models import qwen35_components as components_module
from tinyvllm.models.qwen35_components import (
    build_qwen35_full_attention_decoder_layer,
)
from tinyvllm.models.qwen35_mtp import (
    Qwen35NativeMTP,
    build_qwen35_native_mtp,
)


def _offset_rmsnorm_reference(
    tensor: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    tensor_fp32 = tensor.float()
    normalized = tensor_fp32 * torch.rsqrt(
        tensor_fp32.pow(2).mean(dim=-1, keepdim=True) + eps
    )
    return (normalized * (1.0 + weight.float())).to(tensor.dtype)


class _OffsetNorm(nn.Module):

    def __init__(self, width: int, weight, eps: float):
        super().__init__()
        self.weight = nn.Parameter(
            torch.tensor(weight, dtype=torch.float32).reshape(width)
        )
        self.eps = eps

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        return _offset_rmsnorm_reference(
            tensor,
            self.weight,
            self.eps,
        )


class _TokenEmbedding(nn.Module):

    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(
            [
                [0.25, -0.50, 1.00],
                [1.25, 0.75, -0.25],
                [-1.00, 0.50, 0.25],
                [0.75, -1.25, 0.50],
                [1.50, 0.25, -0.75],
            ],
            dtype=torch.float32,
        ))

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return F.embedding(input_ids, self.weight)


class _Linear(nn.Module):

    def __init__(self, weight):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(
            weight,
            dtype=torch.float32,
        ))

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        return F.linear(tensor, self.weight)


class _PositionMixer(nn.Module):

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        position_row = positions[0] if positions.ndim == 2 else positions
        return hidden_states + position_row.to(
            hidden_states.dtype
        ).unsqueeze(-1) * 0.125


class _Scale(nn.Module):

    def __init__(self, value: float):
        super().__init__()
        self.value = value

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor * self.value


def _mtp_fixture() -> Qwen35NativeMTP:
    embedding = _TokenEmbedding()
    decoder = Qwen35DecoderLayerShell(
        block_type="full_attention",
        input_layernorm=_Scale(0.5),
        post_attention_layernorm=_Scale(0.25),
        mlp=_Scale(-0.2),
        full_attention=_PositionMixer(),
    )
    return Qwen35NativeMTP(
        embed_tokens=embedding,
        lm_head=_Linear([
            [0.5, -0.25, 1.0],
            [-0.5, 0.75, 0.25],
            [1.0, 0.5, -0.75],
            [0.25, 0.25, 0.25],
            [-1.0, 0.5, 0.5],
        ]),
        fc=_Linear([
            [0.50, -0.25, 0.75, 1.00, 0.25, -0.50],
            [-0.50, 0.50, 0.25, -0.75, 1.00, 0.50],
            [0.25, 0.75, -0.50, 0.50, -0.25, 1.00],
        ]),
        layer=Qwen35PackedFullDecoderLayer(decoder),
        norm=_OffsetNorm(3, [0.10, -0.20, 0.30], 1e-5),
        pre_fc_norm_embedding=_OffsetNorm(
            3,
            [-0.10, 0.20, 0.05],
            1e-5,
        ),
        pre_fc_norm_hidden=_OffsetNorm(
            3,
            [0.15, -0.05, 0.25],
            1e-5,
        ),
    )


def _reference_mtp(
    module: Qwen35NativeMTP,
    input_ids: torch.Tensor,
    positions: torch.Tensor,
    target_hidden: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    embedded = module.embed_tokens(input_ids)
    embedded = _offset_rmsnorm_reference(
        embedded,
        module.pre_fc_norm_embedding.weight,
        module.pre_fc_norm_embedding.eps,
    )
    hidden = _offset_rmsnorm_reference(
        target_hidden,
        module.pre_fc_norm_hidden.weight,
        module.pre_fc_norm_hidden.eps,
    )
    fused = F.linear(
        torch.cat((embedded, hidden), dim=-1),
        module.fc.weight,
    )
    decoded = module.layer((len(input_ids),), positions, fused)
    normalized = _offset_rmsnorm_reference(
        decoded,
        module.norm.weight,
        module.norm.eps,
    )
    return normalized, F.linear(normalized, module.lm_head.weight)


def _expect_error(function, message: str) -> None:
    try:
        function()
    except (TypeError, ValueError) as error:
        assert message in str(error), str(error)
    else:
        raise AssertionError(f"expected error containing {message!r}")


def test_forward_step_matches_independent_math_oracle() -> None:
    module = _mtp_fixture()
    input_ids = torch.tensor([1, 3], dtype=torch.int64)
    positions = torch.tensor([4, 5], dtype=torch.int64)
    hidden_states = torch.tensor(
        [[0.5, 1.25, -0.75], [1.5, -0.25, 0.75]],
        dtype=torch.float32,
    )
    expected_hidden, expected_logits = _reference_mtp(
        module,
        input_ids,
        positions,
        hidden_states,
    )
    actual_hidden, actual_logits = module.forward_step(
        input_ids,
        positions,
        hidden_states,
    )
    torch.testing.assert_close(actual_hidden, expected_hidden)
    torch.testing.assert_close(actual_logits, expected_logits)


def test_forward_hidden_matches_oracle_without_materializing_logits() -> None:
    module = _mtp_fixture()
    input_ids = torch.tensor([1, 3], dtype=torch.int64)
    positions = torch.tensor([4, 5], dtype=torch.int64)
    hidden_states = torch.tensor(
        [[0.5, 1.25, -0.75], [1.5, -0.25, 0.75]],
        dtype=torch.float32,
    )
    expected_hidden, _ = _reference_mtp(
        module,
        input_ids,
        positions,
        hidden_states,
    )

    def reject_logits(_):
        raise AssertionError("hidden-only forward must not call lm_head")

    module.lm_head.forward = reject_logits
    actual_hidden = module.forward_hidden(
        input_ids,
        positions,
        hidden_states,
    )

    torch.testing.assert_close(actual_hidden, expected_hidden)


def test_forward_step_uses_embedding_then_hidden_concatenation() -> None:
    module = _mtp_fixture()
    input_ids = torch.tensor([2], dtype=torch.int64)
    positions = torch.tensor([0], dtype=torch.int64)
    hidden_states = torch.tensor(
        [[0.75, -1.0, 0.25]],
        dtype=torch.float32,
    )
    observed = []
    original_forward = module.fc.forward

    def record_fc(tensor):
        observed.append(tensor.clone())
        return original_forward(tensor)

    module.fc.forward = record_fc
    module.forward_step(input_ids, positions, hidden_states)
    expected_embedding = module.pre_fc_norm_embedding(
        module.embed_tokens(input_ids)
    )
    expected_hidden = module.pre_fc_norm_hidden(hidden_states)
    torch.testing.assert_close(
        observed[0],
        torch.cat((expected_embedding, expected_hidden), dim=-1),
    )
    assert module.fc.weight.shape == (3, 6)


def test_forward_step_rejects_non_exact_input_shapes() -> None:
    module = _mtp_fixture()
    valid_ids = torch.tensor([1, 2], dtype=torch.int64)
    valid_positions = torch.tensor([0, 1], dtype=torch.int64)
    valid_hidden = torch.ones(2, 3)
    cases = (
        (
            lambda: module.forward_step(
                valid_ids.reshape(1, 2),
                valid_positions,
                valid_hidden,
            ),
            "input_ids must be rank one",
        ),
        (
            lambda: module.forward_step(
                valid_ids.to(torch.float32),
                valid_positions,
                valid_hidden,
            ),
            "integer dtype",
        ),
        (
            lambda: module.forward_step(
                valid_ids,
                valid_positions[:1],
                valid_hidden,
            ),
            "token count",
        ),
        (
            lambda: module.forward_step(
                valid_ids,
                valid_positions,
                valid_hidden[:1],
            ),
            "token count",
        ),
        (
            lambda: module.forward_step(
                valid_ids,
                valid_positions,
                torch.ones(2, 4),
            ),
            "hidden size",
        ),
    )
    for function, message in cases:
        _expect_error(function, message)


def _config(**overrides):
    values = {
        "dtype": "bfloat16",
        "hidden_size": 8,
        "intermediate_size": 12,
        "num_attention_heads": 2,
        "num_key_value_heads": 2,
        "head_dim": 8,
        "rms_norm_eps": 1e-6,
        "hidden_act": "silu",
        "mtp_num_hidden_layers": 1,
        "mtp_use_dedicated_embeddings": False,
        "tie_word_embeddings": True,
        "rope_parameters": {
            "rope_theta": 1_000_000,
            "partial_rotary_factor": 0.75,
            "mrope_section": (1, 1, 1),
        },
    }
    values.update(overrides)
    return types.SimpleNamespace(**values)


class _Backend(nn.Module):
    pass


class _SharedEmbedding(nn.Module):

    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(16, 8, device="meta"))

    def forward(self, input_ids):
        return F.embedding(input_ids, self.weight)


class _SharedHead(nn.Module):

    def __init__(self, weight):
        super().__init__()
        self.weight = weight

    def forward(self, hidden_states):
        return F.linear(hidden_states, self.weight)


def test_factory_shares_target_modules_at_tp1() -> None:
    embedding = _SharedEmbedding()
    head = _SharedHead(embedding.weight)
    calls = []

    def build_backend(layer_index, local_query_heads, local_kv_heads, head_dim):
        calls.append((
            layer_index,
            local_query_heads,
            local_kv_heads,
            head_dim,
        ))
        return _Backend()

    module = build_qwen35_native_mtp(
        _config(),
        embed_tokens=embedding,
        lm_head=head,
        tensor_parallel_size=1,
        tensor_parallel_rank=0,
        build_attention_backend=build_backend,
        parameter_device="meta",
    )
    assert module.embed_tokens is embedding
    assert module.lm_head is head
    assert module.fc.weight.shape == (8, 16)
    assert module.fc.weight.dtype == torch.bfloat16
    assert calls == [(0, 2, 2, 8)]


def test_factory_uses_requested_tp4_rank_context() -> None:
    for rank in range(4):
        embedding = _SharedEmbedding()
        head = _SharedHead(embedding.weight)
        calls = []

        def build_backend(
            layer_index,
            local_query_heads,
            local_kv_heads,
            head_dim,
        ):
            calls.append((
                layer_index,
                local_query_heads,
                local_kv_heads,
                head_dim,
                torch.distributed.get_world_size(),
                torch.distributed.get_rank(),
            ))
            return _Backend()

        module = build_qwen35_native_mtp(
            _config(
                num_attention_heads=4,
                num_key_value_heads=2,
            ),
            embed_tokens=embedding,
            lm_head=head,
            tensor_parallel_size=4,
            tensor_parallel_rank=rank,
            build_attention_backend=build_backend,
            parameter_device="meta",
        )

        assert module.embed_tokens is embedding
        assert module.lm_head is head
        assert module.fc.weight.shape == (8, 16)
        assert calls == [(0, 1, 1, 8, 4, rank)]


def test_factory_rejects_invalid_tensor_parallel_topology() -> None:
    embedding = _SharedEmbedding()
    head = _SharedHead(embedding.weight)

    def build_backend(*_):
        return _Backend()

    for world_size, rank, message in (
        (0, 0, "positive integer"),
        (True, 0, "positive integer"),
        (1, -1, "tensor_parallel_rank"),
        (1, 1, "tensor_parallel_rank"),
        (4, 4, "tensor_parallel_rank"),
        (4, True, "tensor_parallel_rank"),
    ):
        _expect_error(
            lambda world_size=world_size, rank=rank: (
                build_qwen35_native_mtp(
                    _config(),
                    embed_tokens=embedding,
                    lm_head=head,
                    tensor_parallel_size=world_size,
                    tensor_parallel_rank=rank,
                    build_attention_backend=build_backend,
                    parameter_device="meta",
                )
            ),
            message,
        )

    _expect_error(
        lambda: build_qwen35_native_mtp(
            _config(num_attention_heads=2),
            embed_tokens=embedding,
            lm_head=head,
            tensor_parallel_size=4,
            tensor_parallel_rank=0,
            build_attention_backend=build_backend,
            parameter_device="meta",
        ),
        "divisible",
    )


def test_target_and_mtp_use_the_same_public_full_attention_builder() -> None:
    assert (
        components_module.build_qwen35_full_attention_decoder_layer
        is build_qwen35_full_attention_decoder_layer
    )
    source = Path(components_module.__file__).read_text()
    assert "build_qwen35_full_attention_decoder_layer(" in source
    mtp_source = (ROOT / "tinyvllm/models/qwen35_mtp.py").read_text()
    assert "build_qwen35_full_attention_decoder_layer(" in mtp_source


def main():
    test_forward_step_matches_independent_math_oracle()
    test_forward_step_uses_embedding_then_hidden_concatenation()
    test_forward_step_rejects_non_exact_input_shapes()
    test_factory_shares_target_modules_at_tp1()
    test_factory_uses_requested_tp4_rank_context()
    test_factory_rejects_invalid_tensor_parallel_topology()
    test_target_and_mtp_use_the_same_public_full_attention_builder()
    print("qwen35 native MTP tests passed")


if __name__ == "__main__":
    main()
