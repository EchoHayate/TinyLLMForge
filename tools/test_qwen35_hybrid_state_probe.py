"""CPU-only unit tests for Qwen3.5 hybrid-state normalization helpers.

Run: python3 tools/test_qwen35_hybrid_state_probe.py
"""

from __future__ import annotations

import importlib.util
import io
import json
import os
import sys
import tempfile
from collections import namedtuple
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import torch


THIS_DIR = Path(__file__).resolve().parent


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, os.fspath(path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


contract = _load_module(
    "qwen35_hybrid_state_contract_for_probe_tests",
    THIS_DIR / "qwen35_hybrid_state_contract.py",
)
probe = _load_module(
    "qwen35_hybrid_state_probe_under_test",
    THIS_DIR / "qwen35_hybrid_state_probe.py",
)


def _expect_value_error(callable_, message_fragment):
    try:
        callable_()
    except ValueError as exc:
        assert message_fragment in str(exc)
    else:
        raise AssertionError("expected ValueError")


def _expect_incomplete(callable_, failure_kind):
    try:
        callable_()
    except probe.IncompleteRun as exc:
        assert exc.failure_kind == failure_kind
    else:
        raise AssertionError("expected IncompleteRun")


@dataclass
class _DataclassState:
    recurrent_state: torch.Tensor
    convolution_state: torch.Tensor


_NamedTupleState = namedtuple(
    "_NamedTupleState",
    ("key_cache", "value_cache"),
)


class _AdapterState:
    def __init__(self, hidden, ignored):
        self.hidden = hidden
        self.ignored = ignored


class _FakeLayer:
    def __init__(self, layer_type):
        self.layer_type = layer_type


class _FakeBackbone:
    def __init__(self, layer_types):
        self.layers = [_FakeLayer(layer_type) for layer_type in layer_types]


class _FakeQwen35Model:
    def __init__(self, layer_types, *, apply_dtype_conversion=True):
        self.model = _FakeBackbone(layer_types)
        self.to_calls = []
        self.apply_dtype_conversion = apply_dtype_conversion
        self._parameters = {
            "weight": torch.zeros((2, 2), dtype=torch.float16),
            "state_scale": torch.zeros((1,), dtype=torch.float32),
        }

    def named_parameters(self):
        yield from self._parameters.items()

    def to(self, device=None, dtype=None):
        self.to_calls.append({
            "device": device,
            "dtype": dtype,
        })
        if dtype is not None and self.apply_dtype_conversion:
            self._parameters = {
                name: parameter.to(dtype=dtype)
                for name, parameter in self._parameters.items()
            }
        return self


class _FakeQwen35Config:
    def __init__(self, layer_types):
        self.num_hidden_layers = 24
        self.vocab_size = 152064
        self.layer_types = list(layer_types)
        self.full_attention_interval = 4
        self.linear_num_key_heads = 16
        self.linear_num_value_heads = 16
        self.linear_key_head_dim = 128
        self.linear_value_head_dim = 128
        self.linear_conv_kernel_dim = 4
        self.mamba_ssm_dtype = "float32"


class _FakeQwen35MultimodalConfig:
    def __init__(self, layer_types):
        self.text_config = _FakeQwen35Config(layer_types)


class _FakeTokenizer:
    vocab_size = 151936


class _FakeReferenceStateAdapter:
    vocab_size = 32

    def _state(self, tokens):
        values = [int(value) for value in tokens]
        return {
            "tokens": values,
            "recurrent_state": torch.tensor(
                values[-4:] or [0],
                dtype=torch.float32,
            ),
        }

    def _logits(self, tokens):
        next_token = (sum(tokens) + len(tokens)) % self.vocab_size
        logits = torch.arange(self.vocab_size, dtype=torch.float32).neg()
        logits[next_token] = 1000.0 + len(tokens)
        return logits

    def prefill(self, input_ids, state):
        prefix = [] if state is None else list(state["tokens"])
        tokens = prefix + [int(value) for value in input_ids.flatten()]
        return self._logits(tokens), self._state(tokens)

    def decode_one(self, token_id, state, sequence_length):
        assert sequence_length == len(state["tokens"])
        tokens = [*state["tokens"], int(token_id)]
        return self._logits(tokens), self._state(tokens)

    def one_shot(self, token_ids):
        return self._logits([int(value) for value in token_ids])

    def export_state(
        self,
        state,
        request_id,
        request_generation,
        sequence_length,
    ):
        assert sequence_length == len(state["tokens"])
        return {
            "request_id": request_id,
            "request_generation": request_generation,
            "tokens": list(state["tokens"]),
        }

    def import_state(self, exported):
        return self._state(exported["tokens"])

    def state_sha256(self, state):
        return contract.canonical_json_sha256({
            "tokens": state["tokens"],
            "recurrent_state": state["recurrent_state"].tolist(),
        })


class _FakeCache:
    def __init__(self, tokens):
        self.tokens = tokens

    def to_legacy_cache(self):
        return (self.tokens,)

    @classmethod
    def from_legacy_cache(cls, payload):
        return cls(payload[0])


class _OpaqueCache:
    def __init__(self, tokens):
        self.tokens = tokens


class _FakeHybridLinearLayer:
    def __init__(self):
        self.conv_states = torch.zeros((1, 1, 4), dtype=torch.float32)
        self.recurrent_states = torch.zeros((1, 1, 2, 2), dtype=torch.float32)


class _FakeHybridAttentionLayer:
    def __init__(self):
        self.keys = torch.zeros((1, 1, 0, 1), dtype=torch.float32)
        self.values = torch.zeros((1, 1, 0, 1), dtype=torch.float32)


class _FakeHybridCache:
    def __init__(self, config=None):
        layer_types = (
            ["linear_attention", "full_attention"]
            if config is None
            else config.layer_types
        )
        self.layers = [
            (
                _FakeHybridLinearLayer()
                if layer_type == "linear_attention"
                else _FakeHybridAttentionLayer()
            )
            for layer_type in layer_types
        ]

    def update(self, keys, values, layer_idx):
        layer = self.layers[layer_idx]
        layer.keys = torch.cat((layer.keys, keys), dim=-2)
        layer.values = torch.cat((layer.values, values), dim=-2)
        return layer.keys, layer.values

    def update_conv_state(self, conv_states, layer_idx):
        self.layers[layer_idx].conv_states.copy_(conv_states)
        return self.layers[layer_idx].conv_states

    def update_recurrent_state(self, recurrent_states, layer_idx):
        self.layers[layer_idx].recurrent_states.copy_(recurrent_states)
        return self.layers[layer_idx].recurrent_states

    def get_seq_length(self):
        return int(self.layers[1].keys.shape[-2])


class _FakeHybridCausalModel:
    def __init__(self):
        self.device = torch.device("cpu")
        self.config = SimpleNamespace(
            layer_types=["linear_attention", "full_attention"],
        )

    def __call__(
        self,
        *,
        input_ids,
        past_key_values=None,
        use_cache,
        return_dict,
        cache_position=None,
    ):
        assert use_cache is True
        assert return_dict is True
        cache = past_key_values or _FakeHybridCache(self.config)
        values = input_ids.to(dtype=torch.float32).reshape(1, 1, -1, 1)
        cache.update(values, values * 2, 1)
        tokens = cache.layers[1].keys.flatten()
        conv = torch.zeros((1, 1, 4), dtype=torch.float32)
        tail = tokens[-4:]
        conv[..., -tail.numel():] = tail
        recurrent = torch.full(
            (1, 1, 2, 2),
            float(tokens.sum()),
            dtype=torch.float32,
        )
        cache.update_conv_state(conv, 0)
        cache.update_recurrent_state(recurrent, 0)
        logits = torch.arange(32, dtype=torch.float32).neg()
        logits[int(tokens.sum()) % 32] = 1000.0
        return SimpleNamespace(
            logits=logits.reshape(1, 1, -1),
            past_key_values=cache,
        )


class _FakeCausalModel:
    def __init__(self, cache_type=_FakeCache):
        self.device = torch.device("cpu")
        self.cache_type = cache_type
        self.config = SimpleNamespace()
        self.calls = []

    def __call__(
        self,
        *,
        input_ids,
        past_key_values=None,
        use_cache,
        return_dict,
        cache_position=None,
    ):
        assert use_cache is True
        assert return_dict is True
        prefix = (
            []
            if past_key_values is None
            else past_key_values.tokens.flatten().tolist()
        )
        incoming = input_ids.flatten().tolist()
        tokens = torch.tensor(prefix + incoming, dtype=torch.long)
        vocab_size = 32
        next_token = (int(tokens.sum()) + tokens.numel()) % vocab_size
        logits = torch.arange(vocab_size, dtype=torch.float32).neg()
        logits[next_token] = 1000.0 + tokens.numel()
        logits = logits.reshape(1, 1, -1).repeat(
            1, input_ids.shape[-1], 1
        )
        self.calls.append({
            "input_ids": incoming,
            "cache_position": (
                None
                if cache_position is None
                else cache_position.flatten().tolist()
            ),
        })
        return SimpleNamespace(
            logits=logits,
            past_key_values=self.cache_type(tokens),
        )


class _FakeAutoClass:
    calls = []
    result = None

    @classmethod
    def from_pretrained(cls, model_dir, **kwargs):
        cls.calls.append((os.fspath(model_dir), kwargs))
        return cls.result


def test_load_official_reference_accepts_frozen_requested_dtype():
    layer_types = _canonical_layer_types()
    config = _FakeQwen35Config(layer_types)
    tokenizer = _FakeTokenizer()
    model = _FakeQwen35Model(layer_types)

    class ConfigAuto(_FakeAutoClass):
        result = config

    class TokenizerAuto(_FakeAutoClass):
        result = tokenizer

    class ModelAuto(_FakeAutoClass):
        result = model

    loaded = probe.load_official_reference(
        Path("/immutable/model"),
        requested_dtype="float32",
        auto_config=ConfigAuto,
        auto_tokenizer=TokenizerAuto,
        auto_model=ModelAuto,
    )
    assert loaded["requested_model_dtype"] == "float32"
    assert ModelAuto.calls[-1][1]["dtype"] is torch.float32
    assert "torch_dtype" not in ModelAuto.calls[-1][1]
    assert model.to_calls == [{
        "device": "cuda:0",
        "dtype": torch.float32,
    }]
    assert loaded["architecture"]["parameter_dtypes"] == {
        "float32": 5,
    }


def test_load_official_reference_explicitly_converts_bfloat16_parameters():
    layer_types = _canonical_layer_types()
    config = _FakeQwen35Config(layer_types)
    tokenizer = _FakeTokenizer()
    model = _FakeQwen35Model(layer_types)

    class ConfigAuto(_FakeAutoClass):
        result = config

    class TokenizerAuto(_FakeAutoClass):
        result = tokenizer

    class ModelAuto(_FakeAutoClass):
        result = model

    loaded = probe.load_official_reference(
        Path("/immutable/model"),
        requested_dtype="bfloat16",
        auto_config=ConfigAuto,
        auto_tokenizer=TokenizerAuto,
        auto_model=ModelAuto,
    )
    assert model.to_calls == [{
        "device": "cuda:0",
        "dtype": torch.bfloat16,
    }]
    assert loaded["architecture"]["parameter_dtypes"] == {
        "bfloat16": 5,
    }


def test_load_official_reference_fails_closed_when_conversion_is_ignored():
    layer_types = _canonical_layer_types()
    config = _FakeQwen35Config(layer_types)
    tokenizer = _FakeTokenizer()
    model = _FakeQwen35Model(
        layer_types,
        apply_dtype_conversion=False,
    )

    class ConfigAuto(_FakeAutoClass):
        result = config

    class TokenizerAuto(_FakeAutoClass):
        result = tokenizer

    class ModelAuto(_FakeAutoClass):
        result = model

    _expect_incomplete(
        lambda: probe.load_official_reference(
            Path("/immutable/model"),
            requested_dtype="float32",
            auto_config=ConfigAuto,
            auto_tokenizer=TokenizerAuto,
            auto_model=ModelAuto,
        ),
        "INCOMPLETE_MODEL_LOAD",
    )


class _FakeCuda:
    def __init__(self):
        self.synchronize_calls = 0
        self.reset_peak_calls = 0

    def is_available(self):
        return True

    def synchronize(self):
        self.synchronize_calls += 1

    def memory_allocated(self):
        return 101

    def memory_reserved(self):
        return 202

    def max_memory_allocated(self):
        return 303

    def max_memory_reserved(self):
        return 404

    def reset_peak_memory_stats(self):
        self.reset_peak_calls += 1


class _CasePeakCuda(_FakeCuda):
    def __init__(self):
        super().__init__()
        self.case_index = 0

    def reset_peak_memory_stats(self):
        super().reset_peak_memory_stats()
        self.case_index += 1

    def max_memory_allocated(self):
        return 1000 if self.case_index == 1 else self.case_index * 10

    def max_memory_reserved(self):
        return 2000 if self.case_index == 1 else self.case_index * 20


def _run_complete_reference_matrix(run_dir):
    return probe.run_reference_case_matrix(
        adapter_factory=_FakeReferenceStateAdapter,
        architecture=probe.inspect_model(
            model=_FakeQwen35Model(_canonical_layer_types()),
            config=_FakeQwen35Config(_canonical_layer_types()),
            tokenizer=_FakeTokenizer(),
        ),
        run_dir=run_dir,
        contract_sha256=probe.contract_file_sha256(),
        parameter_bytes=1234,
        cuda_module=_FakeCuda(),
    )


def _canonical_layer_types():
    return tuple(
        "full_attention" if (index + 1) % 4 == 0 else "linear_attention"
        for index in range(24)
    )


def test_walk_tensor_leaves_preserves_explicit_paths_and_aliases():
    storage = torch.arange(8, dtype=torch.float32)
    state = {"layers": [{"key": storage[:4], "value": storage[4:]}]}
    leaves = list(probe.walk_tensor_leaves(state))
    assert [path for path, _ in leaves] == [
        "layers[0].key",
        "layers[0].value",
    ]
    assert leaves[0][1].untyped_storage().data_ptr() == (
        leaves[1][1].untyped_storage().data_ptr()
    )


def test_inspect_model_reconstructs_exact_hybrid_schedule():
    layer_types = _canonical_layer_types()
    result = probe.inspect_model(
        model=_FakeQwen35Model(layer_types),
        config=_FakeQwen35Config(layer_types),
        tokenizer=_FakeTokenizer(),
    )
    assert result["num_hidden_layers"] == 24
    assert result["linear_attention_layers"] == 18
    assert result["full_attention_layers"] == 6
    assert result["full_attention_interval"] == 4
    assert result["layer_schedule"] == {
        str(index): layer_type
        for index, layer_type in enumerate(layer_types)
    }
    assert result["parameter_dtypes"] == {
        "float16": 4,
        "float32": 1,
    }
    assert result["tokenizer_vocab_size"] == _FakeTokenizer.vocab_size
    assert result["model_vocab_size"] == 152064


def test_architecture_mismatch_fails_before_correctness_execution():
    _expect_incomplete(
        lambda: probe.require_canonical_architecture({
            "num_hidden_layers": 23,
            "layer_schedule": {},
        }),
        "INCOMPLETE_ARCHITECTURE",
    )


def test_reference_modes_emit_comparable_step_records():
    adapter = _FakeReferenceStateAdapter()
    oracle = probe.run_one_shot_oracle(
        adapter,
        token_ids=(1, 2, 3),
        decode_steps=2,
    )
    cached = probe.run_cached_decode(
        adapter,
        token_ids=(1, 2, 3),
        decode_steps=2,
    )
    chunked = probe.run_chunked_prefill_decode(
        adapter,
        token_ids=(1, 2, 3),
        chunk_schedule=(1, 2),
        decode_steps=2,
    )
    assert cached["decoded_token_ids"] == oracle["decoded_token_ids"]
    assert chunked["decoded_token_ids"] == oracle["decoded_token_ids"]
    assert len(cached["state_snapshot_ids"]) == 3
    assert len(chunked["state_snapshot_ids"]) == 4
    assert all(
        set(record) == set(contract.LOGIT_RECORD_FIELDS)
        for record in cached["logit_records"]
    )
    assert all(
        record["max_abs_diff"] == 0.0
        for record in cached["logit_records"]
    )
    assert [
        record["position_metadata"]["oracle_greedy_token_id"]
        for record in cached["logit_records"]
    ] == cached["decoded_token_ids"]
    assert all(
        record["position_metadata"]["actual_full_logit_sha256"]
        == record["full_logit_sha256"]
        for record in cached["logit_records"]
    )


def test_logit_record_contains_independent_decision_evidence():
    actual = torch.linspace(-2.0, 2.0, 32, dtype=torch.float32)
    oracle = actual.clone()
    actual[7] = 4.0
    actual[9] = 3.0
    oracle[7] = 3.75
    oracle[9] = 2.75
    record = probe._logit_record(
        logits=actual,
        oracle_logits=oracle,
        request_id="request-0",
        request_generation=0,
        step_index=1,
        sequence_length=18,
        comparison_policy="bf16_decision_preserving",
    )
    assert record["actual_winner_token_id"] == 7
    assert record["oracle_winner_token_id"] == 7
    assert record["actual_runner_up_token_id"] == 9
    assert record["oracle_runner_up_token_id"] == 9
    assert record["actual_winner_margin"] == 1.0
    assert record["oracle_winner_margin"] == 1.0
    assert record["topk_token_ids"] == record["actual_topk_token_ids"]
    assert record["topk_logits"] == record["actual_topk_logits"]
    assert set(record["abs_diff_percentiles"]) == {
        "p50",
        "p95",
        "p99",
        "p99_9",
    }


def test_logit_record_rejects_winner_tie():
    actual = torch.zeros(32)
    oracle = torch.zeros(32)
    actual[3] = actual[4] = 2.0
    oracle[3] = 2.0
    oracle[4] = 1.0
    _expect_incomplete(
        lambda: probe._logit_record(
            logits=actual,
            oracle_logits=oracle,
            request_id="request-0",
            request_generation=0,
            step_index=0,
            sequence_length=17,
            comparison_policy="bf16_decision_preserving",
        ),
        "INCOMPLETE_REFERENCE_SEMANTICS",
    )


def test_fp32_summary_counts_only_values_outside_frozen_allclose():
    oracle = torch.linspace(-1.0, 1.0, 32, dtype=torch.float32)
    oracle[-1] = 4.0
    oracle[-2] = 3.0
    threshold = (
        contract.FP32_ATOL
        + contract.FP32_RTOL * oracle.abs()
    )
    inside = oracle + threshold * 0.5
    outside = inside.clone()
    outside[0] = oracle[0] + threshold[0] * 2.0

    def make_record(actual):
        return probe._logit_record(
            logits=actual,
            oracle_logits=oracle,
            request_id="request-0",
            request_generation=0,
            step_index=0,
            sequence_length=17,
            comparison_policy="fp32_elementwise",
        )

    assert make_record(inside)["allclose_violation_count"] == 0
    assert make_record(outside)["allclose_violation_count"] == 1


def test_comparison_metrics_clamps_float32_cosine_to_definition_domain():
    generator = torch.Generator().manual_seed(17)
    value = torch.randn(32, generator=generator, dtype=torch.float32)
    raw_cosine = torch.nn.functional.cosine_similarity(
        value.reshape(1, -1),
        value.reshape(1, -1),
    )
    assert raw_cosine.item() > 1.0
    metrics = probe._comparison_metrics(value, value.clone())
    assert metrics["cosine_similarity"] == 1.0


def test_export_import_preserves_next_step_logits():
    result = probe.run_export_import_continuation(
        _FakeReferenceStateAdapter(),
        token_ids=(1, 2, 3),
    )
    assert result["decoded_token_ids_equal"] is True
    assert result["full_logit_sha256_equal"] is True
    assert result["max_abs_diff"] == 0.0
    assert result["max_rel_diff"] == 0.0


def test_interleaved_decode_does_not_mutate_inactive_requests():
    result = probe.run_interleaved_requests(
        _FakeReferenceStateAdapter(),
        request_token_ids={
            "slot-0": (1, 2, 3),
            "slot-1": (4, 5),
            "slot-2": (6, 7, 8, 9),
        },
        replacement_token_ids=(10, 11, 12),
    )
    assert result["inactive_request_hash_changes"] == []
    assert result["serial_oracle_mismatches"] == []
    assert len(result["decoded_token_ids"]) == 6
    assert len(result["logit_records"]) == 6
    assert all(
        set(record) == set(contract.LOGIT_RECORD_FIELDS)
        for record in result["logit_records"]
    )


def test_slot_reuse_increments_generation_and_releases_old_state():
    result = probe.run_interleaved_requests(
        _FakeReferenceStateAdapter(),
        request_token_ids={
            "slot-0": (1, 2, 3),
            "slot-1": (4, 5),
            "slot-2": (6, 7, 8, 9),
        },
        replacement_token_ids=(10, 11, 12),
    )
    assert result["slot_generations"]["slot-0"] == [0, 1]
    assert result["released_generations"] == [["slot-0", 0]]
    assert result["stale_state_reads"] == []


def test_reference_state_adapter_uses_explicit_cache_codec():
    model = _FakeCausalModel()
    adapter = probe.ReferenceStateAdapter(
        model=model,
        layer_schedule={0: "linear_attention"},
        vocab_size=32,
        device="cpu",
    )
    logits, state = adapter.prefill(
        torch.tensor([[1, 2, 3]], dtype=torch.long),
        None,
    )
    exported = adapter.export_state(
        state,
        "request-a",
        0,
        3,
    )
    restored = adapter.import_state(exported)
    original_logits, _ = adapter.decode_one(4, state, 3)
    restored_logits, _ = adapter.decode_one(4, restored, 3)
    assert torch.equal(logits, model(
        input_ids=torch.tensor([[1, 2, 3]], dtype=torch.long),
        past_key_values=None,
        use_cache=True,
        return_dict=True,
        cache_position=torch.arange(3),
    ).logits[-1, -1])
    assert torch.equal(original_logits, restored_logits)
    assert exported["cache_codec"] == "legacy_cache"
    assert exported["request_generation"] == 0


def test_reference_state_adapter_round_trips_explicit_hybrid_cache():
    model = _FakeHybridCausalModel()
    adapter = probe.ReferenceStateAdapter(
        model=model,
        layer_schedule={
            0: "linear_attention",
            1: "full_attention",
        },
        vocab_size=32,
        device="cpu",
    )
    _, state = adapter.prefill(
        torch.tensor([[1, 2, 3]], dtype=torch.long),
        None,
    )
    exported = adapter.export_state(state, "request-a", 0, 3)
    restored = adapter.import_state(exported)
    original_logits, _ = adapter.decode_one(4, state, 3)
    restored_logits, _ = adapter.decode_one(4, restored, 3)
    assert exported["cache_codec"] == "hybrid_layers_v1"
    assert torch.equal(original_logits, restored_logits)
    assert adapter.state_for_normalization(restored) == {
        "layers": {
            "0": {
                "linear_convolution_state": (
                    restored.layers[0].conv_states
                ),
                "linear_recurrent_state": (
                    restored.layers[0].recurrent_states
                ),
            },
            "1": {
                "full_attention_key": restored.layers[1].keys,
                "full_attention_value": restored.layers[1].values,
            },
        },
    }


def test_reference_state_adapter_binds_explicit_vocab_size():
    adapter = probe.ReferenceStateAdapter(
        model=_FakeCausalModel(),
        layer_schedule={0: "linear_attention"},
        vocab_size=32,
        model_vocab_size=40,
        device="cpu",
    )
    assert adapter.vocab_size == 32
    assert adapter.model_vocab_size == 40
    token_ids = probe._case_token_ids(adapter, prompt_length=5, seed=7)
    assert len(token_ids) == 5
    assert all(0 < token_id < 32 for token_id in token_ids)


def test_reference_state_adapter_accepts_model_padding_vocab_tokens():
    adapter = probe.ReferenceStateAdapter(
        model=_FakeCausalModel(),
        layer_schedule={0: "linear_attention"},
        vocab_size=32,
        model_vocab_size=40,
        device="cpu",
    )
    logits = adapter.one_shot((1, 2, 35))
    assert logits.shape == (32,)


def test_reference_state_adapter_one_shot_runs_full_token_path():
    model = _FakeCausalModel()
    adapter = probe.ReferenceStateAdapter(
        model=model,
        layer_schedule={0: "linear_attention"},
        vocab_size=32,
        device="cpu",
    )
    logits = adapter.one_shot((1, 2, 3, 4))
    expected_logits = model(
        input_ids=torch.tensor([[1, 2, 3, 4]], dtype=torch.long),
        past_key_values=None,
        use_cache=True,
        return_dict=True,
        cache_position=torch.arange(4),
    ).logits[-1, -1]
    assert torch.equal(logits, expected_logits)
    assert model.calls[0] == {
        "input_ids": [1, 2, 3, 4],
        "cache_position": [0, 1, 2, 3],
    }


def test_reference_state_adapter_runs_cached_worker_with_real_state_rows():
    adapter = probe.ReferenceStateAdapter(
        model=_FakeCausalModel(),
        layer_schedule={0: "linear_attention"},
        vocab_size=32,
        device="cpu",
    )
    observer = probe._StateEvidenceCollector(
        adapter=adapter,
        layer_schedule={0: "linear_attention"},
        cuda=_FakeCuda(),
    )
    case = next(
        case
        for case in contract.build_case_matrix()
        if case.execution_mode == "one_shot_vs_cached"
    )
    row = probe._run_reference_case(case, adapter, observer)
    assert row["complete"] is True
    assert row["logit_records"]
    assert observer.state_components
    assert all(
        component["request_id"] == "request-0"
        for component in observer.state_components
    )


def test_cached_worker_fails_closed_without_one_shot_semantics():
    class _NoOneShotAdapter(_FakeReferenceStateAdapter):
        one_shot = None

    case = next(
        case
        for case in contract.build_case_matrix()
        if case.execution_mode == "one_shot_vs_cached"
    )
    adapter = _NoOneShotAdapter()
    observer = probe._StateEvidenceCollector(
        adapter=adapter,
        layer_schedule={0: "linear_attention"},
        cuda=_FakeCuda(),
    )
    _expect_incomplete(
        lambda: probe._run_reference_case(case, adapter, observer),
        "INCOMPLETE_REFERENCE_SEMANTICS",
    )


def test_reference_state_adapter_fails_closed_without_cache_codec():
    adapter = probe.ReferenceStateAdapter(
        model=_FakeCausalModel(cache_type=_OpaqueCache),
        layer_schedule={0: "linear_attention"},
        vocab_size=32,
        device="cpu",
    )
    _, state = adapter.prefill(
        torch.tensor([[1, 2, 3]], dtype=torch.long),
        None,
    )
    _expect_incomplete(
        lambda: adapter.export_state(state, "request-a", 0, 3),
        "INCOMPLETE_REFERENCE_SEMANTICS",
    )


def test_load_official_reference_uses_local_read_only_arguments():
    layer_types = _canonical_layer_types()
    config = _FakeQwen35Config(layer_types)
    tokenizer = _FakeTokenizer()
    model = _FakeQwen35Model(layer_types)
    config_auto = type("ConfigAuto", (_FakeAutoClass,), {
        "calls": [],
        "result": config,
    })
    tokenizer_auto = type("TokenizerAuto", (_FakeAutoClass,), {
        "calls": [],
        "result": tokenizer,
    })
    model_auto = type("ModelAuto", (_FakeAutoClass,), {
        "calls": [],
        "result": model,
    })
    loaded = probe.load_official_reference(
        Path("/immutable/model"),
        auto_config=config_auto,
        auto_tokenizer=tokenizer_auto,
        auto_model=model_auto,
    )
    assert loaded["config"] is config
    assert loaded["tokenizer"] is tokenizer
    assert loaded["model"] is model
    assert config_auto.calls == [(
        "/immutable/model",
        {
            "local_files_only": True,
            "trust_remote_code": False,
        },
    )]
    assert tokenizer_auto.calls == config_auto.calls
    assert model_auto.calls == [(
        "/immutable/model",
        {
            "local_files_only": True,
            "trust_remote_code": False,
            "dtype": torch.bfloat16,
        },
    )]
    assert loaded["requested_model_dtype"] == "bfloat16"
    assert model.to_calls == [{
        "device": "cuda:0",
        "dtype": torch.bfloat16,
    }]


def test_inspect_model_reads_hybrid_fields_from_nested_text_config():
    layer_types = _canonical_layer_types()
    result = probe.inspect_model(
        model=_FakeQwen35Model(layer_types),
        config=_FakeQwen35MultimodalConfig(layer_types),
        tokenizer=_FakeTokenizer(),
    )
    assert result["config_class"] == "_FakeQwen35MultimodalConfig"
    assert result["num_hidden_layers"] == 24
    assert result["linear_attention_layers"] == 18
    assert result["full_attention_layers"] == 6
    assert result["full_attention_interval"] == 4
    assert result["linear_num_key_heads"] == 16
    assert result["linear_num_value_heads"] == 16
    assert result["linear_key_head_dim"] == 128
    assert result["linear_value_head_dim"] == 128
    assert result["linear_conv_kernel_dim"] == 4
    assert result["mamba_ssm_dtype"] == "float32"


def test_torch_custom_op_annotation_compatibility_is_temporary():
    calls = []

    def operation(input: "torch.Tensor") -> "torch.Tensor":
        return input

    original_annotations = dict(operation.__annotations__)

    def infer_schema(function, mutates_args=()):
        calls.append((
            dict(function.__annotations__),
            tuple(mutates_args),
        ))
        assert function.__annotations__ == {
            "input": torch.Tensor,
            "return": torch.Tensor,
        }
        return "(Tensor input) -> Tensor"

    with probe.torch_custom_op_annotation_compatibility(
        infer_schema_owner=SimpleNamespace(infer_schema=infer_schema),
    ):
        result = probe._resolve_custom_op_schema(
            operation,
            (),
        )
        assert result == "(Tensor input) -> Tensor"
    assert calls == [({
        "input": torch.Tensor,
        "return": torch.Tensor,
    }, ())]
    assert operation.__annotations__ == original_annotations

    already_resolved = lambda input: input
    already_resolved.__annotations__ = {
        "input": torch.Tensor,
        "return": torch.Tensor,
    }
    with probe.torch_custom_op_annotation_compatibility(
        infer_schema_owner=SimpleNamespace(infer_schema=infer_schema),
    ):
        probe._resolve_custom_op_schema(already_resolved, ())
    assert already_resolved.__annotations__ == {
        "input": torch.Tensor,
        "return": torch.Tensor,
    }


def test_load_official_reference_scopes_custom_op_compatibility_to_model():
    layer_types = _canonical_layer_types()
    config = _FakeQwen35Config(layer_types)
    tokenizer = _FakeTokenizer()
    model = _FakeQwen35Model(layer_types)
    events = []

    class Compatibility:
        def __enter__(self):
            events.append("enter")

        def __exit__(self, exc_type, exc, traceback):
            events.append("exit")

    class ConfigAuto(_FakeAutoClass):
        calls = []
        result = config

        @classmethod
        def from_pretrained(cls, model_dir, **kwargs):
            assert events == []
            return super().from_pretrained(model_dir, **kwargs)

    class TokenizerAuto(_FakeAutoClass):
        calls = []
        result = tokenizer

        @classmethod
        def from_pretrained(cls, model_dir, **kwargs):
            assert events == []
            return super().from_pretrained(model_dir, **kwargs)

    class ModelAuto(_FakeAutoClass):
        calls = []
        result = model

        @classmethod
        def from_pretrained(cls, model_dir, **kwargs):
            assert events == ["enter"]
            return super().from_pretrained(model_dir, **kwargs)

    probe.load_official_reference(
        Path("/immutable/model"),
        auto_config=ConfigAuto,
        auto_tokenizer=TokenizerAuto,
        auto_model=ModelAuto,
        custom_op_compatibility=lambda: Compatibility(),
    )
    assert events == ["enter", "exit"]


def test_capture_memory_snapshot_separates_allocator_and_state_ledger():
    components = _synthetic_components()
    fake_cuda = _FakeCuda()
    snapshot = probe.capture_memory_snapshot(
        snapshot_id="memory-0",
        phase="after_prefill",
        request_id="request-a",
        request_generation=0,
        components=components,
        cuda=fake_cuda,
    )
    assert snapshot == {
        "snapshot_id": "memory-0",
        "phase": "after_prefill",
        "request_id": "request-a",
        "request_generation": 0,
        "cuda_allocated_bytes": 101,
        "cuda_reserved_bytes": 202,
        "logical_state_bytes": sum(
            component["logical_bytes"] for component in components
        ),
        "unique_storage_bytes": contract.unique_storage_bytes(components),
    }
    assert fake_cuda.synchronize_calls == 1


def test_emit_raw_probe_artifacts_writes_exact_worker_schemas():
    component = _synthetic_components()[0]
    state_snapshot = {
        "snapshot_id": "state-0",
        "request_id": "request-a",
        "request_generation": 0,
        "lifetime_epoch": 1,
        "sequence_length": 17,
        "component_count": 1,
        "component_sha256": contract.canonical_json_sha256([component]),
    }
    memory_snapshot = {
        "snapshot_id": "memory-0",
        "phase": "after_prefill",
        "request_id": "request-a",
        "request_generation": 0,
        "cuda_allocated_bytes": 101,
        "cuda_reserved_bytes": 202,
        "logical_state_bytes": component["logical_bytes"],
        "unique_storage_bytes": component["storage_nbytes"],
    }
    case_row = {
        "row_id": "row-0",
        "case_id": "case-0",
        "phase": "one_shot_vs_cached",
        "execution_mode": "cached_decode",
        "prompt_length": 17,
        "chunk_schedule": [17],
        "request_count": 1,
        "decode_steps": 1,
        "repeat_index": 0,
        "request_ids": ["request-a"],
        "request_generations": [0],
        "decoded_token_ids": [7],
        "logit_records": [],
        "state_snapshot_ids": ["state-0"],
        "memory_snapshot_ids": ["memory-0"],
        "complete": True,
        "failure_kind": None,
        "failure_detail": None,
        "execution_dtype": "bfloat16",
        "comparison_policy": "bf16_decision_preserving",
    }
    with tempfile.TemporaryDirectory() as temporary:
        run_dir = Path(temporary)
        summary = probe.emit_raw_probe_artifacts(
            run_dir=run_dir,
            architecture={"num_hidden_layers": 24},
            case_rows=[case_row],
            state_snapshots=[state_snapshot],
            state_components=[component],
            memory_snapshots=[memory_snapshot],
            parameter_bytes=1234,
            max_memory_allocated=303,
            max_memory_reserved=404,
        )
        assert summary["parameter_bytes"] == 1234
        assert summary["state_logical_bytes"] == component["logical_bytes"]
        assert summary["state_unique_storage_bytes"] == (
            component["storage_nbytes"]
        )
        assert summary["non_state_peak_allocator_observation_bytes"] == (
            303 - component["storage_nbytes"]
        )
        for filename in (
            "case_rows.jsonl",
            "state_snapshots.jsonl",
            "state_components.jsonl",
            "memory_snapshots.jsonl",
            "summary.json",
        ):
            assert (run_dir / filename).is_file()
        assert not list(run_dir.glob("*.partial"))
        assert set(json.loads(
            (run_dir / "case_rows.jsonl").read_text().strip()
        )) == set(contract.CASE_ROW_FIELDS)


def test_reference_case_matrix_emits_every_frozen_case_exactly_once():
    with tempfile.TemporaryDirectory() as temporary:
        result = _run_complete_reference_matrix(Path(temporary))
        expected_cases = contract.build_case_matrix()
        rows = result["case_rows"]
        assert len(rows) == len(expected_cases) == 35
        assert [row["case_id"] for row in rows] == [
            case.case_id for case in expected_cases
        ]
        assert all(
            set(row) == set(contract.CASE_ROW_FIELDS)
            for row in rows
        )
        assert all(row["complete"] is True for row in rows)
        assert all(row["failure_kind"] is None for row in rows)


def test_reference_case_matrix_routes_fp32_to_dedicated_factory():
    calls = []

    def bf16_factory():
        calls.append("bfloat16")
        return _FakeReferenceStateAdapter()

    def fp32_factory():
        calls.append("float32")
        return _FakeReferenceStateAdapter()

    with tempfile.TemporaryDirectory() as temporary:
        result = probe.run_reference_case_matrix(
            adapter_factory=bf16_factory,
            fp32_adapter_factory=fp32_factory,
            architecture=probe.inspect_model(
                model=_FakeQwen35Model(_canonical_layer_types()),
                config=_FakeQwen35Config(_canonical_layer_types()),
                tokenizer=_FakeTokenizer(),
            ),
            run_dir=Path(temporary),
            contract_sha256=probe.contract_file_sha256(),
            parameter_bytes=1234,
            cuda_module=_FakeCuda(),
        )
    assert calls == ["bfloat16", "float32"]
    control = next(
        row
        for row in result["case_rows"]
        if row["case_id"] == contract.FP32_CONTROL_CASE_ID
    )
    assert control["execution_dtype"] == "float32"
    assert control["comparison_policy"] == "fp32_elementwise"


def test_dtype_profile_reconstructs_parameter_and_state_dtypes():
    components = _synthetic_components()
    components.extend([
        dict(
            components[0],
            state_role="linear_recurrent_state",
            dtype="float32",
        ),
        dict(
            components[0],
            state_role="full_attention_key",
            dtype="bfloat16",
        ),
    ])
    profile = probe._dtype_profile(
        requested_model_dtype="bfloat16",
        architecture={
            "parameter_dtypes": {
                "bfloat16": 100,
                "float32": 10,
            },
        },
        state_components=components,
        logit_dtype="bfloat16",
    )
    assert profile == {
        "requested_model_dtype": "bfloat16",
        "dominant_parameter_dtype": "bfloat16",
        "logit_dtype_before_comparison": "bfloat16",
        "comparison_accumulator_dtype": "float32",
        "recurrent_state_dtypes": ["float32"],
        "kv_state_dtypes": ["bfloat16", "float32"],
    }


def test_architecture_identity_ignores_only_parameter_dtype_counts():
    left = probe.inspect_model(
        model=_FakeQwen35Model(_canonical_layer_types()),
        config=_FakeQwen35Config(_canonical_layer_types()),
        tokenizer=_FakeTokenizer(),
    )
    right = dict(left)
    right["parameter_dtypes"] = {"float32": 5}
    assert probe._architecture_identity(left) == (
        probe._architecture_identity(right)
    )
    right["full_attention_interval"] = 8
    assert probe._architecture_identity(left) != (
        probe._architecture_identity(right)
    )


def test_reference_case_matrix_includes_before_prefill_state_snapshot():
    with tempfile.TemporaryDirectory() as temporary:
        result = _run_complete_reference_matrix(Path(temporary))
        rows_by_id = {
            row["case_id"]: row for row in result["case_rows"]
        }
        for case in contract.build_case_matrix():
            assert len(rows_by_id[case.case_id]["state_snapshot_ids"]) == (
                case.expected_state_snapshots
            )


def test_interleaved_matrix_row_carries_raw_correctness_records():
    with tempfile.TemporaryDirectory() as temporary:
        result = probe.run_reference_case_matrix(
            adapter_factory=_FakeReferenceStateAdapter,
            architecture=probe.inspect_model(
                model=_FakeQwen35Model(_canonical_layer_types()),
                config=_FakeQwen35Config(_canonical_layer_types()),
                tokenizer=_FakeTokenizer(),
            ),
            run_dir=Path(temporary),
            contract_sha256=probe.contract_file_sha256(),
            parameter_bytes=1234,
            cuda_module=_FakeCuda(),
        )
    row = next(
        item
        for item in result["case_rows"]
        if item["phase"] == "interleaved_multi_request"
    )
    assert len(row["decoded_token_ids"]) == (
        len(contract.MULTI_REQUEST_LENGTHS) * contract.DECODE_STEPS
    )
    assert len(row["logit_records"]) == (
        len(contract.MULTI_REQUEST_LENGTHS) * contract.DECODE_STEPS
    )


def test_reference_case_matrix_materializes_real_state_and_allocator_rows():
    with tempfile.TemporaryDirectory() as temporary:
        result = _run_complete_reference_matrix(Path(temporary))
        assert result["state_components"]
        components_by_epoch = {}
        for component in result["state_components"]:
            key = (
                component["request_id"],
                component["request_generation"],
                component["lifetime_epoch"],
            )
            components_by_epoch.setdefault(key, []).append(component)
        nonempty_snapshots = 0
        for snapshot in result["state_snapshots"]:
            if snapshot["sequence_length"] == 0:
                assert snapshot["component_count"] == 0
                continue
            key = (
                snapshot["request_id"],
                snapshot["request_generation"],
                snapshot["lifetime_epoch"],
            )
            components = sorted(
                components_by_epoch[key],
                key=probe._component_sort_key,
            )
            assert snapshot["component_count"] == len(components)
            assert snapshot["component_sha256"] == (
                contract.canonical_json_sha256(components)
            )
            nonempty_snapshots += 1
        assert nonempty_snapshots > 0
        for memory in result["memory_snapshots"]:
            assert memory["cuda_allocated_bytes"] == 101
            assert memory["cuda_reserved_bytes"] == 202


def test_reference_case_matrix_resets_cuda_peak_before_every_case():
    fake_cuda = _FakeCuda()
    with tempfile.TemporaryDirectory() as temporary:
        probe.run_reference_case_matrix(
            adapter_factory=_FakeReferenceStateAdapter,
            architecture=probe.inspect_model(
                model=_FakeQwen35Model(_canonical_layer_types()),
                config=_FakeQwen35Config(_canonical_layer_types()),
                tokenizer=_FakeTokenizer(),
            ),
            run_dir=Path(temporary),
            contract_sha256=probe.contract_file_sha256(),
            parameter_bytes=1234,
            cuda_module=fake_cuda,
        )
    assert fake_cuda.reset_peak_calls == len(contract.build_case_matrix())


def test_same_path_repeatability_uses_identical_prompt_tokens():
    adapter = _FakeReferenceStateAdapter()
    observer = probe._StateEvidenceCollector(
        adapter=adapter,
        layer_schedule={0: "linear_attention"},
        cuda=_FakeCuda(),
    )
    cases = [
        case
        for case in contract.build_case_matrix()
        if (
            case.execution_mode == "cached_repeatability"
            and case.prompt_length == contract.PROMPT_LENGTHS[0]
        )
    ]
    rows = [
        probe._run_reference_case(case, adapter, observer)
        for case in cases
    ]
    assert rows[0]["decoded_token_ids"] == rows[1]["decoded_token_ids"]
    assert [
        record["full_logit_sha256"]
        for record in rows[0]["logit_records"]
    ] == [
        record["full_logit_sha256"]
        for record in rows[1]["logit_records"]
    ]


def test_chunked_and_cached_cases_use_identical_prompt_tokens():
    class RecordingAdapter(_FakeReferenceStateAdapter):
        def __init__(self):
            self.prefill_inputs = []

        def prefill(self, input_ids, state):
            self.prefill_inputs.append(tuple(
                int(value) for value in input_ids.flatten()
            ))
            return super().prefill(input_ids, state)

    adapter = RecordingAdapter()
    observer = probe._StateEvidenceCollector(
        adapter=adapter,
        layer_schedule={0: "linear_attention"},
        cuda=_FakeCuda(),
    )
    cached_case = next(
        case
        for case in contract.build_case_matrix()
        if (
            case.phase == "one_shot_vs_cached"
            and case.prompt_length == 65
        )
    )
    chunked_case = next(
        case
        for case in contract.build_case_matrix()
        if (
            case.phase == "one_shot_vs_chunked"
            and case.prompt_length == 65
            and case.chunk_schedule == (31, 34)
        )
    )
    probe._run_reference_case(cached_case, adapter, observer)
    cached_tokens = adapter.prefill_inputs[0]
    adapter.prefill_inputs.clear()
    probe._run_reference_case(chunked_case, adapter, observer)
    chunked_tokens = tuple(
        token_id
        for chunk in adapter.prefill_inputs
        for token_id in chunk
    )
    assert chunked_tokens == cached_tokens


def test_decode_memory_phases_include_step_index():
    with tempfile.TemporaryDirectory() as temporary:
        result = _run_complete_reference_matrix(Path(temporary))
    phases = {
        row["phase"]
        for row in result["memory_snapshots"]
        if row["phase"].startswith("after_decode")
    }
    assert "after_decode_step_0" in phases
    assert f"after_decode_step_{contract.DECODE_STEPS - 1}" in phases


def test_reference_case_matrix_aggregates_peak_across_cases():
    fake_cuda = _CasePeakCuda()
    with tempfile.TemporaryDirectory() as temporary:
        result = probe.run_reference_case_matrix(
            adapter_factory=_FakeReferenceStateAdapter,
            architecture=probe.inspect_model(
                model=_FakeQwen35Model(_canonical_layer_types()),
                config=_FakeQwen35Config(_canonical_layer_types()),
                tokenizer=_FakeTokenizer(),
            ),
            run_dir=Path(temporary),
            contract_sha256=probe.contract_file_sha256(),
            parameter_bytes=1234,
            cuda_module=fake_cuda,
        )
    assert result["summary"]["max_memory_allocated"] == 1000
    assert result["summary"]["max_memory_reserved"] == 2000


def test_slot_reuse_case_records_release_allocator_phase():
    case = next(
        case
        for case in contract.build_case_matrix()
        if case.execution_mode == "completion_release_slot_reuse"
    )
    adapter = _FakeReferenceStateAdapter()
    observer = probe._StateEvidenceCollector(
        adapter=adapter,
        layer_schedule={0: "linear_attention"},
        cuda=_FakeCuda(),
    )
    probe._run_reference_case(case, adapter, observer)
    assert "after_request_release" in {
        row["phase"] for row in observer.memory_snapshots
    }
    released = [
        component
        for component in observer.state_components
        if component["update_kind"] == "released"
    ]
    assert released
    assert {
        component["request_generation"] for component in released
    } == {0}
    release_snapshot = next(
        row
        for row in observer.state_snapshots
        if row["snapshot_id"].endswith(":after_request_release")
    )
    assert release_snapshot["component_count"] == len(released)


def test_slot_reuse_case_runs_full_interleaved_lifecycle_domain():
    case = next(
        case
        for case in contract.build_case_matrix()
        if case.execution_mode == "completion_release_slot_reuse"
    )
    adapter = _FakeReferenceStateAdapter()
    observer = probe._StateEvidenceCollector(
        adapter=adapter,
        layer_schedule={0: "linear_attention"},
        cuda=_FakeCuda(),
    )
    row = probe._run_reference_case(case, adapter, observer)
    record_counts = {}
    for record in row["logit_records"]:
        key = (
            record["request_id"],
            record["request_generation"],
        )
        record_counts[key] = record_counts.get(key, 0) + 1
    assert record_counts == {
        ("slot-0", 0): 2,
        ("slot-1", 0): contract.DECODE_STEPS,
        ("slot-2", 0): contract.DECODE_STEPS,
        ("slot-0", 1): contract.DECODE_STEPS,
    }
    assert len(row["state_snapshot_ids"]) == 34
    assert row["request_ids"] == [
        "slot-0",
        "slot-1",
        "slot-2",
        "slot-0",
    ]
    assert row["request_generations"] == [0, 0, 0, 1]


def test_reference_worker_rejects_wrong_contract_hash_before_execution():
    adapter_calls = []

    def adapter_factory():
        adapter_calls.append("called")
        return _FakeReferenceStateAdapter()

    with tempfile.TemporaryDirectory() as temporary:
        _expect_incomplete(
            lambda: probe.run_reference_case_matrix(
                adapter_factory=adapter_factory,
                architecture=probe.inspect_model(
                    model=_FakeQwen35Model(_canonical_layer_types()),
                    config=_FakeQwen35Config(_canonical_layer_types()),
                    tokenizer=_FakeTokenizer(),
                ),
                run_dir=Path(temporary),
                contract_sha256="0" * 64,
                parameter_bytes=1234,
                cuda_module=_FakeCuda(),
            ),
            "INCOMPLETE_CONTRACT_MISMATCH",
        )
        assert adapter_calls == []
        assert not list(Path(temporary).iterdir())


def test_run_canonical_cli_writes_complete_raw_artifact_set_atomically():
    with tempfile.TemporaryDirectory() as temporary:
        run_dir = Path(temporary) / "run"
        model_dir = Path(temporary) / "model"
        model_dir.mkdir()
        stdout = io.StringIO()
        exit_code = probe.main(
            [
                "run-canonical",
                "--model-dir",
                os.fspath(model_dir),
                "--run-dir",
                os.fspath(run_dir),
                "--contract-sha256",
                probe.contract_file_sha256(),
            ],
            reference_loader=lambda _model_dir: (
                _FakeQwen35Config(_canonical_layer_types()),
                _FakeTokenizer(),
                _FakeQwen35Model(_canonical_layer_types()),
            ),
            adapter_factory=_FakeReferenceStateAdapter,
            cuda_module=_FakeCuda(),
            stdout=stdout,
        )
        assert exit_code == 0
        assert json.loads(stdout.getvalue())["case_row_count"] == 35
        for filename in (
            "architecture.json",
            "case_rows.jsonl",
            "state_snapshots.jsonl",
            "state_components.jsonl",
            "memory_snapshots.jsonl",
            "summary.json",
        ):
            assert (run_dir / filename).is_file()
        assert not list(run_dir.rglob("*.partial"))
        case_rows = [
            json.loads(line)
            for line in (run_dir / "case_rows.jsonl").read_text().splitlines()
        ]
        assert len(case_rows) == 35
        assert all(
            set(row) == set(contract.CASE_ROW_FIELDS)
            for row in case_rows
        )
        memory_rows = [
            json.loads(line)
            for line in (
                run_dir / "memory_snapshots.jsonl"
            ).read_text().splitlines()
        ]
        assert {"before_model_load", "after_model_load", "after_model_release"} <= {
            row["phase"] for row in memory_rows
        }


def test_run_canonical_cli_preserves_largest_per_case_peak():
    with tempfile.TemporaryDirectory() as temporary:
        run_dir = Path(temporary) / "run"
        model_dir = Path(temporary) / "model"
        model_dir.mkdir()
        stdout = io.StringIO()
        probe.main(
            [
                "run-canonical",
                "--model-dir",
                os.fspath(model_dir),
                "--run-dir",
                os.fspath(run_dir),
                "--contract-sha256",
                probe.contract_file_sha256(),
            ],
            reference_loader=lambda _model_dir: (
                _FakeQwen35Config(_canonical_layer_types()),
                _FakeTokenizer(),
                _FakeQwen35Model(_canonical_layer_types()),
            ),
            adapter_factory=_FakeReferenceStateAdapter,
            cuda_module=_CasePeakCuda(),
            stdout=stdout,
        )
        summary = json.loads(stdout.getvalue())
        assert summary["max_memory_allocated"] == 1000
        assert summary["max_memory_reserved"] == 2000


def test_walk_tensor_leaves_uses_frozen_container_order():
    state = {
        "z": torch.tensor([6]),
        "a": _DataclassState(
            recurrent_state=torch.tensor([1]),
            convolution_state=torch.tensor([2]),
        ),
        "named": _NamedTupleState(
            key_cache=torch.tensor([3]),
            value_cache=torch.tensor([4]),
        ),
        "sequence": [torch.tensor([5])],
    }
    paths = [path for path, _ in probe.walk_tensor_leaves(state)]
    assert paths == [
        "a.recurrent_state",
        "a.convolution_state",
        "named.key_cache",
        "named.value_cache",
        "sequence[0]",
        "z",
    ]


def test_walk_tensor_leaves_rejects_arbitrary_object_attributes():
    state = _AdapterState(
        hidden=torch.tensor([1]),
        ignored=torch.tensor([2]),
    )
    _expect_value_error(
        lambda: list(probe.walk_tensor_leaves(state)),
        "adapter",
    )
    leaves = list(probe.walk_tensor_leaves(
        state,
        adapter_registry={
            _AdapterState: lambda value: {"hidden": value.hidden},
        },
    ))
    assert [path for path, _ in leaves] == ["hidden"]


def test_classify_state_role_covers_frozen_role_domain():
    cases = (
        (
            "layers[0].key_cache",
            "full_attention",
            None,
            "full_attention_key",
        ),
        (
            "layers[0].value_cache",
            "full_attention",
            None,
            "full_attention_value",
        ),
        (
            "layers[0].recurrent_state",
            "linear_attention",
            None,
            "linear_recurrent_state",
        ),
        (
            "layers[0].convolution_state",
            "linear_attention",
            None,
            "linear_convolution_state",
        ),
        (
            "cache_position",
            "metadata",
            None,
            "position_or_sequence_metadata",
        ),
        (
            "layers[0].opaque",
            "linear_attention",
            "adapter_hidden",
            "other_persistent_state",
        ),
    )
    observed = {
        probe.classify_state_role(
            path,
            declared_layer_type=layer_type,
            component_name=component_name,
        )
        for path, layer_type, component_name, _ in cases
    }
    assert observed == set(contract.STATE_ROLES)
    for path, layer_type, component_name, expected in cases:
        assert probe.classify_state_role(
            path,
            declared_layer_type=layer_type,
            component_name=component_name,
        ) == expected


def test_normalization_assigns_request_generation_and_storage_identity():
    tensor = torch.zeros((1, 2, 3), dtype=torch.float32)
    rows = probe.normalize_state_components(
        state={"recurrent_state": tensor},
        request_id="request-a",
        request_generation=2,
        sequence_length=17,
        lifetime_epoch=3,
        layer_schedule={0: "linear_attention"},
    )
    assert rows[0]["request_generation"] == 2
    assert rows[0]["layer_index"] == 0
    assert rows[0]["state_role"] == "linear_recurrent_state"
    assert rows[0]["logical_bytes"] == tensor.numel() * tensor.element_size()
    assert rows[0]["storage_identity"]
    assert len(rows[0]["content_sha256"]) == 64
    assert set(rows[0]) == {
        field.name for field in contract.StateComponent.__dataclass_fields__.values()
    }


def test_normalization_preserves_alias_storage_and_unknown_roles():
    storage = torch.arange(8, dtype=torch.float32)
    rows = probe.normalize_state_components(
        state={
            "layers": [{
                "key_cache": storage[:4],
                "mystery": storage[4:],
            }],
        },
        request_id="request-a",
        request_generation=0,
        sequence_length=4,
        lifetime_epoch=1,
        layer_schedule={0: "full_attention"},
    )
    assert rows[0]["state_role"] == "full_attention_key"
    assert rows[1]["state_role"] == "other_persistent_state"
    assert rows[0]["storage_identity"] == rows[1]["storage_identity"]
    assert rows[0]["storage_nbytes"] == rows[1]["storage_nbytes"]


def _component(
    *,
    role,
    path,
    shape,
    storage_identity,
    content_sha256,
    storage_offset=0,
    storage_nbytes=64,
    generation=0,
):
    dtype = "float32"
    return {
        "request_id": "request-a",
        "request_generation": generation,
        "layer_index": 0,
        "declared_layer_type": "linear_attention",
        "state_role": role,
        "tensor_path": path,
        "shape": list(shape),
        "stride": [1] * len(shape),
        "dtype": dtype,
        "device": "cpu",
        "requires_grad": False,
        "logical_numel": 1 if not shape else int(torch.tensor(shape).prod()),
        "logical_bytes": contract.logical_bytes(tuple(shape), dtype),
        "storage_data_ptr": 1,
        "storage_offset": storage_offset,
        "storage_nbytes": storage_nbytes,
        "storage_identity": storage_identity,
        "lifetime_epoch": 1,
        "sequence_length": 17,
        "update_kind": "created",
        "content_sha256": content_sha256,
    }


def _synthetic_components():
    return [
        _component(
            role="full_attention_key",
            path="layers[0].key_cache",
            shape=(1, 4),
            storage_identity="key-storage",
            content_sha256="a" * 64,
        ),
        _component(
            role="linear_recurrent_state",
            path="layers[0].recurrent_state",
            shape=(1, 4),
            storage_identity="recurrent-storage",
            content_sha256="b" * 64,
        ),
        _component(
            role="linear_convolution_state",
            path="layers[0].convolution_state",
            shape=(1, 4),
            storage_identity="convolution-storage",
            content_sha256="c" * 64,
        ),
        _component(
            role="position_or_sequence_metadata",
            path="sequence_length",
            shape=(1,),
            storage_identity="metadata-storage",
            content_sha256="d" * 64,
        ),
    ]


def test_snapshot_comparison_distinguishes_growth_replacement_and_in_place():
    previous = _synthetic_components()
    current = [dict(row) for row in previous]
    current[0]["shape"] = [1, 5]
    current[0]["logical_numel"] = 5
    current[0]["logical_bytes"] = 20
    current[0]["content_sha256"] = "e" * 64
    current[1]["content_sha256"] = "f" * 64
    current[2]["storage_identity"] = "new-convolution-storage"
    current[2]["storage_data_ptr"] = 2
    current[2]["content_sha256"] = "0" * 64
    transitions = probe.compare_state_snapshots(previous, current)
    assert transitions[probe._component_key(current[0])] == "grown"
    assert transitions[probe._component_key(current[1])] == "mutated_in_place"
    assert transitions[probe._component_key(current[2])] == "replaced"
    assert transitions[probe._component_key(current[3])] == "unchanged"


def test_snapshot_comparison_emits_created_and_released():
    previous = _synthetic_components()
    current = [dict(row) for row in previous[1:]]
    current.append(_component(
        role="full_attention_value",
        path="layers[0].value_cache",
        shape=(1, 4),
        storage_identity="value-storage",
        content_sha256="1" * 64,
    ))
    transitions = probe.compare_state_snapshots(previous, current)
    assert transitions[probe._component_key(previous[0])] == "released"
    assert transitions[probe._component_key(current[-1])] == "created"


def test_snapshot_comparison_keeps_transition_per_component_key():
    first = _component(
        role="linear_recurrent_state",
        path="layers[0].recurrent_state",
        shape=(1, 4),
        storage_identity="recurrent-storage-0",
        content_sha256="a" * 64,
    )
    second = dict(
        _component(
            role="linear_recurrent_state",
            path="layers[1].recurrent_state",
            shape=(1, 4),
            storage_identity="recurrent-storage-1",
            content_sha256="b" * 64,
        ),
        layer_index=1,
    )
    current_first = dict(first, content_sha256="c" * 64)
    current_second = dict(second)
    transitions = probe.compare_state_snapshots(
        [first, second],
        [current_first, current_second],
    )
    assert transitions[probe._component_key(first)] == "mutated_in_place"
    assert transitions[probe._component_key(second)] == "unchanged"


def test_snapshot_comparison_rejects_generation_aliasing():
    previous = _synthetic_components()
    current = [dict(row) for row in previous]
    current.append(dict(current[0], request_generation=1))
    _expect_value_error(
        lambda: probe.compare_state_snapshots(previous, current),
        "request generation",
    )


def test_export_import_round_trip_is_ordered_by_request_layer_and_role():
    components = [
        dict(_synthetic_components()[2], layer_index=2),
        dict(_synthetic_components()[0], layer_index=0),
        dict(_synthetic_components()[1], layer_index=1),
    ]
    payload = probe.export_normalized_state(components)
    restored = probe.import_normalized_state(payload)
    assert [item["layer_index"] for item in restored] == sorted(
        item["layer_index"] for item in restored
    )
    assert contract.canonical_json_sha256(restored) == (
        contract.canonical_json_sha256(
            probe.export_normalized_state(restored)["components"]
        )
    )


def test_export_import_rejects_wrong_schema_and_duplicate_keys():
    payload = probe.export_normalized_state(_synthetic_components())
    _expect_value_error(
        lambda: probe.import_normalized_state(
            dict(payload, schema_version=999)
        ),
        "schema_version",
    )
    duplicated = list(payload["components"])
    duplicated.append(dict(duplicated[0]))
    _expect_value_error(
        lambda: probe.import_normalized_state(
            dict(payload, components=duplicated)
        ),
        "duplicate",
    )


def test_atomic_json_and_jsonl_writers_leave_no_partial_files():
    component = _synthetic_components()[0]
    snapshot = {
        "snapshot_id": "snapshot-0",
        "request_id": "request-a",
        "request_generation": 0,
        "lifetime_epoch": 1,
        "sequence_length": 17,
        "component_count": 1,
        "component_sha256": contract.canonical_json_sha256([component]),
    }
    memory = {
        "snapshot_id": "memory-0",
        "phase": "after_prefill",
        "request_id": "request-a",
        "request_generation": 0,
        "cuda_allocated_bytes": 0,
        "cuda_reserved_bytes": 0,
        "logical_state_bytes": component["logical_bytes"],
        "unique_storage_bytes": component["storage_nbytes"],
    }
    with tempfile.TemporaryDirectory() as temporary:
        run_dir = Path(temporary)
        probe.write_json_atomic(
            run_dir / "summary.json",
            {"schema_version": contract.SCHEMA_VERSION},
        )
        probe.write_jsonl_atomic(
            run_dir / "state_snapshots.jsonl",
            [snapshot],
            required_fields=probe.STATE_SNAPSHOT_FIELDS,
        )
        probe.write_jsonl_atomic(
            run_dir / "state_components.jsonl",
            [component],
            required_fields=tuple(
                contract.StateComponent.__dataclass_fields__
            ),
        )
        probe.write_jsonl_atomic(
            run_dir / "memory_snapshots.jsonl",
            [memory],
            required_fields=probe.MEMORY_SNAPSHOT_FIELDS,
        )
        assert not list(run_dir.glob("*.partial"))
        assert json.loads((run_dir / "summary.json").read_text()) == {
            "schema_version": contract.SCHEMA_VERSION,
        }
        for filename, fields in (
            ("state_snapshots.jsonl", probe.STATE_SNAPSHOT_FIELDS),
            (
                "state_components.jsonl",
                tuple(contract.StateComponent.__dataclass_fields__),
            ),
            ("memory_snapshots.jsonl", probe.MEMORY_SNAPSHOT_FIELDS),
        ):
            lines = (run_dir / filename).read_text().splitlines()
            assert lines
            for line in lines:
                assert set(json.loads(line)) == set(fields)


def main():
    tests = [
        value
        for name, value in sorted(globals().items())
        if name.startswith("test_") and callable(value)
    ]
    for test in tests:
        test()
    print("qwen35 hybrid-state probe unit tests passed")


if __name__ == "__main__":
    main()
